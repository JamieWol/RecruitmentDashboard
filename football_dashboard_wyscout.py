"""
Football Recruitment Dashboard (Streamlit)

Run with:
    streamlit run football_dashboard_wyscout.py

Features:
- CSV / Excel upload
- Auto-detect player/team/league/position/minutes columns
- Infer likely football metrics from numeric columns
- Exclude obvious admin / ID / metadata fields
- League filter when multiple leagues are uploaded
- Team and position filters
- Market opportunity filter (age < 25, contract expiry < 12 months, top score band)
- Selected player from the ranking section drives the charts and tabs
- Transfermarkt links based on the player's full name
- Original-style pizza chart when mplsoccer is available
- White pizza-chart background
- Percentile colour coding in the ranking table
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    from mplsoccer import PyPizza
except ImportError:  # pragma: no cover
    PyPizza = None


# --------------------------------
# PAGE SETTINGS
# --------------------------------
st.set_page_config(page_title="Football Recruitment Dashboard", layout="wide")


# --------------------------------
# COLUMN DETECTION
# --------------------------------
NAME_CANDIDATES = ["Player", "Name", "player", "name", "Footballer"]
TEAM_CANDIDATES = ["Team", "Club", "Squad", "team", "club", "Current Team"]
LEAGUE_CANDIDATES = ["League", "Competition", "competition", "league"]
POSITION_CANDIDATES = ["Position", "Primary Position", "Role", "position"]
MINUTES_CANDIDATES = [
    "Minutes played",
    "Minutes",
    "mins",
    "Min",
    "minutes played",
    "Minutes Played",
    "Minutes (Last 2 years)",
]
AGE_CANDIDATES = ["Age", "age"]
CONTRACT_DAYS_CANDIDATES = [
    "Contract Expiry (days left)",
    "Contract expiry (days left)",
    "Contract days left",
    "Days left on contract",
]
CONTRACT_DATE_CANDIDATES = [
    "Contract expires",
    "Contract Expiry",
    "Contract end",
    "Expiry",
]


@dataclass
class ColumnMap:
    name: str
    team: Optional[str] = None
    league: Optional[str] = None
    position: Optional[str] = None
    minutes: Optional[str] = None
    age: Optional[str] = None
    contract_days: Optional[str] = None
    contract_date: Optional[str] = None


# --------------------------------
# DATA HELPERS
# --------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()].copy()


@st.cache_data
def load_data(file):
    if file.name.lower().endswith(("xlsx", "xls")):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)

    df = clean_columns(df)

    # Best-effort numeric cleanup for mixed uploads.
    for col in df.columns:
        if df[col].dtype == "object":
            numeric_guess = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce",
            )
            if numeric_guess.notna().sum() >= max(3, int(len(df) * 0.7)):
                df[col] = numeric_guess

    return df


def find_first_existing(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_columns(df: pd.DataFrame) -> ColumnMap:
    return ColumnMap(
        name=find_first_existing(df, NAME_CANDIDATES) or "",
        team=find_first_existing(df, TEAM_CANDIDATES),
        league=find_first_existing(df, LEAGUE_CANDIDATES),
        position=find_first_existing(df, POSITION_CANDIDATES),
        minutes=find_first_existing(df, MINUTES_CANDIDATES),
        age=find_first_existing(df, AGE_CANDIDATES),
        contract_days=find_first_existing(df, CONTRACT_DAYS_CANDIDATES),
        contract_date=find_first_existing(df, CONTRACT_DATE_CANDIDATES),
    )


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def make_transfermarkt_url(player_name: str) -> str:
    return (
        "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query="
        + quote_plus(player_name.strip())
    )


def build_rank_view(
    df: pd.DataFrame,
    name_col: str,
    team_col: str | None,
    league_col: str | None,
    position_col: str | None,
) -> pd.DataFrame:
    view = df.copy()

    if name_col in view.columns:
        view["Display Name"] = view[name_col].astype(str)
    if team_col and team_col in view.columns:
        view["Display Team"] = view[team_col].astype(str)
    if league_col and league_col in view.columns:
        view["Display League"] = view[league_col].astype(str)
    if position_col and position_col in view.columns:
        view["Display Position"] = view[position_col].astype(str)

    tm_url_col = None
    for candidate in ["Transfermarkt URL", "Transfermarkt", "TM URL", "tm_url", "Transfermarkt Link"]:
        if candidate in view.columns:
            tm_url_col = candidate
            break

    if tm_url_col:
        view["Transfermarkt Link"] = view[tm_url_col].astype(str)
    elif "Display Name" in view.columns:
        view["Transfermarkt Link"] = [make_transfermarkt_url(n) for n in view["Display Name"].astype(str)]

    return view.loc[:, ~view.columns.duplicated()].copy()


# --------------------------------
# METRIC INFERENCE
# --------------------------------
def infer_metric_columns(df: pd.DataFrame) -> list[str]:
    exclude_exact = {
        "Player", "Name", "Team", "Club", "Squad", "League", "Competition",
        "Position", "Primary Position", "Role", "Age",
        "Minutes played", "Minutes", "mins", "Min", "Minutes Played",
        "Contract expires", "Passport country", "Foot", "Height", "Weight",
        "Valuation", "Contract Expiry (days left)", "Woman player no", "Player no",
        "Match no", "Team no", "Season", "Appearances", "90s Played", "Starting Appearances",
        "__player_name__", "__team__", "__league__", "__position__", "__row_id__",
        "Display Name", "Display Team", "Display League", "Display Position",
        "Transfermarkt Link",
    }

    exclude_keywords = [
        "id", "no", "name", "team", "club", "squad", "player", "match",
        "season", "league", "competition", "birth", "height", "weight",
        "passport", "country", "foot", "shirt", "age", "position", "role",
        "minute",
    ]

    include_keywords = [
        "xg", "xa", "shot", "pass", "carry", "dribble", "duel", "tackle",
        "interception", "press", "clearance", "block", "progressive", "touch",
        "cross", "chance", "key", "goal", "assist", "recover", "foul",
        "save", "action", "possession", "turnover", "expected", "build",
        "final third", "penalty", "chance created", "box", "aerial", "p90",
    ]

    metrics: list[str] = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        c = col.lower()
        if any(k in c for k in exclude_keywords):
            continue

        if any(k in c for k in include_keywords):
            metrics.append(col)
            continue

        nunique = df[col].nunique(dropna=True)
        if nunique >= 5 and not re.fullmatch(r"\d+", c):
            metrics.append(col)

    return unique_preserve_order(metrics)


# --------------------------------
# TRANSFORMS
# --------------------------------
def add_percentiles(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    df = df.copy()
    for m in metrics:
        numeric_series = safe_numeric(df[m])
        df[m + " Percentile"] = (
            numeric_series.rank(pct=True, method="max").mul(100).round(0).fillna(0).astype("Int64")
        )
    return df


def compute_league_average(df: pd.DataFrame, metrics: list[str]) -> list[int]:
    league_avg: list[int] = []
    for m in metrics:
        series = safe_numeric(df[m])
        avg_value = series.mean()
        if len(df) == 0 or pd.isna(avg_value):
            league_avg.append(0)
        else:
            percentile = ((series < avg_value).sum() / len(df)) * 100
            league_avg.append(int(round(percentile)))
    return league_avg


def get_show_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return unique_preserve_order([c for c in cols if c in df.columns])


def display_table(df: pd.DataFrame, cols: list[str]) -> None:
    cols = get_show_columns(df, cols)
    if not cols:
        st.info("No columns to display.")
        return
    safe_df = df.loc[:, ~df.columns.duplicated()].copy()
    st.dataframe(safe_df[cols], use_container_width=True)


def percentile_style_table(df: pd.DataFrame, cols: list[str]) -> pd.io.formats.style.Styler:
    palette = [
        (90, "#d1fae5"),
        (70, "#dcfce7"),
        (50, "#fef08a"),
        (30, "#fdba74"),
        (0, "#fecaca"),
    ]

    def color_value(v):
        try:
            if pd.isna(v):
                return ""
            v = float(v)
        except Exception:
            return ""
        for threshold, color in palette:
            if v >= threshold:
                return f"background-color: {color};"
        return ""

    styled = df[cols].style
    percentile_cols = [c for c in cols if c.endswith(" Percentile") or c == "Overall Score"]
    if percentile_cols:
        if hasattr(styled, "map"):
            styled = styled.map(color_value, subset=percentile_cols)
        else:
            try:
                styled = styled.applymap(color_value, subset=percentile_cols)
            except Exception:
                pass
    return styled


def compute_similarity_frame(df: pd.DataFrame, player_name: str, metrics: list[str], top_n: int = 10) -> pd.DataFrame:
    if not metrics or "Display Name" not in df.columns:
        return pd.DataFrame()

    working_cols = [c for c in ["Display Name", "Display Team", "Display League", "Display Position"] if c in df.columns]
    working_cols += [m for m in metrics if m in df.columns]
    working = df[working_cols].copy()

    metric_cols = [m for m in metrics if m in working.columns]
    if not metric_cols:
        return pd.DataFrame()

    for m in metric_cols:
        working[m] = pd.to_numeric(working[m], errors="coerce")

    working = working.dropna(subset=metric_cols).copy()
    if working.empty:
        return pd.DataFrame()

    player_row = working[working["Display Name"] == player_name]
    if player_row.empty:
        return pd.DataFrame()

    player_vector = player_row.iloc[0][metric_cols].to_numpy(dtype=float)
    other_vectors = working[metric_cols].to_numpy(dtype=float)
    distances = np.linalg.norm(other_vectors - player_vector, axis=1)
    working["Similarity Score"] = 1 / (1 + distances)
    working = working[working["Display Name"] != player_name].sort_values("Similarity Score", ascending=False)
    return working.head(top_n)


# --------------------------------
# CHARTS
# --------------------------------
def plot_radar(labels: list[str], values_list: list[list[float]], labels_list: list[str]):
    if not labels:
        st.info("No metrics selected for radar chart.")
        return

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])

    for values, label in zip(values_list, labels_list):
        vals = values + values[:1]
        ax.plot(angles, vals, linewidth=2, label=label)
        ax.fill(angles, vals, alpha=0.25)

    ax.legend(loc="upper right")
    st.pyplot(fig)
    plt.close(fig)


def plot_pizza_like(player: str, df: pd.DataFrame, metrics: list[str], league_avg: list[int]):
    percentile_cols = [m + " Percentile" for m in metrics]
    player_rows = df.loc[df["__player_name__"] == player, percentile_cols]
    if player_rows.empty:
        st.warning("Player not found for pizza chart.")
        return

    player_values = player_rows.values.flatten().tolist()
    if len(player_values) != len(metrics):
        st.warning("Selected player does not have the right number of percentile values.")
        return

    if PyPizza is not None:
        pizza = PyPizza(
            params=metrics,
            min_range=[0] * len(metrics),
            max_range=[100] * len(metrics),
            background_color="white",
            straight_line_color="black",
            straight_line_lw=1.5,
            last_circle_color="black",
            last_circle_lw=2,
            other_circle_color="none",
        )

        fig, ax = pizza.make_pizza(
            league_avg,
            figsize=(12, 12),
            kwargs_slices=dict(facecolor="#facc15", edgecolor="black", linewidth=2),
            kwargs_params=dict(fontsize=9, color="white", fontweight="bold"),
            kwargs_values=dict(
                fontsize=11,
                fontweight="bold",
                color="black",
                bbox=dict(edgecolor="black", facecolor="#facc15", boxstyle="round,pad=0.25"),
            ),
        )

        pizza.make_pizza(
            player_values,
            ax=ax,
            kwargs_slices=dict(facecolor="#3b82f6", edgecolor="black", linewidth=2, alpha=0.9),
            kwargs_values=dict(
                fontsize=11,
                fontweight="bold",
                color="white",
                bbox=dict(edgecolor="black", facecolor="#3b82f6", boxstyle="round,pad=0.25"),
            ),
        )

        legend_handles = [
            plt.Line2D([0], [0], color="#3b82f6", lw=10, label=player),
            plt.Line2D([0], [0], color="#facc15", lw=10, label="League Average"),
        ]
        ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(1.15, 1.1))
        st.pyplot(fig)
        plt.close(fig)
        return

    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    ax.grid(True, alpha=0.35)

    league_vals = league_avg + league_avg[:1]
    player_vals = player_values + player_values[:1]

    ax.plot(angles, league_vals, linewidth=2, label="League Average")
    ax.fill(angles, league_vals, alpha=0.18)

    ax.plot(angles, player_vals, linewidth=2, label=player)
    ax.fill(angles, player_vals, alpha=0.22)

    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1))
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------
# SELF-TESTS
# --------------------------------
def run_self_tests():
    sample = pd.DataFrame(
        {
            "Player": ["A", "B"],
            "Team": ["X", "Y"],
            "League": ["EPL", "EPL"],
            "Minutes played": [100, 200],
            "xg": [0.2, 0.4],
            "passes": [10, 20],
            "Woman player no": [1, 2],
            "shirt_number": [9, 10],
            "Age": [21, 24],
            "Contract Expiry (days left)": [120, 300],
        }
    )
    sample = clean_columns(sample)
    inferred = infer_metric_columns(sample)
    assert "xg" in inferred, "Expected xg to be inferred as a metric"
    assert "passes" in inferred, "Expected passes to be inferred as a metric"
    assert "Woman player no" not in inferred, "Expected admin columns to be excluded"
    assert "shirt_number" not in inferred, "Expected shirt numbers to be excluded"
    assert "Minutes played" not in inferred, "Expected minutes to be excluded"
    assert "Age" not in inferred, "Expected age to be excluded"

    cols = detect_columns(sample)
    assert cols.league == "League", "Expected League column to be detected"
    assert cols.contract_days == "Contract Expiry (days left)", "Expected contract days column to be detected"

    sim_sample = pd.DataFrame(
        {
            "Display Name": ["A", "B", "C"],
            "Display Team": ["T1", "T2", "T3"],
            "Display League": ["L1", "L1", "L2"],
            "Display Position": ["M", "M", "D"],
            "xg Percentile": [90, 80, np.nan],
            "passes Percentile": [70, 60, 50],
        }
    )
    sim_result = compute_similarity_frame(sim_sample, "A", ["xg Percentile", "passes Percentile"], top_n=10)
    assert not sim_result.empty, "Expected similarity result for valid numeric rows"
    assert sim_result.iloc[0]["Display Name"] == "B", "Expected B to be closest to A in the sample"


# --------------------------------
# SIDEBAR
# --------------------------------
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

with st.sidebar.expander("Developer checks", expanded=False):
    run_checks = st.checkbox("Run inference self-tests", value=False)

if run_checks:
    try:
        run_self_tests()
        st.success("Self-tests passed.")
    except AssertionError as exc:
        st.error(str(exc))
        st.stop()


# --------------------------------
# MAIN APP
# --------------------------------
if not uploaded_file:
    st.title("⚽ Football Recruitment Dashboard")
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()


df = load_data(uploaded_file)
columns = detect_columns(df)

if not columns.name:
    st.error("No player/name column found. Please include a column like 'Player' or 'Name'.")
    st.stop()

work = df.copy()
work["__row_id__"] = np.arange(len(work))
work["__player_name__"] = work[columns.name].astype(str)
work["__team__"] = work[columns.team].astype(str) if columns.team else ""
work["__league__"] = work[columns.league].astype(str) if columns.league else ""
work["__position__"] = work[columns.position].astype(str) if columns.position else ""
work = work[work["__player_name__"].notna() & (work["__player_name__"].str.strip() != "")].copy()

# League filter before ranking and charts.
if columns.league and columns.league in work.columns:
    league_values = sorted(work[columns.league].dropna().astype(str).unique().tolist())
    if len(league_values) > 1:
        selected_league = st.sidebar.selectbox("League", ["All leagues"] + league_values)
        if selected_league != "All leagues":
            work = work[work[columns.league].astype(str) == selected_league].copy()

# Position filter to help browsing large uploads.
if columns.position and columns.position in work.columns:
    position_values = sorted(work[columns.position].dropna().astype(str).unique().tolist())
    if len(position_values) > 1:
        selected_positions = st.sidebar.multiselect("Position", position_values, default=position_values)
        work = work[work[columns.position].astype(str).isin(selected_positions)].copy()

# Clean minutes / age / contract if present.
if columns.minutes and columns.minutes in work.columns:
    work[columns.minutes] = safe_numeric(work[columns.minutes])
    work = work[work[columns.minutes].notna()].copy()

if columns.age and columns.age in work.columns:
    work[columns.age] = safe_numeric(work[columns.age])

if columns.contract_days and columns.contract_days in work.columns:
    work[columns.contract_days] = safe_numeric(work[columns.contract_days])

all_metrics = infer_metric_columns(work)
if len(all_metrics) == 0:
    st.error("No usable numeric performance metrics were found in the uploaded file.")
    st.stop()

st.success(f"{len(work)} rows loaded")


# --------------------------------
# FILTERS
# --------------------------------
st.sidebar.subheader("Filters")

if columns.minutes and columns.minutes in work.columns and work[columns.minutes].notna().any():
    min_mins = int(work[columns.minutes].min())
    max_mins = int(work[columns.minutes].max())
    mins_range = st.sidebar.slider("Minutes Played", min_mins, max_mins, (min_mins, max_mins))
    work = work[work[columns.minutes].between(mins_range[0], mins_range[1])].copy()

if columns.age and columns.age in work.columns and work[columns.age].notna().any():
    min_age = int(work[columns.age].min())
    max_age = int(work[columns.age].max())
    age_range = st.sidebar.slider("Age", min_age, max_age, (min_age, max_age))
    work = work[work[columns.age].between(age_range[0], age_range[1])].copy()

# Market opportunity filter
market_mode = st.sidebar.checkbox("Market opportunity filter", value=False)
if market_mode:
    if columns.age and columns.age in work.columns and work[columns.age].notna().any():
        age_ok = work[columns.age] < 25
    else:
        age_ok = pd.Series(True, index=work.index)

    if columns.contract_days and columns.contract_days in work.columns and work[columns.contract_days].notna().any():
        contract_ok = work[columns.contract_days] < 365
    elif columns.contract_date and columns.contract_date in work.columns:
        contract_ok = pd.to_datetime(work[columns.contract_date], errors="coerce").notna()
    else:
        contract_ok = pd.Series(True, index=work.index)

    work = work.copy()
    work["__market_filter_age_ok__"] = age_ok
    work["__market_filter_contract_ok__"] = contract_ok
else:
    work["__market_filter_age_ok__"] = True
    work["__market_filter_contract_ok__"] = True


# --------------------------------
# METRIC SELECTOR
# --------------------------------
st.sidebar.subheader("Metrics")
auto_mode = st.sidebar.checkbox("Auto-detect metrics", value=True)

if auto_mode:
    default_metrics = all_metrics[: min(10, len(all_metrics))]
    metrics = st.sidebar.multiselect("Select Metrics", all_metrics, default=default_metrics)
else:
    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    default_metrics = numeric_cols[: min(10, len(numeric_cols))]
    metrics = st.sidebar.multiselect("Select Metrics", numeric_cols, default=default_metrics)

if len(metrics) == 0:
    st.warning("Select at least one metric")
    st.stop()

# Team filter sits under the metric selector so the sidebar flows from metrics to browsing filters.
if columns.team and columns.team in work.columns:
    team_values = sorted(work[columns.team].dropna().astype(str).unique().tolist())
    if len(team_values) > 1:
        selected_teams = st.sidebar.multiselect("Team", team_values, default=team_values)
        work = work[work[columns.team].astype(str).isin(selected_teams)].copy()

for m in metrics:
    work[m] = safe_numeric(work[m])


# --------------------------------
# PERCENTILES
# --------------------------------
work = add_percentiles(work, metrics)
percentile_cols = [m + " Percentile" for m in metrics]


# --------------------------------
# LEAGUE AVERAGE
# --------------------------------
league_avg = compute_league_average(work, metrics)


# --------------------------------
# PLAYER RANKING
# --------------------------------
work["Overall Score"] = work[percentile_cols].mean(axis=1).round(0)

# Apply market opportunity filter threshold after score exists.
if market_mode and len(work) > 0:
    score_threshold = work["Overall Score"].quantile(0.80)
    work = work[
        work["__market_filter_age_ok__"]
        & work["__market_filter_contract_ok__"]
        & (work["Overall Score"] >= score_threshold)
    ].copy()
    if len(work) > 0:
        # Recompute from the filtered set.
        work["Overall Score"] = work[percentile_cols].mean(axis=1).round(0)

work = work.sort_values("Overall Score", ascending=False).reset_index(drop=True)
work.index += 1
work.insert(0, "Rank", work.index)


# --------------------------------
# DISPLAY ALIASES / LINKS
# --------------------------------
filtered_df = work.copy()
filtered_df["Display Name"] = filtered_df["__player_name__"]
filtered_df["Display Team"] = filtered_df["__team__"]
filtered_df["Display League"] = filtered_df["__league__"]
filtered_df["Display Position"] = filtered_df["__position__"]
filtered_df = filtered_df.loc[:, ~filtered_df.columns.duplicated()].copy()

pizza_metrics = metrics
position_metrics_map = globals().get("position_metrics_map", {})

rank_view = build_rank_view(work, columns.name, columns.team, columns.league, columns.position)
rank_view["Transfermarkt Link"] = [make_transfermarkt_url(n) for n in rank_view["Display Name"].astype(str)]

# Use the selected player everywhere.
if "active_player" not in st.session_state and len(rank_view) > 0:
    st.session_state["active_player"] = rank_view.iloc[0]["Display Name"]
active_player = st.session_state.get("active_player")


# --------------------------------
# DASHBOARD
# --------------------------------
st.title("⚽ Football Recruitment Dashboard")

c1, c2, c3 = st.columns(3)
c1.metric("Players", f"{len(work)}")
c2.metric("Metrics", f"{len(metrics)}")
c3.metric("Top Score", f"{int(work['Overall Score'].max()) if len(work) else 0}")

st.subheader("🏅 Player Ranking")
player_options = rank_view["Display Name"].dropna().tolist()
if player_options:
    default_player = st.session_state.get("active_player", player_options[0])
    if default_player not in player_options:
        default_player = player_options[0]
    active_player = st.selectbox(
        "Search / select player",
        player_options,
        index=player_options.index(default_player),
        key="active_player_top_selector",
    )
    st.session_state["active_player"] = active_player
else:
    active_player = None
    st.info("No players available in the current filtered set.")

rank_view_display = rank_view.copy()
if active_player and active_player in rank_view_display["Display Name"].astype(str).tolist():
    rank_view_display["Selected"] = ""
    rank_view_display.loc[rank_view_display["Display Name"] == active_player, "Selected"] = "👉"

show_cols = ["Selected", "Rank", "Display Name", "Display Team", "Display League", "Display Position"]
for maybe_col in [columns.age, columns.minutes, columns.contract_days, columns.contract_date]:
    if maybe_col:
        show_cols.append(maybe_col)
for optional_col in ["Valuation", "Contract Expiry (days left)"]:
    if optional_col in rank_view.columns:
        show_cols.append(optional_col)
show_cols.append("Overall Score")
show_cols.extend(metrics)
show_cols.append("Transfermarkt Link")
show_cols = get_show_columns(rank_view_display, show_cols)

styled_rank = percentile_style_table(rank_view_display, show_cols)
st.dataframe(
    styled_rank,
    use_container_width=True,
    column_config={
        "Transfermarkt Link": st.column_config.LinkColumn("Transfermarkt Link"),
    },
)

st.caption("The selected player drives the pizza chart, comparison, similarity, and role profile tabs.")


# --------------------------------
# TOP PERFORMERS
# --------------------------------
st.subheader("⭐ Top Performers")
top_players = []
for m in metrics:
    series = safe_numeric(work[m])
    if series.notna().any():
        top_idx = series.idxmax()
        top_row = work.loc[top_idx]
        top_players.append({"Metric": m, "Player": top_row["__player_name__"], "Value": top_row[m]})

st.dataframe(pd.DataFrame(top_players), use_container_width=True)


# --------------------------------
# CHARTS
# --------------------------------
player_list = work["__player_name__"].tolist()

st.subheader("📊 Pizza Chart")
if active_player and active_player in player_list:
    plot_pizza_like(active_player, work, metrics, league_avg)
else:
    st.info("No players available in the current filtered set.")

st.subheader("📈 Player Comparison")
p1_default = active_player if active_player in player_list else (player_list[0] if player_list else None)
p1 = st.selectbox(
    "Player 1",
    player_list,
    index=player_list.index(p1_default) if p1_default in player_list else 0,
    key="p1",
)
p2_default = 1 if len(player_list) > 1 else 0
p2 = st.selectbox("Player 2", player_list, index=p2_default, key="p2")

if p1 != p2 and len(metrics) > 0:
    vals1 = work.loc[work["__player_name__"] == p1, percentile_cols].values.flatten().tolist()
    vals2 = work.loc[work["__player_name__"] == p2, percentile_cols].values.flatten().tolist()
    if len(vals1) == len(metrics) and len(vals2) == len(metrics):
        plot_radar(metrics, [vals1, vals2], [p1, p2])

st.subheader("📊 Scatter Graph")
two_metrics = st.multiselect("Select 2 metrics", metrics, default=metrics[:2] if len(metrics) >= 2 else metrics)

if len(two_metrics) == 2:
    mX, mY = two_metrics
    df_plot = work.copy()
    df_plot["X"] = pd.to_numeric(df_plot[mX + " Percentile"], errors="coerce") / 100
    df_plot["Y"] = pd.to_numeric(df_plot[mY + " Percentile"], errors="coerce") / 100

    fig, ax = plt.subplots(figsize=(14, 12))

    xv, yv = np.meshgrid(
        np.linspace(-0.05, 1.05, 600),
        np.linspace(-0.05, 1.05, 600),
    )
    Z = ((xv.clip(0, 1)) + (yv.clip(0, 1))) / 2

    ax.imshow(Z, extent=(-0.05, 1.05, -0.05, 1.05), origin="lower", cmap="RdYlGn", alpha=0.6)
    ax.scatter(df_plot["X"], df_plot["Y"], s=140, facecolors="white", edgecolors="black")

    highlight = st.multiselect("Highlight players", df_plot["__player_name__"].tolist())

    for _, r in df_plot.iterrows():
        if r["__player_name__"] not in highlight:
            ax.text(r["X"] + 0.01, r["Y"] + 0.01, r["__player_name__"], fontsize=9)

    for hp in highlight:
        row = df_plot[df_plot["__player_name__"] == hp]
        if not row.empty:
            ax.scatter(row["X"], row["Y"], s=450, facecolors="none", edgecolors="black", linewidths=2)
            ax.text(row["X"].values[0] + 0.015, row["Y"].values[0] + 0.015, hp, fontsize=12, fontweight="bold")

    ax.axhline(0.5, color="black", linestyle="--")
    ax.axvline(0.5, color="black", linestyle="--")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(mX + " Percentile")
    ax.set_ylabel(mY + " Percentile")
    ax.set_title("Player Scatter Graph")
    ax.grid(False)

    def quad_label(x: float, y: float, text: str, color: str) -> None:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor=color,
                linewidth=2,
                alpha=0.95,
            ),
        )

    quad_label(0.28, 0.945, f"Strong in {mY} Only", "#d4a017")
    quad_label(0.72, 0.945, "Strong In Both", "green")
    quad_label(0.28, 0.055, "Weak In Both", "red")
    quad_label(0.72, 0.055, f"Strong in {mX} Only", "#d4a017")

    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("Select exactly 2 metrics.")


# --------------------------------
# TABS
# --------------------------------
tabs = st.tabs([
    "🏆 Shortlist Players",
    "🤝 Similar Players",
    "⚖️ Custom Scoring",
    "📝 Role Profiles",
])

if "shortlist_log" not in st.session_state:
    st.session_state.shortlist_log = pd.DataFrame()

# --- Tab 1: Shortlist Players ---
with tabs[0]:
    st.subheader("🏆 Shortlist Players")
    num_shortlist = st.slider("Number of top players to show", 1, 20, 5)
    top_players = filtered_df.head(num_shortlist).copy()

    shortlist_cols = [c for c in ["Display Name", "Display Team", "Display League", "Display Position", "Overall Score"] if c in top_players.columns]
    shortlist_cols += [m for m in pizza_metrics if m in top_players.columns]
    shortlist_cols = unique_preserve_order(shortlist_cols)
    display_table(top_players, shortlist_cols)

    st.markdown("### 🔍 Select Players to Add to Shortlist")
    if active_player and active_player in filtered_df["Display Name"].dropna().tolist():
        st.info(f"Active player: {active_player}")
    selected_players = st.multiselect(
        "Select players by name",
        options=filtered_df["Display Name"].dropna().tolist(),
        default=[active_player] if active_player in filtered_df["Display Name"].dropna().tolist() else [],
    )

    if st.button("➕ Add Selected Players to Shortlist Log"):
        if selected_players:
            players_to_add = filtered_df[filtered_df["Display Name"].isin(selected_players)][
                [c for c in ["Display Name", "Display Team", "Display League", "Display Position", "Overall Score"] if c in filtered_df.columns]
            ].copy()
            st.session_state.shortlist_log = pd.concat(
                [st.session_state.shortlist_log, players_to_add],
                ignore_index=True,
            ).drop_duplicates(subset=["Display Name"]).reset_index(drop=True)
            st.success(f"{len(players_to_add)} players added to shortlist log.")
        else:
            st.warning("No players selected.")

    st.subheader("📋 Current Shortlist Log by Position")
    if not st.session_state.shortlist_log.empty and "Display Position" in st.session_state.shortlist_log.columns:
        positions = sorted(st.session_state.shortlist_log["Display Position"].dropna().unique())
        max_rows = st.session_state.shortlist_log.groupby("Display Position").size().max()
        table = pd.DataFrame(index=range(max_rows), columns=positions)

        for pos in positions:
            players = st.session_state.shortlist_log[st.session_state.shortlist_log["Display Position"] == pos]["Display Name"].tolist()
            for i, player in enumerate(players):
                table.at[i, pos] = player

        table.index = table.index + 1
        st.dataframe(table.fillna(""), use_container_width=True)

    csv_buffer = io.StringIO()
    st.session_state.shortlist_log.to_csv(csv_buffer, index=False)
    st.download_button(
        label="📥 Download Full Shortlist Log CSV",
        data=csv_buffer.getvalue(),
        file_name="shortlist_log.csv",
        mime="text/csv",
    )

# --- Tab 2: Similar Players ---
with tabs[1]:
    st.subheader("🤝 Find Similar Players")
    if active_player and active_player in filtered_df["Display Name"].dropna().tolist():
        st.info(f"Similarity is currently centered on: {active_player}")
    player_choices = filtered_df["Display Name"].dropna().unique().tolist()
    selected_player_sim = st.selectbox(
        "Select Player to Find Similar Ones",
        player_choices,
        index=player_choices.index(active_player) if active_player in player_choices else 0,
        key="similarity_player",
    )

    if selected_player_sim:
        metrics_for_similarity = [m + " Percentile" for m in pizza_metrics if m + " Percentile" in filtered_df.columns]
        if metrics_for_similarity:
            sim_df = compute_similarity_frame(filtered_df, selected_player_sim, metrics_for_similarity, top_n=10)
            if sim_df.empty:
                st.info("No similar players could be calculated from the current metric set.")
            else:
                sim_cols = [c for c in ["Display Name", "Display Team", "Display League", "Display Position", "Similarity Score"] if c in sim_df.columns]
                sim_cols = unique_preserve_order(sim_cols)
                display_table(sim_df, sim_cols)
        else:
            st.info("No percentile metrics available for similarity comparison.")

# --- Tab 3: Custom Scoring ---
with tabs[2]:
    st.subheader("⚖️ Custom Scoring")
    st.info("Assign weights to metrics for a custom overall score.")

    weights = {}
    for m in pizza_metrics:
        if m + " Percentile" in filtered_df.columns:
            w = st.slider(f"Weight for {m}", 0.0, 2.0, 1.0, step=0.05)
            weights[m] = w

    if weights:
        weighted_cols = [m + " Percentile" for m in weights.keys() if m + " Percentile" in filtered_df.columns]
        weight_values = np.array(list(weights.values()), dtype=float)
        if weight_values.sum() > 0 and weighted_cols:
            filtered_df["Custom Score"] = filtered_df[weighted_cols].values.dot(weight_values) / weight_values.sum()
            custom_cols = [c for c in ["Display Name", "Display Team", "Display League", "Display Position", "Custom Score"] if c in filtered_df.columns]
            custom_cols = unique_preserve_order(custom_cols)
            display_table(filtered_df.sort_values("Custom Score", ascending=False).head(10), custom_cols)

# --- Tab 4: Role Profiles ---
with tabs[3]:
    st.subheader("📝 Role Profiles")
    if "Display Position" in filtered_df.columns and filtered_df["Display Position"].notna().any():
        default_position = None
        if active_player and active_player in filtered_df["Display Name"].dropna().tolist():
            player_rows = filtered_df[filtered_df["Display Name"] == active_player]
            if not player_rows.empty:
                default_position = player_rows.iloc[0]["Display Position"]

        position_choices = sorted(filtered_df["Display Position"].dropna().unique())
        selected_position_role = st.selectbox(
            "Select Position to View Role Profile",
            position_choices,
            index=position_choices.index(default_position) if default_position in position_choices else 0,
        )
        if selected_position_role:
            role_metrics = position_metrics_map.get(selected_position_role, []) if isinstance(position_metrics_map, dict) else []
            role_metrics_present = [m for m in role_metrics if m in filtered_df.columns]
            if role_metrics_present:
                role_avg = filtered_df[role_metrics_present].mean(numeric_only=True)
                st.write(f"Average metrics for {selected_position_role}:")
                st.dataframe(role_avg.to_frame("Average").sort_values("Average", ascending=False), use_container_width=True)
            else:
                st.info("No position-specific role mapping has been defined yet.")
    else:
        st.warning("No position column available.")


# --------------------------------
# EXPORT
# --------------------------------
st.subheader("Download Data")
export_df = filtered_df.copy()
if "Overall Score" not in export_df.columns and "Overall Score" in work.columns:
    export_df["Overall Score"] = work["Overall Score"]

csv = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data", csv, "recruitment_data.csv", "text/csv")

st.caption("Metric inference, duplicate-column protection, league filtering, and Transfermarkt links are enabled.")





















