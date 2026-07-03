from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote_plus

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Integrated Football Apps", layout="wide")


# ================================================================
# Shared helpers
# ================================================================


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()].copy()


@st.cache_data
def _load_any_file(_file_name: str, _raw_bytes: bytes) -> pd.DataFrame:
    suffix = Path(_file_name).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(io.BytesIO(_raw_bytes))
    else:
        df = pd.read_csv(io.BytesIO(_raw_bytes))
    return clean_columns(df)


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    return _load_any_file(uploaded_file.name, uploaded_file.getvalue())


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def make_transfermarkt_url(player_name: str) -> str:
    return "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=" + quote_plus(player_name.strip())


# ================================================================
# Tab 1: Recruitment dashboard helpers
# ================================================================

NAME_CANDIDATES = ["Player", "Name", "player", "name", "Footballer"]
TEAM_CANDIDATES = ["Team", "Club", "Squad", "team", "club", "Current Team"]
LEAGUE_CANDIDATES = ["League", "Competition", "competition", "league"]
POSITION_CANDIDATES = ["Position", "Primary Position", "Role", "position"]
MINUTES_CANDIDATES = [
    "Minutes played", "Minutes", "mins", "Min", "minutes played", "Minutes Played", "Minutes (Last 2 years)",
]
AGE_CANDIDATES = ["Age", "age"]
CONTRACT_DAYS_CANDIDATES = [
    "Contract Expiry (days left)", "Contract expiry (days left)", "Contract days left", "Days left on contract",
]
CONTRACT_DATE_CANDIDATES = ["Contract expires", "Contract Expiry", "Contract end", "Expiry"]


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
        "Transfermarkt Link", "Primary Archetype", "Secondary Archetype", "Player Label",
    }
    exclude_keywords = [
        "id", "name", "team", "club", "squad", "player", "match",
        "season", "league", "competition", "birth", "height", "weight",
        "passport", "country", "foot", "shirt", "age", "position", "role", "minute",
    ]
    include_exact = {
        "Turnovers", "Non-Penalty Goals", "Scoring Contribution", "Goal Conversion%",
        "xG", "xG/Shot", "Aerial Win%", "PAdj Pressures", "Pressure Regains",
        "Touches In Box", "OBV", "Key Passes",
    }
    include_keywords = [
        "xg", "xa", "shot", "pass", "carry", "dribble", "duel", "tackle",
        "interception", "press", "clearance", "block", "progressive", "touch",
        "cross", "chance", "key", "goal", "assist", "recover", "foul",
        "save", "action", "possession", "turnover", "turnovers", "dispossess",
        "miscontrol", "lost possession", "ball lost", "expected", "build",
        "final third", "penalty", "non-penalty", "non penalty", "np", "np xg",
        "scoring contribution", "scoring contributions", "goal contribution",
        "goals+assists", "goals and assists", "chance created", "box", "aerial", "p90",
    ]

    metrics: list[str] = []
    for col in df.columns:
        if col in exclude_exact:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        c = col.strip().lower()
        if col in include_exact:
            metrics.append(col)
            continue
        if any(k in c for k in exclude_keywords):
            continue
        if any(k in c for k in include_keywords):
            metrics.append(col)
            continue
        if df[col].nunique(dropna=True) >= 5 and not re.fullmatch(r"\d+", c):
            metrics.append(col)
    return unique_preserve_order(metrics)


def is_lower_better_metric(metric_name: str) -> bool:
    text = metric_name.lower()
    return any(k in text for k in ["turnover", "turnovers", "dispossess", "miscontrol", "ball lost", "lost possession"])


def add_percentiles(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    df = df.copy()
    for m in metrics:
        s = safe_numeric(df[m])
        pct = s.rank(pct=True, method="max", ascending=not is_lower_better_metric(m))
        df[m + " Percentile"] = pct.mul(100).round(0).fillna(0).astype("Int64")
    return df


def compute_league_average(df: pd.DataFrame, metrics: list[str]) -> list[int]:
    league_avg: list[int] = []
    for m in metrics:
        s = safe_numeric(df[m])
        mean = s.mean()
        if len(df) == 0 or pd.isna(mean):
            league_avg.append(0)
        else:
            league_avg.append(int(round(((s < mean).sum() / len(df)) * 100)))
    return league_avg


def build_player_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    counts = df["__player_name__"].astype(str).value_counts(dropna=False)
    labels: list[str] = []
    for idx, row in df.iterrows():
        name = str(row.get("__player_name__", "")).strip()
        team = str(row.get("__team__", "")).strip()
        league = str(row.get("__league__", "")).strip()
        if counts.get(name, 0) > 1:
            suffix = team or league or f"Row {idx + 1}"
            label = f"{name} ({suffix})"
        else:
            label = name
        if label in labels:
            label = f"{label} #{idx + 1}"
        labels.append(label)
    df["Player Label"] = labels
    return df


def build_rank_view(df: pd.DataFrame, name_col: str, team_col: str | None, league_col: str | None, position_col: str | None) -> pd.DataFrame:
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


def percentile_style_table(df: pd.DataFrame, cols: list[str]) -> pd.io.formats.style.Styler:
    palette = [(90, "#d1fae5"), (70, "#dcfce7"), (50, "#fef08a"), (30, "#fdba74"), (0, "#fecaca")]

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
    pct_cols = [c for c in cols if c.endswith(" Percentile") or c == "Overall Score"]
    if pct_cols:
        if hasattr(styled, "map"):
            styled = styled.map(color_value, subset=pct_cols)
        else:
            try:
                styled = styled.applymap(color_value, subset=pct_cols)
            except Exception:
                pass
    return styled


def infer_position_group(position_value: str | None) -> str:
    text = (position_value or "").lower().strip()
    nums = set(re.findall(r"\d+", text))
    if "10" in nums or any(k in text for k in ["attacking midfielder", "am", "10"]):
        return "AM"
    if nums.intersection({"6", "8"}) or any(k in text for k in ["central midfielder", "cm", "6", "8"]):
        return "CM"
    if nums.intersection({"3", "4"}) or any(k in text for k in ["centre back", "center back", "cb"]):
        return "CB"
    if nums.intersection({"2", "5"}) or any(k in text for k in ["full back", "fullback", "fb", "wing back", "wb", "lb", "rb"]):
        return "FB"
    if "9" in nums or any(k in text for k in ["striker", "forward", "cf", "st"]):
        return "ST"
    if nums.intersection({"7", "11"}) or any(k in text for k in ["winger", "wing", "lw", "rw", "wide"]):
        return "WINGER"
    return "UNKNOWN"


def archetype_scores(row: pd.Series, metric_percentile_cols: list[str]) -> dict[str, float]:
    def value(col: str) -> float:
        try:
            if col not in row.index:
                return 0.0
            v = row[col]
            if pd.isna(v):
                return 0.0
            return float(v)
        except Exception:
            return 0.0

    def score(keys: list[str]) -> float:
        vals = [value(c) for c in metric_percentile_cols if any(k in c.lower() for k in keys)]
        return float(np.mean(vals)) if vals else 0.0

    return {
        "Carrier": np.mean([score(["carry", "dribble", "progressive carry", "ball progression"]), score(["take-on", "1v1"])]),
        "Connector": np.mean([score(["pass", "completion", "completed pass", "short pass", "combination"]), score(["link", "receive", "ball retention", "turnover", "turnovers", "dispossess", "miscontrol"])]),
        "Creator": np.mean([score(["xa", "xA", "key pass", "chance created", "assist", "through ball", "cross", "scoring contribution", "scoring contributions", "goal contribution"]), score(["final third", "box", "big chance"])]),
        "Disruptor": np.mean([score(["tackle", "interception", "pressure", "recover", "duel"]), score(["counterpress", "possession won", "turnover", "turnovers"])]),
        "Finisher": np.mean([score(["goal", "xg", "non-penalty", "non penalty", "np xg", "shot", "shot on target", "box touch", "touch in box"]), score(["conversion", "big chance", "penalty"])]),
        "Progressor": np.mean([score(["progressive pass", "progressive", "final third", "entry", "build", "non-penalty"]), score(["carry", "line break"])]),
        "Protector": np.mean([score(["aerial", "clearance", "block", "defensive duel", "shot block"]), score(["defensive", "duel"])]),
    }


POSITION_ARCHETYPE_MAP = {
    "AM": ["Carrier", "Connector", "Creator", "Disruptor", "Finisher", "Progressor"],
    "CM": ["Carrier", "Connector", "Disruptor", "Progressor", "Protector"],
    "CB": ["Disruptor", "Progressor", "Protector"],
    "FB": ["Carrier", "Connector", "Disruptor", "Progressor", "Protector"],
    "ST": ["Carrier", "Connector", "Creator", "Disruptor", "Finisher"],
    "WINGER": ["Carrier", "Connector", "Creator", "Disruptor", "Finisher", "Progressor"],
    "UNKNOWN": ["Carrier", "Connector", "Creator", "Disruptor", "Finisher", "Progressor", "Protector"],
}


def assign_archetypes(df: pd.DataFrame, metrics: list[str], position_col: str | None = None) -> pd.DataFrame:
    df = df.copy()
    pct_cols = [m + " Percentile" for m in metrics if m + " Percentile" in df.columns]
    if not pct_cols:
        df["Primary Archetype"] = "Unclassified"
        df["Secondary Archetype"] = "Unclassified"
        return df

    primary: list[str] = []
    secondary: list[str] = []
    for _, row in df.iterrows():
        pos_value = row[position_col] if position_col and position_col in row.index else None
        group = infer_position_group(str(pos_value) if pos_value is not None else "")
        allowed = POSITION_ARCHETYPE_MAP.get(group, POSITION_ARCHETYPE_MAP["UNKNOWN"])
        scores = archetype_scores(row, pct_cols)
        allowed_scores = {k: v for k, v in scores.items() if k in allowed}
        if not allowed_scores:
            allowed_scores = scores
        ranked = sorted(allowed_scores.items(), key=lambda kv: kv[1], reverse=True)
        primary.append(ranked[0][0] if ranked else "Unclassified")
        secondary.append(ranked[1][0] if len(ranked) > 1 else "Unclassified")

    df["Primary Archetype"] = primary
    df["Secondary Archetype"] = secondary
    return df


def plot_radar(labels: list[str], values_list: list[list[float]], labels_list: list[str]):
    if not labels:
        st.info("No metrics selected for radar chart.")
        return

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
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


def plot_pizza_like(player_label: str, df: pd.DataFrame, metrics: list[str], league_avg: list[int]):
    percentile_cols = [m + " Percentile" for m in metrics]
    player_rows = df.loc[df["Player Label"] == player_label, percentile_cols]
    if player_rows.empty:
        st.warning("Player not found for pizza chart.")
        return

    player_values = player_rows.values.flatten().tolist()
    if len(player_values) != len(metrics):
        st.warning("Selected player does not have the right number of percentile values.")
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticklabels([])
    league_vals = league_avg + league_avg[:1]
    player_vals = player_values + player_values[:1]
    ax.plot(angles, league_vals, linewidth=2, label="League Average")
    ax.fill(angles, league_vals, alpha=0.18)
    ax.plot(angles, player_vals, linewidth=2, label=player_label)
    ax.fill(angles, player_vals, alpha=0.22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
    st.pyplot(fig)
    plt.close(fig)


# ================================================================
# Tab 2: Native Scout Report helpers
# ================================================================

SCOUT_EXCLUDED = [
    "Account", "Default Radar Template", "Name", "Team", "Competition",
    "Competition Type", "Competition Rank", "Season", "Nationality",
    "Country Code", "Date of Birth", "Age", "Woman Player?",
    "Team Color 1", "Team Color 2", "Player Id", "Player Name",
    "First Name", "Last Name", "Nickname", "Weight", "Height",
    "Birth Date", "Country Id", "Country", "Team Id",
    "Team Color 1st", "Team Color 2nd", "Competition Id",
    "Competition Name", "Season Id", "Seasons",
    "Primary Position", "Secondary Position", "Most Recent Match",
    "Player SBData Id",
    "90s Played", "Appearances", "Minutes Played", "Starting Appearances",
]
LOWER_IS_BETTER = ["Turnovers", "Fouls", "Positioning Error"]


def detect_scout_metrics(data: pd.DataFrame) -> list[str]:
    if data.empty:
        return []
    return [
        k for k in data.columns
        if k not in SCOUT_EXCLUDED and pd.api.types.is_numeric_dtype(data[k])
    ]


def scout_percentiles(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    df = df.copy()
    for m in metrics:
        s = safe_numeric(df[m])
        pct = s.rank(pct=True, method="max")
        if m in LOWER_IS_BETTER:
            pct = 1 - pct
        df[m + " Percentile"] = pct.mul(100).round(0).fillna(0).astype(int)
    return df


def average_rating(row: pd.Series, metrics: list[str]) -> float:
    if not metrics:
        return 0.0
    vals = [float(row.get(f"{m} Percentile", 0)) for m in metrics]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def scout_ranking(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["Average Rating"] = out.apply(lambda r: average_rating(r, metrics), axis=1)
    return out.sort_values("Average Rating", ascending=False).reset_index(drop=True)


def category_score(player: pd.Series, filtered_df: pd.DataFrame, category_metrics: list[str], metrics: list[str]) -> int:
    existing = [m for m in category_metrics if m in metrics]
    if not existing or filtered_df.empty:
        return 0
    scores = []
    for m in existing:
        values = filtered_df[m].tolist()
        if not values:
            continue
        pct = (sum(v <= player[m] for v in values) / len(values)) * 100
        if m in LOWER_IS_BETTER:
            pct = 100 - pct
        scores.append(pct)
    return int(round(sum(scores) / len(scores))) if scores else 0


def simple_photo_name(player_name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", re.sub(r"\s+", "_", player_name.lower().strip()))


def render_pie(ax, score: int, title: str, color: str):
    ax.pie([score, max(0, 100 - score)], startangle=90, counterclock=False, colors=[color, "#e6e6e6"], wedgeprops={"linewidth": 1, "edgecolor": "white"})
    ax.text(0, 0, str(score), ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.axis("equal")


def show_scatter(df: pd.DataFrame, x: str, y: str, highlight: Optional[str] = None):
    if x not in df.columns or y not in df.columns:
        st.info("Select two valid metrics.")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x], df[y], s=35, alpha=0.7)
    if highlight and highlight in df["Player Name"].astype(str).tolist():
        row = df[df["Player Name"].astype(str) == str(highlight)].iloc[0]
        ax.scatter([row[x]], [row[y]], s=140)
        ax.annotate(str(highlight), (row[x], row[y]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close(fig)


def render_scout_summary(player: pd.Series, filtered_df: pd.DataFrame, metrics: list[str]):
    pcts = [float(player.get(f"{m} Percentile", 0)) for m in metrics]
    strengths = [m for m, p in zip(metrics, pcts) if p >= 75]
    weaknesses = [m for m, p in zip(metrics, pcts) if p <= 35]

    st.markdown("### Scout Summary – Strengths & Weaknesses")
    if strengths:
        st.success(", ".join(strengths))
    if weaknesses:
        st.error(", ".join(weaknesses))
    clips_link = player.get("ClipsLink", "") or ""
    st.text_input("Player clips link", value=str(clips_link), key=f"clips_{player.get('Player Name', 'unknown')}")


def build_shadow_position(raw_position: str, existing: list[dict]) -> str:
    text = (raw_position or "").lower()
    if any(k in text for k in ["goalkeeper", "gk"]):
        return "GK"
    if any(k in text for k in ["centre back", "center back", "cb"]):
        return "CB"
    if any(k in text for k in ["full back", "wing back", "lb", "rb", "fb"]):
        return "FB"
    if any(k in text for k in ["winger", "lw", "rw", "wide"]):
        return "LW" if not any(p.get("position") == "LW" for p in existing) else "RW"
    if any(k in text for k in ["striker", "forward", "st", "cf", "9"]):
        return "CF"
    if any(k in text for k in ["midfielder", "cm", "am", "8", "10", "6"]):
        if not any(p.get("position") == "CM1" for p in existing):
            return "CM1"
        if not any(p.get("position") == "CM2" for p in existing):
            return "CM2"
        return "CM1"
    return "CF"


# ================================================================
# Renderers
# ================================================================


def render_recruitment_dashboard():
    st.subheader("Upload data for the recruitment dashboard")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"], key="recruit_upload")
    if not uploaded_file:
        st.info("Upload a CSV or Excel file to begin.")
        return

    df = load_uploaded_file(uploaded_file)
    columns = detect_columns(df)
    if not columns.name:
        st.error("No player/name column found. Please include a column like 'Player' or 'Name'.")
        return

    work = df.copy()
    work["__row_id__"] = np.arange(len(work))
    work["__player_name__"] = work[columns.name].astype(str)
    work["__team__"] = work[columns.team].astype(str) if columns.team else ""
    work["__league__"] = work[columns.league].astype(str) if columns.league else ""
    work["__position__"] = work[columns.position].astype(str) if columns.position else ""
    work = work[work["__player_name__"].notna() & (work["__player_name__"].str.strip() != "")].copy()
    work = build_player_labels(work)

    if columns.league and columns.league in work.columns:
        league_values = sorted(work[columns.league].dropna().astype(str).unique().tolist())
        if len(league_values) > 1:
            selected_league = st.sidebar.selectbox("League", ["All leagues"] + league_values, key="rec_league")
            if selected_league != "All leagues":
                work = work[work[columns.league].astype(str) == selected_league].copy()

    if columns.position and columns.position in work.columns:
        position_values = sorted(work[columns.position].dropna().astype(str).unique().tolist())
        if len(position_values) > 1:
            selected_positions = st.sidebar.multiselect("Position", position_values, default=position_values, key="rec_pos")
            work = work[work[columns.position].astype(str).isin(selected_positions)].copy()

    if columns.minutes and columns.minutes in work.columns:
        work[columns.minutes] = safe_numeric(work[columns.minutes])
        work = work[work[columns.minutes].notna()].copy()
    if columns.age and columns.age in work.columns:
        work[columns.age] = safe_numeric(work[columns.age])
    if columns.contract_days and columns.contract_days in work.columns:
        work[columns.contract_days] = safe_numeric(work[columns.contract_days])

    all_metrics = infer_metric_columns(work)
    if not all_metrics:
        st.error("No usable numeric performance metrics were found in the uploaded file.")
        return

    st.success(f"{len(work)} rows loaded")

    if columns.minutes and columns.minutes in work.columns and work[columns.minutes].notna().any():
        mn, mx = int(work[columns.minutes].min()), int(work[columns.minutes].max())
        minutes_range = st.sidebar.slider("Minutes Played", mn, mx, (mn, mx), key="rec_minutes")
        work = work[work[columns.minutes].between(minutes_range[0], minutes_range[1])].copy()
    if columns.age and columns.age in work.columns and work[columns.age].notna().any():
        mn, mx = int(work[columns.age].min()), int(work[columns.age].max())
        age_range = st.sidebar.slider("Age", mn, mx, (mn, mx), key="rec_age")
        work = work[work[columns.age].between(age_range[0], age_range[1])].copy()

    if columns.team and columns.team in work.columns:
        team_values = sorted(work[columns.team].dropna().astype(str).unique().tolist())
        if len(team_values) > 1:
            selected_teams = st.sidebar.multiselect("Team", team_values, default=team_values, key="rec_team")
            work = work[work[columns.team].astype(str).isin(selected_teams)].copy()

    st.sidebar.subheader("Metrics")
    default_metrics = all_metrics[: min(10, len(all_metrics))]
    metrics = st.sidebar.multiselect("Select Metrics", all_metrics, default=default_metrics, key="rec_metrics")
    if not metrics:
        st.warning("Select at least one metric")
        return

    for m in metrics:
        work[m] = safe_numeric(work[m])

    work = add_percentiles(work, metrics)
    percentile_cols = [m + " Percentile" for m in metrics]
    work = assign_archetypes(work, metrics, position_col=columns.position)
    league_avg = compute_league_average(work, metrics)
    work["Overall Score"] = work[percentile_cols].mean(axis=1).round(0)
    work = work.sort_values("Overall Score", ascending=False).reset_index(drop=True)
    work.index += 1
    work.insert(0, "Rank", work.index)

    rank_view = build_rank_view(work, columns.name, columns.team, columns.league, columns.position)
    rank_view["Player Label"] = work["Player Label"].values
    rank_view["Transfermarkt Link"] = [make_transfermarkt_url(n) for n in rank_view["Display Name"].astype(str)]
    active_player_label = rank_view.iloc[0]["Player Label"] if len(rank_view) else None

    st.title("⚽ Football Recruitment Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Players", f"{len(work)}")
    c2.metric("Metrics", f"{len(metrics)}")
    c3.metric("Top Score", f"{int(work['Overall Score'].max()) if len(work) else 0}")

    st.subheader("🏅 Player Ranking")
    player_search = st.text_input("Search player", value="", placeholder="Type a player name", key="rec_search")
    rank_view_display = rank_view.copy()
    if player_search.strip():
        search_text = player_search.strip()
        rank_view_display = rank_view_display[
            rank_view_display["Display Name"].astype(str).str.contains(search_text, case=False, na=False)
            | rank_view_display["Player Label"].astype(str).str.contains(search_text, case=False, na=False)
            | rank_view_display["Display Team"].astype(str).str.contains(search_text, case=False, na=False)
        ].copy()

    show_cols = ["Rank", "Player Label", "Primary Archetype", "Secondary Archetype", "Display Team", "Display League", "Display Position", "Overall Score"]
    for maybe_col in [columns.age, columns.minutes, columns.contract_days, columns.contract_date]:
        if maybe_col:
            show_cols.append(maybe_col)
    show_cols.extend(metrics)
    show_cols.append("Transfermarkt Link")
    show_cols = [c for c in show_cols if c in rank_view_display.columns]

    st.dataframe(
        percentile_style_table(rank_view_display, show_cols),
        use_container_width=True,
        hide_index=True,
        column_config={"Transfermarkt Link": st.column_config.LinkColumn("Transfermarkt Link")},
    )

    st.subheader("⭐ Top Performers")
    top_players = []
    for m in metrics:
        s = safe_numeric(work[m])
        if s.notna().any():
            top_idx = s.idxmax()
            top_row = work.loc[top_idx]
            top_players.append({
                "Metric": m,
                "Player Label": top_row["Player Label"],
                "Primary Archetype": top_row.get("Primary Archetype", "Unclassified"),
                "Secondary Archetype": top_row.get("Secondary Archetype", "Unclassified"),
                "Value": top_row[m],
            })
    st.dataframe(pd.DataFrame(top_players), use_container_width=True, hide_index=True)

    player_labels = work["Player Label"].tolist()
    if player_labels:
        st.subheader("📊 Pizza Chart")
        selected_player = st.selectbox("Select Player", player_labels, index=0, key="rec_player")
        plot_pizza_like(selected_player, work, metrics, league_avg)

        st.subheader("📈 Player Comparison")
        p1 = st.selectbox("Player 1", player_labels, index=0, key="rec_p1")
        p2 = st.selectbox("Player 2", player_labels, index=1 if len(player_labels) > 1 else 0, key="rec_p2")
        if p1 != p2 and len(metrics) > 0:
            vals1 = work.loc[work["Player Label"] == p1, percentile_cols].values.flatten().tolist()
            vals2 = work.loc[work["Player Label"] == p2, percentile_cols].values.flatten().tolist()
            if len(vals1) == len(metrics) and len(vals2) == len(metrics):
                plot_radar(metrics, [vals1, vals2], [p1, p2])
    else:
        st.info("No players available in the current filtered set.")


def render_native_scout_report():
    st.subheader("Native Streamlit Scout Report")
    uploaded_file = st.file_uploader("Upload scouting CSV", type=["csv", "xlsx", "xls"], key="scout_upload")
    if not uploaded_file:
        st.info("Upload a scouting CSV or Excel file to begin.")
        return

    df = load_uploaded_file(uploaded_file)
    if df.empty:
        st.warning("No data found in the file.")
        return

    # Normalise common column names from the React app.
    rename_map = {
        "Player Name": "Player Name",
        "Team": "Team",
        "Competition Name": "Competition Name",
        "Primary Position": "Primary Position",
        "Minutes Played": "Minutes Played",
    }
    df = df.rename(columns=rename_map)

    if "Player Name" not in df.columns:
        fallback = next((c for c in ["Name", "Player", "player", "name"] if c in df.columns), None)
        if fallback:
            df["Player Name"] = df[fallback].astype(str)
        else:
            st.error("No player name column found. Expected 'Player Name' or 'Name'.")
            return

    if "Primary Position" not in df.columns:
        pos_fallback = next((c for c in ["Position", "Role", "Primary Position"] if c in df.columns), None)
        if pos_fallback:
            df["Primary Position"] = df[pos_fallback].astype(str)
        else:
            df["Primary Position"] = "Unknown"

    metrics = detect_scout_metrics(df)
    if not metrics:
        st.error("No numeric scouting metrics found in the uploaded file.")
        return

    # Clean and compute percentiles.
    for m in metrics:
        df[m] = safe_numeric(df[m])
    df = df.dropna(subset=["Player Name"]).copy()
    df = scout_percentiles(df, metrics)

    competitions = ["All"] + sorted([str(v) for v in df.get("Competition Name", pd.Series(dtype=str)).dropna().unique().tolist()])
    positions = sorted([str(v) for v in df.get("Primary Position", pd.Series(dtype=str)).dropna().unique().tolist()])

    with st.sidebar:
        st.markdown("### Scout Filters")
        competition = st.selectbox("Competition", competitions, key="scout_competition")
        selected_positions = st.multiselect("Position", positions, default=positions, key="scout_positions")
        min_minutes = st.number_input("Min Minutes", min_value=0, value=0, step=50, key="scout_min_minutes")
        max_minutes = st.number_input("Max Minutes", min_value=0, value=99999, step=50, key="scout_max_minutes")
        show_shadow = st.checkbox("Show Shadow Squad", value=True, key="scout_shadow_toggle")

    filtered = df.copy()
    if competition != "All" and "Competition Name" in filtered.columns:
        filtered = filtered[filtered["Competition Name"].astype(str) == competition].copy()
    if selected_positions:
        filtered = filtered[filtered["Primary Position"].astype(str).isin(selected_positions)].copy()
    if "Minutes Played" in filtered.columns:
        filtered["Minutes Played"] = safe_numeric(filtered["Minutes Played"])
        filtered = filtered[(filtered["Minutes Played"].fillna(0) >= min_minutes) & (filtered["Minutes Played"].fillna(0) <= max_minutes)].copy()

    if filtered.empty:
        st.warning("No players match the current filters.")
        return

    # Session state for shadow squad.
    if "shadow_squad" not in st.session_state:
        st.session_state.shadow_squad = []

    ranking = scout_ranking(filtered, metrics)
    ranking["Average Rating"] = ranking["Average Rating"].round(1)

    col_left, col_right = st.columns([1.05, 1.3])
    with col_left:
        st.metric("Players in view", len(filtered))
        st.metric("Metrics detected", len(metrics))
        st.markdown("#### Top Rated Players")
        top_list = ranking[["Player Name", "Average Rating", "Primary Position"]].head(8)
        st.dataframe(top_list, use_container_width=True, hide_index=True)

    player_options = ranking["Player Name"].astype(str).tolist()
    selected_name = st.selectbox("Select player", player_options, index=0, key="scout_player")
    player = ranking[ranking["Player Name"].astype(str) == str(selected_name)].iloc[0]

    # Add to shadow squad.
    add_col, export_col = st.columns([0.4, 0.6])
    with add_col:
        if st.button("➕ Add to Shadow Squad", key="add_shadow"):
            exists = any(item.get("playerName") == selected_name for item in st.session_state.shadow_squad)
            if exists:
                st.warning("Player already in Shadow Squad")
            else:
                raw_position = str(player.get("Primary Position", ""))
                final_position = build_shadow_position(raw_position, st.session_state.shadow_squad)
                st.session_state.shadow_squad.append({
                    "id": player.get("Player Id", selected_name),
                    "playerName": selected_name,
                    "team": str(player.get("Team", player.get("Club", player.get("Squad", "Unknown")))),
                    "position": final_position,
                    "fullPosition": raw_position,
                    "reportUrl": str(player.get("Report URL", "")),
                    "raw": player.to_dict(),
                })
                st.success("Added to Shadow Squad")

    with export_col:
        # lightweight export of a text report (works anywhere; no extra deps)
        report_lines = [
            f"Player: {selected_name}",
            f"Team: {player.get('Team', '')}",
            f"Competition: {player.get('Competition Name', '')}",
            f"Position: {player.get('Primary Position', '')}",
            f"Average Rating: {player.get('Average Rating', '')}",
            "",
            "Strengths:",
            ", ".join([m for m in metrics if float(player.get(f'{m} Percentile', 0)) >= 75]) or "None",
            "",
            "Weaknesses:",
            ", ".join([m for m in metrics if float(player.get(f'{m} Percentile', 0)) <= 35]) or "None",
        ]
        st.download_button(
            "Download player summary",
            data="\n".join(report_lines),
            file_name=f"{simple_photo_name(selected_name) or 'player'}_summary.txt",
            mime="text/plain",
            key="download_summary",
        )

    with col_left:
        st.markdown("#### Player Profile")
        profile_cols = [c for c in ["Player Name", "Team", "Competition Name", "Primary Position", "Age", "Nationality", "Appearances", "Minutes Played", "Average Rating"] if c in ranking.columns]
        st.dataframe(player[profile_cols].to_frame().T, use_container_width=True, hide_index=True)

        st.markdown("#### Category scores")
        attack_metrics = [m for m in metrics if re.search(r"goal|assist|shot|key pass|xg", m, re.I)]
        defend_metrics = [m for m in metrics if re.search(r"tackle|intercept|clearance|block|aerial|pressures", m, re.I)]
        carry_metrics = [m for m in metrics if re.search(r"dribble|carry|progressive|pass", m, re.I)]
        if str(player.get("Primary Position", "")).lower() in {"goalkeeper", "gk"}:
            saving_metrics = [m for m in metrics if re.search(r"save|shot stopped|penalty save", m, re.I)]
            claims_metrics = [m for m in metrics if re.search(r"claim|catch|cross", m, re.I)]
            passing_metrics = [m for m in metrics if re.search(r"pass|distribution", m, re.I)]
            cats = [
                ("Saving", category_score(player, filtered, saving_metrics, metrics), "#2ecc71"),
                ("Claims", category_score(player, filtered, claims_metrics, metrics), "#e74c3c"),
                ("Passing", category_score(player, filtered, passing_metrics, metrics), "#1f77b4"),
            ]
        else:
            cats = [
                ("Attacking", category_score(player, filtered, attack_metrics, metrics), "#2ecc71"),
                ("Defending", category_score(player, filtered, defend_metrics, metrics), "#e74c3c"),
                ("Ball Carrying", category_score(player, filtered, carry_metrics, metrics), "#1f77b4"),
            ]
        pie_cols = st.columns(len(cats))
        for col, (label, score, color) in zip(pie_cols, cats):
            with col:
                fig, ax = plt.subplots(figsize=(2.2, 2.2))
                render_pie(ax, score, label, color)
                st.pyplot(fig)
                plt.close(fig)

    with col_right:
        st.markdown("#### Metric Percentiles")
        for m in metrics:
            pct = int(player.get(f"{m} Percentile", 0))
            raw = player.get(m, 0)
            st.write(f"**{m}**  •  {raw}")
            st.progress(pct / 100.0)
            st.caption(f"Percentile: {pct}%")

        st.markdown("#### Radar / Pizza comparison")
        pcts = [int(player.get(f"{m} Percentile", 0)) for m in metrics]
        league_avg = [int(round((filtered[m].fillna(filtered[m].mean()) < filtered[m].mean()).sum() / len(filtered) * 100)) if len(filtered) else 0 for m in metrics]
        if metrics:
            plot_radar(metrics, [pcts, league_avg], [selected_name, "League Avg"])

        st.markdown("#### Scatter")
        scatter_options = metrics[:]
        x = st.selectbox("Scatter X", scatter_options, index=0 if scatter_options else 0, key="scatter_x")
        y = st.selectbox("Scatter Y", scatter_options, index=1 if len(scatter_options) > 1 else 0, key="scatter_y")
        if x and y:
            show_scatter(filtered, x, y, highlight=selected_name)

        st.markdown("#### Scout Summary")
        render_scout_summary(player, filtered, metrics)

    if show_shadow:
        st.markdown("### Shadow Squad")
        if st.session_state.shadow_squad:
            shadow_df = pd.DataFrame(st.session_state.shadow_squad)
            st.dataframe(shadow_df[[c for c in ["playerName", "team", "position", "fullPosition", "reportUrl"] if c in shadow_df.columns]], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Shadow Squad CSV",
                data=shadow_df.to_csv(index=False),
                file_name="shadow_squad.csv",
                mime="text/csv",
                key="shadow_csv",
            )
        else:
            st.info("No players added yet.")


# ================================================================
# App shell
# ================================================================

st.title("Integrated Football Apps")
st.caption("Two tabs: your recruitment dashboard and a native Streamlit version of the scouting report page.")

recruit_tab, scout_tab = st.tabs(["Recruitment Dashboard", "Scout Report"])

with recruit_tab:
    render_recruitment_dashboard()

with scout_tab:
    render_native_scout_report()



