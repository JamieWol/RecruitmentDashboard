"""
Generic Football Dashboard (Streamlit)

Run locally with:
    streamlit run app.py

What this app does:
- Upload CSV, TSV, TXT, Excel, JSON, Parquet, or Feather files
- For Excel workbooks, pick one sheet or combine all sheets
- Auto-detect player/team/league/position/minutes/age/contract columns
- Auto-detect numeric performance metrics
- Build primary/secondary archetypes from position and metrics
- Rank players, build a player report, and create a simple shadow-squad tab
- Show duplicate player names as Name (Team)
- Keep the dashboard modular with separate tabs
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


st.set_page_config(page_title="Generic Football Dashboard", layout="wide")


# ---------------------------------------------------------------------
# COLUMN DETECTION
# ---------------------------------------------------------------------
NAME_CANDIDATES = ["Player", "Name", "player", "name", "Footballer", "Player Name"]
TEAM_CANDIDATES = ["Team", "Club", "Squad", "team", "club", "Current Team", "Team Name"]
LEAGUE_CANDIDATES = ["League", "Competition", "competition", "league", "Competition Name"]
POSITION_CANDIDATES = ["Position", "Primary Position", "Role", "position"]
MINUTES_CANDIDATES = [
    "Minutes played", "Minutes", "mins", "Min", "minutes played", "Minutes Played", "Minutes (Last 2 years)",
]
AGE_CANDIDATES = ["Age", "age"]
CONTRACT_DAYS_CANDIDATES = [
    "Contract Expiry (days left)", "Contract expiry (days left)", "Contract days left", "Days left on contract",
]
CONTRACT_DATE_CANDIDATES = ["Contract expires", "Contract Expiry", "Contract end", "Expiry"]
SHEET_COL = "__sheet__"


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


# ---------------------------------------------------------------------
# DATA HELPERS
# ---------------------------------------------------------------------
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~df.columns.duplicated()].copy()


def file_extension(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


@st.cache_data(show_spinner=False)
def load_dataset(file_bytes: bytes, filename: str, sheet_choice: str | None) -> pd.DataFrame:
    ext = file_extension(filename)

    if ext in {"csv", "tsv", "txt"}:
        sep = "\t" if ext in {"tsv", "txt"} else ","
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep, engine="python")
        return clean_columns(df)

    if ext in {"xlsx", "xls"}:
        workbook = pd.ExcelFile(io.BytesIO(file_bytes))
        if sheet_choice and sheet_choice != "All sheets (combine)":
            df = pd.read_excel(workbook, sheet_name=sheet_choice)
            df = clean_columns(df)
            df[SHEET_COL] = sheet_choice
            return df

        # Combine all sheets into one table with a sheet label.
        frames: list[pd.DataFrame] = []
        for sheet in workbook.sheet_names:
            part = pd.read_excel(workbook, sheet_name=sheet)
            part = clean_columns(part)
            part[SHEET_COL] = sheet
            frames.append(part)
        if not frames:
            return pd.DataFrame()
        return clean_columns(pd.concat(frames, ignore_index=True, sort=False))

    if ext == "json":
        # Supports common JSON exports; if the file is line-delimited, pandas will also try that.
        try:
            return clean_columns(pd.read_json(io.BytesIO(file_bytes)))
        except ValueError:
            return clean_columns(pd.read_json(io.BytesIO(file_bytes), lines=True))

    if ext == "parquet":
        return clean_columns(pd.read_parquet(io.BytesIO(file_bytes)))

    if ext in {"feather", "arrow"}:
        return clean_columns(pd.read_feather(io.BytesIO(file_bytes)))

    raise ValueError(f"Unsupported file type: .{ext}")


def get_sheet_names(file_bytes: bytes, filename: str) -> list[str]:
    ext = file_extension(filename)
    if ext not in {"xlsx", "xls"}:
        return []
    workbook = pd.ExcelFile(io.BytesIO(file_bytes))
    return workbook.sheet_names


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
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def make_transfermarkt_url(player_name: str) -> str:
    return "https://www.transfermarkt.com/schnellsuche/ergebnis/schnellsuche?query=" + quote_plus(player_name.strip())


def build_player_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    counts = df["__player_name__"].astype(str).value_counts(dropna=False)
    labels: list[str] = []
    for idx, row in df.iterrows():
        name = str(row.get("__player_name__", "")).strip()
        team = str(row.get("__team__", "")).strip()
        sheet = str(row.get(SHEET_COL, "")).strip()
        if counts.get(name, 0) > 1:
            suffix = team or sheet or f"Row {idx + 1}"
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


# ---------------------------------------------------------------------
# METRIC INFERENCE
# ---------------------------------------------------------------------
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
        SHEET_COL,
    }

    exclude_keywords = [
        "id", "name", "team", "club", "squad", "player", "match",
        "season", "league", "competition", "birth", "height", "weight",
        "passport", "country", "foot", "shirt", "age", "position", "role", "minute",
    ]

    include_exact = {
        "Turnovers",
        "Non-Penalty Goals",
        "Scoring Contribution",
        "Goal Conversion%",
        "xG",
        "xG/Shot",
        "Aerial Win%",
        "PAdj Pressures",
        "Pressure Regains",
        "Touches In Box",
        "OBV",
        "Key Passes",
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


# ---------------------------------------------------------------------
# TRANSFORMS
# ---------------------------------------------------------------------
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


def get_show_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return unique_preserve_order([c for c in cols if c in df.columns])


def display_table(df: pd.DataFrame, cols: list[str]) -> None:
    cols = get_show_columns(df, cols)
    if not cols:
        st.info("No columns to display.")
        return
    st.dataframe(df.loc[:, ~df.columns.duplicated()][cols], use_container_width=True, hide_index=True)


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


def compute_similarity_frame(df: pd.DataFrame, player_label: str, metrics: list[str], top_n: int = 10) -> pd.DataFrame:
    if not metrics or "Player Label" not in df.columns:
        return pd.DataFrame()

    working_cols = [c for c in ["Player Label", "Display Name", "Display Team", "Display League", "Display Position", "Primary Archetype", "Secondary Archetype"] if c in df.columns]
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

    player_row = working[working["Player Label"] == player_label]
    if player_row.empty:
        return pd.DataFrame()

    player_vector = player_row.iloc[0][metric_cols].to_numpy(dtype=float)
    other_vectors = working[metric_cols].to_numpy(dtype=float)
    distances = np.linalg.norm(other_vectors - player_vector, axis=1)
    working["Similarity Score"] = 1 / (1 + distances)
    working = working[working["Player Label"] != player_label].sort_values("Similarity Score", ascending=False)
    return working.head(top_n)


# ---------------------------------------------------------------------
# POSITION-BASED ARCHETYPES
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# CHARTS
# ---------------------------------------------------------------------
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


def plot_pizza_or_radar(player_label: str, df: pd.DataFrame, metrics: list[str], league_avg: list[int]):
    percentile_cols = [m + " Percentile" for m in metrics]
    player_rows = df.loc[df["Player Label"] == player_label, percentile_cols]
    if player_rows.empty:
        st.warning("Player not found for chart.")
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
            kwargs_values=dict(fontsize=11, fontweight="bold", color="black", bbox=dict(edgecolor="black", facecolor="#facc15", boxstyle="round,pad=0.25")),
        )
        pizza.make_pizza(
            player_values,
            ax=ax,
            kwargs_slices=dict(facecolor="#3b82f6", edgecolor="black", linewidth=2, alpha=0.9),
            kwargs_values=dict(fontsize=11, fontweight="bold", color="white", bbox=dict(edgecolor="black", facecolor="#3b82f6", boxstyle="round,pad=0.25")),
        )
        ax.legend(handles=[plt.Line2D([0], [0], color="#3b82f6", lw=10, label=player_label), plt.Line2D([0], [0], color="#facc15", lw=10, label="League Average")], loc="upper right", bbox_to_anchor=(1.15, 1.1))
        st.pyplot(fig)
        plt.close(fig)
        return

    # Fallback radar chart when mplsoccer is not available.
    plot_radar(metrics, [league_avg, player_values], ["League Average", player_label])


# ---------------------------------------------------------------------
# SIMPLE SHADOW SQUAD ASSIGNMENT
# ---------------------------------------------------------------------
FORMATION_SLOTS = {
    "3-4-3": [
        {"id": "GK", "label": "GK", "base": "GK", "side": "C"},
        {"id": "LCB", "label": "LCB", "base": "CB", "side": "L"},
        {"id": "CB", "label": "CB", "base": "CB", "side": "C"},
        {"id": "RCB", "label": "RCB", "base": "CB", "side": "R"},
        {"id": "LWB", "label": "LWB", "base": "FB", "side": "L"},
        {"id": "LCM", "label": "LCM", "base": "CM", "side": "L"},
        {"id": "RCM", "label": "RCM", "base": "CM", "side": "R"},
        {"id": "RWB", "label": "RWB", "base": "FB", "side": "R"},
        {"id": "L10", "label": "L10", "base": "AM", "side": "L"},
        {"id": "R10", "label": "R10", "base": "AM", "side": "R"},
        {"id": "CF", "label": "CF", "base": "ST", "side": "C"},
    ]
}


def infer_shadow_role(position_text: str) -> tuple[str, str]:
    group = infer_position_group(position_text)
    if group == "CB":
        return "CB", "C"
    if group == "FB":
        return "FB", "L"
    if group == "CM":
        return "CM", "C"
    if group == "AM":
        return "AM", "C"
    if group == "WINGER":
        return "AM", "L"
    if group == "ST":
        return "ST", "C"
    return "ST", "C"


def auto_assign_shadow_slot(player_row: pd.Series, current_assignments: dict[str, str], formation_name: str) -> Optional[str]:
    slots = FORMATION_SLOTS.get(formation_name, [])
    taken = set(current_assignments.values())
    role, side = infer_shadow_role(str(player_row.get("Display Position", "") or player_row.get("__position__", "")))

    candidates = [
        lambda s: s["base"] == role and s["side"] == side,
        lambda s: s["base"] == role and s["side"] == "C",
        lambda s: s["base"] == role,
    ]
    for pred in candidates:
        for s in slots:
            if s["id"] not in taken and pred(s):
                return s["id"]
    return None


# ---------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------
if "shadow_squad" not in st.session_state:
    st.session_state.shadow_squad = []


# ---------------------------------------------------------------------
# SIDEBAR: FILE UPLOAD AND SOURCE SELECTION
# ---------------------------------------------------------------------
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file",
    type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet", "feather"],
)

if not uploaded_file:
    st.title("⚽ Generic Football Dashboard")
    st.info("Upload a file to begin.")
    st.stop()

file_bytes = uploaded_file.getvalue()
filename = uploaded_file.name
sheet_options = get_sheet_names(file_bytes, filename)

sheet_choice: str | None = None
if sheet_options:
    sheet_choice = st.sidebar.selectbox("Excel sheet", ["All sheets (combine)"] + sheet_options)

try:
    df = load_dataset(file_bytes, filename, sheet_choice)
except Exception as exc:
    st.error(f"Could not read file: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded file did not contain any rows.")
    st.stop()

# If all sheets were combined, allow filtering by sheet afterwards.
if SHEET_COL in df.columns:
    sheet_filter_values = ["All sheets"] + sorted(df[SHEET_COL].dropna().astype(str).unique().tolist())
    sheet_filter = st.sidebar.selectbox("Filter sheet", sheet_filter_values)
    if sheet_filter != "All sheets":
        df = df[df[SHEET_COL].astype(str) == sheet_filter].copy()

columns = detect_columns(df)
if not columns.name:
    st.error("No player/name column found. Include a column like 'Player', 'Name', or 'Player Name'.")
    st.stop()

# ---------------------------------------------------------------------
# STANDARDISE THE TABLE
# ---------------------------------------------------------------------
work = df.copy()
work["__row_id__"] = np.arange(len(work))
work["__player_name__"] = work[columns.name].astype(str)
work["__team__"] = work[columns.team].astype(str) if columns.team else ""
work["__league__"] = work[columns.league].astype(str) if columns.league else ""
work["__position__"] = work[columns.position].astype(str) if columns.position else ""
work = work[work["__player_name__"].notna() & (work["__player_name__"].str.strip() != "")].copy()
work = build_player_labels(work)

# Sidebar filters
st.sidebar.subheader("Filters")
if columns.league and columns.league in work.columns:
    league_values = sorted(work[columns.league].dropna().astype(str).unique().tolist())
    if len(league_values) > 1:
        selected_leagues = st.sidebar.multiselect("League", league_values, default=league_values)
        work = work[work[columns.league].astype(str).isin(selected_leagues)].copy()

if columns.team and columns.team in work.columns:
    team_values = sorted(work[columns.team].dropna().astype(str).unique().tolist())
    if len(team_values) > 1:
        selected_teams = st.sidebar.multiselect("Team", team_values, default=team_values)
        work = work[work[columns.team].astype(str).isin(selected_teams)].copy()

if columns.position and columns.position in work.columns:
    position_values = sorted(work[columns.position].dropna().astype(str).unique().tolist())
    if len(position_values) > 1:
        selected_positions = st.sidebar.multiselect("Position", position_values, default=position_values)
        work = work[work[columns.position].astype(str).isin(selected_positions)].copy()

if columns.minutes and columns.minutes in work.columns:
    work[columns.minutes] = safe_numeric(work[columns.minutes])
    if work[columns.minutes].notna().any():
        mn, mx = int(work[columns.minutes].min()), int(work[columns.minutes].max())
        min_range = st.sidebar.slider("Minutes Played", mn, mx, (mn, mx))
        work = work[work[columns.minutes].between(min_range[0], min_range[1])].copy()

if columns.age and columns.age in work.columns:
    work[columns.age] = safe_numeric(work[columns.age])
    if work[columns.age].notna().any():
        mn, mx = int(work[columns.age].min()), int(work[columns.age].max())
        age_range = st.sidebar.slider("Age", mn, mx, (mn, mx))
        work = work[work[columns.age].between(age_range[0], age_range[1])].copy()

if columns.contract_days and columns.contract_days in work.columns:
    work[columns.contract_days] = safe_numeric(work[columns.contract_days])
if columns.contract_date and columns.contract_date in work.columns:
    work[columns.contract_date] = work[columns.contract_date].astype(str)

market_mode = st.sidebar.checkbox("Market opportunity filter", value=False)

# Metric selection
all_metrics = infer_metric_columns(work)
if not all_metrics:
    st.error("No usable numeric performance metrics were found in the uploaded file.")
    st.stop()

st.sidebar.subheader("Metrics")
auto_mode = st.sidebar.checkbox("Auto-detect metrics", value=True)
if auto_mode:
    default_metrics = all_metrics[: min(10, len(all_metrics))]
    metrics = st.sidebar.multiselect("Select Metrics", all_metrics, default=default_metrics)
else:
    numeric_cols = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c])]
    default_metrics = numeric_cols[: min(10, len(numeric_cols))]
    metrics = st.sidebar.multiselect("Select Metrics", numeric_cols, default=default_metrics)
if not metrics:
    st.warning("Select at least one metric")
    st.stop()

for m in metrics:
    work[m] = safe_numeric(work[m])

# Apply market filter after metrics are in place.
if market_mode:
    age_ok = (
        work[columns.age] < 25
        if columns.age and columns.age in work.columns and work[columns.age].notna().any()
        else pd.Series(True, index=work.index)
    )
    if columns.contract_days and columns.contract_days in work.columns and work[columns.contract_days].notna().any():
        contract_ok = work[columns.contract_days] < 365
    elif columns.contract_date and columns.contract_date in work.columns:
        contract_ok = pd.to_datetime(work[columns.contract_date], errors="coerce").notna()
    else:
        contract_ok = pd.Series(True, index=work.index)
    work = work[age_ok & contract_ok].copy()

work = add_percentiles(work, metrics)
work = assign_archetypes(work, metrics, position_col=columns.position)
league_avg = compute_league_average(work, metrics)
percentile_cols = [m + " Percentile" for m in metrics]
work["Overall Score"] = work[percentile_cols].mean(axis=1).round(0)
work = work.sort_values("Overall Score", ascending=False).reset_index(drop=True)
work.index += 1
work.insert(0, "Rank", work.index)

# Shared display table
rank_view = build_rank_view(work, columns.name, columns.team, columns.league, columns.position)
rank_view["Player Label"] = work["Player Label"].values
rank_view["Transfermarkt Link"] = [make_transfermarkt_url(n) for n in rank_view["Display Name"].astype(str)]

if "active_player_label" not in st.session_state and len(rank_view) > 0:
    st.session_state["active_player_label"] = rank_view.iloc[0]["Player Label"]
active_player_label = st.session_state.get("active_player_label")

# ---------------------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------------------
st.title("⚽ Generic Football Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(work)}")
c2.metric("Metrics", f"{len(metrics)}")
c3.metric("Sheets", f"{len(sheet_options) if sheet_options else 1}")
c4.metric("Top Score", f"{int(work['Overall Score'].max()) if len(work) else 0}")

# Tabs
rank_tab, report_tab, squad_tab, data_tab = st.tabs(["Rankings", "Scout Report", "Shadow Squad", "Data Explorer"])

# ---------------------------------------------------------------------
# TAB 1: RANKINGS
# ---------------------------------------------------------------------
with rank_tab:
    st.subheader("🏅 Player Rankings")
    search_text = st.text_input("Search player", value="", placeholder="Type a player name")

    rank_view_display = rank_view.copy()
    if search_text.strip():
        q = search_text.strip()
        rank_view_display = rank_view_display[
            rank_view_display["Display Name"].astype(str).str.contains(q, case=False, na=False)
            | rank_view_display["Player Label"].astype(str).str.contains(q, case=False, na=False)
            | rank_view_display["Display Team"].astype(str).str.contains(q, case=False, na=False)
        ].copy()

    show_cols = [
        "Rank",
        "Player Label",
        "Primary Archetype",
        "Secondary Archetype",
        "Display Team",
        "Display League",
        "Display Position",
        "Overall Score",
    ]
    if columns.minutes:
        show_cols.append(columns.minutes)
    if columns.age:
        show_cols.append(columns.age)
    if columns.contract_days:
        show_cols.append(columns.contract_days)
    if columns.contract_date:
        show_cols.append(columns.contract_date)
    show_cols += metrics
    show_cols.append("Transfermarkt Link")
    show_cols = get_show_columns(rank_view_display, show_cols)

    st.dataframe(
        percentile_style_table(rank_view_display, show_cols),
        use_container_width=True,
        hide_index=True,
        column_config={"Transfermarkt Link": st.column_config.LinkColumn("Transfermarkt Link")},
    )

    st.caption("Duplicate names appear as Name (Team/Sheet).")

# ---------------------------------------------------------------------
# TAB 2: SCOUT REPORT
# ---------------------------------------------------------------------
with report_tab:
    st.subheader("📝 Scout Report")
    player_labels = work["Player Label"].tolist()
    if not player_labels:
        st.info("No players available.")
    else:
        selected_player_label = st.selectbox(
            "Select a player",
            player_labels,
            index=player_labels.index(active_player_label) if active_player_label in player_labels else 0,
            key="scout_player_select",
        )
        st.session_state["active_player_label"] = selected_player_label

        player_row = work[work["Player Label"] == selected_player_label].iloc[0]
        summary_cols = st.columns(4)
        summary_cols[0].metric("Overall Score", f"{int(player_row['Overall Score'])}")
        summary_cols[1].metric("Primary", str(player_row.get("Primary Archetype", "-")))
        summary_cols[2].metric("Secondary", str(player_row.get("Secondary Archetype", "-")))
        summary_cols[3].metric("Transfermarkt", "Open link in rankings")

        left, right = st.columns([1, 1])
        with left:
            if columns.team and columns.team in player_row.index:
                st.write(f"**Team:** {player_row.get('Display Team', '')}")
            if columns.league and columns.league in player_row.index:
                st.write(f"**League:** {player_row.get('Display League', '')}")
            if columns.position and columns.position in player_row.index:
                st.write(f"**Position:** {player_row.get('Display Position', '')}")
            if columns.age and columns.age in player_row.index:
                st.write(f"**Age:** {player_row.get(columns.age, '')}")
            if columns.minutes and columns.minutes in player_row.index:
                st.write(f"**Minutes:** {player_row.get(columns.minutes, '')}")

            st.markdown("### Strengths / Weaknesses")
            pcts = [
                (m, int(player_row[m + " Percentile"]))
                for m in metrics
                if m + " Percentile" in player_row.index and pd.notna(player_row[m + " Percentile"])
            ]
            strengths = [m for m, v in pcts if v >= 75]
            weaknesses = [m for m, v in pcts if v <= 35]
            st.write(f"**Strengths:** {', '.join(strengths) if strengths else 'None highlighted'}")
            st.write(f"**Weaknesses:** {', '.join(weaknesses) if weaknesses else 'None highlighted'}")

            if "Transfermarkt Link" in rank_view.columns:
                link = rank_view.loc[rank_view["Player Label"] == selected_player_label, "Transfermarkt Link"].iloc[0]
                st.markdown(f"[Open Transfermarkt link]({link})")

        with right:
            st.markdown("### Player Chart")
            plot_pizza_or_radar(selected_player_label, work, metrics, league_avg)

            st.markdown("### Metric Percentiles")
            if pcts:
                metric_df = pd.DataFrame(pcts, columns=["Metric", "Percentile"])
                st.dataframe(metric_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 3: SHADOW SQUAD
# ---------------------------------------------------------------------
with squad_tab:
    st.subheader("🟩 Shadow Squad")
    formation = st.selectbox("Formation", list(FORMATION_SLOTS.keys()), index=0)
    available_players = work["Player Label"].tolist()

    squad_pick = st.multiselect(
        "Pick players to add",
        available_players,
        default=[active_player_label] if active_player_label in available_players else [],
    )

    if st.button("Add selected players"):
        current_ids = {p["id"] for p in st.session_state.shadow_squad}
        for label in squad_pick:
            if any(p["label"] == label for p in st.session_state.shadow_squad):
                continue
            row = work[work["Player Label"] == label].iloc[0]
            slot = auto_assign_shadow_slot(row, {p["label"]: p.get("slot", "") for p in st.session_state.shadow_squad}, formation)
            st.session_state.shadow_squad.append(
                {
                    "id": str(row.get("__row_id__", label)),
                    "label": label,
                    "name": str(row.get("__player_name__", label)),
                    "team": str(row.get("Display Team", row.get("__team__", ""))),
                    "slot": slot or "Bench",
                    "primary": str(row.get("Primary Archetype", "-")),
                    "secondary": str(row.get("Secondary Archetype", "-")),
                    "sheet": str(row.get(SHEET_COL, "")),
                }
            )
        st.success("Selected players added to the shadow squad.")

    if st.button("Clear shadow squad"):
        st.session_state.shadow_squad = []
        st.success("Shadow squad cleared.")

    if st.session_state.shadow_squad:
        squad_df = pd.DataFrame(st.session_state.shadow_squad)
        st.dataframe(squad_df[["label", "team", "slot", "primary", "secondary", "sheet"]], use_container_width=True, hide_index=True)
    else:
        st.info("No players in the shadow squad yet.")

    st.markdown("### Formation slots")
    slot_rows = []
    taken = {p["slot"]: p for p in st.session_state.shadow_squad if p.get("slot") and p.get("slot") != "Bench"}
    for slot in FORMATION_SLOTS[formation]:
        occupant = taken.get(slot["id"])
        slot_rows.append(
            {
                "Slot": slot["label"],
                "Player": occupant["label"] if occupant else "",
                "Team": occupant["team"] if occupant else "",
                "Archetype": occupant["primary"] if occupant else "",
            }
        )
    st.dataframe(pd.DataFrame(slot_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 4: DATA EXPLORER
# ---------------------------------------------------------------------
with data_tab:
    st.subheader("📂 Data Explorer")
    st.write(f"Source file: **{filename}**")
    if SHEET_COL in work.columns:
        st.write("Loaded from multiple sheets.")
    st.markdown("### Column summary")
    summary = pd.DataFrame({
        "Column": work.columns,
        "Dtype": [str(work[c].dtype) for c in work.columns],
        "Non-null": [int(work[c].notna().sum()) for c in work.columns],
        "Example": [str(work[c

