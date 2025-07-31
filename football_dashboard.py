import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import PyPizza
import numpy as np

import streamlit as st
import pandas as pd
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="Football Recruitment Dashboard", page_icon="⚽", layout="wide")

# --- Sidebar Upload ---
st.sidebar.title("Upload Your Player Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- Welcome Page ---
if uploaded_file is None:
    st.title("📊 Football Recruitment Dashboard")
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6e/Football_Icon.svg", width=100)

    st.markdown("""
    ## ⚽️ Welcome!
    
    This dashboard helps you scout and assess players based on position-specific metrics using radar charts and percentiles.
    
    To get started:
    - Upload your CSV file using the **sidebar**
    - Explore player insights and rankings
    - Use filters for position, league, minutes played, and more

    ---

    👈 **Use the sidebar to upload your data file now.**
    """)
    st.stop()

# --- Position to Metrics Mapping ---
position_metrics_map = {
    # Centre Backs
    "Centre Back": [
        "Scoring Contribution", "PAdj Interceptions", "Aerial Win%", "Aerial Wins",
        "Dribbles Successfully Defended%", "PAdj Tackles", "Tack/Dribbled Past%",
        "xGBuildup", "Pass Forward%", "Long Ball%", "Passing%", "Pressured Pass%", "OBV"
    ],
    "Left Centre Back": [
        "Scoring Contribution", "PAdj Interceptions", "Aerial Win%", "Aerial Wins",
        "Dribbles Successfully Defended%", "PAdj Tackles", "Tack/Dribbled Past%",
        "xGBuildup", "Pass Forward%", "Long Ball%", "Passing%", "Pressured Pass%", "OBV"
    ],
    "Right Centre Back": [
        "Scoring Contribution", "PAdj Interceptions", "Aerial Win%", "Aerial Wins",
        "Dribbles Successfully Defended%", "PAdj Tackles", "Tack/Dribbled Past%",
        "xGBuildup", "Pass Forward%", "Long Ball%", "Passing%", "Pressured Pass%", "OBV"
    ],

    # Full Backs & Wing Backs
    "Full Back": [
        "Key Passes", "xG Assisted", "Successful Dribbles", "Scoring Contribution",
        "PAdj Interceptions", "Dribbles Successfully Defended%", "PAdj Tackles",
        "Tack/Dribbled Past%", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV"
    ],
    "Left Back": [
        "Key Passes", "xG Assisted", "Successful Dribbles", "Scoring Contribution",
        "PAdj Interceptions", "Dribbles Successfully Defended%", "PAdj Tackles",
        "Tack/Dribbled Past%", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV"
    ],
    "Right Back": [
        "Key Passes", "xG Assisted", "Successful Dribbles", "Scoring Contribution",
        "PAdj Interceptions", "Dribbles Successfully Defended%", "PAdj Tackles",
        "Tack/Dribbled Past%", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV"
    ],
    "Left Wing Back": [
        "Key Passes", "xG Assisted", "Successful Dribbles", "Scoring Contribution",
        "PAdj Interceptions", "Dribbles Successfully Defended%", "PAdj Tackles",
        "Tack/Dribbled Past%", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV"
    ],
    "Right Wing Back": [
        "Key Passes", "xG Assisted", "Successful Dribbles", "Scoring Contribution",
        "PAdj Interceptions", "Dribbles Successfully Defended%", "PAdj Tackles",
        "Tack/Dribbled Past%", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV"
    ],

    # Wide Midfielders
    "Left Midfielder": [
        "Key Passes", "Dribble%", "Scoring Contribution", "Non-Penalty Goals", "xG",
        "xG/Shot", "PAdj Pressures", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV", "OBV"
    ],
    "Right Midfielder": [
        "Key Passes", "Dribble%", "Scoring Contribution", "Non-Penalty Goals", "xG",
        "xG/Shot", "PAdj Pressures", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV", "OBV"
    ],

    # Defensive Midfielders
    "Centre Defensive Midfielder": [
        "Key Passes", "Scoring Contribution", "Shots", "xG", "PAdj Interceptions",
        "PAdj Tackles", "Tack/Dribbled Past%", "PAdj Pressures", "Deep Progressions",
        "Pass Forward%", "Long Ball%", "Passing%", "OBV"
    ],
    "Left Defensive Midfielder": [
        "Key Passes", "Scoring Contribution", "Shots", "xG", "PAdj Interceptions",
        "PAdj Tackles", "Tack/Dribbled Past%", "PAdj Pressures", "Deep Progressions",
        "Pass Forward%", "Long Ball%", "Passing%", "OBV"
    ],
    "Right Defensive Midfielder": [
        "Key Passes", "Scoring Contribution", "Shots", "xG", "PAdj Interceptions",
        "PAdj Tackles", "Tack/Dribbled Past%", "PAdj Pressures", "Deep Progressions",
        "Pass Forward%", "Long Ball%", "Passing%", "OBV"
    ],
    "Left Centre Midfielder": [
        "Key Passes", "Scoring Contribution", "Shots", "xG", "PAdj Interceptions",
        "PAdj Tackles", "Tack/Dribbled Past%", "PAdj Pressures", "Deep Progressions",
        "Pass Forward%", "Long Ball%", "Passing%", "OBV"
    ],
    "Right Centre Midfielder": [
        "Key Passes", "Scoring Contribution", "Shots", "xG", "PAdj Interceptions",
        "PAdj Tackles", "Tack/Dribbled Past%", "PAdj Pressures", "Deep Progressions",
        "Pass Forward%", "Long Ball%", "Passing%", "OBV"
    ],

    # Attacking Midfielders / Wingers
    "Centre Attacking Midfielder": [
        "Key Passes", "Dribble%", "Scoring Contribution", "Non-Penalty Goals", "xG",
        "xG/Shot", "PAdj Pressures", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV", "OBV"
    ],
    "Left Attacking Midfielder": [
        "Key Passes", "Dribble%", "Scoring Contribution", "Non-Penalty Goals", "xG",
        "xG/Shot", "PAdj Pressures", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV", "OBV"
    ],
    "Right Attacking Midfielder": [
        "Key Passes", "Dribble%", "Scoring Contribution", "Non-Penalty Goals", "xG",
        "xG/Shot", "PAdj Pressures", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV", "OBV"
    ],
    "Left Wing": [
        "Key Passes", "Dribble%", "Scoring Contribution", "Non-Penalty Goals", "xG",
        "xG/Shot", "PAdj Pressures", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV", "OBV"
    ],
    "Right Wing": [
        "Key Passes", "Dribble%", "Scoring Contribution", "Non-Penalty Goals", "xG",
        "xG/Shot", "PAdj Pressures", "Crossing%", "Deep Completions", "Deep Progressions",
        "Passing%", "Dribble & Carry OBV", "OBV"
    ],

    # Forwards
    "Centre Forward": [
        "Key Passes", "Turnovers", "Scoring Contribution", "Goal Conversion%",
        "Non-Penalty Goals", "xG", "xG/Shot", "Aerial Win%", "PAdj Pressures",
        "Deep Completions", "Touches In Box", "Passing%", "OBV"
    ],
    "Right Centre Forward": [
        "Key Passes", "Turnovers", "Scoring Contribution", "Goal Conversion%",
        "Non-Penalty Goals", "xG", "xG/Shot", "Aerial Win%", "PAdj Pressures",
        "Deep Completions", "Touches In Box", "Passing%", "OBV"
    ],
    "Left Centre Forward": [
        "Key Passes", "Turnovers", "Scoring Contribution", "Goal Conversion%",
        "Non-Penalty Goals", "xG", "xG/Shot", "Aerial Win%", "PAdj Pressures",
        "Deep Completions", "Touches In Box", "Passing%", "OBV"
    ],

    # Goalkeepers
    "Goalkeeper": [
        "Claims - CCAA%", "Pass into Danger%", "Pass into Pressure%", "Positive Outcome",
        "Expected Save%", "Goals Saved Above Average", "Positioning Error",
        "PSxG Faced", "Save%", "Goalkeeper OBV"
    ]
}


metric_higher_better = {
    "Goals Conceded": False,
    "Positioning Error": False,
    # Add others where lower is better...
}

def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    else:
        return pd.DataFrame()

# Load the file
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
df = load_data(uploaded_file)

if df.empty:
    st.title("⚽ Recruitment Dashboard")
    st.write("Upload a CSV file using the sidebar to begin.")
    st.stop()

# Data cleaning
df = df.dropna(subset=["Age", "Minutes Played", "Name", "Primary Position", "Team", "Competition"])
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df["Minutes Played"] = pd.to_numeric(df["Minutes Played"], errors='coerce')
df["Name"] = df["Name"].astype(str)
df["Primary Position"] = df["Primary Position"].astype(str)
df["Team"] = df["Team"].astype(str)
df["Competition"] = df["Competition"].astype(str)

# Now your filters and rest of code...



# --- Sidebar Filters ---
st.sidebar.header("Filters")
positions = sorted(df["Primary Position"].dropna().unique())
selected_positions = st.sidebar.multiselect("Position", positions, default=positions)
selected_age = st.sidebar.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
selected_minutes = st.sidebar.slider("Minutes Played Range", int(df["Minutes Played"].min()), int(df["Minutes Played"].max()), (int(df["Minutes Played"].min()), int(df["Minutes Played"].max())))
selected_teams = st.sidebar.multiselect("Team", sorted(df["Team"].dropna().unique()), default=sorted(df["Team"].dropna().unique()))
selected_comps = st.sidebar.multiselect("Competition", sorted(df["Competition"].dropna().unique()), default=sorted(df["Competition"].dropna().unique()))
selected_names = st.sidebar.multiselect("Player Name", sorted(df["Name"].dropna().unique()), default=[])

# --- Metrics for Positions ---
pizza_metrics = []
for pos in selected_positions:
    pizza_metrics += position_metrics_map.get(pos, [])
pizza_metrics = list(set(pizza_metrics))

if not pizza_metrics:
    st.error("No metrics found for selected position(s). Please update the position_metrics_map.")
    st.stop()

# --- Apply Filters ---
filtered_df = df[
    (df["Primary Position"].isin(selected_positions)) &
    (df["Age"].between(*selected_age)) &
    (df["Minutes Played"].between(*selected_minutes)) &
    (df["Team"].isin(selected_teams)) &
    (df["Competition"].isin(selected_comps))
].copy()

if selected_names:
    filtered_df = filtered_df[filtered_df["Name"].isin(selected_names)]

if filtered_df.empty:
    st.warning("No players match the filter criteria.")
    st.stop()

# --- Percentile Rankings ---
for m in pizza_metrics:
    if m in filtered_df.columns:
        filtered_df[m + " Percentile"] = filtered_df[m].rank(pct=True)
    else:
        st.warning(f"Metric '{m}' not found in data.")

percentile_columns = [m + " Percentile" for m in pizza_metrics if m + " Percentile" in filtered_df.columns]
league_avg_percentiles = filtered_df[percentile_columns].mean().values


filtered_df["Overall Score"] = filtered_df[[m + " Percentile" for m in pizza_metrics if m + " Percentile" in filtered_df.columns]].mean(axis=1)
filtered_df = filtered_df.sort_values("Overall Score", ascending=False).reset_index(drop=True)
filtered_df.index += 1

# --- Dashboard Title ---
st.title("⚽ Recruitment Dashboard")

# --- Player Ranking Table ---
st.subheader("🏅 Player Ranking")
st.dataframe(
    filtered_df[
        ["Name", "Team", "Primary Position", "Age", "Minutes Played", "Competition", "Overall Score"] + pizza_metrics
    ]
    .style.format({"Overall Score": "{:.3f}"})
    .highlight_max(subset=["Overall Score"], color="lightgreen")
    .set_caption("Players ranked by average percentile across key metrics")
)

# --- Pizza Chart Function ---
def plot_pizza(player, data, league_avg, metrics_list):
    cols = [m + " Percentile" for m in metrics_list]
    cols = [c for c in cols if c in data.columns]
    player_percentiles_raw = data.loc[data["Name"] == player, cols].values.flatten().tolist()
    league_avg_raw = league_avg

    player_percentiles_str = [f"{int(p * 100)}%" for p in player_percentiles_raw]
    league_percentiles_str = [f"{int(p * 100)}%" for p in league_avg_raw]

    pizza = PyPizza(
        params=metrics_list,
        min_range=[0] * len(metrics_list),
        max_range=[1] * len(metrics_list),
        background_color="#f0f8ff",
        straight_line_color="black",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=0
    )

    fig, ax = pizza.make_pizza(
        league_avg_raw,
        figsize=(6, 6),
        color_blank_space="same",
        kwargs_slices=dict(
            facecolor="#FFFF00",
            edgecolor="black",
            linewidth=1.5,
            alpha=1,
            zorder=5
        ),
        kwargs_params=dict(
            color="black",
            fontsize=6,
            fontweight="bold"
        ),
        kwargs_values=dict(
            color="black",
            fontsize=8,
            fontweight="bold",
            zorder=10,
            bbox=dict(edgecolor="black", facecolor="#FFFF00", boxstyle="round,pad=0.15")
        )
    )

    for text_obj, pct_str in zip(ax.texts[-len(metrics_list):], league_percentiles_str):
        text_obj.set_text(pct_str)
        text_obj.set_zorder(15)

    pizza.make_pizza(
        player_percentiles_raw,
        ax=ax,
        color_blank_space="same",
        kwargs_slices=dict(
            facecolor="#1a78cf",
            edgecolor="black",
            linewidth=2,
            alpha=0.8,
            zorder=5
        ),
        kwargs_params=dict(
            color="black",
            fontsize=6,
            fontweight="bold"
        ),
        kwargs_values=dict(
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=20,
            bbox=dict(edgecolor="#000000", facecolor="#1a78cf", boxstyle="round,pad=0.2", alpha=0.7)
        )
    )

    for text_obj, pct_str in zip(ax.texts[-len(metrics_list):], player_percentiles_str):
        text_obj.set_text(pct_str)
        text_obj.set_zorder(25)

    from matplotlib.patches import Patch

    legend_patches = [
        Patch(facecolor="#1a78cf", edgecolor="black", label=player),
        Patch(facecolor="#FFFF00", edgecolor="black", label="League Average")
    ]
    legend = ax.legend(
        handles=legend_patches,
        loc="upper right",
        bbox_to_anchor=(1.1, 1.05),
        fontsize=8,
        facecolor="#222222",
        edgecolor="white"
    )
    for text in legend.get_texts():
        text.set_color("white")

    st.pyplot(fig)
    plt.close(fig)

# --- Radar Chart Function ---
def plot_radar(labels, values_list, labels_list, colors):
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.spines['polar'].set_visible(True)
    ax.grid(linewidth=0.8, color='black', alpha=0.6)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)

    for r in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for angle in angles[:-1]:
            ax.text(angle, r, f"{int(r * 100)}", size=9, fontweight='bold', color="black", ha='center', va='center')

    for values, label, color in zip(values_list, labels_list, colors):
        vals = values + values[:1]
        ax.plot(angles, vals, color=color, linewidth=2.5, label=label)
        ax.fill(angles, vals, color=color, alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig)
    plt.close(fig)

# --- Pizza Chart Section ---
st.subheader("📊 Pizza Chart: Player vs League Average")
players_available = filtered_df["Name"].unique().tolist()
selected_player = st.selectbox("Select Player for Pizza Chart", players_available)
if selected_player:
    plot_pizza(selected_player, filtered_df, league_avg_percentiles, pizza_metrics)

# --- Radar Chart Section ---
st.subheader("📈 Radar Chart: Player vs Player Comparison")
p1 = st.selectbox("Select Player 1", players_available)
p2 = st.selectbox("Select Player 2", players_available, index=1)
if p1 != p2:
    vals1 = filtered_df.loc[filtered_df["Name"] == p1, [m + " Percentile" for m in pizza_metrics]].values.flatten().tolist()
    vals2 = filtered_df.loc[filtered_df["Name"] == p2, [m + " Percentile" for m in pizza_metrics]].values.flatten().tolist()
    plot_radar(pizza_metrics, [vals1, vals2], [p1, p2], ["red", "blue"])
else:
    st.info("Select two different players.")

st.subheader("📊 Radar Chart: Player vs League Average")
p3 = st.selectbox("Select Player", players_available, key="league_player")
league_avg_vals = league_avg_percentiles.tolist()
p3_vals = filtered_df.loc[filtered_df["Name"] == p3, [m + " Percentile" for m in pizza_metrics]].values.flatten().tolist()
plot_radar(pizza_metrics, [p3_vals, league_avg_vals], [p3, "League Average"], ["green", "red"])

# --- CSV Export Section ---
st.subheader("⬇️ Export Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Filtered Player Data as CSV",
    data=csv,
    file_name='filtered_players.csv',
    mime='text/csv'
)



