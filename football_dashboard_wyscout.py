import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import PyPizza
import numpy as np
from matplotlib.patches import Patch

# -------------------------------
# Position to Metrics Mapping
# -------------------------------
position_metrics_map_wyscout = {
    "CB": [
        "xG", "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
        "Shots blocked per 90", "PAdj Interceptions", "Accurate passes, %",
        "Accurate forward passes, %", "Accurate long passes, %"
    ],
    "6s": [
        "Duels won, %", "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
        "PAdj Interceptions", "xG", "Shots per 90", "Progressive runs per 90",
        "Accurate passes, %", "Successful dribbles, %", "Accurate forward passes, %",
        "Accurate long passes, %", "Offensive duels won, %", "Key passes per 90",
        "Deep completions per 90", "Progressive passes per 90", "xA", "Accelerations per 90"
    ],
    "WB": [
        "xG", "xA", "Successful defensive actions per 90", "Defensive duels per 90",
        "Defensive duels won, %", "PAdj Interceptions", "Accurate crosses, %",
        "Successful dribbles, %", "Progressive runs per 90", "Accelerations per 90",
        "Accurate passes, %", "Key passes per 90", "Deep completions per 90"
    ],
    "CF": [
        "xG", "xA", "Successful defensive actions per 90", "Aerial duels won, %",
        "Non-penalty goals per 90", "Goal conversion, %", "Offensive duels won, %",
        "Touches in box per 90", "Accurate passes, %", "Key passes per 90",
        "Deep completions per 90"
    ],
    "10s": [
        "xG", "Goals per 90", "Non-penalty goals per 90", "Shots per 90",
        "Accurate crosses, %", "Dribbles per 90", "Successful dribbles, %",
        "Accurate passes, %", "Key passes per 90", "Deep completions per 90"
    ],
    "GK": [
        "Average long pass length, m", "Save rate, %", "Prevented goals",
        "Prevented goals per 90", "Exits per 90", "Aerial duels per 90"
    ]
}

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_excel(file) if file.name.endswith(("xlsx", "xls")) else pd.read_csv(file)

    df = df.dropna(subset=["Player", "Minutes played"])

    df["Player"] = df["Player"].astype(str)
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors='coerce')

    return df


# -------------------------------
# Pizza Chart
# -------------------------------
def plot_pizza(player, data, league_avg, metrics_list):

    cols = [m + " Percentile" for m in metrics_list]

    player_percentiles = (
        data.loc[data["Player"] == player, cols]
        .values.flatten()
        .tolist()
    )

    player_percentiles = [int(round(p)) for p in player_percentiles]

    pizza = PyPizza(
        params=metrics_list,
        min_range=[0] * len(metrics_list),
        max_range=[100] * len(metrics_list),
        background_color="#f0f8ff",
        straight_line_color="black",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=0
    )

    fig, ax = pizza.make_pizza(
        league_avg,
        figsize=(7,7),
        color_blank_space="same",
        kwargs_slices=dict(
            facecolor="#FFFF00",
            edgecolor="black",
            linewidth=1.5,
            alpha=1
        ),
        kwargs_params=dict(color="black", fontsize=8, fontweight="bold"),
        kwargs_values=dict(color="black", fontsize=9)
    )

    pizza.make_pizza(
        player_percentiles,
        ax=ax,
        color_blank_space="same",
        kwargs_slices=dict(
            facecolor="#1a78cf",
            edgecolor="black",
            linewidth=2,
            alpha=0.8
        ),
        kwargs_params=dict(color="black", fontsize=8),
        kwargs_values=dict(color="white", fontsize=9)
    )

    legend_patches = [
        Patch(facecolor="#1a78cf", edgecolor="black", label=player),
        Patch(facecolor="#FFFF00", edgecolor="black", label="League Average")
    ]

    ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.2,1.1))

    st.pyplot(fig)
    plt.close(fig)


# -------------------------------
# Radar Chart
# -------------------------------
def plot_radar(labels, values_list, labels_list):

    num_vars = len(labels)

    angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylim(0,100)
    ax.set_yticklabels([])

    for values, label in zip(values_list, labels_list):

        vals = values + values[:1]

        ax.plot(angles, vals, linewidth=2, label=label)
        ax.fill(angles, vals, alpha=0.25)

    ax.legend(loc="upper right", bbox_to_anchor=(1.2,1.1))

    st.pyplot(fig)
    plt.close(fig)


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.sidebar.header("Upload Wyscout File")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv","xlsx","xls"]
)

if uploaded_file:

    df = load_data(uploaded_file)

    st.success(f"Loaded {len(df)} rows")

    # -------------------------------
    # POSITION
    # -------------------------------
    position_options = list(position_metrics_map_wyscout.keys())

    selected_position = st.sidebar.selectbox(
        "Select Position",
        position_options
    )

    metrics = [
        m for m in position_metrics_map_wyscout[selected_position]
        if m in df.columns
    ]

    # -------------------------------
    # FILTERS
    # -------------------------------
    st.sidebar.subheader("Filters")

    min_minutes = int(df["Minutes played"].min())
    max_minutes = int(df["Minutes played"].max())

    minutes_range = st.sidebar.slider(
        "Minutes Played",
        min_minutes,
        max_minutes,
        (min_minutes, max_minutes),
        step=50
    )

    df = df[df["Minutes played"].between(minutes_range[0], minutes_range[1])]

    if "Age" in df.columns:

        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())

        age_range = st.sidebar.slider(
            "Age",
            min_age,
            max_age,
            (min_age, max_age)
        )

        df = df[df["Age"].between(age_range[0], age_range[1])]

    if df.empty:

        st.warning("No players match the filters")

        st.stop()

    # -------------------------------
    # PERCENTILES
    # -------------------------------
    for m in metrics:

        df[m + " Percentile"] = (
            df[m]
            .rank(pct=True)
            .mul(100)
            .round(0)
            .astype(int)
        )

    percentile_columns = [m + " Percentile" for m in metrics]

    league_avg_percentiles = (
        df[percentile_columns]
        .mean()
        .round(0)
        .astype(int)
        .tolist()
    )

    # -------------------------------
    # OVERALL SCORE
    # -------------------------------
    df["Overall Score"] = df[percentile_columns].mean(axis=1).round(0)

    df = df.sort_values("Overall Score", ascending=False).reset_index(drop=True)

    df.index += 1

    # -------------------------------
    # DASHBOARD
    # -------------------------------
    st.title(f"⚽ Recruitment Dashboard - {selected_position}")

    st.subheader("🏅 Player Ranking")

    info_columns = [
        "Player","Team","Position","Age","Overall Score",
        "Contract expires","Minutes played","Passport country",
        "Foot","Height","Weight"
    ]

    existing_info_columns = [c for c in info_columns if c in df.columns]

    st.dataframe(
        df[existing_info_columns + metrics]
        .style.format({"Overall Score":"{:.0f}"})
        .highlight_max(subset=["Overall Score"], color="lightgreen")
    )

    # -------------------------------
    # PIZZA CHART
    # -------------------------------
    st.subheader("📊 Pizza Chart")

    player_list = df["Player"].tolist()

    selected_player = st.selectbox(
        "Select Player",
        player_list
    )

    plot_pizza(
        selected_player,
        df,
        league_avg_percentiles,
        metrics
    )

    # -------------------------------
    # RADAR PLAYER VS PLAYER
    # -------------------------------
    st.subheader("📈 Player Comparison")

    if len(player_list) >= 2:

        p1 = st.selectbox("Player 1", player_list)

        p2 = st.selectbox("Player 2", player_list, index=1)

        if p1 != p2:

            vals1 = df.loc[df["Player"] == p1, percentile_columns].values.flatten().tolist()

            vals2 = df.loc[df["Player"] == p2, percentile_columns].values.flatten().tolist()

            plot_radar(
                metrics,
                [vals1, vals2],
                [p1, p2]
            )

    # -------------------------------
    # PLAYER VS LEAGUE
    # -------------------------------
    st.subheader("📊 Player vs League Average")

    p3 = st.selectbox(
        "Select Player",
        player_list,
        key="league_player"
    )

    vals3 = df.loc[df["Player"] == p3, percentile_columns].values.flatten().tolist()

    plot_radar(
        metrics,
        [vals3, league_avg_percentiles],
        [p3, "League Average"]
    )

    # -------------------------------
    # EXPORT
    # -------------------------------
    st.subheader("Download Data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Data",
        csv,
        "filtered_wyscout.csv",
        "text/csv"
    )
