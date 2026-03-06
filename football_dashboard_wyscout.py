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
    "CB": ["xG", "Successful defensive actions per 90", "Defensive duels per 90",
           "Defensive duels won, %", "Aerial duels per 90", "Aerial duels won, %",
           "Shots blocked per 90", "PAdj Interceptions", "Accurate passes, %",
           "Accurate forward passes, %", "Accurate long passes, %"],
    "6s": ["Duels won, %", "Defensive duels won, %", "Aerial duels won, %", "PAdj Interceptions", "xG",
           "Shots per 90", "Progressive runs per 90", "Accurate passes, %",
           "Accurate forward passes, %", "Accurate long passes, %",
           "Key passes per 90", "Deep completions per 90", "Progressive passes per 90"],
    "WB": ["xG", "xA", "Successful defensive actions per 90", "Defensive duels per 90",
           "Defensive duels won, %", "PAdj Interceptions", "Accurate crosses, %",
           "Successful dribbles, %", "Progressive runs per 90", "Accelerations per 90",
           "Accurate passes, %", "Key passes per 90", "Deep completions per 90"],
    "CF": ["xG", "xA", "Successful defensive actions per 90", "Aerial duels won, %",
           "Non-penalty goals per 90", "Goal conversion, %", "Offensive duels won, %",
           "Touches in box per 90", "Accurate passes, %", "Key passes per 90",
           "Deep completions per 90"],
    "10s": ["xG", "Goals per 90", "Non-penalty goals per 90", "Shots per 90",
            "Accurate crosses, %", "Dribbles per 90", "Successful dribbles, %",
            "Accurate passes, %", "Key passes per 90", "Deep completions per 90"],
    "GK": ["Average long pass length, m", "Save rate, %", "Prevented goals",
           "Prevented goals per 90", "Exits per 90", "Aerial duels per 90"]
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
    cols = [m + " Percentile" for m in metrics_list if m + " Percentile" in data.columns]
    player_percentiles = data.loc[data["Player"] == player, cols].values.flatten().tolist()
    league_percentiles = league_avg

    pizza = PyPizza(
        params=metrics_list,
        min_range=[0]*len(metrics_list),
        max_range=[100]*len(metrics_list),
        background_color="#f0f8ff",
        straight_line_color="black",
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=0
    )

    # Draw league average (yellow) with slice-end labels (metric names only)
    fig, ax = pizza.make_pizza(
        league_percentiles,
        figsize=(7,7),
        color_blank_space="same",
        kwargs_slices=dict(facecolor="#FFFF00", edgecolor="black", linewidth=1.5, alpha=0.8),
        kwargs_params=dict(color="black", fontsize=8, fontweight="bold"),  # metric names
        kwargs_values=dict(color="none")  # hide default values
    )

    # Draw player (blue) on top
    pizza.make_pizza(
        player_percentiles,
        ax=ax,
        color_blank_space="same",
        kwargs_slices=dict(facecolor="#1a78cf", edgecolor="black", linewidth=2, alpha=0.8),
        kwargs_params=dict(color="none"),  # hide metric names on top layer
        kwargs_values=dict(color="none")  # hide default values
    )

    # Boxed labels for percentages at slice ends
    player_percentiles_int = [int(round(p)) for p in player_percentiles]
    league_percentiles_int = [int(round(p)) for p in league_percentiles]

    # Player percentages (blue)
    for text_obj, pct in zip(ax.texts[-len(metrics_list):], player_percentiles_int):
        text_obj.set_text(f"{pct}%")
        text_obj.set_bbox(dict(facecolor="#1a78cf", edgecolor="black", boxstyle="round,pad=0.25", alpha=0.9))

    # League percentages (yellow)
    for text_obj, pct in zip(ax.texts[-2*len(metrics_list):-len(metrics_list)], league_percentiles_int):
        text_obj.set_text(f"{pct}%")
        text_obj.set_bbox(dict(facecolor="#FFFF00", edgecolor="black", boxstyle="round,pad=0.25", alpha=0.9))

    # Legend
    legend_patches = [
        Patch(facecolor="#1a78cf", edgecolor="black", label=player),
        Patch(facecolor="#FFFF00", edgecolor="black", label="League Average")
    ]
    ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.1,1.05))

    st.pyplot(fig)
    plt.close(fig)
