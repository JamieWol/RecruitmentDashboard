import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import PyPizza
from matplotlib.patches import Patch

# --------------------------------
# PAGE SETTINGS
# --------------------------------

st.set_page_config(
    page_title="Football Recruitment Dashboard",
    layout="wide"
)

# --------------------------------
# LOAD DATA
# --------------------------------

@st.cache_data
def load_data(file):

    df = pd.read_excel(file) if file.name.endswith(("xlsx","xls")) else pd.read_csv(file)

    df = df.dropna(subset=["Player","Minutes played"])

    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")

    return df


# --------------------------------
# PIZZA CHART
# --------------------------------

def plot_pizza(player, df, metrics, league_avg):

    percentile_cols = [m + " Percentile" for m in metrics]

    player_values = (
        df.loc[df["Player"] == player, percentile_cols]
        .values.flatten()
        .tolist()
    )

    pizza = PyPizza(
        params=metrics,
        min_range=[0]*len(metrics),
        max_range=[100]*len(metrics),

        background_color="white",

        straight_line_color="black",
        straight_line_lw=1.5,

        last_circle_color="black",
        last_circle_lw=2,

        other_circle_color="black",
        other_circle_lw=1
    )

    fig, ax = pizza.make_pizza(
        league_avg,
        figsize=(10,10),

        kwargs_slices=dict(
            facecolor="#facc15",
            edgecolor="black",
            linewidth=2
        ),

        kwargs_params=dict(
            fontsize=8,
            color="white",
            fontweight="bold"
        ),

        kwargs_values=dict(
            fontsize=8,
            color="black",
            bbox=dict(
                edgecolor="white",
                facecolor="#facc15",
                boxstyle="round,pad=0.25"
            )
        )
    )

    pizza.make_pizza(
    player_values,
    ax=ax,

    kwargs_slices=dict(
        facecolor="#3b82f6",
        edgecolor="black",
        linewidth=2,
        alpha=0.9
    ),

    kwargs_values=dict(
        fontsize=8,
        color="black",
        bbox=dict(
            edgecolor="#3b82f6",
            facecolor="#3b82f6",
            boxstyle="round,pad=0.25"
        )
    )
)

    legend = [
        Patch(facecolor="#3b82f6", label=player),
        Patch(facecolor="#facc15", label="League Average")
    ]

    ax.legend(handles=legend, loc="upper right", bbox_to_anchor=(1.15,1.1))

    st.pyplot(fig)
    plt.close(fig)
    
# --------------------------------
# RADAR CHART
# --------------------------------

def plot_radar(labels, values_list, labels_list):

    num_vars = len(labels)

    angles = np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels,fontsize=10)

    ax.set_ylim(0,100)
    ax.set_yticklabels([])

    for values,label in zip(values_list,labels_list):

        vals = values + values[:1]

        ax.plot(angles,vals,linewidth=2,label=label)
        ax.fill(angles,vals,alpha=0.25)

    ax.legend(loc="upper right")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.header("Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv","xlsx","xls"]
)

# --------------------------------
# MAIN APP
# --------------------------------

if uploaded_file:

    df = load_data(uploaded_file)

    st.success(f"{len(df)} rows loaded")

    ignore_cols = [
        "Player","Team","Position","Age","Minutes played",
        "Contract expires","Passport country","Foot",
        "Height","Weight"
    ]

    all_metrics = [
        col for col in df.columns
        if col not in ignore_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # --------------------------------
    # FILTERS
    # --------------------------------

    st.sidebar.subheader("Filters")

    min_mins = int(df["Minutes played"].min())
    max_mins = int(df["Minutes played"].max())

    mins_range = st.sidebar.slider(
        "Minutes Played",
        min_mins,
        max_mins,
        (min_mins,max_mins)
    )

    df = df[df["Minutes played"].between(mins_range[0],mins_range[1])]

    if "Age" in df.columns:

        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())

        age_range = st.sidebar.slider(
            "Age",
            min_age,
            max_age,
            (min_age,max_age)
        )

        df = df[df["Age"].between(age_range[0],age_range[1])]

    # --------------------------------
    # METRIC SELECTOR
    # --------------------------------

    st.sidebar.subheader("Metrics")

    metrics = st.sidebar.multiselect(
        "Select Metrics",
        all_metrics,
        default=all_metrics[:10]
    )

    if len(metrics) == 0:
        st.warning("Select at least one metric")
        st.stop()

    # --------------------------------
    # PERCENTILES
    # --------------------------------

    for m in metrics:

        df[m+" Percentile"] = (
            df[m]
            .rank(pct=True, method="max")
            .mul(100)
            .round(0)
            .astype(int)
        )

    percentile_cols = [m+" Percentile" for m in metrics]

    # --------------------------------
    # LEAGUE AVERAGE (FIXED)
    # --------------------------------

    league_avg = []

    for m in metrics:

        avg_value = df[m].mean()

        percentile = ((df[m] < avg_value).sum() / len(df)) * 100

        league_avg.append(round(percentile))

    # --------------------------------
    # PLAYER RANKING
    # --------------------------------

    df["Overall Score"] = (
        df[percentile_cols]
        .mean(axis=1)
        .round(0)
    )

    df = df.sort_values("Overall Score", ascending=False).reset_index(drop=True)

    df.index += 1
    df.insert(0, "Rank", df.index)

    # --------------------------------
    # DASHBOARD
    # --------------------------------

    st.title("⚽ Football Recruitment Dashboard")

    st.subheader("🏅 Player Ranking")

    st.dataframe(
        df[["Rank","Player","Team","Minutes played","Overall Score"] + metrics]
    )

    # --------------------------------
    # TOP PLAYER FINDER
    # --------------------------------

    st.subheader("⭐ Top Performers")

    top_players = []

    for m in metrics:

        top_row = df.loc[df[m].idxmax()]

        top_players.append({
            "Metric": m,
            "Player": top_row["Player"],
            "Value": top_row[m]
        })

    st.dataframe(pd.DataFrame(top_players))

    # --------------------------------
    # PIZZA CHART
    # --------------------------------

    st.subheader("📊 Pizza Chart")

    player_list = df["Player"].tolist()

    selected_player = st.selectbox("Select Player", player_list)

    plot_pizza(selected_player, df, metrics, league_avg)

    # --------------------------------
    # RADAR COMPARISON
    # --------------------------------

    st.subheader("📈 Player Comparison")

    p1 = st.selectbox("Player 1", player_list)
    p2 = st.selectbox("Player 2", player_list, index=1)

    if p1 != p2:

        vals1 = df.loc[df["Player"]==p1,percentile_cols].values.flatten().tolist()
        vals2 = df.loc[df["Player"]==p2,percentile_cols].values.flatten().tolist()

        plot_radar(metrics,[vals1,vals2],[p1,p2])

    # --------------------------------
    # SCATTER GRAPH
    # --------------------------------

    st.subheader("📊 Scatter Graph")

    two_metrics = st.multiselect(
        "Select 2 metrics",
        metrics,
        default=metrics[:2]
    )

    if len(two_metrics) == 2:

        mX, mY = two_metrics

        df_plot = df.copy()

        df_plot["X"] = df_plot[mX+" Percentile"]/100
        df_plot["Y"] = df_plot[mY+" Percentile"]/100

        fig, ax = plt.subplots(figsize=(14,12))

        xv, yv = np.meshgrid(
            np.linspace(-0.05,1.05,600),
            np.linspace(-0.05,1.05,600)
        )

        Z = ((xv.clip(0,1)) + (yv.clip(0,1))) / 2

        ax.imshow(
            Z,
            extent=(-0.05,1.05,-0.05,1.05),
            origin="lower",
            cmap="RdYlGn",
            alpha=0.6
        )

        ax.scatter(
            df_plot["X"],
            df_plot["Y"],
            s=140,
            facecolors="white",
            edgecolors="black"
        )

        highlight = st.multiselect(
            "Highlight players",
            df_plot["Player"]
        )

        for _, r in df_plot.iterrows():

            if r["Player"] not in highlight:

                ax.text(
                    r["X"]+0.01,
                    r["Y"]+0.01,
                    r["Player"],
                    fontsize=9
                )

        for hp in highlight:

            row = df_plot[df_plot["Player"]==hp]

            ax.scatter(
                row["X"],
                row["Y"],
                s=450,
                facecolors="none",
                edgecolors="black",
                linewidths=2
            )

            ax.text(
                row["X"].values[0]+0.015,
                row["Y"].values[0]+0.015,
                hp,
                fontsize=12,
                fontweight="bold"
            )

        ax.axhline(0.5,color="black",linestyle="--")
        ax.axvline(0.5,color="black",linestyle="--")

        ax.set_xlim(-0.05,1.05)
        ax.set_ylim(-0.05,1.05)

        ax.set_xlabel(mX+" Percentile")
        ax.set_ylabel(mY+" Percentile")

        ax.set_title("Player Scatter Graph")

        ax.grid(False)

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info("Select exactly 2 metrics.")

    # --------------------------------
    # EXPORT
    # --------------------------------

    st.subheader("Download Data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Data",
        csv,
        "recruitment_data.csv",
        "text/csv"
    )
