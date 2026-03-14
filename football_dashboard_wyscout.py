import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import PyPizza
import numpy as np
from matplotlib.patches import Patch

# -------------------------------
# POSITION METRICS
# -------------------------------

position_metrics_map_wyscout = {
    "CB": [
        "xG","Successful defensive actions per 90","Defensive duels per 90",
        "Defensive duels won, %","Aerial duels per 90","Aerial duels won, %",
        "Shots blocked per 90","PAdj Interceptions","Accurate passes, %",
        "Accurate forward passes, %","Accurate long passes, %"
    ],
    "6": [
        "Duels won, %","Successful defensive actions per 90","Defensive duels per 90",
        "Defensive duels won, %","Aerial duels per 90","Aerial duels won, %",
        "PAdj Interceptions","xG","Shots per 90","Progressive runs per 90",
        "Accurate passes, %","Successful dribbles, %","Accurate forward passes, %",
        "Accurate long passes, %","Offensive duels won, %","Key passes per 90",
        "Deep completions per 90","Progressive passes per 90","xA","Accelerations per 90"
    ],
    "WB": [
        "xG","xA","Successful defensive actions per 90","Defensive duels per 90",
        "Defensive duels won, %","PAdj Interceptions","Accurate crosses, %",
        "Successful dribbles, %","Progressive runs per 90","Accelerations per 90",
        "Accurate passes, %","Key passes per 90","Deep completions per 90"
    ],
    "CF": [
        "xG","xA","Successful defensive actions per 90","Aerial duels won, %",
        "Non-penalty goals per 90","Goal conversion, %","Offensive duels won, %",
        "Touches in box per 90","Accurate passes, %","Key passes per 90",
        "Deep completions per 90"
    ]
}

# -------------------------------
# LOAD DATA
# -------------------------------

@st.cache_data
def load_data(file):
    df = pd.read_excel(file) if file.name.endswith(("xlsx","xls")) else pd.read_csv(file)
    df = df.dropna(subset=["Player","Minutes played"])
    df["Minutes played"] = pd.to_numeric(df["Minutes played"], errors="coerce")
    return df


# -------------------------------
# PIZZA CHART
# -------------------------------

def plot_pizza(player, df, metrics):

    cols = [m + " Percentile" for m in metrics]

    player_values = (
        df.loc[df["Player"] == player, cols]
        .values.flatten()
        .tolist()
    )

    league_avg = [50]*len(metrics)

    pizza = PyPizza(
        params=metrics,
        background_color="#dce6e6",
        straight_line_color="#4a6f66",
        straight_line_lw=1.5,
        last_circle_lw=1,
        other_circle_lw=0
    )

    fig, ax = pizza.make_pizza(
        league_avg,
        figsize=(8,8),
        kwargs_slices=dict(
            facecolor="yellow",
            edgecolor="black",
            linewidth=2
        ),
        kwargs_params=dict(
            fontsize=10,
            fontweight="bold"
        ),
        kwargs_values=dict(
            color="black",
            fontsize=10,
            bbox=dict(
                edgecolor="black",
                facecolor="yellow",
                boxstyle="round,pad=0.2"
            )
        )
    )

    pizza.make_pizza(
        player_values,
        ax=ax,
        kwargs_slices=dict(
            facecolor="#1a78cf",
            edgecolor="black",
            linewidth=2,
            alpha=0.85
        ),
        kwargs_values=dict(
            color="white",
            fontsize=10,
            bbox=dict(
                edgecolor="#1a78cf",
                facecolor="#1a78cf",
                boxstyle="round,pad=0.2"
            )
        )
    )

    legend = [
        Patch(facecolor="#1a78cf", label=player),
        Patch(facecolor="yellow", label="League Average")
    ]

    ax.legend(handles=legend, loc="upper right")

    st.pyplot(fig)
    plt.close(fig)


# -------------------------------
# RADAR CHART
# -------------------------------

def plot_radar(labels, values_list, labels_list):

    num_vars=len(labels)

    angles=np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist()
    angles+=angles[:1]

    fig, ax=plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels,fontsize=10)

    ax.set_ylim(0,100)
    ax.set_yticklabels([])

    for values,label in zip(values_list,labels_list):

        vals=values+values[:1]

        ax.plot(angles,vals,linewidth=2,label=label)
        ax.fill(angles,vals,alpha=0.25)

    ax.legend(loc="upper right")

    st.pyplot(fig)
    plt.close(fig)


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.sidebar.header("Upload Wyscout File")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv","xlsx","xls"]
)

if uploaded_file:

    df = load_data(uploaded_file)

    st.success(f"{len(df)} rows loaded")

    # -------------------------------
    # POSITION
    # -------------------------------

    position = st.sidebar.selectbox(
        "Position",
        list(position_metrics_map_wyscout.keys())
    )

    metrics = [
        m for m in position_metrics_map_wyscout[position]
        if m in df.columns
    ]

    # -------------------------------
    # FILTERS
    # -------------------------------

    st.sidebar.subheader("Filters")

    min_mins=int(df["Minutes played"].min())
    max_mins=int(df["Minutes played"].max())

    mins_range=st.sidebar.slider(
        "Minutes Played",
        min_mins,
        max_mins,
        (min_mins,max_mins)
    )

    df=df[df["Minutes played"].between(mins_range[0],mins_range[1])]

    # -------------------------------
    # PERCENTILES
    # -------------------------------

    for m in metrics:

        df[m+" Percentile"] = (
            df[m]
            .rank(pct=True)
        )

    percentile_cols=[m+" Percentile" for m in metrics]

    # -------------------------------
    # DASHBOARD
    # -------------------------------

    st.title(f"⚽ Recruitment Dashboard - {position}")

    st.subheader("Player Table")

    st.dataframe(df[["Player","Team","Minutes played"]+metrics])

    # -------------------------------
    # PIZZA
    # -------------------------------

    st.subheader("Pizza Chart")

    player_list=df["Player"].tolist()

    selected_player=st.selectbox("Select Player",player_list)

    plot_pizza(selected_player,df,metrics)

    # -------------------------------
    # RADAR
    # -------------------------------

    st.subheader("Player Comparison")

    p1=st.selectbox("Player 1",player_list)
    p2=st.selectbox("Player 2",player_list,index=1)

    if p1!=p2:

        vals1=df.loc[df["Player"]==p1,percentile_cols].values.flatten().tolist()
        vals2=df.loc[df["Player"]==p2,percentile_cols].values.flatten().tolist()

        plot_radar(metrics,[vals1,vals2],[p1,p2])

    # -------------------------------
    # SCATTER
    # -------------------------------

    st.subheader("📊 Scatter Graph")

    two_metrics = st.multiselect(
        "Select 2 metrics",
        metrics,
        default=metrics[:2]
    )

    if len(two_metrics)==2:

        mX,mY=two_metrics

        df_plot=df.copy()

        df_plot["X"]=df_plot[mX+" Percentile"].clip(0,1)
        df_plot["Y"]=df_plot[mY+" Percentile"].clip(0,1)

        fig, ax = plt.subplots(figsize=(14,12))

        xv,yv=np.meshgrid(
            np.linspace(-0.05,1.05,600),
            np.linspace(-0.05,1.05,600)
        )

        Z=((xv.clip(0,1))+(yv.clip(0,1)))/2

        ax.imshow(
            Z,
            extent=(-0.05,1.05,-0.05,1.05),
            origin="lower",
            cmap="RdYlGn",
            alpha=0.6,
            aspect="auto"
        )

        ax.scatter(
            df_plot["X"],
            df_plot["Y"],
            s=140,
            facecolors="white",
            edgecolors="black"
        )

        highlight = st.multiselect(
            "Highlight player(s)",
            df_plot["Player"].unique()
        )

        for _,r in df_plot.iterrows():

            if r["Player"] not in highlight:

                ax.text(
                    r["X"]+0.012,
                    r["Y"]+0.012,
                    r["Player"],
                    fontsize=9
                )

        for hp in highlight:

            row=df_plot[df_plot["Player"]==hp]

            ax.scatter(
                row["X"],
                row["Y"],
                s=480,
                facecolors="none",
                edgecolors="black",
                linewidths=2.4
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
