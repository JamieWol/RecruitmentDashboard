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
        "xG","Successful defensive actions per 90","Defensive duels per 90",
        "Defensive duels won, %","Aerial duels per 90","Aerial duels won, %",
        "Shots blocked per 90","PAdj Interceptions","Accurate passes, %",
        "Accurate forward passes, %","Accurate long passes, %"
    ],
    "6s": [
        "Duels won, %","Defensive duels won, %","Aerial duels won, %",
        "PAdj Interceptions","xG","Shots per 90","Progressive runs per 90",
        "Accurate passes, %","Accurate forward passes, %",
        "Accurate long passes, %","Key passes per 90",
        "Deep completions per 90","Progressive passes per 90"
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
    ],
    "10s": [
        "xG","Goals per 90","Non-penalty goals per 90","Shots per 90",
        "Accurate crosses, %","Dribbles per 90","Successful dribbles, %",
        "Accurate passes, %","Key passes per 90","Deep completions per 90"
    ],
    "GK": [
        "Average long pass length, m","Save rate, %",
        "Prevented goals","Prevented goals per 90",
        "Exits per 90","Aerial duels per 90"
    ]
}

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data(file):

    df = pd.read_excel(file) if file.name.endswith(("xlsx","xls")) else pd.read_csv(file)

    df = df.dropna(subset=["Player","Minutes played"])

    df["Player"] = df["Player"].astype(str)
    df["Minutes played"] = pd.to_numeric(df["Minutes played"],errors="coerce")

    return df


# -------------------------------
# Pizza Chart
# -------------------------------
def plot_pizza(player,data,league_avg,metrics):

    cols = [m+" Percentile" for m in metrics if m+" Percentile" in data.columns]

    player_vals = data.loc[data["Player"]==player,cols].values.flatten().tolist()
    league_vals = league_avg

    player_labels = [f"{int(v)}%" for v in player_vals]
    league_labels = [f"{int(v)}%" for v in league_vals]

    pizza = PyPizza(
        params=metrics,
        min_range=[0]*len(metrics),
        max_range=[100]*len(metrics),
        background_color="#f0f8ff",
        straight_line_color="black"
    )

    fig,ax = pizza.make_pizza(
        league_vals,
        figsize=(7,7),
        kwargs_slices=dict(facecolor="#FFFF00",edgecolor="black"),
        kwargs_params=dict(color="black",fontsize=7,fontweight="bold"),
        kwargs_values=dict(color="black",fontsize=9,fontweight="bold")
    )

    for txt,label in zip(ax.texts[-len(metrics):],league_labels):
        txt.set_text(label)

    pizza.make_pizza(
        player_vals,
        ax=ax,
        kwargs_slices=dict(facecolor="#1a78cf",edgecolor="black",alpha=0.8),
        kwargs_params=dict(color="black",fontsize=7,fontweight="bold"),
        kwargs_values=dict(color="white",fontsize=9,fontweight="bold")
    )

    for txt,label in zip(ax.texts[-len(metrics):],player_labels):
        txt.set_text(label)

    legend = [
        Patch(facecolor="#1a78cf",label=player),
        Patch(facecolor="#FFFF00",label="League Average")
    ]

    ax.legend(handles=legend,loc="upper right",bbox_to_anchor=(1.1,1.05))

    st.pyplot(fig)
    plt.close()


# -------------------------------
# Radar Chart
# -------------------------------
def plot_radar(labels,values_list,names,colors):

    angles = np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist()
    angles += angles[:1]

    fig,ax = plt.subplots(figsize=(7,7),subplot_kw=dict(polar=True))

    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels,fontsize=10)

    ax.set_ylim(0,100)

    for vals,name,color in zip(values_list,names,colors):

        vals = vals + vals[:1]

        ax.plot(angles,vals,color=color,linewidth=2,label=name)
        ax.fill(angles,vals,color=color,alpha=0.25)

    ax.legend(loc="upper right",bbox_to_anchor=(1.2,1.1))

    st.pyplot(fig)
    plt.close()


# -------------------------------
# Streamlit UI
# -------------------------------
st.sidebar.header("Upload Wyscout File")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv","xlsx","xls"]
)

if uploaded_file:

    df = load_data(uploaded_file)

    st.success(f"{len(df)} rows loaded")

    position = st.sidebar.selectbox(
        "Select Position",
        list(position_metrics_map_wyscout.keys())
    )

    metrics = position_metrics_map_wyscout[position]

    # ---------------- Filters ----------------

    st.sidebar.subheader("Filters")

    min_min = int(df["Minutes played"].min())
    max_min = int(df["Minutes played"].max())

    minutes = st.sidebar.slider(
        "Minutes Played",
        min_min,
        max_min,
        (min_min,max_min),
        step=50
    )

    df = df[df["Minutes played"].between(minutes[0],minutes[1])]

    if "Age" in df.columns:

        min_age = int(df["Age"].min())
        max_age = int(df["Age"].max())

        ages = st.sidebar.slider(
            "Age",
            min_age,
            max_age,
            (min_age,max_age)
        )

        df = df[df["Age"].between(ages[0],ages[1])]

    st.sidebar.success(f"{len(df)} players after filters")

    # ---------------- Percentiles ----------------

    for m in metrics:

        if m in df.columns:

            df[m+" Percentile"] = (
                df[m]
                .rank(pct=True)
                .mul(100)
                .round(0)
                .fillna(0)
                .astype(int)
            )

    percentile_cols = [m+" Percentile" for m in metrics if m+" Percentile" in df.columns]

    league_avg = (
        df[percentile_cols]
        .mean()
        .fillna(0)
        .round(0)
        .astype(int)
        .values
    )

    # ---------------- Ranking ----------------

    df["Overall Score"] = (
        df[percentile_cols]
        .mean(axis=1)
        .fillna(0)
        .round(0)
        .astype(int)
    )

    df = df.sort_values("Overall Score",ascending=False).reset_index(drop=True)
    df.index += 1

    # ---------------- Dashboard ----------------

    st.title(f"⚽ Recruitment Dashboard - {position}")

    st.subheader("🏅 Player Ranking")

    st.dataframe(
        df[["Player","Minutes played","Overall Score"]+metrics]
        .highlight_max(subset=["Overall Score"],color="lightgreen")
    )

    # ---------------- Pizza Chart ----------------

    st.subheader("📊 Pizza Chart")

    players = df["Player"].tolist()

    player = st.selectbox("Select Player",players)

    plot_pizza(player,df,league_avg,metrics)

    # ---------------- Player vs Player ----------------

    st.subheader("📈 Player Comparison Radar")

    if len(players) >= 2:

        p1 = st.selectbox("Player 1",players)
        p2 = st.selectbox("Player 2",players,index=1)

        if p1 != p2:

            vals1 = df.loc[df["Player"]==p1,percentile_cols].values.flatten().tolist()
            vals2 = df.loc[df["Player"]==p2,percentile_cols].values.flatten().tolist()

            plot_radar(metrics,[vals1,vals2],[p1,p2],["red","blue"])

    # ---------------- Player vs League ----------------

    st.subheader("📊 Player vs League Radar")

    p3 = st.selectbox("Player vs League Average",players,key="league")

    vals = df.loc[df["Player"]==p3,percentile_cols].values.flatten().tolist()

    plot_radar(metrics,[vals,league_avg.tolist()],[p3,"League Average"],["green","black"])

    # ---------------- Export ----------------

    st.subheader("⬇️ Export")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Filtered Data",
        csv,
        "filtered_wyscout.csv",
        "text/csv"
    )
