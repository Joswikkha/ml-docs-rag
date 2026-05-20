import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Benchmark Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retrieval Strategy Benchmark Results")
st.caption("Local scoring · 10 questions · 4 metrics · 3 strategies")

METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "context_precision"
]

COLORS = ["#6ee7b7", "#818cf8", "#f472b6"]
STRATEGY_COL = "strategy"


@st.cache_data
def load_results():
    path = ROOT / "results" / "benchmark_results.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)


df = load_results()

if df is None:
    st.error("No results found at results/benchmark_results.csv")
    st.stop()

st.subheader("Context Recall — headline metric")

cols = st.columns(len(df))
baseline = df.iloc[0]["context_recall"]

for i, row in df.iterrows():
    delta = (row["context_recall"] - baseline) / max(baseline, 0.0001) * 100

    cols[i].metric(
        label=row[STRATEGY_COL],
        value=f"{row['context_recall']:.4f}",
        delta=f"{delta:+.1f}% vs baseline" if i > 0 else "baseline"
    )

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Radar — all 4 metrics")

    fig_radar = go.Figure()

    for i, row in df.iterrows():
        vals = [row[m] for m in METRICS]
        labels = [m.replace("_", " ").title() for m in METRICS]

        fig_radar.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=labels + [labels[0]],
                name=row[STRATEGY_COL],
                fill="toself",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                fillcolor=COLORS[i % len(COLORS)],
                opacity=0.2
            )
        )

    max_val = df[METRICS].max().max()

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_val * 1.3]
            )
        ),
        height=420,
        legend=dict(orientation="h", y=-0.25)
    )

    st.plotly_chart(fig_radar, use_container_width=True)

with col2:
    st.subheader("Grouped bar — per metric")

    df_long = df.melt(
        id_vars=STRATEGY_COL,
        value_vars=METRICS,
        var_name="Metric",
        value_name="Score"
    )

    df_long["Metric"] = df_long["Metric"].str.replace("_", " ").str.title()

    fig_bar = px.bar(
        df_long,
        x="Metric",
        y="Score",
        color=STRATEGY_COL,
        barmode="group",
        color_discrete_sequence=COLORS
    )

    fig_bar.update_layout(
        height=420,
        legend_title="Strategy",
        legend=dict(orientation="h", y=-0.35)
    )

    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Full results table")

display = df.set_index(STRATEGY_COL)[METRICS]

st.dataframe(
    display.style
    .format("{:.4f}")
    .highlight_max(axis=0, color="#1a3a2a")
    .highlight_min(axis=0, color="#3a1a1a"),
    use_container_width=True
)

best_idx = df["context_recall"].idxmax()
winner = df.loc[best_idx, STRATEGY_COL]
best_val = df.loc[best_idx, "context_recall"]
improvement = (best_val - baseline) / max(baseline, 0.0001) * 100

st.success(
    f"🏆 **Winner: {winner}** — "
    f"Context Recall {best_val:.4f} "
    f"({improvement:+.1f}% over baseline)"
)

st.divider()
st.subheader("Strategy detail")

choice = st.selectbox(
    "Pick a strategy",
    df[STRATEGY_COL].tolist()
)

filtered = df[df[STRATEGY_COL] == choice][METRICS].T.reset_index()
filtered.columns = ["Metric", "Score"]
filtered["Metric"] = filtered["Metric"].str.replace("_", " ").str.title()

st.dataframe(
    filtered.style.format({"Score": "{:.4f}"}),
    use_container_width=True
)