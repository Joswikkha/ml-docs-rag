import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

st.set_page_config(page_title="Benchmark Dashboard", page_icon="📊", layout="wide")
st.title("📊 Retrieval Strategy Benchmark Results")

COLORS = ["#6ee7b7", "#818cf8", "#f472b6"]
STRATEGY_COL = "strategy"
ALL_POSSIBLE_METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

@st.cache_data
def load_results():
    path = os.path.join(ROOT, "results", "benchmark_results.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

df = load_results()

if df is None:
    st.error("No results found at results/benchmark_results.csv")
    st.stop()

# Keep only the most recent row per strategy — repeated incremental
# benchmark runs can leave duplicate/stale rows in the CSV
df = df.drop_duplicates(subset=STRATEGY_COL, keep="last").reset_index(drop=True)

# Only use metrics that actually exist in this CSV
METRICS = [m for m in ALL_POSSIBLE_METRICS if m in df.columns]
if not METRICS:
    st.error("No recognized metric columns found in benchmark_results.csv")
    st.stop()

headline = METRICS[0]
st.caption(f"Local scoring · {len(df)} strategies · metrics: {', '.join(METRICS)}")

# ── KPI cards ──────────────────────────────────────────────────────
st.subheader(f"{headline.replace('_', ' ').title()} — headline metric (higher = better)")
cols = st.columns(len(df))
baseline = df.iloc[0][headline]
for i, row in df.iterrows():
    delta = (row[headline] - baseline) / max(baseline, 0.0001) * 100
    cols[i].metric(
        label=row[STRATEGY_COL],
        value=f"{row[headline]:.4f}",
        delta=f"{delta:+.1f}% vs baseline" if i > 0 else "baseline"
    )

st.divider()

# ── Radar (only if 3+ metrics) + Bar ───────────────────────────────
if len(METRICS) >= 3:
    col1, col2 = st.columns(2)
else:
    col1 = st.container()
    col2 = st.container()

with col1:
    if len(METRICS) >= 3:
        st.subheader("Radar — all metrics")
        fig_radar = go.Figure()
        for i, row in df.iterrows():
            vals   = [row[m] for m in METRICS]
            labels = [m.replace("_", " ").title() for m in METRICS]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=labels + [labels[0]],
                name=row[STRATEGY_COL],
                fill="toself",
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                fillcolor=COLORS[i % len(COLORS)],
                opacity=0.2
            ))
        max_val = df[METRICS].max().max()
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.3])),
            height=420,
            legend=dict(orientation="h", y=-0.25)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

with col2:
    st.subheader("Bar — per metric")
    df_long = df.melt(
        id_vars=STRATEGY_COL,
        value_vars=METRICS,
        var_name="Metric",
        value_name="Score"
    )
    df_long["Metric"] = df_long["Metric"].str.replace("_", " ").str.title()
    fig_bar = px.bar(
        df_long, x="Metric", y="Score", color=STRATEGY_COL,
        barmode="group",
        color_discrete_sequence=COLORS
    )
    fig_bar.update_layout(height=420, legend_title="Strategy", legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Full summary table ─────────────────────────────────────────────
st.subheader("Full results table")
display = df.set_index(STRATEGY_COL)[METRICS]
st.dataframe(
    display.style.format("{:.4f}").highlight_max(axis=0, color="#1a3a2a").highlight_min(axis=0, color="#3a1a1a"),
    use_container_width=True
)

# ── Winner callout ─────────────────────────────────────────────────
best_idx    = df[headline].idxmax()
winner      = df.loc[best_idx, STRATEGY_COL]
best_val    = df.loc[best_idx, headline]
improvement = (best_val - baseline) / max(baseline, 0.0001) * 100
st.success(
    f"🏆 **Winner: {winner}** — "
    f"{headline.replace('_',' ').title()} {best_val:.4f} "
    f"({improvement:+.1f}% over baseline)"
)

# ── Per-row drilldown ─────────────────────────────────────────────
st.divider()
st.subheader("Strategy detail")
choice   = st.selectbox("Pick a strategy", df[STRATEGY_COL].tolist())
filtered = df[df[STRATEGY_COL] == choice][METRICS].T.reset_index()
filtered.columns = ["Metric", "Score"]
filtered["Metric"] = filtered["Metric"].str.replace("_", " ").str.title()
st.dataframe(filtered.style.format({"Score": "{:.4f}"}), use_container_width=True)