import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

st.set_page_config(page_title="Benchmark Dashboard", page_icon="📊", layout="wide")
st.title("📊 Retrieval Strategy Benchmark Results")
st.caption("Local scoring · 10 questions · 4 metrics · 3 strategies")

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
COLORS  = ["#6ee7b7", "#818cf8", "#f472b6"]

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

# Show raw columns for debugging (remove later)
st.caption(f"CSV columns detected: {df.columns.tolist()}")

# ── Find the strategy column automatically ─────────────────────────
strategy_col = None
for col in df.columns:
    if "strategy" in col.lower() or "name" in col.lower():
        strategy_col = col
        break

if strategy_col is None:
    # If no strategy column, use index
    df["strategy"] = df.index.astype(str)
    strategy_col = "strategy"

# ── Find available metric columns ─────────────────────────────────
available_metrics = [m for m in METRICS if m in df.columns]
if not available_metrics:
    # Try partial matches
    available_metrics = [c for c in df.columns if c != strategy_col]

# ── Summary table ─────────────────────────────────────────────────
summary = df.groupby(strategy_col)[available_metrics].mean().reset_index()

# ── KPI cards ─────────────────────────────────────────────────────
main_metric = available_metrics[-1]  # use last metric as headline
st.subheader(f"Scores by strategy — {main_metric.replace('_',' ').title()}")
cols = st.columns(len(summary))
baseline = summary.iloc[0][main_metric]
for i, row in summary.iterrows():
    delta = (row[main_metric] - baseline) / max(baseline, 0.0001) * 100
    cols[i].metric(
        label=row[strategy_col],
        value=f"{row[main_metric]:.4f}",
        delta=f"{delta:+.1f}% vs baseline" if i > 0 else "baseline"
    )

st.divider()

# ── Radar + Bar ────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Radar — all metrics")
    fig_radar = go.Figure()
    for i, row in summary.iterrows():
        vals   = [row[m] for m in available_metrics]
        labels = [m.replace("_", " ").title() for m in available_metrics]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            name=str(row[strategy_col]),
            fill="toself",
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            fillcolor=COLORS[i % len(COLORS)],
            opacity=0.2
        ))
    max_val = summary[available_metrics].max().max()
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max_val * 1.3])),
        height=400,
        legend=dict(orientation="h", y=-0.25)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col2:
    st.subheader("Grouped bar — per metric")
    df_long = summary.melt(
        id_vars=strategy_col,
        value_vars=available_metrics,
        var_name="Metric", value_name="Score"
    )
    df_long["Metric"] = df_long["Metric"].str.replace("_", " ").str.title()
    fig_bar = px.bar(
        df_long, x="Metric", y="Score", color=strategy_col,
        barmode="group",
        color_discrete_sequence=COLORS
    )
    fig_bar.update_layout(height=400, legend=dict(orientation="h", y=-0.35))
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Full table ────────────────────────────────────────────────────
st.subheader("Summary table")
display = summary.set_index(strategy_col)[available_metrics]
st.dataframe(
    display.style
    .format("{:.4f}")
    .highlight_max(axis=0, color="#1a3a2a")
    .highlight_min(axis=0, color="#3a1a1a"),
    use_container_width=True
)

# ── Winner ────────────────────────────────────────────────────────
best_idx  = summary[main_metric].idxmax()
winner    = summary.loc[best_idx, strategy_col]
best_val  = summary.loc[best_idx, main_metric]
improvement = (best_val - baseline) / max(baseline, 0.0001) * 100
st.success(f"🏆 **Winner: {winner}** — {main_metric.replace('_',' ').title()} {best_val:.4f} ({improvement:+.1f}% over baseline)")

# ── Per-question drilldown ────────────────────────────────────────
st.divider()
st.subheader("Per-question drilldown")
choices = summary[strategy_col].tolist()
choice  = st.selectbox("Pick a strategy", choices)
filtered = df[df[strategy_col] == choice]

# Show whatever columns exist
show_cols = [c for c in filtered.columns if c != strategy_col]
fmt_cols  = {m: "{:.4f}" for m in available_metrics if m in show_cols}
st.dataframe(
    filtered[show_cols].reset_index(drop=True)
    .style.format(fmt_cols),
    use_container_width=True
)
