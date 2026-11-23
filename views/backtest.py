import pandas as pd
import streamlit as st

from engine import ui
from engine.backtest import build_panel, summarize

st.set_page_config(page_title="FinPulse: Backtest", layout="wide")
st.title("Backtest: sentiment vs. forward returns")
st.markdown(
    "For each stock and trading day, the mean FinBERT score of headlines known by the "
    "15:30 IST close is compared with the return from that close to the close 1 and 5 "
    "trading days later. Prices: Yahoo Finance, split/dividend adjusted.\n\n"
    "**Correlation is not prediction**, and small samples are noisy. Nothing here is tuned "
    "or filtered to look better, and none of it is financial advice."
)

heads = ui.headlines()
if heads.empty:
    st.info("No stored headlines yet. Run `python scripts/collect.py` first.")
    st.stop()

when = pd.to_datetime(heads["published_at"].fillna(heads["fetched_at"]), utc=True,
                      errors="coerce", format="ISO8601")
start = (when.min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
try:
    with st.spinner("Downloading prices..."):
        px = ui.prices(tuple(sorted(heads["ticker"].unique())), start)
except Exception as exc:
    st.error(f"Could not download prices: {exc}")
    st.stop()

panel = build_panel(heads, px)
summary = summarize(panel)

horizon = st.radio("Horizon", ["1d", "5d"], horizontal=True)
ret_col = f"ret_{horizon}"
pooled = summary[(summary.ticker == "ALL") & (summary.horizon == horizon)].iloc[0]

st.caption(f"{len(heads)} headlines · {heads['ticker'].nunique()} tickers · "
           f"{when.min():%Y-%m-%d} to {when.max():%Y-%m-%d}")
c = st.columns(5)
c[0].metric("Observations (stock-days)", int(pooled.n))
c[1].metric("Pearson r", "n/a" if pd.isna(pooled.pearson) else f"{pooled.pearson:+.3f}")
c[2].metric("Spearman ρ", "n/a" if pd.isna(pooled.spearman) else f"{pooled.spearman:+.3f}")
c[3].metric("Hit rate", "n/a" if pd.isna(pooled.hit_rate) else f"{pooled.hit_rate:.1%}")
c[4].metric("Up-day rate (baseline)", "n/a" if pd.isna(pooled.up_rate) else f"{pooled.up_rate:.1%}")

if pooled.n < 100:
    st.warning(f"Only {int(pooled.n)} observations with a known {horizon} forward return. "
               "Treat every number on this page as anecdotal until history builds up.")

data = panel.dropna(subset=[ret_col])
if not data.empty:
    st.subheader("Daily score vs. forward return")
    st.scatter_chart(data.assign(**{ret_col: data[ret_col] * 100}),
                     x="score", y=ret_col, color="ticker", height=420)
    st.caption(f"y axis: {horizon} forward return in %")

per = summary[(summary.ticker != "ALL") & (summary.horizon == horizon)].set_index("ticker")
per_valid = per.dropna(subset=["pearson"])
if not per_valid.empty:
    st.subheader("Per-stock correlation")
    st.bar_chart(per_valid["pearson"], height=320)

st.subheader("All statistics")
st.dataframe(summary, hide_index=True, width="stretch",
             column_config={k: st.column_config.NumberColumn(format="%.3f")
                            for k in ["pearson", "spearman", "t_stat", "hit_rate", "up_rate"]})
with st.expander("Definitions"):
    st.markdown(
        "- **n**: stock-days with at least one headline and a known forward return.\n"
        "- **Pearson / Spearman**: correlation of daily mean score with forward return.\n"
        "- **t_stat**: r·√((n−2)/(1−r²)). A rough guide only; pooled stock-days are not independent.\n"
        "- **Hit rate**: share of days where sign(score) = sign(return), excluding zeros.\n"
        "- **Up-day rate**: share of those days with a positive return, i.e. what "
        "always guessing *up* would score. Compare hit rate against this.")
