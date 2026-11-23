import pandas as pd
import streamlit as st

from engine import storage, ui
from engine.config import FINBERT_MODEL, NIFTY50

st.set_page_config(page_title="FinPulse: Nifty 50 Sentiment", layout="wide")

st.title("FinPulse: Nifty 50 News Sentiment")
st.markdown(
    "Scores recent Yahoo Finance headlines for **Nifty 50** stocks with "
    "[FinBERT](https://huggingface.co/ProsusAI/finbert). "
    "Score = P(positive) − P(negative), from −1 to +1. "
    "*Research tool only, not financial advice.*"
)

ticker = st.selectbox("Stock", NIFTY50, index=NIFTY50.index("RELIANCE.NS"), format_func=ui.short)

col1, col2 = st.columns([1, 3])
run = col1.button("Fetch & score headlines", type="primary")
save = col2.checkbox("Save results to the history database", value=False,
                     help="On Streamlit Community Cloud the disk is reset on restart; "
                          "the scheduled GitHub Action is what builds lasting history.")

if run:
    try:
        with st.spinner("Fetching headlines..."):
            items = ui.news(ticker)
    except Exception as exc:
        st.error(f"Could not reach Yahoo Finance after retries: {exc}")
        items = None

    if items == []:
        st.warning("Yahoo Finance returned no usable headlines for this ticker right now.")
    elif items:
        with st.spinner("Scoring with FinBERT..."):
            results = ui.scored(tuple(it["title"] for it in items))
        rows = [{**it, **res} for it, res in zip(items, results)]
        df = pd.DataFrame(rows)

        m1, m2, m3 = st.columns(3)
        m1.metric("Headlines", len(df))
        m2.metric("Mean score", f"{df['score'].mean():+.2f}")
        m3.metric("Positive / Negative / Neutral",
                  " / ".join(str((df["label"] == lbl).sum())
                             for lbl in ("positive", "negative", "neutral")))

        st.dataframe(
            df[["published_at", "title", "publisher", "label", "confidence", "score", "link"]],
            hide_index=True, width="stretch",
            column_config={
                "link": st.column_config.LinkColumn("link", display_text="open"),
                "confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1, format="%.2f"),
                "score": st.column_config.NumberColumn("score", format="%+.2f"),
            },
        )
        if save:
            written = storage.save_scored(rows, FINBERT_MODEL)
            ui.headlines.clear()
            st.success(f"Saved {written} new headline(s); {len(rows) - written} were already stored.")

st.divider()
st.subheader(f"Stored history: {ui.short(ticker)}")
hist = ui.headlines(ticker)
if hist.empty:
    st.info("No stored headlines for this ticker yet. Run `python scripts/collect.py` "
            "or let the scheduled GitHub Action collect them.")
else:
    when = pd.to_datetime(hist["published_at"].fillna(hist["fetched_at"]), utc=True,
                          errors="coerce", format="ISO8601")
    daily = (hist.assign(date=when.dt.tz_convert("Asia/Kolkata").dt.date)
                 .groupby("date")["score"].agg(["mean", "size"])
                 .rename(columns={"mean": "mean score", "size": "headlines"}))
    st.caption(f"{len(hist)} headlines from {daily.index.min()} to {daily.index.max()} (IST dates)")
    st.line_chart(daily["mean score"])
    st.dataframe(hist[["published_at", "title", "publisher", "label", "confidence", "score"]],
                 hide_index=True, width="stretch")
