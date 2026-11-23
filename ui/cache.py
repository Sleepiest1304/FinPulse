"""Streamlit-cached wrappers shared by the app pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from finpulse import storage
from finpulse.news import fetch_news
from finpulse.prices import fetch_prices
from finpulse.sentiment import get_sentiment, load_model


@st.cache_resource(show_spinner="Loading FinBERT (first run downloads ~440 MB)...")
def model():
    return load_model()


@st.cache_data(ttl=900, show_spinner=False)
def news(ticker: str) -> list[dict]:
    return fetch_news(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def scored(titles: tuple[str, ...]) -> list[dict]:
    model()  # make sure the cached model is loaded once per process
    return get_sentiment(list(titles))


@st.cache_data(ttl=3600, show_spinner=False)
def prices(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return fetch_prices(list(tickers), start=start)


@st.cache_data(ttl=300, show_spinner=False)
def headlines(ticker: str | None = None) -> pd.DataFrame:
    return storage.load_headlines(ticker)


def short(ticker: str) -> str:
    return ticker.removesuffix(".NS")
