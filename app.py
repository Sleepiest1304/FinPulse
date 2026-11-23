"""Entry point: defines the sidebar navigation and runs the selected page."""
import streamlit as st

st.set_page_config(page_title="FinPulse: Nifty 50 Sentiment", layout="wide")

page = st.navigation([
    st.Page("views/home.py", title="Home", default=True),
    st.Page("views/backtest.py", title="Backtest", url_path="Backtest"),
    st.Page("views/model_evaluation.py", title="Model Evaluation", url_path="Model_Evaluation"),
])
page.run()
