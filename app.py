"""Entry point: defines the sidebar navigation and runs the selected page."""
import streamlit as st

st.set_page_config(page_title="FinPulse: Nifty 50 Sentiment", layout="wide")

page = st.navigation([
    st.Page("ui/pages/home.py", title="Home", default=True),
    st.Page("ui/pages/backtest.py", title="Backtest", url_path="Backtest"),
    st.Page("ui/pages/model_evaluation.py", title="Model Evaluation", url_path="Model_Evaluation"),
])
page.run()
