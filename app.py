import streamlit as st
import pandas as pd
from engine.data_fetch import fetch_nifty_news
from engine.sentiment import get_sentiment

st.set_page_config(page_title="FinPulse: Nifty 50 AI", layout="wide")

st.title("📈 FinPulse: Indian Market Sentiment Engine")
st.markdown("Analyzing real-time headlines for **Nifty 50** stocks using FinBERT.")

# User selection
ticker = st.selectbox("Select a Nifty 50 Stock", ["RELIANCE.NS", "TCS.NS", "HDFC BANK.NS", "INFY.NS", "ICICIBANK.NS"])

if st.button("Run AI Analysis"):
    with st.spinner("Processing news data..."):
        headlines = fetch_nifty_news(ticker)
        
        if headlines:
            scores = get_sentiment(headlines)
            
            for i, text in enumerate(headlines):
                with st.expander(f"Headline {i+1}: {text[:60]}..."):
                    st.write(f"**Full Headline:** {text}")
                    
                    # Create data for the chart
                    sentiment_df = pd.DataFrame({
                        'Sentiment': ['Positive', 'Negative', 'Neutral'],
                        'Confidence': scores[i].detach().numpy().tolist()
                    })
                    st.bar_chart(sentiment_df.set_index('Sentiment'))
        else:
            st.error("No news found for this ticker.")