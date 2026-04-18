# FinPulse: Nifty 50 AI Sentiment Engine 📈

**FinPulse** is an end-to-end data analytics tool designed for the Indian stock market. It utilizes **Deep Learning (NLP)** to quantify market sentiment from real-time news headlines of Nifty 50 companies, helping investors distinguish market "noise" from actionable sentiment triggers.

## 🚀 The Problem
Retail investors in the Indian market are often overwhelmed by a high volume of financial news. FinPulse solves this by providing a numerical "Sentiment Score" for Blue Chip stocks, allowing for a data-driven approach to market volatility.

## 🛠️ Tech Stack
* **Language:** Python
* **AI/ML:** PyTorch, Hugging Face Transformers (**FinBERT**)
* **Data Ingestion:** yfinance API (Real-time Nifty 50 data)
* **Dashboard:** Streamlit (Interactive UI)
* **Environment:** Virtual Environments (venv), Git/GitHub

## 🌟 Key Features
* **Real-time News Pipeline:** Automated ingestion of the latest headlines for top Indian tickers (Reliance, TCS, HDFC, etc.).
* **Financial NLP:** Implements the **FinBERT** model, specifically pre-trained on financial corpora for superior accuracy over generic sentiment tools.
* **Interactive Visualization:** A custom-built Streamlit dashboard that allows users to toggle between stocks and visualize sentiment confidence levels.
* **Robust Error Handling:** Custom logic to handle dynamic API schema changes and nested JSON data structures.

## 📂 Project Structure
```text
FinPulse/
├── app.py              # Main Streamlit Dashboard
├── engine/
│   ├── data_fetch.py   # Yahoo Finance API Integration
│   └── sentiment.py    # FinBERT Model Inference Logic
├── requirements.txt    # Dependency Management
└── README.md           # Project Documentation
