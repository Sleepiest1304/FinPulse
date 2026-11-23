"""FinPulse: Nifty 50 headline sentiment with FinBERT.

Core library with no Streamlit dependency. Modules:
    config      tickers, paths, input validation
    news        Yahoo Finance headlines (schema-tolerant parsing)
    prices      Yahoo Finance daily closes
    retry       retry with exponential backoff
    sentiment   FinBERT inference (lazy-loaded)
    storage     SQLite persistence
    collect     one fetch -> score -> store pass
    backtest    sentiment vs forward returns
    evaluation  FinBERT vs baselines on Financial PhraseBank
    cli         `python -m finpulse collect|backtest|evaluate`
"""
