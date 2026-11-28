"""FinPulse: Nifty 50 headline sentiment with FinBERT.

Core library with no Streamlit dependency. Modules:
    config       tickers, paths, input validation
    news         Yahoo Finance headlines (schema-tolerant parsing)
    prices       Yahoo Finance daily closes
    retry        retry with exponential backoff
    sentiment    FinBERT inference (lazy-loaded)
    storage      SQLite persistence
    collect      one fetch -> score -> store pass
    backtest     sentiment vs forward returns
    evaluation   model comparison (PhraseBank, labelled Indian headlines, event tagger)
    novelty      headline de-duplication
    events       rule-based event-type tagging
    eventstudy   abnormal returns, events and point-in-time earnings surprise
    stats        clustered OLS, t p-values, Holm and Benjamini-Hochberg
    v3tests      the pre-registered v3 tests
    gdelt        GDELT GKG pull and Nifty 50 name matching
    replication  the pre-registered H3 replication on GDELT headlines
    cli          `python -m finpulse <command>`
"""
