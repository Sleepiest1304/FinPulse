# FinPulse: Nifty 50 News Sentiment

[![CI](https://github.com/Sleepiest1304/FinPulse/actions/workflows/ci.yml/badge.svg)](https://github.com/Sleepiest1304/FinPulse/actions/workflows/ci.yml)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://finpulse-nifty.streamlit.app/)

**Live demo:** https://finpulse-nifty.streamlit.app/

FinPulse collects recent Yahoo Finance headlines for the 50 stocks in India's Nifty 50 index and scores each one with [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned for financial sentiment. It stores every scored headline in SQLite so a sentiment history builds up over time. It then tests that history against next-day and next-5-day stock returns and reports the results as they come out, including when they are weak.

> **Not financial advice.** FinPulse is a research and learning project. A correlation between headline sentiment and returns is not a prediction, and nothing here is a trading signal.

## Screenshots

| Live analysis | Backtest | Model evaluation |
|---|---|---|
| ![Live analysis](docs/screenshots/live.png) | ![Backtest](docs/screenshots/backtest.png) | ![Model evaluation](docs/screenshots/model_eval.png) |

Note the top row of the live analysis: FinBERT labels a fair-value *cut* as positive (0.93 confidence). It is a real example of the headline-level errors described under [Limitations](#limitations).


## What it does

- **Collect** (`python -m finpulse collect`): fetches headlines for all Nifty 50 tickers (or a subset), scores the ones not seen before with FinBERT, and saves the ticker, title, publisher, link, publish time, fetch time, label, confidence and class probabilities to `data/finpulse.db`. A scheduled GitHub Action runs it twice each weekday and commits the database back to the repo.
- **Dashboard** (`streamlit run app.py`, *Home* page): pick a stock, score its latest headlines live, optionally save them, and browse that stock's stored history.
- **Backtest** (`python -m finpulse backtest`, *Backtest* page): per ticker and pooled, compares the daily mean sentiment score with the forward 1-day and 5-day returns. Reports Pearson and Spearman correlation, a rough t-statistic, hit rate, the always-up baseline and sample size.
- **Model evaluation** (`python -m finpulse evaluate`, *Model Evaluation* page): compares FinBERT with VADER and a majority-class baseline on Financial PhraseBank (accuracy, macro F1, weighted F1). With `--labelled`, it compares them, plus a zero-shot NLI model, on labelled Indian headlines that none of the models were trained on. `--alternatives` reruns the scoring with borderline labels swapped.
- **Event study** (`python -m finpulse eventstudy --prices CSV --earnings CSV`): runs the pre-registered v3 tests on abnormal returns, using locally saved price and earnings files. Those files are not in the repo.
- **Event-type tagger check** (`python -m finpulse tagger-eval`): compares the rule-based event-type tagger with labelled headlines (accuracy, per-class precision and recall).

## Architecture

```mermaid
flowchart LR
    subgraph Sources
        YN[Yahoo Finance news<br/>via yfinance]
        YP[Yahoo Finance prices<br/>via yfinance]
        FPB[Financial PhraseBank<br/>via Hugging Face Hub]
    end

    subgraph finpulse ["finpulse package"]
        N[news.py<br/>schema-tolerant parsing,<br/>retries + backoff]
        PR[prices.py]
        S[sentiment.py<br/>FinBERT, lazy-loaded]
        ST[storage.py<br/>SQLite]
        BT[backtest.py<br/>no-look-ahead alignment,<br/>stats]
        EV[evaluation.py<br/>accuracy / F1]
    end

    GHA[[GitHub Actions<br/>collect.yml, cron]] --> CLI[cli.py<br/>python -m finpulse]
    CLI --> C[collect.py]
    YN --> N --> C
    C --> S --> C
    C --> ST
    ST --> DB[(data/finpulse.db)]
    DB --> BT
    YP --> PR --> BT
    FPB --> EV
    S --> EV

    subgraph ui ["Streamlit app: app.py + ui/"]
        A[Home<br/>live + history]
        P1[Backtest page]
        P2[Model Evaluation page]
    end
    DB --> A
    BT --> P1
    EV -- results/model_eval.json --> P2
```

### Backtest method

1. Each headline is assigned to the first NSE close at which it was public. A headline published by 15:30 IST counts for that day's close. A later one counts for the next trading day. Weekends and holidays roll forward. When the source gives no publish time, the (later) fetch time is used.
2. Per stock and trading day *D*, the signal is the mean FinBERT score (P(positive) − P(negative), from −1 to +1) of that day's headlines.
3. The target is the adjusted-close return from *D* to *D+1* and *D+5*. The move on the headline's own day is **not** counted, so any reaction during that day is excluded by design.
4. **Hit rate** is the share of days where the sign of the score matches the sign of the return. It should be read against the **up-day rate**, the hit rate of always guessing "up".

## Results

### Model evaluation (Financial PhraseBank v1.0)

Measured 2026-11-23 by the *Evaluate models* workflow ([`results/model_eval.md`](results/model_eval.md)):

| Subset | Model | n | Accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| allagree | majority (neutral) | 2264 | 0.614 | 0.254 | 0.468 |
| allagree | VADER | 2264 | 0.571 | 0.487 | 0.579 |
| allagree | FinBERT | 2264 | 0.972 | 0.963 | 0.972 |
| 50agree | majority (neutral) | 4846 | 0.594 | 0.248 | 0.443 |
| 50agree | VADER | 4846 | 0.543 | 0.489 | 0.549 |
| 50agree | FinBERT | 4846 | 0.890 | 0.883 | 0.891 |

Re-run it from Actions → *Evaluate models* → *Run workflow*.

**Important caveat:** `ProsusAI/finbert` was fine-tuned on Financial PhraseBank, so its scores on this dataset are largely in-sample and **overstate** how well it does on unseen headlines. The 0.97 on the unanimous-agreement subset says more about fit to training data than about how well it generalises. The 50agree subset adds sentences that human annotators disagreed on, which are harder for any model, and FinBERT drops to 0.89 there. VADER, a general-purpose lexicon model, has never seen this data, so it is the fairer reference. On accuracy it scores *below* the trivial "always neutral" baseline, but it does much better on macro F1.

### Model evaluation (Indian headlines, unseen by every model)

Measured 2026-11-25 on 150 headlines from Indian outlets, sampled from GDELT 2.0 GKG files (January to November 2026). Each headline was labelled positive, negative or neutral from a shareholder's point of view. **The labels were drafted with an LLM and then reviewed and confirmed by the project owner.** None of the models below was trained or tuned on these headlines. Full output is in [`results/model_eval_indian.md`](results/model_eval_indian.md).

Label mix: 73 neutral, 40 positive, 37 negative.

| Model | Accuracy [95% CI] | Macro F1 [95% CI] |
|---|---|---|
| majority (neutral) | 0.487 [0.407, 0.567] | 0.218 [0.193, 0.241] |
| VADER | 0.487 [0.407, 0.573] | 0.489 [0.403, 0.571] |
| FinBERT | **0.700** [0.620, 0.773] | **0.698** [0.618, 0.766] |
| Zero-shot NLI (`facebook/bart-large-mnli`) | 0.500 [0.420, 0.580] | 0.456 [0.398, 0.507] |

The CIs are bootstrap intervals over headlines (2,000 resamples).

- **FinBERT vs VADER:** FinBERT scores about 0.21 higher on both accuracy and macro F1, and the two confidence intervals do not overlap. On these headlines FinBERT is better than VADER. VADER's accuracy is no better than always guessing "neutral", but its macro F1 is much higher, as it was on Financial PhraseBank.
- **Zero-shot:** this is an open NLI classifier, not a large generative LLM. It labels few headlines neutral, so its accuracy is barely above the majority baseline and its macro F1 is below VADER's.

**Label sensitivity.** 12 of the 150 headlines were flagged as borderline when the labels were reviewed, each with a plausible alternative label. Swapping all 12 to their alternative gives:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| majority (neutral) | 0.433 | 0.202 |
| VADER | 0.500 | 0.502 |
| FinBERT | 0.680 | 0.683 |
| Zero-shot NLI | 0.553 | 0.485 |

The macro-F1 ranking is unchanged: FinBERT, then VADER, then zero-shot, then majority. The accuracy ranking *does* change, but only at the bottom. VADER and the majority baseline were exactly tied at 73/150 and the swap separates them. FinBERT stays first and zero-shot second on accuracy. The FinBERT-over-VADER result does not depend on how these borderline items are labelled.

### Event-type tagger check

The pre-registration (`docs/v3-preregistration.md`, section 2) requires the rule-based event-type tagger to agree with labelled headlines at least 70% of the time before event-type results get p-values. On 100 headlines drawn at random (seed 2026) from the project's headline database, labelled with the same six categories the tagger uses (labels drafted with an LLM and reviewed by the project owner), the results are in [`results/event_tagger_eval.md`](results/event_tagger_eval.md):

| Type | Labelled | Tagged | Precision | Recall |
|---|---:|---:|---:|---:|
| earnings | 26 | 27 | 0.93 | 0.96 |
| regulatory | 7 | 4 | 0.75 | 0.43 |
| rumour | 6 | 2 | 1.00 | 0.33 |
| management | 2 | 2 | 0.50 | 0.50 |
| macro | 10 | 9 | 0.22 | 0.20 |
| other | 49 | 56 | 0.75 | 0.86 |

Overall accuracy is **0.75** (95% CI 0.67 to 0.83) and macro F1 is 0.58, so the 70% threshold is met, though the lower end of the interval falls below it. The tagger is reliable for earnings and over-assigns "other". Recall for regulatory, rumour and macro is poor: the keyword lists miss most of these, so the small per-type groups in the event study are probably both undercounted and mixed. When the 10 borderline event-type labels are swapped to their alternatives, accuracy is 0.74 and the threshold is still met, but macro F1 falls to 0.48 (rumour precision and recall drop to 0 on 2 items).

### Backtest

First collection run, 2026-11-23: 370 headlines for 47 tickers (Yahoo returned none for ITC, JSWSTEEL and TRENT). Yahoo returns at most 10 headlines per ticker, but not only recent ones, so publish dates run from December 2025 to November 2026, mostly July to November 2026. Pooled across tickers:

| Horizon | n (stock-days) | Pearson r | Spearman ρ | t-stat | Hit rate | Up-day rate |
|---|---:|---:|---:|---:|---:|---:|
| 1 day | 305 | +0.003 | +0.020 | 0.06 | 55.4% | 54.1% |
| 5 day | 291 | −0.153 | −0.169 | −2.63 | 49.8% | 54.3% |

How to read this:

- **1 day:** no measurable relationship. The hit rate is about one point above the always-up baseline, which is within noise.
- **5 day:** a weak *negative* correlation. More positive headlines went with slightly lower returns over the next five days, and the hit rate is *below* the always-up baseline. The t-stat looks notable but is overstated: 5-day windows on consecutive days overlap, stocks co-move, and this is one of 2 horizons × 47 tickers tested. Treat it as noise until it holds up on later, out-of-sample data.
- **Sample:** the sample is small and not a random draw. It contains only the headlines Yahoo still served on one day, spread unevenly over a year (1 to 10 per ticker).

The collector workflow refreshes these numbers in [`results/backtest_summary.csv`](results/backtest_summary.csv), including per-ticker rows, and the Backtest page computes them live. The table above records the first run and will not update itself.

### Event study (v3, pre-registered)

The question: does negative news mean something different for stocks whose latest earnings beat or missed expectations, and do post-news drops reverse over the following days? The hypotheses, windows, groups and tests were written down in [`docs/v3-preregistration.md`](docs/v3-preregistration.md) and committed before any test was run. Later clarifications are logged in that file's amendment log.

**Method, briefly:**
- **Abnormal returns:** stock return minus the Nifty 50 index return. CAR[a,b] sums these from trading day a to day b, where day 0 is the first trading day on which the headline could be acted on.
- **Events:** an event is one stock-day of headlines, scored by FinBERT. Only novel headlines count; near-duplicates of the same stock's last 3 days are dropped. An event is "negative" if its mean score is below 0.
- **Fundamentals:** the earnings surprise % from the most recent release that was public *before* day 0, so there is no look-ahead. Releases are split into terciles using cut-offs from all 228 releases in the sample.
- **Earnings window:** events with an earnings date between day −1 and day 5 are left out of H1–H4, so the tests aren't just picking up the results announcement.
- **Inference:** standard errors are clustered by date and by ticker, and each test also gets a within-month permutation test (10,000 shuffles). p-values are Holm-adjusted across H1–H4.

**Data:** the database frozen at commit `6445562`: 398 headlines, 47 tickers, 331 stock-day events. 77 of the events were negative, and 75 events fell near an earnings date. Results: [`results/v3_confirmatory.csv`](results/v3_confirmatory.csv), with exploratory tests and robustness checks in `results/v3_exploratory.csv` and `results/v3_robustness.csv`.

| Test | Question | n | Estimate [95% CI] | p (clustered) | p (permutation) | p (Holm) |
|---|---|---:|---|---:|---:|---:|
| H1 | Mean CAR[2,5] after negative events (reversion?) | 55 | +0.19 pp [−1.07, +1.45] | 0.76 | 0.68 | 0.76 |
| H2 | Slope of CAR[2,5] on CAR[0,1]: negative events minus no-news days | 55 vs 9,536 | +0.48 [+0.05, +0.90] | 0.028 | 0.039 | 0.083 |
| **H3** | CAR[0,1] after negative events: top minus bottom surprise tercile | 35 (19 / 16) | **+2.27 pp** [+1.17, +3.37] | 0.0003 | 0.0036 | **0.0013** |
| H4 | CAR[2,5] after negative events: top minus bottom surprise tercile | 33 (17 / 16) | +2.29 pp [+0.05, +4.53] | 0.045 | 0.029 | 0.090 |

How to read this:

- **No reversion (H1, H2).** After negative news, days 2–5 show no measurable bounce-back: +0.19 pp, with an interval that comfortably includes zero. H2's slope points the other way from reversion: after negative events, the day 0–1 move tended to *continue* slightly more than on no-news days. That is not significant after the Holm correction.
- **Fundamentals matter for the reaction (H3).** This is the only result that survives the correction. Negative news about a stock whose last results beat expectations went with a CAR[0,1] of **+1.06 pp** on average (19 events). For a stock whose last results missed, the figure was **−1.21 pp** (16 events), a gap of 2.27 pp. The gap holds when any single ticker is dropped (2.06 to 2.51 pp), and its CAR[2,5] counterpart (H4) points the same way, though H4 doesn't survive the correction.
- **Exploratory tests** (Benjamini–Hochberg, q = 0.10): most event-type groups had fewer than 15 negative events, so they were reported as untested, as pre-registered. Negative earnings headlines (n = 21) showed nothing measurable. The sentiment score's slope on next-day and days 1–5 returns is close to zero (r = +0.01 and −0.07). The score × surprise interaction (S14) is significant (q = 0.015), but it uses the same data as H3, so it is not independent evidence.

**Treat H3 as a hypothesis to test again, not a finding.** Reasons:
- **The sample is small:** 35 events, 21 tickers, 31 dates.
- **The result is fragile in the robustness checks.** With a stricter definition of "negative" (score ≤ −0.3), or one event per ticker per 5 trading days, the groups fall below 15 and the test cannot be run. With earnings-window events kept in, the gap shrinks to 1.44 pp (Holm p 0.084).
- **It isn't specific to bad news.** A post-hoc check, not pre-registered, gives a top-minus-bottom gap of +0.94 pp for *non-negative* events too. So part of H3 may be a general post-earnings drift between beats and misses, rather than something specific to negative headlines.
- **The news is noisy.** Many headlines in these groups are only loosely about the stock they are filed under.

A fair test would use a fresh, later sample, for example the GDELT Indian-outlet headlines described above, with the same pre-registered rules. This is descriptive research, not a trading signal.

## Limitations

- **Correlation is not prediction.** Even a statistically significant correlation in this setup would not mean you could trade on it profitably: transaction costs, slippage and multiple-testing across 50 tickers × 2 horizons are not accounted for.
- **Small, biased news sample.** Yahoo Finance's feed is mostly English-language, international-facing coverage, often syndicated or auto-generated ("fair value estimate" style stories). It is not representative of the news Indian market participants act on. Some items are only loosely related to the stock they are filed under (for example, a general piece on the US housing market listed for Apollo Hospitals), which adds noise to that stock's score.
- **Headline-only, off-domain model.** FinBERT reads only titles, truncated to 128 tokens, and was fine-tuned on English financial news about Finnish companies (Financial PhraseBank), not Indian market news.
- **Small, LLM-assisted label sets.** The Indian-headline comparison uses 150 headlines and the tagger check uses 100, so the confidence intervals are wide (about ±0.08 on accuracy). Both sets were drafted with an LLM and reviewed and confirmed by the project owner. That is one reviewer, with no inter-annotator agreement measured, and the review may be anchored on the LLM's draft. The labelled files hold third-party headline text and are not in this repository. Results from a single label pass should be treated as indicative only.
- **FinBERT's training data.** FinBERT was fine-tuned on Financial PhraseBank, so its PhraseBank scores are in-sample. The Indian-headline set is the out-of-sample check, and FinBERT's accuracy drops from 0.89–0.97 there to 0.70.
- **Event study power.** The pre-registered power calculation assumed about 70 negative events and put the smallest reliably detectable mean CAR[2,5] at about 1.1 pp. Only 55 remained after exclusions, so the real threshold is higher. Smaller reversions would be missed, so the H1 null result does not rule out a small effect. The earnings surprises come from Yahoo Finance and have not been checked against exchange filings.
- **Pooled observations are not independent.** Stocks move together, so the t-statistic is a rough guide, not a valid p-value.
- **Survivorship and membership drift.** The ticker list in `finpulse/config.py` is a snapshot of current Nifty 50 members and is not point-in-time.
- **Ephemeral storage on Streamlit Cloud.** Headlines saved from the deployed app's *Save* checkbox are lost when the app restarts. Lasting history comes from the GitHub Action.

## Project structure

```text
FinPulse/
├── app.py                     # Streamlit entry point: sidebar navigation only
├── finpulse/                  # core library (no Streamlit imports)
│   ├── __main__.py            # enables `python -m finpulse`
│   ├── cli.py                 # all commands (collect, backtest, evaluate, eventstudy, ...)
│   ├── config.py              # Nifty 50 tickers, paths, input validation
│   ├── news.py                # Yahoo headlines, schema-tolerant parsing
│   ├── prices.py              # Yahoo daily closes
│   ├── retry.py               # retry with exponential backoff
│   ├── sentiment.py           # FinBERT inference (lazy-loaded)
│   ├── storage.py             # SQLite persistence
│   ├── collect.py             # one fetch → score → store pass
│   ├── backtest.py            # no-look-ahead alignment + statistics
│   ├── evaluation.py          # model comparison and event-tagger check
│   ├── novelty.py             # headline de-duplication
│   ├── events.py              # rule-based event-type tagger
│   ├── eventstudy.py          # abnormal returns, events, point-in-time surprise
│   ├── stats.py               # clustered OLS, p-values, multiple-testing corrections
│   ├── v3tests.py             # pre-registered v3 tests
│   ├── gdelt.py               # resumable GDELT pull, Nifty 50 name matching
│   └── replication.py         # pre-registered H3 replication
├── ui/                        # Streamlit layer, depends on finpulse/
│   ├── cache.py               # st.cache wrappers around finpulse calls
│   └── pages/
│       ├── home.py            # live scoring + stored history
│       ├── backtest.py        # sentiment vs forward returns
│       └── model_evaluation.py
├── tests/                     # tests for every finpulse module; network mocked
├── data/finpulse.db           # headline history, committed by the collector
├── results/                   # backtest, model evaluation and v3 summary tables
├── docs/                      # v3 pre-registration, screenshots
├── .github/workflows/         # ci.yml, collect.yml, evaluate.yml
├── pyproject.toml             # tool config (pytest)
├── requirements.txt           # runtime deps (also used by Streamlit Cloud)
└── requirements-dev.txt       # pytest, vaderSentiment
```

Dependencies point one way: `ui/` imports `finpulse/`, never the reverse. So the core logic can be tested and run from the CLI without Streamlit.

## Running locally

```bash
git clone https://github.com/Sleepiest1304/FinPulse.git
cd FinPulse
python -m venv venv
# Windows: .\venv\Scripts\activate    macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python -m finpulse collect              # all 50 tickers (first run downloads FinBERT, ~440 MB)
python -m finpulse collect TCS INFY     # or a subset
python -m finpulse backtest             # writes results/backtest_*.csv
streamlit run app.py
```

Model evaluation and tests:

```bash
pip install -r requirements-dev.txt
python -m finpulse evaluate             # downloads Financial PhraseBank from the Hugging Face Hub
pytest
```

Run `python -m finpulse <command> --help` for each command's options.

Tests mock all network calls and never load FinBERT, so they run without torch or transformers installed. That is exactly how CI runs them.

The database path can be overridden with the `FINPULSE_DB` environment variable. Running the collector locally changes `data/finpulse.db`, which the scheduled workflow also commits, so pull before you push.

## Deploying to Streamlit Community Cloud

No secrets or API keys are needed.

1. Push this repository to GitHub.
2. Under **Settings → Actions → General → Workflow permissions**, choose **Read and write permissions** so the collector can commit `data/finpulse.db`.
3. Open **Actions → Collect headlines → Run workflow** once to seed the database. After that it runs on its schedule (about 08:53 and 16:53 IST, Monday to Friday; GitHub may start scheduled runs late).
4. Go to [share.streamlit.io](https://share.streamlit.io), click **Create app**, and choose this repo, branch `main`, main file `app.py`.
5. Under **Advanced settings**, set Python to **3.12**, then **Deploy**.
6. Streamlit Cloud redeploys when the branch changes, so each collector commit brings new history into the app.

`requirements.txt` pulls CPU-only PyTorch wheels to keep the install small. The app's first FinBERT load downloads about 440 MB of weights, and the model needs several hundred MB of RAM. Check that it stays within Community Cloud's resource limits after you deploy.

## Data and licences

- News and prices: Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance), subject to Yahoo's terms. Not for redistribution or commercial use.
- [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (Malo et al., 2014): CC BY-NC-SA 3.0. It is downloaded at runtime and not stored in this repo.
- [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert) (Araci, 2019).
- Indian headlines for the model comparison: [The GDELT Project](https://www.gdeltproject.org/) (GDELT 2.0 Global Knowledge Graph), free to use with citation. Headline text is not stored in this repo.
- [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli), the zero-shot model, used through the Hugging Face transformers zero-shot pipeline.
