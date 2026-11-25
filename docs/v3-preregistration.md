# FinPulse v3: pre-registration

Status: **draft, written before any v3 test was run.** The commit that adds this file is the registration timestamp. Any change after that point goes in the amendment log at the bottom, with the date and the reason, and is made before the affected test is run.

This is a research and analytics exercise. It tests whether patterns exist in past data. It makes no forecasts and gives no trading recommendations. A null result will be reported as a null result.

## 0. What has already been seen

Not everything below is a clean pre-registration, because some results on the same data were looked at during v1/v2:

- Raw 5-day correlation between sentiment and return: −0.17. Market-adjusted (stock minus Nifty 50): −0.07, not significant. The 1-day correlation is flat.
- Abnormal 5-day return on negative-score days minus non-negative days: +0.3 pp (permutation p ≈ 0.50).

Not yet looked at: anything conditional on earnings surprise, the day 2–5 windows, event types and novelty. The hypotheses in section 3 cover only these.

## 1. Data (frozen)

| Input | Definition |
|---|---|
| Headlines | `data/finpulse.db` as of commit `6445562` (2026-11-25): 398 headlines, 47 tickers. Later bot commits are excluded from the primary analysis. |
| Prices | Daily adjusted closes from yfinance, for the 47 tickers and `^NSEI`, from 2025-11-01. Downloaded at analysis time and never committed. |
| Earnings | yfinance earnings dates with EPS estimate, reported EPS and surprise %. Kept local and never committed. |
| Sentiment | FinBERT score = P(positive) − P(negative), as already stored. |

New headline sources (GDELT, BSE) are **not** part of this registration. If one is added, it gets its own amendment, written before any return is joined to it, and is analysed as a separate replication sample.

## 2. Definitions

**Entry day D.** The first NSE close that reflects the headline: 15:30 IST or earlier counts for that day, later rolls to the next trading day (same rule as `finpulse/backtest.py`).

**Abnormal return.** AR_t = r_stock,t − r_Nifty50,t, using simple daily returns. This is a market-adjusted model; there is no beta estimate, because each stock has only about a year of history. CAR[a,b] is the sum of AR over trading days D+a … D+b.

| Window | Meaning |
|---|---|
| CAR[0,1] | Initial reaction: entry day plus the next day |
| CAR[1,5] | Post-entry drift (what v1 measured, now market-adjusted) |
| CAR[2,5] | Reversion window |

Day 0 is only partly after the news for headlines published during market hours. That is a known limitation and is reported as such.

**Novelty (de-duplication).** A title is normalised: Unicode NFKC, lowercase, publisher suffix after " - " or " | " removed, punctuation removed, small stopword list removed, then turned into a set of tokens. A headline is a **repeat** if the same ticker has an earlier headline in the previous 3 calendar days with token Jaccard similarity ≥ 0.6. Otherwise it is **novel**. A title stored under 3 or more tickers on the same day is flagged `multi_ticker`.

**Event.** One (ticker, D) pair. Event score = mean FinBERT score of the *novel* headlines on that pair. A pair with only repeat headlines is dropped from the primary tests.

**Negative event.** Event score < 0. Robustness check: score ≤ −0.3.

**Event type.** A rule-based keyword tagger. Each headline gets one primary type, checked in this order:

1. `earnings`: results, earnings, quarterly, Q1–Q4, profit, net income, revenue, EPS, EBITDA, margin, guidance, beat, miss
2. `regulatory`: SEBI, RBI, CCI, NCLT, court, tribunal, regulator, probe, investigation, penalty, fine, ban, licence/license, lawsuit, tax demand, GST notice, Enforcement Directorate
3. `rumour`: reportedly, sources, people familiar, said to, in talks, mulls, weighs, considering, rumour/rumor, speculation
4. `management`: CEO, CFO, MD, chairman/chairperson, board, director, resigns/resignation, appoints/appointed, steps down, succession, promoter, founder
5. `macro`: Nifty, Sensex, rupee, inflation, GDP, repo rate, interest rate, Fed, tariff, crude/oil prices, monetary policy, budget, economy, stocks to watch, top gainers/losers; plus any `multi_ticker` headline
6. otherwise `other`

Matching is on whole words and is case-insensitive. An event takes the type of its novel headlines. If they disagree, it takes the type earliest in the order above. Before any event-type test, the tagger is checked against 100 randomly drawn headlines labelled by hand (seed 2026). If agreement is below 70%, the event-type results are reported as descriptive only, with no p-values.

**Fundamentals (point in time).** For an event on day D, the measure is the surprise % from the most recent earnings release whose *availability day* is strictly before D. The availability day is the first NSE trading day strictly after the earnings calendar date in IST, because Yahoo's intraday times for Indian earnings are placeholders. A release more than 120 calendar days old counts as no data. Surprise is winsorised at the 5th and 95th percentiles.

**Surprise terciles.** Cut points are the 33rd and 67th percentiles of winsorised surprise across all earnings releases with a reported surprise from 2025-09-01 to 2026-11-25, giving bottom / middle / top. The cut points use the whole cross-section, which is a mild departure from point-in-time. The robustness check that is strictly point-in-time is a sign split: surprise < 0 versus ≥ 0.

**Earnings exclusion.** For H1–H4, events with any earnings date in [D−1, D+5] are dropped, so the surprise itself is not what moves the window. A robustness check keeps them.

## 3. Hypotheses (confirmatory family)

All tests are two-sided and run on negative events after the earnings exclusion.

| ID | Question | Estimate | Null |
|---|---|---|---|
| H1 | Do post-news moves revert on average? | mean CAR[2,5] | = 0 |
| H2 | Is reversion after a negative event stronger than ordinary short-term reversal? | β_neg − β_base, from CAR[2,5] = a + β·CAR[0,1]. β_base uses all stock-days with no headline in [D−5, D+5]. | = 0 |
| H3 | Does the initial reaction differ with fundamentals? | mean CAR[0,1], top surprise tercile minus bottom | = 0 |
| H4 | Does reversion differ with fundamentals? | mean CAR[2,5], top tercile minus bottom | = 0 |

**Multiple comparisons.** Holm correction across H1–H4 at family-wise α = 0.05.

## 4. Secondary (exploratory family)

These use the Benjamini–Hochberg false discovery rate at q = 0.10 across all of them. They are labelled exploratory everywhere they are reported.

- S1–S5: for each event type except `other`, mean CAR[0,1] for negative events.
- S6–S10: the same with CAR[2,5].
- S11: the correlation between event score and CAR[1,5] on novel events only. This repeats the −0.07 check with de-duplicated data.
- S12: the same for CAR[1,1].
- S13: mean CAR[0,1] for negative events, novel minus repeat. Repeat-only pairs are included here.
- S14: across all events, the interaction between score and surprise in CAR[1,5] = a + b·score + c·surprise_w + d·score×surprise_w. The test is on d.

A group with fewer than 15 events is reported as "n too small" and is not tested. It still counts in the BH family, at p = 1.

## 5. Inference

- **Standard errors.** Means and regressions are estimated with OLS. Standard errors are two-way clustered by entry date and by ticker (Cameron–Gelbach–Miller), with the small-sample factor G/(G−1) on each component. p-values come from a t distribution with min(G_date, G_ticker) − 1 degrees of freedom.
- **Randomisation check.** Each confirmatory test is also run with 10,000 permutations of the group labels (or of the score) within calendar month, seed 2026. If the clustered result and the permutation result disagree on significance, both are reported and the claim is not made.
- **Robustness checks.** These are reported but are not tests: CARs winsorised at 1/99, the negative-event threshold ≤ −0.3, keeping events in the earnings window, and keeping only the first event when one ticker has events within 5 trading days of each other.

## 6. Power (stated before running)

The standard deviations come from prices only, with no sentiment involved: the stock-day spread of CAR[0,1] is 2.0%, of CAR[1,5] 3.2% and of CAR[2,5] 2.8%. The v1 panel has 80 negative (ticker, day) pairs. That count was taken before de-duplication and the earnings exclusion, so the real n will be lower.

- H1 (n ≈ 70): the minimum detectable mean CAR[2,5] is about **1.1 pp** at 80% power, using the strictest Holm level of 0.0125. That figure assumes independent events, and clustering makes it worse.
- H3/H4 (about 22 per tercile): the minimum detectable top-minus-bottom difference is about **2.0 pp** for CAR[0,1] and **2.8 pp** for CAR[2,5].

These tests can only detect large effects. Not rejecting the null does **not** show that there is no effect. The write-up will say so and report confidence intervals, not just p-values.

## 7. Reporting commitments

- Every test in sections 3 and 4 is reported, whatever its result, with n, the estimate, the 95% confidence interval, the clustered p, the permutation p and the adjusted p.
- No buy/sell language, forecasts or "top picks". Results describe past data only.
- Data files (prices, earnings, any new headline source) stay out of the repository. Only code, tests and summary tables are committed.

## Amendment log

_None yet._
