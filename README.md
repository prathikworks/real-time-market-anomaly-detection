# Real-Time Market Anomaly Detection & News Attribution System

An evolutionary-model project that monitors Nifty and Sensex indices for
sudden price movements, identifies the probable cause from financial news,
and delivers a plain-English explanation to the user.

---

## Project Structure

```
real-time-market-anomaly-detection/
├── config.py               # Centralised config loader (reads .env)
├── detector.py             # Pure anomaly-detection logic (I/O-free, fully testable)
├── price_monitor.py        # yfinance price fetcher + rolling PriceBuffer
├── news_fetcher.py         # NewsSource ABC + NewsAPISource + NewsIngester
├── relevance_filter.py     # Keyword-based article relevance scoring (offline)
├── causal_prefilter.py     # Offline causal ranking (recency, keywords, source tier)
├── causal_analyzer.py      # LLM causal analysis layer (Gemini, pluggable)
├── main.py                 # Entry point — live monitor + backtest mode
├── requirements.txt        # Runtime dependencies
├── .env.example            # Copy → .env and fill in your values
├── tests/
│   ├── __init__.py
│   ├── test_detector.py        # Iteration 1 — 20 offline unit tests
│   ├── test_news_ingestion.py  # Iteration 2 — 39 offline unit tests
│   ├── test_causal_iteration3.py  # Iteration 3 — 37 offline unit tests
│   └── test_fixes.py           # Targeted fix tests — 33 offline unit tests
└── README.md
```

---

## Iteration 1 — Price Movement Detection ✅

### What was built

| Module | Responsibility |
|---|---|
| `config.py` | Loads all config from `.env` and exposes typed constants. |
| `detector.py` | `check_window()` — tests one snapshot for an anomaly. `scan_series_for_events()` — slides the window across historical data. No I/O. |
| `price_monitor.py` | `fetch_history()` / `fetch_history_range()` / `fetch_latest_price()` via yfinance. `PriceBuffer` keeps a 60-min rolling buffer in memory. |
| `main.py` | Live polling loop (`default`) and historical replay (`--backtest`). `on_event()` is the hook for future iterations. |
| `tests/test_detector.py` | 20 unit tests: threshold boundaries, edge cases (empty/single/naive index), direction labels, event attributes, backtest scan. |

### Key design decisions & defaults

| Config key | Default | Notes |
|---|---|---|
| `POLL_INTERVAL_SECONDS` | `60` | 1-minute poll; yfinance 1-min bars are the finest live granularity for NSE indices. |
| `WINDOW_MINUTES` | `5` | 5-minute rolling window to measure % change. |
| `ANOMALY_THRESHOLD_PERCENT` | `0.5` | 0.5% move in 5 minutes. Adjust to taste. |
| `WATCH_SYMBOLS` | `^NSEI,^BSESN` | Nifty 50 + BSE Sensex yfinance tickers. |

### Quick start

```bash
# 1. Copy and configure
cp .env.example .env           # edit .env with your API keys

# 2. Install dependencies (into existing venv)
pip install -r requirements.txt

# 3. Run all unit tests (offline, no internet needed)
python -m pytest tests/ -v

# 4a. Replay today's 1-min history — compact summary
python main.py --backtest

# 4b. Replay a specific past trading day (must be within last 7 days)
python main.py --backtest --date 2026-10-07

# 4c. Detection-only (no NewsAPI / Gemini calls — useful for threshold tuning)
python main.py --detect-only

# 4d. Replay with full article detail (snippet + timestamp + URL)
python main.py --backtest --verbose

# 4e. Start live monitoring (Ctrl-C to stop)
python main.py
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| _(none)_ | — | Live monitoring mode |
| `--backtest` | — | Replay 1-min history (today or `--date`) end-to-end |
| `--date YYYY-MM-DD` | today | Replay a specific past trading day. Must be within last 7 days (yfinance 1-min limit); fails with a clear message if out of range. |
| `--detect-only` | off | Backtest without any NewsAPI or Gemini calls — prints event counts only. Useful for threshold tuning without burning API quota. |
| `--verbose` | off | Print full article detail per news match (snippet, timestamp, URL). Without it, one compact line per article is shown. Full detail is always written to `market_anomaly.log` at DEBUG level. |

### How to validate detection logic offline

The `--backtest` mode downloads 1-minute OHLCV bars for each symbol
and slides the detection window across the full day.  Any flagged events are
printed to the console **and** written to `market_anomaly.log`.

You can also construct a synthetic price series in Python and call
`scan_series_for_events()` directly:

```python
import pandas as pd
from datetime import datetime, timezone, timedelta
from detector import scan_series_for_events

# Simulate a 2% spike at minute 10
base = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
ts = [base + timedelta(minutes=i) for i in range(20)]
prices = [21000.0] * 10 + [21420.0] * 10   # +2% jump

series = pd.Series(prices, index=pd.DatetimeIndex(ts))
events = scan_series_for_events(series, symbol="^NSEI", window_minutes=5, threshold_pct=0.5)
for ev in events:
    print(ev)
```

---

### Open questions / refinements before Iteration 2

1. **Threshold tuning** — 0.5% in 5 minutes is a starting point. Indian indices
   can move 0.3–0.5% on routine block trades; you may want to start at 1.0%
   and lower it only after observing the noise level with real data.

2. **yfinance rate limits** — Polling every 60 s should be well within yfinance's
   unofficial limits, but if you see `HTTPError 429` during live mode, increase
   `POLL_INTERVAL_SECONDS` to 120–300 s.

3. **Market-hours awareness** — The current live loop runs 24/7.  Outside NSE
   trading hours (09:15–15:30 IST, Mon–Fri) yfinance returns the last bar, so
   you'll see no meaningful movement.  A market-hours guard can be added to
   Iteration 2.

4. **yfinance 1-min granularity window** — yfinance only returns 1-min bars
   for the **last 7 days**; for longer back-tests you'll need 5-min or 1-day
   intervals.

---

## Iteration 2 — News Ingestion ✅

### What was built

| Module | Responsibility |
|---|---|
| `relevance_filter.py` | Keyword-based relevance scoring (3-tier weighted keywords). No I/O. Fully unit-testable. |
| `news_fetcher.py` | `NewsSource` ABC, `NewsAPISource` (newsapi.org), `NewsIngester` orchestrator. Pluggable: add new sources with `register_source()`. |
| `tests/test_news_ingestion.py` | 39 offline unit tests: scoring, threshold boundaries, mock ingester, raw article parser. |
| `main.py` (updated) | `on_event()` now calls `_fetch_and_log_news()`. Stage placeholders for Iterations 3 & 4 documented. |

### New config keys

| Config key | Default | Notes |
|---|---|---|
| `NEWSAPI_KEY` | _(required)_ | Get a free key at [newsapi.org/register](https://newsapi.org/register) |
| `NEWS_TIME_WINDOW_MINUTES` | `15` | Articles published within ±15 min of the event |
| `NEWS_MAX_ARTICLES` | `20` | Max articles fetched per source (free tier cap: 100) |
| `NEWS_RELEVANCE_MIN_SCORE` | `1.0` | Minimum keyword-weight score. 1.0 = any market term; 3.0 = must name Nifty/Sensex/NSE/BSE |

### Keyword scoring tiers

| Tier | Weight | Examples |
|---|---|---|
| 1 | 3.0 | Nifty, Sensex, NSE, BSE, Dalal Street |
| 2 | 2.0 | RBI, SEBI, FII, repo rate, Union Budget, Rupee, Midcap |
| 3 | 1.0 | rally, crash, correction, volatile, equity, F&O, IPO |

Scores accumulate across multiple keyword hits. Symbol-specific keywords (e.g. "Nifty" for `^NSEI`) receive a bonus to rank index-specific articles higher.

### How to add a new news source

```python
# In news_fetcher.py (or a new file), subclass NewsSource:
class MoneycontrolRSSSource(NewsSource):
    name = "MoneycontrolRSS"

    def fetch(self, query, from_time, to_time, max_articles):
        # parse RSS feed, return list[NewsArticle]
        ...

# In build_default_ingester():
ingester.register_source(MoneycontrolRSSSource())
```

### How to test news ingestion without an API key

```bash
# All unit tests run offline (mocked sources)
python -m pytest tests/ -v
```

### NewsAPI free-tier limitations (flagged explicitly)

- Articles are delayed by **~1 hour** on the Developer plan — so a live event at 10:00 IST may not appear in NewsAPI results until ~11:00.
- **100 requests/day** rate limit. At 60 s polling and events on both symbols, budget carefully.
- Date range filter has **minute-level precision** only.

For a production system, a paid plan or a direct RSS feed (Moneycontrol, ET Markets) would be preferred. For coursework validation, the developer plan is sufficient.

---

## Iteration 3 — Causal Analysis ✅

### What was built

| Module | Responsibility |
|---|---|
| `causal_prefilter.py` | Offline causal ranking (5 factors: recency decay, keyword density, source tier, India specificity, relevance pass-through). No I/O. |
| `causal_analyzer.py` | Pluggable `LLMClient` ABC, `GeminiClient` implementation, prompt construction, JSON parsing with validation, fallback builder. |
| `tests/test_causal_iteration3.py` | 37 offline unit tests covering all scoring factors and analyzer behaviour. |
| `main.py` (updated) | `on_event()` now runs the full pipeline: Detection → News → Pre-filter → LLM → log result. |

### New config keys

| Config key | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | _(required for LLM)_ | Google AI Studio key — [aistudio.google.com](https://aistudio.google.com) |
| `LLM_PROVIDER` | `gemini` | Currently only `gemini` supported |
| `LLM_MODEL` | `gemini-2.5-flash` | Any Gemini model accessible on your key |
| `CAUSAL_TOP_N_ARTICLES` | `3` | How many top causal articles to send to the LLM |
| `LLM_TIMEOUT_SECONDS` | `20` | Per-call LLM timeout |

### LLM fallback behaviour

If no articles are found, the LLM is **not called** — a "no relevant news" result is returned immediately.  If the LLM call fails (non-retryable error) or times out, a rule-based fallback using the top headline is returned. The pipeline never crashes due to an LLM failure.

### LLM retry policy

| Condition | Behaviour |
|---|---|
| 429 / 5xx / network / timeout | Retry with exponential backoff (2 s, 4 s, 8 s …) |
| 400 / 401 / 403 / 404 | Fail immediately — no retry |
| No articles | Skip LLM entirely — return no-news fallback |

---

## Fixes applied (post Iteration 3)

| # | File | Fix |
|---|---|---|
| 1 | `main.py`, `price_monitor.py` | `--date YYYY-MM-DD` backtest flag: replay a specific past trading day. Validates the 7-day yfinance window; fails with a clear message if date is out of range or in the future. |
| 2 | `main.py` | `--detect-only` flag: price detection only — no NewsAPI or Gemini calls. Prints event counts. |
| 3 | `causal_analyzer.py` | Empty article list now short-circuits before any LLM call, returning an immediate "no relevant news" fallback. |
| 4 | `causal_analyzer.py` | `_is_retryable()` helper: exponential backoff on 429/5xx/network/timeout; no retry on 400/401/403/404. |
| 5 | `news_fetcher.py` | Query deduplication: `index_name` (Nifty/Sensex) is no longer prepended to the base query when it already appears in it. |

All fixes are covered by offline tests in `tests/test_fixes.py` (33 tests).

---

## Iteration 4 — Notification (planned)

*Will add: delivery channel (e.g. desktop notification, Telegram, email).*

