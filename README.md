# Real-Time Market Anomaly Detection & News Attribution System

An evolutionary-model project that monitors Nifty and Sensex indices for
sudden price movements, identifies the probable cause from financial news,
and delivers a plain-English explanation to the user.

---

## Project Structure

```
real-time-market-anomaly-detection/
├── config.py            # Centralised config loader (reads .env)
├── detector.py          # Pure anomaly-detection logic (I/O-free, fully testable)
├── price_monitor.py     # yfinance price fetcher + rolling PriceBuffer
├── main.py              # Entry point (live monitor + backtest mode)
├── requirements.txt     # Runtime dependencies
├── .env.example         # Copy → .env and fill in your values
├── tests/
│   ├── __init__.py
│   └── test_detector.py # 20 offline unit tests
└── README.md
```

---

## Iteration 1 — Price Movement Detection ✅

### What was built

| Module | Responsibility |
|---|---|
| `config.py` | Loads all config from `.env` and exposes typed constants. |
| `detector.py` | `check_window()` — tests one snapshot for an anomaly. `scan_series_for_events()` — slides the window across historical data. No I/O. |
| `price_monitor.py` | `fetch_history()` / `fetch_latest_price()` via yfinance. `PriceBuffer` keeps a 60-min rolling buffer in memory. |
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
cp .env.example .env           # edit .env if you want non-default thresholds

# 2. Install dependencies (into existing venv)
pip install -r requirements.txt

# 3. Run unit tests (offline, no internet needed)
python -m pytest tests/ -v

# 4a. Replay today's 1-min history to validate detection
python main.py --backtest

# 4b. Start live monitoring (Ctrl-C to stop)
python main.py
```

### How to validate detection logic offline

The `--backtest` mode downloads today's 1-minute OHLCV bars for each symbol
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
# All 59 unit tests run offline (mocked sources)
python -m pytest tests/ -v
```

### NewsAPI free-tier limitations (flagged explicitly)

- Articles are delayed by **~1 hour** on the Developer plan — so a live event at 10:00 IST may not appear in NewsAPI results until ~11:00.
- **100 requests/day** rate limit. At 60 s polling and events on both symbols, budget carefully.
- Date range filter has **minute-level precision** only.

For a production system, a paid plan or a direct RSS feed (Moneycontrol, ET Markets) would be preferred. For coursework validation, the developer plan is sufficient.

---

### Open questions / refinements before Iteration 3

1. **NewsAPI delay** — Free tier articles are ~1-hour delayed. For backtest validation this means: run the backtest with yesterday's data and check if articles timestamped around yesterday's events are returned. Live validation is harder without a paid plan.
2. **Relevance threshold tuning** — `NEWS_RELEVANCE_MIN_SCORE=1.0` is intentionally permissive. If you're seeing too much noise (generic "Indian economy" articles), raise it to `3.0` to require an index name in the headline.
3. **Query breadth** — The current query is broad ("Nifty OR Sensex OR NSE OR BSE...") to cast a wide net before filtering. If you hit the 100-req/day cap quickly, narrow the query via `NEWS_MAX_ARTICLES`.
4. **`[Removed]` articles** — NewsAPI marks deleted articles with `[Removed]`; these are discarded. Some relevant articles may be removed by publishers; nothing to do here.

---

## Iteration 3 — Causal Analysis (planned)

*Will add: LLM prompt engineering + configurable API call.*

## Iteration 4 — Notification (planned)

*Will add: end-to-end pipeline wiring + delivery channel.*
