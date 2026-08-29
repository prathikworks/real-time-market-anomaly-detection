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

## Iteration 2 — News Ingestion (coming next)

*Will add: RSS/API-based news retrieval triggered on a `MovementEvent`, keyword
filtering, graceful failure handling.*

## Iteration 3 — Causal Analysis (planned)

*Will add: LLM prompt engineering + configurable API call.*

## Iteration 4 — Notification (planned)

*Will add: end-to-end pipeline wiring + delivery channel.*
