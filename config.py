"""
config.py — Central configuration loader for all iterations.

Reads values from the .env file (or environment variables) and exposes
typed, validated constants.  All other modules import from here — no
hardcoded values anywhere else.
"""

import os
from dotenv import load_dotenv

# Load .env into the environment (does nothing if .env doesn't exist)
load_dotenv()


def _get_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Config error: {key}={raw!r} is not a valid float.")


def _get_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Config error: {key}={raw!r} is not a valid integer.")


def _get_list(key: str, default: list[str]) -> list[str]:
    raw = os.getenv(key)
    if raw is None:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


# ── Price Monitor ──────────────────────────────────────────────────────────────

#: How often (seconds) the live monitor polls for new prices.
POLL_INTERVAL_SECONDS: int = _get_int("POLL_INTERVAL_SECONDS", 60)

#: Rolling window (minutes) used to compute % change for anomaly detection.
WINDOW_MINUTES: int = _get_int("WINDOW_MINUTES", 5)

#: Alert fires when |% change over window| >= this value (percentage points).
ANOMALY_THRESHOLD_PERCENT: float = _get_float("ANOMALY_THRESHOLD_PERCENT", 0.5)

#: yfinance ticker symbols to monitor.
#: ^NSEI = Nifty 50,  ^BSESN = BSE Sensex
WATCH_SYMBOLS: list[str] = _get_list("WATCH_SYMBOLS", ["^NSEI", "^BSESN"])

# ── News Ingestion (Iteration 2) ───────────────────────────────────────────────

#: Minutes around a movement event to search for news articles.
NEWS_TIME_WINDOW_MINUTES: int = _get_int("NEWS_TIME_WINDOW_MINUTES", 15)

#: Maximum articles to request per source per event.
NEWS_MAX_ARTICLES: int = _get_int("NEWS_MAX_ARTICLES", 20)

#: Minimum relevance score an article must reach to pass the filter.
#: Score is a cumulative keyword-weight sum (see relevance_filter.py).
#: 1.0 = at least one Tier-3 keyword hit; 3.0 = at least one Tier-1 (index name).
NEWS_RELEVANCE_MIN_SCORE: float = _get_float("NEWS_RELEVANCE_MIN_SCORE", 1.0)

NEWS_API_KEY: str | None = os.getenv("NEWSAPI_KEY")

# ── Causal Analysis (Iteration 3) ─────────────────────────────────────────────

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# ── Notification (Iteration 4) ────────────────────────────────────────────────

NOTIFICATION_CHANNEL: str = os.getenv("NOTIFICATION_CHANNEL", "console")
