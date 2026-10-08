"""
price_monitor.py — Live & historical price fetching via yfinance.

Responsibilities
────────────────
1. Fetch a recent slice of price history for each watched symbol
   (used to seed the rolling window on startup and for back-testing).
2. Fetch the latest tick (used by the live polling loop in main.py).
3. Maintain an in-memory rolling buffer of recent prices so the
   detector always has enough data to compute the window.

Design notes
────────────
- yfinance is used exclusively — it's free, covers ^NSEI and ^BSESN,
  and requires no API key.
- The module is intentionally I/O-only; detection logic lives in
  detector.py so each layer can be tested in isolation.
- During Indian market hours yfinance's 1-minute bars are live; outside
  hours you'll get the last available bar (normal for an index).
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Iterator

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)

# How many minutes of history to seed the buffer with at startup.
# Must be > WINDOW_MINUTES so the detector has something to work with immediately.
_SEED_MINUTES: int = max(config.WINDOW_MINUTES * 3, 30)


# ── Fetching helpers ──────────────────────────────────────────────────────────


def fetch_history(
    symbol: str,
    period: str = "1d",
    interval: str = "1m",
) -> pd.Series:
    """
    Download 1-minute OHLCV bars for *symbol* and return the Close price series.

    Parameters
    ----------
    symbol : str
        yfinance ticker, e.g. ``"^NSEI"`` or ``"^BSESN"``.
    period : str
        yfinance period string (``"1d"``, ``"5d"``, …).
    interval : str
        Bar interval (``"1m"``, ``"2m"``, ``"5m"``, …).

    Returns
    -------
    pd.Series with DatetimeIndex (UTC-aware), sorted oldest→newest.
    Raises on network failure so the caller can handle gracefully.
    """
    logger.debug("Fetching history: symbol=%s period=%s interval=%s", symbol, period, interval)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {symbol!r} (period={period})")
    series = df["Close"].dropna()
    series.index = series.index.tz_convert("UTC")
    series = series.sort_index()
    logger.info("Fetched %d bars for %s (latest: %.2f)", len(series), symbol, series.iloc[-1])
    return series


def fetch_history_range(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str = "1m",
) -> pd.Series:
    """
    Download 1-minute bars for *symbol* between *start* and *end* (both UTC).

    Used by the ``--date`` backtest option to replay a specific past trading day.
    yfinance supports ``start``/``end`` with ``interval="1m"`` for up to 7 days ago.

    Returns
    -------
    pd.Series with DatetimeIndex (UTC-aware), sorted oldest→newest.
    May be empty if the market was closed or the date is outside the 7-day window.
    Raises on network failure.
    """
    logger.debug(
        "Fetching history range: symbol=%s start=%s end=%s interval=%s",
        symbol, start.isoformat(), end.isoformat(), interval,
    )
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
    if df.empty:
        return pd.Series(dtype=float)
    series = df["Close"].dropna()
    series.index = series.index.tz_convert("UTC")
    series = series.sort_index()
    logger.info(
        "Fetched %d bars for %s in range [%s, %s]",
        len(series), symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    return series


def fetch_latest_price(symbol: str) -> tuple[datetime, float]:
    """
    Fetch the single most-recent close price for *symbol*.

    Returns
    -------
    (timestamp_utc, price) tuple.
    """
    series = fetch_history(symbol, period="1d", interval="1m")
    ts = series.index[-1].to_pydatetime()
    price = float(series.iloc[-1])
    return ts, price


# ── Rolling Price Buffer ──────────────────────────────────────────────────────


class PriceBuffer:
    """
    In-memory FIFO buffer of (timestamp, price) observations.

    Keeps only the most recent ``keep_minutes`` minutes of data to
    avoid unbounded memory growth during long monitoring sessions.
    """

    def __init__(self, symbol: str, keep_minutes: int = 60) -> None:
        self.symbol = symbol
        self.keep_minutes = keep_minutes
        # deque of (pd.Timestamp, float) pairs, oldest first
        self._data: deque[tuple[pd.Timestamp, float]] = deque()

    def seed(self, series: pd.Series) -> None:
        """Bulk-load an existing price series into the buffer."""
        for ts, price in series.items():
            self._data.append((ts, float(price)))
        self._evict()
        logger.debug("%s buffer seeded with %d points.", self.symbol, len(self._data))

    def push(self, ts: datetime | pd.Timestamp, price: float) -> None:
        """Add a single new observation."""
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts, tz="UTC") if ts.tzinfo else pd.Timestamp(ts).tz_localize("UTC")
        self._data.append((ts, price))
        self._evict()

    def to_series(self) -> pd.Series:
        """Return a ``pd.Series`` snapshot suitable for passing to the detector."""
        if not self._data:
            return pd.Series(dtype=float)
        timestamps, prices = zip(*self._data)
        return pd.Series(prices, index=pd.DatetimeIndex(timestamps), name=self.symbol)

    def latest(self) -> tuple[pd.Timestamp, float] | None:
        """Return the most recent (timestamp, price) or ``None`` if empty."""
        return self._data[-1] if self._data else None

    def _evict(self) -> None:
        """Drop observations older than keep_minutes from the front."""
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=self.keep_minutes)
        while self._data and self._data[0][0] < cutoff:
            self._data.popleft()

    def __len__(self) -> int:
        return len(self._data)
