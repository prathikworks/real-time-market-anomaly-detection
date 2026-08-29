"""
detector.py — Pure, stateless anomaly-detection logic.

This module contains no I/O and no external dependencies beyond the
standard library and pandas.  It can therefore be fully unit-tested
offline without a network connection or a live data source.

Core concept
────────────
Given a time-series of (timestamp, price) observations for a single
index, a "movement event" is flagged when the percentage change
between the oldest and newest observations in a rolling window
exceeds a configurable threshold:

    pct_change = (price_now - price_window_start) / price_window_start * 100

If |pct_change| >= threshold → MovementEvent is returned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MovementEvent:
    """Represents a detected price-anomaly event."""

    symbol: str
    detected_at: datetime          # wall-clock time the event was flagged
    window_start: datetime         # oldest price in the window
    window_end: datetime           # newest price in the window
    price_start: float             # price at window_start
    price_end: float               # price at window_end
    pct_change: float              # signed % change (negative = drop)
    window_minutes: int            # actual elapsed minutes
    threshold_used: float          # threshold that was applied

    @property
    def direction(self) -> str:
        return "UP" if self.pct_change >= 0 else "DOWN"

    def __str__(self) -> str:
        return (
            f"[ANOMALY] {self.symbol} {self.direction} {self.pct_change:+.3f}% "
            f"over {self.window_minutes}min "
            f"(from {self.price_start:.2f} -> {self.price_end:.2f}) "
            f"at {self.detected_at.strftime('%Y-%m-%d %H:%M:%S %Z')}"
        )


# ── Core Detection Function ───────────────────────────────────────────────────


def check_window(
    series: pd.Series,
    *,
    symbol: str,
    window_minutes: int,
    threshold_pct: float,
    now: datetime | None = None,
) -> MovementEvent | None:
    """
    Inspect the most-recent ``window_minutes`` of price data and return a
    :class:`MovementEvent` if the threshold is breached, else ``None``.

    Parameters
    ----------
    series:
        A ``pd.Series`` with a ``DatetimeIndex`` (timezone-aware) and float
        price values, sorted oldest→newest.
    symbol:
        Human-readable name for the index (used in log messages / event).
    window_minutes:
        How far back (in minutes) from the latest data point to look.
    threshold_pct:
        Minimum absolute % change to flag as an anomaly.
    now:
        Override for "current time" — useful for testing.  When ``None``
        the latest timestamp in *series* is used as the reference point.

    Returns
    -------
    MovementEvent | None
    """
    if series.empty:
        logger.debug("%s: empty series — skipping.", symbol)
        return None

    # Ensure index is a DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")

    # Make sure timestamps are tz-aware (use UTC if naive)
    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("UTC")

    series = series.sort_index()

    reference_time: datetime = now if now is not None else series.index[-1].to_pydatetime()
    window_start_time: datetime = reference_time - timedelta(minutes=window_minutes)

    window_data = series[series.index >= window_start_time]

    if len(window_data) < 2:
        logger.debug(
            "%s: only %d data point(s) in window — need at least 2.",
            symbol,
            len(window_data),
        )
        return None

    price_start = float(window_data.iloc[0])
    price_end = float(window_data.iloc[-1])

    if price_start == 0:
        logger.warning("%s: price_start is 0 — cannot compute % change.", symbol)
        return None

    pct_change = (price_end - price_start) / price_start * 100
    actual_minutes = int(
        (window_data.index[-1] - window_data.index[0]).total_seconds() / 60
    )

    logger.debug(
        "%s: window [%s → %s]  %.2f → %.2f  (%+.3f%%)",
        symbol,
        window_data.index[0].strftime("%H:%M:%S"),
        window_data.index[-1].strftime("%H:%M:%S"),
        price_start,
        price_end,
        pct_change,
    )

    if abs(pct_change) >= threshold_pct:
        event = MovementEvent(
            symbol=symbol,
            detected_at=reference_time,
            window_start=window_data.index[0].to_pydatetime(),
            window_end=window_data.index[-1].to_pydatetime(),
            price_start=price_start,
            price_end=price_end,
            pct_change=pct_change,
            window_minutes=actual_minutes,
            threshold_used=threshold_pct,
        )
        logger.info(str(event))
        return event

    return None


def scan_series_for_events(
    series: pd.Series,
    *,
    symbol: str,
    window_minutes: int,
    threshold_pct: float,
    step_minutes: int = 1,
) -> list[MovementEvent]:
    """
    Slide a rolling window across *series* and collect **all** anomaly events.

    Useful for back-testing / validating detection logic against historical data.

    Parameters
    ----------
    series:
        Full historical price series (DatetimeIndex, sorted oldest→newest).
    symbol:
        Index name.
    window_minutes:
        Window length in minutes.
    threshold_pct:
        Alert threshold (absolute %).
    step_minutes:
        How many minutes to advance the window on each step (default 1 min).
        Smaller = finer scan; larger = faster but may miss short spikes.

    Returns
    -------
    List of :class:`MovementEvent` objects, one per breached window.
    """
    if series.empty:
        return []

    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("UTC")

    series = series.sort_index()

    events: list[MovementEvent] = []
    start = series.index[0].to_pydatetime() + timedelta(minutes=window_minutes)
    end = series.index[-1].to_pydatetime()

    current = start
    while current <= end:
        event = check_window(
            series,
            symbol=symbol,
            window_minutes=window_minutes,
            threshold_pct=threshold_pct,
            now=current,
        )
        if event is not None:
            events.append(event)
        current += timedelta(minutes=step_minutes)

    return events
