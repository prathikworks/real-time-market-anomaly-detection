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

Bug-fix history
───────────────
v1.1  Fixed check_window(): the window must be sliced to [now-W, now]
      from both ends, not just from the left.  The original code kept
      the full tail of the series as "window_data", so iloc[-1] was
      always the last bar of the day instead of the bar closest to
      `now` — producing reported windows of hundreds of minutes.

      Fixed scan_series_for_events(): added cooldown suppression so a
      sustained trend produces one event, not one per step-minute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    window_minutes: int            # actual elapsed minutes (should be <= WINDOW_MINUTES)
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
    Inspect the ``window_minutes``-wide slice of price data ending at ``now``
    and return a :class:`MovementEvent` if the threshold is breached.

    Parameters
    ----------
    series:
        A ``pd.Series`` with a ``DatetimeIndex`` (timezone-aware) and float
        price values.  May be longer than ``window_minutes``; this function
        always slices it to exactly ``[now - window_minutes, now]``.
    symbol:
        Human-readable name for the index (used in log messages / event).
    window_minutes:
        Width of the detection window in minutes.
    threshold_pct:
        Minimum absolute % change to flag as an anomaly.
    now:
        Override for "current time" — useful for backtesting.  When ``None``
        the latest timestamp in *series* is used as the reference point.

    Returns
    -------
    MovementEvent | None

    Notes
    -----
    **Key implementation constraint**: the window is sliced from *both* ends:

        window_data = series[(series.index >= window_start_time)
                             & (series.index <= reference_time)]

    Without the upper-bound filter the last bar in the series (often the
    end of the trading day) bleeds into every earlier window, causing
    the reported duration to be hundreds of minutes and the % change to
    reflect the full-day move rather than the 5-minute move.
    """
    if series.empty:
        logger.debug("%s: empty series -- skipping.", symbol)
        return None

    # Ensure index is a DatetimeIndex
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")

    # Make sure timestamps are tz-aware (use UTC if naive)
    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("UTC")

    series = series.sort_index()

    # ── Reference point and window bounds ────────────────────────────────────
    reference_time: datetime = now if now is not None else series.index[-1].to_pydatetime()
    # Normalise to UTC-aware if caller passed a naive datetime
    if reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    window_start_time: datetime = reference_time - timedelta(minutes=window_minutes)

    # CRITICAL: filter BOTH ends so the window is exactly [start, now].
    # Without the upper-bound (series.index <= reference_time) the slice
    # includes all future bars up to the end of the series, making
    # window_data.iloc[-1] always the last bar of the day.
    window_data = series[
        (series.index >= window_start_time) & (series.index <= reference_time)
    ]

    if len(window_data) < 2:
        logger.debug(
            "%s: only %d data point(s) in window [%s -> %s] -- need at least 2.",
            symbol,
            len(window_data),
            window_start_time.strftime("%H:%M:%S"),
            reference_time.strftime("%H:%M:%S"),
        )
        return None

    price_start = float(window_data.iloc[0])
    price_end = float(window_data.iloc[-1])

    if price_start == 0:
        logger.warning("%s: price_start is 0 -- cannot compute %% change.", symbol)
        return None

    pct_change = (price_end - price_start) / price_start * 100
    actual_minutes = int(
        (window_data.index[-1] - window_data.index[0]).total_seconds() / 60
    )

    logger.debug(
        "%s: window [%s -> %s]  %.2f -> %.2f  (%+.3f%%)",
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


# ── Backtest scan with cooldown ───────────────────────────────────────────────


def scan_series_for_events(
    series: pd.Series,
    *,
    symbol: str,
    window_minutes: int,
    threshold_pct: float,
    step_minutes: int = 1,
    cooldown_minutes: int = 15,
) -> list[MovementEvent]:
    """
    Slide a rolling window across *series* and collect anomaly events,
    applying a cooldown to avoid flooding on sustained trends.

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
    cooldown_minutes:
        After an event fires, suppress further events for this many minutes.
        Default 15.  Set to 0 to disable (not recommended for backtests).

    Returns
    -------
    List of :class:`MovementEvent` objects after deduplication.

    Design notes
    ────────────
    Cooldown is applied at the *scan* level, not inside check_window, so
    check_window remains pure and testable in isolation.  When cooldown is
    active the loop simply skips check_window entirely, avoiding both
    spurious events and wasted computation.
    """
    if series.empty:
        return []

    if series.index.tz is None:
        series = series.copy()
        series.index = series.index.tz_localize("UTC")

    series = series.sort_index()

    events: list[MovementEvent] = []
    cooldown_until: datetime | None = None  # None = no active cooldown

    start = series.index[0].to_pydatetime() + timedelta(minutes=window_minutes)
    end   = series.index[-1].to_pydatetime()

    current = start
    while current <= end:

        # ── Cooldown suppression ──────────────────────────────────────────────
        if cooldown_until is not None and current < cooldown_until:
            current += timedelta(minutes=step_minutes)
            continue

        event = check_window(
            series,
            symbol=symbol,
            window_minutes=window_minutes,
            threshold_pct=threshold_pct,
            now=current,
        )

        if event is not None:
            events.append(event)
            if cooldown_minutes > 0:
                cooldown_until = current + timedelta(minutes=cooldown_minutes)

        current += timedelta(minutes=step_minutes)

    return events
