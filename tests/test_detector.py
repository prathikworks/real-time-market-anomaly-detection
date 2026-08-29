"""
tests/test_detector.py — Unit tests for the anomaly detection logic.

All tests are fully offline: they construct synthetic price series directly
as pd.Series objects so no network or API access is needed.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from detector import MovementEvent, check_window, scan_series_for_events

# ── Helpers ───────────────────────────────────────────────────────────────────

UTC = timezone.utc
BASE_TIME = datetime(2024, 1, 15, 9, 30, 0, tzinfo=UTC)
SYMBOL = "TEST_INDEX"


def make_series(
    prices: list[float],
    start: datetime = BASE_TIME,
    freq_seconds: int = 60,
) -> pd.Series:
    """
    Build a tz-aware DatetimeIndex price series from a list of prices.
    Timestamps are spaced ``freq_seconds`` seconds apart starting at ``start``.
    """
    # start is already UTC-aware; pd.Timestamp() preserves tzinfo automatically.
    timestamps = [
        pd.Timestamp(start + timedelta(seconds=i * freq_seconds))
        for i in range(len(prices))
    ]
    return pd.Series(prices, index=pd.DatetimeIndex(timestamps), name=SYMBOL)


# ── check_window: basic threshold logic ──────────────────────────────────────


class TestCheckWindowThreshold:
    """Tests that the threshold comparison is correct for various % changes."""

    def test_no_event_when_change_below_threshold(self):
        """A 0.3% change should NOT fire when threshold is 0.5%."""
        prices = [100.0, 100.1, 100.2, 100.3]  # +0.3%
        series = make_series(prices)
        now = BASE_TIME + timedelta(minutes=3)
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5, now=now)
        assert result is None

    def test_event_when_change_exactly_at_threshold(self):
        """A change exactly equal to the threshold SHOULD fire (>=)."""
        prices = [100.0, 100.25, 100.5]  # +0.5%
        series = make_series(prices)
        now = BASE_TIME + timedelta(minutes=2)
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5, now=now)
        assert result is not None
        assert abs(result.pct_change - 0.5) < 1e-6

    def test_event_when_change_above_threshold(self):
        """A 1.2% spike above threshold should fire."""
        prices = [21000.0, 21100.0, 21252.0]  # ~1.2%
        series = make_series(prices)
        now = BASE_TIME + timedelta(minutes=2)
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5, now=now)
        assert result is not None
        assert result.pct_change > 0

    def test_downward_spike_detected(self):
        """A sharp fall should also be detected (threshold applies to |pct_change|)."""
        prices = [21000.0, 20900.0, 20790.0]  # ~-1.0%
        series = make_series(prices)
        now = BASE_TIME + timedelta(minutes=2)
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5, now=now)
        assert result is not None
        assert result.pct_change < 0
        assert result.direction == "DOWN"

    def test_upward_spike_direction_label(self):
        """Upward spike should have direction == 'UP'."""
        prices = [100.0, 101.0]  # +1%
        series = make_series(prices)
        now = BASE_TIME + timedelta(minutes=1)
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5, now=now)
        assert result is not None
        assert result.direction == "UP"

    def test_no_event_with_flat_prices(self):
        """A completely flat series should never fire."""
        prices = [22000.0] * 10
        series = make_series(prices)
        now = BASE_TIME + timedelta(minutes=9)
        result = check_window(series, symbol=SYMBOL, window_minutes=10, threshold_pct=0.1, now=now)
        assert result is None


# ── check_window: edge cases ──────────────────────────────────────────────────


class TestCheckWindowEdgeCases:

    def test_empty_series_returns_none(self):
        series = pd.Series(dtype=float)
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5)
        assert result is None

    def test_single_data_point_returns_none(self):
        """Need at least 2 points to compute a % change."""
        series = make_series([21000.0])
        now = BASE_TIME
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5, now=now)
        assert result is None

    def test_window_excludes_old_data(self):
        """
        Points outside the window should be ignored.
        Put a huge spike at t=0, flat data within window → no event.
        """
        # Spike at t=0 (+10%), but within the 5-min window prices are flat
        old_prices = [100.0, 110.0]          # t=0, t=1 min — outside 5-min window
        new_prices = [110.0, 110.0, 110.0]   # t=6, t=7, t=8 min — inside window

        old_ts = [
            pd.Timestamp(BASE_TIME),
            pd.Timestamp(BASE_TIME + timedelta(minutes=1)),
        ]
        new_ts = [
            pd.Timestamp(BASE_TIME + timedelta(minutes=6)),
            pd.Timestamp(BASE_TIME + timedelta(minutes=7)),
            pd.Timestamp(BASE_TIME + timedelta(minutes=8)),
        ]
        series = pd.Series(
            old_prices + new_prices,
            index=pd.DatetimeIndex(old_ts + new_ts),
        )
        now = BASE_TIME + timedelta(minutes=8)
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5, now=now)
        # Within the 5-min window the price is flat at 110 — no event
        assert result is None

    def test_series_with_naive_datetimeindex_gets_utc_localized(self):
        """Naive timestamps should be treated as UTC without crashing."""
        ts = [datetime(2024, 1, 15, 9, i, 0) for i in range(5)]  # naive
        prices = [100.0, 100.2, 100.4, 100.6, 101.0]
        series = pd.Series(prices, index=pd.DatetimeIndex(ts))
        now = datetime(2024, 1, 15, 9, 4, 0, tzinfo=UTC)
        # Should not raise; +1% spike
        result = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5, now=now)
        assert result is not None

    def test_non_datetimeindex_raises_typeerror(self):
        series = pd.Series([100.0, 101.0], index=[0, 1])
        with pytest.raises(TypeError):
            check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.5)


# ── MovementEvent properties ──────────────────────────────────────────────────


class TestMovementEvent:

    def _make_event(self, pct: float) -> MovementEvent:
        now = BASE_TIME + timedelta(minutes=5)
        series = make_series([100.0, 100 + pct])
        return check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=0.1, now=now)

    def test_pct_change_accuracy(self):
        event = self._make_event(1.0)
        assert event is not None
        assert abs(event.pct_change - 1.0) < 0.01

    def test_str_representation_contains_symbol(self):
        event = self._make_event(1.0)
        assert SYMBOL in str(event)

    def test_str_representation_contains_direction(self):
        up_event = self._make_event(1.0)
        dn_event = self._make_event(-1.0)
        assert "UP" in str(up_event)
        assert "DOWN" in str(dn_event)

    def test_threshold_stored_on_event(self):
        now = BASE_TIME + timedelta(minutes=5)
        series = make_series([100.0, 101.5])
        event = check_window(series, symbol=SYMBOL, window_minutes=5, threshold_pct=1.0, now=now)
        assert event is not None
        assert event.threshold_used == 1.0


# ── scan_series_for_events ────────────────────────────────────────────────────


class TestScanSeriesForEvents:

    def test_finds_single_spike(self):
        """A series with one clear spike should produce at least one event."""
        # Build: flat for 10 min, then +2% spike, then flat again
        flat_before = [100.0] * 10
        spike = [100.0, 102.0, 102.0, 102.0, 102.0]  # sharp jump
        flat_after = [102.0] * 5
        prices = flat_before + spike + flat_after
        series = make_series(prices)

        events = scan_series_for_events(
            series,
            symbol=SYMBOL,
            window_minutes=3,
            threshold_pct=1.5,
            step_minutes=1,
        )
        assert len(events) > 0, "Expected at least one event for a 2% spike"

    def test_returns_empty_for_quiet_market(self):
        """A completely flat series should yield no events."""
        series = make_series([22000.0] * 30)
        events = scan_series_for_events(
            series,
            symbol=SYMBOL,
            window_minutes=5,
            threshold_pct=0.3,
            step_minutes=1,
        )
        assert events == []

    def test_empty_series_returns_empty_list(self):
        series = pd.Series(dtype=float)
        events = scan_series_for_events(
            series,
            symbol=SYMBOL,
            window_minutes=5,
            threshold_pct=0.5,
        )
        assert events == []

    def test_high_threshold_suppresses_small_moves(self):
        """A 0.5% move should not appear when threshold is 1.0%."""
        prices = [100.0] * 5 + [100.5] * 5  # +0.5% step
        series = make_series(prices)
        events = scan_series_for_events(
            series,
            symbol=SYMBOL,
            window_minutes=5,
            threshold_pct=1.0,
            step_minutes=1,
        )
        assert events == []

    def test_all_events_reference_correct_symbol(self):
        """Every returned event should carry the symbol we passed in."""
        prices = [100.0] * 5 + [102.5] * 5  # +2.5%
        series = make_series(prices)
        events = scan_series_for_events(
            series,
            symbol="MY_INDEX",
            window_minutes=3,
            threshold_pct=1.0,
            step_minutes=1,
        )
        for ev in events:
            assert ev.symbol == "MY_INDEX"
