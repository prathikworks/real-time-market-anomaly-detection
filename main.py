"""
main.py — Entry point for Iteration 1: live price monitoring loop.

Run modes
─────────
  python main.py               # live monitoring (uses config from .env)
  python main.py --backtest    # replay today's 1-min history for each symbol
  python main.py --help        # show options

The script polls each watched symbol every POLL_INTERVAL_SECONDS seconds,
feeds new prices into a PriceBuffer, runs the detector, and logs any
MovementEvents found.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone

import config
import price_monitor as pm
from detector import check_window, scan_series_for_events, MovementEvent

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("market_anomaly.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


# ── Event handling ────────────────────────────────────────────────────────────


def on_event(event: MovementEvent) -> None:
    """
    Called whenever a movement event is detected.

    Iteration 1: just log it clearly.
    Iterations 2-4 will add news ingestion → analysis → notification here.
    """
    separator = "=" * 72
    logger.warning(separator)
    logger.warning("MOVEMENT EVENT DETECTED")
    logger.warning("  Symbol      : %s", event.symbol)
    logger.warning("  Direction   : %s", event.direction)
    logger.warning("  Change      : %+.3f%%", event.pct_change)
    logger.warning("  Window      : %d min  (%s -> %s)", event.window_minutes,
                   event.window_start.strftime("%H:%M:%S UTC"),
                   event.window_end.strftime("%H:%M:%S UTC"))
    logger.warning("  Price range : %.2f -> %.2f", event.price_start, event.price_end)
    logger.warning("  Threshold   : %.2f%%", event.threshold_used)
    logger.warning("  Detected at : %s", event.detected_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
    logger.warning(separator)


# ── Backtest mode ─────────────────────────────────────────────────────────────


def run_backtest() -> None:
    """
    Download today's 1-minute data for each symbol and slide the detection
    window across the full history.  Prints a summary of all events found.
    Useful for validating that the detector fires correctly on real data.
    """
    logger.info("--- BACKTEST MODE ---------------------------------------------------")
    logger.info("Symbols      : %s", ", ".join(config.WATCH_SYMBOLS))
    logger.info("Window       : %d min", config.WINDOW_MINUTES)
    logger.info("Threshold    : %.2f%%", config.ANOMALY_THRESHOLD_PERCENT)

    total_events = 0

    for symbol in config.WATCH_SYMBOLS:
        logger.info("\nFetching history for %s ...", symbol)
        try:
            series = pm.fetch_history(symbol, period="1d", interval="1m")
        except Exception as exc:
            logger.error("  Could not fetch %s: %s", symbol, exc)
            continue

        logger.info("  %d bars  (%s -> %s)",
                    len(series),
                    series.index[0].strftime("%Y-%m-%d %H:%M UTC"),
                    series.index[-1].strftime("%Y-%m-%d %H:%M UTC"))

        events = scan_series_for_events(
            series,
            symbol=symbol,
            window_minutes=config.WINDOW_MINUTES,
            threshold_pct=config.ANOMALY_THRESHOLD_PERCENT,
            step_minutes=1,
        )

        logger.info("  Events found: %d", len(events))
        for ev in events:
            on_event(ev)

        total_events += len(events)

    logger.info("\nBacktest complete. Total events: %d", total_events)


# ── Live monitoring loop ──────────────────────────────────────────────────────


def run_live() -> None:
    """
    Seed each symbol's buffer with recent history, then poll for new
    prices every POLL_INTERVAL_SECONDS.  Runs until Ctrl-C.
    """
    logger.info("--- LIVE MONITORING MODE --------------------------------------------")
    logger.info("Symbols       : %s", ", ".join(config.WATCH_SYMBOLS))
    logger.info("Poll interval : %d sec", config.POLL_INTERVAL_SECONDS)
    logger.info("Window        : %d min", config.WINDOW_MINUTES)
    logger.info("Threshold     : %.2f%%", config.ANOMALY_THRESHOLD_PERCENT)
    logger.info("Press Ctrl-C to stop.\n")

    # Initialise a PriceBuffer per symbol, seeded with recent history
    buffers: dict[str, pm.PriceBuffer] = {}
    for symbol in config.WATCH_SYMBOLS:
        buf = pm.PriceBuffer(symbol, keep_minutes=60)
        try:
            history = pm.fetch_history(symbol, period="1d", interval="1m")
            buf.seed(history)
            logger.info("Seeded %s buffer with %d bars.", symbol, len(buf))
        except Exception as exc:
            logger.warning("Could not seed %s: %s — starting empty.", symbol, exc)
        buffers[symbol] = buf

    # Polling loop
    try:
        while True:
            for symbol, buf in buffers.items():
                try:
                    ts, price = pm.fetch_latest_price(symbol)
                    buf.push(ts, price)
                    logger.debug("%s  %s  %.2f", symbol, ts.strftime("%H:%M:%S"), price)

                    event = check_window(
                        buf.to_series(),
                        symbol=symbol,
                        window_minutes=config.WINDOW_MINUTES,
                        threshold_pct=config.ANOMALY_THRESHOLD_PERCENT,
                    )
                    if event:
                        on_event(event)

                except Exception as exc:
                    logger.error("Error processing %s: %s", symbol, exc)

            logger.info("Next poll in %d s …", config.POLL_INTERVAL_SECONDS)
            time.sleep(config.POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-Time Market Anomaly Detection — Iteration 1"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Replay today's 1-min history for detection validation (no live polling).",
    )
    args = parser.parse_args()

    if args.backtest:
        run_backtest()
    else:
        run_live()


if __name__ == "__main__":
    main()
