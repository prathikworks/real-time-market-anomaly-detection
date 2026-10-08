"""
main.py — Entry point: live price monitoring & backtesting pipeline.

Run modes
---------
  python main.py               # live monitoring (uses config from .env)
  python main.py --backtest    # replay today's 1-min history for each symbol
  python main.py --help        # show options

Pipeline (per MovementEvent)
----------------------------
  Iteration 1: detect price anomaly -> log
  Iteration 2: detect -> fetch & filter news -> log headlines
  Iteration 3: detect -> news -> LLM causal analysis (coming)
  Iteration 4: detect -> news -> analysis -> notification (coming)
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
from news_fetcher import NewsIngester, build_default_ingester, NewsArticle
from causal_prefilter import rank_for_causality
from causal_analyzer import CausalAnalyzer, CausalResult, build_default_analyzer

# ── Logging setup ─────────────────────────────────────────────────────────────

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)

_file_handler = logging.FileHandler("market_anomaly.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)   # full detail always in the log file

logging.basicConfig(
    level=logging.DEBUG,               # root at DEBUG so handlers can filter
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[_stream_handler, _file_handler],
)
logger = logging.getLogger("main")

# Build the news ingester once at module level (reuses HTTP session across events)
_ingester: NewsIngester = build_default_ingester()

# Build the causal analyzer once (lazy-inits LLM client on first use)
_analyzer: CausalAnalyzer = build_default_analyzer()

# Verbose mode flag — set to True by --verbose CLI flag.
# When False: one-line-per-article summary (default, screenshot-friendly).
# When True:  full detail — snippet, timestamp, URL per article.
_verbose: bool = False


# ── Event handling ────────────────────────────────────────────────────────────


def on_event(event: MovementEvent) -> None:
    """
    Central pipeline hook called on every detected MovementEvent.

    Stage 1 (Iteration 1): Log the price anomaly.
    Stage 2 (Iteration 2): Fetch & filter relevant news headlines.
    Stage 3 (Iteration 3): LLM causal analysis — TODO.
    Stage 4 (Iteration 4): Deliver notification — TODO.
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

    # ── Stage 2: News ingestion ──────────────────────────────────────────────
    articles = _fetch_and_log_news(event)

    # ── Stage 3: Causal analysis ─────────────────────────────────────────────
    top_articles = rank_for_causality(
        articles,
        event_time=event.detected_at,
        top_n=config.CAUSAL_TOP_N_ARTICLES,
    )
    result = _analyzer.analyse(event, top_articles)
    _log_causal_result(result)

    # ── Stage 4: Notification (Iteration 4 placeholder) ─────────────────────
    # notifier.send(event, result)  # TODO


def _fetch_and_log_news(event: MovementEvent) -> list[NewsArticle]:
    """
    Fetch relevant news for *event*, log matched headlines, and return the
    article list so Stage 3 (causal analysis) can consume it.

    Output mode is controlled by the module-level ``_verbose`` flag:
      - Default (INFO):  one condensed line per article — [score] Title (Source)
      - Verbose (--verbose): full detail — title + snippet + timestamp + URL

    Snippet, timestamp, and URL are always emitted at DEBUG level so they
    are captured in market_anomaly.log regardless of the console mode.
    """
    articles = _ingester.fetch_for_event(
        event_symbol=event.symbol,
        event_time=event.detected_at,
    )

    news_sep = "-" * 72
    if not articles:
        logger.info("%s", news_sep)
        logger.info("NEWS: No relevant articles found for this event.")
        logger.info("%s", news_sep)
        return []

    logger.info("%s", news_sep)
    logger.info("NEWS: %d relevant article(s) | window=+/-%d min | threshold=%.1f",
                len(articles), config.NEWS_TIME_WINDOW_MINUTES, config.NEWS_RELEVANCE_MIN_SCORE)

    for i, art in enumerate(articles, 1):
        if _verbose:
            # ── Verbose: full four-line block ──────────────────────────────
            logger.info("  [%d] [score=%.1f] %s", i, art.relevance_score, art.title)
            if art.description:
                logger.info("      %s", art.description[:120])
            logger.info("      Published: %s | Source: %s",
                        art.published_at.strftime("%Y-%m-%d %H:%M UTC"),
                        art.source_name)
            logger.info("      URL: %s", art.url)
        else:
            # ── Compact: one line, screenshot-friendly ─────────────────────
            logger.info("  [%d] [%.1f] %s  (%s)",
                        i, art.relevance_score, art.title, art.source_name)
            # Always preserve full detail at DEBUG so the log file has it
            logger.debug("      snippet  : %s", art.description[:120] if art.description else "—")
            logger.debug("      published: %s", art.published_at.strftime("%Y-%m-%d %H:%M UTC"))
            logger.debug("      url      : %s", art.url)

    logger.info("%s", news_sep)
    if not _verbose:
        logger.info("  (run with --verbose for snippet, timestamp, and URL)")
        logger.info("%s", news_sep)

    return articles


def _log_causal_result(result: CausalResult) -> None:
    """
    Log the causal analysis result to the console and log file.
    Fallback results are flagged clearly so users know the LLM was unavailable.
    """
    sep = "~" * 72
    tag = "[FALLBACK] " if result.is_fallback else ""
    logger.warning(sep)
    logger.warning("CAUSAL ANALYSIS  %s[%s]", tag, result.confidence.upper())
    logger.warning("  %s", result.explanation)
    if result.source_refs:
        refs = "; ".join(result.source_refs[:3])
        logger.warning("  Sources: %s", refs)
    logger.warning(sep)


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
            cooldown_minutes=config.EVENT_COOLDOWN_MINUTES,
        )

        logger.info("  Events found: %d (cooldown=%d min)",
                    len(events), config.EVENT_COOLDOWN_MINUTES)

        # Apply news-lookup cap to protect the NewsAPI quota
        cap = config.BACKTEST_MAX_EVENTS_PER_RUN
        events_for_news = events if cap == 0 else events[:cap]
        if cap > 0 and len(events) > cap:
            logger.warning(
                "  [CAP] %d event(s) found but only %d will trigger news lookup "
                "(BACKTEST_MAX_EVENTS_PER_RUN=%d). "
                "Lower the threshold or raise the cap to process all.",
                len(events), cap, cap,
            )

        for ev in events_for_news:
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
    global _verbose

    parser = argparse.ArgumentParser(
        description="Real-Time Market Anomaly Detection — Iterations 1, 2 & 3"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Replay today's 1-min history for detection validation (no live polling).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print full article detail (snippet, timestamp, URL) for each news match. "
            "Default is one-line-per-article summary."
        ),
    )
    args = parser.parse_args()
    _verbose = args.verbose

    if args.backtest:
        run_backtest()
    else:
        run_live()


if __name__ == "__main__":
    main()
