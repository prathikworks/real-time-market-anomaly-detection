"""
main.py — Entry point: live price monitoring & backtesting pipeline.

Run modes
---------
  python main.py               # live monitoring (uses config from .env)
  python main.py --backtest    # replay today's 1-min history for each symbol
  python main.py --help        # show options

Pipeline (per MovementEvent)
----------------------------
  1. Detect price anomaly.
  2. Fetch & filter news articles from NewsAPI within the time window.
  3. Score and rank articles with deterministic rule-based analysis.
  4. Display top ranked news as possible context (no LLM required).
  5. Notification (placeholder — extend in Stage 4).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import unicodedata
from datetime import date as _date
from datetime import datetime, timedelta, timezone

import config
import price_monitor as pm
from detector import check_window, scan_series_for_events, MovementEvent
from news_fetcher import NewsIngester, build_default_ingester, NewsArticle
from causal_prefilter import rank_for_causality
from causal_analyzer import (
    CausalAnalyzer,
    CausalResult,
    build_default_analyzer,
    pre_event_articles,
    select_causal_articles,
)
import notifier

# ── Logging setup ─────────────────────────────────────────────────────────────

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

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

# Notification flags — set by --notify / --notify-dry-run CLI flags.
# Default is OFF so existing backtests never send emails.
_notify: bool = False
_notify_dry_run: bool = False


def _console_safe_text(value: object) -> str:
    """Remove invisible control characters and replace console-unsafe glyphs."""
    text = unicodedata.normalize("NFC", str(value))
    text = "".join(char for char in text if unicodedata.category(char) != "Cf")
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        return text.encode("utf-8", errors="replace").decode("utf-8")


# ── Event handling ────────────────────────────────────────────────────────────


def on_event(event: MovementEvent) -> None:
    """
    Central pipeline hook called on every detected MovementEvent.

    Stage 1: Log the price anomaly.
    Stage 2: Fetch & filter relevant news headlines.
    Stage 3: Rule-based scoring — rank articles and display as possible context.
    Stage 4: Deliver notification (placeholder).
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

    # ── Stage 3: Rule-based ranking & context display ────────────────────────
    causal_candidates = pre_event_articles(articles, event.detected_at)
    ranked_articles = rank_for_causality(
        causal_candidates,
        event_time=event.detected_at,
        top_n=len(causal_candidates),
    )
    top_articles = select_causal_articles(
        ranked_articles, event.detected_at
    )[:config.CAUSAL_TOP_N_ARTICLES]
    result = _analyzer.analyse(event, top_articles)
    _log_causal_result(result, event, top_articles)

    # ── Stage 4: Email notification ────────────────────────────────────────
    if _notify or _notify_dry_run:
        notifier.send_event_email(
            event,
            result,
            top_articles,
            dry_run=_notify_dry_run,
        )


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
                logger.info("      %s", _console_safe_text(art.description[:120]))
            logger.info("      Published: %s | Source: %s",
                        art.published_at.strftime("%Y-%m-%d %H:%M UTC"),
                        art.source_name)
            logger.info("      URL: %s", art.url)
        else:
            # ── Compact: one line, screenshot-friendly ─────────────────────
            logger.info("  [%d] [%.1f] %s  (%s)",
                        i, art.relevance_score, art.title, art.source_name)
            # Always preserve full detail at DEBUG so the log file has it
            snippet = art.description[:120] if art.description else "—"
            logger.debug("      snippet  : %s", _console_safe_text(snippet))
            logger.debug("      published: %s", art.published_at.strftime("%Y-%m-%d %H:%M UTC"))
            logger.debug("      url      : %s", art.url)

    logger.info("%s", news_sep)
    if not _verbose:
        logger.info("  (run with --verbose for snippet, timestamp, and URL)")
        logger.info("%s", news_sep)

    return articles


def _log_causal_result(
    result: CausalResult,
    event: MovementEvent | None = None,
    top_articles: list | None = None,
) -> None:
    """
    Log the rule-based news context result to the console and log file.

    Displays up to 5 ranked articles with:
      - rank number
      - pre-event / post-event timing label
      - headline
      - source name
      - publication time
      - relevance score
      - URL

    Post-event articles are clearly distinguished so they are not
    mistaken for causal evidence.
    """
    sep = "~" * 72
    logger.warning(sep)
    logger.warning("RELEVANT NEWS CONTEXT  [%s]", result.confidence.upper())

    if result.confidence == "none" or not result.source_refs:
        logger.warning("  %s", result.explanation)
        logger.warning(sep)
        return

    # Print the header line
    from causal_analyzer import _timing_label
    from datetime import timezone
    logger.warning(
        "  Possible context for the %s move (%s). "
        "Ranked news around the event window:",
        event.direction.lower() if event else "?",
        f"{event.pct_change:+.2f}%" if event else "",
    )
    logger.warning(
        "  NOTE: Post-event articles report on the move — they are NOT evidence of its cause."
    )

    articles_to_show = top_articles or []
    for i, sa in enumerate(articles_to_show, 1):
        art    = sa.article
        ev_t   = event.detected_at if event else None
        timing = _timing_label(art.published_at, ev_t) if ev_t else "?"
        pub    = art.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)

        logger.warning("")
        logger.warning("  [%d] %s", i, art.title)
        logger.warning("       Source  : %s", art.source_name)
        logger.warning("       Time    : %s  [%s]",
                       pub.strftime("%Y-%m-%d %H:%M UTC"), timing)
        logger.warning("       Score   : %.3f  (relevance=%.1f)",
                       sa.causal_score, art.relevance_score)
        logger.warning("       URL     : %s", art.url)

    logger.warning(sep)


# ── Backtest mode ─────────────────────────────────────────────────────────────


def run_backtest(
    backtest_date: _date | None = None,
    detect_only: bool = False,
) -> None:
    """
    Slide the detection window across 1-minute history for each symbol.

    Parameters
    ----------
    backtest_date:
        A specific calendar date to replay (YYYY-MM-DD).  Must be within the
        last 7 days (yfinance 1-min data limit).  ``None`` → today.
    detect_only:
        When True, run only price detection and print event counts; skip all
        NewsAPI and Gemini calls.  Useful for threshold tuning without burning
        API quota.
    """
    today = _date.today()
    if backtest_date is None:
        backtest_date = today

    # Validate: yfinance keeps only ~7 days of 1-min history.
    age_days = (today - backtest_date).days
    if age_days < 0:
        logger.error("--date %s is in the future. Use a past date.", backtest_date)
        sys.exit(1)
    if age_days > 29:
        logger.error(
            "--date %s is %d days ago. yfinance only stores ~7 days of 1-min data. "
            "Use a more recent date or switch to a coarser interval.",
            backtest_date, age_days,
        )
        sys.exit(1)

    logger.info("--- BACKTEST MODE ---------------------------------------------------")
    logger.info("Date         : %s%s", backtest_date,
                " (today)" if backtest_date == today else "")
    logger.info("Symbols      : %s", ", ".join(config.WATCH_SYMBOLS))
    logger.info("Window       : %d min", config.WINDOW_MINUTES)
    logger.info("Threshold    : %.2f%%", config.ANOMALY_THRESHOLD_PERCENT)
    if detect_only:
        logger.info("Mode         : DETECT-ONLY (no NewsAPI calls)")

    total_events = 0

    for symbol in config.WATCH_SYMBOLS:
        logger.info("\nFetching history for %s ...", symbol)
        try:
            if backtest_date == today:
                series = pm.fetch_history(symbol, period="1d", interval="1m")
            else:
                # yfinance: use start/end for specific past day
                start_dt = datetime.combine(backtest_date,
                                            datetime.min.time()).replace(tzinfo=timezone.utc)
                end_dt   = start_dt + timedelta(days=1)
                series   = pm.fetch_history_range(
                    symbol,
                    start=start_dt,
                    end=end_dt,
                    interval="1m",
                )
        except Exception as exc:
            logger.error("  Could not fetch %s: %s", symbol, exc)
            continue

        if series.empty:
            logger.warning("  No data returned for %s on %s — "
                           "market may have been closed or date is outside "
                           "yfinance's 7-day 1-min window.", symbol, backtest_date)
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

        if detect_only:
            for ev in events:
                logger.info("    %s", ev)
            total_events += len(events)
            continue

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
    global _verbose, _notify, _notify_dry_run

    parser = argparse.ArgumentParser(
        description="Real-Time Market Anomaly Detection — Iterations 1, 2 & 3"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Replay 1-min history for detection validation (no live polling).",
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        default=None,
        help=(
            "Replay a specific trading day (default: today). "
            "Must be within the last 7 days (yfinance 1-min data limit). "
            "Example: --backtest --date 2026-10-07"
        ),
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        dest="detect_only",
        help=(
            "Backtest mode only: run price detection and print event counts "
            "without making any NewsAPI calls."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print full article detail (snippet, timestamp, URL) for each news match. "
            "Default is one-line-per-article summary."
        ),
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help=(
            "Send an email notification for each detected event. "
            "Requires SMTP_USER, SMTP_PASSWORD, and EMAIL_TO in .env or environment. "
            "Default is OFF — backtests never send emails unless this flag is passed."
        ),
    )
    parser.add_argument(
        "--notify-dry-run",
        action="store_true",
        dest="notify_dry_run",
        help=(
            "Build the email and print it to the console (and save it to "
            "logs/last_email.txt) without actually sending it. "
            "Useful for verifying email formatting without SMTP credentials."
        ),
    )
    args = parser.parse_args()
    _verbose = args.verbose
    _notify = args.notify
    _notify_dry_run = args.notify_dry_run

    if args.backtest or args.detect_only or args.date:
        # Parse --date
        backtest_date: _date | None = None
        if args.date:
            try:
                backtest_date = _date.fromisoformat(args.date)
            except ValueError:
                logger.error(
                    "Invalid --date format %r — expected YYYY-MM-DD.", args.date
                )
                sys.exit(1)
        run_backtest(backtest_date=backtest_date, detect_only=args.detect_only)
    else:
        run_live()


if __name__ == "__main__":
    main()
