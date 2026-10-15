"""
notifier.py - Email notification module for market anomaly events.

Sends a plain-text + HTML email via SMTP STARTTLS (stdlib only).
Configuration is read from environment variables (or a .env file when
python-dotenv is available). Credentials are NEVER hardcoded.

Public API
----------
    send_event_email(event, causal_result, articles)
        Build and send (or dry-run) an email for a MovementEvent.
        Called from main.py after the causal analysis is complete.

Environment variables
---------------------
    SMTP_HOST        SMTP server hostname          (default: smtp.gmail.com)
    SMTP_PORT        SMTP STARTTLS port            (default: 587)
    SMTP_USER        Login username / address      (required)
    SMTP_PASSWORD    Login password / app-password (required)
    EMAIL_FROM       Sender address                (default: SMTP_USER)
    EMAIL_TO         Comma-separated recipients    (required)

Dry-run mode
------------
    When dry_run=True the email is built and printed to the console; the
    raw text is also saved to logs/last_email.txt. Nothing is sent.

Failure policy
--------------
    Any exception is caught, logged to the file logger, and reported on
    the console as a single WARNING line. The main pipeline is never
    crashed.
"""

from __future__ import annotations

import logging
import os
import smtplib
import textwrap
from datetime import timezone
from email.message import EmailMessage
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from causal_analyzer import CausalResult
    from causal_prefilter import ScoredArticle
    from detector import MovementEvent

logger = logging.getLogger(__name__)

# IST offset (UTC+5:30) - no pytz required
_IST_OFFSET_SECONDS = 5 * 3600 + 30 * 60


def _to_ist(dt) -> str:
    """Return a datetime formatted in IST (UTC+05:30)."""
    from datetime import timedelta
    ist_tz = timezone(timedelta(seconds=_IST_OFFSET_SECONDS))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    ist_dt = dt.astimezone(ist_tz)
    return ist_dt.strftime("%H:%M IST")


# Config loader

def _load_config() -> dict:
    """
    Load SMTP configuration from environment variables.

    Attempts to load a .env file via python-dotenv if the library is
    available. Falls back to os.environ silently if it is not installed.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)  # do not override already-set env vars
    except ImportError:
        pass  # python-dotenv not installed; os.environ used as-is

    smtp_user = os.environ.get("SMTP_USER", "")
    return {
        "host":     os.environ.get("SMTP_HOST", "smtp.gmail.com"),
        "port":     int(os.environ.get("SMTP_PORT", "587")),
        "user":     smtp_user,
        "password": os.environ.get("SMTP_PASSWORD", ""),
        "from":     os.environ.get("EMAIL_FROM", smtp_user),
        "to":       [
            addr.strip()
            for addr in os.environ.get("EMAIL_TO", "").split(",")
            if addr.strip()
        ],
    }


# Subject builder

def _build_subject(event: "MovementEvent") -> str:
    """
    Build the email subject line.

    Example: "[Market Alert] ^BSESN UP +1.22% in 5 min"
    """
    return (
        f"[Market Alert] {event.symbol} {event.direction} "
        f"{event.pct_change:+.2f}% in {event.window_minutes} min"
    )


# Plain-text body helpers

def _articles_section_text(
    causal_result: "CausalResult",
    articles: "list[ScoredArticle]",
) -> str:
    """Plain-text probable-cause section (top 3 ranked articles)."""
    conf = causal_result.confidence.upper()
    lines = [f"Probable Cause  [confidence: {conf}]"]
    lines.append("-" * 40)

    if causal_result.confidence.lower() == "none" or not articles:
        lines.append(
            "No strongly relevant pre-event news found; cause undetermined."
        )
        return "\n".join(lines)

    top3 = articles[:3]
    for i, sa in enumerate(top3, 1):
        art = sa.article
        pub = art.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        pub_utc = pub.strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"[{i}] {art.title}")
        lines.append(f"     Source    : {art.source_name}")
        lines.append(f"     Published : {pub_utc}")
        lines.append(f"     URL       : {art.url}")
        lines.append("")

    return "\n".join(lines)


def _build_plain_text(
    event: "MovementEvent",
    causal_result: "CausalResult",
    articles: "list[ScoredArticle]",
) -> str:
    """Assemble the full plain-text email body."""
    w_start = event.window_start
    w_end   = event.window_end
    if w_start.tzinfo is None:
        w_start = w_start.replace(tzinfo=timezone.utc)
    if w_end.tzinfo is None:
        w_end = w_end.replace(tzinfo=timezone.utc)

    w_start_utc = w_start.strftime("%H:%M:%S UTC")
    w_end_utc   = w_end.strftime("%H:%M:%S UTC")
    w_start_ist = _to_ist(w_start)
    w_end_ist   = _to_ist(w_end)

    header = (
        "========================================\n"
        "MARKET ANOMALY ALERT\n"
        "========================================\n"
        f"Symbol      : {event.symbol}\n"
        f"Direction   : {event.direction}\n"
        f"Change      : {event.pct_change:+.2f}%\n"
        f"Window      : {event.window_minutes} min\n"
        f"              {w_start_utc} -> {w_end_utc}\n"
        f"              {w_start_ist} -> {w_end_ist}\n"
        f"Price Range : {event.price_start:.2f} -> {event.price_end:.2f}\n"
        f"Threshold   : {event.threshold_used:.2f}%\n"
        "----------------------------------------\n"
    )

    cause_section = _articles_section_text(causal_result, articles)

    footer = (
        "\n"
        "----------------------------------------\n"
        "Automated analysis. News correlation is not proof of causation.\n"
    )

    return header + "\n" + cause_section + footer


# HTML body builder

def _build_html(
    event: "MovementEvent",
    causal_result: "CausalResult",
    articles: "list[ScoredArticle]",
) -> str:
    """Assemble a minimal HTML alternative body."""
    from html import escape

    w_start = event.window_start
    w_end   = event.window_end
    if w_start.tzinfo is None:
        w_start = w_start.replace(tzinfo=timezone.utc)
    if w_end.tzinfo is None:
        w_end = w_end.replace(tzinfo=timezone.utc)

    w_start_utc = w_start.strftime("%H:%M:%S UTC")
    w_end_utc   = w_end.strftime("%H:%M:%S UTC")
    w_start_ist = _to_ist(w_start)
    w_end_ist   = _to_ist(w_end)

    direction_colour = "#2ecc71" if event.direction == "UP" else "#e74c3c"

    rows = [
        ("Symbol",      escape(event.symbol)),
        ("Direction",
         f'<span style="color:{direction_colour};font-weight:bold">'
         f'{escape(event.direction)}</span>'),
        ("Change",      f"{event.pct_change:+.2f}%"),
        ("Window",
         f"{event.window_minutes} min &nbsp;|&nbsp; "
         f"{escape(w_start_utc)} &rarr; {escape(w_end_utc)} &nbsp;|&nbsp; "
         f"{escape(w_start_ist)} &rarr; {escape(w_end_ist)}"),
        ("Price Range", f"{event.price_start:.2f} &rarr; {event.price_end:.2f}"),
        ("Threshold",   f"{event.threshold_used:.2f}%"),
    ]

    table_rows = "".join(
        f"<tr><td style='padding:4px 8px;color:#888'>{k}</td>"
        f"<td style='padding:4px 8px;font-weight:bold'>{v}</td></tr>"
        for k, v in rows
    )

    conf = escape(causal_result.confidence.upper())

    if causal_result.confidence.lower() == "none" or not articles:
        cause_html = (
            "<p style='color:#888'>No strongly relevant pre-event news found; "
            "cause undetermined.</p>"
        )
    else:
        article_items = []
        for i, sa in enumerate(articles[:3], 1):
            art = sa.article
            pub = art.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            pub_utc = pub.strftime("%Y-%m-%d %H:%M UTC")
            safe_title  = escape(art.title)
            safe_source = escape(art.source_name)
            safe_url    = escape(art.url)
            article_items.append(
                f"<li style='margin-bottom:10px'>"
                f"<strong>{i}. {safe_title}</strong><br>"
                f"<span style='color:#888'>{safe_source} &nbsp;|&nbsp; {pub_utc}</span><br>"
                f"<a href='{safe_url}' style='color:#3498db'>{safe_url}</a>"
                f"</li>"
            )
        cause_html = "<ol style='padding-left:18px'>" + "".join(article_items) + "</ol>"

    return (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<body style=\"font-family:Arial,sans-serif;max-width:640px;margin:auto;padding:20px\">\n"
        "  <h2 style=\"border-bottom:2px solid #333;padding-bottom:8px\">\n"
        "    &#9888; Market Anomaly Alert\n"
        "  </h2>\n"
        "  <table style=\"border-collapse:collapse;width:100%\">\n"
        f"    {table_rows}\n"
        "  </table>\n"
        "  <hr style=\"margin:16px 0\">\n"
        f"  <h3 style=\"margin-bottom:6px\">Probable Cause &nbsp;<small style=\"color:#888\">[{conf}]</small></h3>\n"
        f"  {cause_html}\n"
        "  <hr style=\"margin:16px 0\">\n"
        "  <p style=\"color:#aaa;font-size:11px\">\n"
        "    Automated analysis. News correlation is not proof of causation.\n"
        "  </p>\n"
        "</body>\n"
        "</html>\n"
    )


# EmailMessage builder

def _build_message(
    event: "MovementEvent",
    causal_result: "CausalResult",
    articles: "list[ScoredArticle]",
    cfg: dict,
) -> EmailMessage:
    """Assemble the EmailMessage object (plain + HTML alternatives)."""
    msg = EmailMessage()
    msg["Subject"] = _build_subject(event)
    msg["From"]    = cfg["from"]
    msg["To"]      = ", ".join(cfg["to"]) if cfg["to"] else "undisclosed-recipients:;"

    plain = _build_plain_text(event, causal_result, articles)
    html  = _build_html(event, causal_result, articles)

    msg.set_content(plain)
    msg.add_alternative(html, subtype="html")
    return msg


# Dry-run helper

def _dry_run_output(msg: EmailMessage) -> None:
    """Print the email to stdout and save it to logs/last_email.txt."""
    header_block = (
        f"To      : {msg['To']}\n"
        f"From    : {msg['From']}\n"
        f"Subject : {msg['Subject']}\n"
    )

    plain_body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                plain_body = part.get_payload(decode=True).decode(
                    part.get_content_charset() or "utf-8", errors="replace"
                )
                break
    else:
        plain_body = msg.get_payload(decode=True).decode("utf-8", errors="replace")

    separator = "=" * 60
    output = (
        f"\n{separator}\n"
        f"DRY-RUN EMAIL (not sent)\n"
        f"{separator}\n"
        f"{header_block}"
        f"{separator}\n"
        f"{plain_body}\n"
        f"{separator}\n"
    )

    print(output)

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    last_email_path = logs_dir / "last_email.txt"
    try:
        last_email_path.write_text(output, encoding="utf-8")
        print(f"  [dry-run] Email saved to {last_email_path}")
    except OSError as exc:
        logger.warning("Could not save dry-run email to %s: %s", last_email_path, exc)


# Public API

def send_event_email(
    event: "MovementEvent",
    causal_result: "CausalResult",
    articles: "list[ScoredArticle]",
    *,
    dry_run: bool = False,
) -> None:
    """
    Build and (optionally) send an email for a detected MovementEvent.

    Parameters
    ----------
    event:
        The MovementEvent that triggered the alert.
    causal_result:
        Output of the causal analyzer (confidence + explanation).
    articles:
        Ranked ScoredArticle list (top-3 are used in the email).
    dry_run:
        When True, build and print the email but do NOT connect to SMTP.
        The email is also saved to logs/last_email.txt.

    Failures
    --------
        Any exception is caught, logged at ERROR level, and reported to
        the console as one WARNING line. The pipeline is never crashed.
    """
    try:
        cfg = _load_config()
        msg = _build_message(event, causal_result, articles, cfg)

        if dry_run:
            _dry_run_output(msg)
            return

        # Validate required config before attempting SMTP
        if not cfg["user"]:
            raise ValueError("SMTP_USER is not set -- cannot send email.")
        if not cfg["password"]:
            raise ValueError("SMTP_PASSWORD is not set -- cannot send email.")
        if not cfg["to"]:
            raise ValueError("EMAIL_TO is not set -- no recipients configured.")

        # Send via STARTTLS (10-second timeout)
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=10) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(cfg["user"], cfg["password"])
            smtp.send_message(msg)

        logger.info(
            "Email notification sent: %s -> %s",
            _build_subject(event),
            cfg["to"],
        )

    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        logger.error("Email notification failed: %s", reason, exc_info=True)
        print(f"WARNING: email notification failed ({reason})")
