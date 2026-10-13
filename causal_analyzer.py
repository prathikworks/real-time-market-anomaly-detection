"""
causal_analyzer.py — Deterministic rule-based causal context for price events.

This module replaces the previous LLM-powered (Gemini) implementation with a
fully offline, deterministic scoring system.  No API key, no network call, no
model is required.

Architecture
────────────
                 ┌──────────────────────────────────────────────────────┐
                 │           RuleBasedAnalyzer                          │
                 │  (scores ScoredArticles, picks top-N, emits result)  │
                 └──────────────────────────────────────────────────────┘

Public API  (unchanged from LLM version)
────────────────────────────────────────
    result = RuleBasedAnalyzer().analyse(event, top_articles)
    print(result.explanation)   # framed as "possible context", never causal claim
    print(result.confidence)    # "high" | "medium" | "low" | "none"
    print(result.source_refs)   # titles of top articles
    print(result.is_fallback)   # always False — this IS the primary path

Scoring signals (all used in causal_prefilter.py already; re-read here for
combined confidence estimation)
────────────────────────────────
 1. Broadcast scope  - broad-market articles score higher than single-stock.
 2. Keyword density  - count of unique causal-keyword categories matched.
 3. Recency          - articles published before the event rank above post-event.
 4. Source tier      - tier-1 India finance sources get a bonus.
 5. Relevance score  - pass-through from the existing relevance_filter layer.

Language policy
───────────────
  Output is always framed as "relevant news" or "possible context" — never
  a confirmed explanation of the market movement.  This prevents fabrication.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from causal_prefilter import ScoredArticle
    from detector import MovementEvent

logger = logging.getLogger(__name__)


# ── Result dataclass (same shape as before) ───────────────────────────────────


@dataclass
class CausalResult:
    """
    Output of a causal context analysis run.

    Fields
    ------
    explanation:
        Plain-English context summary.  Always framed as
        "possible context" — never a fabricated causal claim.
    confidence:
        "high" | "medium" | "low" | "none"  (derived from article scores).
    source_refs:
        Titles of the top articles used to build the explanation.
    is_fallback:
        Always False for RuleBasedAnalyzer (rule-based IS the primary path).
    raw_response:
        Empty string (kept for API compatibility with previous LLM version).
    """
    explanation: str
    confidence: str                        # high | medium | low | none
    source_refs: list[str] = field(default_factory=list)
    is_fallback: bool = False
    raw_response: str = ""

    def __str__(self) -> str:
        tag = "[FALLBACK] " if self.is_fallback else ""
        refs = ("; ".join(self.source_refs[:2])) if self.source_refs else "n/a"
        return (
            f"{tag}RELEVANT NEWS CONTEXT [{self.confidence.upper()}]\n"
            f"  {self.explanation}\n"
            f"  Sources: {refs}"
        )


# ── Macro-signal patterns (used for confidence estimation) ────────────────────

# Each pattern represents a distinct macro-signal category.
# Matching is case-insensitive.  Only the FIRST match per category is counted
# so a single keyword cannot inflate the score.
_MACRO_SIGNALS: list[re.Pattern] = [
    re.compile(r"\b(FII|FPI|foreign.?institutional|foreign.?portfolio)\b", re.IGNORECASE),
    re.compile(r"\b(crude oil|Brent|WTI|OPEC)\b", re.IGNORECASE),
    re.compile(r"\b(rupee|INR)\b", re.IGNORECASE),
    re.compile(r"\b(bond yield|10.year|g-sec|government securities?)\b", re.IGNORECASE),
    re.compile(r"\b(RBI|repo rate|monetary policy|rate cut|rate hike)\b", re.IGNORECASE),
    re.compile(r"\b(inflation|CPI|WPI)\b", re.IGNORECASE),
    re.compile(r"\b(geopolit|war|sanctions?|global market)\b", re.IGNORECASE),
    re.compile(r"\b(GDP|growth|recession|slowdown)\b", re.IGNORECASE),
    re.compile(r"\b(Sensex|Nifty|BSE|NSE|Dalal Street)\b", re.IGNORECASE),
]

_CAUSAL_CONTEXT_SIGNAL = re.compile(
    r"\b(FII|FPI|foreign.?institutional|foreign.?portfolio|crude oil|Brent|WTI|OPEC|"
    r"rupee|INR|US Treasury yields?|bond yields?|West Asia|Middle East|"
    r"geopolit\w*|RBI|repo rate|monetary policy|inflation|CPI|WPI|"
    r"sanctions?|global sell.?off)\b",
    re.IGNORECASE,
)


def _count_macro_signals(text: str) -> int:
    """Count how many distinct macro-signal categories appear in *text*."""
    return sum(1 for pat in _MACRO_SIGNALS if pat.search(text))


# ── Confidence estimator ──────────────────────────────────────────────────────


def _estimate_confidence(
    top_articles: "list[ScoredArticle]",
    event_time: datetime,
) -> str:
    """
    Derive a confidence level from the top articles' collective signal strength.

    Rules
    -----
    - "none"   : no articles, or best causal_score < 0.45
    - "low"    : best score in [0.45, 0.58) or top article is post-event
    - "medium" : best score in [0.58, 0.72) with a pre-event article at top
    - "high"   : best score >= 0.72 AND top article is pre-event
                 AND >= 3 distinct macro signal categories match
    """
    if not top_articles:
        return "none"

    best       = top_articles[0]
    best_score = best.causal_score
    best_text  = f"{best.article.title} {best.article.description}"
    macro_hits = _count_macro_signals(best_text)

    ev_time = event_time
    if ev_time.tzinfo is None:
        ev_time = ev_time.replace(tzinfo=timezone.utc)

    pub = best.article.published_at
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    is_pre_event = pub <= ev_time

    if best_score < 0.45:
        return "none"
    if best_score >= 0.72 and is_pre_event and macro_hits >= 3:
        return "high"
    if best_score >= 0.58 and is_pre_event:
        return "medium"
    return "low"


# ── Timing label ──────────────────────────────────────────────────────────────


def _timing_label(art_time: datetime, event_time: datetime) -> str:
    """Return 'pre-event' or 'post-event' relative to the movement."""
    pub = art_time
    ev  = event_time
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    if ev.tzinfo is None:
        ev = ev.replace(tzinfo=timezone.utc)
    return "pre-event" if pub <= ev else "post-event"


def pre_event_articles(articles: list, event_time: datetime) -> list:
    """Return only articles published at or before *event_time*.

    The helper accepts either NewsArticle instances or ScoredArticle wrappers
    so the same rule is applied before causal ranking and again defensively
    inside the analyzer.
    """
    event_utc = event_time if event_time.tzinfo else event_time.replace(tzinfo=timezone.utc)
    eligible: list = []
    for item in articles:
        article = getattr(item, "article", item)
        published = article.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if published <= event_utc:
            eligible.append(item)
    return eligible


def select_causal_articles(
    articles: "list[ScoredArticle]",
    event_time: datetime,
) -> "list[ScoredArticle]":
    """Keep only pre-event, broad-market articles with a concrete driver."""
    return [
        item for item in pre_event_articles(articles, event_time)
        if item.broadcast_score >= 0.7
        and _CAUSAL_CONTEXT_SIGNAL.search(
            f"{item.article.title} {item.article.description}"
        )
    ]


# ── Main analyzer ─────────────────────────────────────────────────────────────


class RuleBasedAnalyzer:
    """
    Deterministic, offline causal-context analyzer.

    Uses the causal_prefilter scores (already computed) plus additional
    macro-signal counting to derive a confidence level and produce a
    plain-English context summary.  No LLM, no network call.

    Parameters
    ----------
    top_display : int
        How many top articles to surface in the result (default 5).
    """

    def __init__(self, top_display: int = 5) -> None:
        self._top_display = top_display

    def analyse(
        self,
        event: "MovementEvent",
        top_articles: "list[ScoredArticle]",
    ) -> CausalResult:
        """
        Build a CausalResult from the pre-scored article list.

        Never fabricates a causal explanation.  Only articles available at or
        before the event are eligible for possible-cause analysis.
        """
        top_articles = select_causal_articles(top_articles, event.detected_at)
        if not top_articles:
            logger.info("Rule-based analyzer: no articles — returning no-news result.")
            return CausalResult(
                explanation="No strongly relevant pre-event news found; cause undetermined.",
                confidence="none",
                source_refs=[],
                is_fallback=False,
            )

        confidence = _estimate_confidence(top_articles, event.detected_at)
        logger.info(
            "Rule-based causal context: confidence=%s  top_article=%s",
            confidence,
            top_articles[0].article.title[:80],
        )

        display = top_articles[:self._top_display]
        lines: list[str] = []
        for i, sa in enumerate(display, 1):
            art    = sa.article
            timing = _timing_label(art.published_at, event.detected_at)
            pub    = art.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            lines.append(
                f"[{i}] ({timing}) {art.title} "
                f"— {art.source_name}, {pub.strftime('%H:%M UTC')}"
            )

        explanation = (
            f"Possible context for the {event.direction.lower()} move in {event.symbol} "
            f"({event.pct_change:+.2f}%). Top relevant news around "
            f"{event.detected_at.strftime('%H:%M UTC')}:\n"
            + "\n".join(lines)
        )

        source_refs = [sa.article.title for sa in display]

        return CausalResult(
            explanation=explanation,
            confidence=confidence,
            source_refs=source_refs,
            is_fallback=False,
        )


# Alias for callers that previously imported CausalAnalyzer
CausalAnalyzer = RuleBasedAnalyzer


# ── Factory ───────────────────────────────────────────────────────────────────


def build_default_analyzer() -> RuleBasedAnalyzer:
    """
    Build a RuleBasedAnalyzer using current config.
    Never raises — no API key or network required.
    """
    import config as _config
    top_n = getattr(_config, "CAUSAL_TOP_N_DISPLAY", 5)
    return RuleBasedAnalyzer(top_display=top_n)
