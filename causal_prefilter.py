"""
causal_prefilter.py — Lightweight offline pre-filter for causal analysis (Iteration 3).

Purpose
───────
After the relevance filter in Iteration 2 removes clearly unrelated articles,
this module re-ranks the survivors for *causal probability* — how likely each
article is to be the CAUSE of the price movement, not just topically related.

It is intentionally lightweight and offline:
  - No network calls, no ML models, no heavy dependencies.
  - Uses a multi-factor scoring formula with configurable weights.

Scoring factors (all normalised to [0, 1] before weighting)
────────────────────────────────────────────────────────────
1. Recency / proximity  – articles published closest to the event time score
   higher.  Uses exponential decay: score = exp(-dt_minutes / DECAY_HALFLIFE).

2. Causal keyword density – count of "trigger" words (rate, RBI, policy,
   decision, report, result, crash, halt, circuit, ban, sanction …) in
   title + description, normalised by text length.

3. Source tier weight – tier-1 sources (Business Standard, Economic Times,
   Moneycontrol, Bloomberg, Reuters, NDTV) get a bonus; generic/aggregator
   sources get a penalty.

4. Indian-market specificity bonus – articles that name Indian institutions
   or markets specifically (RBI, SEBI, NSE, BSE, Sensex, Nifty, Rupee) get
   a bonus; US-only stories (S&P 500, Fed, NASDAQ, Dow Jones) are penalised.

5. Relevance score pass-through – the relevance_score from Iteration 2's
   filter is used as a base so the ordering from that layer is not discarded.

All factors are combined into a single causal_score:
    causal_score = (
        w_recency     * recency_score
      + w_keywords    * keyword_score
      + w_source      * source_score
      + w_india       * india_score
      + w_relevance   * normalised_relevance_score
    )

The top CAUSAL_TOP_N_ARTICLES by causal_score are returned.
"""

from __future__ import annotations

import math
import re
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from news_fetcher import NewsArticle

logger = logging.getLogger(__name__)


# ── Causal keyword tables ─────────────────────────────────────────────────────

_CAUSAL_KEYWORDS: list[str] = [
    # Market events
    "rate", "hike", "cut", "policy", "decision", "announcement", "report",
    "result", "earnings", "gdp", "inflation", "data", "signal", "signal",
    # Crisis / halt words
    "crash", "circuit breaker", "halt", "suspend", "ban", "sanction",
    "default", "bankruptcy", "downgrade", "upgrade",
    # Macro triggers
    "rbi", "sebi", "fed", "fomc", "budget", "fiscal", "monetary",
    "repo", "cpi", "wpi", "current account", "trade deficit",
    # Corporate actions
    "merger", "acquisition", "ipo", "stake", "buyback", "dividend",
    "quarterly", "profit", "loss", "revenue", "guidance",
    # Geo/political triggers
    "war", "geopolitical", "oil", "crude", "opec", "sanction",
]
_CAUSAL_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _CAUSAL_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# ── Source tier weights ───────────────────────────────────────────────────────

# Tier 1: reputable, India-focused financial sources
_TIER1_SOURCES = {
    "business standard", "economic times", "moneycontrol", "livemint", "mint",
    "bloomberg", "reuters", "cnbc", "ndtv profit", "et markets", "bse india",
    "nse india", "the hindu businessline", "financial express",
}
# Tier 3: low-credibility or generic aggregators
_TIER3_SOURCES = {
    "benzinga", "seekingalpha", "reddit", "twitter", "x.com",
    "investing.com", "stockanalysis",
}

# ── India vs non-India patterns ───────────────────────────────────────────────

_INDIA_POSITIVE = re.compile(
    r"\b(RBI|SEBI|NSE|BSE|Sensex|Nifty|Rupee|INR|Dalal Street|"
    r"Indian (market|economy|stock|equit)|HDFC|Reliance|Infosys|TCS|Adani|"
    r"Tata|Wipro|Bajaj|ICICI|SBI|Kotak)\b",
    re.IGNORECASE,
)
_US_ONLY = re.compile(
    r"\b(S&P 500|Nasdaq|Dow Jones|NYSE|Wall Street|"
    r"Federal Reserve(?! India)|US (GDP|CPI|inflation|jobs report)|"
    r"Jackson Hole|FOMC meeting)\b",
    re.IGNORECASE,
)
_INDIA_NEGATIVE_PURE_US = re.compile(
    # If article has lots of US markers and zero India markers it's US-only
    r"\bUS (stocks?|market|economy|equities)\b",
    re.IGNORECASE,
)

# ── Scoring weights (must sum to 1.0) ────────────────────────────────────────

_W_RECENCY   = 0.35
_W_KEYWORDS  = 0.25
_W_SOURCE    = 0.15
_W_INDIA     = 0.15
_W_RELEVANCE = 0.10

assert abs(_W_RECENCY + _W_KEYWORDS + _W_SOURCE + _W_INDIA + _W_RELEVANCE - 1.0) < 1e-9

# Exponential decay half-life in minutes for recency scoring
_DECAY_HALFLIFE_MIN = 30.0


# ── Public API ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoredArticle:
    """A NewsArticle decorated with its causal pre-filter score."""

    article: "NewsArticle"
    causal_score: float          # combined score in [0, ~1]
    recency_score: float
    keyword_score: float
    source_score: float
    india_score: float
    relevance_score_norm: float  # the Iteration-2 relevance score, normalised

    def __str__(self) -> str:
        return (
            f"[causal={self.causal_score:.3f}] "
            f"[rec={self.recency_score:.2f} kw={self.keyword_score:.2f} "
            f"src={self.source_score:.2f} in={self.india_score:.2f}] "
            f"{self.article.title[:80]}"
        )


def rank_for_causality(
    articles: list["NewsArticle"],
    event_time: datetime,
    top_n: int = 3,
    decay_halflife: float = _DECAY_HALFLIFE_MIN,
) -> list[ScoredArticle]:
    """
    Score and rank *articles* by their probability of being the causal driver
    of the price movement that occurred at *event_time*.

    Parameters
    ----------
    articles:
        Articles that have already passed the relevance filter (Iteration 2).
    event_time:
        UTC-aware datetime of the detected movement event.
    top_n:
        Maximum number of articles to return.
    decay_halflife:
        Minutes for exponential recency decay (default 30 min).

    Returns
    -------
    List of ScoredArticle, sorted by causal_score descending, length <= top_n.
    """
    if not articles:
        return []

    # Pre-compute the max relevance score for normalisation
    max_rel = max(a.relevance_score for a in articles) or 1.0

    scored: list[ScoredArticle] = []
    for art in articles:
        rec   = _recency_score(art, event_time, decay_halflife)
        kw    = _keyword_score(art)
        src   = _source_score(art)
        india = _india_score(art)
        rel_n = art.relevance_score / max_rel

        total = (
            _W_RECENCY   * rec
            + _W_KEYWORDS  * kw
            + _W_SOURCE    * src
            + _W_INDIA     * india
            + _W_RELEVANCE * rel_n
        )
        scored.append(ScoredArticle(
            article=art,
            causal_score=round(total, 4),
            recency_score=round(rec, 4),
            keyword_score=round(kw, 4),
            source_score=round(src, 4),
            india_score=round(india, 4),
            relevance_score_norm=round(rel_n, 4),
        ))
        logger.debug(str(scored[-1]))

    scored.sort(key=lambda s: s.causal_score, reverse=True)
    top = scored[:top_n]
    logger.info("Causal pre-filter: %d -> %d articles (top_n=%d)", len(articles), len(top), top_n)
    return top


# ── Factor implementations (all return values in [0, 1]) ─────────────────────


def _recency_score(art: "NewsArticle", event_time: datetime, halflife: float) -> float:
    """
    Exponential decay: score = exp(-ln2 * dt / halflife).
    Articles published AT event_time score 1.0; articles published
    halflife minutes away score 0.5.
    """
    pub = art.published_at
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)
    dt_minutes = abs((event_time - pub).total_seconds()) / 60.0
    return math.exp(-math.log(2) * dt_minutes / halflife)


def _keyword_score(art: "NewsArticle") -> float:
    """
    Density of causal keywords in title + description.
    Normalised to [0, 1] with a saturation cap at 5 matches.
    """
    text = f"{art.title} {art.description}"
    matches = len(_CAUSAL_PATTERN.findall(text))
    return min(matches / 5.0, 1.0)


def _source_score(art: "NewsArticle") -> float:
    """
    Tier-1 source → 1.0, tier-3 → 0.2, all others → 0.6.
    Matching is done against the lowercased source_name.
    """
    name_lower = art.source_name.lower()
    for t1 in _TIER1_SOURCES:
        if t1 in name_lower:
            return 1.0
    for t3 in _TIER3_SOURCES:
        if t3 in name_lower:
            return 0.2
    return 0.6


def _india_score(art: "NewsArticle") -> float:
    """
    Score in [0, 1] for Indian-market specificity.
    Positive: India keywords present.
    Negative: US-only keywords present and India keywords absent.
    """
    text = f"{art.title} {art.description}"
    india_hits = len(_INDIA_POSITIVE.findall(text))
    us_hits    = len(_US_ONLY.findall(text))

    if india_hits > 0 and us_hits == 0:
        return 1.0                             # purely Indian
    if india_hits > 0 and us_hits > 0:
        return 0.6                             # mixed
    if india_hits == 0 and us_hits > 0:
        return 0.1                             # US-only penalty
    return 0.5                                 # neutral / no signal
