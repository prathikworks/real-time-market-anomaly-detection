"""
relevance_filter.py — Keyword-based relevance scoring for news articles.

Design
──────
Relevance is computed as a weighted keyword hit count against the article's
title + description.  The scoring is intentionally simple and transparent
so it can be tuned or swapped for an embedding/ML approach in future
without touching any other module.

Scoring weights
───────────────
  Tier 1 (weight 3): Direct index names — "Nifty", "Sensex", "NSE", "BSE"
  Tier 2 (weight 2): Macro / market movers — RBI, SEBI, FII, budget, etc.
  Tier 3 (weight 1): General market terms — rally, crash, volatile, etc.

An article must accumulate a score >= min_score to pass the filter.
The default min_score (config.NEWS_RELEVANCE_MIN_SCORE = 1) is intentionally
low — it means at least ONE Tier-1 match needed — to avoid discarding
legitimate but tersely-written headlines.

This module has zero I/O and no external dependencies beyond the stdlib,
so it is fully unit-testable offline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular import at runtime; NewsArticle is a dataclass with no methods
    from news_fetcher import NewsArticle

logger = logging.getLogger(__name__)


# ── Keyword tables ────────────────────────────────────────────────────────────

# Each entry is (pattern, weight).
# Patterns are matched case-insensitively against: title + " " + description
# Use word-boundary anchors (\b) where the keyword could appear as a substring.

_TIER1_KEYWORDS: list[tuple[str, float]] = [
    (r"\bNifty\b",       3.0),
    (r"\bSensex\b",      3.0),
    (r"\bNSE\b",         3.0),
    (r"\bBSE\b",         3.0),
    (r"\bNifty ?50\b",   3.0),
    (r"\bBSE ?200\b",    3.0),
    (r"\bDalal Street\b", 3.0),
]

_TIER2_KEYWORDS: list[tuple[str, float]] = [
    (r"\bRBI\b",                         2.0),
    (r"\bSEBI\b",                        2.0),
    (r"\bFII\b",                         2.0),
    (r"\bDII\b",                         2.0),
    (r"\bforeign institutional\b",       2.0),
    (r"\bdomestic institutional\b",      2.0),
    (r"\brepo rate\b",                   2.0),
    (r"\bmonetary policy\b",             2.0),
    (r"\bUnion Budget\b",                2.0),
    (r"\bGDP\b",                         2.0),
    (r"\bInflation\b",                   2.0),
    (r"\bRupee\b",                       2.0),
    (r"\bINR\b",                         2.0),
    (r"\bcurrent account\b",             2.0),
    (r"\btrade deficit\b",               2.0),
    (r"\bfiscal deficit\b",              2.0),
    (r"\bIndian (stock|equity|market)\b", 2.0),
    (r"\bD-Street\b",                    2.0),
    (r"\bBombay Stock Exchange\b",       2.0),
    (r"\bNational Stock Exchange\b",     2.0),
    (r"\bMidcap\b",                      2.0),
    (r"\bSmallcap\b",                    2.0),
    (r"\bLargecap\b",                    2.0),
]

_TIER3_KEYWORDS: list[tuple[str, float]] = [
    (r"\bIndian (economy|government|finance)\b", 1.0),
    (r"\bstock market\b",                        1.0),
    (r"\bequity market\b",                       1.0),
    (r"\bbull(ish)?\b",                          1.0),
    (r"\bbear(ish)?\b",                          1.0),
    (r"\brally\b",                               1.0),
    (r"\bsell[- ]?off\b",                        1.0),
    (r"\bcrash\b",                               1.0),
    (r"\bcorrection\b",                          1.0),
    (r"\bvolatil(e|ity)\b",                      1.0),
    (r"\bmarket cap\b",                          1.0),
    (r"\bIPO\b",                                 1.0),
    (r"\bshares?\b",                             1.0),
    (r"\bequit(y|ies)\b",                        1.0),
    (r"\bsector\b",                              1.0),
    (r"\bindex\b",                               1.0),
    (r"\bindices\b",                             1.0),
    (r"\bDerivatives\b",                         1.0),
    (r"\bF&O\b",                                 1.0),
    (r"\bfutures\b",                             1.0),
    (r"\boptions\b",                             1.0),
]

# Symbol → additional Tier-1 keywords that are specific to that index
_SYMBOL_EXTRA_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "^NSEI": [
        (r"\bNifty\b",    3.0),
        (r"\bNSE\b",      3.0),
    ],
    "^BSESN": [
        (r"\bSensex\b",   3.0),
        (r"\bBSE\b",      3.0),
    ],
}

# Pre-compile all patterns for speed
_ALL_PATTERNS: list[tuple[re.Pattern, float]] = [
    (re.compile(p, re.IGNORECASE), w)
    for p, w in (_TIER1_KEYWORDS + _TIER2_KEYWORDS + _TIER3_KEYWORDS)
]

_SYMBOL_PATTERNS: dict[str, list[tuple[re.Pattern, float]]] = {
    sym: [(re.compile(p, re.IGNORECASE), w) for p, w in kws]
    for sym, kws in _SYMBOL_EXTRA_KEYWORDS.items()
}


# ── Scoring function ──────────────────────────────────────────────────────────


def score_article(article: "NewsArticle", symbol: str = "") -> float:
    """
    Compute a relevance score for *article* relative to *symbol*.

    Parameters
    ----------
    article : NewsArticle
        The article to score.
    symbol : str
        Optional yfinance ticker (used to apply symbol-specific extra weights).

    Returns
    -------
    float — cumulative keyword weight (0.0 = no match).
    """
    text = f"{article.title} {article.description}"
    total = 0.0

    for pattern, weight in _ALL_PATTERNS:
        if pattern.search(text):
            total += weight

    # Apply symbol-specific bonus patterns (may double-count — intentional,
    # articles naming the exact index are more relevant)
    for pattern, weight in _SYMBOL_PATTERNS.get(symbol, []):
        if pattern.search(text):
            total += weight

    return total


def is_relevant(article: "NewsArticle", symbol: str = "", min_score: float = 1.0) -> bool:
    """Return True if *article* meets the minimum relevance threshold."""
    return score_article(article, symbol) >= min_score


# ── Filter class ──────────────────────────────────────────────────────────────


class RelevanceFilter:
    """
    Applies scoring to a list of NewsArticles and returns only those that
    meet the minimum score threshold, with the score attached.

    The class wrapper (rather than bare functions) makes it easy to inject
    a mock in unit tests, or swap to an ML-based scorer later.
    """

    def filter_and_score(
        self,
        articles: list["NewsArticle"],
        symbol: str = "",
        min_score: float = 1.0,
    ) -> list["NewsArticle"]:
        """
        Score every article in *articles* and return those with score >= min_score.

        The returned articles are new instances with ``relevance_score`` set.

        Parameters
        ----------
        articles : list[NewsArticle]
        symbol : str
            yfinance ticker for symbol-specific keyword bonuses.
        min_score : float
            Minimum score to pass. Articles below this are silently discarded.

        Returns
        -------
        List of scored NewsArticles (same order as input, but filtered).
        """
        scored: list["NewsArticle"] = []
        for art in articles:
            s = score_article(art, symbol)
            if s >= min_score:
                # NewsArticle is frozen; use dataclasses.replace to set score
                scored.append(replace(art, relevance_score=s))
            else:
                logger.debug("Discarded (score=%.1f): %s", s, art.title[:80])

        return scored
