"""
news_fetcher.py — News ingestion layer (Iteration 2).

Architecture
────────────
                     ┌─────────────────────────────────────────────┐
                     │              NewsIngester                    │
                     │  (orchestrates multiple sources, dedupes,   │
                     │   passes results to relevance filter)        │
                     └────────────┬────────────────────────────────┘
                                  │  calls each registered source
                     ┌────────────▼────────────┐
                     │      NewsSource (ABC)    │   ← plug in new sources here
                     └─────────┬───────────────┘
                               │
               ┌───────────────┘
               │
    ┌──────────▼──────────┐
    │   NewsAPISource      │   (newsapi.org — free tier)
    └─────────────────────┘

To add a new source (e.g. MoneycontrolRSSSource):
  1. Subclass NewsSource and implement fetch().
  2. Register it with NewsIngester.register_source().

Data model
──────────
NewsArticle — a dataclass carrying everything the downstream layers need.
All timestamps are stored as UTC-aware datetimes.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

import config
from relevance_filter import RelevanceFilter, is_relevant, score_article

logger = logging.getLogger(__name__)


# ── Data Model ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NewsArticle:
    """Canonical representation of a single news article from any source."""

    title: str
    description: str          # article snippet / lead paragraph (may be empty)
    url: str
    published_at: datetime    # UTC-aware
    source_name: str          # e.g. "NewsAPI", "MoneycontrolRSS"
    relevance_score: float = 0.0   # filled in by the relevance filter

    def __str__(self) -> str:
        ts = self.published_at.strftime("%Y-%m-%d %H:%M UTC")
        score_tag = f"[score={self.relevance_score:.1f}]" if self.relevance_score else ""
        return f"[{ts}] {score_tag} {self.title} ({self.source_name})"


# ── Abstract Base ─────────────────────────────────────────────────────────────


class NewsSource(ABC):
    """
    Abstract base class for a news data source.

    Subclass this and implement ``fetch()`` to add a new provider.
    The source is responsible only for I/O and basic parsing — relevance
    filtering happens in the NewsIngester layer.
    """

    # Human-readable name used in logging and NewsArticle.source_name
    name: str = "UnknownSource"

    @abstractmethod
    def fetch(
        self,
        query: str,
        from_time: datetime,
        to_time: datetime,
        max_articles: int,
    ) -> list[NewsArticle]:
        """
        Retrieve articles matching *query* published between *from_time* and
        *to_time* (both UTC-aware).

        Implementations MUST:
        - Return an empty list (not raise) when no articles are found.
        - Catch network / rate-limit exceptions internally, log them, and
          return an empty list so the pipeline stays alive.
        - Never return articles outside [from_time, to_time] if avoidable.
        """
        ...


# ── NewsAPI Source ─────────────────────────────────────────────────────────────


class NewsAPISource(NewsSource):
    """
    Fetches articles from https://newsapi.org (free tier).

    Free-tier limitations (flagged explicitly):
    - Results are delayed by ~1 hour on the Developer plan.
    - Only the most recent 100 results per query.
    - 100 requests/day rate limit.
    - The ``from`` / ``to`` time filter has minute-level precision.

    These constraints are acceptable for a coursework project where we're
    validating detection logic rather than running a production feed.
    """

    name = "NewsAPI"
    _BASE_URL = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "MarketAnomalyDetector/2.0"})

    def fetch(
        self,
        query: str,
        from_time: datetime,
        to_time: datetime,
        max_articles: int = 20,
    ) -> list[NewsArticle]:
        """
        Call the NewsAPI /everything endpoint and return parsed NewsArticles.

        Failures (network error, auth, rate-limit, unexpected shape) are all
        caught here — the method returns [] so the pipeline is never crashed
        by a news source failure.
        """
        params: dict[str, Any] = {
            "q": query,
            "from": from_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": to_time.strftime("%Y-%m-%dT%H:%M:%S"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(max_articles, 100),   # free tier cap
            "apiKey": self._api_key,
        }

        logger.debug(
            "NewsAPI request: query=%r from=%s to=%s",
            query,
            from_time.strftime("%H:%M UTC"),
            to_time.strftime("%H:%M UTC"),
        )

        try:
            resp = self._session.get(self._BASE_URL, params=params, timeout=10)
        except requests.exceptions.ConnectionError as exc:
            logger.warning("NewsAPI: network error — %s", exc)
            return []
        except requests.exceptions.Timeout:
            logger.warning("NewsAPI: request timed out after 10 s.")
            return []

        if resp.status_code == 401:
            logger.error("NewsAPI: invalid or missing API key (HTTP 401). "
                         "Set NEWSAPI_KEY in .env.")
            return []
        if resp.status_code == 429:
            logger.warning("NewsAPI: rate limit hit (HTTP 429). "
                           "Free tier allows 100 requests/day.")
            return []
        if resp.status_code != 200:
            logger.warning("NewsAPI: unexpected HTTP %d — %s",
                           resp.status_code, resp.text[:200])
            return []

        try:
            payload = resp.json()
        except ValueError:
            logger.warning("NewsAPI: response is not valid JSON.")
            return []

        if payload.get("status") != "ok":
            logger.warning("NewsAPI: status=%r message=%r",
                           payload.get("status"), payload.get("message"))
            return []

        articles: list[NewsArticle] = []
        for raw in payload.get("articles", []):
            article = self._parse_article(raw)
            if article is not None:
                articles.append(article)

        logger.info(
            "NewsAPI: fetched %d article(s) for query=%r",
            len(articles), query,
        )
        return articles

    # ── private ──

    def _parse_article(self, raw: dict) -> NewsArticle | None:
        """Parse one raw API dict into a NewsArticle.  Returns None on bad data."""
        try:
            pub_str = raw.get("publishedAt", "")
            # NewsAPI returns ISO-8601 with trailing 'Z'
            pub_str = pub_str.replace("Z", "+00:00")
            published_at = datetime.fromisoformat(pub_str)
            if published_at.tzinfo is None:
                published_at = published_at.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            logger.debug("NewsAPI: could not parse publishedAt=%r — skipping.", raw.get("publishedAt"))
            return None

        title = (raw.get("title") or "").strip()
        if not title or title == "[Removed]":
            return None

        publisher = (raw.get("source") or {}).get("name") or self.name
        return NewsArticle(
            title=title,
            description=(raw.get("description") or "").strip(),
            url=raw.get("url") or "",
            published_at=published_at,
            source_name=publisher.strip(),
        )


# ── Orchestrator ──────────────────────────────────────────────────────────────


class NewsIngester:
    """
    Orchestrates one or more NewsSource instances.

    Usage
    ─────
    ingester = NewsIngester()
    ingester.register_source(NewsAPISource(api_key=config.NEWS_API_KEY))
    # ingester.register_source(MoneycontrolRSSSource())  ← future

    articles = ingester.fetch_for_event(event)

    Design decisions flagged explicitly
    ────────────────────────────────────
    - The query sent to all sources is a fixed Indian-market keyword string
      plus the index name derived from the symbol.  This is intentionally
      broad so the relevance filter (not the API) does the fine-grained work.
    - Deduplication is by normalised URL so the same article from two sources
      doesn't appear twice in downstream layers.
    - If no sources are registered or all return empty, returns [] with a
      WARNING log — does not raise.
    """

    # Symbol → human name used in query construction
    _SYMBOL_NAMES: dict[str, str] = {
        "^NSEI": "Nifty",
        "^BSESN": "Sensex",
    }

    # Default broad query — cast wide; the relevance filter narrows down
    _BASE_QUERY = "Nifty OR Sensex OR NSE OR BSE OR \"stock market\" OR \"Indian market\""
    _CONTEXT_QUERIES: tuple[str, ...] = (
        "(Sensex OR Nifty OR NSE OR BSE OR \"Indian market\") AND (sell-off OR selloff OR fall OR drop OR slump OR correction)",
        "(India OR Indian) AND (\"crude oil\" OR Brent OR WTI OR OPEC)",
        "(India OR Indian) AND (FII OR FPI OR \"foreign institutional\" OR \"foreign portfolio\") AND (outflow OR selling OR sell-off)",
        "(India OR Indian) AND (\"US Treasury yield\" OR \"US Treasury yields\" OR \"bond yield\")",
        "(rupee OR INR) AND (weakness OR weakens OR fall OR depreciation) AND (India OR Indian)",
        "(\"West Asia\" OR \"Middle East\" OR geopolitical OR geopolitics) AND (India OR Indian)",
        "(auto sector OR auto stocks OR automobile sector) AND (India OR Indian)",
    )
    _BROAD_MARKET_RE = re.compile(
        r"\b(Sensex|Nifty|NSE|BSE|Dalal Street|Indian (?:market|bourses|equities|stocks)|"
        r"market (?:sell-?off|fall|drop|slump|correction|decline)|benchmark index)\b",
        re.IGNORECASE,
    )
    _TRACKING_QUERY_PARAMS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid"})

    def __init__(self) -> None:
        self._sources: list[NewsSource] = []
        self._filter = RelevanceFilter()

    def register_source(self, source: NewsSource) -> None:
        """Add a news source to the pool."""
        self._sources.append(source)
        logger.debug("Registered news source: %s", source.name)

    def fetch_for_event(
        self,
        event_symbol: str,
        event_time: datetime,
        *,
        window_minutes: int | None = None,
        max_articles: int | None = None,
        min_relevance_score: float | None = None,
    ) -> list[NewsArticle]:
        """
        Fetch and filter news articles around a movement event.

        Parameters
        ----------
        event_symbol : str
            The yfinance ticker, e.g. ``"^NSEI"``.
        event_time : datetime
            UTC-aware timestamp of the movement event.
        window_minutes : int, optional
            Override for how many minutes before/after the event to search.
            Defaults to ``config.NEWS_TIME_WINDOW_MINUTES``.
        max_articles : int, optional
            Maximum articles to request per source.
            Defaults to ``config.NEWS_MAX_ARTICLES``.
        min_relevance_score : float, optional
            Minimum relevance score to pass articles downstream.
            Defaults to ``config.NEWS_RELEVANCE_MIN_SCORE``.

        Returns
        -------
        List of relevant NewsArticles sorted by relevance score (desc),
        then by published_at (most recent first).
        """
        if not self._sources:
            logger.warning("NewsIngester: no sources registered — skipping news fetch.")
            return []

        win = window_minutes if window_minutes is not None else config.NEWS_TIME_WINDOW_MINUTES
        max_arts = max_articles if max_articles is not None else config.NEWS_MAX_ARTICLES
        min_score = min_relevance_score if min_relevance_score is not None else config.NEWS_RELEVANCE_MIN_SCORE

        from_time = event_time - timedelta(minutes=win)
        to_time = event_time + timedelta(minutes=win)

        queries = self._queries_for_symbol(event_symbol)

        logger.info(
            "News fetch: symbol=%s  window=+/-%d min  [%s -> %s]",
            event_symbol, win,
            from_time.strftime("%H:%M UTC"),
            to_time.strftime("%H:%M UTC"),
        )

        # Collect from all sources, deduplicate by normalised URL
        seen_articles: set[str] = set()
        raw_articles: list[NewsArticle] = []

        for source in self._sources:
            for query in queries:
                try:
                    results = source.fetch(query, from_time, to_time, max_articles=max_arts)
                except Exception as exc:
                    logger.error("Source %s raised unexpectedly: %s", source.name, exc)
                    results = []

                for art in results:
                    dedupe_key = self._dedupe_key(art)
                    if dedupe_key not in seen_articles:
                        seen_articles.add(dedupe_key)
                        raw_articles.append(art)
            continue
            try:
                results = source.fetch(query, from_time, to_time, max_articles=max_arts)
            except Exception as exc:
                # Defensive catch — sources should handle their own errors,
                # but this ensures a buggy source never kills the pipeline.
                logger.error("Source %s raised unexpectedly: %s", source.name, exc)
                results = []

            for art in results:
                norm_url = self._dedupe_key(art)
                if norm_url not in seen_articles:
                    seen_articles.add(norm_url)
                    raw_articles.append(art)

        logger.info(
            "News fetch: %d unique article(s) across %d source(s) before filtering.",
            len(raw_articles), len(self._sources),
        )

        if not raw_articles:
            logger.warning(
                "No news articles found for %s around %s.",
                event_symbol, event_time.strftime("%Y-%m-%d %H:%M UTC"),
            )
            return []

        # Relevance filtering
        filtered = self._filter.filter_and_score(
            raw_articles,
            symbol=event_symbol,
            min_score=min_score,
        )

        logger.info(
            "News fetch: %d article(s) passed relevance filter (min_score=%.1f).",
            len(filtered), min_score,
        )

        if not filtered:
            logger.warning(
                "No relevant news found for %s around %s (all below score %.1f).",
                event_symbol, event_time.strftime("%Y-%m-%d %H:%M UTC"), min_score,
            )

        # Sort: highest relevance first, then most recent
        filtered.sort(key=lambda a: self._ranking_key(a, event_time))
        return filtered

    @classmethod
    def _queries_for_symbol(cls, event_symbol: str) -> tuple[str, ...]:
        """Return distinct NewsAPI searches for an index movement event."""
        index_name = cls._SYMBOL_NAMES.get(event_symbol, event_symbol)
        index_query = cls._BASE_QUERY
        if index_name.lower() not in index_query.lower():
            index_query = f"{index_name} OR {index_query}"
        return (index_query, *cls._CONTEXT_QUERIES)

    @classmethod
    def _dedupe_key(cls, article: NewsArticle) -> str:
        """Canonicalise a URL for matching without changing the displayed URL."""
        if article.url:
            parts = urlsplit(article.url)
            query = urlencode(sorted(
                (key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
                if not key.lower().startswith("utm_") and key.lower() not in cls._TRACKING_QUERY_PARAMS
            ))
            canonical = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), query, ""))
            return f"url:{canonical.lower()}"
        title = re.sub(r"\s+", " ", article.title).strip().lower()
        return f"fallback:{title}|{article.published_at.isoformat()}|{article.source_name.lower()}"

    @classmethod
    def _ranking_key(cls, article: NewsArticle, event_time: datetime) -> tuple[float, int, float, float]:
        published = article.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        is_post_event = published > event_time
        distance_minutes = abs((published - event_time).total_seconds()) / 60.0
        broad_market = bool(cls._BROAD_MARKET_RE.search(f"{article.title} {article.description}"))
        return (-float(broad_market), int(is_post_event), distance_minutes, -article.relevance_score)


# ── Convenience factory ───────────────────────────────────────────────────────


def build_default_ingester() -> NewsIngester:
    """
    Build a NewsIngester with all configured sources registered.
    Called once at startup in main.py.

    Returns a fully wired ingester even if NewsAPI key is missing —
    in that case, no sources are registered and fetch_for_event()
    returns [] with a warning.
    """
    ingester = NewsIngester()

    if config.NEWS_API_KEY:
        ingester.register_source(NewsAPISource(api_key=config.NEWS_API_KEY))
        logger.info("NewsAPI source registered.")
    else:
        logger.warning(
            "NEWSAPI_KEY not set in .env — news ingestion disabled. "
            "Set the key to enable Iteration 2."
        )

    # Future sources:
    # if config.MONEYCONTROL_RSS_ENABLED:
    #     ingester.register_source(MoneycontrolRSSSource())

    return ingester
