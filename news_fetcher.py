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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

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

        return NewsArticle(
            title=title,
            description=(raw.get("description") or "").strip(),
            url=raw.get("url") or "",
            published_at=published_at,
            source_name=self.name,
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
    _BASE_QUERY = (
        "Nifty OR Sensex OR NSE OR BSE OR \"stock market\" OR \"Indian market\" "
        "OR RBI OR SEBI"
    )

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

        index_name = self._SYMBOL_NAMES.get(event_symbol, event_symbol)
        query = f"{index_name} OR {self._BASE_QUERY}"

        logger.info(
            "News fetch: symbol=%s  window=+/-%d min  [%s -> %s]",
            event_symbol, win,
            from_time.strftime("%H:%M UTC"),
            to_time.strftime("%H:%M UTC"),
        )

        # Collect from all sources, deduplicate by normalised URL
        seen_urls: set[str] = set()
        raw_articles: list[NewsArticle] = []

        for source in self._sources:
            try:
                results = source.fetch(query, from_time, to_time, max_articles=max_arts)
            except Exception as exc:
                # Defensive catch — sources should handle their own errors,
                # but this ensures a buggy source never kills the pipeline.
                logger.error("Source %s raised unexpectedly: %s", source.name, exc)
                results = []

            for art in results:
                norm_url = art.url.rstrip("/").lower()
                if norm_url not in seen_urls:
                    seen_urls.add(norm_url)
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
        filtered.sort(key=lambda a: (-a.relevance_score, -a.published_at.timestamp()))
        return filtered


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
