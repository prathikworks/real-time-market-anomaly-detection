"""
tests/test_news_ingestion.py — Offline unit tests for Iteration 2.

Coverage
────────
  RelevanceFilter / score_article / is_relevant:
    - Tier-1 keywords (Nifty, Sensex, NSE, BSE)
    - Tier-2 keywords (RBI, budget, etc.)
    - Tier-3 keywords (rally, crash, etc.)
    - Symbol-specific scoring bonus
    - Articles that SHOULD be discarded
    - Min-score threshold boundary

  NewsIngester (mocked NewsAPISource):
    - Happy path: articles fetched, filtered, returned sorted
    - No sources registered: returns []
    - Source raises unexpectedly: returns [] without crashing
    - All articles fail relevance: returns [] with warning
    - Deduplication by URL across sources
    - Time-window parameters passed correctly to source

  NewsAPISource._parse_article (unit tests on the private parser):
    - Valid article dict parsed correctly
    - Missing/None title returns None
    - "[Removed]" title returns None
    - Bad publishedAt returns None

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Shared test fixtures ──────────────────────────────────────────────────────

UTC = timezone.utc
EVENT_TIME = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
SYMBOL = "^NSEI"


def make_article(
    title: str,
    description: str = "",
    url: str = "https://example.com/article",
    published_at: datetime | None = None,
    source_name: str = "TestSource",
) -> "NewsArticle":
    """Helper that imports NewsArticle lazily (avoids config import at module level)."""
    from news_fetcher import NewsArticle
    return NewsArticle(
        title=title,
        description=description,
        url=url,
        published_at=published_at or EVENT_TIME,
        source_name=source_name,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — RelevanceFilter / scoring functions
# ══════════════════════════════════════════════════════════════════════════════


class TestScoreArticle:
    """Tests for relevance_filter.score_article()."""

    def test_tier1_nifty_in_title(self):
        from relevance_filter import score_article
        art = make_article("Nifty falls 200 points on weak global cues")
        assert score_article(art, SYMBOL) >= 3.0

    def test_tier1_sensex_in_title(self):
        from relevance_filter import score_article
        art = make_article("Sensex drops 500 points; markets rattled")
        assert score_article(art, "^BSESN") >= 3.0

    def test_tier1_nse_keyword_matches(self):
        from relevance_filter import score_article
        art = make_article("NSE trading halted briefly due to technical glitch")
        assert score_article(art, SYMBOL) >= 3.0

    def test_tier1_bse_keyword_matches(self):
        from relevance_filter import score_article
        art = make_article("BSE Sensex surges 600 points in early trade")
        assert score_article(art) >= 3.0

    def test_tier2_rbi_in_description(self):
        from relevance_filter import score_article
        art = make_article("Markets react to rate decision", description="RBI raises repo rate by 25 bps")
        assert score_article(art) >= 2.0

    def test_tier2_budget_keyword(self):
        from relevance_filter import score_article
        art = make_article("Union Budget 2024: What it means for investors")
        assert score_article(art) >= 2.0

    def test_tier3_rally_keyword(self):
        from relevance_filter import score_article
        art = make_article("Indian stock market stages a strong rally")
        assert score_article(art) >= 1.0

    def test_tier3_crash_keyword(self):
        from relevance_filter import score_article
        art = make_article("Markets crash as FIIs pull out funds")
        # "crash" (T3=1) + "FIIs" matches FII pattern (T2=2)
        assert score_article(art) >= 1.0

    def test_completely_irrelevant_article_scores_zero(self):
        from relevance_filter import score_article
        art = make_article("Best pizza recipes for summer 2024",
                           description="Try these amazing pizza toppings")
        assert score_article(art) == 0.0

    def test_sports_article_scores_zero(self):
        from relevance_filter import score_article
        art = make_article("India wins cricket series against Australia",
                           description="Team India beats Australia 3-1 in the ODI series")
        assert score_article(art) == 0.0

    def test_score_cumulates_multiple_keywords(self):
        from relevance_filter import score_article
        # Has Nifty (T1), RBI (T2), rally (T3) — score should be sum
        art = make_article("Nifty rallies after RBI holds repo rate steady")
        score = score_article(art, SYMBOL)
        assert score >= 6.0   # at least Nifty(3) + RBI(2) + rally(1)

    def test_symbol_specific_bonus_for_nsei(self):
        from relevance_filter import score_article
        art_nifty = make_article("Nifty hits record high")
        art_sensex = make_article("Sensex hits record high")
        # ^NSEI gets bonus on "Nifty" keyword
        score_nifty_for_nsei = score_article(art_nifty, "^NSEI")
        score_sensex_for_nsei = score_article(art_sensex, "^NSEI")
        assert score_nifty_for_nsei > score_sensex_for_nsei

    def test_symbol_specific_bonus_for_bsesn(self):
        from relevance_filter import score_article
        art = make_article("Sensex crosses 70000 mark for first time")
        score = score_article(art, "^BSESN")
        # Should get both the global T1 "Sensex" AND the symbol-specific bonus
        assert score >= 6.0


class TestIsRelevant:
    """Tests for relevance_filter.is_relevant()."""

    def test_relevant_article_passes_default_threshold(self):
        from relevance_filter import is_relevant
        art = make_article("Nifty tumbles 300 points on profit booking")
        assert is_relevant(art, SYMBOL) is True

    def test_irrelevant_article_fails_default_threshold(self):
        from relevance_filter import is_relevant
        art = make_article("Weather forecast: heavy rains expected in Mumbai")
        assert is_relevant(art, SYMBOL) is False

    def test_boundary_exactly_at_min_score_passes(self):
        from relevance_filter import score_article, is_relevant
        art = make_article("Stock market rally", description="equities rise")
        # Explicitly compute score and check boundary
        score = score_article(art)
        assert is_relevant(art, min_score=score) is True

    def test_boundary_just_below_min_score_fails(self):
        from relevance_filter import score_article, is_relevant
        art = make_article("Stock market rally", description="equities rise")
        score = score_article(art)
        assert is_relevant(art, min_score=score + 0.01) is False

    def test_high_min_score_rejects_weak_matches(self):
        from relevance_filter import is_relevant
        # "shares" alone (T3=1) should fail a min_score of 5
        art = make_article("Company XYZ issues new shares to employees")
        assert is_relevant(art, min_score=5.0) is False


class TestRelevanceFilter:
    """Tests for RelevanceFilter.filter_and_score()."""

    def test_returns_scored_articles(self):
        from relevance_filter import RelevanceFilter
        rf = RelevanceFilter()
        articles = [
            make_article("Nifty drops 400 points"),
            make_article("Best pizza recipes"),
        ]
        result = rf.filter_and_score(articles, symbol=SYMBOL, min_score=1.0)
        assert len(result) == 1
        assert result[0].title == "Nifty drops 400 points"
        assert result[0].relevance_score > 0

    def test_all_irrelevant_returns_empty(self):
        from relevance_filter import RelevanceFilter
        rf = RelevanceFilter()
        articles = [
            make_article("Pizza recipe"),
            make_article("Cricket match results"),
        ]
        result = rf.filter_and_score(articles, symbol=SYMBOL, min_score=1.0)
        assert result == []

    def test_empty_input_returns_empty(self):
        from relevance_filter import RelevanceFilter
        result = RelevanceFilter().filter_and_score([], symbol=SYMBOL)
        assert result == []

    def test_score_is_attached_to_returned_article(self):
        from relevance_filter import RelevanceFilter, score_article
        rf = RelevanceFilter()
        art = make_article("Sensex gains 500 points on FII buying")
        [scored] = rf.filter_and_score([art], symbol="^BSESN", min_score=1.0)
        expected = score_article(art, "^BSESN")
        assert abs(scored.relevance_score - expected) < 1e-9

    def test_original_article_is_not_mutated(self):
        """filter_and_score must not modify the input NewsArticle objects."""
        from relevance_filter import RelevanceFilter
        art = make_article("Nifty drops 400 points")
        original_score = art.relevance_score   # 0.0
        RelevanceFilter().filter_and_score([art], symbol=SYMBOL, min_score=1.0)
        assert art.relevance_score == original_score  # still 0.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — NewsIngester (mocked sources)
# ══════════════════════════════════════════════════════════════════════════════


def _mock_source(name: str, articles: list) -> MagicMock:
    """Build a mock NewsSource whose fetch() returns *articles*."""
    from news_fetcher import NewsSource
    src = MagicMock(spec=NewsSource)
    src.name = name
    src.fetch.return_value = articles
    return src


class TestNewsIngester:
    """Tests for NewsIngester.fetch_for_event()."""

    def test_happy_path_returns_filtered_articles(self):
        from news_fetcher import NewsIngester
        ingester = NewsIngester()
        relevant = make_article("Nifty crashes 500 points after RBI surprise")
        irrelevant = make_article("Cricket: India vs Australia highlights")
        ingester.register_source(_mock_source("MockSource", [relevant, irrelevant]))
        results = ingester.fetch_for_event(SYMBOL, EVENT_TIME, min_relevance_score=1.0)
        assert len(results) == 1
        assert results[0].title == relevant.title

    def test_no_sources_returns_empty(self):
        from news_fetcher import NewsIngester
        ingester = NewsIngester()   # no sources registered
        results = ingester.fetch_for_event(SYMBOL, EVENT_TIME)
        assert results == []

    def test_source_raises_exception_returns_empty_gracefully(self):
        """A buggy source must not propagate an exception up the call stack."""
        from news_fetcher import NewsIngester, NewsSource
        ingester = NewsIngester()
        bad_src = MagicMock(spec=NewsSource)
        bad_src.name = "BadSource"
        bad_src.fetch.side_effect = RuntimeError("network exploded")
        ingester.register_source(bad_src)
        # Should not raise
        results = ingester.fetch_for_event(SYMBOL, EVENT_TIME)
        assert results == []

    def test_all_articles_below_min_score_returns_empty(self):
        from news_fetcher import NewsIngester
        ingester = NewsIngester()
        arts = [
            make_article("Weather: Mumbai braces for heavy rains"),
            make_article("Sports round-up: hockey, cricket, tennis"),
        ]
        ingester.register_source(_mock_source("MockSource", arts))
        results = ingester.fetch_for_event(SYMBOL, EVENT_TIME, min_relevance_score=1.0)
        assert results == []

    def test_deduplication_by_url(self):
        """The same article from two sources should appear only once."""
        from news_fetcher import NewsIngester
        url = "https://example.com/nifty-crash"
        art1 = make_article("Nifty drops 400 points", url=url, source_name="Source1")
        art2 = make_article("Nifty drops 400 points", url=url, source_name="Source2")
        ingester = NewsIngester()
        ingester.register_source(_mock_source("Source1", [art1]))
        ingester.register_source(_mock_source("Source2", [art2]))
        results = ingester.fetch_for_event(SYMBOL, EVENT_TIME, min_relevance_score=1.0)
        assert len(results) == 1

    def test_results_sorted_by_relevance_descending(self):
        """Highest-scoring article should come first in the returned list."""
        from news_fetcher import NewsIngester
        low = make_article("Indian stock market", url="https://example.com/a1")
        high = make_article("Nifty collapses as RBI hikes repo rate sharply", url="https://example.com/a2")
        ingester = NewsIngester()
        ingester.register_source(_mock_source("MockSource", [low, high]))
        results = ingester.fetch_for_event(SYMBOL, EVENT_TIME, min_relevance_score=1.0)
        assert len(results) == 2
        assert results[0].relevance_score >= results[1].relevance_score

    def test_time_window_passed_to_source(self):
        """fetch() on the source should receive from_time / to_time derived from window_minutes."""
        from news_fetcher import NewsIngester
        ingester = NewsIngester()
        src = _mock_source("MockSource", [])
        ingester.register_source(src)
        ingester.fetch_for_event(SYMBOL, EVENT_TIME, window_minutes=30)

        _, call_kwargs = src.fetch.call_args
        from_time = src.fetch.call_args[0][1]
        to_time   = src.fetch.call_args[0][2]
        expected_from = EVENT_TIME - timedelta(minutes=30)
        expected_to   = EVENT_TIME + timedelta(minutes=30)
        assert abs((from_time - expected_from).total_seconds()) < 1
        assert abs((to_time   - expected_to).total_seconds())   < 1

    def test_multiple_sources_combined(self):
        """Articles from two sources should both appear (minus duplicates)."""
        from news_fetcher import NewsIngester
        art_a = make_article("Nifty up 1%", url="https://example.com/a")
        art_b = make_article("Sensex surges on positive sentiment", url="https://example.com/b")
        ingester = NewsIngester()
        ingester.register_source(_mock_source("SourceA", [art_a]))
        ingester.register_source(_mock_source("SourceB", [art_b]))
        results = ingester.fetch_for_event(SYMBOL, EVENT_TIME, min_relevance_score=1.0)
        assert len(results) == 2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NewsAPISource._parse_article (unit tests on the raw parser)
# ══════════════════════════════════════════════════════════════════════════════


class TestNewsAPISourceParser:
    """Tests for the internal _parse_article method of NewsAPISource.

    We instantiate the class with a dummy key so no network call is made.
    """

    def _make_source(self):
        from news_fetcher import NewsAPISource
        return NewsAPISource(api_key="dummy_key_for_test")

    def _valid_raw(self, **overrides) -> dict:
        base = {
            "title": "Nifty drops 300 points",
            "description": "Indian markets fall amid global sell-off",
            "url": "https://example.com/nifty",
            "publishedAt": "2024-01-15T10:00:00Z",
            "source": {"name": "Reuters"},
        }
        base.update(overrides)
        return base

    def test_valid_article_parsed_correctly(self):
        src = self._make_source()
        art = src._parse_article(self._valid_raw())
        assert art is not None
        assert art.title == "Nifty drops 300 points"
        assert art.published_at.tzinfo is not None
        assert art.published_at.year == 2024

    def test_missing_title_returns_none(self):
        src = self._make_source()
        assert src._parse_article(self._valid_raw(title=None)) is None

    def test_empty_title_returns_none(self):
        src = self._make_source()
        assert src._parse_article(self._valid_raw(title="")) is None

    def test_removed_title_returns_none(self):
        """NewsAPI sometimes returns '[Removed]' for taken-down articles."""
        src = self._make_source()
        assert src._parse_article(self._valid_raw(title="[Removed]")) is None

    def test_bad_published_at_returns_none(self):
        src = self._make_source()
        assert src._parse_article(self._valid_raw(publishedAt="not-a-date")) is None

    def test_missing_description_defaults_to_empty_string(self):
        src = self._make_source()
        art = src._parse_article(self._valid_raw(description=None))
        assert art is not None
        assert art.description == ""

    def test_published_at_is_utc_aware(self):
        src = self._make_source()
        art = src._parse_article(self._valid_raw())
        assert art.published_at.tzinfo is not None
        assert art.published_at.utcoffset().total_seconds() == 0

    def test_source_name_is_newsapi(self):
        src = self._make_source()
        art = src._parse_article(self._valid_raw())
        assert art.source_name == "NewsAPI"
