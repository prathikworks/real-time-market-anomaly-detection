"""
tests/test_causal_iteration3.py — Unit tests for Iteration 3 causal analysis.

Coverage
────────
causal_prefilter : rank_for_causality, _recency_score, _keyword_score,
                   _source_score, _india_score
causal_analyzer  : build_prompt, parse_llm_response, build_fallback_result,
                   CausalAnalyzer.analyse (mock LLM, fallback, retry, 429)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest


# ── Shared helpers ────────────────────────────────────────────────────────────


def _dt(offset_minutes: int = 0) -> datetime:
    base = datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc)
    return base + timedelta(minutes=offset_minutes)


def _make_article(
    *,
    title: str = "Nifty rises on strong earnings",
    description: str = "Indian equity markets advanced.",
    source_name: str = "Economic Times",
    published_offset_minutes: int = 0,
    relevance_score: float = 2.0,
    url: str = "https://example.com/article",
):
    from news_fetcher import NewsArticle
    return NewsArticle(
        title=title,
        description=description,
        url=url,
        source_name=source_name,
        published_at=_dt(published_offset_minutes),
        relevance_score=relevance_score,
    )


def _make_event(symbol: str = "^NSEI"):
    from detector import MovementEvent
    return MovementEvent(
        symbol=symbol,
        detected_at=_dt(0),
        window_start=_dt(-5),
        window_end=_dt(0),
        price_start=22000.0,
        price_end=22330.0,
        pct_change=1.5,
        window_minutes=5,
        threshold_used=0.5,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# causal_prefilter
# ═══════════════════════════════════════════════════════════════════════════════


class TestRankForCausality:

    def test_empty_returns_empty(self):
        from causal_prefilter import rank_for_causality
        assert rank_for_causality([], event_time=_dt(), top_n=3) == []

    def test_returns_at_most_top_n(self):
        from causal_prefilter import rank_for_causality
        arts = [_make_article(title=str(i)) for i in range(10)]
        assert len(rank_for_causality(arts, event_time=_dt(), top_n=3)) == 3

    def test_fewer_than_top_n_returns_all(self):
        from causal_prefilter import rank_for_causality
        arts = [_make_article(), _make_article(title="B")]
        assert len(rank_for_causality(arts, event_time=_dt(), top_n=5)) == 2

    def test_sorted_descending_by_causal_score(self):
        from causal_prefilter import rank_for_causality
        # article at event time should score higher than one from 2h ago
        a1 = _make_article(title="RBI rate hike Nifty crash", published_offset_minutes=0)
        a2 = _make_article(title="Generic update", published_offset_minutes=-120)
        result = rank_for_causality([a2, a1], event_time=_dt(), top_n=5)
        assert result[0].causal_score >= result[1].causal_score

    def test_causal_score_bounded_0_to_1(self):
        from causal_prefilter import rank_for_causality
        arts = [
            _make_article(title="RBI repo crash halt", source_name="Economic Times",
                          published_offset_minutes=0),
            _make_article(title="S&P 500 FOMC", source_name="Reddit",
                          published_offset_minutes=-200),
        ]
        for sa in rank_for_causality(arts, event_time=_dt(), top_n=5):
            assert 0.0 <= sa.causal_score <= 1.0, f"score out of range: {sa.causal_score}"


class TestRecencyScore:

    def test_at_event_time_scores_1(self):
        from causal_prefilter import _recency_score
        art = _make_article(published_offset_minutes=0)
        assert abs(_recency_score(art, _dt(), 30.0) - 1.0) < 1e-6

    def test_at_halflife_scores_half(self):
        from causal_prefilter import _recency_score
        art = _make_article(published_offset_minutes=-30)
        assert abs(_recency_score(art, _dt(), 30.0) - 0.5) < 1e-6

    def test_very_old_near_zero(self):
        from causal_prefilter import _recency_score
        art = _make_article(published_offset_minutes=-480)
        assert _recency_score(art, _dt(), 30.0) < 0.05

    def test_naive_datetime_no_error(self):
        from causal_prefilter import _recency_score
        from news_fetcher import NewsArticle
        art = NewsArticle(
            title="T", description="", url="https://x.com",
            source_name="T", published_at=datetime(2024, 1, 15, 9, 30),
            relevance_score=1.0,
        )
        score = _recency_score(art, _dt(), 30.0)
        assert 0.0 <= score <= 1.0


class TestKeywordScore:

    def test_high_density_scores_high(self):
        from causal_prefilter import _keyword_score
        art = _make_article(title="RBI rate hike policy decision report",
                            description="GDP inflation data signals repo cut budget")
        assert _keyword_score(art) > 0.5

    def test_generic_article_scores_low(self):
        from causal_prefilter import _keyword_score
        art = _make_article(title="Markets open today", description="Stocks traded.")
        assert _keyword_score(art) < 0.3

    def test_score_capped_at_1(self):
        from causal_prefilter import _keyword_score
        art = _make_article(
            title="rate hike cut policy decision report result earnings gdp inflation",
            description="crash halt circuit ban sanction default bankruptcy merger acquisition ipo",
        )
        assert _keyword_score(art) == 1.0


class TestSourceScore:

    def test_tier1_scores_1(self):
        from causal_prefilter import _source_score
        for src in ["Economic Times", "Bloomberg", "moneycontrol", "Reuters"]:
            assert _source_score(_make_article(source_name=src)) == 1.0, src

    def test_tier3_scores_02(self):
        from causal_prefilter import _source_score
        for src in ["Benzinga", "SeekingAlpha", "Reddit"]:
            assert _source_score(_make_article(source_name=src)) == 0.2, src

    def test_unknown_scores_neutral(self):
        from causal_prefilter import _source_score
        assert _source_score(_make_article(source_name="Some Random Blog")) == 0.6


class TestIndiaScore:

    def test_purely_indian_scores_1(self):
        from causal_prefilter import _india_score
        art = _make_article(title="RBI cuts repo rate",
                            description="Nifty and Sensex surged on NSE.")
        assert _india_score(art) == 1.0

    def test_mixed_scores_06(self):
        from causal_prefilter import _india_score
        art = _make_article(title="RBI amid US Fed signals",
                            description="Nifty up S&P 500 flat.")
        assert _india_score(art) == 0.6

    def test_us_only_scores_low(self):
        from causal_prefilter import _india_score
        art = _make_article(title="S&P 500 rises FOMC meeting",
                            description="Wall Street gains Federal Reserve.")
        assert _india_score(art) == 0.1

    def test_neutral_scores_05(self):
        from causal_prefilter import _india_score
        art = _make_article(title="Oil prices rise", description="Crude oil at 80.")
        assert _india_score(art) == 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# causal_analyzer
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildPrompt:

    def test_contains_symbol(self):
        from causal_analyzer import build_prompt
        from causal_prefilter import rank_for_causality
        top = rank_for_causality([_make_article()], event_time=_dt())
        assert "^NSEI" in build_prompt(_make_event("^NSEI"), top)

    def test_contains_direction(self):
        from causal_analyzer import build_prompt
        assert "UP" in build_prompt(_make_event(), [])

    def test_contains_article_title(self):
        from causal_analyzer import build_prompt
        from causal_prefilter import rank_for_causality
        art = _make_article(title="RBI hikes repo rate by 25bps")
        top = rank_for_causality([art], event_time=_dt())
        assert "RBI hikes repo rate by 25bps" in build_prompt(_make_event(), top)

    def test_no_articles_handled(self):
        from causal_analyzer import build_prompt
        assert "NEWS ARTICLES" in build_prompt(_make_event(), [])


class TestParseLLMResponse:

    def test_clean_json(self):
        from causal_analyzer import parse_llm_response
        raw = json.dumps({
            "explanation": "RBI rate hike caused sell-off.",
            "confidence": "high",
            "source_refs": ["Art A"],
        })
        r = parse_llm_response(raw)
        assert r.explanation == "RBI rate hike caused sell-off."
        assert r.confidence == "high"
        assert r.source_refs == ["Art A"]
        assert r.is_fallback is False

    def test_markdown_fences_stripped(self):
        from causal_analyzer import parse_llm_response
        # Build with chr(96) to avoid escaping issues inside the string
        fence = chr(96) * 3
        raw = (fence + "json\n"
               + json.dumps({"explanation": "Budget rally.", "confidence": "medium",
                             "source_refs": []})
               + "\n" + fence)
        r = parse_llm_response(raw)
        assert r.explanation == "Budget rally."
        assert r.confidence == "medium"

    def test_invalid_confidence_clamped_to_low(self):
        from causal_analyzer import parse_llm_response
        raw = json.dumps({"explanation": "X happened.", "confidence": "very_high",
                          "source_refs": []})
        assert parse_llm_response(raw).confidence == "low"

    def test_empty_explanation_raises(self):
        from causal_analyzer import parse_llm_response
        with pytest.raises(ValueError):
            parse_llm_response(json.dumps({"explanation": "", "confidence": "low",
                                           "source_refs": []}))

    def test_totally_unparseable_raises(self):
        from causal_analyzer import parse_llm_response
        with pytest.raises(ValueError):
            parse_llm_response("This is not JSON at all.")


class TestBuildFallbackResult:

    def test_uses_top_headline(self):
        from causal_analyzer import build_fallback_result
        from causal_prefilter import rank_for_causality
        top = rank_for_causality([_make_article(title="RBI policy unchanged")],
                                 event_time=_dt())
        r = build_fallback_result(top)
        assert r.is_fallback is True
        assert "RBI policy unchanged" in r.explanation

    def test_empty_articles(self):
        from causal_analyzer import build_fallback_result
        r = build_fallback_result([])
        assert r.is_fallback is True
        assert r.source_refs == []

    def test_reason_in_explanation(self):
        from causal_analyzer import build_fallback_result
        r = build_fallback_result([], reason="timeout")
        assert "timeout" in r.explanation


class TestCausalAnalyzer:

    def _scored(self):
        from causal_prefilter import rank_for_causality
        return rank_for_causality([_make_article(title="RBI rate cut boosts Nifty")],
                                  event_time=_dt())

    def test_real_result_on_valid_response(self):
        from causal_analyzer import CausalAnalyzer, LLMClient

        class MockClient(LLMClient):
            def complete(self, prompt: str, timeout_seconds: float = 20.0) -> str:
                return json.dumps({
                    "explanation": "RBI rate cut boosted market.",
                    "confidence": "high",
                    "source_refs": ["RBI rate cut boosts Nifty"],
                })

        r = CausalAnalyzer(client=MockClient()).analyse(_make_event(), self._scored())
        assert r.is_fallback is False
        assert r.confidence == "high"
        assert "RBI" in r.explanation

    def test_fallback_when_no_client(self):
        from causal_analyzer import CausalAnalyzer
        analyzer = CausalAnalyzer(client=None)
        with patch.object(analyzer, "_get_client", return_value=None):
            r = analyzer.analyse(_make_event(), self._scored())
        assert r.is_fallback is True

    def test_fallback_on_exception(self):
        from causal_analyzer import CausalAnalyzer, LLMClient

        class BrokenClient(LLMClient):
            def complete(self, prompt: str, timeout_seconds: float = 20.0) -> str:
                raise RuntimeError("Network error")

        r = CausalAnalyzer(client=BrokenClient(), max_retries=0).analyse(
            _make_event(), self._scored()
        )
        assert r.is_fallback is True

    def test_429_is_retried(self):
        """429 is now treated as retryable (rate-limit is transient)."""
        from causal_analyzer import CausalAnalyzer, LLMClient

        call_count = []

        class RateLimitClient(LLMClient):
            def complete(self, prompt: str, timeout_seconds: float = 20.0) -> str:
                call_count.append(1)
                raise Exception("429 Too Many Requests")

        with patch("causal_analyzer.time.sleep"):
            CausalAnalyzer(client=RateLimitClient(), max_retries=1).analyse(
                _make_event(), self._scored()
            )
        # max_retries=1 → 2 total attempts (1 original + 1 retry)
        assert len(call_count) == 2, "429 should be retried once"

    def test_retries_on_transient_error(self):
        from causal_analyzer import CausalAnalyzer, LLMClient

        attempts = []

        class FlakyClient(LLMClient):
            def complete(self, prompt: str, timeout_seconds: float = 20.0) -> str:
                attempts.append(1)
                if len(attempts) < 2:
                    raise RuntimeError("Transient failure")
                return json.dumps({
                    "explanation": "Recovered on retry.",
                    "confidence": "low",
                    "source_refs": [],
                })

        with patch("causal_analyzer.time.sleep"):
            r = CausalAnalyzer(client=FlakyClient(), max_retries=1).analyse(
                _make_event(), self._scored()
            )

        assert len(attempts) == 2
        assert r.is_fallback is False
        assert r.explanation == "Recovered on retry."

    def test_empty_articles_no_crash(self):
        """Empty articles now short-circuit before LLM — always returns is_fallback=True."""
        from causal_analyzer import CausalAnalyzer, LLMClient

        llm_called = []

        class MockClient(LLMClient):
            def complete(self, prompt: str, timeout_seconds: float = 20.0) -> str:
                llm_called.append(1)
                return json.dumps({
                    "explanation": "No clear cause found.",
                    "confidence": "none",
                    "source_refs": [],
                })

        r = CausalAnalyzer(client=MockClient()).analyse(_make_event(), [])
        assert isinstance(r.explanation, str) and len(r.explanation) > 0
        assert r.is_fallback is True
        assert len(llm_called) == 0, "LLM must NOT be called when articles list is empty"
