"""
tests/test_notifier.py - Dry-run test for the notifier module.

Calls send_event_email with a hardcoded fake MovementEvent and fake
ScoredArticle list in dry_run=True mode. Verifies:
  - No exception is raised
  - logs/last_email.txt is created and contains expected content

Does NOT connect to SMTP, does NOT call NewsAPI.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure the project root is on sys.path when run from the tests/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

import notifier


# Minimal stubs (no imports from the real modules needed)

@dataclass(frozen=True)
class _FakeArticle:
    title: str
    source_name: str
    published_at: datetime
    url: str
    description: str = ""
    relevance_score: float = 3.0


@dataclass(frozen=True)
class _FakeScoredArticle:
    article: _FakeArticle
    causal_score: float
    recency_score: float = 0.9
    keyword_score: float = 0.8
    source_score: float  = 0.7
    india_score: float   = 0.6
    broadcast_score: float = 0.85
    relevance_score_norm: float = 0.75


@dataclass(frozen=True)
class _FakeMovementEvent:
    symbol: str
    detected_at: datetime
    window_start: datetime
    window_end: datetime
    price_start: float
    price_end: float
    pct_change: float
    window_minutes: int
    threshold_used: float

    @property
    def direction(self) -> str:
        return "UP" if self.pct_change >= 0 else "DOWN"


@dataclass
class _FakeCausalResult:
    explanation: str
    confidence: str
    source_refs: list = field(default_factory=list)
    is_fallback: bool = False
    raw_response: str = ""


# Test fixtures

_NOW = datetime(2026, 10, 15, 9, 0, 0, tzinfo=timezone.utc)

_FAKE_EVENT = _FakeMovementEvent(
    symbol="^BSESN",
    detected_at=_NOW,
    window_start=_NOW - timedelta(minutes=5),
    window_end=_NOW,
    price_start=80100.00,
    price_end=81078.22,
    pct_change=+1.22,
    window_minutes=5,
    threshold_used=0.4,
)

_FAKE_ARTICLES = [
    _FakeScoredArticle(
        article=_FakeArticle(
            title="RBI unexpectedly cuts repo rate by 25 bps",
            source_name="Economic Times",
            published_at=_NOW - timedelta(minutes=20),
            url="https://economictimes.indiatimes.com/fake-rbi-cut",
            description="The Reserve Bank of India cut the benchmark repo rate.",
        ),
        causal_score=0.87,
    ),
    _FakeScoredArticle(
        article=_FakeArticle(
            title="FII inflows surge; Sensex rallies on foreign buying",
            source_name="Business Standard",
            published_at=_NOW - timedelta(minutes=10),
            url="https://business-standard.com/fake-fii-rally",
            description="Foreign institutional investors poured funds into Indian equities.",
        ),
        causal_score=0.74,
    ),
    _FakeScoredArticle(
        article=_FakeArticle(
            title="Crude oil prices fall sharply on OPEC supply agreement",
            source_name="Reuters",
            published_at=_NOW - timedelta(minutes=35),
            url="https://reuters.com/fake-crude-opec",
            description="Brent crude slipped 3% after OPEC+ announced an output increase.",
        ),
        causal_score=0.61,
    ),
]

_FAKE_RESULT_HIGH = _FakeCausalResult(
    explanation="Possible context for UP move...",
    confidence="high",
    source_refs=["RBI unexpectedly cuts repo rate by 25 bps"],
)

_FAKE_RESULT_NONE = _FakeCausalResult(
    explanation="No strongly relevant pre-event news found; cause undetermined.",
    confidence="none",
    source_refs=[],
)


def test_dry_run_with_articles(tmp_path, monkeypatch, capsys):
    """Dry-run with high-confidence result: email is printed and saved."""
    # Redirect logs/last_email.txt to a temp location so tests don't write to cwd
    monkeypatch.chdir(tmp_path)

    notifier.send_event_email(
        _FAKE_EVENT,
        _FAKE_RESULT_HIGH,
        _FAKE_ARTICLES,
        dry_run=True,
    )

    captured = capsys.readouterr()
    assert "DRY-RUN EMAIL" in captured.out
    assert "^BSESN" in captured.out
    assert "+1.22%" in captured.out
    assert "HIGH" in captured.out
    assert "RBI" in captured.out

    saved = (tmp_path / "logs" / "last_email.txt").read_text(encoding="utf-8")
    assert "^BSESN" in saved
    assert "Market Alert" in saved


def test_dry_run_no_articles(tmp_path, monkeypatch, capsys):
    """Dry-run with no-confidence result: 'cause undetermined' appears."""
    monkeypatch.chdir(tmp_path)

    notifier.send_event_email(
        _FAKE_EVENT,
        _FAKE_RESULT_NONE,
        [],
        dry_run=True,
    )

    captured = capsys.readouterr()
    assert "cause undetermined" in captured.out.lower()
    assert "NONE" in captured.out


def test_subject_format():
    """Subject line matches the expected format."""
    subject = notifier._build_subject(_FAKE_EVENT)
    assert subject == "[Market Alert] ^BSESN UP +1.22% in 5 min"


def test_send_failure_does_not_raise(tmp_path, monkeypatch, capsys):
    """When SMTP creds are missing, send_event_email prints WARNING and does not raise."""
    monkeypatch.chdir(tmp_path)
    # Ensure no SMTP creds in env
    monkeypatch.delenv("SMTP_USER", raising=False)
    monkeypatch.delenv("SMTP_PASSWORD", raising=False)
    monkeypatch.delenv("EMAIL_TO", raising=False)

    # Should not raise, should print WARNING
    notifier.send_event_email(
        _FAKE_EVENT,
        _FAKE_RESULT_HIGH,
        _FAKE_ARTICLES,
        dry_run=False,  # attempt a real send
    )

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
