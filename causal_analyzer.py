"""
causal_analyzer.py — LLM-powered causal analysis for price movement events (Iteration 3).

Architecture
────────────
                 ┌──────────────────────────────────────────────────────┐
                 │               CausalAnalyzer                         │
                 │  (builds prompt, calls LLMClient, parses response)   │
                 └───────────────────┬──────────────────────────────────┘
                                     │
                    ┌────────────────▼──────────────────┐
                    │          LLMClient (ABC)            │  ← pluggable
                    └────────────────┬──────────────────┘
                                     │
                         ┌───────────▼───────────┐
                         │   GeminiClient         │  (google-genai SDK)
                         └───────────────────────┘

Public API
──────────
    result = CausalAnalyzer().analyse(event, top_articles)
    print(result.explanation)   # 1-3 sentence plain English
    print(result.confidence)    # "high" | "medium" | "low" | "none"
    print(result.source_refs)   # list[str] — titles of articles relied on
    print(result.is_fallback)   # True if LLM was unavailable

Failure modes handled
─────────────────────
- API key missing    → immediate fallback (no network attempt)
- Timeout / network  → one retry, then fallback
- Rate limit (429)   → no retry, fallback with warning
- Bad/empty response → fallback with warning
- Any other exception → fallback with error log

Fallback result
───────────────
    explanation = "Possible driver: <top article headline>. Automated analysis unavailable."
    confidence  = "none"
    is_fallback = True
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import config

if TYPE_CHECKING:
    from causal_prefilter import ScoredArticle
    from detector import MovementEvent

logger = logging.getLogger(__name__)


# ── Retry helpers ————————————————————————————————————————————————

# HTTP status codes that are never worth retrying (client-side errors).
_NO_RETRY_STATUSES = frozenset({400, 401, 403, 404})


def _extract_http_status(exc_str: str) -> int | None:
    """
    Attempt to extract an HTTP status code from an exception message string.
    Returns the int status code, or None if none found.
    """
    import re as _re
    m = _re.search(r"\b([1-5]\d{2})\b", exc_str)
    return int(m.group(1)) if m else None


def _is_retryable(exc: Exception) -> bool:
    """
    Return True iff the exception warrants a retry.

    Retryable  : 429 (rate-limit), 5xx server errors, network/timeout errors.
    Not-retryable: 400 Bad Request, 401 Unauthorised, 403 Forbidden, 404 Not Found,
                   or any exception that doesn't look like a transient failure.
    """
    exc_str = str(exc).lower()

    # Explicit network / timeout keywords — always transient
    transient_keywords = ("timeout", "connection", "network", "connectionerror",
                          "connecttimeout", "readtimeout", "remotedisconnected")
    if any(k in exc_str for k in transient_keywords):
        return True

    status = _extract_http_status(str(exc))
    if status is not None:
        if status in _NO_RETRY_STATUSES:
            return False          # definitive client error — don't retry
        if status == 429 or status >= 500:
            return True           # rate-limit or server error — retry
        return False              # other 4xx — don't retry

    # Unknown exception type — retry cautiously (e.g. SDK-internal errors)
    return True


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class CausalResult:
    """
    Output of a causal analysis run.

    Fields
    ------
    explanation:
        Plain-English probable cause, max ~3 sentences.
        "No clear cause found." when news doesn't explain the move.
    confidence:
        "high" | "medium" | "low" | "none"
    source_refs:
        Titles of articles the model cited as evidence.
    is_fallback:
        True when the LLM was unavailable and a rule-based fallback was used.
    raw_response:
        Raw text from the LLM (empty string for fallbacks).
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
            f"{tag}CAUSAL ANALYSIS [{self.confidence.upper()}]\n"
            f"  {self.explanation}\n"
            f"  Sources: {refs}"
        )


# ── LLMClient interface ───────────────────────────────────────────────────────


class LLMClient(ABC):
    """Provider-agnostic interface for a text-completion LLM."""

    @abstractmethod
    def complete(self, prompt: str, timeout_seconds: float = 20.0) -> str:
        """
        Send *prompt* to the model and return the raw text response.
        Raises any exception on error — the caller handles retries/fallback.
        """


# ── Gemini implementation ─────────────────────────────────────────────────────


class GeminiClient(LLMClient):
    """
    LLMClient implementation using the official google-genai SDK (v2.x).
    Uses config.LLM_MODEL and config.GEMINI_API_KEY.
    """

    def __init__(self) -> None:
        from google import genai as _genai  # type: ignore[import]
        from google.genai import types as _types  # type: ignore[import]

        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in config/.env")

        self._client = _genai.Client(api_key=config.GEMINI_API_KEY)
        self._model  = config.LLM_MODEL
        self._types  = _types
        logger.info("GeminiClient ready (model=%s)", self._model)

    def complete(self, prompt: str, timeout_seconds: float = 20.0) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=0.2,          # low temperature for factual output
                max_output_tokens=512,
            ),
        )
        return response.text or ""


# ── Prompt construction ───────────────────────────────────────────────────────

_SYSTEM_INSTRUCTIONS = """\
You are a financial news analyst specializing in Indian equity markets (NSE/BSE).
Your task: given a detected price movement in a market index and a set of
recent news headlines, identify the most probable cause of the price move.

Rules:
1. Answer ONLY from the provided news context. Do NOT invent causes not
   mentioned in the articles.
2. If the news does not clearly explain the move, say "No clear cause found."
3. Keep your explanation to 1-3 plain-English sentences.
4. State your confidence as exactly one of: high, medium, low, none.
5. List the title(s) of the article(s) you relied on in the source_refs field.
6. Respond ONLY with valid JSON matching this schema:
   {
     "explanation": "<string>",
     "confidence": "high|medium|low|none",
     "source_refs": ["<article title>", ...]
   }
"""


def build_prompt(event: "MovementEvent", top_articles: list["ScoredArticle"]) -> str:
    """
    Construct the full prompt to send to the LLM.

    Parameters
    ----------
    event:
        The detected MovementEvent (symbol, direction, %, times).
    top_articles:
        Ranked ScoredArticle list from the causal pre-filter (top N).

    Returns
    -------
    A single string prompt ready to send to any LLM.
    """
    event_block = (
        f"MARKET EVENT\n"
        f"  Index     : {event.symbol}\n"
        f"  Direction : {event.direction}\n"
        f"  Change    : {event.pct_change:+.3f}%\n"
        f"  Window    : {event.window_minutes} min "
        f"({event.window_start.strftime('%H:%M UTC')} -> "
        f"{event.window_end.strftime('%H:%M UTC')})\n"
        f"  Detected  : {event.detected_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
    )

    articles_block_lines = ["NEWS ARTICLES (ranked by causal probability, most likely first)"]
    for i, sa in enumerate(top_articles, 1):
        art = sa.article
        snippet = art.description[:200].strip() if art.description else "(no snippet)"
        articles_block_lines.append(
            f"\n[{i}] Title     : {art.title}\n"
            f"    Published : {art.published_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"    Snippet   : {snippet}\n"
            f"    Source    : {art.source_name}"
        )
    articles_block = "\n".join(articles_block_lines)

    if not top_articles:
        articles_block = "NEWS ARTICLES: (none available)"

    return (
        f"{_SYSTEM_INSTRUCTIONS}\n\n"
        f"---\n\n"
        f"{event_block}\n"
        f"{articles_block}\n\n"
        f"---\n\n"
        f"Respond with the JSON object only. No markdown fences."
    )


# ── Response parsing ──────────────────────────────────────────────────────────


def parse_llm_response(raw: str) -> CausalResult:
    """
    Parse the LLM's raw text response into a CausalResult.

    Attempts strict JSON parsing first, then falls back to regex extraction
    for malformed responses (e.g. if the model wraps in markdown fences).

    Parameters
    ----------
    raw:
        Raw text string from the LLM.

    Returns
    -------
    CausalResult (is_fallback=False if parsing succeeded).

    Raises
    ------
    ValueError if the response cannot be parsed at all.
    """
    # Strip markdown fences if present
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage with regex
        exp_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', text)
        conf_match = re.search(r'"confidence"\s*:\s*"(high|medium|low|none)"', text, re.IGNORECASE)
        refs_match = re.findall(r'"source_refs"\s*:\s*\[([^\]]*)\]', text)

        if not exp_match:
            raise ValueError(f"Cannot parse LLM response: {raw[:200]}")

        explanation = exp_match.group(1)
        confidence  = conf_match.group(1).lower() if conf_match else "low"
        source_refs: list[str] = []
        if refs_match:
            source_refs = [r.strip().strip('"') for r in refs_match[0].split(",") if r.strip()]

        return CausalResult(
            explanation=explanation,
            confidence=confidence,
            source_refs=source_refs,
            raw_response=raw,
        )

    explanation = str(data.get("explanation", "")).strip()
    confidence  = str(data.get("confidence", "low")).lower().strip()
    source_refs = [str(r) for r in data.get("source_refs", [])]

    if confidence not in {"high", "medium", "low", "none"}:
        confidence = "low"
    if not explanation:
        raise ValueError("LLM returned empty explanation field")

    return CausalResult(
        explanation=explanation,
        confidence=confidence,
        source_refs=source_refs,
        raw_response=raw,
    )


# ── Fallback builder ──────────────────────────────────────────────────────────


def build_fallback_result(top_articles: list["ScoredArticle"], reason: str = "") -> CausalResult:
    """
    Build a rule-based fallback result when the LLM is unavailable.

    Uses the title of the highest-ranked causal article as the probable cause.
    """
    if top_articles:
        headline = top_articles[0].article.title
        explanation = (
            f"Possible driver: {headline}. "
            f"Automated analysis unavailable{(' (' + reason + ')') if reason else ''}."
        )
    else:
        explanation = (
            f"No news articles found. "
            f"Automated analysis unavailable{(' (' + reason + ')') if reason else ''}."
        )
    return CausalResult(
        explanation=explanation,
        confidence="none",
        source_refs=[top_articles[0].article.title] if top_articles else [],
        is_fallback=True,
    )


# ── Main analyzer ─────────────────────────────────────────────────────────────


class CausalAnalyzer:
    """
    Orchestrates prompt construction, LLM call, response parsing, and fallback.

    Parameters
    ----------
    client:
        An LLMClient implementation.  When None, GeminiClient is instantiated
        automatically using config.LLM_PROVIDER / config.GEMINI_API_KEY.
    timeout:
        Per-call timeout in seconds (default: config.LLM_TIMEOUT_SECONDS).
    max_retries:
        Number of retries on transient errors (default: 1).
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        timeout: float | None = None,
        max_retries: int = 1,
    ) -> None:
        self._timeout     = timeout or float(config.LLM_TIMEOUT_SECONDS)
        self._max_retries = max_retries
        self._client      = client  # lazy-init if None

    def _get_client(self) -> LLMClient | None:
        """Lazily initialise the LLM client; return None on configuration error."""
        if self._client is not None:
            return self._client

        provider = config.LLM_PROVIDER.lower()
        if provider == "gemini":
            if not config.GEMINI_API_KEY:
                logger.warning("GEMINI_API_KEY not set — causal analysis will use fallback.")
                return None
            try:
                self._client = GeminiClient()
                return self._client
            except Exception as exc:
                logger.warning("Could not initialise GeminiClient: %s", exc)
                return None
        else:
            logger.warning("Unsupported LLM_PROVIDER=%r — using fallback.", provider)
            return None

    def analyse(
        self,
        event: "MovementEvent",
        top_articles: list["ScoredArticle"],
    ) -> CausalResult:
        """
        Run the full causal analysis pipeline for *event*.

        Pipeline
        ────────
        1. If no articles are available, return a "no news" fallback immediately
           (no LLM call).
        2. Build prompt from event + top_articles.
        3. Send to LLM with timeout, exponential backoff and retry.
        4. Parse the JSON response.
        5. Return CausalResult (or fallback on any non-retryable / exhausted error).

        Retry policy
        ────────────
        - Retried : 429, 5xx, network/timeout errors.
        - Not retried : 400, 401, 403, 404 — fail immediately.
        - Backoff  : 2 ** attempt seconds (2 s, 4 s, 8 s …).
        """
        # Short-circuit: no articles → no point calling the LLM
        if not top_articles:
            logger.info("Causal analysis: no articles — returning no-news fallback.")
            return CausalResult(
                explanation="No relevant news articles were found around this event.",
                confidence="none",
                source_refs=[],
                is_fallback=True,
            )

        client = self._get_client()
        if client is None:
            return build_fallback_result(top_articles, reason="no LLM client configured")

        prompt = build_prompt(event, top_articles)
        logger.debug("LLM prompt (%d chars):\n%s", len(prompt), prompt)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = client.complete(prompt, timeout_seconds=self._timeout)
                logger.debug("LLM raw response: %s", raw[:300])
                result = parse_llm_response(raw)
                logger.info(
                    "Causal analysis: confidence=%s  explanation=%s",
                    result.confidence, result.explanation[:80],
                )
                return result

            except Exception as exc:
                last_exc = exc
                exc_str  = str(exc)

                if not _is_retryable(exc):
                    logger.warning(
                        "LLM call failed with non-retryable error: %s", exc_str
                    )
                    break

                if attempt < self._max_retries:
                    wait = 2 ** (attempt + 1)   # 2, 4, 8 … seconds
                    logger.warning(
                        "LLM attempt %d/%d failed (%s) — retrying in %ds …",
                        attempt + 1, self._max_retries + 1, exc_str, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "LLM call failed after %d attempt(s): %s",
                        attempt + 1, exc_str,
                    )

        reason = type(last_exc).__name__ if last_exc else "unknown"
        return build_fallback_result(top_articles, reason=reason)


# ── Factory ───────────────────────────────────────────────────────────────────


def build_default_analyzer() -> CausalAnalyzer:
    """
    Build a CausalAnalyzer using the current config.
    Does NOT raise — if the API key is missing the analyzer will use fallback.
    """
    return CausalAnalyzer()
