"""Cheap LLM rubric judge / quality filter.

The :class:`QualityJudge` scores an ``(instruction, response)`` pair on a small
rubric -- correctness, idiomatic Go, completeness, and instruction<->response
alignment -- and reduces those axes to a single overall score in ``[0, 1]``. It
is the second half of the forge quality gate: where :class:`~src.core.verifier`
proves that code *compiles*, the judge filters out pairs that compile but are
low quality (wrong, sloppy, or off-task).

It talks to the same OpenAI-compatible endpoint family as the teacher and reuses
the same pricing / spend-tracking pattern:

* ``FORGE_JUDGE_BASE_URL`` / ``FORGE_JUDGE_API_KEY`` / ``FORGE_JUDGE_MODEL``
* ``FORGE_JUDGE_PRICE_IN`` / ``FORGE_JUDGE_PRICE_OUT``

each of which falls back to the corresponding ``FORGE_TEACHER_*`` value, so a
single endpoint configuration drives both stages. Use a *cheap* model here -- the
judge is a coarse filter, not an oracle.

An :pyattr:`offline` mode produces a deterministic pseudo-score from a content
hash so tests and CI run with no network and no ``openai`` dependency.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.core.instruction_gen import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_PRICE_IN,
    DEFAULT_PRICE_OUT,
    MAX_CONTEXT_CHARS,
    _loads_lenient,
)

logger = logging.getLogger(__name__)

#: Rubric axes scored independently and averaged into the overall score.
JUDGE_AXES: tuple[str, ...] = ("correctness", "idiomatic", "completeness", "alignment")


def _env(name: str, fallback: str, default: str) -> str:
    """Read ``name`` from the env, then ``fallback``, then ``default``."""
    return os.environ.get(name) or os.environ.get(fallback) or default


def _env_price(name: str, fallback: str, default: float) -> float:
    raw = os.environ.get(name) or os.environ.get(fallback)
    return float(raw) if raw is not None else default


@dataclass
class JudgeResult:
    """A judge verdict for one pair.

    Attributes:
        score: Overall quality in ``[0, 1]`` (mean of the per-axis scores).
        axes: Per-axis scores keyed by :data:`JUDGE_AXES`.
        rationale: Short free-text justification (may be empty offline).
    """

    score: float
    axes: dict[str, float] = field(default_factory=dict)
    rationale: str = ""


class QualityJudge:
    """Score and filter instruction/response pairs against a quality rubric."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        offline: bool = False,
        max_context_chars: int = MAX_CONTEXT_CHARS,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        temperature: float = 0.0,
        request_timeout: float = 120.0,
        price_in: float | None = None,
        price_out: float | None = None,
    ) -> None:
        """Initialize the judge.

        Args:
            base_url: OpenAI-compatible base URL. Falls back to
                ``FORGE_JUDGE_BASE_URL`` then ``FORGE_TEACHER_BASE_URL``.
            api_key: API key. Falls back to ``FORGE_JUDGE_API_KEY`` then
                ``FORGE_TEACHER_API_KEY``.
            model: Model id. Falls back to ``FORGE_JUDGE_MODEL`` then
                ``FORGE_TEACHER_MODEL``.
            offline: When ``True``, score deterministically from a content hash
                with no network call or ``openai`` dependency.
            max_context_chars: Truncate each field to this many characters.
            max_retries: Maximum judge-call attempts before giving up.
            backoff_base: Base seconds for exponential backoff between retries.
            temperature: Sampling temperature (``0`` for stable scoring).
            request_timeout: Per-request timeout in seconds.
            price_in: USD per 1M input tokens. Falls back to
                ``FORGE_JUDGE_PRICE_IN`` then ``FORGE_TEACHER_PRICE_IN``.
            price_out: USD per 1M output tokens. Falls back to
                ``FORGE_JUDGE_PRICE_OUT`` then ``FORGE_TEACHER_PRICE_OUT``.
        """
        self.base_url = base_url or _env("FORGE_JUDGE_BASE_URL", "FORGE_TEACHER_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = api_key or _env("FORGE_JUDGE_API_KEY", "FORGE_TEACHER_API_KEY", "not-needed")
        self.model = model or _env("FORGE_JUDGE_MODEL", "FORGE_TEACHER_MODEL", DEFAULT_MODEL)
        self.offline = offline
        self.max_context_chars = max_context_chars
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.price_in = (
            price_in if price_in is not None
            else _env_price("FORGE_JUDGE_PRICE_IN", "FORGE_TEACHER_PRICE_IN", DEFAULT_PRICE_IN)
        )
        self.price_out = (
            price_out if price_out is not None
            else _env_price("FORGE_JUDGE_PRICE_OUT", "FORGE_TEACHER_PRICE_OUT", DEFAULT_PRICE_OUT)
        )
        #: Running estimate of judge spend in USD, updated after every online call.
        self.spend_usd: float = 0.0
        self._client: Any | None = None

    # -- public API ---------------------------------------------------------

    def judge_pair(self, pair: dict[str, Any]) -> JudgeResult:
        """Score a single instruction/response pair on the rubric.

        Never raises: online failures (network/parse) are logged and reported as
        a conservative zero score so a misbehaving judge cannot smuggle low
        quality through a positive threshold.

        Args:
            pair: A dict with at least ``instruction`` and ``response`` keys
                (``context`` is used when present).

        Returns:
            A :class:`JudgeResult`.
        """
        instruction = str(pair.get("instruction", "")).strip()
        response = str(pair.get("response", "")).strip()
        if not instruction or not response:
            return JudgeResult(0.0, {axis: 0.0 for axis in JUDGE_AXES}, "empty instruction or response")

        if self.offline:
            return self._offline_result(instruction, response)

        context = str(pair.get("context", "")).strip()
        try:
            raw, usage = self._call_judge(instruction, response, context)
        except Exception as exc:  # network/parse errors must not crash a batch
            logger.error("Judge call failed: %s", exc)
            return JudgeResult(0.0, {axis: 0.0 for axis in JUDGE_AXES}, "judge call failed")

        self._update_spend(usage)
        return self._parse_result(raw)

    def filter_pairs(
        self, pairs: Iterable[dict[str, Any]], threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        """Keep only pairs whose overall judge score is ``>= threshold``.

        Args:
            pairs: Instruction/response pairs to score.
            threshold: Minimum overall score to keep a pair.

        Returns:
            The surviving pairs (copies), each annotated with a ``judge_score``.
        """
        kept: list[dict[str, Any]] = []
        for pair in pairs:
            result = self.judge_pair(pair)
            if result.score >= threshold:
                enriched = dict(pair)
                enriched["judge_score"] = result.score
                kept.append(enriched)
        return kept

    # -- spend --------------------------------------------------------------

    def _update_spend(self, usage: Any) -> float:
        """Update :pyattr:`spend_usd` from a judge ``usage`` payload."""
        if usage is None:
            return 0.0
        if isinstance(usage, dict):
            in_tok = int(usage.get("prompt_tokens", 0) or 0)
            out_tok = int(usage.get("completion_tokens", 0) or 0)
        else:
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = (in_tok / 1_000_000) * self.price_in + (out_tok / 1_000_000) * self.price_out
        self.spend_usd += cost
        logger.debug(
            "Judge call cost ~$%.5f (in=%d, out=%d); running spend ~$%.4f.",
            cost, in_tok, out_tok, self.spend_usd,
        )
        return cost

    # -- offline ------------------------------------------------------------

    def _offline_result(self, instruction: str, response: str) -> JudgeResult:
        """Deterministic pseudo-score seeded by the pair's content hash."""
        digest = hashlib.sha256(f"{instruction}\x00{response}".encode()).hexdigest()
        axes: dict[str, float] = {}
        for index, axis in enumerate(JUDGE_AXES):
            chunk = digest[index * 4 : index * 4 + 4] or "0000"
            axes[axis] = int(chunk, 16) / 0xFFFF
        score = sum(axes.values()) / len(axes)
        return JudgeResult(round(score, 6), axes, "[offline] deterministic pseudo-score")

    # -- online -------------------------------------------------------------

    def _client_handle(self) -> Any:
        """Lazily construct and cache the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - exercised only online
                raise RuntimeError(
                    "The 'openai' package is required for online judging. "
                    "Install it or use offline=True."
                ) from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_context_chars:
            return text
        return text[: self.max_context_chars] + "\n... [truncated]"

    def _build_messages(self, instruction: str, response: str, context: str) -> list[dict[str, str]]:
        """Build the rubric-scoring messages."""
        system = (
            "You are a strict but fair grader of Go instruction-tuning data. You score a "
            "single (instruction, response) pair on a rubric and return STRICT JSON only."
        )
        ctx = f"\n\n=== GROUNDING CONTEXT ===\n{self._truncate(context)}" if context else ""
        user = (
            "Grade the pair below on four axes, each a float in [0, 1]:\n"
            "- correctness: is the response technically correct?\n"
            "- idiomatic: is the Go idiomatic and following standard-library conventions?\n"
            "- completeness: does it fully address the instruction (no missing pieces)?\n"
            "- alignment: does the response actually answer THIS instruction?\n\n"
            "Return ONLY a JSON object of the form:\n"
            '{"correctness": 0.0, "idiomatic": 0.0, "completeness": 0.0, '
            '"alignment": 0.0, "rationale": "..."}\n\n'
            f"=== INSTRUCTION ===\n{self._truncate(instruction)}{ctx}\n\n"
            f"=== RESPONSE ===\n{self._truncate(response)}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _call_judge(self, instruction: str, response: str, context: str) -> tuple[str, Any]:
        """Call the judge model with retries and exponential backoff."""
        messages = self._build_messages(instruction, response, context)
        client = self._client_handle()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    response_format={"type": "json_object"},
                )
                content = completion.choices[0].message.content or ""
                usage = getattr(completion, "usage", None)
                return content, usage
            except Exception as exc:  # broad: SDK raises many error subtypes
                last_error = exc
                logger.warning("Judge attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    # -- parsing ------------------------------------------------------------

    @staticmethod
    def _parse_result(raw: str) -> JudgeResult:
        """Parse a judge response into a :class:`JudgeResult` (lenient)."""
        if not raw or not raw.strip():
            return JudgeResult(0.0, {axis: 0.0 for axis in JUDGE_AXES}, "empty judge output")
        candidate = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()

        data = _loads_lenient(candidate)
        if not isinstance(data, dict):
            return JudgeResult(0.0, {axis: 0.0 for axis in JUDGE_AXES}, "unparseable judge output")

        axes: dict[str, float] = {}
        for axis in JUDGE_AXES:
            axes[axis] = _clamp(data.get(axis))
        score = sum(axes.values()) / len(axes)
        rationale = str(data.get("rationale", "")).strip()
        return JudgeResult(round(score, 6), axes, rationale)


def _clamp(value: Any) -> float:
    """Coerce ``value`` to a float clamped into ``[0, 1]`` (missing -> 0)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:  # NaN
        return 0.0
    return max(0.0, min(1.0, number))


def judge_and_rank(
    pairs: Sequence[dict[str, Any]], judge: QualityJudge, threshold: float = 0.0
) -> list[dict[str, Any]]:
    """Convenience: score ``pairs``, drop below ``threshold``, sort best-first."""
    kept = judge.filter_pairs(pairs, threshold)
    kept.sort(key=lambda p: p.get("judge_score", 0.0), reverse=True)
    return kept
