"""Teacher-model instruction generation.

Turns a piece of source content into supervised fine-tuning
``(instruction, response)`` pairs by prompting a *teacher* model exposed through
any OpenAI-compatible endpoint (vLLM, TGI, OpenAI, Together, ...).

Configuration is read from the environment:

* ``FORGE_TEACHER_BASE_URL`` -- OpenAI-compatible base URL.
* ``FORGE_TEACHER_API_KEY``  -- API key (any non-empty string for local servers).
* ``FORGE_TEACHER_MODEL``    -- Model identifier.
* ``FORGE_TEACHER_PRICE_IN`` -- USD per 1M input tokens (cost estimation).
* ``FORGE_TEACHER_PRICE_OUT``-- USD per 1M output tokens (cost estimation).

An :pyattr:`offline` mode produces deterministic synthetic pairs so that tests
and CI run without a network or the ``openai`` package installed.

For cost-controlled bulk generation use :meth:`InstructionGenerator.generate_dataset`,
which honours hard ``max_pairs`` and ``max_spend_usd`` caps and tracks a running
spend estimate derived from the teacher's reported token usage.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from src.core.judge import QualityJudge
    from src.core.verifier import GoVerifier

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "teacher"
MAX_CONTEXT_CHARS = 8_000

# Default per-token prices (USD per 1M tokens). Defaults track DeepSeek-V3
# (``deepseek-chat``), a cheap and strong OpenAI-compatible teacher.
DEFAULT_PRICE_IN = 0.14
DEFAULT_PRICE_OUT = 0.28


# --------------------------------------------------------------------------- #
# Role registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Role:
    """A teacher persona used to steer instruction generation.

    Attributes:
        name: Registry key (e.g. ``"go_reviewer"``).
        system_prompt: System message sent to the teacher model.
        task: Short description of the instruction style, also used to seed the
            deterministic offline output.
    """

    name: str
    system_prompt: str
    task: str


ROLE_REGISTRY: dict[str, Role] = {}


def register_role(role: Role) -> None:
    """Register (or overwrite) a role in the global registry."""
    ROLE_REGISTRY[role.name] = role


def get_role(name: str) -> Role:
    """Look up a role by name.

    Raises:
        ValueError: If the role is not registered.
    """
    try:
        return ROLE_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(ROLE_REGISTRY))
        raise ValueError(f"Unknown role {name!r}. Available: {available}") from None


for _role in (
    Role(
        "go_reviewer",
        "You are a senior Go engineer performing a rigorous code review. You find "
        "bugs, race conditions, nil-pointer and error-handling mistakes, and idiom "
        "violations, and you always propose concrete, idiomatic fixes with corrected "
        "code. You write strict JSON only.",
        "invent a realistic, self-contained code-review request whose instruction "
        "embeds a concrete Go snippet to review, then give a thorough review that "
        "lists specific issues and shows the idiomatic fix for each",
    ),
    Role(
        "go_generator",
        "You are an expert Go programmer. You write idiomatic, correct, compilable Go "
        "that follows the standard library's conventions, handles errors explicitly, "
        "and avoids needless allocation. You write strict JSON only.",
        "invent a realistic, self-contained Go programming problem grounded in the "
        "snippet, then provide a complete, idiomatic Go solution with a short "
        "explanation of the approach",
    ),
    Role(
        "go_explainer",
        "You are a technical writer who explains Go code clearly and precisely for an "
        "intermediate engineer, covering purpose, structure, control flow, and key "
        "mechanisms (goroutines, channels, interfaces, error handling). You write "
        "strict JSON only.",
        "invent a self-contained request to explain a Go snippet (embedding the code "
        "in the instruction), then give a clear, well-structured explanation of what "
        "it does and how its main components fit together",
    ),
    Role(
        "go_tester",
        "You are a Go testing expert. You write thorough, idiomatic table-driven tests "
        "with the standard testing package, covering happy paths, edge cases, and "
        "error conditions, using subtests and t.Run. You write strict JSON only.",
        "invent a self-contained request to test a Go function or component grounded "
        "in the snippet, then provide a comprehensive table-driven test suite using "
        "the standard testing package",
    ),
):
    register_role(_role)


# --------------------------------------------------------------------------- #
# Quality-gate bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class GateStats:
    """Counters describing what the quality gate did during generation."""

    generated: int = 0
    verified_dropped: int = 0
    judged_dropped: int = 0
    kept: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a plain dictionary of the counters."""
        return {
            "generated": self.generated,
            "verified_dropped": self.verified_dropped,
            "judged_dropped": self.judged_dropped,
            "kept": self.kept,
        }


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class InstructionGenerator:
    """Generate SFT instruction/response pairs from source content."""

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
        temperature: float = 0.4,
        request_timeout: float = 120.0,
        price_in: float | None = None,
        price_out: float | None = None,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
    ) -> None:
        """Initialize the generator.

        Args:
            base_url: OpenAI-compatible base URL. Falls back to
                ``FORGE_TEACHER_BASE_URL`` then a local default.
            api_key: API key. Falls back to ``FORGE_TEACHER_API_KEY``.
            model: Model id. Falls back to ``FORGE_TEACHER_MODEL`` then a default.
            offline: When ``True``, generate deterministic synthetic pairs without
                any network call or ``openai`` dependency.
            max_context_chars: Truncate context to this many characters.
            max_retries: Maximum teacher-call attempts before giving up.
            backoff_base: Base seconds for exponential backoff between retries.
            temperature: Sampling temperature for the teacher model.
            request_timeout: Per-request timeout in seconds.
            price_in: USD per 1M input tokens for cost estimation. Falls back to
                ``FORGE_TEACHER_PRICE_IN`` then :data:`DEFAULT_PRICE_IN`.
            price_out: USD per 1M output tokens. Falls back to
                ``FORGE_TEACHER_PRICE_OUT`` then :data:`DEFAULT_PRICE_OUT`.
        """
        self.base_url = base_url or os.environ.get("FORGE_TEACHER_BASE_URL", DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("FORGE_TEACHER_API_KEY", "not-needed")
        self.model = model or os.environ.get("FORGE_TEACHER_MODEL", DEFAULT_MODEL)
        self.offline = offline
        self.max_context_chars = max_context_chars
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.temperature = temperature
        self.request_timeout = request_timeout
        self.price_in = (
            price_in if price_in is not None
            else float(os.environ.get("FORGE_TEACHER_PRICE_IN", DEFAULT_PRICE_IN))
        )
        self.price_out = (
            price_out if price_out is not None
            else float(os.environ.get("FORGE_TEACHER_PRICE_OUT", DEFAULT_PRICE_OUT))
        )
        self.max_tokens = max_tokens
        # Reasoning models (e.g. gpt-oss) otherwise spend their whole budget on
        # hidden reasoning and return empty content; "low" keeps answers compact.
        self.reasoning_effort = reasoning_effort or os.environ.get("FORGE_TEACHER_REASONING_EFFORT")
        #: Running estimate of teacher spend in USD, updated after every online call.
        self.spend_usd: float = 0.0
        self._client: Any | None = None

    # -- public API ---------------------------------------------------------

    def generate_pairs(
        self, content: str, role: str = "go_explainer", max_pairs: int = 1
    ) -> list[dict[str, str]]:
        """Generate up to ``max_pairs`` instruction/response pairs from ``content``.

        Args:
            content: Source content to ground the instructions in.
            role: Registered teacher role name.
            max_pairs: Number of pairs to request.

        Returns:
            A list of dicts with ``instruction``, ``response`` and ``context`` keys.
            Always returns a list (possibly empty); failures are logged, not raised.
        """
        role_obj = get_role(role)
        context = self._truncate(content)
        if not context.strip():
            return []

        if self.offline:
            return self._synthetic_pairs(context, role_obj, max_pairs)

        try:
            raw, usage = self._call_teacher(context, role_obj, max_pairs)
        except Exception as exc:  # network/parse errors must not crash a batch
            logger.error("Teacher call failed for role %s: %s", role, exc)
            return []

        self._update_spend(usage)
        pairs = self._parse_pairs(raw)
        for pair in pairs:
            pair["context"] = context
        return pairs[:max_pairs]

    def generate_dataset(
        self,
        documents: Iterable[dict[str, Any]],
        *,
        roles: Sequence[str],
        max_pairs_per_doc: int = 1,
        max_pairs: int | None = None,
        max_spend_usd: float | None = None,
        content_key: str = "content",
        verifier: GoVerifier | None = None,
        judge: QualityJudge | None = None,
        judge_threshold: float = 0.0,
        rejection_samples: int = 1,
    ) -> list[dict[str, str]]:
        """Generate pairs across many documents under hard budget caps and a quality gate.

        Iterates ``documents`` x ``roles``, accumulating instruction/response
        pairs. The loop stops *cleanly* (returning whatever was produced so far,
        never raising) as soon as either cap is reached:

        * ``max_pairs`` -- stop once this many pairs have been collected.
        * ``max_spend_usd`` -- stop once the running spend estimate reaches this.
          When a ``judge`` is supplied, **its** spend counts toward the cap too.

        When any of ``verifier`` / ``judge`` is set or ``rejection_samples > 1``,
        the *quality gate* is active. For each keep-slot per (document, role) it
        samples ``rejection_samples`` candidates and keeps the best surviving one:

        * **verify gate** -- candidates whose code fails execution verification
          (per the verifier's ``strict`` policy) are dropped.
        * **judge gate** -- candidates scoring below ``judge_threshold`` are
          dropped; survivors are ranked by judge score (best wins).

        With no gate active this is exactly the original budget-capped behaviour.

        Args:
            documents: Iterable of document dicts (or raw strings).
            roles: Teacher role names to apply to each document.
            max_pairs_per_doc: Pairs (keep-slots) per (document, role).
            max_pairs: Optional hard cap on total pairs.
            max_spend_usd: Optional hard cap on estimated spend (USD).
            content_key: Key holding the source content when documents are dicts.
            verifier: Optional :class:`~src.core.verifier.GoVerifier` gate.
            judge: Optional :class:`~src.core.judge.QualityJudge` gate.
            judge_threshold: Minimum judge score to keep a candidate.
            rejection_samples: Candidates sampled per keep-slot (best survivor wins).

        Returns:
            The collected instruction/response/context pairs.
        """
        gate_active = verifier is not None or judge is not None or rejection_samples > 1
        stats = GateStats()
        pairs: list[dict[str, str]] = []

        for doc in documents:
            content = doc.get(content_key, "") if isinstance(doc, dict) else str(doc)
            for role in roles:
                if max_pairs is not None and len(pairs) >= max_pairs:
                    logger.info(
                        "Reached max_pairs cap (%d); stopping. Estimated spend ~$%.4f.",
                        max_pairs, self._total_spend(judge),
                    )
                    self._log_gate(stats, len(pairs), judge)
                    return pairs
                if max_spend_usd is not None and self._total_spend(judge) >= max_spend_usd:
                    logger.info(
                        "Reached max_spend_usd cap ($%.2f); stopping. Estimated spend ~$%.4f.",
                        max_spend_usd, self._total_spend(judge),
                    )
                    self._log_gate(stats, len(pairs), judge)
                    return pairs

                if gate_active:
                    produced = self._gated_pairs(
                        content, role,
                        n_keep=max_pairs_per_doc,
                        rejection_samples=max(1, rejection_samples),
                        verifier=verifier,
                        judge=judge,
                        judge_threshold=judge_threshold,
                        max_spend_usd=max_spend_usd,
                        stats=stats,
                    )
                else:
                    produced = self.generate_pairs(content, role=role, max_pairs=max_pairs_per_doc)
                    stats.generated += len(produced)

                for pair in produced:
                    pairs.append(pair)
                    if max_pairs is not None and len(pairs) >= max_pairs:
                        break

        self._log_gate(stats, len(pairs), judge)
        return pairs

    def _total_spend(self, judge: QualityJudge | None) -> float:
        """Combined teacher + judge spend estimate in USD."""
        return self.spend_usd + (judge.spend_usd if judge is not None else 0.0)

    def _log_gate(self, stats: GateStats, kept: int, judge: QualityJudge | None) -> None:
        stats.kept = kept
        logger.info(
            "Quality gate: generated=%d verified_dropped=%d judged_dropped=%d kept=%d; "
            "total estimated spend ~$%.4f.",
            stats.generated, stats.verified_dropped, stats.judged_dropped, kept,
            self._total_spend(judge),
        )

    def _gated_pairs(
        self,
        content: str,
        role: str,
        *,
        n_keep: int,
        rejection_samples: int,
        verifier: GoVerifier | None,
        judge: QualityJudge | None,
        judge_threshold: float,
        max_spend_usd: float | None,
        stats: GateStats,
    ) -> list[dict[str, str]]:
        """Produce up to ``n_keep`` gated pairs for one (content, role).

        For each keep-slot, sample ``rejection_samples`` candidates, drop those
        that fail verification or fall below the judge threshold, and keep the
        best survivor (highest judge score, else the first survivor).
        """
        kept: list[dict[str, str]] = []
        for _slot in range(max(1, n_keep)):
            survivors: list[dict[str, str]] = []
            for _sample in range(rejection_samples):
                if max_spend_usd is not None and self._total_spend(judge) >= max_spend_usd:
                    break
                candidates = self.generate_pairs(content, role=role, max_pairs=1)
                if not candidates:
                    continue
                candidate = candidates[0]
                stats.generated += 1

                if verifier is not None:
                    result = verifier.verify(candidate.get("response", ""), role=role)
                    if not result.passed:
                        stats.verified_dropped += 1
                        continue
                    candidate["verify_status"] = result.status

                if judge is not None:
                    verdict = judge.judge_pair(candidate)
                    candidate["judge_score"] = verdict.score
                    if verdict.score < judge_threshold:
                        stats.judged_dropped += 1
                        continue

                survivors.append(candidate)

            if not survivors:
                continue
            if judge is not None:
                best = max(survivors, key=lambda p: float(p.get("judge_score", 0.0)))
            else:
                best = survivors[0]
            kept.append(best)
        return kept

    def _update_spend(self, usage: Any) -> float:
        """Update :pyattr:`spend_usd` from a teacher ``usage`` payload.

        Args:
            usage: An OpenAI ``usage`` object or dict with ``prompt_tokens`` /
                ``completion_tokens``. ``None`` is treated as zero cost.

        Returns:
            The estimated cost (USD) of this single call.
        """
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
            "Teacher call cost ~$%.5f (in=%d, out=%d); running spend ~$%.4f.",
            cost, in_tok, out_tok, self.spend_usd,
        )
        return cost

    # -- offline ------------------------------------------------------------

    def _synthetic_pairs(self, context: str, role: Role, max_pairs: int) -> list[dict[str, str]]:
        """Build deterministic synthetic pairs (seeded by content hash)."""
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        head = context.strip().splitlines()[0][:80] if context.strip() else ""
        pairs: list[dict[str, str]] = []
        for index in range(max_pairs):
            seed = digest[index * 4 : index * 4 + 8] or digest[:8]
            instruction = (
                f"As a {role.name.replace('_', ' ')}, {role.task}. "
                f"Focus on the snippet starting with: {head!r}."
            )
            response = (
                f"[offline:{role.name}:{seed}] This deterministic placeholder stands in "
                f"for a teacher-generated answer that would {role.task}. "
                f"It references {len(context)} characters of context."
            )
            pairs.append({"instruction": instruction, "response": response, "context": context})
        return pairs

    # -- online -------------------------------------------------------------

    def _client_handle(self) -> Any:
        """Lazily construct and cache the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - exercised only online
                raise RuntimeError(
                    "The 'openai' package is required for online generation. "
                    "Install it or use offline=True."
                ) from exc
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _build_messages(self, context: str, role: Role, max_pairs: int) -> list[dict[str, str]]:
        """Build OSS-Instruct-style messages grounding the task in a real snippet."""
        user_prompt = (
            f"Below is a REAL Go code snippet drawn from an open-source project. Use it "
            f"as inspiration to {role.task}.\n\n"
            f"Produce exactly {max_pairs} high-quality instruction/response pair(s). "
            "Each instruction MUST be fully self-contained: a reader who has never seen "
            "the snippet below must be able to act on it, so embed any code the task "
            "refers to directly inside the instruction. The response must be complete, "
            "correct, and idiomatic Go.\n\n"
            "Return ONLY a JSON object of the form:\n"
            '{"pairs": [{"instruction": "...", "response": "..."}]}\n\n'
            f"=== GO SNIPPET START ===\n{context}\n=== GO SNIPPET END ==="
        )
        return [
            {"role": "system", "content": role.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_teacher(self, context: str, role: Role, max_pairs: int) -> tuple[str, Any]:
        """Call the teacher model with retries and exponential backoff.

        Returns:
            A ``(content, usage)`` tuple where ``usage`` is the SDK usage object
            (or ``None`` if the endpoint did not report token counts).
        """
        messages = self._build_messages(context, role, max_pairs)
        client = self._client_handle()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    **extra,
                )
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                return content, usage
            except Exception as exc:  # broad: SDK raises many error subtypes
                last_error = exc
                logger.warning(
                    "Teacher attempt %d/%d failed: %s", attempt, self.max_retries, exc
                )
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)))
        assert last_error is not None
        raise last_error

    # -- parsing / helpers --------------------------------------------------

    def _truncate(self, content: str) -> str:
        """Truncate content to the configured context budget on a line boundary."""
        if len(content) <= self.max_context_chars:
            return content
        truncated = content[: self.max_context_chars]
        newline = truncated.rfind("\n")
        if newline > self.max_context_chars // 2:
            truncated = truncated[:newline]
        return truncated + "\n... [truncated]"

    @staticmethod
    def _parse_pairs(raw: str) -> list[dict[str, str]]:
        """Robustly parse teacher output into a list of instruction/response dicts.

        Tolerates markdown code fences, surrounding prose, and either a top-level
        list or a ``{"pairs": [...]}`` wrapper.
        """
        if not raw or not raw.strip():
            return []
        candidate = raw.strip()

        fence = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()

        data = _loads_lenient(candidate)
        if data is None:
            return []

        items = data.get("pairs", data) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []

        pairs: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            instruction = str(item.get("instruction", "")).strip()
            response = str(item.get("response", "")).strip()
            if instruction and response:
                pairs.append({"instruction": instruction, "response": response})
        return pairs


def _loads_lenient(text: str) -> Any | None:
    """Parse JSON, falling back to the first balanced ``{...}`` / ``[...]`` block."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None
