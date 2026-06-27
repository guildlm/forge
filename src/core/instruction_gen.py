"""Teacher-model instruction generation.

Turns a piece of source content into supervised fine-tuning
``(instruction, response)`` pairs by prompting a *teacher* model exposed through
any OpenAI-compatible endpoint (vLLM, TGI, OpenAI, Together, ...).

Configuration is read from the environment:

* ``FORGE_TEACHER_BASE_URL`` -- OpenAI-compatible base URL.
* ``FORGE_TEACHER_API_KEY``  -- API key (any non-empty string for local servers).
* ``FORGE_TEACHER_MODEL``    -- Model identifier.

An :pyattr:`offline` mode produces deterministic synthetic pairs so that tests
and CI run without a network or the ``openai`` package installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "teacher"
MAX_CONTEXT_CHARS = 8_000


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
        "You are a senior Go engineer performing a rigorous code review. Identify "
        "bugs, race conditions, and idiom violations, and suggest concrete fixes.",
        "review the code and explain the issues found and how to fix them",
    ),
    Role(
        "go_generator",
        "You are an expert Go programmer. Given context, write idiomatic, correct, "
        "well-tested Go code that fulfils a clearly stated requirement.",
        "write idiomatic Go code that satisfies a requirement derived from the context",
    ),
    Role(
        "go_explainer",
        "You are a technical writer. Explain Go code clearly and concisely for an "
        "intermediate engineer, covering purpose, structure, and key mechanisms.",
        "explain what the code does and how its main components fit together",
    ),
    Role(
        "go_tester",
        "You are a Go testing expert. Write thorough, idiomatic table-driven tests "
        "(using the standard testing package) that cover happy paths, edge cases, "
        "and error conditions for the given code.",
        "write a comprehensive table-driven test suite for the code",
    ),
):
    register_role(_role)


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
            raw = self._call_teacher(context, role_obj, max_pairs)
        except Exception as exc:  # network/parse errors must not crash a batch
            logger.error("Teacher call failed for role %s: %s", role, exc)
            return []

        pairs = self._parse_pairs(raw)
        for pair in pairs:
            pair["context"] = context
        return pairs[:max_pairs]

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
        user_prompt = (
            f"Produce exactly {max_pairs} high-quality instruction/response pair(s) "
            f"that {role.task}.\n\n"
            "Return ONLY a JSON object of the form:\n"
            '{"pairs": [{"instruction": "...", "response": "..."}]}\n'
            "The instruction must be self-contained; the response must be complete "
            "and correct.\n\n"
            f"=== CONTEXT START ===\n{context}\n=== CONTEXT END ==="
        )
        return [
            {"role": "system", "content": role.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_teacher(self, context: str, role: Role, max_pairs: int) -> str:
        """Call the teacher model with retries and exponential backoff."""
        messages = self._build_messages(context, role, max_pairs)
        client = self._client_handle()
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    timeout=self.request_timeout,
                    response_format={"type": "json_object"},
                )
                return response.choices[0].message.content or ""
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
