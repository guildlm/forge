"""HuggingFace instruction-dataset source.

Unlike :mod:`src.sources.github` (which discovers *repositories* to clone and
process), this source curates *existing* open instruction datasets hosted on the
HuggingFace Hub and emits ready-to-build Forge pairs
(``{instruction, response, context}``) directly. It is the engine behind the
``forge import`` command and ``mode: import`` configs.

Two design constraints make it cheap and safe:

* **Streaming only.** Datasets such as ``nvidia/OpenCodeInstruct`` contain
  millions of rows; we open them with ``streaming=True`` and stop as soon as
  ``max_results`` matching rows have been collected, never materializing the
  full dataset.
* **Guarded dependency.** The heavy ``datasets`` library is imported lazily
  inside :meth:`HFDatasetsSource._load_stream`, so importing this module (and
  running the rest of the test suite) never requires it. Install it with
  ``pip install 'guildlm-forge[hf]'``.

Row normalization maps the many field-name conventions used across instruction
datasets onto Forge's pair schema, and a language filter keeps only rows whose
code matches the requested language (Go by default).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Iterator
from typing import Any

from src.sources.base import Source, SourceRecord

logger = logging.getLogger(__name__)

# Candidate field names, in priority order, for each part of a pair.
INSTRUCTION_KEYS: tuple[str, ...] = (
    "instruction", "problem", "prompt", "question", "query", "task",
)
RESPONSE_KEYS: tuple[str, ...] = (
    "response", "output", "solution", "completion", "answer", "code",
)
CONTEXT_KEYS: tuple[str, ...] = ("context", "input")

# Go detection heuristics used when a row carries no explicit language field.
_GO_FENCE = re.compile(r"```go\b", re.IGNORECASE)
_GO_MARKERS: tuple[str, ...] = ("package ", "func ", ":= ", "import (")
_GO_LANG_ALIASES: frozenset[str] = frozenset({"go", "golang"})


def _first(row: dict[str, Any], keys: Iterable[str]) -> str:
    """Return the first present, non-empty string value among ``keys``."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _from_messages(messages: list[Any]) -> tuple[str, str, str]:
    """Extract ``(instruction, response, context)`` from a chat ``messages`` list.

    The last user turn becomes the instruction, the last assistant turn the
    response, and any system turns are joined into the context.
    """
    instruction = ""
    response = ""
    system_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "system":
            system_parts.append(content.strip())
        elif role == "user":
            instruction = content.strip()
        elif role == "assistant":
            response = content.strip()
    return instruction, response, "\n".join(system_parts)


def normalize_row(row: dict[str, Any]) -> dict[str, str] | None:
    """Map a raw dataset row onto Forge's pair schema.

    Handles the common instruction-dataset conventions: ``{instruction,
    output|response}``, ``{problem, solution}`` (Magicoder OSS-Instruct),
    ``{prompt, completion}``, Alpaca-style ``{instruction, input, output}``, and
    chat ``{messages: [...]}``.

    Args:
        row: A single dataset row.

    Returns:
        A ``{instruction, response, context}`` dict, or ``None`` if the row lacks
        a usable instruction/response pair.
    """
    if not isinstance(row, dict):
        return None

    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        instruction, response, context = _from_messages(messages)
    else:
        instruction = _first(row, INSTRUCTION_KEYS)
        response = _first(row, RESPONSE_KEYS)
        context = _first(row, CONTEXT_KEYS)

    instruction = instruction.strip()
    response = response.strip()
    context = context.strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response, "context": context}


def is_language_row(row: dict[str, Any], pair: dict[str, str], language: str = "go") -> bool:
    """Decide whether ``pair`` is in the requested ``language``.

    Prefers an explicit ``lang``/``language`` field on the row; otherwise falls
    back to a heuristic. The heuristic is Go-specific (fenced ```go blocks or Go
    source markers), so non-Go languages are only accepted when the dataset
    provides an explicit language field.

    Args:
        row: The raw dataset row (may hold a ``lang``/``language`` field).
        pair: The normalized pair whose text is inspected by the heuristic.
        language: Target language (lowercased on comparison).

    Returns:
        ``True`` if the row should be kept.
    """
    target = language.strip().lower()
    declared = row.get("lang") or row.get("language")
    if isinstance(declared, str) and declared.strip():
        declared_norm = declared.strip().lower()
        if target == "go":
            return declared_norm in _GO_LANG_ALIASES
        return declared_norm == target

    if target != "go":
        return False

    text = f"{pair['instruction']}\n{pair['response']}\n{pair.get('context', '')}"
    if _GO_FENCE.search(text):
        return True
    return any(marker in text for marker in _GO_MARKERS)


class HFDatasetsSource(Source):
    """Curate an existing HuggingFace instruction dataset into Forge pairs.

    The dataset id is passed as the ``query`` argument to :meth:`search` (so it
    plugs into the generic discovery machinery), with ``split`` and ``language``
    supplied as keyword arguments.
    """

    name = "hf_datasets"

    def _load_stream(self, dataset: str, split: str) -> Iterator[dict[str, Any]]:
        """Open ``dataset`` in streaming mode and yield raw rows.

        Imports the ``datasets`` library lazily so the rest of Forge runs without
        it. Overridden in tests to avoid any network access.

        Raises:
            RuntimeError: If the ``datasets`` package is not installed.
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise RuntimeError(
                "The 'datasets' package is required for the hf_datasets source. "
                "Install it with: pip install 'guildlm-forge[hf]'."
            ) from exc
        logger.info("Streaming HF dataset %r (split=%r).", dataset, split)
        return iter(load_dataset(dataset, split=split, streaming=True))

    def search(
        self,
        query: str,
        max_results: int = 100,
        *,
        split: str = "train",
        language: str = "go",
        max_scan: int | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Stream ``query`` (a dataset id), normalize, language-filter, and collect.

        Args:
            query: HuggingFace dataset id (e.g. ``"ise-uiuc/Magicoder-OSS-Instruct-75K"``).
            max_results: Stop after this many matching pairs are collected.
            split: Dataset split to stream (default ``"train"``).
            language: Programming language to keep (default ``"go"``).
            max_scan: Optional safety cap on rows scanned before giving up.

        Returns:
            A list of ``{instruction, response, context}`` pair dictionaries.
        """
        pairs: list[dict[str, Any]] = []
        scanned = 0
        for row in self._load_stream(query, split):
            scanned += 1
            normalized = normalize_row(row)
            if normalized is not None and is_language_row(row, normalized, language):
                pairs.append(normalized)
                if len(pairs) >= max_results:
                    break
            if max_scan is not None and scanned >= max_scan:
                logger.info("Reached max_scan cap (%d rows); stopping.", max_scan)
                break
        logger.info(
            "Imported %d %s pair(s) from %r after scanning %d row(s).",
            len(pairs), language, query, scanned,
        )
        return pairs

    @staticmethod
    def describe(dataset: str, split: str = "train") -> dict[str, Any]:
        """Build a :class:`SourceRecord`-shaped descriptor for the dataset.

        Useful for provenance/manifest purposes; not part of the hot path.
        """
        return SourceRecord(
            source="hf_datasets",
            id=dataset,
            url=f"https://huggingface.co/datasets/{dataset}",
            title=dataset,
            extra={"split": split},
        ).to_dict()
