"""Final dataset assembly.

Takes generated instruction pairs, validates them against the SFT schema, splits
into train/validation sets, and exports to JSONL (always) and Parquet (optional,
guarded behind ``pyarrow``). A manifest with per-file content hashes and stats is
written alongside the data.

The resulting dataset is what the downstream training stage (Anvil) consumes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = "You are a GuildLM specialist."

# Fields present on every exported record.
SCHEMA_FIELDS = ("instruction", "response", "context", "messages")


class SchemaError(ValueError):
    """Raised when an instruction pair fails schema validation."""


@dataclass
class BuildManifest:
    """Metadata describing a built dataset."""

    name: str
    created_at: str
    total_records: int
    splits: dict[str, int]
    files: list[dict[str, Any]] = field(default_factory=list)
    schema_fields: list[str] = field(default_factory=lambda: list(SCHEMA_FIELDS))
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "created_at": self.created_at,
            "total_records": self.total_records,
            "splits": self.splits,
            "files": self.files,
            "schema_fields": self.schema_fields,
            "stats": self.stats,
        }


class DatasetBuilder:
    """Validate, split, and export instruction pairs as a training dataset."""

    def __init__(self, output_dir: str = "data/datasets", system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        """Args:
        output_dir: Directory where dataset files and the manifest are written.
        system_prompt: System message embedded in each record's chat transcript.
        """
        self.output_dir = output_dir
        self.system_prompt = system_prompt
        os.makedirs(self.output_dir, exist_ok=True)

    # -- normalization / validation ----------------------------------------

    def normalize(self, pair: dict[str, Any]) -> dict[str, Any]:
        """Validate a raw pair and expand it into the exported record schema.

        Raises:
            SchemaError: If ``instruction`` or ``response`` is missing/empty.
        """
        instruction = str(pair.get("instruction", "")).strip()
        response = str(pair.get("response", "")).strip()
        if not instruction:
            raise SchemaError("record is missing a non-empty 'instruction'")
        if not response:
            raise SchemaError("record is missing a non-empty 'response'")
        context = str(pair.get("context", "")).strip()

        user_content = instruction if not context else f"{instruction}\n\n{context}"
        return {
            "instruction": instruction,
            "response": response,
            "context": context,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ],
        }

    # -- splitting ----------------------------------------------------------

    @staticmethod
    def split(
        records: list[dict[str, Any]], val_ratio: float, seed: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Deterministically split records into ``(train, val)``."""
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be in [0, 1)")
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        n_val = int(len(shuffled) * val_ratio)
        return shuffled[n_val:], shuffled[:n_val]

    # -- export -------------------------------------------------------------

    def build(
        self,
        pairs: Iterable[dict[str, Any]],
        name: str,
        *,
        val_ratio: float = 0.1,
        seed: int = 42,
        formats: Iterable[str] = ("jsonl",),
        source_stats: dict[str, Any] | None = None,
    ) -> BuildManifest:
        """Build and export the dataset.

        Args:
            pairs: Raw instruction pairs.
            name: Dataset name; used as a filename prefix.
            val_ratio: Fraction reserved for the validation split.
            seed: Seed for the deterministic split.
            formats: Any of ``"jsonl"`` and ``"parquet"``.
            source_stats: Optional upstream stats embedded in the manifest.

        Returns:
            The :class:`BuildManifest` describing the export (also written to disk).
        """
        records = [self.normalize(p) for p in pairs]
        train, val = self.split(records, val_ratio, seed)
        splits = {"train": train, "validation": val}
        formats = list(formats)

        manifest = BuildManifest(
            name=name,
            created_at=datetime.now(UTC).isoformat(),
            total_records=len(records),
            splits={k: len(v) for k, v in splits.items()},
            stats=source_stats or {},
        )

        for split_name, split_records in splits.items():
            if not split_records and split_name == "validation":
                continue
            if "jsonl" in formats:
                path = os.path.join(self.output_dir, f"{name}.{split_name}.jsonl")
                self._write_jsonl(split_records, path)
                manifest.files.append(self._file_entry(path, split_name, "jsonl", len(split_records)))
            if "parquet" in formats:
                path = os.path.join(self.output_dir, f"{name}.{split_name}.parquet")
                self._write_parquet(split_records, path)
                manifest.files.append(self._file_entry(path, split_name, "parquet", len(split_records)))

        manifest_path = os.path.join(self.output_dir, f"{name}.manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest.to_dict(), handle, indent=2)
        logger.info("Built dataset %r: %d records -> %s", name, len(records), self.output_dir)
        return manifest

    def export_to_jsonl(self, pairs: Iterable[dict[str, Any]], filename: str) -> str:
        """Export all pairs (no split) to a single JSONL file. Returns its path."""
        records = [self.normalize(p) for p in pairs]
        path = os.path.join(self.output_dir, filename)
        self._write_jsonl(records, path)
        return os.path.abspath(path)

    @staticmethod
    def load_jsonl(path: str) -> list[dict[str, Any]]:
        """Read a JSONL file back into a list of records."""
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    # -- internal writers ---------------------------------------------------

    @staticmethod
    def _write_jsonl(records: list[dict[str, Any]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_parquet(records: list[dict[str, Any]], path: str) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise RuntimeError(
                "Parquet export requires pyarrow. Install with: pip install 'guildlm-forge[parquet]'."
            ) from exc
        columns = {
            "instruction": [r["instruction"] for r in records],
            "response": [r["response"] for r in records],
            "context": [r["context"] for r in records],
            "messages": [json.dumps(r["messages"], ensure_ascii=False) for r in records],
        }
        pq.write_table(pa.table(columns), path)

    @staticmethod
    def _file_entry(path: str, split: str, fmt: str, count: int) -> dict[str, Any]:
        sha256 = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                sha256.update(chunk)
        return {
            "path": os.path.basename(path),
            "split": split,
            "format": fmt,
            "records": count,
            "bytes": os.path.getsize(path),
            "sha256": sha256.hexdigest(),
        }
