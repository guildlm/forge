"""Document extraction and cleaning.

The :class:`Processor` turns raw inputs (e.g. cloned repositories) into a clean
corpus of documents ready for instruction generation. Cleaning is fully
dependency-free and includes:

* exact deduplication via SHA-256 content hashing,
* near-duplicate detection via pure-Python MinHash over token shingles,
* permissive-license allowlist filtering (SPDX),
* PII / secret scrubbing (emails, keys, tokens, private-key blocks),
* minimum / maximum length filters,
* text encoding / printability validation.

Every run returns rich :class:`CleaningStats`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)

# SPDX identifiers we consider safe for redistribution / model training.
PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "MIT",
        "MIT-0",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "BSD-3-Clause-Clear",
        "ISC",
        "0BSD",
        "Unlicense",
        "Zlib",
        "MPL-2.0",
        "BSL-1.0",
        "PostgreSQL",
        "CC0-1.0",
        "WTFPL",
    }
)

# Compiled PII / secret patterns. Order matters: more specific first.
PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "private_key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----", re.DOTALL),
        "[REDACTED_PRIVATE_KEY]",
    ),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "[REDACTED_GH_TOKEN]"),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "[REDACTED_API_KEY]"),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "[REDACTED_SLACK_TOKEN]"),
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
]


@dataclass
class CleaningStats:
    """Counts describing what a cleaning pass did."""

    total_in: int = 0
    total_out: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    license_filtered: int = 0
    length_filtered: int = 0
    encoding_filtered: int = 0
    pii_redactions: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a plain dictionary of the counters."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# PII scrubbing
# --------------------------------------------------------------------------- #
def scrub_pii(text: str) -> tuple[str, int]:
    """Redact emails, API keys, tokens and private-key blocks from ``text``.

    Returns:
        ``(scrubbed_text, redaction_count)``.
    """
    count = 0
    for _name, pattern, replacement in PII_PATTERNS:
        text, n = pattern.subn(replacement, text)
        count += n
    return text, count


# --------------------------------------------------------------------------- #
# Near-duplicate detection (pure-Python MinHash)
# --------------------------------------------------------------------------- #
class MinHasher:
    """Deterministic MinHash signatures over k-shingles, no external deps."""

    _MERSENNE = (1 << 61) - 1  # large prime modulus

    def __init__(self, num_perm: int = 64, shingle_size: int = 5, seed: int = 1) -> None:
        """Args:
        num_perm: Number of permutations (signature length). More = more precise.
        shingle_size: Number of consecutive tokens per shingle.
        seed: Seed for the hash-coefficient generator (reproducibility).
        """
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = _LCG(seed)
        self._a = [rng.next_nonzero() for _ in range(num_perm)]
        self._b = [rng.next() for _ in range(num_perm)]

    def _shingles(self, text: str) -> set[int]:
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) < self.shingle_size:
            grams = {" ".join(tokens)} if tokens else set()
        else:
            grams = {
                " ".join(tokens[i : i + self.shingle_size])
                for i in range(len(tokens) - self.shingle_size + 1)
            }
        return {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams}

    def signature(self, text: str) -> tuple[int, ...]:
        """Compute the MinHash signature of ``text``."""
        shingles = self._shingles(text)
        if not shingles:
            return tuple([0] * self.num_perm)
        sig = []
        for a, b in zip(self._a, self._b, strict=True):
            sig.append(min(((a * h + b) % self._MERSENNE) for h in shingles))
        return tuple(sig)

    @staticmethod
    def similarity(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> float:
        """Estimated Jaccard similarity from two equal-length signatures."""
        if not sig_a:
            return 0.0
        equal = sum(1 for x, y in zip(sig_a, sig_b, strict=True) if x == y)
        return equal / len(sig_a)


class _LCG:
    """Tiny deterministic linear-congruential generator for hash coefficients."""

    def __init__(self, seed: int) -> None:
        self._state = (seed & 0xFFFFFFFF) or 1

    def next(self) -> int:
        self._state = (1103515245 * self._state + 12345) & 0x7FFFFFFF
        return self._state

    def next_nonzero(self) -> int:
        value = self.next()
        return value or 1


# --------------------------------------------------------------------------- #
# Processor
# --------------------------------------------------------------------------- #
class Processor:
    """Extract documents from repositories and clean a document corpus."""

    DEFAULT_EXCLUDES = [
        "vendor/", "node_modules/", ".git/", "mocks/", "testdata/",
        "_test.go", "mock_", ".pb.go", "_string.go",
    ]

    def __init__(
        self,
        include_extensions: Iterable[str] | None = None,
        exclude_patterns: Iterable[str] | None = None,
        *,
        min_length: int = 50,
        max_length: int = 100_000,
        license_allowlist: Iterable[str] | None = None,
        allow_unknown_license: bool = True,
        near_dup_threshold: float = 0.85,
        scrub: bool = True,
        min_printable_ratio: float = 0.85,
    ) -> None:
        """Configure extraction and cleaning behaviour.

        Args:
            include_extensions: File extensions to extract (default ``['.go']``).
            exclude_patterns: Path substrings to skip during extraction.
            min_length: Minimum document length in characters.
            max_length: Maximum document length in characters.
            license_allowlist: Permitted SPDX ids (default permissive set).
            allow_unknown_license: Keep documents with no/unknown license.
            near_dup_threshold: MinHash similarity above which docs are dropped.
            scrub: Whether to run PII/secret scrubbing.
            min_printable_ratio: Minimum ratio of printable characters required.
        """
        self.include_extensions = list(include_extensions or [".go"])
        self.exclude_patterns = list(exclude_patterns or self.DEFAULT_EXCLUDES)
        self.min_length = min_length
        self.max_length = max_length
        self.license_allowlist = frozenset(license_allowlist or PERMISSIVE_LICENSES)
        self.allow_unknown_license = allow_unknown_license
        self.near_dup_threshold = near_dup_threshold
        self.scrub = scrub
        self.min_printable_ratio = min_printable_ratio

    # -- extraction ---------------------------------------------------------

    def _should_process(self, rel_path: str) -> bool:
        norm = rel_path.replace("\\", "/")
        if not any(norm.endswith(ext) for ext in self.include_extensions):
            return False
        return not any(pat in norm for pat in self.exclude_patterns)

    def process_repository(self, repo_path: str, license: str | None = None) -> Iterator[dict[str, Any]]:
        """Walk a cloned repository and yield raw document dictionaries.

        Args:
            repo_path: Path to the local repository checkout.
            license: SPDX license to attach to every document (from discovery).

        Yields:
            Documents with ``id``, ``repo``, ``file_path``, ``content`` and
            ``license`` keys.
        """
        if not os.path.isdir(repo_path):
            logger.warning("Repository path does not exist: %s", repo_path)
            return
        repo_name = os.path.basename(repo_path.rstrip("/"))
        logger.info("Extracting documents from %s.", repo_name)

        for root, _dirs, files in os.walk(repo_path):
            for file in files:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, repo_path)
                if not self._should_process(rel_path):
                    continue
                try:
                    with open(filepath, encoding="utf-8") as handle:
                        content = handle.read()
                except (UnicodeDecodeError, OSError):
                    continue
                if not content.strip():
                    continue
                yield {
                    "id": f"{repo_name}:{rel_path}",
                    "repo": repo_name,
                    "file_path": rel_path,
                    "content": content,
                    "license": license,
                }

    # -- cleaning -----------------------------------------------------------

    @staticmethod
    def _printable_ratio(text: str) -> float:
        if not text:
            return 0.0
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
        return printable / len(text)

    def clean(self, documents: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], CleaningStats]:
        """Run the full cleaning pipeline over ``documents``.

        Each document must contain a ``content`` key; ``license`` is optional.

        Returns:
            ``(cleaned_documents, stats)``. Cleaned documents gain a
            ``content_hash`` field.
        """
        stats = CleaningStats()
        hasher = MinHasher()
        seen_hashes: set[str] = set()
        kept_signatures: list[tuple[int, ...]] = []
        out: list[dict[str, Any]] = []

        for doc in documents:
            stats.total_in += 1
            content = doc.get("content", "")

            # 1. Encoding / printability validation.
            if self._printable_ratio(content) < self.min_printable_ratio:
                stats.encoding_filtered += 1
                continue

            # 2. License allowlist.
            license_id = doc.get("license")
            if license_id is None:
                if not self.allow_unknown_license:
                    stats.license_filtered += 1
                    continue
            elif license_id not in self.license_allowlist:
                stats.license_filtered += 1
                continue

            # 3. PII / secret scrubbing.
            if self.scrub:
                content, redactions = scrub_pii(content)
                stats.pii_redactions += redactions

            # 4. Length filters (after scrubbing).
            length = len(content)
            if length < self.min_length or length > self.max_length:
                stats.length_filtered += 1
                continue

            # 5. Exact dedup.
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                stats.exact_duplicates += 1
                continue
            seen_hashes.add(content_hash)

            # 6. Near dedup.
            signature = hasher.signature(content)
            if any(
                MinHasher.similarity(signature, kept) >= self.near_dup_threshold
                for kept in kept_signatures
            ):
                stats.near_duplicates += 1
                continue
            kept_signatures.append(signature)

            cleaned = dict(doc)
            cleaned["content"] = content
            cleaned["content_hash"] = content_hash
            out.append(cleaned)

        stats.total_out = len(out)
        logger.info("Cleaning complete: %s", stats.to_dict())
        return out, stats
