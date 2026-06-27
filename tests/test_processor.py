"""Tests for the cleaning processor: dedup, PII, license, length, encoding."""

from __future__ import annotations

from src.core.processor import MinHasher, Processor, scrub_pii


def _doc(content: str, license: str | None = "MIT", doc_id: str = "x") -> dict:
    return {"id": doc_id, "content": content, "license": license}


def test_exact_dedup_removes_identical_documents() -> None:
    body = "package main\n\nfunc Add(a, b int) int { return a + b }\n" * 3
    docs = [_doc(body, doc_id="a"), _doc(body, doc_id="b")]
    proc = Processor(near_dup_threshold=1.1)  # disable near-dup so we isolate exact
    cleaned, stats = proc.clean(docs)
    assert len(cleaned) == 1
    assert stats.exact_duplicates == 1
    assert "content_hash" in cleaned[0]


def test_near_dedup_removes_similar_documents() -> None:
    base = " ".join(f"token{i}" for i in range(200))
    near = base + " token_extra_at_end token_extra_two"
    docs = [_doc(base, doc_id="a"), _doc(near, doc_id="b")]
    cleaned, stats = Processor(near_dup_threshold=0.8).clean(docs)
    assert len(cleaned) == 1
    assert stats.near_duplicates == 1


def test_pii_scrubbing_redacts_secrets() -> None:
    text = (
        "Contact me at john.doe@example.com using key sk-ABCDEFGHIJ1234567890XYZ "
        "and token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 plus AKIAIOSFODNN7EXAMPLE."
    )
    scrubbed, count = scrub_pii(text)
    assert count == 4
    assert "john.doe@example.com" not in scrubbed
    assert "sk-ABCDEFGHIJ" not in scrubbed
    assert "ghp_" not in scrubbed
    assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed


def test_pii_scrubbing_counts_in_stats() -> None:
    body = "func main(){}\n// reach me at dev@guildlm.ai\n" + "padding line\n" * 5
    cleaned, stats = Processor(near_dup_threshold=1.1).clean([_doc(body)])
    assert stats.pii_redactions == 1
    assert "dev@guildlm.ai" not in cleaned[0]["content"]


def test_license_allowlist_filters_non_permissive() -> None:
    body = "package main\n" + "x := 1\n" * 30
    docs = [_doc(body, license="GPL-3.0", doc_id="a"), _doc(body, license="MIT", doc_id="b")]
    cleaned, stats = Processor(near_dup_threshold=1.1).clean(docs)
    licenses = {d["license"] for d in cleaned}
    assert "GPL-3.0" not in licenses
    assert stats.license_filtered == 1


def test_unknown_license_can_be_rejected() -> None:
    body = "package main\n" + "x := 1\n" * 30
    cleaned, stats = Processor(allow_unknown_license=False).clean([_doc(body, license=None)])
    assert cleaned == []
    assert stats.license_filtered == 1


def test_length_filters() -> None:
    short = _doc("x", doc_id="s")
    long = _doc("y" * 5000, doc_id="l")
    cleaned, stats = Processor(min_length=10, max_length=1000, near_dup_threshold=1.1).clean([short, long])
    assert cleaned == []
    assert stats.length_filtered == 2


def test_encoding_validation_drops_binary_like() -> None:
    binary = "".join(chr(c) for c in range(0, 31)) * 50
    cleaned, stats = Processor().clean([_doc(binary)])
    assert cleaned == []
    assert stats.encoding_filtered == 1


def test_minhash_self_similarity_is_one() -> None:
    hasher = MinHasher()
    sig = hasher.signature("the quick brown fox jumps over the lazy dog repeatedly")
    assert MinHasher.similarity(sig, sig) == 1.0


def test_stats_total_in_out_consistent() -> None:
    body = "package main\n" + "stmt := value\n" * 40
    cleaned, stats = Processor(near_dup_threshold=1.1).clean([_doc(body, doc_id=str(i)) for i in range(1)])
    assert stats.total_in == 1
    assert stats.total_out == len(cleaned)
