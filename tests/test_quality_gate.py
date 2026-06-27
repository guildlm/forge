"""Tests for the wired quality gate inside generate_dataset.

Uses stub teacher / verifier / judge so everything runs offline with no network
and no Go toolchain.
"""

from __future__ import annotations

import re

from src.core.instruction_gen import InstructionGenerator
from src.core.judge import JudgeResult
from src.core.verifier import VerifyResult


class StubVerifier:
    """Pass responses for which ``predicate(response)`` is true."""

    def __init__(self, predicate) -> None:  # noqa: ANN001 - test stub
        self.predicate = predicate

    def verify(self, response: str, role: str = "") -> VerifyResult:
        ok = self.predicate(response)
        return VerifyResult(ok, "ok" if ok else "build", "ok" if ok else "failed", "")


class StubJudgeByText:
    """Score a pair by reading ``score=<float>`` out of its response text."""

    def __init__(self, cost: float = 0.0) -> None:
        self.spend_usd = 0.0
        self.cost = cost

    def judge_pair(self, pair: dict) -> JudgeResult:
        self.spend_usd += self.cost
        match = re.search(r"score=([0-9.]+)", str(pair.get("response", "")))
        score = float(match.group(1)) if match else 0.0
        return JudgeResult(score, {}, "stub")


def _cycle_teacher(gen: InstructionGenerator, responses, monkeypatch) -> None:
    """Make ``gen.generate_pairs`` return ``responses`` in round-robin order."""
    state = {"i": 0}

    def fake_generate_pairs(content, role="go_generator", max_pairs=1):  # noqa: ANN001
        out = []
        for _ in range(max_pairs):
            resp = responses[state["i"] % len(responses)]
            state["i"] += 1
            out.append({"instruction": "Q", "response": resp, "context": content})
        return out

    monkeypatch.setattr(gen, "generate_pairs", fake_generate_pairs)


# --- rejection sampling keeps the best survivor ----------------------------- #


def test_rejection_sampling_keeps_best(monkeypatch) -> None:
    gen = InstructionGenerator(offline=True)
    _cycle_teacher(gen, ["a score=0.2", "b score=0.9", "c score=0.5"], monkeypatch)

    pairs = gen.generate_dataset(
        [{"content": "doc"}],
        roles=["go_generator"],
        max_pairs_per_doc=1,
        rejection_samples=3,
        judge=StubJudgeByText(),
        judge_threshold=0.0,
    )
    assert len(pairs) == 1
    assert "score=0.9" in pairs[0]["response"]
    assert abs(pairs[0]["judge_score"] - 0.9) < 1e-9


# --- verify gate drops failures --------------------------------------------- #


def test_verify_gate_drops_failures(monkeypatch) -> None:
    gen = InstructionGenerator(offline=True)
    _cycle_teacher(gen, ["BAD code", "BAD code"], monkeypatch)
    verifier = StubVerifier(lambda r: "GOOD" in r)

    pairs = gen.generate_dataset(
        [{"content": "doc"}],
        roles=["go_generator"],
        max_pairs_per_doc=1,
        rejection_samples=2,
        verifier=verifier,
    )
    assert pairs == []  # every candidate failed verification


def test_verify_gate_keeps_passing(monkeypatch) -> None:
    gen = InstructionGenerator(offline=True)
    _cycle_teacher(gen, ["GOOD code"], monkeypatch)
    verifier = StubVerifier(lambda r: "GOOD" in r)

    pairs = gen.generate_dataset(
        [{"content": "doc"}],
        roles=["go_generator"],
        max_pairs_per_doc=1,
        rejection_samples=1,
        verifier=verifier,
    )
    assert len(pairs) == 1
    assert pairs[0]["verify_status"] == "ok"


# --- judge gate drops low scores -------------------------------------------- #


def test_judge_gate_drops_below_threshold(monkeypatch) -> None:
    gen = InstructionGenerator(offline=True)
    _cycle_teacher(gen, ["x score=0.3"], monkeypatch)

    pairs = gen.generate_dataset(
        [{"content": "doc"}],
        roles=["go_generator"],
        max_pairs_per_doc=1,
        rejection_samples=1,
        judge=StubJudgeByText(),
        judge_threshold=0.6,
    )
    assert pairs == []


def test_judge_gate_keeps_above_threshold(monkeypatch) -> None:
    gen = InstructionGenerator(offline=True)
    _cycle_teacher(gen, ["x score=0.8"], monkeypatch)

    pairs = gen.generate_dataset(
        [{"content": "doc"}],
        roles=["go_generator"],
        max_pairs_per_doc=1,
        rejection_samples=1,
        judge=StubJudgeByText(),
        judge_threshold=0.6,
    )
    assert len(pairs) == 1


# --- budget caps still honored, including judge spend ----------------------- #


def test_max_pairs_cap_with_gate(monkeypatch) -> None:
    gen = InstructionGenerator(offline=True)
    _cycle_teacher(gen, ["ok score=0.9"], monkeypatch)
    docs = [{"content": f"doc{i}"} for i in range(10)]

    pairs = gen.generate_dataset(
        docs,
        roles=["go_generator"],
        max_pairs_per_doc=1,
        max_pairs=3,
        rejection_samples=1,
        judge=StubJudgeByText(),
        judge_threshold=0.0,
    )
    assert len(pairs) == 3


def test_judge_spend_counts_toward_budget() -> None:
    # No teacher spend; each judge call costs $1. Cap at $2.5 -> stop after 3.
    gen = InstructionGenerator(offline=False, price_in=0.0, price_out=0.0)

    def fake_call(context, role, max_pairs):  # noqa: ANN001 - test stub
        return '{"pairs": [{"instruction": "Q", "response": "score=0.9"}]}', {
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    gen._call_teacher = fake_call  # type: ignore[method-assign]
    judge = StubJudgeByText(cost=1.0)
    docs = [{"content": f"doc{i}"} for i in range(10)]

    pairs = gen.generate_dataset(
        docs,
        roles=["go_generator"],
        max_pairs_per_doc=1,
        max_spend_usd=2.5,
        rejection_samples=1,
        judge=judge,
        judge_threshold=0.0,
    )
    assert len(pairs) == 3
    assert judge.spend_usd >= 2.5
    assert gen.spend_usd == 0.0  # all spend was the judge's
