"""Tests for the quality judge: offline determinism, filtering, parse/retry, spend."""

from __future__ import annotations

import pytest

from src.core.judge import JUDGE_AXES, JudgeResult, QualityJudge

PAIR = {
    "instruction": "Write a Go function that adds two ints.",
    "response": "```go\nfunc Add(a, b int) int { return a + b }\n```",
}


# --- offline determinism ---------------------------------------------------- #


def test_offline_score_is_deterministic() -> None:
    judge = QualityJudge(offline=True)
    first = judge.judge_pair(PAIR)
    second = judge.judge_pair(PAIR)
    assert first == second
    assert 0.0 <= first.score <= 1.0
    assert set(first.axes) == set(JUDGE_AXES)
    for value in first.axes.values():
        assert 0.0 <= value <= 1.0


def test_offline_score_varies_by_content() -> None:
    judge = QualityJudge(offline=True)
    other = {"instruction": "Different task", "response": "Different answer"}
    assert judge.judge_pair(PAIR).score != judge.judge_pair(other).score


def test_offline_makes_no_spend() -> None:
    judge = QualityJudge(offline=True)
    judge.judge_pair(PAIR)
    assert judge.spend_usd == 0.0


def test_empty_pair_scores_zero() -> None:
    judge = QualityJudge(offline=True)
    assert judge.judge_pair({"instruction": "", "response": "x"}).score == 0.0


# --- filtering -------------------------------------------------------------- #


def test_filter_pairs_threshold() -> None:
    judge = QualityJudge(offline=True)
    pairs = [
        {"instruction": f"task {i}", "response": f"answer {i}"} for i in range(40)
    ]
    kept_low = judge.filter_pairs(pairs, threshold=0.0)
    kept_high = judge.filter_pairs(pairs, threshold=0.9)
    assert len(kept_low) == len(pairs)
    assert len(kept_high) < len(kept_low)
    for pair in kept_high:
        assert pair["judge_score"] >= 0.9


# --- mocked client: parse + retry + spend ----------------------------------- #


class _FakeUsage:
    prompt_tokens = 1_000_000
    completion_tokens = 0


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


def test_online_parse_and_spend(monkeypatch) -> None:
    judge = QualityJudge(offline=False, price_in=1.0, price_out=1.0)

    def fake_call(instruction, response, context):  # noqa: ANN001 - test stub
        raw = (
            '{"correctness": 1.0, "idiomatic": 0.5, "completeness": 0.5, '
            '"alignment": 1.0, "rationale": "good"}'
        )
        return raw, _FakeUsage()

    monkeypatch.setattr(judge, "_call_judge", fake_call)
    result = judge.judge_pair(PAIR)
    assert isinstance(result, JudgeResult)
    assert result.axes["correctness"] == 1.0
    assert abs(result.score - 0.75) < 1e-9
    assert result.rationale == "good"
    assert abs(judge.spend_usd - 1.0) < 1e-9  # 1M input tokens @ $1/1M


def test_online_retry_then_success(monkeypatch) -> None:
    judge = QualityJudge(offline=False, max_retries=3, backoff_base=0.0)
    calls = {"n": 0}

    class _Client:
        class chat:  # noqa: N801 - mirror SDK shape
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    calls["n"] += 1
                    if calls["n"] < 2:
                        raise RuntimeError("transient")
                    return _FakeCompletion('{"correctness": 0.8, "idiomatic": 0.8, '
                                           '"completeness": 0.8, "alignment": 0.8}')

    monkeypatch.setattr(judge, "_client_handle", lambda: _Client())
    result = judge.judge_pair(PAIR)
    assert calls["n"] == 2
    assert abs(result.score - 0.8) < 1e-9


def test_online_failure_scores_zero(monkeypatch) -> None:
    judge = QualityJudge(offline=False, max_retries=2, backoff_base=0.0)

    class _Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("down")

    monkeypatch.setattr(judge, "_client_handle", lambda: _Client())
    result = judge.judge_pair(PAIR)
    assert result.score == 0.0


def test_parse_result_tolerates_garbage() -> None:
    assert QualityJudge._parse_result("not json").score == 0.0
    assert QualityJudge._parse_result("").score == 0.0


def test_parse_result_clamps_and_handles_wrapped_json() -> None:
    raw = '```json\n{"correctness": 2.0, "idiomatic": -1, "completeness": 0.5, "alignment": 0.5}\n```'
    result = QualityJudge._parse_result(raw)
    assert result.axes["correctness"] == 1.0  # clamped
    assert result.axes["idiomatic"] == 0.0  # clamped


# --- env fallback ----------------------------------------------------------- #


def test_judge_env_falls_back_to_teacher(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_JUDGE_BASE_URL", raising=False)
    monkeypatch.setenv("FORGE_TEACHER_BASE_URL", "https://teacher.example/v1")
    monkeypatch.setenv("FORGE_TEACHER_PRICE_IN", "0.5")
    judge = QualityJudge(offline=True)
    assert judge.base_url == "https://teacher.example/v1"
    assert judge.price_in == 0.5


def test_judge_env_override_wins(monkeypatch) -> None:
    monkeypatch.setenv("FORGE_TEACHER_BASE_URL", "https://teacher.example/v1")
    monkeypatch.setenv("FORGE_JUDGE_BASE_URL", "https://judge.example/v1")
    judge = QualityJudge(offline=True)
    assert judge.base_url == "https://judge.example/v1"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
