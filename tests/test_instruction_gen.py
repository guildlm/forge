"""Tests for instruction generation: offline determinism, parsing, roles."""

from __future__ import annotations

import pytest

from src.core.instruction_gen import InstructionGenerator, Role, get_role, register_role


def test_offline_mode_is_deterministic() -> None:
    gen = InstructionGenerator(offline=True)
    code = "package main\nfunc main() { println(\"hi\") }"
    first = gen.generate_pairs(code, role="go_explainer", max_pairs=2)
    second = gen.generate_pairs(code, role="go_explainer", max_pairs=2)
    assert first == second
    assert len(first) == 2
    for pair in first:
        assert pair["instruction"]
        assert pair["response"]
        assert pair["context"]


def test_offline_mode_varies_by_role() -> None:
    gen = InstructionGenerator(offline=True)
    code = "package main\nfunc main() {}"
    reviewer = gen.generate_pairs(code, role="go_reviewer")[0]
    explainer = gen.generate_pairs(code, role="go_explainer")[0]
    assert reviewer["response"] != explainer["response"]


def test_offline_empty_content_returns_empty() -> None:
    gen = InstructionGenerator(offline=True)
    assert gen.generate_pairs("   ", role="go_explainer") == []


def test_unknown_role_raises() -> None:
    with pytest.raises(ValueError):
        InstructionGenerator(offline=True).generate_pairs("code", role="nope")


def test_role_registry_is_extensible() -> None:
    register_role(Role("sql_optimizer", "You optimize SQL.", "optimize the query"))
    role = get_role("sql_optimizer")
    assert role.name == "sql_optimizer"


def test_truncation_respects_budget() -> None:
    gen = InstructionGenerator(offline=True, max_context_chars=100)
    long = "line\n" * 1000
    pair = gen.generate_pairs(long, role="go_explainer")[0]
    assert len(pair["context"]) <= 100 + len("\n... [truncated]")
    assert pair["context"].endswith("[truncated]")


def test_parse_pairs_handles_wrapped_json() -> None:
    raw = 'Here you go:\n```json\n{"pairs": [{"instruction": "Q", "response": "A"}]}\n```'
    parsed = InstructionGenerator._parse_pairs(raw)
    assert parsed == [{"instruction": "Q", "response": "A"}]


def test_parse_pairs_handles_top_level_list() -> None:
    raw = '[{"instruction": "Q1", "response": "A1"}, {"instruction": "", "response": "x"}]'
    parsed = InstructionGenerator._parse_pairs(raw)
    assert parsed == [{"instruction": "Q1", "response": "A1"}]


def test_parse_pairs_handles_garbage() -> None:
    assert InstructionGenerator._parse_pairs("not json at all") == []
    assert InstructionGenerator._parse_pairs("") == []
