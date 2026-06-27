"""Tests for the hf_datasets source: row normalization, language filter, streaming.

All tests stub the HuggingFace loader so they run fully offline (no network and
no ``datasets`` install required).
"""

from __future__ import annotations

from src.sources import get_source
from src.sources.hf_datasets import (
    HFDatasetsSource,
    is_language_row,
    normalize_row,
)

GO_CODE = "package main\n\nfunc Add(a, b int) int { return a + b }"
PY_CODE = "def add(a, b):\n    return a + b"


# --- row normalization across field-name variants --------------------------- #


def test_normalize_instruction_output() -> None:
    row = {"instruction": "Add two ints", "output": GO_CODE}
    assert normalize_row(row) == {"instruction": "Add two ints", "response": GO_CODE, "context": ""}


def test_normalize_instruction_response() -> None:
    row = {"instruction": "Q", "response": "A"}
    assert normalize_row(row) == {"instruction": "Q", "response": "A", "context": ""}


def test_normalize_problem_solution_magicoder() -> None:
    row = {"problem": "Implement Add", "solution": GO_CODE, "lang": "go"}
    assert normalize_row(row) == {"instruction": "Implement Add", "response": GO_CODE, "context": ""}


def test_normalize_prompt_completion() -> None:
    row = {"prompt": "Write Add", "completion": GO_CODE}
    assert normalize_row(row) == {"instruction": "Write Add", "response": GO_CODE, "context": ""}


def test_normalize_alpaca_input_becomes_context() -> None:
    row = {"instruction": "Fix it", "input": GO_CODE, "output": "fixed"}
    pair = normalize_row(row)
    assert pair == {"instruction": "Fix it", "response": "fixed", "context": GO_CODE}


def test_normalize_messages_chat_format() -> None:
    row = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Write Add"},
            {"role": "assistant", "content": GO_CODE},
        ]
    }
    pair = normalize_row(row)
    assert pair == {"instruction": "Write Add", "response": GO_CODE, "context": "You are helpful."}


def test_normalize_missing_fields_returns_none() -> None:
    assert normalize_row({"foo": "bar"}) is None
    assert normalize_row({"instruction": "only question"}) is None
    assert normalize_row({"response": "only answer"}) is None
    assert normalize_row("not a dict") is None  # type: ignore[arg-type]


# --- language filter -------------------------------------------------------- #


def test_language_filter_explicit_field_positive() -> None:
    pair = {"instruction": "x", "response": "y", "context": ""}
    assert is_language_row({"lang": "go"}, pair, "go") is True
    assert is_language_row({"language": "Golang"}, pair, "go") is True


def test_language_filter_explicit_field_negative() -> None:
    pair = {"instruction": "x", "response": "y", "context": ""}
    assert is_language_row({"lang": "python"}, pair, "go") is False


def test_language_filter_heuristic_positive() -> None:
    pair = {"instruction": "Solve it", "response": GO_CODE, "context": ""}
    assert is_language_row({}, pair, "go") is True
    fenced = {"instruction": "x", "response": "```go\nfmt.Println()\n```", "context": ""}
    assert is_language_row({}, fenced, "go") is True


def test_language_filter_heuristic_negative() -> None:
    pair = {"instruction": "Solve it", "response": PY_CODE, "context": ""}
    assert is_language_row({}, pair, "go") is False


def test_language_filter_non_go_requires_declared_field() -> None:
    pair = {"instruction": "x", "response": PY_CODE, "context": ""}
    # No declared language -> heuristic is Go-only, so non-go target is rejected.
    assert is_language_row({}, pair, "python") is False
    assert is_language_row({"lang": "python"}, pair, "python") is True


# --- streaming search ------------------------------------------------------- #


def test_search_streams_filters_and_caps(monkeypatch) -> None:
    rows = [
        {"problem": "Add ints", "solution": GO_CODE, "lang": "go"},
        {"problem": "Add floats", "solution": PY_CODE, "lang": "python"},
        {"problem": "Sub ints", "solution": GO_CODE.replace("Add", "Sub"), "lang": "go"},
        {"problem": "Mul ints", "solution": GO_CODE.replace("Add", "Mul"), "lang": "go"},
    ]
    source = HFDatasetsSource()
    monkeypatch.setattr(source, "_load_stream", lambda dataset, split: iter(rows))

    pairs = source.search("some/dataset", max_results=2, language="go")
    assert len(pairs) == 2  # capped, and the Python row was filtered out
    assert all("instruction" in p and "response" in p for p in pairs)
    assert pairs[0]["instruction"] == "Add ints"
    assert pairs[1]["instruction"] == "Sub ints"


def test_source_registered() -> None:
    assert isinstance(get_source("hf_datasets"), HFDatasetsSource)
