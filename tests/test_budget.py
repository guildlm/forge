"""Tests for budget-capped generation and the import CLI (all offline)."""

from __future__ import annotations

import json

from typer.testing import CliRunner

import src.sources.hf_datasets as hf_datasets
from src.cli import app
from src.core.instruction_gen import InstructionGenerator

runner = CliRunner()

GO_CODE = "package main\n\nfunc Add(a, b int) int { return a + b }"


# --- max_pairs cap ---------------------------------------------------------- #


def test_generate_dataset_stops_at_max_pairs() -> None:
    gen = InstructionGenerator(offline=True)
    docs = [{"content": f"package p{i}\nfunc f{i}() {{}}"} for i in range(10)]
    pairs = gen.generate_dataset(
        docs, roles=["go_explainer"], max_pairs_per_doc=1, max_pairs=3
    )
    assert len(pairs) == 3


# --- max_spend_usd cap ------------------------------------------------------ #


def test_generate_dataset_stops_at_max_spend(monkeypatch) -> None:
    # Prices of $1/1M tokens with 500k in + 500k out => $1.00 per call.
    gen = InstructionGenerator(offline=False, price_in=1.0, price_out=1.0)

    def fake_call(context, role, max_pairs):  # noqa: ANN001 - test stub
        usage = {"prompt_tokens": 500_000, "completion_tokens": 500_000}
        return '{"pairs": [{"instruction": "Q", "response": "A"}]}', usage

    monkeypatch.setattr(gen, "_call_teacher", fake_call)
    docs = [{"content": f"package p{i}\nfunc f{i}() {{}}"} for i in range(10)]

    pairs = gen.generate_dataset(
        docs, roles=["go_explainer"], max_pairs_per_doc=1, max_spend_usd=2.5
    )
    # Calls cost $1 each; loop stops before the call that would start at spend>=2.5,
    # i.e. after 3 calls (spend == $3.00).
    assert len(pairs) == 3
    assert gen.spend_usd >= 2.5
    assert abs(gen.spend_usd - 3.0) < 1e-9


def test_spend_tracking_handles_usage_object() -> None:
    gen = InstructionGenerator(offline=False, price_in=1.0, price_out=1.0)

    class _Usage:
        prompt_tokens = 1_000_000
        completion_tokens = 0

    cost = gen._update_spend(_Usage())
    assert abs(cost - 1.0) < 1e-9
    assert abs(gen.spend_usd - 1.0) < 1e-9
    assert gen._update_spend(None) == 0.0


# --- import CLI on a local fixture (stubbed loader) ------------------------- #


def test_import_cli_on_local_fixture(tmp_path, monkeypatch) -> None:
    rows = [
        {"problem": "Implement Add", "solution": GO_CODE, "lang": "go"},
        {"problem": "Implement Add", "solution": GO_CODE, "lang": "go"},  # exact duplicate
        {"problem": "Python add", "solution": "def add(a, b): return a + b", "lang": "python"},
        {"problem": "Implement Sub", "solution": GO_CODE.replace("Add", "Sub"), "lang": "go"},
    ]
    monkeypatch.setattr(
        hf_datasets.HFDatasetsSource,
        "_load_stream",
        lambda self, dataset, split: iter(rows),
    )
    out = tmp_path / "pairs.json"
    result = runner.invoke(
        app,
        ["import", "--dataset", "x/y", "--language", "go", "--max", "10",
         "--min-length", "10", "--output", str(out)],
    )
    assert result.exit_code == 0, result.output
    pairs = json.loads(out.read_text(encoding="utf-8"))
    # Python row filtered out; exact-duplicate Go row deduped -> 2 unique Go pairs.
    assert len(pairs) == 2
    instructions = {p["instruction"] for p in pairs}
    assert instructions == {"Implement Add", "Implement Sub"}
    for pair in pairs:
        assert pair["instruction"] and pair["response"]
