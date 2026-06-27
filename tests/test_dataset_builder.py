"""Tests for dataset building: schema validation, split, JSONL roundtrip, manifest."""

from __future__ import annotations

import json

import pytest

from src.core.dataset_builder import DatasetBuilder, SchemaError


def _pairs(n: int) -> list[dict]:
    return [
        {"instruction": f"Explain snippet {i}", "context": f"x{i} := {i}", "response": f"It sets x{i}."}
        for i in range(n)
    ]


def test_jsonl_roundtrip(tmp_path) -> None:
    builder = DatasetBuilder(str(tmp_path))
    path = builder.export_to_jsonl(_pairs(3), "data.jsonl")
    records = DatasetBuilder.load_jsonl(path)
    assert len(records) == 3
    first = records[0]
    assert first["instruction"] == "Explain snippet 0"
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][-1]["role"] == "assistant"
    assert first["messages"][-1]["content"] == "It sets x0."


def test_build_split_is_deterministic(tmp_path) -> None:
    builder = DatasetBuilder(str(tmp_path))
    manifest = builder.build(_pairs(10), "ds", val_ratio=0.2, seed=7)
    assert manifest.splits == {"train": 8, "validation": 2}
    assert manifest.total_records == 10

    train = DatasetBuilder.load_jsonl(str(tmp_path / "ds.train.jsonl"))
    val = DatasetBuilder.load_jsonl(str(tmp_path / "ds.validation.jsonl"))
    assert len(train) == 8 and len(val) == 2

    # Re-running with the same seed reproduces the same split assignment.
    builder2 = DatasetBuilder(str(tmp_path / "again"))
    manifest2 = builder2.build(_pairs(10), "ds", val_ratio=0.2, seed=7)
    val2 = DatasetBuilder.load_jsonl(str(tmp_path / "again" / "ds.validation.jsonl"))
    assert [r["instruction"] for r in val] == [r["instruction"] for r in val2]
    assert manifest2.splits == manifest.splits


def test_manifest_has_hashes_and_stats(tmp_path) -> None:
    builder = DatasetBuilder(str(tmp_path))
    builder.build(_pairs(5), "ds", val_ratio=0.0, source_stats={"total_out": 5})
    manifest_path = tmp_path / "ds.manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["stats"] == {"total_out": 5}
    assert data["files"]
    for entry in data["files"]:
        assert len(entry["sha256"]) == 64
        assert entry["records"] >= 0


def test_schema_validation_rejects_empty(tmp_path) -> None:
    builder = DatasetBuilder(str(tmp_path))
    with pytest.raises(SchemaError):
        builder.normalize({"instruction": "", "response": "x"})
    with pytest.raises(SchemaError):
        builder.normalize({"instruction": "x", "response": "  "})


def test_no_validation_file_when_ratio_zero(tmp_path) -> None:
    builder = DatasetBuilder(str(tmp_path))
    builder.build(_pairs(4), "ds", val_ratio=0.0)
    assert (tmp_path / "ds.train.jsonl").exists()
    assert not (tmp_path / "ds.validation.jsonl").exists()


def test_parquet_export_when_available(tmp_path) -> None:
    pytest.importorskip("pyarrow")
    builder = DatasetBuilder(str(tmp_path))
    manifest = builder.build(_pairs(4), "ds", val_ratio=0.0, formats=["jsonl", "parquet"])
    assert (tmp_path / "ds.train.parquet").exists()
    assert any(f["format"] == "parquet" for f in manifest.files)
