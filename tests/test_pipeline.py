"""End-to-end pipeline test: extract -> clean -> generate (offline) -> build."""

from __future__ import annotations

from src.core.dataset_builder import DatasetBuilder
from src.core.instruction_gen import InstructionGenerator
from src.core.processor import Processor

GO_FILE = """package main

import \"fmt\"

// Greeter prints a friendly message.
func Greeter(name string) string {
    return fmt.Sprintf(\"Hello, %s!\", name)
}

func main() {
    fmt.Println(Greeter(\"GuildLM\"))
}
"""


def test_full_offline_pipeline(tmp_path) -> None:
    # Build a tiny fake repo on disk.
    repo = tmp_path / "demo_repo"
    repo.mkdir()
    (repo / "main.go").write_text(GO_FILE, encoding="utf-8")
    (repo / "ignore_test.go").write_text("package main\n", encoding="utf-8")

    processor = Processor(include_extensions=[".go"], min_length=20)
    raw = list(processor.process_repository(str(repo), license="MIT"))
    # The *_test.go file is excluded by default patterns.
    assert [d["file_path"] for d in raw] == ["main.go"]

    documents, stats = processor.clean(raw)
    assert stats.total_out == 1

    generator = InstructionGenerator(offline=True)
    pairs = []
    for doc in documents:
        pairs.extend(generator.generate_pairs(doc["content"], role="go_explainer"))
    assert pairs

    builder = DatasetBuilder(str(tmp_path / "ds"))
    manifest = builder.build(pairs, "demo", val_ratio=0.0, source_stats=stats.to_dict())
    assert manifest.total_records == len(pairs)
    assert manifest.stats["total_out"] == 1
