"""Forge command-line interface.

Exposes each pipeline stage as a subcommand plus an end-to-end ``run`` driven by
a YAML config:

    forge discover --source github --query "language:go stars:>2000" --max 5
    forge download --input repos.json
    forge process  --input repos.json
    forge generate --input docs.json --role go_explainer
    forge build    --input pairs.json --name go_guild_v1
    forge run      --config configs/example.yaml
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import typer
import yaml

from src.core.dataset_builder import DatasetBuilder
from src.core.discoverer import Discoverer
from src.core.downloader import Downloader
from src.core.instruction_gen import InstructionGenerator
from src.core.processor import Processor

app = typer.Typer(
    add_completion=False,
    help="Forge -- the domain-agnostic data pipeline for GuildLM.",
    no_args_is_help=True,
)
logger = logging.getLogger("forge")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging.")) -> None:
    """Global options."""
    _configure_logging(verbose)


@app.command()
def discover(
    source: str = typer.Option("github", help="Registered source name."),
    query: str = typer.Option(..., help="Source-specific search query."),
    max_results: int = typer.Option(20, "--max", help="Maximum records."),
    output: Path = typer.Option(Path("data/discovered.json"), help="Output JSON path."),
) -> None:
    """Discover items from a source and write standardized records to JSON."""
    records = Discoverer().discover(source, query, max_results=max_results)
    _write_json(output, records)
    typer.echo(f"Discovered {len(records)} record(s) -> {output}")


@app.command()
def download(
    input: Path = typer.Option(..., help="JSON file of discovered records."),
    output_dir: Path = typer.Option(Path("data/raw"), help="Directory for clones."),
    output: Path = typer.Option(Path("data/downloaded.json"), help="Results JSON path."),
    max_workers: int = typer.Option(4, help="Concurrency."),
) -> None:
    """Clone discovered repositories concurrently."""
    records = _read_json(input)
    results = Downloader(str(output_dir), max_workers=max_workers).clone_all(records)
    _write_json(output, [r.to_dict() for r in results])
    ok = sum(1 for r in results if r.status in ("success", "cached"))
    typer.echo(f"Downloaded {ok}/{len(results)} -> {output}")


@app.command()
def process(
    input: Path = typer.Option(..., help="JSON file of download results."),
    output: Path = typer.Option(Path("data/documents.json"), help="Clean documents JSON path."),
    extensions: str = typer.Option(".go", help="Comma-separated file extensions."),
) -> None:
    """Extract and clean documents from downloaded repositories."""
    results = _read_json(input)
    proc = Processor(include_extensions=[e.strip() for e in extensions.split(",") if e.strip()])
    raw: list[dict[str, Any]] = []
    for result in results:
        path = result.get("local_path")
        if path:
            raw.extend(proc.process_repository(path, license=result.get("license")))
    cleaned, stats = proc.clean(raw)
    _write_json(output, cleaned)
    typer.echo(f"Cleaned documents: {stats.to_dict()} -> {output}")


@app.command()
def generate(
    input: Path = typer.Option(..., help="JSON file of clean documents."),
    output: Path = typer.Option(Path("data/pairs.json"), help="Pairs JSON path."),
    role: str = typer.Option("go_explainer", help="Teacher role."),
    max_pairs: int = typer.Option(1, help="Pairs per document."),
    offline: bool = typer.Option(False, help="Use deterministic offline teacher."),
) -> None:
    """Generate instruction/response pairs from documents."""
    documents = _read_json(input)
    gen = InstructionGenerator(offline=offline)
    pairs: list[dict[str, Any]] = []
    for doc in documents:
        pairs.extend(gen.generate_pairs(doc.get("content", ""), role=role, max_pairs=max_pairs))
    _write_json(output, pairs)
    typer.echo(f"Generated {len(pairs)} pair(s) -> {output}")


@app.command()
def build(
    input: Path = typer.Option(..., help="JSON file of instruction pairs."),
    name: str = typer.Option(..., help="Dataset name (filename prefix)."),
    output_dir: Path = typer.Option(Path("data/datasets"), help="Output directory."),
    val_ratio: float = typer.Option(0.1, help="Validation split fraction."),
    parquet: bool = typer.Option(False, help="Also export Parquet (needs pyarrow)."),
) -> None:
    """Validate, split, and export pairs into a training dataset."""
    pairs = _read_json(input)
    formats = ["jsonl"] + (["parquet"] if parquet else [])
    manifest = DatasetBuilder(str(output_dir)).build(
        pairs, name, val_ratio=val_ratio, formats=formats
    )
    typer.echo(f"Built {manifest.total_records} record(s): {manifest.splits} -> {output_dir}")


@app.command()
def run(config: Path = typer.Option(..., help="YAML pipeline config.")) -> None:
    """Run the full discover -> download -> process -> generate -> build pipeline."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    manifest = run_pipeline(cfg)
    typer.echo(
        f"Pipeline complete: dataset {manifest.name!r} "
        f"({manifest.total_records} records, splits={manifest.splits})."
    )


def run_pipeline(cfg: dict[str, Any]):
    """Execute the end-to-end pipeline from a parsed config dict.

    Returns the resulting :class:`~src.core.dataset_builder.BuildManifest`.
    """
    dl_cfg = cfg.get("download", {})
    proc_cfg = cfg.get("process", {})
    gen_cfg = cfg.get("generate", {})
    build_cfg = cfg.get("build", {})

    records = Discoverer().discover(
        cfg.get("source", "github"),
        cfg["query"],
        max_results=cfg.get("max_results", 10),
    )

    downloader = Downloader(
        dl_cfg.get("output_dir", "data/raw"),
        max_workers=dl_cfg.get("max_workers", 4),
    )
    download_results = downloader.clone_all(records)

    processor = Processor(
        include_extensions=proc_cfg.get("include_extensions", [".go"]),
        min_length=proc_cfg.get("min_length", 50),
        max_length=proc_cfg.get("max_length", 100_000),
        near_dup_threshold=proc_cfg.get("near_dup_threshold", 0.85),
        allow_unknown_license=proc_cfg.get("allow_unknown_license", True),
    )
    raw_docs: list[dict[str, Any]] = []
    for result in download_results:
        if result.local_path:
            raw_docs.extend(processor.process_repository(result.local_path, license=result.extra.get("license")))
    documents, clean_stats = processor.clean(raw_docs)

    generator = InstructionGenerator(offline=gen_cfg.get("offline", False))
    roles = gen_cfg.get("roles", ["go_explainer"])
    max_pairs = gen_cfg.get("max_pairs_per_doc", 1)
    pairs: list[dict[str, Any]] = []
    for doc in documents:
        for role in roles:
            pairs.extend(generator.generate_pairs(doc["content"], role=role, max_pairs=max_pairs))

    builder = DatasetBuilder(build_cfg.get("output_dir", "data/datasets"))
    return builder.build(
        pairs,
        build_cfg.get("name", "forge_dataset"),
        val_ratio=build_cfg.get("val_ratio", 0.1),
        seed=build_cfg.get("seed", 42),
        formats=build_cfg.get("formats", ["jsonl"]),
        source_stats=clean_stats.to_dict(),
    )


if __name__ == "__main__":  # pragma: no cover
    app()
