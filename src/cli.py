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
from src.core.judge import QualityJudge
from src.core.processor import CleaningStats, Processor, scrub_pii
from src.core.verifier import GoVerifier
from src.sources import get_source

#: Source names that already yield instruction/response pairs (import route).
IMPORT_SOURCES: frozenset[str] = frozenset({"hf_datasets"})

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


def _clean_pairs(
    raw_pairs: list[dict[str, Any]],
    *,
    min_length: int = 40,
    max_length: int = 20_000,
    near_dup_threshold: float = 0.85,
) -> tuple[list[dict[str, Any]], CleaningStats]:
    """Scrub and clean imported instruction pairs, reusing :class:`Processor`.

    PII/secret scrubbing is applied to each pair field with the processor's
    :func:`scrub_pii`; exact/near deduplication, length and printability filters
    are delegated to :class:`Processor` over the combined pair text.

    Args:
        raw_pairs: Pairs with ``instruction``/``response``/``context`` keys.
        min_length: Minimum combined-pair length in characters.
        max_length: Maximum combined-pair length in characters.
        near_dup_threshold: MinHash similarity above which pairs are dropped.

    Returns:
        ``(clean_pairs, stats)`` where ``stats`` carries the cleaning counters
        (with ``pii_redactions`` reflecting the field-level scrubbing).
    """
    processor = Processor(
        min_length=min_length,
        max_length=max_length,
        near_dup_threshold=near_dup_threshold,
        allow_unknown_license=True,
        scrub=False,  # we scrub each field below to keep them individually clean
    )
    redactions = 0
    docs: list[dict[str, Any]] = []
    for pair in raw_pairs:
        instruction, n_i = scrub_pii(str(pair.get("instruction", "")))
        response, n_r = scrub_pii(str(pair.get("response", "")))
        context, n_c = scrub_pii(str(pair.get("context", "")))
        if not instruction.strip() or not response.strip():
            continue
        redactions += n_i + n_r + n_c
        combined = "\n\n".join(part for part in (instruction, context, response) if part)
        docs.append(
            {
                "instruction": instruction,
                "response": response,
                "context": context,
                "content": combined,
                "license": None,
            }
        )
    cleaned, stats = processor.clean(docs)
    stats.pii_redactions = redactions
    pairs = [
        {"instruction": d["instruction"], "response": d["response"], "context": d["context"]}
        for d in cleaned
    ]
    return pairs, stats


def _refine_pairs(
    pairs: list[dict[str, Any]],
    *,
    verifier: GoVerifier | None,
    judge: QualityJudge | None,
    judge_threshold: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply the quality gate to an existing list of pairs.

    Each pair is execution-verified (dropped per the verifier's strict policy)
    and/or judge-scored (dropped below ``judge_threshold``). Pairs without a
    ``role`` key are verified with build/vet only (no ``go test``).

    Returns:
        ``(kept_pairs, stats)`` where ``stats`` counts what the gate did.
    """
    kept: list[dict[str, Any]] = []
    stats = {"input": len(pairs), "verified_dropped": 0, "judged_dropped": 0, "kept": 0}
    for pair in pairs:
        enriched = dict(pair)
        if verifier is not None:
            result = verifier.verify(str(enriched.get("response", "")), role=str(enriched.get("role", "")))
            if not result.passed:
                stats["verified_dropped"] += 1
                continue
            enriched["verify_status"] = result.status
        if judge is not None:
            verdict = judge.judge_pair(enriched)
            enriched["judge_score"] = verdict.score
            if verdict.score < judge_threshold:
                stats["judged_dropped"] += 1
                continue
        kept.append(enriched)
    stats["kept"] = len(kept)
    return kept, stats


def _apply_refine_stage(
    pairs: list[dict[str, Any]], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply an optional top-level ``refine:`` config stage to ``pairs``.

    The ``refine`` block accepts ``verify`` (bool), ``strict_verify`` (bool),
    ``judge`` (bool), ``judge_threshold`` (float) and ``offline`` (bool). When no
    ``refine`` block is present the pairs are returned unchanged.
    """
    refine_cfg = cfg.get("refine")
    if not refine_cfg:
        return pairs
    offline = refine_cfg.get("offline", cfg.get("generate", {}).get("offline", False))
    verifier = (
        GoVerifier(strict=refine_cfg.get("strict_verify", False))
        if refine_cfg.get("verify", True)
        else None
    )
    judge = QualityJudge(offline=offline) if refine_cfg.get("judge", True) else None
    kept, stats = _refine_pairs(
        pairs, verifier=verifier, judge=judge, judge_threshold=refine_cfg.get("judge_threshold", 0.6)
    )
    logger.info("Refine stage: %s", stats)
    return kept


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


@app.command(name="import")
def import_dataset(
    dataset: str = typer.Option(..., help="HuggingFace dataset id, e.g. ise-uiuc/Magicoder-OSS-Instruct-75K."),
    split: str = typer.Option("train", help="Dataset split to stream."),
    language: str = typer.Option("go", help="Programming language to keep."),
    max_records: int = typer.Option(3000, "--max", help="Max matching pairs to collect."),
    output: Path = typer.Option(Path("data/curated.json"), help="Output pairs JSON path."),
    min_length: int = typer.Option(40, help="Min combined-pair length (chars)."),
    max_length: int = typer.Option(20_000, help="Max combined-pair length (chars)."),
    near_dup_threshold: float = typer.Option(0.85, help="Near-dup MinHash threshold."),
) -> None:
    """Import an existing HuggingFace instruction dataset ($0 curation route).

    Streams the dataset, normalizes rows to Forge pairs, keeps only the requested
    language, runs the processor (dedup/PII/length), and writes pairs ready for
    ``forge build``.
    """
    source = get_source("hf_datasets")
    raw = source.search(dataset, max_results=max_records, split=split, language=language)
    pairs, stats = _clean_pairs(
        raw, min_length=min_length, max_length=max_length, near_dup_threshold=near_dup_threshold
    )
    _write_json(output, pairs)
    typer.echo(f"Imported {len(pairs)} pair(s) [{stats.to_dict()}] -> {output}")


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
    role: str = typer.Option("go_explainer", help="Teacher role (comma-separated for several)."),
    max_pairs_per_doc: int = typer.Option(1, help="Pairs requested per (document, role)."),
    max_pairs: int | None = typer.Option(None, help="Hard cap: stop after N total pairs."),
    max_spend_usd: float | None = typer.Option(
        None, help="Hard cap: stop once estimated teacher+judge spend (USD) reaches this."
    ),
    offline: bool = typer.Option(False, help="Use deterministic offline teacher (and judge)."),
    verify: bool = typer.Option(False, "--verify/--no-verify", help="Execution-verify Go code."),
    strict_verify: bool = typer.Option(
        False, "--strict-verify", help="Drop pairs whose code is unverifiable (no toolchain / no code)."
    ),
    judge: bool = typer.Option(False, "--judge/--no-judge", help="Rubric-judge and filter pairs."),
    judge_threshold: float = typer.Option(0.0, help="Minimum judge score to keep a pair."),
    rejection_samples: int = typer.Option(
        1, help="Candidates sampled per keep-slot; the best survivor is kept."
    ),
) -> None:
    """Generate instruction/response pairs from documents under budget caps.

    With a real teacher, ``--max-pairs`` and ``--max-spend-usd`` cap cost: the
    loop stops cleanly when either is hit and reports the estimated spend. The
    quality gate (``--verify`` / ``--judge`` / ``--rejection-samples``) verifies
    that code compiles and filters low-quality pairs; judge spend also counts
    toward ``--max-spend-usd``.
    """
    documents = _read_json(input)
    roles = [r.strip() for r in role.split(",") if r.strip()]
    gen = InstructionGenerator(offline=offline)
    verifier = GoVerifier(strict=strict_verify) if verify else None
    quality_judge = QualityJudge(offline=offline) if judge else None
    pairs = gen.generate_dataset(
        documents,
        roles=roles,
        max_pairs_per_doc=max_pairs_per_doc,
        max_pairs=max_pairs,
        max_spend_usd=max_spend_usd,
        verifier=verifier,
        judge=quality_judge,
        judge_threshold=judge_threshold,
        rejection_samples=rejection_samples,
    )
    _write_json(output, pairs)
    spend = gen.spend_usd + (quality_judge.spend_usd if quality_judge else 0.0)
    typer.echo(f"Generated {len(pairs)} pair(s) (estimated spend ~${spend:.4f}) -> {output}")


@app.command()
def refine(
    input: Path = typer.Option(..., help="JSON file of existing instruction pairs."),
    output: Path = typer.Option(Path("data/refined.json"), help="Kept pairs JSON path."),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Execution-verify Go code."),
    strict_verify: bool = typer.Option(
        False, "--strict-verify", help="Drop pairs whose code is unverifiable (no toolchain / no code)."
    ),
    judge: bool = typer.Option(True, "--judge/--no-judge", help="Rubric-judge and filter pairs."),
    judge_threshold: float = typer.Option(0.6, help="Minimum judge score to keep a pair."),
    offline: bool = typer.Option(False, help="Use the deterministic offline judge."),
) -> None:
    """Apply the quality gate to an EXISTING pairs file (verify + judge).

    Runs the same gate as ``forge generate`` over pairs you already have -- e.g.
    Route-A curated data -- so any dataset can be filtered to verified, judged
    pairs. Reads/writes the Forge pairs JSON schema.
    """
    pairs = _read_json(input)
    verifier = GoVerifier(strict=strict_verify) if verify else None
    quality_judge = QualityJudge(offline=offline) if judge else None
    kept, stats = _refine_pairs(
        pairs, verifier=verifier, judge=quality_judge, judge_threshold=judge_threshold
    )
    _write_json(output, kept)
    spend = quality_judge.spend_usd if quality_judge else 0.0
    typer.echo(f"Refined {stats['kept']}/{stats['input']} pair(s) [{stats}] "
               f"(estimated judge spend ~${spend:.4f}) -> {output}")


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

    Two routes are supported, selected by ``mode`` (or auto-detected from the
    source):

    * ``import`` -- curate an existing instruction dataset (e.g. ``hf_datasets``):
      the source already yields pairs, so teacher generation is skipped.
    * ``generate`` -- the classic discover -> download -> process -> generate
      -> build flow with a teacher model.

    Returns the resulting :class:`~src.core.dataset_builder.BuildManifest`.
    """
    source_name = cfg.get("source", "github")
    mode = cfg.get("mode") or ("import" if source_name in IMPORT_SOURCES else "generate")
    if mode == "import":
        return _run_import_pipeline(cfg, source_name)
    return _run_generate_pipeline(cfg, source_name)


def _run_import_pipeline(cfg: dict[str, Any], source_name: str):
    """Curate an existing instruction dataset into a built Forge dataset."""
    proc_cfg = cfg.get("process", {})
    build_cfg = cfg.get("build", {})
    dataset = cfg.get("dataset") or cfg.get("query")
    if not dataset:
        raise ValueError("import mode requires a 'dataset' (or 'query') key in the config")

    raw_pairs = Discoverer().discover(
        source_name,
        dataset,
        max_results=cfg.get("max_records", cfg.get("max_results", 3000)),
        split=cfg.get("split", "train"),
        language=cfg.get("language", "go"),
    )
    pairs, stats = _clean_pairs(
        raw_pairs,
        min_length=proc_cfg.get("min_length", 40),
        max_length=proc_cfg.get("max_length", 20_000),
        near_dup_threshold=proc_cfg.get("near_dup_threshold", 0.85),
    )
    pairs = _apply_refine_stage(pairs, cfg)
    builder = DatasetBuilder(build_cfg.get("output_dir", "data/datasets"))
    return builder.build(
        pairs,
        build_cfg.get("name", "forge_curated"),
        val_ratio=build_cfg.get("val_ratio", 0.1),
        seed=build_cfg.get("seed", 42),
        formats=build_cfg.get("formats", ["jsonl"]),
        source_stats=stats.to_dict(),
    )


def _run_generate_pipeline(cfg: dict[str, Any], source_name: str):
    """Classic discover -> download -> process -> generate -> build flow."""
    dl_cfg = cfg.get("download", {})
    proc_cfg = cfg.get("process", {})
    gen_cfg = cfg.get("generate", {})
    build_cfg = cfg.get("build", {})

    records = Discoverer().discover(
        source_name,
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

    offline = gen_cfg.get("offline", False)
    generator = InstructionGenerator(offline=offline)
    verifier = (
        GoVerifier(strict=gen_cfg.get("strict_verify", False))
        if gen_cfg.get("verify", False)
        else None
    )
    judge = QualityJudge(offline=offline) if gen_cfg.get("judge", False) else None
    pairs = generator.generate_dataset(
        documents,
        roles=gen_cfg.get("roles", ["go_explainer"]),
        max_pairs_per_doc=gen_cfg.get("max_pairs_per_doc", 1),
        max_pairs=gen_cfg.get("max_pairs"),
        max_spend_usd=gen_cfg.get("max_spend_usd"),
        verifier=verifier,
        judge=judge,
        judge_threshold=gen_cfg.get("judge_threshold", 0.0),
        rejection_samples=gen_cfg.get("rejection_samples", 1),
    )

    pairs = _apply_refine_stage(pairs, cfg)
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
