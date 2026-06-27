# Forge

[![CI](https://github.com/guildlm/forge/actions/workflows/ci.yml/badge.svg)](https://github.com/guildlm/forge/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

**Forge is the domain-agnostic data pipeline of [GuildLM](https://github.com/guildlm/guildlm.github.io).**

GuildLM trains small, specialist LLMs grouped into *guilds* and routed by a
*brain*. Forge produces the supervised fine-tuning (SFT) datasets those
specialists learn from. The core engine knows **nothing** about any specific
domain: all platform and domain knowledge lives in pluggable `sources/` and
`plugins/`. Point Forge at a new source and a new teacher role, and it produces a
new guild's dataset without touching the engine.

```
discover ──▶ download ──▶ process ──▶ generate ──▶ build
 (sources)   (clone/fetch)  (clean)    (teacher)    (JSONL/Parquet + manifest)
```

---

## Architecture role

| Stage | Module | Responsibility |
| --- | --- | --- |
| **Discover** | `src/core/discoverer.py` + `src/sources/` | Find candidate items (repos, papers, ...) via a registered `Source`. |
| **Download** | `src/core/downloader.py` | Concurrently and politely clone repos / fetch URLs with retries. |
| **Process** | `src/core/processor.py` | Extract documents and clean them (dedup, PII, license, length, encoding). |
| **Generate** | `src/core/instruction_gen.py` | Prompt a teacher model for `(instruction, response)` pairs. |
| **Build** | `src/core/dataset_builder.py` | Validate, split, and export the training dataset with a manifest. |

Everything domain-specific is isolated:

- `src/sources/` — *where the data comes from* (GitHub, arXiv, ...).
- `src/plugins/` — *domain conventions* (e.g. which file types matter for a guild).

---

## Quickstart

```bash
git clone https://github.com/guildlm/forge.git
cd forge
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'          # add ',parquet' for Parquet, ',hf' for dataset curation

pytest -q                        # tests run fully offline
```

Run the whole pipeline from the example config (offline teacher, no GPU needed):

```bash
forge run --config configs/example.yaml
```

This discovers a few high-quality Go repositories, clones them, cleans the Go
files, generates synthetic instruction pairs (offline), and writes a dataset to
`data/datasets/`.

---

## CLI usage

`forge` exposes each stage as a subcommand plus an end-to-end `run`:

```bash
# 1. Discover (writes standardized records to JSON)
forge discover --source github --query "language:go stars:>2000" --max 5 \
               --output data/discovered.json

# 2. Download (shallow git clone, concurrent + polite)
forge download --input data/discovered.json --output-dir data/raw \
               --output data/downloaded.json

# 3. Process (extract + clean -> documents)
forge process  --input data/downloaded.json --extensions .go \
               --output data/documents.json

# 4. Generate (teacher model -> instruction pairs; --offline for CI)
forge generate --input data/documents.json --role go_explainer \
               --offline --output data/pairs.json

# 5. Build (validate, split, export, manifest)
forge build    --input data/pairs.json --name go_guild_v1 --val-ratio 0.1
```

`forge generate` accepts a comma-separated `--role` list and **hard budget
caps** for cost control (see below):

```bash
forge generate --input data/documents.json --role go_reviewer,go_generator \
               --max-pairs-per-doc 1 --max-pairs 4000 --max-spend-usd 4.0 \
               --output data/pairs.json
```

| Flag | Meaning | Default |
| --- | --- | --- |
| `--max-pairs-per-doc` | Pairs requested per (document, role). | `1` |
| `--max-pairs` | Hard cap: stop after N total pairs. | unset |
| `--max-spend-usd` | Hard cap: stop once estimated teacher spend (USD) hits this. | unset |

Generation stops *cleanly* at whichever cap is hit first and still writes
whatever it produced. Spend is estimated from the teacher's reported token usage.

There is also a `$0` curation route that skips the teacher entirely:

```bash
# Curate an existing open dataset, filter to Go, clean -> ready-to-build pairs.
forge import --dataset ise-uiuc/Magicoder-OSS-Instruct-75K --split train \
             --language go --max 3000 --output data/go_curated.json
forge build  --input data/go_curated.json --name go_curated_v1
```

### Teacher model configuration

Online generation talks to any OpenAI-compatible endpoint (vLLM, TGI, OpenAI,
Together, ...) via the `openai` SDK. Configure it with environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `FORGE_TEACHER_BASE_URL` | OpenAI-compatible base URL | `http://localhost:8000/v1` |
| `FORGE_TEACHER_API_KEY` | API key (any non-empty string for local servers) | `not-needed` |
| `FORGE_TEACHER_MODEL` | Model identifier | `teacher` |
| `FORGE_TEACHER_PRICE_IN` | USD per 1M input tokens (cost estimation) | `0.14` |
| `FORGE_TEACHER_PRICE_OUT` | USD per 1M output tokens (cost estimation) | `0.28` |

Drop `--offline` to use it. The offline mode produces deterministic synthetic
pairs so tests and CI never need a network or a teacher model. The price
defaults track DeepSeek-V3 (`deepseek-chat`) and drive the `--max-spend-usd` cap.

---

## Building a real dataset cheaply

The example config is an offline smoke-test. To produce a **real,
training-quality Go SFT dataset** there are two cheap routes — build either or
both. Both are honest: curated and teacher-generated data is genuinely useful
for fine-tuning. (We make no benchmark claims here — measure downstream.)

### Route A — curate existing open datasets (`$0`)

Reuse high-quality, already-instruction-tuned datasets from the HuggingFace Hub.
No teacher, no GPU, no spend. Needs the `datasets` library:

```bash
pip install 'guildlm-forge[hf]'

# Stream the dataset, keep only Go rows, clean (dedup/PII/length), write pairs.
forge import --dataset ise-uiuc/Magicoder-OSS-Instruct-75K --split train \
             --language go --max 3000 --output data/go_curated.json
forge build  --input data/go_curated.json --name go_curated_v1
```

Datasets are **streamed** (`streaming=True`) and stop as soon as `--max` matching
rows are collected, so multi-million-row datasets are never materialized. The
row-normalizer maps the common field conventions (`{instruction, output}`,
`{problem, solution}`, `{prompt, completion}`, chat `{messages: [...]}`, Alpaca
`{instruction, input, output}`) onto Forge's pair schema, and the language filter
keeps Go rows via an explicit `lang`/`language` field or a Go-source heuristic.

Recommended sources:

| Dataset | Notes |
| --- | --- |
| `ise-uiuc/Magicoder-OSS-Instruct-75K` | OSS-Instruct pairs; has a `lang` field — filter to Go. |
| `nvidia/OpenCodeInstruct` | ~5M rows, CC-BY-4.0; stream and filter to Go. |

The whole curate → clean → build flow also runs from one config (`mode: import`,
auto-detected for `source: hf_datasets`) — see `configs/go_curated.yaml`:

```bash
forge run --config configs/go_curated.yaml
```

### Route B — generate grounded pairs with a cheap teacher (~`$2-3`)

Discover top idiomatic Go repos, then prompt a cheap OpenAI-compatible teacher to
invent OSS-Instruct-style problems grounded in real snippets and answer them
idiomatically (concrete reviews, idiomatic code, table-driven tests, clear
explanations). Use **DeepSeek-V3**, which is strong and cheap:

```bash
export FORGE_TEACHER_BASE_URL=https://api.deepseek.com
export FORGE_TEACHER_MODEL=deepseek-chat
export FORGE_TEACHER_API_KEY=sk-...        # your DeepSeek key
export GITHUB_TOKEN=ghp_...                # lifts the 60 req/hr GitHub limit

forge run --config configs/go_reviewer_real.yaml
```

**Cost math.** DeepSeek-V3 is ~`$0.14` / `$1M` input and `$0.28` / `$1M` output
tokens. A grounded pair is roughly ~1–2k input + ~0.5–1k output tokens, i.e. a
fraction of a cent each; a few thousand pairs lands around **`$2-3`**. The recipe
hard-caps spend at `max_spend_usd: 4.0` and `max_pairs: 4000` — generation stops
cleanly at whichever is hit first and still builds whatever it made. Spend is
estimated live from the teacher's reported token usage.

---

## Quality gate — verified, judged data

Teacher output is *plausible*, not *guaranteed*. For code datasets the single
biggest quality lever is **execution verification**: actually compiling every
code example (and running tests for the tester role) instead of trusting that it
looks right. Forge adds a two-stage gate on top of generation:

1. **Execution verification** (`src/core/verifier.py`) — extract the Go from each
   teacher response, write it to a throwaway module, and run the local `go`
   toolchain: `go build ./...` and `go vet ./...`, plus `go test ./...` for the
   `go_tester` role. Candidates whose code does not compile are dropped. This is
   best-effort: if `go` is not on `PATH` the verifier reports `unavailable`
   instead of crashing, and `--strict-verify` controls whether unverifiable
   pairs are dropped or kept.
2. **Rubric judge** (`src/core/judge.py`) — a *cheap* LLM grades each surviving
   pair on correctness, idiomatic Go, completeness, and instruction↔response
   alignment, producing an overall score in `[0, 1]`. Pairs below
   `--judge-threshold` are dropped.

With **rejection sampling** (`--rejection-samples K`) Forge generates `K`
candidates per slot and keeps the best survivor (must pass verification; ranked
by judge score). All existing budget caps still apply, and **judge spend now
counts toward `--max-spend-usd`** too.

```bash
# Generate with the full gate.
forge generate --input data/documents.json --role go_reviewer,go_tester \
               --verify --judge --judge-threshold 0.6 --rejection-samples 2 \
               --max-spend-usd 4.0 --output data/pairs.json

# Or curate an EXISTING pairs file (e.g. Route-A data) through the same gate.
forge refine --input data/go_curated.json --output data/go_verified.json \
             --verify --judge --judge-threshold 0.6
```

| Flag (`generate` / `refine`) | Meaning | Default |
| --- | --- | --- |
| `--verify` / `--no-verify` | Compile-check extracted Go with the local toolchain. | off / on |
| `--strict-verify` | Drop pairs that can't be verified (no toolchain / no code). | off |
| `--judge` / `--no-judge` | Rubric-judge and filter pairs with a cheap LLM. | off / on |
| `--judge-threshold` | Minimum overall judge score to keep a pair. | `0.0` / `0.6` |
| `--rejection-samples` | Candidates per slot; the best survivor is kept (`generate` only). | `1` |

The judge reuses the teacher endpoint config and adds optional `FORGE_JUDGE_*`
overrides (`FORGE_JUDGE_BASE_URL`, `FORGE_JUDGE_API_KEY`, `FORGE_JUDGE_MODEL`,
`FORGE_JUDGE_PRICE_IN`, `FORGE_JUDGE_PRICE_OUT`); each falls back to the matching
`FORGE_TEACHER_*` value. Both stages have a deterministic `offline` mode so tests
and CI run with no network and no Go required.

**Recommended real recipe.** `configs/go_reviewer_real.yaml` now enables the gate
by default — `verify: true`, `judge: true`, `judge_threshold: 0.6`,
`rejection_samples: 2`. This is the recommended Route-B quality recipe.

**Be honest about what this buys you.** Verification *guarantees the code
compiles* (and that tests pass for the tester role) — nothing more. The judge
raises *average* quality but is itself a cheap model, not an oracle: it reduces
obvious junk, it does not certify correctness. We make **no benchmark claims** —
measure downstream.

## Config schema (`forge run`)

`forge run` supports two routes via `mode` (auto-detected from the source):
`generate` (discover → download → process → teacher → build) and `import`
(curate an existing instruction dataset; the teacher stage is skipped).

```yaml
source: github                       # registered source name (github | arxiv | hf_datasets)
mode: generate                       # generate | import (auto: import for hf_datasets)
query: "language:go stars:>2000"     # source-specific search expression
max_results: 5                       # cap on discovered items

download:
  output_dir: data/raw
  max_workers: 4                     # bounded concurrency

process:
  include_extensions: [".go"]
  min_length: 200                    # chars
  max_length: 50000                  # chars
  near_dup_threshold: 0.85           # MinHash Jaccard; >= this is a near-dup
  allow_unknown_license: true        # keep docs with no SPDX license

generate:
  offline: true                      # deterministic synthetic pairs
  roles: [go_explainer, go_reviewer] # one set of pairs per role, per document
  max_pairs_per_doc: 1
  max_pairs: 4000                    # optional hard cap: stop after N pairs
  max_spend_usd: 4.0                 # optional hard cap: stop at this estimated spend (teacher+judge)
  verify: false                      # execution-verify Go code with the local toolchain
  strict_verify: false               # drop pairs that can't be verified
  judge: false                       # rubric-judge and filter pairs
  judge_threshold: 0.0               # minimum judge score to keep a pair
  rejection_samples: 1               # candidates per slot; keep the best survivor

# Optional top-level stage: apply the quality gate to the produced pairs (works
# for both the generate and import routes — e.g. to curate Route-A data).
refine:
  verify: true
  strict_verify: false
  judge: true
  judge_threshold: 0.6

build:
  name: go_guild_v1
  output_dir: data/datasets
  val_ratio: 0.1
  seed: 42
  formats: ["jsonl"]                 # add "parquet" if pyarrow is installed
```

For `mode: import` (e.g. `source: hf_datasets`) the `download`/`generate`
sections are unused; instead provide the dataset keys:

```yaml
source: hf_datasets
mode: import                         # auto-detected for hf_datasets
dataset: ise-uiuc/Magicoder-OSS-Instruct-75K   # HuggingFace dataset id
split: train                         # dataset split to stream
language: go                         # language to keep
max_records: 3000                    # stop after this many matching pairs
process:                             # dedup/PII/length over the pair text
  min_length: 40
  max_length: 20000
  near_dup_threshold: 0.85
build: { name: go_curated_v1, val_ratio: 0.1 }
```

---

## Data schema (output)

Each line of the exported JSONL is one training record:

| Field | Type | Description |
| --- | --- | --- |
| `instruction` | string | Self-contained task for the student model. |
| `response` | string | Target completion from the teacher. |
| `context` | string | Grounding source snippet (may be empty). |
| `messages` | array | Chat transcript: `system` → `user` (instruction + context) → `assistant` (response). Ready for HF SFT trainers. |

A `<name>.manifest.json` accompanies every build with per-file SHA-256 hashes,
record counts, split sizes, the schema, and upstream cleaning stats.

---

## Extending Forge

### Add a new source

1. Create `src/sources/<name>.py` with a class subclassing
   `src.sources.base.Source`, setting `name` and implementing `search()` to
   return standardized record dicts (use `SourceRecord` to build them).
2. Register it in `src/sources/__init__.py` `SOURCE_REGISTRY` (or call
   `register_source(name, factory)` at runtime).

See `src/sources/arxiv.py` for a stdlib-only Atom-feed example.

### Add a new teacher role

Register a `Role` (system prompt + task description) in
`src/core/instruction_gen.py`:

```python
from src.core.instruction_gen import Role, register_role

register_role(Role("sql_optimizer", "You optimize SQL queries.", "rewrite the query for performance"))
```

### Add domain conventions

Put domain-specific extraction/filtering helpers under `src/plugins/`.

---

## Development

```bash
pip install -e '.[dev]'
ruff check .          # lint
pytest -q             # tests (offline, no network)
```

CI runs the same checks on Python 3.11 and 3.12 (`.github/workflows/ci.yml`).

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines. Licensed under
[Apache 2.0](./LICENSE).
