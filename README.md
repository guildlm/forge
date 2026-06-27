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
pip install -e '.[dev]'          # add ',parquet' for Parquet export

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

### Teacher model configuration

Online generation talks to any OpenAI-compatible endpoint (vLLM, TGI, OpenAI,
Together, ...) via the `openai` SDK. Configure it with environment variables:

| Variable | Meaning | Default |
| --- | --- | --- |
| `FORGE_TEACHER_BASE_URL` | OpenAI-compatible base URL | `http://localhost:8000/v1` |
| `FORGE_TEACHER_API_KEY` | API key (any non-empty string for local servers) | `not-needed` |
| `FORGE_TEACHER_MODEL` | Model identifier | `teacher` |

Drop `--offline` to use it. The offline mode produces deterministic synthetic
pairs so tests and CI never need a network or a teacher model.

---

## Config schema (`forge run`)

```yaml
source: github                       # registered source name (github | arxiv)
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

build:
  name: go_guild_v1
  output_dir: data/datasets
  val_ratio: 0.1
  seed: 42
  formats: ["jsonl"]                 # add "parquet" if pyarrow is installed
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
