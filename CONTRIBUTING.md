# Contributing to Forge

Thanks for your interest in improving Forge, the data pipeline for GuildLM!

## Ground rules

- **Keep the core domain-agnostic.** The pipeline engine (`src/core/`) must not
  hard-code any platform or domain. New platforms go in `src/sources/`; new
  domain conventions go in `src/plugins/`.
- **Dependency-light.** Prefer the standard library. New runtime dependencies
  need a clear justification; heavy/optional ones go behind extras in
  `pyproject.toml` with a guarded import.
- **Everything must run offline.** Tests and CI must pass with no network and
  only the dev dependencies. Network-dependent paths get an offline/deterministic
  mode (see `InstructionGenerator(offline=True)`).

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## Before opening a PR

```bash
ruff check .     # lint must be clean
pytest -q        # all tests must pass
```

- Add or update tests for any behaviour you change.
- Use type hints and concise docstrings on public functions and classes.
- Add logging (not `print`) for pipeline-relevant events.
- Keep functions focused; avoid dead code and leftover scaffolding.

## Commit messages

Write clear, imperative commit subjects (e.g. "Add arXiv source"). Reference
related issues where relevant.

## Adding a source

1. Subclass `src.sources.base.Source`, set `name`, implement `search()`.
2. Register it in `SOURCE_REGISTRY` (`src/sources/__init__.py`).
3. Add a parser unit test using a saved sample payload (see
   `tests/test_sources.py` and `tests/data/arxiv_sample.xml`).

## License

By contributing, you agree that your contributions are licensed under the
Apache License 2.0.
