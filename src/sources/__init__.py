"""Forge data sources and their registry.

Sources carry all platform-specific knowledge. The core engine resolves them by
name through :func:`get_source` / :data:`SOURCE_REGISTRY`.
"""

from __future__ import annotations

from collections.abc import Callable

from src.sources.arxiv import ArxivSource
from src.sources.base import Source, SourceProtocol, SourceRecord
from src.sources.github import GitHubSource

__all__ = [
    "Source",
    "SourceProtocol",
    "SourceRecord",
    "GitHubSource",
    "ArxivSource",
    "SOURCE_REGISTRY",
    "get_source",
    "register_source",
]

#: Maps a source name to a zero-argument factory that constructs it.
SOURCE_REGISTRY: dict[str, Callable[[], Source]] = {
    GitHubSource.name: GitHubSource,
    ArxivSource.name: ArxivSource,
}


def register_source(name: str, factory: Callable[[], Source]) -> None:
    """Register a new source factory under ``name`` (idempotent overwrite)."""
    SOURCE_REGISTRY[name] = factory


def get_source(name: str) -> Source:
    """Instantiate a registered source by name.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    try:
        factory = SOURCE_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(SOURCE_REGISTRY))
        raise ValueError(f"Unknown source {name!r}. Available: {available}") from None
    return factory()
