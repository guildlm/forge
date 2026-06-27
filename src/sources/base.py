"""Source abstractions for the Forge data pipeline.

A :class:`Source` is the only place in Forge that carries *platform* knowledge
(GitHub, arXiv, ...). The core engine treats every source uniformly through the
:meth:`Source.search` method, which returns a list of standardized
:class:`SourceRecord`-shaped dictionaries.

Records are plain ``dict`` objects (not dataclass instances) so they remain
trivially JSON-serializable as they flow through the pipeline. :class:`SourceRecord`
is provided as a typed helper for building those dictionaries consistently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = ["SourceRecord", "Source", "SourceProtocol"]


@dataclass
class SourceRecord:
    """A standardized metadata record describing a single discovered item.

    Attributes:
        source: Name of the originating source (e.g. ``"github"``).
        id: Stable identifier within the source (e.g. ``"owner/repo"`` or an
            arXiv id). Unique per source.
        url: Human-facing URL for the item.
        title: Short human-readable title.
        license: SPDX license identifier when known, else ``None``.
        fetch_url: URL or clone URL used by the downloader to retrieve content.
        extra: Source-specific fields (stars, authors, summary, ...).
    """

    source: str
    id: str
    url: str
    title: str = ""
    license: str | None = None
    fetch_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, JSON-serializable dictionary representation."""
        data = asdict(self)
        return data


@runtime_checkable
class SourceProtocol(Protocol):
    """Structural type describing anything usable as a Forge source."""

    name: str

    def search(self, query: str, max_results: int = 100, **kwargs: Any) -> list[dict[str, Any]]:
        ...


class Source(ABC):
    """Abstract base class implemented by every concrete data source.

    Concrete subclasses must set the :attr:`name` class attribute and implement
    :meth:`search`, returning a list of standardized record dictionaries (see
    :class:`SourceRecord`).
    """

    #: Unique, lowercase source name used in the registry and in records.
    name: str = ""

    @abstractmethod
    def search(self, query: str, max_results: int = 100, **kwargs: Any) -> list[dict[str, Any]]:
        """Discover items matching ``query``.

        Args:
            query: Source-specific search expression.
            max_results: Maximum number of records to return.
            **kwargs: Source-specific options.

        Returns:
            A list of standardized record dictionaries.
        """
        raise NotImplementedError
