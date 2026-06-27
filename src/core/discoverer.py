"""Source-agnostic discovery engine.

The :class:`Discoverer` resolves a source by name from the registry and delegates
the actual query to it. It holds no platform knowledge of its own.
"""

from __future__ import annotations

import logging
from typing import Any

from src.sources import get_source

logger = logging.getLogger(__name__)


class Discoverer:
    """Discover items from any registered :class:`~src.sources.base.Source`."""

    def discover(
        self, source_name: str, query: str, max_results: int = 100, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Discover items from ``source_name`` matching ``query``.

        Args:
            source_name: Registered source name (e.g. ``"github"``, ``"arxiv"``).
            query: Source-specific search expression.
            max_results: Maximum number of records to return.
            **kwargs: Forwarded to the source's ``search`` method.

        Returns:
            A list of standardized record dictionaries.
        """
        source = get_source(source_name)
        logger.info("Discovering via %r: %r", source_name, query)
        return source.search(query, max_results=max_results, **kwargs)

    def discover_code_guild_targets(
        self, language: str = "go", min_stars: int = 1000, max_results: int = 100
    ) -> list[dict[str, Any]]:
        """Convenience helper: high-quality repositories for a language.

        Excludes common tutorial/awesome-list noise to keep dataset quality high.
        """
        query = (
            f"language:{language} stars:>{min_stars} "
            f"NOT awesome NOT tutorial NOT 'learn {language}'"
        )
        return self.discover("github", query=query, max_results=max_results)
