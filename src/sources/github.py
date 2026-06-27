"""GitHub repository discovery source.

Uses the GitHub REST search API to find repositories matching a query and
returns standardized :class:`~src.sources.base.SourceRecord` dictionaries.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

from src.sources.base import Source, SourceRecord

logger = logging.getLogger(__name__)


class GitHubSource(Source):
    """Discover repositories on GitHub using the REST search API.

    Handles pagination, primary rate limiting, and output standardization. A
    Personal Access Token (``GITHUB_TOKEN``) is strongly recommended to lift the
    unauthenticated 60 requests/hour limit.
    """

    name = "github"
    BASE_URL = "https://api.github.com"

    def __init__(self, token: str | None = None, session: requests.Session | None = None) -> None:
        """Initialize the source.

        Args:
            token: GitHub Personal Access Token. Falls back to the ``GITHUB_TOKEN``
                environment variable.
            session: Optional pre-configured :class:`requests.Session` (useful for
                testing). A new session is created when omitted.
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = session or requests.Session()
        self.session.headers.setdefault("Accept", "application/vnd.github.v3+json")
        if self.token:
            self.session.headers["Authorization"] = f"token {self.token}"
        else:
            logger.warning(
                "No GITHUB_TOKEN provided; GitHub API requests are limited to 60/hour."
            )

    def _handle_rate_limit(self, response: requests.Response) -> bool:
        """Sleep until the rate-limit window resets. Returns True if it slept."""
        if response.status_code == 403 and "rate limit" in response.text.lower():
            reset_time = int(response.headers.get("X-RateLimit-Reset", "0"))
            sleep_for = max(0, reset_time - int(time.time())) + 5
            logger.warning("GitHub rate limit hit; sleeping for %ss.", sleep_for)
            time.sleep(sleep_for)
            return True
        return False

    def search(
        self,
        query: str,
        max_results: int = 100,
        *,
        sort: str = "stars",
        order: str = "desc",
        page_delay: float = 2.0,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Search repositories matching ``query``.

        Args:
            query: GitHub search syntax, e.g. ``"language:go stars:>1000"``.
            max_results: Maximum repositories to return.
            sort: Sort field (``stars``, ``forks``, ``updated``).
            order: ``asc`` or ``desc``.
            page_delay: Seconds to sleep between pages (search API politeness).

        Returns:
            Standardized record dictionaries.
        """
        endpoint = f"{self.BASE_URL}/search/repositories"
        per_page = min(100, max(1, max_results))
        page = 1
        raw: list[dict[str, Any]] = []

        logger.info("Searching GitHub for %r (max %d).", query, max_results)
        while len(raw) < max_results:
            params = {
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": per_page,
                "page": page,
            }
            response = self.session.get(endpoint, params=params, timeout=30)
            if self._handle_rate_limit(response):
                continue
            response.raise_for_status()

            items = response.json().get("items", [])
            if not items:
                break
            raw.extend(items)
            if len(items) < per_page:
                break
            page += 1
            time.sleep(page_delay)

        records = [self._standardize(repo) for repo in raw[:max_results]]
        logger.info("Discovered %d repositories.", len(records))
        return records

    @staticmethod
    def _standardize(repo: dict[str, Any]) -> dict[str, Any]:
        """Convert a raw GitHub repo payload into a standardized record dict."""
        license_block = repo.get("license") or {}
        record = SourceRecord(
            source="github",
            id=repo["full_name"],
            url=repo["html_url"],
            title=repo.get("description") or repo["full_name"],
            license=license_block.get("spdx_id"),
            fetch_url=repo["clone_url"],
            extra={
                "clone_url": repo["clone_url"],
                "default_branch": repo.get("default_branch"),
                "stars": repo.get("stargazers_count"),
                "language": repo.get("language"),
                "updated_at": repo.get("updated_at"),
            },
        )
        return record.to_dict()

    def search_repositories(self, query: str, max_results: int = 100, **kwargs: Any) -> list[dict[str, Any]]:
        """Deprecated alias for :meth:`search`, kept for backwards compatibility."""
        return self.search(query, max_results, **kwargs)
