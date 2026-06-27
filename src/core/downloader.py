"""Concurrent, polite content downloader.

Two retrieval strategies are provided over a shared bounded thread pool:

* :meth:`Downloader.fetch_all` -- HTTP GET many URLs (uses ``requests``).
* :meth:`Downloader.clone_all` -- shallow ``git clone`` many repositories.

Both enforce per-host politeness delays, retry transient failures with
exponential backoff, and return uniform :class:`DownloadResult` records.
Dependency-light by design: only ``requests`` plus the standard library.
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


@dataclass
class DownloadResult:
    """Outcome of a single download attempt."""

    id: str
    status: str  # "success" | "cached" | "failed"
    url: str | None = None
    local_path: str | None = None
    content: str | None = None
    error: str | None = None
    attempts: int = 0
    elapsed: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary (content omitted if large)."""
        return asdict(self)


class _HostThrottle:
    """Enforces a minimum interval between requests to the same host."""

    def __init__(self, min_interval: float) -> None:
        self.min_interval = min_interval
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}

    def wait(self, host: str) -> None:
        """Block until it is polite to hit ``host`` again."""
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            earliest = self._last.get(host, 0.0) + self.min_interval
            sleep_for = max(0.0, earliest - now)
            self._last[host] = max(now, earliest)
        if sleep_for > 0:
            time.sleep(sleep_for)


class Downloader:
    """Fetch URLs or clone repositories concurrently and politely."""

    def __init__(
        self,
        output_dir: str = "data/raw",
        *,
        max_workers: int = 4,
        max_retries: int = 3,
        timeout: float = 30.0,
        min_host_interval: float = 0.0,
        backoff_base: float = 0.5,
        session: requests.Session | None = None,
    ) -> None:
        """Initialize the downloader.

        Args:
            output_dir: Base directory for cloned repositories / downloaded files.
            max_workers: Bounded thread-pool size.
            max_retries: Maximum attempts per item before giving up.
            timeout: Per-request timeout in seconds.
            min_host_interval: Minimum seconds between requests to one host.
            backoff_base: Base seconds for exponential backoff between retries.
            session: Optional shared :class:`requests.Session` (e.g. for tests).
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self.backoff_base = backoff_base
        self.session = session or requests.Session()
        self._throttle = _HostThrottle(min_host_interval)
        os.makedirs(self.output_dir, exist_ok=True)

    # -- HTTP fetch ---------------------------------------------------------

    def fetch_all(self, items: list[dict[str, Any]]) -> list[DownloadResult]:
        """Fetch many URLs concurrently.

        Args:
            items: Records with at least ``id`` and ``url`` keys.

        Returns:
            One :class:`DownloadResult` per item.
        """
        return self._run(self._fetch_one, items, label="fetch")

    def _fetch_one(self, item: dict[str, Any]) -> DownloadResult:
        url = item.get("url") or item.get("fetch_url")
        identifier = str(item.get("id", url))
        if not url:
            return DownloadResult(id=identifier, status="failed", error="no url")

        host = urlparse(url).netloc
        start = time.monotonic()
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            self._throttle.wait(host)
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                return DownloadResult(
                    id=identifier,
                    status="success",
                    url=url,
                    content=response.text,
                    attempts=attempt,
                    elapsed=time.monotonic() - start,
                )
            except requests.RequestException as exc:
                last_error = str(exc)
                logger.warning("Fetch %s attempt %d/%d failed: %s",
                               identifier, attempt, self.max_retries, exc)
                self._sleep_backoff(attempt)
        return DownloadResult(
            id=identifier,
            status="failed",
            url=url,
            error=last_error,
            attempts=self.max_retries,
            elapsed=time.monotonic() - start,
        )

    # -- git clone ----------------------------------------------------------

    def clone_all(self, repositories: list[dict[str, Any]]) -> list[DownloadResult]:
        """Shallow-clone many repositories concurrently.

        Args:
            repositories: Records with ``id`` and a clone URL (``clone_url`` or
                ``fetch_url``).

        Returns:
            One :class:`DownloadResult` per repository.
        """
        return self._run(self._clone_one, repositories, label="clone")

    def _clone_one(self, repo: dict[str, Any]) -> DownloadResult:
        identifier = str(repo["id"])
        clone_url = repo.get("clone_url") or repo.get("fetch_url") or repo.get("url")
        safe_name = identifier.replace("/", "_")
        target = os.path.join(self.output_dir, safe_name)

        if os.path.isdir(target) and os.listdir(target):
            return DownloadResult(id=identifier, status="cached", url=clone_url, local_path=target)
        if not clone_url:
            return DownloadResult(id=identifier, status="failed", error="no clone url")

        host = urlparse(clone_url).netloc
        start = time.monotonic()
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            self._throttle.wait(host)
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", clone_url, target],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return DownloadResult(
                    id=identifier,
                    status="success",
                    url=clone_url,
                    local_path=target,
                    attempts=attempt,
                    elapsed=time.monotonic() - start,
                )
            except subprocess.CalledProcessError as exc:
                last_error = (exc.stderr or "").strip()
                logger.warning("Clone %s attempt %d/%d failed: %s",
                               identifier, attempt, self.max_retries, last_error)
                self._sleep_backoff(attempt)
        return DownloadResult(
            id=identifier,
            status="failed",
            url=clone_url,
            error=last_error,
            attempts=self.max_retries,
            elapsed=time.monotonic() - start,
        )

    # -- shared helpers -----------------------------------------------------

    def _sleep_backoff(self, attempt: int) -> None:
        """Sleep using exponential backoff with jitter (skip after last attempt)."""
        if attempt >= self.max_retries:
            return
        delay = self.backoff_base * (2 ** (attempt - 1))
        time.sleep(delay + random.uniform(0, self.backoff_base))

    def _run(self, worker, items, *, label: str) -> list[DownloadResult]:
        logger.info("Starting %s of %d item(s) (workers=%d).", label, len(items), self.max_workers)
        results: list[DownloadResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(worker, item): item for item in items}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Unexpected error processing %s", item.get("id"))
                    results.append(
                        DownloadResult(id=str(item.get("id", "?")), status="failed", error=str(exc))
                    )
        ok = sum(1 for r in results if r.status in ("success", "cached"))
        logger.info("%s complete: %d/%d succeeded.", label.capitalize(), ok, len(items))
        return results
