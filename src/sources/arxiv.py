"""arXiv discovery source.

Queries the public arXiv export API (``export.arxiv.org/api/query``) and parses
the returned Atom feed with the Python standard library only. No third-party
XML or arXiv client is required.
"""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from src.sources.base import Source, SourceRecord

logger = logging.getLogger(__name__)

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# arXiv content is distributed under non-OSI terms; we tag it explicitly rather
# than guessing an SPDX id so downstream license filtering stays honest.
ARXIV_LICENSE = "arXiv-nonexclusive"


class ArxivSource(Source):
    """Discover papers on arXiv via the Atom export API."""

    name = "arxiv"
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, request_delay: float = 3.0, user_agent: str = "guildlm-forge/0.1") -> None:
        """Initialize the source.

        Args:
            request_delay: Seconds to sleep between API requests. arXiv requests a
                minimum of 3 seconds between calls.
            user_agent: User-Agent header sent with each request.
        """
        self.request_delay = request_delay
        self.user_agent = user_agent

    def search(
        self,
        query: str,
        max_results: int = 100,
        *,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        page_size: int = 100,
        **_: Any,
    ) -> list[dict[str, Any]]:
        """Search arXiv for ``query``.

        Args:
            query: arXiv search expression, e.g. ``"cat:cs.LG AND ti:transformer"``.
            max_results: Maximum number of records to return.
            sort_by: ``relevance``, ``lastUpdatedDate`` or ``submittedDate``.
            sort_order: ``ascending`` or ``descending``.
            page_size: Results requested per API call (arXiv caps at ~2000).

        Returns:
            Standardized record dictionaries.
        """
        records: list[dict[str, Any]] = []
        start = 0
        page_size = min(page_size, max(1, max_results))

        logger.info("Searching arXiv for %r (max %d).", query, max_results)
        while len(records) < max_results:
            params = {
                "search_query": query,
                "start": start,
                "max_results": min(page_size, max_results - len(records)),
                "sortBy": sort_by,
                "sortOrder": sort_order,
            }
            xml = self._fetch(params)
            page = self.parse_feed(xml)
            if not page:
                break
            records.extend(page)
            start += len(page)
            if len(page) < params["max_results"]:
                break
            time.sleep(self.request_delay)

        records = records[:max_results]
        logger.info("Discovered %d arXiv papers.", len(records))
        return records

    def _fetch(self, params: dict[str, Any]) -> str:
        """Perform a single HTTP GET against the export API, returning XML text."""
        url = f"{self.BASE_URL}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=30) as response:  # noqa: S310 (trusted host)
            return response.read().decode("utf-8")

    @staticmethod
    def parse_feed(xml_text: str) -> list[dict[str, Any]]:
        """Parse an arXiv Atom feed into standardized record dictionaries.

        This is a pure function (no network) so it is unit-testable against a
        saved sample feed.

        Args:
            xml_text: Raw Atom XML returned by the arXiv export API.

        Returns:
            A list of standardized record dictionaries.
        """
        root = ET.fromstring(xml_text)
        records: list[dict[str, Any]] = []
        for entry in root.findall(f"{_ATOM}entry"):
            arxiv_url = (entry.findtext(f"{_ATOM}id") or "").strip()
            arxiv_id = arxiv_url.rsplit("/", 1)[-1] if arxiv_url else ""
            title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
            summary = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())

            authors = [
                (a.findtext(f"{_ATOM}name") or "").strip()
                for a in entry.findall(f"{_ATOM}author")
            ]
            categories = [
                c.get("term", "")
                for c in entry.findall(f"{_ATOM}category")
                if c.get("term")
            ]

            pdf_url = None
            for link in entry.findall(f"{_ATOM}link"):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
                    break

            record = SourceRecord(
                source="arxiv",
                id=arxiv_id,
                url=arxiv_url,
                title=title,
                license=ARXIV_LICENSE,
                fetch_url=pdf_url or arxiv_url,
                extra={
                    "summary": summary,
                    "authors": authors,
                    "categories": categories,
                    "primary_category": (
                        entry.find(f"{_ARXIV}primary_category").get("term")
                        if entry.find(f"{_ARXIV}primary_category") is not None
                        else None
                    ),
                    "published": (entry.findtext(f"{_ATOM}published") or "").strip(),
                    "updated": (entry.findtext(f"{_ATOM}updated") or "").strip(),
                    "pdf_url": pdf_url,
                },
            )
            records.append(record.to_dict())
        return records
