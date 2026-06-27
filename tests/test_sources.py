"""Tests for sources: arXiv parser, ABC conformance, registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.sources import ArxivSource, GitHubSource, get_source
from src.sources.arxiv import ARXIV_LICENSE
from src.sources.base import Source, SourceProtocol

SAMPLE_XML = (Path(__file__).parent / "data" / "arxiv_sample.xml").read_text(encoding="utf-8")


def test_arxiv_parser_extracts_records() -> None:
    records = ArxivSource.parse_feed(SAMPLE_XML)
    assert len(records) == 2

    first = records[0]
    assert first["source"] == "arxiv"
    assert first["id"] == "2106.01345v2"
    assert first["title"] == "Decision Transformer: Reinforcement Learning via Sequence Modeling"
    assert first["license"] == ARXIV_LICENSE
    assert first["fetch_url"] == "http://arxiv.org/pdf/2106.01345v2"
    assert first["extra"]["authors"] == ["Lili Chen", "Kevin Lu"]
    assert first["extra"]["primary_category"] == "cs.LG"
    assert "cs.AI" in first["extra"]["categories"]
    assert first["extra"]["summary"].startswith("We introduce a framework")


def test_arxiv_parser_empty_feed() -> None:
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert ArxivSource.parse_feed(empty) == []


def test_sources_conform_to_abc_and_protocol() -> None:
    for cls in (GitHubSource, ArxivSource):
        assert issubclass(cls, Source)
        instance = cls()
        assert isinstance(instance, SourceProtocol)
        assert isinstance(instance.name, str) and instance.name
        assert callable(instance.search)


def test_registry_resolves_known_sources() -> None:
    assert isinstance(get_source("github"), GitHubSource)
    assert isinstance(get_source("arxiv"), ArxivSource)


def test_registry_rejects_unknown_source() -> None:
    with pytest.raises(ValueError):
        get_source("does-not-exist")
