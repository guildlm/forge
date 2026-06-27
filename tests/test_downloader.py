"""Tests for the downloader's throttle and result records (no network)."""

from __future__ import annotations

import time

from src.core.downloader import Downloader, DownloadResult, _HostThrottle


def test_host_throttle_enforces_interval() -> None:
    throttle = _HostThrottle(min_interval=0.05)
    start = time.monotonic()
    throttle.wait("example.com")
    throttle.wait("example.com")
    throttle.wait("example.com")
    elapsed = time.monotonic() - start
    assert elapsed >= 0.10  # two enforced gaps of 0.05s


def test_host_throttle_independent_hosts() -> None:
    throttle = _HostThrottle(min_interval=0.05)
    start = time.monotonic()
    throttle.wait("a.com")
    throttle.wait("b.com")
    assert time.monotonic() - start < 0.05


def test_download_result_serializable(tmp_path) -> None:
    result = DownloadResult(id="x/y", status="success", local_path="/tmp/x")
    data = result.to_dict()
    assert data["id"] == "x/y"
    assert data["status"] == "success"


def test_fetch_missing_url_fails_cleanly(tmp_path) -> None:
    dl = Downloader(str(tmp_path), max_retries=1)
    results = dl.fetch_all([{"id": "nourl"}])
    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error == "no url"
