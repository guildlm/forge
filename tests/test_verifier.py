"""Tests for execution verification: extraction, mocked runner, real toolchain."""

from __future__ import annotations

import shutil

import pytest

from src.core.verifier import GoVerifier, VerifyResult, extract_go_code

VALID_GO = """package main

import "fmt"

func Add(a, b int) int { return a + b }

func main() { fmt.Println(Add(1, 2)) }
"""

BROKEN_GO = """package main

func main() {
    x := // syntax error, no value
}
"""

GO_WITH_TEST = """package calc

func Add(a, b int) int { return a + b }
"""

GO_TEST_FILE = """package calc

import "testing"

func TestAdd(t *testing.T) {
    if Add(1, 2) != 3 {
        t.Fatalf("want 3")
    }
}
"""


# --- extract_go_code -------------------------------------------------------- #


def test_extract_go_fenced_block() -> None:
    text = f"Here is code:\n```go\n{VALID_GO}```\nDone."
    blocks = extract_go_code(text)
    assert len(blocks) == 1
    assert "func Add" in blocks[0]
    assert not blocks[0].startswith("\n")


def test_extract_prefers_go_over_bare_fence() -> None:
    text = "```python\nprint('x')\n```\n```go\npackage main\n```"
    blocks = extract_go_code(text)
    assert blocks == ["package main"]


def test_extract_falls_back_to_bare_fence() -> None:
    text = "```\npackage main\nfunc main() {}\n```"
    blocks = extract_go_code(text)
    assert len(blocks) == 1
    assert "package main" in blocks[0]


def test_extract_multiple_go_blocks_preserves_order() -> None:
    text = "```go\npackage a\n```\nmiddle\n```go\npackage b\n```"
    assert extract_go_code(text) == ["package a", "package b"]


def test_extract_no_code_returns_empty() -> None:
    assert extract_go_code("just prose, no fences") == []
    assert extract_go_code("") == []


# --- mocked runner: ok / failed -------------------------------------------- #


def _runner_all_ok(cmd, *, cwd, env, timeout):  # noqa: ANN001 - test stub
    return 0, ""


def _runner_fail_on(stage_cmd):
    """Build a runner that fails (rc=1) when the command contains ``stage_cmd``."""

    def runner(cmd, *, cwd, env, timeout):  # noqa: ANN001 - test stub
        if stage_cmd in cmd:
            return 1, f"{stage_cmd}: simulated failure"
        return 0, ""

    return runner


def test_verify_ok_with_mocked_runner() -> None:
    v = GoVerifier(runner=_runner_all_ok, go_available=True)
    result = v.verify(f"```go\n{VALID_GO}```", role="go_generator")
    assert isinstance(result, VerifyResult)
    assert result.passed
    assert result.status == "ok"
    assert result.stage == "ok"


def test_verify_failed_build() -> None:
    v = GoVerifier(runner=_runner_fail_on("build"), go_available=True)
    result = v.verify(f"```go\n{BROKEN_GO}```")
    assert not result.passed
    assert result.status == "failed"
    assert result.stage == "build"
    assert "simulated failure" in result.diagnostics


def test_verify_tester_role_runs_test_stage() -> None:
    v = GoVerifier(runner=_runner_fail_on("test"), go_available=True)
    result = v.verify(f"```go\n{GO_WITH_TEST}```\n```go\n{GO_TEST_FILE}```", role="go_tester")
    assert not result.passed
    assert result.stage == "test"


def test_verify_non_tester_role_skips_test_stage() -> None:
    # A runner that fails only on `go test` should NOT affect a non-tester role.
    v = GoVerifier(runner=_runner_fail_on("test"), go_available=True)
    result = v.verify(f"```go\n{VALID_GO}```", role="go_generator")
    assert result.passed


# --- unavailable / no_code + strict policy --------------------------------- #


def test_unavailable_kept_when_not_strict() -> None:
    v = GoVerifier(go_available=False, strict=False)
    result = v.verify(f"```go\n{VALID_GO}```")
    assert result.status == "unavailable"
    assert result.passed  # best-effort: kept


def test_unavailable_dropped_when_strict() -> None:
    v = GoVerifier(go_available=False, strict=True)
    result = v.verify(f"```go\n{VALID_GO}```")
    assert result.status == "unavailable"
    assert not result.passed


def test_no_code_kept_when_not_strict() -> None:
    v = GoVerifier(go_available=True, strict=False, runner=_runner_all_ok)
    result = v.verify("a response with no code blocks at all")
    assert result.status == "no_code"
    assert result.passed


def test_no_code_dropped_when_strict() -> None:
    v = GoVerifier(go_available=True, strict=True, runner=_runner_all_ok)
    result = v.verify("a response with no code blocks at all")
    assert result.status == "no_code"
    assert not result.passed


# --- real toolchain (only when `go` is installed) -------------------------- #


@pytest.mark.skipif(shutil.which("go") is None, reason="go toolchain not installed")
def test_real_verifier_end_to_end() -> None:
    v = GoVerifier(timeout=120.0)
    assert v.available()

    ok = v.verify(f"```go\n{VALID_GO}```", role="go_generator")
    assert ok.passed, ok.diagnostics
    assert ok.status == "ok"

    bad = v.verify(f"```go\n{BROKEN_GO}```", role="go_generator")
    assert not bad.passed
    assert bad.status == "failed"

    tested = v.verify(
        f"```go\n{GO_WITH_TEST}```\n```go\n{GO_TEST_FILE}```", role="go_tester"
    )
    assert tested.passed, tested.diagnostics
