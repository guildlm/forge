"""Execution verification of teacher-generated Go code.

The :class:`GoVerifier` is the core quality lever of the forge quality gate: it
takes a teacher *response* string, extracts the Go code it contains, and proves
that code actually **compiles** (and, for the ``go_tester`` role, that its tests
**pass**) by invoking the local ``go`` toolchain in an isolated temp module.

Two layers keep this testable and safe:

* :func:`extract_go_code` is pure Python (no subprocess, no toolchain) and is
  fully unit-testable -- it parses ```` ```go ```` fenced blocks (falling back to
  bare ```` ``` ```` fences).
* The subprocess boundary is an injectable ``runner`` callable, so tests cover
  the pass / fail / unavailable / no-code paths *without* needing Go installed.

The toolchain is treated as best-effort: if ``go`` is not on ``PATH`` the
verifier returns ``status="unavailable"`` rather than crashing. A ``strict``
flag controls policy -- ``strict=True`` drops ``unavailable`` / ``no_code``
pairs, ``strict=False`` keeps them.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Cap on captured diagnostics so failure output never bloats the dataset/logs.
DIAGNOSTICS_LIMIT = 4_000

#: Default per-check timeout in seconds for a single ``go`` invocation.
DEFAULT_TIMEOUT = 60.0

#: Signature of an injectable subprocess runner.
#: ``runner(cmd, *, cwd, env, timeout) -> (returncode, combined_output)``.
Runner = Callable[..., "tuple[int, str]"]

_GO_FENCE = re.compile(r"```go\b[^\n]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
# Detects an unfenced response that is itself Go source.
_RAW_GO = re.compile(r"^\s*package\s+\w+.*\bfunc\b", re.DOTALL)
_TEST_MARKER = re.compile(r'^\s*func\s+(?:Test|Benchmark|Example|Fuzz)\w*\s*\(', re.MULTILINE)
_TESTING_IMPORT = re.compile(r'"testing"')


def extract_go_code(text: str) -> list[str]:
    """Extract Go code blocks from a teacher response.

    Prefers explicitly tagged ```` ```go ```` fences; if there are none, falls
    back to any fenced ```` ``` ```` block (teachers sometimes omit the language
    tag); as a last resort, if the *whole* response is unfenced but looks like Go
    source (declares a ``package`` and at least one ``func``), the entire text is
    treated as one block. Only the inner code is returned, with surrounding blank
    lines trimmed.

    Args:
        text: The raw teacher response.

    Returns:
        A list of code blocks (possibly empty). Order is preserved.
    """
    if not text:
        return []
    blocks = [b.strip("\n") for b in _GO_FENCE.findall(text)]
    blocks = [b for b in blocks if b.strip()]
    if blocks:
        return blocks
    fallback = [b.strip("\n") for b in _ANY_FENCE.findall(text)]
    fallback = [b for b in fallback if b.strip()]
    if fallback:
        return fallback
    # Last resort: many teachers return the answer as raw Go with no fences.
    if _RAW_GO.search(text):
        return [text.strip("\n")]
    return []


def _looks_like_test(block: str) -> bool:
    """Whether a code block is a Go test file (needs the ``_test.go`` suffix)."""
    return bool(_TEST_MARKER.search(block)) or bool(_TESTING_IMPORT.search(block))


@dataclass
class VerifyResult:
    """Outcome of verifying one teacher response.

    Attributes:
        passed: Whether the pair survives the gate under the active policy.
        stage: The check that produced the verdict
            (``"extract"``, ``"toolchain"``, ``"mod"``, ``"build"``, ``"vet"``,
            ``"test"`` or ``"ok"``).
        status: One of ``"ok"``, ``"failed"``, ``"unavailable"``, ``"no_code"``.
        diagnostics: Toolchain output for a failure (truncated), else ``""``.
    """

    passed: bool
    stage: str
    status: str
    diagnostics: str = ""


def _default_runner(cmd: list[str], *, cwd: str, env: dict[str, str], timeout: float) -> tuple[int, str]:
    """Run a command, returning ``(returncode, stdout+stderr)``.

    Timeouts and a missing executable are reported as non-zero return codes
    rather than raised, so the verifier degrades gracefully.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - trusted, fixed go subcommands
            cmd,
            cwd=cwd,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s: {' '.join(cmd)}"
    except FileNotFoundError as exc:  # pragma: no cover - exercised only without go
        return 127, str(exc)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


class GoVerifier:
    """Verify extracted Go code against the local ``go`` toolchain."""

    def __init__(
        self,
        *,
        strict: bool = False,
        timeout: float = DEFAULT_TIMEOUT,
        runner: Runner | None = None,
        go_available: bool | None = None,
        module_name: str = "forgeverify",
    ) -> None:
        """Configure verification behaviour.

        Args:
            strict: When ``True``, ``unavailable`` and ``no_code`` results fail
                the gate; when ``False`` they pass (best-effort verification).
            timeout: Per-check timeout in seconds.
            runner: Injectable subprocess runner (defaults to a real
                :func:`subprocess.run` wrapper). Tests pass a stub to exercise
                every path without Go installed.
            go_available: Force toolchain availability for tests. When ``None``
                it is auto-detected via :func:`shutil.which`.
            module_name: Module path used for the throwaway ``go.mod``.
        """
        self.strict = strict
        self.timeout = timeout
        self._runner: Runner = runner or _default_runner
        self._go_available = go_available
        self.module_name = module_name

    # -- availability -------------------------------------------------------

    def available(self) -> bool:
        """Whether a ``go`` toolchain is usable for verification."""
        if self._go_available is not None:
            return self._go_available
        return shutil.which("go") is not None

    # -- public API ---------------------------------------------------------

    def verify(self, response: str, role: str = "") -> VerifyResult:
        """Extract and execution-verify the Go code in a teacher ``response``.

        Args:
            response: The teacher response text containing fenced Go code.
            role: The teacher role; ``"go_tester"`` additionally runs
                ``go test ./...``.

        Returns:
            A :class:`VerifyResult`. The ``passed`` flag already reflects the
            ``strict`` policy, so callers can simply drop pairs where it is
            ``False``.
        """
        blocks = extract_go_code(response)
        if not blocks:
            return VerifyResult(
                passed=not self.strict,
                stage="extract",
                status="no_code",
                diagnostics="no Go code block found in response",
            )
        if not self.available():
            return VerifyResult(
                passed=not self.strict,
                stage="toolchain",
                status="unavailable",
                diagnostics="go toolchain not found on PATH",
            )
        return self._run_checks(blocks, role)

    # -- internals ----------------------------------------------------------

    def _env(self) -> dict[str, str]:
        """Toolchain environment: module mode, offline, no network."""
        env = dict(os.environ)
        env.update(
            {
                "GOFLAGS": "-mod=mod",
                "GOPROXY": "off",
                "GOSUMDB": "off",
                "GOTOOLCHAIN": "local",
                "GO111MODULE": "on",
                "CGO_ENABLED": "0",
            }
        )
        return env

    def _write_module(self, work_dir: str, blocks: list[str]) -> None:
        """Write each code block to its own file (test blocks get ``_test.go``)."""
        for index, block in enumerate(blocks):
            suffix = "_test.go" if _looks_like_test(block) else ".go"
            path = os.path.join(work_dir, f"code{index}{suffix}")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(block if block.endswith("\n") else block + "\n")

    def _run_checks(self, blocks: list[str], role: str) -> VerifyResult:
        with tempfile.TemporaryDirectory(prefix="forge-verify-") as work_dir:
            env = self._env()
            self._write_module(work_dir, blocks)

            rc, out = self._runner(
                ["go", "mod", "init", self.module_name], cwd=work_dir, env=env, timeout=self.timeout
            )
            if rc != 0:
                return VerifyResult(False, "mod", "failed", _clip(out))

            checks: list[tuple[str, list[str]]] = [
                ("build", ["go", "build", "./..."]),
                ("vet", ["go", "vet", "./..."]),
            ]
            if role == "go_tester":
                checks.append(("test", ["go", "test", "./..."]))

            for stage, cmd in checks:
                rc, out = self._runner(cmd, cwd=work_dir, env=env, timeout=self.timeout)
                if rc != 0:
                    return VerifyResult(False, stage, "failed", _clip(out))

        return VerifyResult(True, "ok", "ok", "")


def _clip(text: str) -> str:
    """Trim toolchain output to :data:`DIAGNOSTICS_LIMIT` characters."""
    text = (text or "").strip()
    if len(text) <= DIAGNOSTICS_LIMIT:
        return text
    return text[:DIAGNOSTICS_LIMIT] + "\n... [truncated]"
