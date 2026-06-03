"""Tests for D-182 P-AUTORESUME-OFF-SPEC-WARNING remediation in tools/run_long.py.

Covers:
  - Spec-sentinel helpers (write/read/verify) in isolation.
  - verify_spec_sentinel returns ok=True on a fresh ckpt-dir (pre-launch path
    for a clean run; no refusal).
  - verify_spec_sentinel returns ok=False on legacy ckpt-dir without sentinel
    (the D-155 scenario: ckpts present, no sentinel, refuse to launch).
  - verify_spec_sentinel returns ok=False on spec-fingerprint mismatch (the
    Phase-respec scenario: ckpts exist but manifest SHA changed, refuse).
  - verify_spec_sentinel returns ok=True on matching sentinel (expected
    re-launch path; auto-resume proceeds).
  - compute_spec_fingerprint degrades gracefully on missing manifest / unset
    LAB_PROGRAM (records explicit "<none>" marker instead of crashing).

Does not exercise the full subprocess flow (no training binary spawned in
tests). The legacy rc=0+WARNING → rc=2 fail-fast branch is left untested by
this file because that path is D-125 scope and was already vetted; D-182
adds the pre-launch sentinel which is what this file covers.
"""

from __future__ import annotations

import importlib
import os
import pathlib
import sys
from typing import Iterator

import pytest

TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

run_long = importlib.import_module("run_long")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ckpt_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return an empty, real-on-disk ckpt dir for each test."""
    d = tmp_path / "ckpts"
    d.mkdir()
    return d


@pytest.fixture()
def fake_program(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create a fake programs/<name>/spec_invariants.yaml + point run_long at it.

    Returns the program name. The fake ROOT is injected via monkeypatch so
    compute_spec_fingerprint picks up the fake manifest instead of the real
    Program-2 one.
    """
    program = "program_test_autoresume"
    fake_root = tmp_path / "fake_root"
    manifest_dir = fake_root / "programs" / program
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "spec_invariants.yaml"
    manifest.write_text(
        "arm = test\n"
        "model = test\n"
        "tokenizer_basename = test.bin\n"
        "vocab_size = 100\n"
        "param_count_total = 1000\n"
    )
    monkeypatch.setattr(run_long, "ROOT", str(fake_root))
    return program


def _touch_ckpt(ckpt_dir: pathlib.Path, step: int) -> pathlib.Path:
    """Create a minimally-named ckpt file the launcher treats as resumable."""
    path = ckpt_dir / f"step_{step:06d}.ckpt"
    path.write_bytes(b"\x00" * 16)
    return path


# ---------------------------------------------------------------------------
# compute_spec_fingerprint
# ---------------------------------------------------------------------------


def test_compute_spec_fingerprint_empty_program() -> None:
    """Unset LAB_PROGRAM → explicit '<none>:<none>' marker, no crash."""
    assert run_long.compute_spec_fingerprint("") == "<none>:<none>"


def test_compute_spec_fingerprint_missing_manifest(tmp_path: pathlib.Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """Program name set but no manifest file → '<program>:<missing-manifest>'."""
    fake_root = tmp_path / "fake_root"
    fake_root.mkdir()
    monkeypatch.setattr(run_long, "ROOT", str(fake_root))
    fp = run_long.compute_spec_fingerprint("nonexistent_program")
    assert fp == "nonexistent_program:<missing-manifest>"


def test_compute_spec_fingerprint_deterministic(fake_program: str) -> None:
    """Same manifest content → same fingerprint across calls."""
    fp1 = run_long.compute_spec_fingerprint(fake_program)
    fp2 = run_long.compute_spec_fingerprint(fake_program)
    assert fp1 == fp2
    assert fp1.startswith(f"{fake_program}:")
    # 12-char sha1 prefix
    _, digest = fp1.split(":", 1)
    assert len(digest) == 12
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# Sentinel write / read round-trip
# ---------------------------------------------------------------------------


def test_sentinel_roundtrip(ckpt_dir: pathlib.Path) -> None:
    run_long.write_sentinel(str(ckpt_dir), "prog_x:abcdef012345")
    assert run_long.read_sentinel(str(ckpt_dir)) == "prog_x:abcdef012345"


def test_read_sentinel_missing_returns_none(ckpt_dir: pathlib.Path) -> None:
    assert run_long.read_sentinel(str(ckpt_dir)) is None


# ---------------------------------------------------------------------------
# verify_spec_sentinel scenarios
# ---------------------------------------------------------------------------


def test_verify_fresh_ckpt_dir_ok(ckpt_dir: pathlib.Path, fake_program: str) -> None:
    """Empty ckpt-dir → ok=True (fresh launch; caller writes sentinel after)."""
    ok, reason = run_long.verify_spec_sentinel(str(ckpt_dir), fake_program)
    assert ok is True
    assert "fresh" in reason


def test_verify_legacy_ckpts_without_sentinel_refused(
    ckpt_dir: pathlib.Path, fake_program: str
) -> None:
    """This is the D-155 scenario: ckpts present, no sentinel → REFUSE."""
    _touch_ckpt(ckpt_dir, 500)
    ok, reason = run_long.verify_spec_sentinel(str(ckpt_dir), fake_program)
    assert ok is False
    assert "no spec-sentinel" in reason
    assert "P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE" in reason


def test_verify_fingerprint_mismatch_refused(
    ckpt_dir: pathlib.Path, fake_program: str
) -> None:
    """Ckpts present, sentinel exists but spec changed → REFUSE."""
    _touch_ckpt(ckpt_dir, 1000)
    # Write a deliberately-wrong sentinel (simulates a prior program name).
    run_long.write_sentinel(str(ckpt_dir), "old_program:000000000000")
    ok, reason = run_long.verify_spec_sentinel(str(ckpt_dir), fake_program)
    assert ok is False
    assert "spec-sentinel mismatch" in reason
    assert "old_program" in reason


def test_verify_matching_sentinel_ok(
    ckpt_dir: pathlib.Path, fake_program: str
) -> None:
    """Ckpts present, sentinel matches current spec → OK, auto-resume proceeds."""
    _touch_ckpt(ckpt_dir, 1500)
    fp = run_long.compute_spec_fingerprint(fake_program)
    run_long.write_sentinel(str(ckpt_dir), fp)
    ok, reason = run_long.verify_spec_sentinel(str(ckpt_dir), fake_program)
    assert ok is True
    assert "sentinel match" in reason
    assert fp in reason


# ---------------------------------------------------------------------------
# ckpt_dir_has_ckpts edge cases
# ---------------------------------------------------------------------------


def test_ckpt_dir_has_ckpts_missing(tmp_path: pathlib.Path) -> None:
    """Non-existent dir → False (not a crash)."""
    assert run_long.ckpt_dir_has_ckpts(str(tmp_path / "does_not_exist")) is False


def test_ckpt_dir_has_ckpts_only_sentinel(ckpt_dir: pathlib.Path) -> None:
    """Sentinel-only dir → False (sentinel without ckpts is not resumable)."""
    run_long.write_sentinel(str(ckpt_dir), "p:abc")
    assert run_long.ckpt_dir_has_ckpts(str(ckpt_dir)) is False


def test_ckpt_dir_has_ckpts_positive(ckpt_dir: pathlib.Path) -> None:
    _touch_ckpt(ckpt_dir, 100)
    assert run_long.ckpt_dir_has_ckpts(str(ckpt_dir)) is True


# ---------------------------------------------------------------------------
# Misc regression guards
# ---------------------------------------------------------------------------


def test_sentinel_basename_is_hidden() -> None:
    """Sentinel must be dot-prefixed so it doesn't show up in `ls` or ckpt
    globs; CKPT_RE only matches ``step_\\d{6}\\.ckpt`` so the sentinel is
    already invisible to auto-resume."""
    assert run_long.SENTINEL_BASENAME.startswith(".")
    assert run_long.CKPT_RE.match(run_long.SENTINEL_BASENAME) is None


def test_extract_ckpt_dir_resilient() -> None:
    """Sanity: the D-122 auto-resume extractor still parses both arg forms."""
    assert run_long.extract_ckpt_dir("bin --checkpoint-dir foo/bar") == "foo/bar"
    assert run_long.extract_ckpt_dir("bin --checkpoint-dir=foo/bar") == "foo/bar"
    assert run_long.extract_ckpt_dir("bin --steps 5000") is None


# ---------------------------------------------------------------------------
# Perf-band parser (D-243 — step_time sentinel for compute-commitment authority)
# ---------------------------------------------------------------------------


def test_parse_step_times_real_format(tmp_path) -> None:
    """Parser pulls (step, ms) tuples from scale_experiment's print_step output.

    Format from src/training/scale_experiment.c:
        '%5d  | %8.4f (%6.3f + %6.4f + %6.3f) | %9.2f | %6.3f | %7.5f | %6.4f | %2d | %.0f'
    Last column is step_time_ms.
    """
    log = tmp_path / "stdout.log"
    log.write_text(
        "    1  |  10.5000 ( 0.000 + 0.0000 + 10.500) |  50000.00 | 12.000 | 0.00000 | 0.0000 |  0 | 8500\n"
        "    2  |  10.4000 ( 0.000 + 0.0000 + 10.400) |  45000.00 | 11.500 | 0.00000 | 0.0000 |  0 | 8650\n"
        "[load] step=2 layer=0 K=1 ema=[1.00000] min=1.00000 max=1.00000 ratio=1.00\n"
        "    3  |  10.3000 ( 0.000 + 0.0000 + 10.300) |  40000.00 | 11.000 | 0.00000 | 0.0000 |  0 | 9200\n",
        encoding="utf-8",
    )
    steps = run_long._parse_step_times(str(log))
    assert steps == [(1, 8500.0), (2, 8650.0), (3, 9200.0)]


def test_parse_step_times_after_step_filter(tmp_path) -> None:
    """`after_step` filter lets a watcher resume incrementally without
    re-parsing earlier lines."""
    log = tmp_path / "stdout.log"
    log.write_text(
        "    1  |  9.0 ( 0 + 0 + 9 ) | 50000 | 11 | 0 | 0 |  0 | 8500\n"
        "    2  |  9.0 ( 0 + 0 + 9 ) | 50000 | 11 | 0 | 0 |  0 | 8600\n"
        "    3  |  9.0 ( 0 + 0 + 9 ) | 50000 | 11 | 0 | 0 |  0 | 8700\n",
        encoding="utf-8",
    )
    later = run_long._parse_step_times(str(log), after_step=1)
    assert [s for s, _ in later] == [2, 3]


def test_parse_step_times_missing_file_safe() -> None:
    """Missing log returns empty list, not an exception."""
    assert run_long._parse_step_times("/nonexistent/path/foo.log") == []


def test_perf_drift_rc_is_4() -> None:
    """rc=4 is the orchestrator-recognized PERF_DRIFT signal. Hard-coded;
    must not collide with rc=2 (sentinel-fail), rc=3 (off-spec ckpt block),
    or 0/1 (success/generic-fail)."""
    assert run_long.PERF_DRIFT_RC == 4
