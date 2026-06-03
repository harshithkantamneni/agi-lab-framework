"""test_checkpoint_load_dump.py -- regression guard for tools/checkpoint_load_dump.py.

Covers the Cycle 30 fix for the V5 parse bug:
  - V4 (Cycle 25)          : sanity metadata + EMA sums ~1.0 per layer.
  - V5 early (Cycle 27)    : 128-byte TrainConfig; known worst ratio 16.84.
  - V5 late  (Cycle 29)    : 168-byte TrainConfig; known worst ratio 1.66
                              at step 300 on the 300-step smoke.

Each assertion is keyed to a known value from the live post-mortem record
(data/experiments/.../training.log or D-094/D-096). If any drift, the tool
or the checkpoints changed -- investigate immediately.

Run manually:
    python3 tests/test_checkpoint_load_dump.py

Pytest-compatible: `pytest tests/test_checkpoint_load_dump.py -v`
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "checkpoint_load_dump.py"


def _run(ckpt: Path) -> str:
    res = subprocess.run(
        [sys.executable, str(TOOL), str(ckpt)],
        capture_output=True, text=True, timeout=30,
    )
    # The tool exits non-zero iff any layer kill-violates (ratio > --kill=10).
    # We don't care about exit code for the regression test -- output parsing
    # is what we verify.
    return res.stdout


def _parse_meta_line(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("# version="):
            # e.g. "# version=5 step=300 epoch=0 tokens=153600 best_ppl=62.78 best_loss=4.1396"
            parts = line.lstrip("# ").split()
            out = {}
            for kv in parts:
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    out[k] = v
            return out
    raise AssertionError(f"missing '# version=' line in output:\n{stdout}")


def _parse_worst_ratio(stdout: str) -> float:
    for line in stdout.splitlines():
        if line.startswith("# WORST layer ratio:"):
            return float(line.split(":")[-1].strip())
    raise AssertionError(f"missing '# WORST layer ratio' in output:\n{stdout}")


def _no_sanity_warnings(stdout: str) -> bool:
    return "SANITY WARNING" not in stdout


def test_v4_cycle25():
    ckpt = REPO / "data" / "checkpoints" / "cycle25_phaseb_gatecheck" / "step_000500.ckpt"
    if not ckpt.exists():
        print(f"SKIP test_v4_cycle25: {ckpt} missing")
        return
    stdout = _run(ckpt)
    meta = _parse_meta_line(stdout)
    assert meta["version"] == "4", meta
    assert meta["step"] == "500", meta
    assert float(meta["best_ppl"]) < 100.0, meta
    assert _no_sanity_warnings(stdout), stdout
    print("OK test_v4_cycle25")


def test_v5_early_cycle27():
    ckpt = REPO / "data" / "checkpoints" / "cycle27_phaseb_regate_planb" / "step_000500.ckpt"
    if not ckpt.exists():
        print(f"SKIP test_v5_early_cycle27: {ckpt} missing")
        return
    stdout = _run(ckpt)
    meta = _parse_meta_line(stdout)
    assert meta["version"] == "5", meta
    assert meta["step"] == "500", meta
    worst = _parse_worst_ratio(stdout)
    # D-094 canonical: layer 3 = 16.84:1.
    assert 16.5 < worst < 17.5, f"worst_ratio={worst} (expected ~16.84)"
    assert _no_sanity_warnings(stdout), stdout
    print(f"OK test_v5_early_cycle27 (worst_ratio={worst})")


def test_v5_late_cycle29_smoke():
    ckpt = (REPO / "data" / "experiments" / "cycle_29_rev2_300step_smoke"
            / "checkpoints" / "final.ckpt")
    if not ckpt.exists():
        print(f"SKIP test_v5_late_cycle29_smoke: {ckpt} missing")
        return
    stdout = _run(ckpt)
    meta = _parse_meta_line(stdout)
    assert meta["version"] == "5", meta
    assert meta["step"] == "300", meta
    worst = _parse_worst_ratio(stdout)
    # qa_cycle29_rev2_300step_smoke.md §3 canonical: worst = 1.66.
    assert 1.5 < worst < 1.8, f"worst_ratio={worst} (expected ~1.66)"
    assert _no_sanity_warnings(stdout), stdout
    print(f"OK test_v5_late_cycle29_smoke (worst_ratio={worst})")


def test_ckpt_version_tcfg_size_guard_present():
    """Regression guard for devops Cycle 31 P2 followup.

    Cycle 29 Rev-2 added 40B to TrainConfig without bumping CKPT_VERSION.
    The Python tool had to paper over the resulting two-layouts-under-one-
    version mess via auto-detection. This test locks in the compile-time
    static_assert added in src/training/checkpoint.c so a future edit that
    deletes or weakens the guard fails this test loudly.

    We don't invoke clang here (the build already exercises the guard);
    instead we verify the guard source is present with the expected
    structure. If anyone renames CKPT_CURRENT_TCFG_SIZE or removes the
    _Static_assert, this test fails.
    """
    checkpoint_c = REPO / "src" / "training" / "checkpoint.c"
    assert checkpoint_c.exists(), f"missing {checkpoint_c}"
    src = checkpoint_c.read_text()
    assert "#define CKPT_CURRENT_TCFG_SIZE" in src, (
        "CKPT_CURRENT_TCFG_SIZE macro missing from checkpoint.c -- "
        "the TrainConfig version-bump guard was removed. Restore it "
        "per the file header's migration checklist."
    )
    assert "sizeof(TrainConfig) == CKPT_CURRENT_TCFG_SIZE" in src, (
        "_Static_assert on sizeof(TrainConfig) missing -- the "
        "version-bump guard was weakened. Restore it per the file "
        "header's migration checklist."
    )
    assert "CKPT_VERSION == 5" in src, (
        "CKPT_VERSION pin missing from the guard. If CKPT_VERSION was "
        "intentionally bumped, the CKPT_V{old}_TCFG_SIZE archive step "
        "in checkpoint.h must also be completed -- see the header "
        "comment in checkpoint.c."
    )
    # Archive-size sentinels must stay frozen.
    assert "CKPT_V1_TCFG_SIZE == 100" in src
    assert "CKPT_V2_TCFG_SIZE == 104" in src
    assert "CKPT_V4_TCFG_SIZE == 116" in src
    print("OK test_ckpt_version_tcfg_size_guard_present")


if __name__ == "__main__":
    test_v4_cycle25()
    test_v5_early_cycle27()
    test_v5_late_cycle29_smoke()
    test_ckpt_version_tcfg_size_guard_present()
    print("all pass")
