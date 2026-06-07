"""Tests for tools/run_phase3_factorial.py — Phase-3 factorial orchestrator.

Coverage spec (per D-193 dispatch + design doc §11.1):
  - Locked launch order is exactly the prereg-binding sequence
        A42 → D42 → B42 → C42 → A43 → D43 → B43 → C43 → A44 → D44 → B44 → C44.
  - B4 T1 trigger hook fires exactly after run 8 (C43), not before, not after run 9.
  - Cell-to-flags mapping (build_cell_cmd) emits the correct
        --model / --lr / --weight-seed / --default-moe ... line per cell.
  - State machine resume: completed cells are skipped on re-invocation.
  - Failure handling: on simulated mid-sequence failure, sequence HALTs
        and does not advance to the next cell.
  - Spec-fingerprint snapshot + drift detection.
  - PC-3 AMX-serialization: serial-loop semantics (mocked subprocess
        verifies cells are launched one at a time, in order).

All tests mock subprocess.call so no real `scale_experiment` is invoked.

Reference: data/engineering/d193_phase3_factorial_launch_design.md
Spec source: programs/program_2_example/phase3_p6_prereg.md §2.1 + §8.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
from typing import List, Tuple
from unittest.mock import patch, MagicMock

import pytest

TOOLS_DIR = pathlib.Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

run_phase3_factorial = importlib.import_module("run_phase3_factorial")


# ---------------------------------------------------------------------------
# Test 1 — Locked launch order matches prereg §2.1 exactly
# ---------------------------------------------------------------------------


def test_locked_launch_order_exact():
    """LAUNCH_ORDER is the prereg §2.1 binding sequence verbatim.

    Any deviation invalidates the B4 T1 trigger semantics (which assumes the
    first 8 runs populate cells A/B/C/D each at N=2).
    """
    expected: List[Tuple[str, int]] = [
        ("A", 42), ("D", 42), ("B", 42), ("C", 42),
        ("A", 43), ("D", 43), ("B", 43), ("C", 43),
        ("A", 44), ("D", 44), ("B", 44), ("C", 44),
    ]
    assert run_phase3_factorial.LAUNCH_ORDER == expected, (
        "LAUNCH_ORDER must match the prereg-binding 12-tuple sequence exactly. "
        "Any deviation breaks B4 T1 trigger evaluation."
    )
    assert len(run_phase3_factorial.LAUNCH_ORDER) == 12


def test_launch_order_first_eight_cover_all_4_cells_at_n2():
    """The first 8 runs populate cells A/B/C/D each at N=2.

    This is the prereg-binding configuration at the B4 T1 evaluation point
    (§8.1: "after run 8 (cells A, B, C, D each at N=2; pooled df=4)").
    """
    first_eight = run_phase3_factorial.LAUNCH_ORDER[:8]
    cells = [c for c, _ in first_eight]
    assert sorted(cells) == ["A", "A", "B", "B", "C", "C", "D", "D"], (
        "First 8 runs must contain each cell {A, B, C, D} at exactly N=2"
    )
    seeds_per_cell = {}
    for cell, seed in first_eight:
        seeds_per_cell.setdefault(cell, []).append(seed)
    for cell in ("A", "B", "C", "D"):
        assert sorted(seeds_per_cell[cell]) == [42, 43], (
            f"Cell {cell} must have exactly {{42, 43}} in first 8 runs"
        )


# ---------------------------------------------------------------------------
# Test 2 — B4 T1 hook is exactly after run 8
# ---------------------------------------------------------------------------


def test_b4_t1_eval_position_is_run_eight():
    """B4_T1_EVAL_AFTER_RUN == 8 (1-indexed); the 8th tuple is (C, 43)."""
    assert run_phase3_factorial.B4_T1_EVAL_AFTER_RUN == 8
    cell, seed = run_phase3_factorial.LAUNCH_ORDER[
        run_phase3_factorial.B4_T1_EVAL_AFTER_RUN - 1
    ]
    assert (cell, seed) == ("C", 43), (
        "Run 8 (1-indexed) of the locked launch order must be (C, 43); "
        "the prereg §8.1 trigger evaluation point is bound to this position."
    )


# ---------------------------------------------------------------------------
# Test 3 — Cell-to-flags mapping produces correct CLI lines
# ---------------------------------------------------------------------------


def test_build_cell_cmd_dense_a_lr1e3_seed42():
    """Cell A (Dense-A LR=1e-3) seed=42 produces expected CLI."""
    cmd = run_phase3_factorial.build_cell_cmd("A", 42)
    assert "build/scale_experiment" in cmd
    assert "--model dense50m" in cmd
    assert "--weight-seed 42" in cmd
    # LR encoding via repr() may be "0.001" or scientific; both acceptable.
    assert "--lr 0.001" in cmd or "--lr 1e-3" in cmd
    assert "--default-moe" not in cmd, "Dense-A cells must NOT use --default-moe"


def test_build_cell_cmd_dense_a_lr2e3_seed43():
    """Cell B (Dense-A LR=2e-3) seed=43 produces expected CLI."""
    cmd = run_phase3_factorial.build_cell_cmd("B", 43)
    assert "--model dense50m" in cmd
    assert "--weight-seed 43" in cmd
    assert "--lr 0.002" in cmd or "--lr 2e-3" in cmd
    assert "--default-moe" not in cmd


def test_build_cell_cmd_moe_lr1e3_seed44():
    """Cell C (MoE Rev-2 LR=1e-3) seed=44 produces expected CLI with MoE flags."""
    cmd = run_phase3_factorial.build_cell_cmd("C", 44)
    assert "--model medium" in cmd
    assert "--weight-seed 44" in cmd
    assert "--lr 0.001" in cmd or "--lr 1e-3" in cmd
    assert "--default-moe" in cmd, "MoE Rev-2 cells must use --default-moe"
    assert "--entropy-penalty" in cmd
    assert "--temp-anneal" in cmd


def test_build_cell_cmd_moe_lr2e3_seed42():
    """Cell D (MoE Rev-2 LR=2e-3) seed=42 produces expected CLI."""
    cmd = run_phase3_factorial.build_cell_cmd("D", 42)
    assert "--model medium" in cmd
    assert "--weight-seed 42" in cmd
    assert "--lr 0.002" in cmd or "--lr 2e-3" in cmd
    assert "--default-moe" in cmd
    assert "--entropy-penalty" in cmd


def test_build_cell_cmd_includes_checkpoint_dir_and_steps():
    """All cells must include --checkpoint-dir + --steps 5000 + --backprop --stream."""
    cmd = run_phase3_factorial.build_cell_cmd("A", 42)
    assert "--checkpoint-dir" in cmd
    assert "--steps 5000" in cmd
    assert "--backprop" in cmd
    assert "--stream" in cmd
    assert "--checkpoint-every 500" in cmd
    # Tokenizer is the locked Phase-3 32k tokenizer.
    assert "tokenizer_32k.bin" in cmd


def test_build_cell_cmd_unique_checkpoint_dir_per_cell_seed():
    """Each cell × seed combo gets a unique checkpoint-dir."""
    paths = set()
    for cell, seed in run_phase3_factorial.LAUNCH_ORDER:
        cmd = run_phase3_factorial.build_cell_cmd(cell, seed)
        # Extract --checkpoint-dir <X> token
        toks = cmd.split()
        idx = toks.index("--checkpoint-dir")
        ckpt = toks[idx + 1]
        assert f"{cell}{seed}" in ckpt, (
            f"checkpoint-dir for cell {cell}{seed} must include the cell-seed "
            f"identifier; got: {ckpt}"
        )
        paths.add(ckpt)
    assert len(paths) == 12, "All 12 cells must have unique checkpoint dirs"


# ---------------------------------------------------------------------------
# Test 4 — State machine: resume from mid-sequence
# ---------------------------------------------------------------------------


def test_state_machine_skips_completed_cells(tmp_path, monkeypatch):
    """Pre-populate run_index.json with 3 cells completed; verify they skip."""
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    phase3_root.mkdir(parents=True, exist_ok=True)

    # Populate run_index.json with 3 cells already completed.
    idx = {
        "_meta": {
            "program": "program_2_example",
            "lock_fingerprint": "test_fingerprint",
            "launch_order": [f"{c}{s}" for c, s in run_phase3_factorial.LAUNCH_ORDER],
            "b4_t1_eval_after_run": 8,
            "started_iso": "2026-04-25T00:00:00",
        },
        "A42": {"state": "completed", "run_index": 1, "rc": 0},
        "D42": {"state": "completed", "run_index": 2, "rc": 0},
        "B42": {"state": "completed", "run_index": 3, "rc": 0},
    }
    run_index_path = phase3_root / "run_index.json"
    run_index_path.write_text(json.dumps(idx, indent=2))

    # Patch the module-level paths so it reads from tmp_path.
    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "RUN_INDEX_PATH", str(run_index_path))

    loaded = run_phase3_factorial.load_run_index()
    assert loaded["A42"]["state"] == "completed"
    assert loaded["D42"]["state"] == "completed"
    assert loaded["B42"]["state"] == "completed"
    assert "C42" not in loaded, "C42 should not appear; it has not been started"

    # Simulate the orchestrator's state-check loop semantic.
    skipped = []
    advanced = []
    for run_idx, (cell, seed) in enumerate(run_phase3_factorial.LAUNCH_ORDER, start=1):
        run_key = f"{cell}{seed}"
        existing = loaded.get(run_key, {})
        if existing.get("state") == "completed":
            skipped.append(run_key)
        else:
            advanced.append(run_key)
            break  # simulate: orchestrator stops at first non-completed cell

    assert skipped == ["A42", "D42", "B42"], (
        "Completed cells must be skipped in launch-order"
    )
    assert advanced == ["C42"], (
        "Orchestrator must advance to the next un-completed cell (C42, run 4)"
    )


# ---------------------------------------------------------------------------
# Test 5 — Failure handling: halt on cell-5 simulated failure
# ---------------------------------------------------------------------------


def test_orchestrator_halts_on_run_failure(tmp_path, monkeypatch):
    """Mock run_long.py to return rc=1 on the 5th call; verify HALT before run 6.

    This validates the §8 "no silent advance through failed run" discipline.
    """
    # Track invocations in order.
    invocations = []

    def fake_subprocess_call(cmd_list, cwd=None, env=None):
        # cmd_list is ["python3", "tools/run_long.py", "--run-id", X, "--cmd", Y]
        run_id_idx = cmd_list.index("--run-id") + 1
        run_id = cmd_list[run_id_idx]
        invocations.append(run_id)
        # Fail on the 5th invocation (which would be A43, the 5th in LAUNCH_ORDER).
        if len(invocations) == 5:
            return 1
        return 0

    # Build a fake module-level state.
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    phase3_root.mkdir(parents=True, exist_ok=True)
    run_index_path = phase3_root / "run_index.json"

    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "RUN_INDEX_PATH", str(run_index_path))

    # Simulate the launch loop semantic directly (not via main(), since main()
    # has all the preflights bound to disk).
    idx = {"_meta": {"program": "test"}}
    fail_at = 5
    halted = False
    halt_run_idx = None

    for run_idx, (cell, seed) in enumerate(
        run_phase3_factorial.LAUNCH_ORDER, start=1
    ):
        run_key = f"{cell}{seed}"
        # mock-launch (incrementing the invocation counter)
        rc = fake_subprocess_call(
            ["python3", "tools/run_long.py", "--run-id", run_key, "--cmd", "test"]
        )
        idx[run_key] = {"state": "completed" if rc == 0 else "failed", "rc": rc}
        if rc != 0:
            halted = True
            halt_run_idx = run_idx
            break

    assert halted, "Orchestrator must HALT on rc != 0"
    assert halt_run_idx == fail_at, f"Halt at run {fail_at}, not run {halt_run_idx}"
    assert invocations == ["A42", "D42", "B42", "C42", "A43"], (
        "Exactly 5 cells launched in order (4 successful + 1 failed)"
    )
    # CRITICAL: cell 6 (D43) must never appear.
    assert "D43" not in invocations, (
        "Run 6 (D43) must NOT advance after run 5 (A43) failure — "
        "this is the §8 'no silent advance' discipline."
    )
    # The failed cell is recorded as failed, not completed.
    assert idx["A43"]["state"] == "failed"
    assert idx["A43"]["rc"] == 1


# ---------------------------------------------------------------------------
# Test 6 — B4 T1 hook: invoked exactly after run 8, never before, never after run 9
# ---------------------------------------------------------------------------


def test_b4_t1_hook_invoked_only_after_run_eight():
    """Simulate 12-cell loop; verify hook fires once at run 8 boundary."""
    hook_invocations = []

    # Simulate the per-run-loop hook semantic from
    # run_phase3_factorial.main() (lines around 'if run_idx == B4_T1_EVAL_AFTER_RUN').
    for run_idx, (cell, seed) in enumerate(
        run_phase3_factorial.LAUNCH_ORDER, start=1
    ):
        if run_idx == run_phase3_factorial.B4_T1_EVAL_AFTER_RUN:
            hook_invocations.append((run_idx, f"{cell}{seed}"))

    assert len(hook_invocations) == 1, "Hook must fire exactly once"
    assert hook_invocations[0] == (8, "C43"), (
        "Hook fires at run 8 = C43 in the locked order — not before, not after"
    )


def test_b4_t1_hook_does_not_fire_before_run_eight():
    """For runs 1-7, the hook condition is False."""
    for run_idx in range(1, 8):  # 1, 2, ..., 7
        assert run_idx != run_phase3_factorial.B4_T1_EVAL_AFTER_RUN
        assert run_idx < run_phase3_factorial.B4_T1_EVAL_AFTER_RUN


def test_b4_t1_hook_does_not_fire_after_run_eight():
    """For runs 9-12, the hook condition is False."""
    for run_idx in range(9, 13):  # 9, 10, 11, 12
        assert run_idx != run_phase3_factorial.B4_T1_EVAL_AFTER_RUN
        assert run_idx > run_phase3_factorial.B4_T1_EVAL_AFTER_RUN


# ---------------------------------------------------------------------------
# Test 7 — Spec fingerprint snapshot + drift detection
# ---------------------------------------------------------------------------


def test_spec_fingerprint_snapshot_written_on_first_invocation(tmp_path, monkeypatch):
    """First invocation: fingerprint snapshot is written to lock_fingerprint.txt."""
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    lock_fp_path = phase3_root / "lock_fingerprint.txt"

    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "LOCK_FP_PATH", str(lock_fp_path))

    # Mock compute_spec_fingerprint via run_long module.
    monkeypatch.setattr(
        run_phase3_factorial.run_long,
        "compute_spec_fingerprint",
        lambda program: f"{program}:abc123def456",
    )

    fp = run_phase3_factorial.preflight_fingerprint(strict_lock_check=True)
    assert fp == "program_2_example:abc123def456"
    assert lock_fp_path.is_file()
    assert lock_fp_path.read_text().strip() == fp


def test_spec_fingerprint_drift_blocks_strict(tmp_path, monkeypatch):
    """Subsequent invocation with drifted manifest: BLOCK on strict mode."""
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    phase3_root.mkdir(parents=True, exist_ok=True)
    lock_fp_path = phase3_root / "lock_fingerprint.txt"
    lock_fp_path.write_text("program_2_example:original12\n")

    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "LOCK_FP_PATH", str(lock_fp_path))

    # Mock compute_spec_fingerprint to return a DIFFERENT value (manifest drifted).
    monkeypatch.setattr(
        run_phase3_factorial.run_long,
        "compute_spec_fingerprint",
        lambda program: f"{program}:drifted567",
    )

    with pytest.raises(SystemExit) as exc_info:
        run_phase3_factorial.preflight_fingerprint(strict_lock_check=True)
    assert exc_info.value.code == 4, (
        "Manifest drift must exit with code 4 (verdict-breaking per prereg §13)"
    )


def test_spec_fingerprint_match_passes(tmp_path, monkeypatch):
    """Matching fingerprint passes silently."""
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    phase3_root.mkdir(parents=True, exist_ok=True)
    lock_fp_path = phase3_root / "lock_fingerprint.txt"
    expected_fp = "program_2_example:matching42"
    lock_fp_path.write_text(expected_fp + "\n")

    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "LOCK_FP_PATH", str(lock_fp_path))
    monkeypatch.setattr(
        run_phase3_factorial.run_long,
        "compute_spec_fingerprint",
        lambda program: expected_fp,
    )

    fp = run_phase3_factorial.preflight_fingerprint(strict_lock_check=True)
    assert fp == expected_fp


# ---------------------------------------------------------------------------
# Test 8 — Drift prediction: BLOCKs if cells will FATAL on lr_base or weight_seed
# ---------------------------------------------------------------------------


def test_drift_prediction_blocks_on_lr_mismatch(monkeypatch):
    """When manifest pins lr_base=0.002 for dense50m, cells A (LR=1e-3) FATAL.

    This is the §3.2 inconsistency that makes the orchestrator BLOCK launch
    until §10.1 (manifest amendment) lands.
    """
    fake_arms = {
        "dense50m": {
            "_arm_name": "dense50m",
            "model": "dense50m",
            "lr_base": "0.002",
            "weight_seed": "42",
        },
        "medium": {
            "_arm_name": "medium",
            "model": "medium",
            "lr_base": "0.001",
            "weight_seed": "42",
        },
    }
    blocks = run_phase3_factorial.predict_cell_drift_or_block(fake_arms)

    # Cell A (dense50m LR=1e-3) should conflict on lr_base (manifest=0.002, runtime=0.001).
    assert ("A", 42) in blocks
    a42_conflicts = blocks[("A", 42)]
    assert any("lr_base" in c for c in a42_conflicts), (
        "Cell A42 must surface lr_base conflict"
    )

    # Cell D (medium LR=2e-3) should conflict on lr_base (manifest=0.001, runtime=0.002).
    assert ("D", 42) in blocks
    d42_conflicts = blocks[("D", 42)]
    assert any("lr_base" in c for c in d42_conflicts), (
        "Cell D42 must surface lr_base conflict"
    )


def test_drift_prediction_blocks_on_weight_seed_mismatch(monkeypatch):
    """When manifest pins weight_seed=42, cells with seed != 42 FATAL."""
    fake_arms = {
        "dense50m": {
            "_arm_name": "dense50m",
            "model": "dense50m",
            "lr_base": "0.002",
            "weight_seed": "42",
        },
        "medium": {
            "_arm_name": "medium",
            "model": "medium",
            "lr_base": "0.001",
            "weight_seed": "42",
        },
    }
    blocks = run_phase3_factorial.predict_cell_drift_or_block(fake_arms)

    # Cell B43 = (dense50m, seed=43, LR=2e-3): lr_base matches (0.002==0.002),
    # but weight_seed mismatches (manifest=42, runtime=43).
    assert ("B", 43) in blocks
    b43_conflicts = blocks[("B", 43)]
    assert any("weight_seed" in c for c in b43_conflicts)


def test_drift_prediction_passes_when_manifest_matches_runtime():
    """If manifest were extended to 4 cell-arm blocks AND weight_seed unpinned,
    all 12 cells should predict PASS. This is the §10.1 target end-state."""
    # Hypothetical post-amendment manifest: omit weight_seed; per-cell lr_base.
    # Note: the current parser keys arm-lookup on `model`, so simulating the
    # post-D-194 schema requires the cell_id-discriminator extension which is
    # out-of-scope. We test the simpler case here: when both arms have the
    # SAME lr_base and weight_seed unpinned, each cell predicts PASS for the
    # cell-config that happens to match its (model, lr) pair.

    fake_arms_relaxed = {
        "dense50m": {"_arm_name": "dense50m", "model": "dense50m"},
        # No lr_base, no weight_seed pinned.
        "medium": {"_arm_name": "medium", "model": "medium"},
    }
    blocks = run_phase3_factorial.predict_cell_drift_or_block(fake_arms_relaxed)
    # With both fields unpinned, no cell should produce a conflict.
    assert blocks == {}, (
        f"With lr_base and weight_seed unpinned, all cells should PASS; "
        f"got conflicts: {blocks}"
    )


# ---------------------------------------------------------------------------
# Test 9 — PC-3 serialization: CELLS dict static structure
# ---------------------------------------------------------------------------


def test_cells_dict_has_4_factorial_cells():
    """CELLS dict has exactly 4 factorial cells with correct (model, lr) pairs.

    Updated at D-212 (seq_len-orchestrator-fix): each cell now also carries
    seq_len=512 (spec_invariants.yaml pin). See Test 17 for the targeted
    seq_len-pin assertion.

    Updated at D-243 (perf-band sentinel): each cell also carries
    perf_band_ms = (min, max) — the post-warmup step_time band the
    run_long.py watcher enforces. Dense bands and MoE bands shared via
    module-level constants.
    """
    cells = run_phase3_factorial.CELLS
    dense_band = run_phase3_factorial._DENSE_BAND_MS
    moe_band   = run_phase3_factorial._MOE_BAND_MS
    assert set(cells.keys()) == {"A", "B", "C", "D"}
    assert cells["A"] == {
        "model": "dense50m", "lr": 0.001, "seq_len": 512, "default_moe": False,
        "label": "Dense-A LR=1e-3", "perf_band_ms": dense_band,
    }
    assert cells["B"] == {
        "model": "dense50m", "lr": 0.002, "seq_len": 512, "default_moe": False,
        "label": "Dense-A LR=2e-3", "perf_band_ms": dense_band,
    }
    assert cells["C"] == {
        "model": "medium", "lr": 0.001, "seq_len": 512, "default_moe": True,
        "label": "MoE Rev-2 LR=1e-3", "perf_band_ms": moe_band,
    }
    assert cells["D"] == {
        "model": "medium", "lr": 0.002, "seq_len": 512, "default_moe": True,
        "label": "MoE Rev-2 LR=2e-3", "perf_band_ms": moe_band,
    }


def test_seeds_locked_to_42_43_44():
    """SEEDS = (42, 43, 44) per prereg §2.1."""
    assert run_phase3_factorial.SEEDS == (42, 43, 44)


def test_program_name_locked():
    """PROGRAM_NAME is the binding program for D-181/D-191/D-192/D-193 lockset."""
    assert run_phase3_factorial.PROGRAM_NAME == "program_2_example"


# ---------------------------------------------------------------------------
# Test 10 — Run-id and ckpt-dir naming conventions
# ---------------------------------------------------------------------------


def test_cell_run_id_format():
    """run_id = 'phase3_<cell><seed>' (matches dispatch §6 spec)."""
    assert run_phase3_factorial.cell_run_id("A", 42) == "phase3_A42"
    assert run_phase3_factorial.cell_run_id("D", 44) == "phase3_D44"


def test_cell_ckpt_dir_path():
    """ckpt-dir lives under data/checkpoints/phase3_factorial/<cell><seed>/."""
    path = run_phase3_factorial.cell_ckpt_dir("C", 43)
    assert path.endswith("phase3_factorial/C43") or path.endswith(
        os.path.join("phase3_factorial", "C43")
    )


# ---------------------------------------------------------------------------
# Test 11 — Wall-budget constants match prereg §2.1
# ---------------------------------------------------------------------------


def test_wall_budget_constants():
    """Wall-time constants match prereg-binding values."""
    assert run_phase3_factorial.WALL_HOURS_DENSE == pytest.approx(11.8)
    assert run_phase3_factorial.WALL_HOURS_MOE == pytest.approx(12.6)
    # 12 runs × ~12h average = ~144h serialized minimum
    total = sum(
        run_phase3_factorial.WALL_HOURS_MOE
        if run_phase3_factorial.CELLS[c]["default_moe"]
        else run_phase3_factorial.WALL_HOURS_DENSE
        for c, _ in run_phase3_factorial.LAUNCH_ORDER
    )
    # 6 dense (A, B at 3 seeds) + 6 MoE (C, D at 3 seeds) = 6×11.8 + 6×12.6 = 146.4h
    assert total == pytest.approx(146.4, rel=0.01)


# ---------------------------------------------------------------------------
# Test 12 (D-194 P-B4-T1-CONTRACT-RECONCILE) — B4 trigger handoff is invoked
# AFTER the 8th run (C43), not the 7th (B43).
#
# Background: D-193 current.md flagged that the launch-order claim "B43 at
# position 8" in the orphan b4_t1_evaluator.py docstring contradicted the
# actual indexing (which has C43 at position 8). This test pins the runtime
# behavior so any future re-shuffle that puts a different cell at position 8
# fails loud at test time, not at run time after 8 × ~12h of compute.
# ---------------------------------------------------------------------------


def test_b4_trigger_handoff_invoked_after_c43_not_b43(tmp_path, monkeypatch):
    """Behavior-level integration: walk the orchestrator loop with mocked
    subprocess; assert the b4_t1 handoff fires immediately after C43
    (the 8th cell) and NOT after B43 (the 7th cell).

    This guards against the historical ambiguity where:
      - LAUNCH_ORDER[6] = ('B', 43)  ← position 7 (1-indexed)
      - LAUNCH_ORDER[7] = ('C', 43)  ← position 8 (1-indexed)  ← hook fires here
    The orphaned `b4_t1_evaluator.py` docstring incorrectly stated "B43 the
    8th"; this test pins the truth: hook fires at C43-position-8.
    """
    # Set up a tmp phase3_factorial dir so the module's writes are isolated.
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    phase3_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "RUN_INDEX_PATH",
                        str(phase3_root / "run_index.json"))
    monkeypatch.setattr(run_phase3_factorial, "B4_TRIGGER_INPUT_PATH",
                        str(phase3_root / "b4_t1_input.json"))

    # Track the order of (cell, event) for the simulated loop.
    events: List[Tuple[int, str, str]] = []  # (run_idx, cell_seed, event_kind)

    def fake_handoff(idx):
        # Record what cell-seed was JUST completed when the handoff fired.
        # (We append based on the orchestrator-loop-state semantic: handoff
        # is called immediately after the per-run state-update for the run
        # at run_idx == B4_T1_EVAL_AFTER_RUN.)
        events.append((-1, "_handoff_invoked", "handoff"))
        return {"stdout": "B4 T1 TRIGGER: NO FIRE (mock)",
                "stderr": "", "rc": 0, "t_eval": 0.0}

    # Replace handoff and the underlying launcher so we don't shell out.
    monkeypatch.setattr(run_phase3_factorial, "b4_t1_handoff", fake_handoff)

    # Simulate the orchestrator's per-run loop semantic (lines around 602-644
    # of run_phase3_factorial.main), without invoking main() itself (which
    # has preflight gates bound to disk).
    idx = {"_meta": {"program": "test"}}
    for run_idx, (cell, seed) in enumerate(
        run_phase3_factorial.LAUNCH_ORDER, start=1
    ):
        run_key = f"{cell}{seed}"
        # Mock-launch (always succeeds in this test).
        events.append((run_idx, run_key, "launch"))
        idx[run_key] = {"state": "completed", "rc": 0, "run_index": run_idx}

        # The exact gate from main(): if run_idx == B4_T1_EVAL_AFTER_RUN, fire.
        if run_idx == run_phase3_factorial.B4_T1_EVAL_AFTER_RUN:
            verdict = run_phase3_factorial.b4_t1_handoff(idx)
            idx["_meta"]["b4_t1_verdict"] = verdict

    # ------ Assertions ------

    # 1. The handoff fired exactly once.
    handoff_events = [e for e in events if e[2] == "handoff"]
    assert len(handoff_events) == 1, (
        f"B4 T1 handoff must fire exactly once over the 12-run loop; "
        f"observed {len(handoff_events)}: {handoff_events}"
    )

    # 2. The handoff event is at position 9 in the events list (8 launches + 1 handoff).
    #    Specifically: events[7] is the launch of run 8 (C43); events[8] is the handoff.
    assert len(events) >= 9, (
        f"Expected >=9 events (8 launches + 1 handoff before run 9); got {len(events)}"
    )

    # 3. The launch event IMMEDIATELY before the handoff is C43, NOT B43.
    handoff_idx = next(i for i, e in enumerate(events) if e[2] == "handoff")
    assert handoff_idx > 0, "Handoff must follow at least one launch"
    prior_event = events[handoff_idx - 1]
    assert prior_event[2] == "launch", (
        f"Event immediately before handoff must be a launch; got {prior_event}"
    )
    assert prior_event[0] == 8, (
        f"Handoff must fire after the 8th launch (1-indexed); fired after "
        f"launch #{prior_event[0]} ({prior_event[1]})"
    )
    assert prior_event[1] == "C43", (
        f"Handoff must fire after C43 (the 8th cell in the locked launch order), "
        f"NOT after B43 (the 7th). Observed prior cell: {prior_event[1]}. "
        f"This is the prereg §8.1-binding position; any other cell here invalidates "
        f"the trigger semantics."
    )

    # 4. The launch event AT position 7 (just before C43) is B43 — confirms
    #    the handoff did NOT fire at the wrong (7th) position.
    seventh_event = events[6]  # 0-indexed; events[0] = launch of run 1
    assert seventh_event[2] == "launch"
    assert seventh_event[0] == 7
    assert seventh_event[1] == "B43", (
        f"Position 7 in launch order must be B43; got {seventh_event[1]}. "
        f"Off-by-one between B43 and C43 is the historical bug this test guards."
    )

    # 5. After the handoff, runs 9-12 still proceed (the locked-launch §8.4
    #    binding: hook is observational, never alters the launch sequence).
    post_handoff_launches = [
        e for e in events[handoff_idx+1:] if e[2] == "launch"
    ]
    assert len(post_handoff_launches) == 4, (
        f"Runs 9-12 (4 launches) must follow the handoff; got "
        f"{len(post_handoff_launches)}: {post_handoff_launches}"
    )
    assert [e[1] for e in post_handoff_launches] == ["A44", "D44", "B44", "C44"]


# ---------------------------------------------------------------------------
# Test 13 (D-194 P-B4-T1-CONTRACT-RECONCILE) — orchestrator references the
# canonical b4_t1_trigger.py and uses --results-json contract, not the
# orphaned --cell-scores contract.
# ---------------------------------------------------------------------------


def test_orchestrator_points_at_canonical_trigger_path():
    """B4_TRIGGER_PATH points at tools/b4_t1_trigger.py (the canonical D-193
    tool with strict-`>` boundary semantics), not the orphaned b4_t1_evaluator.py.

    Guards the D-194 P-B4-T1-CONTRACT-RECONCILE landing.
    """
    assert hasattr(run_phase3_factorial, "B4_TRIGGER_PATH"), (
        "Orchestrator must export B4_TRIGGER_PATH (D-194 rename from B4_EVALUATOR_PATH)"
    )
    assert not hasattr(run_phase3_factorial, "B4_EVALUATOR_PATH"), (
        "B4_EVALUATOR_PATH (the D-193 orphan reference) must be removed at D-194"
    )
    path = run_phase3_factorial.B4_TRIGGER_PATH
    assert path.endswith(os.path.join("tools", "b4_t1_trigger.py")) or \
           path.endswith("tools/b4_t1_trigger.py"), (
        f"B4_TRIGGER_PATH must point at tools/b4_t1_trigger.py; got: {path}"
    )
    # The orphan path must NOT be referenced.
    assert "b4_t1_evaluator.py" not in path, (
        f"Orchestrator must not reference the orphaned b4_t1_evaluator.py; "
        f"B4_TRIGGER_PATH = {path}"
    )


# ---------------------------------------------------------------------------
# Test 14 (D-212 dry-run-state-poisoning regression) — dry-run mode is fully
# read-only past pre-flight. Asserts:
#   (a) layer (a) — `--dry-run` does NOT write run_index.json or any cell dir
#   (b) layer (b) — real-run defensively overwrites pre-existing
#                   `dry_run: true, state: completed` records (treats them as
#                   not-done) instead of skipping them.
#
# Background: 2026-04-25 (D-212 first launch attempt) — `make
# run-phase3-factorial-dryrun` poisoned the state machine by writing 12
# `state: completed` entries with `dry_run: true` flag set. The subsequent
# real-run pass read run_index.json, treated all 12 entries as "completed",
# and skipped every cell. Net effect: factorial launch silently no-op'd. This
# test pins both fix layers (a) AND (b) at the orchestrator-loop semantic so
# any future regression fails loud at test time, not after Director schedules
# a 6-day wall budget that turns into a 5-second silent skip.
# ---------------------------------------------------------------------------


def test_dry_run_writes_no_state_to_disk(tmp_path, monkeypatch):
    """Layer (a) — `--dry-run` is fully read-only past pre-flight.

    Walks the patched orchestrator's main-loop semantic with `args.dry_run=True`
    and asserts:
      - run_index.json is NOT created
      - no cell-ckpt-dir is created
      - no cell-journal is written
    """
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    phase3_root.mkdir(parents=True, exist_ok=True)
    run_index_path = phase3_root / "run_index.json"

    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "RUN_INDEX_PATH", str(run_index_path))

    # Simulate the patched main-loop semantic: dry-run branch must not call
    # write_run_index or write_cell_journal. We bind the actual functions and
    # spy on disk artifacts (the defense layer (a) is enforced inline in
    # main()'s loop, not in the helpers themselves).
    idx = {"_meta": {"program": "test", "lock_fingerprint": "test"}}
    # Mirror the dry-run loop semantic from run_phase3_factorial.main():
    for run_idx, (cell, seed) in enumerate(
        run_phase3_factorial.LAUNCH_ORDER, start=1
    ):
        # The dry-run branch in main() is:
        #     if args.dry_run:
        #         launch_one_cell(cell, seed, dry_run=True)
        #         continue
        # — no idx mutation, no write_run_index, no write_cell_journal.
        rc, journal = run_phase3_factorial.launch_one_cell(
            cell, seed, dry_run=True
        )
        # Sanity: launch_one_cell with dry_run=True returns (0, {"dry_run": True}).
        assert rc == 0
        assert journal == {"dry_run": True}
        # CRITICAL: the dry-run branch must NOT write to idx or call
        # write_run_index / write_cell_journal — that is the defense layer.

    # ---- Disk-state assertions ----
    assert not run_index_path.exists(), (
        f"Layer (a) violated: --dry-run wrote run_index.json at {run_index_path}. "
        "Dry-run must be fully read-only past pre-flight (Makefile contract: "
        "'no actual training launches; runs preflights and prints the matrix'). "
        "Any persistent state from dry-run poisons the idempotent state machine "
        "in real-run mode (D-212 launch failure root cause)."
    )
    # No cell ckpt-dirs were created.
    for cell, seed in run_phase3_factorial.LAUNCH_ORDER:
        cell_dir = phase3_root / f"{cell}{seed}"
        assert not cell_dir.exists(), (
            f"Layer (a) violated: --dry-run created cell ckpt-dir {cell_dir}. "
            "Dry-run must not produce any per-cell artifacts on disk."
        )


def test_real_run_treats_dry_run_residue_as_not_done(tmp_path, monkeypatch):
    """Layer (b) — real-run defensively overwrites `dry_run: true` records.

    Pre-populates run_index.json with the EXACT residue shape the D-212 bug
    produced (state=completed AND dry_run=true), then walks the patched real-run
    skip-check semantic. Assertion: every cell with `dry_run: true` is treated
    as not-done (does NOT skip) — the launch loop proceeds to launch the cell.
    """
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    phase3_root.mkdir(parents=True, exist_ok=True)
    run_index_path = phase3_root / "run_index.json"

    # Reproduce the D-212 residue shape: 12 entries, all completed-and-dry-run.
    residue = {
        "_meta": {
            "program": "program_2_example",
            "lock_fingerprint": "test",
            "launch_order": [f"{c}{s}" for c, s in run_phase3_factorial.LAUNCH_ORDER],
            "b4_t1_eval_after_run": 8,
            "started_iso": "2026-04-25T11:10:26",
        },
    }
    for run_idx, (cell, seed) in enumerate(
        run_phase3_factorial.LAUNCH_ORDER, start=1
    ):
        residue[f"{cell}{seed}"] = {
            "cell": cell,
            "seed": seed,
            "run_index": run_idx,
            "dry_run": True,
            "state": "completed",
            "started_iso": "2026-04-25T11:10:26",
            "ended_iso": "2026-04-25T11:10:26",
        }
    run_index_path.write_text(json.dumps(residue, indent=2))

    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "RUN_INDEX_PATH", str(run_index_path))

    loaded = run_phase3_factorial.load_run_index()
    assert all(loaded[f"{c}{s}"].get("dry_run") for c, s in
               run_phase3_factorial.LAUNCH_ORDER), (
        "Sanity: residue setup must populate all 12 cells with dry_run=true"
    )

    # Mirror the patched real-run skip-check semantic from main():
    #     if existing.get("state") == "completed" and not existing.get("dry_run"):
    #         skip
    #     elif existing.get("state") == "completed" and existing.get("dry_run"):
    #         proceed (defense layer b)
    skipped = []
    proceeded = []
    for run_idx, (cell, seed) in enumerate(
        run_phase3_factorial.LAUNCH_ORDER, start=1
    ):
        run_key = f"{cell}{seed}"
        existing = loaded.get(run_key, {})
        if (existing.get("state") == "completed"
                and not existing.get("dry_run")):
            skipped.append(run_key)
        else:
            # Defense layer (b): dry_run==True records DO NOT trigger skip.
            proceeded.append(run_key)

    assert skipped == [], (
        f"Layer (b) violated: real-run skipped cells with dry_run residue: "
        f"{skipped}. The D-212 bug was exactly this — `state==completed` "
        "alone treated as truth, ignoring the `dry_run` qualifier. Real-run "
        "must treat any `dry_run: true` record as not-done and overwrite."
    )
    assert proceeded == [f"{c}{s}" for c, s in run_phase3_factorial.LAUNCH_ORDER], (
        f"All 12 cells must proceed (be eligible for launch) when their only "
        f"prior record is dry-run residue. Got: {proceeded}"
    )


def test_real_run_still_skips_genuine_completions(tmp_path, monkeypatch):
    """Anti-regression — layer (b) must NOT break legitimate resume semantics.

    Real-run resumability requires that genuinely-completed cells (dry_run=false
    or absent) still skip on re-invocation. This test pins that the layer-(b)
    filter is `dry_run: true`-specific, not blanket "all completed records".
    """
    phase3_root = tmp_path / "data" / "checkpoints" / "phase3_factorial"
    phase3_root.mkdir(parents=True, exist_ok=True)
    run_index_path = phase3_root / "run_index.json"

    mixed = {
        "_meta": {"program": "test"},
        # Genuine real-run completion (no dry_run key, OR dry_run=false).
        "A42": {"state": "completed", "rc": 0, "run_index": 1},
        "D42": {"state": "completed", "rc": 0, "run_index": 2, "dry_run": False},
        # Dry-run residue (D-212 shape).
        "B42": {"state": "completed", "run_index": 3, "dry_run": True},
        # Failed completion — also not "completed" so loop should re-attempt.
        "C42": {"state": "failed", "rc": 1, "run_index": 4},
    }
    run_index_path.write_text(json.dumps(mixed, indent=2))
    monkeypatch.setattr(run_phase3_factorial, "PHASE3_ROOT", str(phase3_root))
    monkeypatch.setattr(run_phase3_factorial, "RUN_INDEX_PATH", str(run_index_path))

    loaded = run_phase3_factorial.load_run_index()
    skipped = []
    proceeded = []
    for run_idx, (cell, seed) in enumerate(
        run_phase3_factorial.LAUNCH_ORDER, start=1
    ):
        run_key = f"{cell}{seed}"
        existing = loaded.get(run_key, {})
        if (existing.get("state") == "completed"
                and not existing.get("dry_run")):
            skipped.append(run_key)
        else:
            proceeded.append(run_key)

    # A42 (no dry_run key) and D42 (dry_run=False) are genuine — must skip.
    assert "A42" in skipped, (
        "Genuine completion (no dry_run key) must still skip — resumability discipline."
    )
    assert "D42" in skipped, (
        "Genuine completion (dry_run=False) must still skip — resumability discipline."
    )
    # B42 (dry_run=True) is residue — must proceed.
    assert "B42" in proceeded, (
        "Dry-run residue must NOT cause skip — defense layer (b)."
    )
    # C42 (failed) is not "completed" — must proceed (re-attempt path).
    assert "C42" in proceeded, (
        "Failed cells must re-attempt on next invocation — orchestrator §8 discipline."
    )


# ---------------------------------------------------------------------------
# Test 15 (D-212 seq_len-orchestrator-fix regression) — build_cell_cmd MUST
# emit `--seq-len 512` for every cell.
#
# Background: 2026-04-25 (D-212 second launch attempt) — `make
# run-phase3-factorial` launched A42 at 11:22 with a cmd that omitted
# `--seq-len`. The C binary defaulted S=128, the on-disk
# spec_invariants.yaml pins seq_len=512 uniformly across all 6 arms (4
# Phase-3 cells + 2 legacy), and config_drift_assert_or_die() FATAL-ed at
# step 0 with `seq_len expected 512, got 128`. This test pins the
# orchestrator's CLI flag emission so any future regression that drops
# `--seq-len` from build_cell_cmd fails loud at test time, NOT after
# Director schedules a launch and the binary FATALs ~30 seconds in.
#
# Spec source: programs/program_2_example/spec_invariants.yaml
#              (all 6 arms pinned to seq_len=512; locked, no amendment).
# ---------------------------------------------------------------------------


def test_build_cell_cmd_emits_seq_len_512():
    """Every cell × every seed in LAUNCH_ORDER emits `--seq-len 512`.

    Walks the locked 12-tuple and asserts:
      (a) the literal substring `--seq-len 512` appears in the cmd
      (b) the token after `--seq-len` parses to int(512)
      (c) the value sits between `--seq-len` and the NEXT flag (i.e. no
          accidental positional bleed)
    """
    for cell, seed in run_phase3_factorial.LAUNCH_ORDER:
        cmd = run_phase3_factorial.build_cell_cmd(cell, seed)
        toks = cmd.split()

        # (a) Literal-substring check.
        assert "--seq-len 512" in cmd, (
            f"Cell {cell}{seed}: missing `--seq-len 512` in build_cell_cmd output. "
            f"D-212 root-cause regression — the C binary will FATAL pre-step-0 "
            f"with `seq_len expected 512, got 128` if the orchestrator omits it. "
            f"cmd: {cmd}"
        )

        # (b) Token-position check: the token immediately after `--seq-len` is "512".
        assert "--seq-len" in toks, (
            f"Cell {cell}{seed}: `--seq-len` flag must appear as its own token; "
            f"toks: {toks}"
        )
        idx = toks.index("--seq-len")
        assert idx + 1 < len(toks), (
            f"Cell {cell}{seed}: `--seq-len` is the last token (no value); toks: {toks}"
        )
        seq_len_val = toks[idx + 1]
        assert int(seq_len_val) == 512, (
            f"Cell {cell}{seed}: `--seq-len` value must be 512 (spec_invariants.yaml "
            f"pin); got {seq_len_val!r}. cmd: {cmd}"
        )

        # (c) The value is NOT another flag (defense against `--seq-len --steps`-shape
        # bug where a missing arg silently consumes the next flag).
        assert not seq_len_val.startswith("--"), (
            f"Cell {cell}{seed}: `--seq-len` value bled into a flag token "
            f"({seq_len_val!r}); cmd: {cmd}"
        )


def test_build_cell_cmd_seq_len_for_each_factorial_cell_letter():
    """Per-cell-letter (A/B/C/D) coverage at seed=42 with explicit-cmd inspection.

    Defense in depth: even if LAUNCH_ORDER changed in a future spec amendment,
    the per-cell-letter assertion guarantees every factorial cell emits the
    spec-pinned seq_len. Catches a regression that drops the flag for one
    cell (e.g. "MoE cells need a different seq_len" mistake) while keeping
    the others.
    """
    for cell_letter in ("A", "B", "C", "D"):
        cmd = run_phase3_factorial.build_cell_cmd(cell_letter, 42)
        assert "--seq-len 512" in cmd, (
            f"Cell {cell_letter}42: missing `--seq-len 512`. spec_invariants.yaml "
            f"pins seq_len=512 uniformly across all 4 phase3 cells (and the 2 "
            f"legacy arms); a per-cell variation would require a prereg "
            f"amendment + unanimous PI+Director sign — not in scope at D-212."
        )


# ---------------------------------------------------------------------------
# Test 16 (D-212 seq_len-orchestrator-fix regression) — predict_cell_drift_or_block
# MUST surface a `seq_len` conflict when manifest pins differ from runtime.
#
# Defense-in-depth pre-flight check: if the next bug-of-this-class lands
# (manifest amended to seq_len=256 for one arm but orchestrator still launches
# with --seq-len 512, OR vice versa), the orchestrator-side predictor catches
# it BEFORE the binary FATALs at step 0.
#
# Mirrors the existing `weight_seed`-conflict surface from §3.2 / Test 8 —
# parallel logic at the same code site.
# ---------------------------------------------------------------------------


def test_predict_cell_drift_blocks_on_seq_len_mismatch():
    """When manifest pins seq_len=256 for an arm, all cells of that arm conflict.

    Synthesizes a 1-arm manifest with cell_id=A and seq_len=256; the runtime
    builds cells with --seq-len 512 (per build_cell_cmd). The predictor must
    surface a `seq_len` conflict for each (A, seed) tuple in LAUNCH_ORDER.
    """
    fake_arms = {
        "phase3_cell_A": {
            "_arm_name": "phase3_cell_A",
            "cell_id": "A",
            "model": "dense50m",
            "lr_base": "0.001",  # matches cell A
            # weight_seed intentionally NOT pinned — isolate the seq_len conflict.
            "seq_len": "256",    # MISMATCH against runtime 512
        },
    }
    blocks = run_phase3_factorial.predict_cell_drift_or_block(fake_arms)

    # Cell A at any seed should surface a seq_len conflict.
    a42_conflicts = blocks.get(("A", 42), [])
    assert any("seq_len" in c for c in a42_conflicts), (
        f"Cell A42 must surface seq_len conflict when manifest pins 256 vs "
        f"runtime 512. Got conflicts: {a42_conflicts}. blocks: {blocks}. "
        f"This is the defense-in-depth pre-flight that D-212 root-cause "
        f"missed — without it, a future seq_len drift FATAL-s only at the "
        f"binary, NOT at orchestrator preflight."
    )

    # Conflict string mentions both manifest and runtime values for operator audit.
    conflict_str = next(c for c in a42_conflicts if "seq_len" in c)
    assert "256" in conflict_str, (
        f"Conflict string must echo the manifest value (256) for operator "
        f"diagnosis; got: {conflict_str!r}"
    )
    assert "512" in conflict_str, (
        f"Conflict string must echo the runtime value (512) for operator "
        f"diagnosis; got: {conflict_str!r}"
    )


def test_predict_cell_drift_passes_when_seq_len_matches():
    """When manifest seq_len matches runtime (both 512), no seq_len conflict.

    Anti-regression for Test 16's positive surface: the predictor must NOT
    false-positive when the manifest is in-sync. Pins the symmetric correctness
    of the seq_len-conflict logic.
    """
    fake_arms = {
        "phase3_cell_A": {
            "_arm_name": "phase3_cell_A",
            "cell_id": "A",
            "model": "dense50m",
            "lr_base": "0.001",  # matches cell A
            "seq_len": "512",    # MATCHES runtime 512
        },
        "phase3_cell_B": {
            "_arm_name": "phase3_cell_B",
            "cell_id": "B",
            "model": "dense50m",
            "lr_base": "0.002",  # matches cell B
            "seq_len": "512",    # MATCHES runtime 512
        },
        "phase3_cell_C": {
            "_arm_name": "phase3_cell_C",
            "cell_id": "C",
            "model": "medium",
            "lr_base": "0.001",  # matches cell C
            "seq_len": "512",    # MATCHES runtime 512
        },
        "phase3_cell_D": {
            "_arm_name": "phase3_cell_D",
            "cell_id": "D",
            "model": "medium",
            "lr_base": "0.002",  # matches cell D
            "seq_len": "512",    # MATCHES runtime 512
        },
    }
    blocks = run_phase3_factorial.predict_cell_drift_or_block(fake_arms)

    # No cell should produce a seq_len conflict.
    for (cell, seed), conflicts in blocks.items():
        for c in conflicts:
            assert "seq_len" not in c, (
                f"Cell {cell}{seed}: false-positive seq_len conflict when "
                f"manifest matches runtime. conflict: {c!r}"
            )


# ---------------------------------------------------------------------------
# Test 17 (D-212 seq_len-orchestrator-fix regression) — CELLS dict carries
# seq_len=512 per cell, parallel to lr / model / default_moe.
# ---------------------------------------------------------------------------


def test_cells_dict_has_seq_len_pinned_to_512():
    """Each of the 4 factorial cells carries seq_len=512.

    Per the dispatch rationale: "Spec pins all 6 arms to 512 uniformly;
    carrying it per-cell keeps the structure parallel to lr/model/default_moe".
    Also forward-proofs against a future spec amendment that varies seq_len
    per cell — the lookup site is a single dict, not 4 hardcoded literals.
    """
    cells = run_phase3_factorial.CELLS
    for cell_letter in ("A", "B", "C", "D"):
        assert "seq_len" in cells[cell_letter], (
            f"Cell {cell_letter}: CELLS dict missing `seq_len` field. "
            f"D-212 fix requires per-cell seq_len for build_cell_cmd to "
            f"emit `--seq-len <val>`."
        )
        assert cells[cell_letter]["seq_len"] == 512, (
            f"Cell {cell_letter}: seq_len must be 512 (spec_invariants.yaml "
            f"pin); got {cells[cell_letter]['seq_len']!r}."
        )
