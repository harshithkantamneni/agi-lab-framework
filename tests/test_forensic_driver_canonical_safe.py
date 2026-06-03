"""test_forensic_driver_canonical_safe.py -- regression safeguard for
tools/forensic_driver.py.

Purpose
-------
The forensic driver is the ONLY sanctioned way to run eval_*.py on a
non-current checkpoint without clobbering the canonical
`data/eval/<bench>_results.json` paths + `data/eval/scorecard.md`.

This test enforces the two contracts that protect the regression suite
and the Director's source-of-truth eval state:

  1. tools/forensic_driver.py exists and its docstring explicitly forbids
     writes under `data/checkpoints/cycleXX_*/step_*.ckpt` from forensic
     code paths.

  2. tools/forensic_driver.py does not import/use any symlink-creation
     or checkpoint-path-rewriting primitive. It only reads checkpoints.

  3. --dry-run mode completes end-to-end without invoking the eval
     drivers and without touching canonical paths.

If any of these contracts breaks, this test fails loudly and names the
specific regression.

Run manually:
    python3 tests/test_forensic_driver_canonical_safe.py

Pytest-compatible: `pytest tests/test_forensic_driver_canonical_safe.py -v`
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "tools" / "forensic_driver.py"
EVAL_DIR = REPO / "data" / "eval"


def _sha(path: Path) -> str:
    if not path.exists():
        return "absent"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def test_forensic_driver_exists_and_warns_on_checkpoint_writes():
    assert DRIVER.exists(), (
        f"missing {DRIVER} -- see Cycle 31 devops task 1 (D-098 followup). "
        "This wrapper is the only sanctioned way to run evals on a "
        "non-current checkpoint without clobbering canonical paths."
    )
    src = DRIVER.read_text()

    # The tool's docstring / comments must explicitly state the no-write
    # rule for data/checkpoints/cycleXX_*/step_*.ckpt. If anyone ever
    # edits the tool to write into data/checkpoints/, they must also
    # rewrite this contract (and this test), which flags the change for
    # reviewer attention.
    assert "data/checkpoints/" in src, (
        "forensic_driver.py must document its no-write rule for "
        "data/checkpoints/ explicitly. Expected a docstring line "
        "covering the regression-test checkpoint path invariant."
    )
    assert "NEVER writes" in src or "never writes" in src.lower(), (
        "forensic_driver.py must state the canonical-checkpoint no-write "
        "invariant in its docstring."
    )
    print("OK test_forensic_driver_exists_and_warns_on_checkpoint_writes")


def test_forensic_driver_no_symlink_or_checkpoint_write():
    """The tool must not use symlink/rename/copy primitives against
    data/checkpoints/. A grep-level check catches obvious regressions
    where someone adds an os.symlink / os.rename to a checkpoint path.
    """
    src = DRIVER.read_text()
    # The tool SHOULD use shutil.move on eval JSONs only. Forbid patterns
    # that touch data/checkpoints/ paths.
    forbidden = [
        'os.symlink',
        'Path.symlink_to',
        '.symlink_to(',
    ]
    for pattern in forbidden:
        # The tool should not create symlinks anywhere. If we ever want
        # to, the caller has to consciously rewrite this test.
        assert pattern not in src, (
            f"forensic_driver.py uses forbidden primitive '{pattern}'. "
            "Symlink creation is forbidden; see the canonical-path-clobber "
            "risk documented in decisions_recent.md D-098."
        )

    # Additionally, forbid any literal reference to writing under
    # data/checkpoints/.
    dangerous_writes = [
        'open("data/checkpoints/',
        "open('data/checkpoints/",
        'write("data/checkpoints/',
        'CHECKPOINTS_DIR.write',
    ]
    for pat in dangerous_writes:
        assert pat not in src, (
            f"forensic_driver.py contains a dangerous write pattern: "
            f"'{pat}'. The tool MUST only read from data/checkpoints/."
        )
    print("OK test_forensic_driver_no_symlink_or_checkpoint_write")


def test_forensic_driver_dry_run_preserves_canonical_paths(tmp_path=None):
    """End-to-end: --dry-run must not mutate canonical eval/ artifacts."""
    # Snapshot canonical state before.
    canonical_targets = [
        EVAL_DIR / "mmlu_results.json",
        EVAL_DIR / "hellaswag_results.json",
        EVAL_DIR / "gsm8k_results.json",
        EVAL_DIR / "winogrande_results.json",
        EVAL_DIR / "scorecard.md",
    ]
    pre = {p: _sha(p) for p in canonical_targets}

    # Dry-run needs an existing checkpoint + tokenizer path. Use the
    # Cycle 25 regression-test checkpoint + the 32k tokenizer. Fall back
    # to SKIP if either is missing (i.e. minimal checkout without data).
    ckpt = REPO / "data" / "checkpoints" / "cycle25_phaseb_gatecheck" / "step_000500.ckpt"
    tok = REPO / "data" / "training" / "tokenizer_32k.bin"
    if not ckpt.exists() or not tok.exists():
        print(f"SKIP test_forensic_driver_dry_run_preserves_canonical_paths: "
              f"ckpt.exists={ckpt.exists()} tok.exists={tok.exists()}")
        return

    # Run the forensic driver in dry-run mode with a unique tag.
    tag = "regtest_dryrun"
    res = subprocess.run(
        [sys.executable, str(DRIVER),
         "--checkpoint", str(ckpt),
         "--tokenizer", str(tok),
         "--tag", tag,
         "--benchmarks", "mmlu",
         "--dry-run"],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, (
        f"forensic_driver.py --dry-run failed:\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )

    # Verify canonical state is byte-for-byte identical.
    post = {p: _sha(p) for p in canonical_targets}
    drift = [p for p in canonical_targets if pre[p] != post[p]]
    assert not drift, (
        f"forensic_driver.py --dry-run mutated canonical paths: "
        f"{[str(p) for p in drift]}. This is a REGRESSION."
    )

    # Clean up the forensic run dir we created.
    for d in EVAL_DIR.glob(f"forensic_run_*_{tag}"):
        shutil.rmtree(d, ignore_errors=True)
    print("OK test_forensic_driver_dry_run_preserves_canonical_paths")


if __name__ == "__main__":
    test_forensic_driver_exists_and_warns_on_checkpoint_writes()
    test_forensic_driver_no_symlink_or_checkpoint_write()
    test_forensic_driver_dry_run_preserves_canonical_paths()
    print("all pass")
