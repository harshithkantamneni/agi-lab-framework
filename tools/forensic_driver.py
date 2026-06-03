#!/usr/bin/env python3
"""forensic_driver.py -- Namespaced wrapper around tools/eval_*.py.

PROBLEM this solves
-------------------
The four eval drivers (`tools/eval_mmlu.py`, `tools/eval_hellaswag.py`,
`tools/eval_gsm8k.py`, `tools/eval_winogrande.py`) ALL write their results
to canonical paths under `data/eval/`:

    data/eval/mmlu_results.json
    data/eval/hellaswag_results.json
    data/eval/gsm8k_results.json
    data/eval/winogrande_results.json
    data/eval/scorecard.md

These canonical paths are authoritative for the "current state of the
model" (the Director, evaluator, and dashboard all read them). But when
we run a FORENSIC eval on an old or preserved checkpoint (e.g. the Cycle
26 step_500 post-mortem -- see `data/eval/forensic_cycle26_step500.md`),
those canonical paths get clobbered as a side effect of running the eval
drivers. benchmark_lead noted this explicitly in the Cycle 26 forensic
report, §Methodology line 80:

    "Canonical `data/eval/<benchmark>_results.json` paths were overwritten
     as a side effect (drivers always write to canonical paths)."

Consequence: any regression test, dashboard snapshot, or reviewer trying
to read "the current model's MMLU score" right after a forensic run
reads the FORENSIC checkpoint's score, not the current model's.

Related concern tracked in D-098 §"Dispositions NOT addressed this cycle"
(evaluator Cycle 28 §3, Cycle 29 §7): the forensic driver could also
clobber the canonical `data/checkpoints/cycleXX_*/step_000500.ckpt`
regression-test target via symlink rewriting. The tests at
`tests/test_checkpoint_load_dump.py` lock specific canonical numbers
(Cycle 25 = 2.61, Cycle 27 = 16.84, Cycle 29 = 1.66) -- a clobber breaks
the regression suite.

WHAT this tool does
-------------------
A thin wrapper that:

  1. Creates a namespaced output directory under
     `data/eval/forensic_run_<timestamp>_<tag>/` and copies all existing
     canonical artifacts there as a baseline snapshot.
  2. Before each eval-driver run: moves the canonical
     `data/eval/<bench>_results.json` + `data/eval/scorecard.md` aside
     to a .forensic-save backup.
  3. Runs the eval driver (which writes new canonical outputs).
  4. After the eval driver finishes: moves the just-written canonical
     outputs into the namespaced forensic dir, then restores the
     .forensic-save backups to the canonical paths.

Net effect: after the tool returns, the canonical `data/eval/*` state is
byte-identical to what it was before the tool was invoked. Only the
namespaced `data/eval/forensic_run_<timestamp>_<tag>/` contains the
forensic eval's output.

Checkpoint files: this tool READS checkpoint files only (it passes
`--checkpoint <path>` to the eval drivers). It NEVER writes to
`data/checkpoints/`. It NEVER creates, moves, or rewrites symlinks.
If the eval drivers or the C eval binary were ever to symlink-clobber
the canonical `step_000500.ckpt` targets, that would be THEIR bug --
this wrapper only contributes read access. Grep sanity-verified 2026-04-17
Cycle 31: no tool under tools/ writes to data/checkpoints/.

USAGE
-----
    # Run all 4 benchmarks on a forensic checkpoint, namespaced output.
    python3 tools/forensic_driver.py \\
        --checkpoint data/checkpoints/cycle25_phaseb_gatecheck/step_000500.ckpt \\
        --tokenizer  data/training/tokenizer_32k.bin \\
        --tag cycle25_step500_rerun \\
        --benchmarks mmlu,hellaswag,gsm8k,winogrande

    # Single benchmark with sample cap:
    python3 tools/forensic_driver.py \\
        --checkpoint <ckpt> --tokenizer <tok> --tag my_experiment \\
        --benchmarks mmlu --mmlu-per-subject 4

    # Use --dry-run to see what commands would execute without running
    # the C eval binary:
    python3 tools/forensic_driver.py --checkpoint <ckpt> --tokenizer <tok> \\
        --tag dry --benchmarks mmlu --dry-run

Output structure
----------------
    data/eval/forensic_run_<UTC_ISO>_<tag>/
        baseline_canonical/
            mmlu_results.json         # pre-run snapshot (if existed)
            hellaswag_results.json
            gsm8k_results.json
            winogrande_results.json
            scorecard.md
        forensic_results/
            mmlu_results.json         # new outputs from forensic run
            hellaswag_results.json
            gsm8k_results.json
            winogrande_results.json
            scorecard.md
        metadata.json                 # checkpoint, tokenizer, timestamp, tag,
                                       # benchmarks, exit codes, per-driver
                                       # stdout/stderr tail

Regression safeguard
--------------------
A pytest at `tests/test_forensic_driver_canonical_safe.py` (paired with
this tool) asserts that running forensic_driver.py leaves the canonical
`data/eval/*_results.json` mtimes/contents untouched. If any eval-driver
author ever changes the canonical write path without a compat migration,
that test fails.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "data"
EVAL_DIR = DATA_DIR / "eval"

# Map driver name -> (script_path, canonical_json_name, extra_arg_builder).
# Keep extra_arg_builder a function so each driver can map --max-* args.
DRIVERS = {
    "mmlu": {
        "script": REPO / "tools" / "eval_mmlu.py",
        "canonical_json": "mmlu_results.json",
    },
    "hellaswag": {
        "script": REPO / "tools" / "eval_hellaswag.py",
        "canonical_json": "hellaswag_results.json",
    },
    "gsm8k": {
        "script": REPO / "tools" / "eval_gsm8k.py",
        "canonical_json": "gsm8k_results.json",
    },
    "winogrande": {
        "script": REPO / "tools" / "eval_winogrande.py",
        "canonical_json": "winogrande_results.json",
    },
}
CANONICAL_SCORECARD = "scorecard.md"


def _sha256_short(path: Path) -> str:
    if not path.exists():
        return "absent"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _snapshot_canonical(target_dir: Path) -> dict:
    """Copy current canonical artifacts into target_dir; return {name: sha}."""
    target_dir.mkdir(parents=True, exist_ok=True)
    shas = {}
    for spec in DRIVERS.values():
        src = EVAL_DIR / spec["canonical_json"]
        shas[spec["canonical_json"]] = _sha256_short(src)
        if src.exists():
            shutil.copy2(src, target_dir / spec["canonical_json"])
    sc_src = EVAL_DIR / CANONICAL_SCORECARD
    shas[CANONICAL_SCORECARD] = _sha256_short(sc_src)
    if sc_src.exists():
        shutil.copy2(sc_src, target_dir / CANONICAL_SCORECARD)
    return shas


def _save_aside(canonical_path: Path) -> Path | None:
    """Move canonical_path to canonical_path.with_suffix('.forensic-save')."""
    if not canonical_path.exists():
        return None
    save = canonical_path.with_suffix(canonical_path.suffix + ".forensic-save")
    # Use move (rename) rather than copy to make the canonical path
    # genuinely absent for the duration of the eval-driver run, so we can
    # unambiguously detect whether the driver wrote a new file.
    shutil.move(str(canonical_path), str(save))
    return save


def _restore(save: Path | None, canonical_path: Path) -> None:
    if save is None:
        # There was nothing to save -> if the eval driver wrote one, it
        # still remains at canonical_path. Caller will handle harvesting.
        return
    # Overwrite whatever the driver wrote (harvested separately before).
    if canonical_path.exists():
        canonical_path.unlink()
    shutil.move(str(save), str(canonical_path))


def _harvest(canonical_path: Path, dest_dir: Path) -> Path | None:
    """Move a freshly-written canonical output into dest_dir.

    Returns the new path inside dest_dir, or None if the driver didn't
    write one.
    """
    if not canonical_path.exists():
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / canonical_path.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(canonical_path), str(dest))
    return dest


def _run_driver(
    name: str,
    spec: dict,
    *,
    checkpoint: Path,
    tokenizer: Path,
    per_driver_args: list[str],
    dry_run: bool,
) -> dict:
    """Run a single eval driver. Returns metadata dict."""
    cmd = [
        sys.executable,
        str(spec["script"]),
        "--checkpoint", str(checkpoint),
        "--tokenizer", str(tokenizer),
        *per_driver_args,
    ]
    if dry_run:
        print(f"[forensic:{name}] DRY-RUN would invoke: {' '.join(cmd)}")
        return {"driver": name, "cmd": cmd, "exit_code": None, "dry_run": True}
    print(f"[forensic:{name}] invoking: {' '.join(cmd)}")
    # nice -n 19 ourselves if a live training job is running.
    res = subprocess.run(cmd, capture_output=True, text=True)
    tail = "\n".join(res.stdout.splitlines()[-20:])
    err_tail = "\n".join(res.stderr.splitlines()[-20:])
    return {
        "driver": name,
        "cmd": cmd,
        "exit_code": res.returncode,
        "stdout_tail": tail,
        "stderr_tail": err_tail,
        "dry_run": False,
    }


def run_forensic(
    *,
    checkpoint: Path,
    tokenizer: Path,
    tag: str,
    benchmarks: list[str],
    per_driver_args: dict[str, list[str]],
    dry_run: bool = False,
) -> Path:
    """Main orchestration. Returns the forensic run directory."""
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_tag = "".join(c for c in tag if c.isalnum() or c in "-_")[:48] or "untagged"
    run_dir = EVAL_DIR / f"forensic_run_{ts}_{safe_tag}"
    baseline_dir = run_dir / "baseline_canonical"
    forensic_dir = run_dir / "forensic_results"
    run_dir.mkdir(parents=True, exist_ok=False)  # fail if dir already exists

    # 1. Snapshot pre-run canonical state.
    pre_shas = _snapshot_canonical(baseline_dir)
    print(f"[forensic] baseline snapshot -> {baseline_dir}")

    metadata = {
        "timestamp": ts,
        "tag": safe_tag,
        "checkpoint": str(checkpoint),
        "tokenizer": str(tokenizer),
        "benchmarks": benchmarks,
        "pre_run_canonical_sha256_short": pre_shas,
        "driver_runs": [],
    }

    # 2. Per-driver cycle: save aside, run, harvest, restore.
    scorecard_path = EVAL_DIR / CANONICAL_SCORECARD
    scorecard_save = _save_aside(scorecard_path) if not dry_run else None

    try:
        for bench in benchmarks:
            if bench not in DRIVERS:
                print(f"[forensic] WARNING unknown benchmark '{bench}', skipping")
                continue
            spec = DRIVERS[bench]
            canonical_json = EVAL_DIR / spec["canonical_json"]

            # Save aside the canonical JSON.
            json_save = _save_aside(canonical_json) if not dry_run else None

            run_meta = _run_driver(
                bench, spec,
                checkpoint=checkpoint, tokenizer=tokenizer,
                per_driver_args=per_driver_args.get(bench, []),
                dry_run=dry_run,
            )
            metadata["driver_runs"].append(run_meta)

            # Harvest the new canonical JSON into the forensic dir.
            if not dry_run:
                harvested = _harvest(canonical_json, forensic_dir)
                if harvested:
                    run_meta["harvested_json"] = str(
                        harvested.relative_to(REPO)
                    )
                _restore(json_save, canonical_json)

        # Harvest the forensic scorecard (written by the drivers).
        if not dry_run:
            harvested_sc = _harvest(scorecard_path, forensic_dir)
            if harvested_sc:
                metadata["harvested_scorecard"] = str(
                    harvested_sc.relative_to(REPO)
                )
            _restore(scorecard_save, scorecard_path)
    finally:
        # Defensive: make sure we never leave the canonical state in a
        # half-restored state on exception. If restore already happened,
        # these are no-ops.
        if not dry_run:
            for spec in DRIVERS.values():
                save = (EVAL_DIR / (spec["canonical_json"] + ".forensic-save"))
                canonical = EVAL_DIR / spec["canonical_json"]
                if save.exists() and not canonical.exists():
                    shutil.move(str(save), str(canonical))
            sc_save = scorecard_path.with_suffix(".md.forensic-save")
            if sc_save.exists() and not scorecard_path.exists():
                shutil.move(str(sc_save), str(scorecard_path))

    # 3. Post-run: verify canonical state is identical to baseline.
    post_shas = {
        spec["canonical_json"]: _sha256_short(EVAL_DIR / spec["canonical_json"])
        for spec in DRIVERS.values()
    }
    post_shas[CANONICAL_SCORECARD] = _sha256_short(scorecard_path)
    metadata["post_run_canonical_sha256_short"] = post_shas

    drift = [
        name for name, sha in post_shas.items()
        if pre_shas.get(name) != sha
    ]
    metadata["canonical_paths_drifted"] = drift
    if drift and not dry_run:
        print(
            f"[forensic] !!! WARNING !!! canonical paths drifted: {drift}. "
            "The wrapper's save/restore logic missed something. "
            "Investigate tools/forensic_driver.py before trusting canonical "
            "eval state."
        )

    # 4. Write metadata.json for reviewers.
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"[forensic] done -> {run_dir}")
    if not drift:
        print("[forensic] canonical eval paths verified unchanged "
              "(sha256 match).")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forensic eval wrapper: runs eval_*.py on a checkpoint "
                    "while preserving canonical data/eval/ paths."
    )
    parser.add_argument("--checkpoint", required=True, type=Path,
                        help="Path to checkpoint (read-only).")
    parser.add_argument("--tokenizer", required=True, type=Path,
                        help="Path to tokenizer (read-only).")
    parser.add_argument("--tag", required=True, type=str,
                        help="Short tag for the forensic run dir name.")
    parser.add_argument(
        "--benchmarks", default="mmlu,hellaswag,gsm8k,winogrande",
        help="Comma-separated list from: mmlu, hellaswag, gsm8k, winogrande.",
    )
    # Per-driver passthrough sample caps (match each driver's CLI).
    parser.add_argument("--mmlu-per-subject", type=int, default=None,
                        help="--max-per-subject for eval_mmlu.py")
    parser.add_argument("--hellaswag-max-items", type=int, default=None,
                        help="--max-items for eval_hellaswag.py")
    parser.add_argument("--gsm8k-max-items", type=int, default=None,
                        help="--max-items for eval_gsm8k.py")
    parser.add_argument("--winogrande-max-items", type=int, default=None,
                        help="--max-items for eval_winogrande.py")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands that would run, but don't "
                             "invoke the eval drivers. Still creates the "
                             "forensic run directory and metadata.json.")
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    per_driver_args: dict[str, list[str]] = {b: [] for b in benchmarks}
    if args.mmlu_per_subject is not None and "mmlu" in per_driver_args:
        per_driver_args["mmlu"] += [
            "--max-per-subject", str(args.mmlu_per_subject)
        ]
    if args.hellaswag_max_items is not None and "hellaswag" in per_driver_args:
        per_driver_args["hellaswag"] += [
            "--max-items", str(args.hellaswag_max_items)
        ]
    if args.gsm8k_max_items is not None and "gsm8k" in per_driver_args:
        per_driver_args["gsm8k"] += [
            "--max-items", str(args.gsm8k_max_items)
        ]
    if args.winogrande_max_items is not None and "winogrande" in per_driver_args:
        per_driver_args["winogrande"] += [
            "--max-items", str(args.winogrande_max_items)
        ]

    # Fail-fast validation.
    if not args.checkpoint.exists():
        sys.exit(f"ERROR: checkpoint not found: {args.checkpoint}")
    if not args.tokenizer.exists():
        sys.exit(f"ERROR: tokenizer not found: {args.tokenizer}")
    for b in benchmarks:
        if b not in DRIVERS:
            sys.exit(
                f"ERROR: unknown benchmark '{b}'. "
                f"Known: {sorted(DRIVERS.keys())}"
            )
        if not DRIVERS[b]["script"].exists():
            sys.exit(f"ERROR: driver script missing: {DRIVERS[b]['script']}")

    run_forensic(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        tag=args.tag,
        benchmarks=benchmarks,
        per_driver_args=per_driver_args,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
