#!/usr/bin/env python3
"""tools/session_productivity.py — post-session productivity check.

Detects sessions where the Director (or any agent) generated a lot of
tokens but produced very little durable output — the within-session
analog of D-202..D-210's between-session holding loop. Output: write
a warning to data/diagnostics/ if productivity is low; emit a one-line
summary to stdout for the runner to fold into telemetry.

Concept (Lab Weakness #5, addressed D-233):
    productivity = output_bytes / session_log_bytes
    where:
        output_bytes  = git diff size of (data/, programs/) since
                        session start, EXCLUDING data/infra/session_logs/
        session_log_bytes = size of the session's primary log file

A productive session converts session_log into deliverables: log.md
entries, current.md updates, programs/ deliverables, agent episodic
records, code changes. An unproductive session writes lots of session
log but nothing else (e.g., chief_scientist looping on synthesis without
producing the final document).

Threshold: ratio < MIN_RATIO for a session that is also above MIN_LOG
size (otherwise small honest sessions trip the detector). Default
MIN_RATIO=0.05 — output must be at least 5% the size of the session log.

Output:
    stdout: one line `PRODUCTIVITY: <ratio> session=<bytes> output=<bytes> verdict=<good|low>`
    side-effect: if verdict=low, append to data/diagnostics/low_productivity_sessions.md
                 with timestamp + session number + the metrics, so the
                 next Director session sees it on startup.

Usage:
    python3 tools/session_productivity.py SESSION_LOG_PATH SESSION_NUM

Invoked by run_agi_lab.sh after each session.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DIAGNOSTICS_DIR = REPO_ROOT / "data" / "diagnostics"
WARNING_FILE = DIAGNOSTICS_DIR / "low_productivity_sessions.md"

MIN_RATIO = 0.05
MIN_LOG_BYTES = 20_000


def session_log_size(path: Path) -> int:
    if not path.is_file():
        return 0
    return path.stat().st_size


def output_size_since(commit_or_marker: str) -> int:
    """Return weighted bytes of REAL durable output produced since the
    given marker.

    Why weighted: holding-pattern sessions (D-198..D-225 incident) edit
    `data/memories/current.md` and `data/memories/log.md` every cycle
    regardless of whether real work happened. Counting those as
    "output" makes the detector blind to the failure mode it's meant
    to detect. So we weight by file class:

      Real work signals (weight 1.0):
        - data/agents/*/episodic/*  (agent dispatch records)
        - programs/<*>/*            (program deliverables)
        - data/findings/*            (cross-role findings)
        - data/memories/programs/*  (program-level memory)
        - tools/*, src/*, tests/*   (code changes)

      Self-write signals (weight 0.2):
        - data/memories/current.md  (Director writes every session)
        - data/memories/log.md      (Director prepends every session)
        - data/session_exit.md      (Director writes every session)

      Excluded (weight 0):
        - **/session_logs/*         (the input we're measuring)
        - **/__pycache__/*

    This matches "did the Director dispatch real agents that produced
    deliverables?" rather than "did any byte change?".
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--numstat", commit_or_marker],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return 0
        total_weighted_lines = 0.0
        for line in result.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            added, removed, fname = parts[0], parts[1], parts[2]
            if added == "-" or removed == "-":
                continue
            try:
                lines = int(added) + int(removed)
            except ValueError:
                continue
            # Excluded
            if "session_logs" in fname or "__pycache__" in fname:
                continue
            # Self-write — heavily down-weighted
            if fname in (
                "data/memories/current.md",
                "data/memories/log.md",
                "data/session_exit.md",
            ):
                weight = 0.2
            # Real-work signals — full weight
            elif (
                fname.startswith("data/agents/") and "/episodic/" in fname
            ) or (
                fname.startswith("programs/")
            ) or (
                fname.startswith("data/findings/")
            ) or (
                fname.startswith("data/memories/programs/")
            ) or (
                fname.startswith("tools/")
                or fname.startswith("src/")
                or fname.startswith("tests/")
            ):
                weight = 1.0
            # Other data/ + memories/ — moderate weight (procedural updates,
            # killed-ideas append, etc. are real but rarer)
            else:
                weight = 0.5
            total_weighted_lines += lines * weight
        return int(total_weighted_lines * 80)
    except (FileNotFoundError, subprocess.SubprocessError):
        return 0


def append_warning(session_num: int, ratio: float,
                   session_bytes: int, output_bytes: int,
                   session_log: Path) -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = (
        f"\n## Session {session_num} — {timestamp}\n\n"
        f"- session log: `{session_log.relative_to(REPO_ROOT)}` "
        f"({session_bytes:,} bytes)\n"
        f"- durable output: ~{output_bytes:,} bytes "
        f"(git diff --numstat data/ programs/, excl. session_logs)\n"
        f"- ratio: {ratio:.4f} "
        f"(threshold: {MIN_RATIO}, min-log: {MIN_LOG_BYTES:,})\n"
        f"- verdict: LOW PRODUCTIVITY — session generated tokens without "
        f"durable output. Director should investigate which agent looped, "
        f"and either route around the failure or escalate to operator if "
        f"the same agent loops across multiple consecutive sessions.\n"
    )
    if not WARNING_FILE.exists():
        WARNING_FILE.write_text(
            "# Low-productivity session log\n\n"
            "*Auto-generated by `tools/session_productivity.py` "
            "(invoked from `run_agi_lab.sh` after each session). "
            "Tracks sessions where token-spend was high but durable "
            "output was low — the within-session analog of D-202..D-210. "
            "Director should read this on startup; if the pattern "
            "repeats across 3+ consecutive sessions, escalate to "
            "operator via `OPERATOR_REVIEW_*.md` in the current "
            "program directory.*\n",
            encoding="utf-8",
        )
    with WARNING_FILE.open("a", encoding="utf-8") as f:
        f.write(entry)


def main() -> int:
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: session_productivity.py SESSION_LOG SESSION_NUM\n")
        return 2
    session_log = Path(sys.argv[1])
    try:
        session_num = int(sys.argv[2])
    except ValueError:
        session_num = -1

    session_bytes = session_log_size(session_log)
    if session_bytes < MIN_LOG_BYTES:
        # Small honest session — skip the check entirely.
        print(
            f"PRODUCTIVITY: skip session={session_bytes} "
            f"output=- verdict=skip-small-session"
        )
        return 0

    # Compare against HEAD before this session — best-effort. The runner
    # commits between sessions so HEAD~1 is a reasonable marker. If git
    # history is shallow or commits aren't reliable, we fall through to
    # zero output and the verdict will be conservatively "low".
    output_bytes = output_size_since("HEAD~1")
    ratio = output_bytes / max(session_bytes, 1)
    verdict = "good" if ratio >= MIN_RATIO else "low"

    print(
        f"PRODUCTIVITY: ratio={ratio:.4f} "
        f"session={session_bytes} output={output_bytes} verdict={verdict}"
    )
    if verdict == "low":
        append_warning(session_num, ratio, session_bytes,
                       output_bytes, session_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
