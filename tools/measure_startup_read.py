#!/usr/bin/env python3
"""tools/measure_startup_read.py — measure Director startup-orientation read size.

Parses a session log for the first N `Read <path>` tool calls (those performed
before the Director dispatches its first specialist agent — the "orientation"
phase) and sums the byte sizes of the files it read. Emits `startup_kb=N` on
stdout.

Used by run_agi_lab.sh post-session to append a pre-refactor baseline to
data/infra/memory_telemetry.log. Post-memory-refactor, the same measurement
method will show the <=10 KB target is hit (or not).

Usage:
    python3 tools/measure_startup_read.py <session_log_path>
    python3 tools/measure_startup_read.py <session_log_path> --before-dispatch
    python3 tools/measure_startup_read.py <session_log_path> --limit-lines 200

Default behavior: count Read calls before the first `>>> AGENT` dispatch
marker, OR the first 100 tool calls if no dispatch occurred (safety cap).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Strip ANSI escape sequences from streamed log lines.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")

# Match the stream_formatter's tool-call line for Read:
#   <ANSI>┌<ANSI> <ANSI>Read<ANSI> <path>
# After ANSI stripping:
#   ┌ Read <path>
READ_RE = re.compile(r"^\s*┌\s+Read\s+(\S+)")

# First agent dispatch marker.
DISPATCH_RE = re.compile(r">>> AGENT")


def measure(log_path: Path, before_dispatch: bool = True, limit_lines: int = 200) -> dict:
    """Return dict with startup_kb, read_count, and file list."""
    if not log_path.exists():
        return {"startup_kb": 0, "read_count": 0, "files": [], "error": f"log not found: {log_path}"}

    read_paths: list[str] = []
    line_count = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line_count += 1
            stripped = ANSI_RE.sub("", line)
            if before_dispatch and DISPATCH_RE.search(stripped):
                break
            if line_count > limit_lines:
                break
            m = READ_RE.match(stripped)
            if m:
                read_paths.append(m.group(1))

    total_bytes = 0
    sized_files: list[tuple[str, int]] = []
    for p in read_paths:
        # Resolve relative to repo root. Skip paths we can't stat.
        candidate = (REPO_ROOT / p).resolve() if not Path(p).is_absolute() else Path(p)
        try:
            if candidate.exists() and candidate.is_file():
                size = candidate.stat().st_size
                total_bytes += size
                sized_files.append((p, size))
        except OSError:
            continue

    startup_kb = total_bytes / 1024.0
    return {
        "startup_kb": round(startup_kb, 2),
        "read_count": len(read_paths),
        "sized_count": len(sized_files),
        "files": sized_files,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure Director startup-orientation read size from a session log.")
    parser.add_argument("log_path", type=Path, help="Path to session log file")
    parser.add_argument("--limit-lines", type=int, default=200,
                        help="Safety cap: don't scan past N lines (default 200)")
    parser.add_argument("--no-before-dispatch", action="store_false", dest="before_dispatch",
                        help="Don't stop at first >>> AGENT marker; use limit-lines only")
    parser.add_argument("--verbose", action="store_true", help="Print each read with its size")
    args = parser.parse_args()

    result = measure(args.log_path, before_dispatch=args.before_dispatch, limit_lines=args.limit_lines)
    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        for path, size in result["files"]:
            print(f"  {size // 1024:4d}K  {path}", file=sys.stderr)
        print(f"  read_count={result['read_count']} sized={result['sized_count']}", file=sys.stderr)

    # Machine-readable output on stdout (key=value). The runner greps this.
    print(f"startup_kb={result['startup_kb']}")


if __name__ == "__main__":
    main()
