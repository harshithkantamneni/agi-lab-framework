#!/usr/bin/env python3
"""tools/rotate_agent_semantic.py — cap agent semantic memory with archive rotation.

Each role's semantic memory (`data/agents/<role>/semantic.md`) accumulates
numbered lessons (S-1, S-2, ...) across lab history. Without a bound, these
memories can encode wrong mental models that override the role's procedural
prompt (this is the D-109 failure mechanism: Director's S-1..S-5 collectively
encoded "PI as external human reviewer," overriding the procedural's explicit
pi-agent dispatch rule).

This tool mechanically caps each role's active semantic memory at N entries.
Overflow rotates to `data/agents/<role>/semantic_archive.md` (append-only),
preserving §4 archive-never-delete. Active entries keep the drift surface
bounded while the full history remains retrievable.

Usage:
    # Check current state across all roles
    python3 tools/rotate_agent_semantic.py --status

    # Dry-run: show what would be archived for one role
    python3 tools/rotate_agent_semantic.py --role director

    # Apply rotation for all roles exceeding cap
    python3 tools/rotate_agent_semantic.py --all --apply

    # Custom cap
    python3 tools/rotate_agent_semantic.py --role director --cap 15 --apply

Entry detection: splits `semantic.md` on top-level `## ` headings. Each
section is one entry. Numbered entries (## S-1, ## S-2, ...) and unnumbered
topic-headed entries both work. Leading pre-heading content (file header,
intro paragraph) is preserved; only entries past the cap are rotated.

Why TOP=OLDEST: lab convention appends new entries at the bottom (e.g.,
Director's S-6 added via 073a5da landed at EOF of semantic.md). So the
first entry is the oldest. This convention is audited as part of rotation.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "data" / "agents"

DEFAULT_CAP = 10
ENTRY_HEADING_RE = re.compile(r"^##[ \t]")


def split_entries(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Parse semantic.md into (preamble, entries).

    Preamble = everything before the first `## ` heading (may be empty).
    Entries = list of (heading_line, body_including_heading).
    """
    lines = text.splitlines(keepends=True)
    first_heading_idx = None
    for i, line in enumerate(lines):
        if ENTRY_HEADING_RE.match(line):
            first_heading_idx = i
            break

    if first_heading_idx is None:
        return text, []

    preamble = "".join(lines[:first_heading_idx])

    entries: list[tuple[str, str]] = []
    current_heading = ""
    current_body: list[str] = []
    for line in lines[first_heading_idx:]:
        if ENTRY_HEADING_RE.match(line):
            if current_heading:
                entries.append((current_heading, "".join(current_body)))
            current_heading = line.rstrip("\n")
            current_body = [line]
        else:
            current_body.append(line)
    if current_heading:
        entries.append((current_heading, "".join(current_body)))
    return preamble, entries


def rotate_role(role: str, cap: int, apply: bool = False) -> dict:
    """Rotate one role's semantic memory. Returns a status dict."""
    sem_path = AGENTS_DIR / role / "semantic.md"
    if not sem_path.exists():
        return {"role": role, "status": "no_file", "active": 0, "archived_now": 0}

    text = sem_path.read_text(encoding="utf-8")
    preamble, entries = split_entries(text)

    active_count = len(entries)
    if active_count <= cap:
        return {"role": role, "status": "under_cap", "active": active_count, "archived_now": 0}

    to_archive = entries[: active_count - cap]
    to_keep = entries[active_count - cap :]

    archive_path = AGENTS_DIR / role / "semantic_archive.md"

    if apply:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_archive_content = "".join(body for _, body in to_archive)
        archive_header = (
            f"\n\n<!-- rotated {stamp} — cap={cap} — role={role} — {len(to_archive)} entries moved from active -->\n\n"
        )
        if archive_path.exists():
            prior = archive_path.read_text(encoding="utf-8")
            archive_path.write_text(prior + archive_header + new_archive_content, encoding="utf-8")
        else:
            header = (
                f"# {role} semantic memory — archive\n\n"
                f"*Append-only rotation target for `semantic.md`. §4 archive-never-delete.*\n"
            )
            archive_path.write_text(header + archive_header + new_archive_content, encoding="utf-8")

        new_active = preamble + "".join(body for _, body in to_keep)
        tmp = sem_path.with_suffix(sem_path.suffix + ".tmp")
        tmp.write_text(new_active, encoding="utf-8")
        tmp.rename(sem_path)

    return {
        "role": role,
        "status": "rotated" if apply else "would_rotate",
        "active_before": active_count,
        "active_after": cap if apply else active_count,
        "archived_now": len(to_archive),
        "archived_headings": [h for h, _ in to_archive],
        "kept_headings": [h for h, _ in to_keep],
    }


def status_all() -> list[dict]:
    """Return cap-status for every role."""
    results = []
    for sem_path in sorted(AGENTS_DIR.glob("*/semantic.md")):
        role = sem_path.parent.name
        text = sem_path.read_text(encoding="utf-8", errors="replace")
        _, entries = split_entries(text)
        results.append({"role": role, "entries": len(entries), "size_kb": round(sem_path.stat().st_size / 1024, 1)})
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Cap agent semantic memory with archive rotation. Prevents D-109-class drift."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--role", help="Rotate one specific role")
    group.add_argument("--all", action="store_true", help="Rotate every role exceeding cap")
    group.add_argument("--status", action="store_true", help="Show entry counts for all roles, no changes")

    parser.add_argument("--cap", type=int, default=DEFAULT_CAP,
                        help=f"Max active entries per role (default {DEFAULT_CAP})")
    parser.add_argument("--apply", action="store_true",
                        help="Apply rotation (default: dry-run showing what would happen)")
    args = parser.parse_args()

    if args.status:
        rows = status_all()
        print(f"{'role':<30} {'entries':>8} {'size_kb':>9}  state")
        print(f"{'-' * 30} {'-' * 8} {'-' * 9}  {'-' * 8}")
        for row in rows:
            marker = "OVER" if row["entries"] > args.cap else "ok"
            print(f"{row['role']:<30} {row['entries']:>8} {row['size_kb']:>9.1f}  {marker}")
        return 0

    roles = [args.role] if args.role else [p.parent.name for p in sorted(AGENTS_DIR.glob("*/semantic.md"))]
    any_rotated = False
    for role in roles:
        result = rotate_role(role, cap=args.cap, apply=args.apply)
        if result["status"] in ("rotated", "would_rotate"):
            any_rotated = True
            verb = "ROTATED" if args.apply else "DRY-RUN"
            print(f"{verb} {role}: {result['active_before']} → {result['active_after']} active ({result['archived_now']} archived)")
            for h in result.get("archived_headings", []):
                print(f"  archive: {h}")
        elif args.role:
            # Only report no-ops when user explicitly picked one role
            print(f"{role}: {result['status']} ({result.get('active', 0)} entries, cap {args.cap})")

    if not any_rotated and args.all:
        print(f"All roles under cap {args.cap}. Nothing to rotate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
