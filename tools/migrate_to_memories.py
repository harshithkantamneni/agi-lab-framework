#!/usr/bin/env python3
"""tools/migrate_to_memories.py -- one-time bootstrap for data/memories/.

Run once at migration gate (Program 2 Phase 2 close). Deleted from the tree
after 3 consecutive clean phases post-migration per spec §12.3.

Spec: docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md §13
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.memory import Memory  # noqa: E402


LEGACY_TO_WIKI = {
    "data/pi_notes.md":                           "governance/pi_notes.md",
    "data/values.md":                             "governance/values.md",
    "data/procedures.md":                         "procedures.md",
    "data/shared_knowledge.md":                   "shared.md",
    "data/killed_ideas.md":                       "killed.md",
    "data/mission_reframe_2026-04-18.md":         "mission_reframe_2026-04-18.md",
    "data/checkpoints/ARCHIVED.md":               "checkpoints_archived.md",
    "programs/portfolio.md":                      "programs/portfolio.md",
}


def _snapshot_all(repo_root: Path) -> Path:
    snap_root = repo_root / "data" / "archives" / "snapshots" / "pre-memory-refactor"
    snap_root.mkdir(parents=True, exist_ok=True)
    candidates = [
        "data/state.md", "data/pi_notes.md", "data/values.md",
        "data/shared_knowledge.md", "data/killed_ideas.md",
        "data/procedures.md", "data/index.md",
        "data/director_log.md", "data/decisions_recent.md",
        "data/mission_reframe_2026-04-18.md",
        "data/checkpoints/ARCHIVED.md",
        "programs/portfolio.md",
    ]
    for rel in candidates:
        src = repo_root / rel
        if src.exists():
            dest = snap_root / Path(rel).name
            shutil.copy(src, dest)
    return snap_root


def _extract_mission(repo_root: Path) -> str:
    claude = (repo_root / "CLAUDE.md").read_text() if (repo_root / "CLAUDE.md").exists() else ""
    reframe = ""
    reframe_path = repo_root / "data" / "mission_reframe_2026-04-18.md"
    if reframe_path.exists():
        reframe = reframe_path.read_text()
    parts = ["# Mission\n\n"]
    if "## Mission" in claude:
        # Grab the Mission section from CLAUDE.md
        start = claude.index("## Mission")
        end = claude.find("\n## ", start + 1)
        parts.append(claude[start: end if end != -1 else len(claude)] + "\n")
    if reframe:
        parts.append("\n---\n\n## Reframe context (2026-04-18)\n\n")
        parts.append(reframe)
    return "".join(parts)


def _build_current_md(repo_root: Path) -> str:
    state = repo_root / "data" / "state.md"
    if not state.exists():
        return "# Current state\n\n(empty — populate at next phase transition)\n"
    raw = state.read_text()
    # Pass through for now; in real migration the Director-team prunes it at
    # first phase transition. We keep the header and truncate at 36 KB (90% of cap).
    head = "# Current state\n\n(migrated from data/state.md on first cutover;"
    head += " Director curates down to active-only content)\n\n"
    return head + raw[:36 * 1024]


def _merge_logs(repo_root: Path) -> tuple[str, str]:
    """Merge director_log.md + decisions_recent.md, keeping newest under 30KB."""
    parts = []
    for rel in ("data/decisions_recent.md", "data/director_log.md"):
        f = repo_root / rel
        if f.exists():
            parts.append(f.read_text())
    merged = "\n\n".join(parts)
    # Split on `## ` headings, keep newest N that fit in cap.
    lines = merged.splitlines(keepends=True)
    segments = []
    buf = []
    for line in lines:
        if line.startswith("## ") and buf:
            segments.append("".join(buf))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        segments.append("".join(buf))
    cap = 30 * 1024
    kept, kept_size = [], 0
    overflow_start = len(segments)
    for i, seg in enumerate(segments):
        size = len(seg.encode("utf-8"))
        if kept_size + size > cap:
            overflow_start = i
            break
        kept.append(seg)
        kept_size += size
    overflow = segments[overflow_start:]
    return "".join(kept) if kept else "# Log\n\n(empty)\n", "".join(overflow)


def _summarize_roster(repo_root: Path) -> str:
    import json
    agents = repo_root / "data" / "agents" / "agents.json"
    if not agents.exists():
        return "# Roster\n\n(agents.json not found)\n"
    data = json.loads(agents.read_text())
    lines = ["# Roster\n\n", "See `data/agents/agents.json` for authoritative definitions.\n\n"]
    for key, entry in data.items():
        name = entry.get("name", key) if isinstance(entry, dict) else key
        desc = entry.get("description", "") if isinstance(entry, dict) else ""
        lines.append(f"- **{name}** — {desc[:80]}\n")
    return "".join(lines)


def migrate(repo_root: Path, force: bool = False) -> None:
    repo_root = Path(repo_root).resolve()
    mem_root = repo_root / "data" / "memories"
    arch_root = repo_root / "data" / "archives" / "memories-deletes"
    snap_root = repo_root / "data" / "archives" / "memories-snapshots"

    # Safety guard: refuse to clobber a populated memories/ unless --force
    if mem_root.exists() and any(mem_root.iterdir()) and not force:
        raise RuntimeError(
            f"memories already exists and is non-empty at {mem_root}. "
            f"Pass force=True (or --force on CLI) to proceed. "
            f"Existing content will be atomically overwritten file-by-file."
        )

    _snapshot_all(repo_root)

    m = Memory(root=mem_root, archive_root=arch_root, snapshot_root=snap_root)

    # Hot tier
    m.create("current.md", _build_current_md(repo_root))

    # Wiki tier
    m.create("mission.md", _extract_mission(repo_root))
    for legacy, dest in LEGACY_TO_WIKI.items():
        src = repo_root / legacy
        if src.exists():
            m.create(dest, src.read_text())
    m.create("roster.md", _summarize_roster(repo_root))

    # Programs: seed active.md with a stub pointer; Director fills at first session.
    active_stub = (
        "# Active program state\n\n"
        "*To be populated by Director at first post-migration phase transition.*\n"
    )
    if not (mem_root / "programs" / "active.md").exists():
        m.create("programs/active.md", active_stub)

    # Log tier (merged + rotated)
    live_log, overflow = _merge_logs(repo_root)
    m.create("log.md", live_log)
    if overflow:
        from datetime import datetime, timezone
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        arch_dir = repo_root / "data" / "archives" / stamp
        arch_dir.mkdir(parents=True, exist_ok=True)
        (arch_dir / "log.md").write_text(overflow)
        m.create(
            "history.md",
            f"# Session history\n\n- {stamp}: pre-migration overflow → `{arch_dir}/log.md`\n",
        )
    else:
        m.create("history.md", "# Session history\n\n(no rotations yet)\n")

    # INDEX
    m.index()


def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=".", help="repo root (default: cwd)")
    p.add_argument("--dry-run", action="store_true",
                   help="run in temp dir, don't touch repo")
    p.add_argument("--force", action="store_true",
                   help="overwrite existing data/memories/ if non-empty")
    args = p.parse_args()
    if args.dry_run:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp) / "repo"
            shutil.copytree(args.repo_root, tmp_root, symlinks=True,
                            ignore=shutil.ignore_patterns(
                                ".git", ".venv", "__pycache__", "*.db",
                                "archives", "checkpoints",
                                "*.safetensors", "*.bin", "*.pt",
                            ))
            migrate(tmp_root, force=args.force)
            print(f"Dry-run complete. Inspect: {tmp_root / 'data' / 'memories'}")
    else:
        migrate(Path(args.repo_root), force=args.force)
        print("Migration complete.")


if __name__ == "__main__":
    _cli()
