#!/usr/bin/env python3
"""tools/memory.py -- Anthropic memory_20250818-protocol tool for AGI lab.

Implements the six protocol commands (view/create/str_replace/insert/delete/rename)
plus AGI extensions (search/snapshot) and admin commands (index/audit/rotate-log).

Spec: docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md

Deviation from Anthropic protocol: `delete` archives + breadcrumbs rather than
destroying. Values §4 archive-never-delete.
"""
from __future__ import annotations

import sys
from pathlib import Path


class MemoryError(Exception):
    """Base for memory-tool errors."""


class PathOutsideRootError(MemoryError):
    """Path resolves outside the memory root."""


class MemoryFileNotFoundError(MemoryError):
    """Requested memory path does not exist."""


class NonUniqueMatchError(MemoryError):
    """str_replace `old_str` not unique in target file."""


class NoMatchError(MemoryError):
    """str_replace `old_str` not found in target file."""


class UnresolvedReferenceError(MemoryError):
    """Reference tag does not resolve to an existing path."""


def _get_lab_memory():
    from tools.lab_memory import LabMemory
    lm = LabMemory("tools/lab_memory.db")
    return lm


class Memory:
    # Tier caps in bytes
    CAP_HOT = 40 * 1024
    CAP_WIKI_TOTAL = 100 * 1024  # raised 50→100 KB at D-184 per D-182 retro + PI unanimous
    CAP_LOG = 30 * 1024
    WIKI_PER_FILE_SOFT = 15 * 1024

    # Tier membership by filename (first-tier is the convention)
    HOT_FILES = {"current.md"}
    LOG_FILES = {"log.md"}

    def __init__(self, root: Path | str, archive_root: Path | str | None = None,
                 snapshot_root: Path | str | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if archive_root is None:
            archive_root = self.root.parent / "archives" / "deletes"
        self.archive_root = Path(archive_root).resolve()
        if snapshot_root is None:
            snapshot_root = self.root.parent / "archives" / "snapshots"
        self.snapshot_root = Path(snapshot_root).resolve()
        # repo_root defaults to memory root's parent.parent (data/memories → repo root)
        self.repo_root = self.root.parent.parent

    def _resolve(self, path: str) -> Path:
        if Path(path).is_absolute():
            raise PathOutsideRootError(f"absolute path rejected: {path}")
        candidate = (self.root / path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PathOutsideRootError(f"path escapes memory root: {path}")
        return candidate

    def view(self, path: str, view_range: tuple[int, int] | None = None):
        target = self._resolve(path) if path else self.root
        if target.is_dir():
            children = []
            for child in sorted(target.iterdir()):
                children.append(child.name + ("/" if child.is_dir() else ""))
            return children
        if not target.exists():
            raise MemoryFileNotFoundError(str(target.relative_to(self.root)))
        text = target.read_text()
        if view_range is None:
            return text
        start, end = view_range  # 1-indexed inclusive
        lines = text.splitlines(keepends=True)
        return "".join(lines[start - 1 : end])

    def create(self, path: str, file_text: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(file_text)
        tmp.rename(target)

    def str_replace(self, path: str, old_str: str, new_str: str) -> None:
        target = self._resolve(path)
        if not target.exists():
            raise MemoryFileNotFoundError(str(target.relative_to(self.root)))
        text = target.read_text()
        count = text.count(old_str)
        if count == 0:
            raise NoMatchError(f"old_str not found in {path}")
        if count > 1:
            raise NonUniqueMatchError(
                f"old_str matches {count} places in {path}; require unique match"
            )
        self.create(path, text.replace(old_str, new_str, 1))

    def insert(self, path: str, insert_line: int, insert_text: str) -> None:
        target = self._resolve(path)
        if not target.exists():
            raise MemoryFileNotFoundError(str(target.relative_to(self.root)))
        lines = target.read_text().splitlines(keepends=True)
        if insert_line < 0 or insert_line > len(lines):
            raise ValueError(f"insert_line {insert_line} out of range")
        new_lines = lines[:insert_line] + [insert_text] + lines[insert_line:]
        self.create(path, "".join(new_lines))

    def rename(self, old_path: str, new_path: str) -> None:
        src = self._resolve(old_path)
        dst = self._resolve(new_path)
        if not src.exists():
            raise MemoryFileNotFoundError(str(src.relative_to(self.root)))
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

    def delete(self, path: str) -> Path:
        from datetime import datetime, timezone
        target = self._resolve(path)
        if not target.exists():
            raise MemoryFileNotFoundError(str(target.relative_to(self.root)))
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_dir = self.archive_root / stamp
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / target.relative_to(self.root)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        # Copy content to archive
        archive_path.write_text(target.read_text())
        # Replace original with breadcrumb
        breadcrumb = f"<!-- archived {stamp} → {archive_path} -->\n"
        self.create(path, breadcrumb)
        return archive_path

    def snapshot(self, path: str) -> Path:
        from datetime import datetime, timezone
        target = self._resolve(path)
        if not target.exists():
            raise MemoryFileNotFoundError(str(target.relative_to(self.root)))
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%f")
        dest = self.snapshot_root / stamp / target.relative_to(self.root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(target.read_text())
        return dest

    def search(self, query: str, k: int = 5):
        """Semantic search via lab_memory.py. Returns list[lab_memory.Hit]."""
        lm = _get_lab_memory()
        return lm.search(query, top_k=k)

    def resolve(self, reference: str) -> Path:
        import re
        match = re.match(r"\[(\w+):([^\]]+)\]", reference.strip())
        if not match:
            raise UnresolvedReferenceError(f"malformed reference: {reference}")
        tag, body = match.group(1), match.group(2)
        if tag == "episodic":
            # [episodic:<role>/<file>]
            rel = Path(body + ".md")
            candidate = self.repo_root / "data" / "agents" / rel.parts[0] / "episodic" / Path(*rel.parts[1:])
            # Enforce that resolved candidate stays under data/agents/ to block .. traversal
            agents_root = (self.repo_root / "data" / "agents").resolve()
            try:
                candidate.resolve().relative_to(agents_root)
            except ValueError:
                raise UnresolvedReferenceError(
                    f"episodic path escapes data/agents: {reference}"
                )
            if candidate.exists():
                return candidate.resolve()
            raise UnresolvedReferenceError(f"episodic not found: {candidate}")
        if tag == "log":
            # [log:YYYY-MM-DD#D-NNN] — try live log first, then archives.
            # Anchor matches only as a `## <anchor>` line header (not substring),
            # so D-1 does not accidentally match D-11 or D-100.
            if "#" in body:
                date_part, anchor = body.split("#", 1)
            else:
                date_part, anchor = body, None
            pattern = (
                re.compile(rf"^##\s+{re.escape(anchor)}\b", re.MULTILINE)
                if anchor
                else None
            )
            live = self.root / "log.md"
            if live.exists() and pattern is not None and pattern.search(live.read_text()):
                return live.resolve()
            archive = self.repo_root / "data" / "archives" / date_part / "log.md"
            if archive.exists():
                if pattern is None or pattern.search(archive.read_text()):
                    return archive.resolve()
            raise UnresolvedReferenceError(f"log reference unresolved: {reference}")
        if tag == "session":
            hist = self.root / "history.md"
            if hist.exists() and body in hist.read_text():
                return hist.resolve()
            raise UnresolvedReferenceError(f"session reference unresolved: {reference}")
        raise UnresolvedReferenceError(f"unknown reference tag: {tag}")

    def _tier_of(self, rel_path: str) -> str:
        name = Path(rel_path).name
        if name in self.HOT_FILES:
            return "hot"
        if name in self.LOG_FILES or rel_path.startswith("log/"):
            return "log"
        if rel_path in {"session_brief.md", "INDEX.md", "history.md"}:
            return "meta"
        return "wiki"

    def audit(self) -> list[str]:
        breaches = []
        wiki_total = 0
        for f in sorted(self.root.rglob("*.md")):
            rel = str(f.relative_to(self.root))
            size = f.stat().st_size
            tier = self._tier_of(rel)
            if tier == "hot" and size > self.CAP_HOT:
                breaches.append(f"HOT {rel}: {size} > {self.CAP_HOT}")
            elif tier == "log" and size > self.CAP_LOG:
                breaches.append(f"LOG {rel}: {size} > {self.CAP_LOG}")
            elif tier == "wiki":
                wiki_total += size
                if size > self.WIKI_PER_FILE_SOFT:
                    breaches.append(
                        f"WIKI-SOFT {rel}: {size} > {self.WIKI_PER_FILE_SOFT} (soft per-file)"
                    )
        if wiki_total > self.CAP_WIKI_TOTAL:
            breaches.append(f"WIKI-TOTAL: {wiki_total} > {self.CAP_WIKI_TOTAL}")
        return breaches

    def index(self) -> None:
        lines = ["# Memory INDEX\n", "\n",
                 "*Auto-regenerated by `memory.py index`. Do not hand-edit.*\n", "\n"]
        for f in sorted(self.root.rglob("*.md")):
            if f.name == "INDEX.md":
                continue
            rel = f.relative_to(self.root)
            size = f.stat().st_size
            # Grab first non-empty line as description, skipping YAML frontmatter
            desc = ""
            in_frontmatter = False
            for i, line in enumerate(f.read_text().splitlines()):
                stripped = line.strip()
                if stripped == "---":
                    if i == 0:
                        in_frontmatter = True
                        continue
                    if in_frontmatter:
                        in_frontmatter = False
                        continue
                if in_frontmatter:
                    continue
                if stripped:
                    desc = stripped.lstrip("# ").strip()[:80]
                    break
            lines.append(f"- [`{rel}`]({rel}) — {size // 1024}K — {desc}\n")
        self.create("INDEX.md", "".join(lines))

    def rotate_log(self, cap_kb: int = 30) -> Path | None:
        from datetime import datetime, timezone
        log = self._resolve("log.md")
        if not log.exists():
            return None
        cap_bytes = cap_kb * 1024
        content = log.read_text()
        if len(content.encode("utf-8")) <= cap_bytes:
            return None
        # Split on entry boundaries (lines starting with `## `)
        segments = []
        buf = []
        for line in content.splitlines(keepends=True):
            if line.startswith("## ") and buf:
                segments.append("".join(buf))
                buf = [line]
            else:
                buf.append(line)
        if buf:
            segments.append("".join(buf))
        # Keep newest-first convention: first N segments that fit in cap stay; rest archive
        kept = []
        kept_size = 0
        overflow_start = 0
        for i, seg in enumerate(segments):
            seg_bytes = len(seg.encode("utf-8"))
            if kept_size + seg_bytes > cap_bytes:
                overflow_start = i
                break
            kept.append(seg)
            kept_size += seg_bytes
        else:
            overflow_start = len(segments)
        overflow = segments[overflow_start:]
        if not overflow:
            return None
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_dir = self.archive_root.parent / stamp
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_path = archive_dir / "log.md"
        existing = archive_path.read_text() if archive_path.exists() else ""
        archive_path.write_text(existing + "".join(overflow))
        self.create("log.md", "".join(kept))
        # Breadcrumb in history.md
        hist = self._resolve("history.md")
        hist_line = f"- {stamp}: rotated {len(overflow)} entries to `{archive_path}`\n"
        if hist.exists():
            self.insert("history.md", 0, hist_line)
        else:
            self.create("history.md", "# Session history\n\n" + hist_line)
        return archive_path


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description=(
            "Anthropic memory_20250818-compatible tool for AGI lab. "
            "Deviation: delete archives + breadcrumbs (Values §4 archive-never-delete). "
            "Extensions: search, snapshot. Admin: index, audit, rotate-log."
        )
    )
    parser.add_argument("--root", default="data/memories",
                        help="memory root directory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_view = sub.add_parser("view", help="view file or directory")
    p_view.add_argument("path")
    p_view.add_argument("--lines", help="line range START-END (1-indexed inclusive)")

    p_create = sub.add_parser("create", help="create or overwrite file")
    p_create.add_argument("path")
    p_create.add_argument("--text", required=True)

    p_sr = sub.add_parser("str-replace", help="exact-match unique str_replace")
    p_sr.add_argument("path"); p_sr.add_argument("--old", required=True); p_sr.add_argument("--new", required=True)

    p_ins = sub.add_parser("insert", help="insert text at line")
    p_ins.add_argument("path"); p_ins.add_argument("--line", type=int, required=True); p_ins.add_argument("--text", required=True)

    p_del = sub.add_parser("delete", help="archive + breadcrumb (NOT destroy)")
    p_del.add_argument("path")

    p_ren = sub.add_parser("rename", help="move within memory tree")
    p_ren.add_argument("src"); p_ren.add_argument("dst")

    p_srch = sub.add_parser("search", help="semantic search via lab_memory.py")
    p_srch.add_argument("query"); p_srch.add_argument("-k", type=int, default=5)

    p_snap = sub.add_parser("snapshot", help="defensive pre-edit copy")
    p_snap.add_argument("path")

    p_res = sub.add_parser("resolve", help="resolve a [log:...] / [episodic:...] / [session:...] reference to a path")
    p_res.add_argument("reference")

    sub.add_parser("index", help="regenerate INDEX.md")
    sub.add_parser("audit", help="report size-cap breaches")
    p_rot = sub.add_parser("rotate-log", help="rotate overflow entries to archive")
    p_rot.add_argument("--cap-kb", type=int, default=30)

    args = parser.parse_args()
    m = Memory(root=args.root)

    if args.cmd == "view":
        rng = None
        if args.lines:
            a, b = args.lines.split("-")
            rng = (int(a), int(b))
        result = m.view(args.path, view_range=rng)
        if isinstance(result, list):
            for child in result:
                print(child)
        else:
            print(result, end="")
    elif args.cmd == "create":
        m.create(args.path, args.text)
    elif args.cmd == "str-replace":
        m.str_replace(args.path, args.old, args.new)
    elif args.cmd == "insert":
        m.insert(args.path, args.line, args.text)
    elif args.cmd == "delete":
        archived = m.delete(args.path)
        print(f"archived: {archived}")
    elif args.cmd == "rename":
        m.rename(args.src, args.dst)
    elif args.cmd == "search":
        hits = m.search(args.query, k=args.k)
        print(json.dumps(hits, indent=2, default=str))
    elif args.cmd == "snapshot":
        snap = m.snapshot(args.path)
        print(f"snapshot: {snap}")
    elif args.cmd == "resolve":
        resolved = m.resolve(args.reference)
        print(resolved)
    elif args.cmd == "index":
        m.index()
    elif args.cmd == "audit":
        breaches = m.audit()
        if not breaches:
            print("OK: no breaches")
        else:
            for b in breaches:
                print(b)
            sys.exit(1)
    elif args.cmd == "rotate-log":
        arch = m.rotate_log(cap_kb=args.cap_kb)
        if arch is None:
            print("log under cap, no rotation needed")
        else:
            print(f"rotated to: {arch}")


if __name__ == "__main__":
    main()
