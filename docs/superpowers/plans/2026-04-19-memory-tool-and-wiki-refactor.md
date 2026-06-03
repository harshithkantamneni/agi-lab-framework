# Memory Tool + Wiki Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Director-level orchestration state out of unbounded `state.md` / `decisions_recent.md` / `director_log.md` into a capped three-tier model (hot / wiki / log) accessed through an Anthropic `memory_20250818`-compatible tool, delivering ≤10 KB Director session-startup reads and 5–7 sessions per compaction window.

**Architecture:** `tools/memory.py` implements Anthropic's memory protocol (view / create / str_replace / insert / delete / rename) with AGI extensions (search via `lab_memory.py`, snapshot) and a `delete`→archive override per Values §4. A one-time bootstrap script seeds `data/memories/` from existing files. The runner writes a `session_brief.md` before each Director launch. Evaluator gains 7 memory-discipline checks (total 18). Phase-gated migration executes after Program 2 Phase 2 close; reversible via pre-migration snapshot.

**Tech Stack:** Python 3 (stdlib), pytest, sqlite-vec + sentence-transformers (via existing `lab_memory.py`), bash (runner + Makefile), markdown (prompts, memories).

**Spec:** `docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md` (committed 0a199ba)

---

## Stages and gates

| Stage | Tasks | Gate to advance |
|---|---|---|
| **1. Build tool** | 1–11 | All unit tests pass; `tools/memory.py --help` shows protocol + extensions |
| **2. Bootstrap & prompts** | 12–17 | Bootstrap dry-run on fixture succeeds; prompt updates committed |
| **3. Runner & integration** | 18–24 | Runner writes brief in dry-run; dashboard/slack paths work against fixture |
| **4. Migration execution** | 25–29 | **GATE: Program 2 Phase 2 must close before starting.** Post-migration verification clean. |
| **5. Post-validation** | 30–31 | **GATE: 3 consecutive clean phases post-migration.** Compaction bump + cleanup. |

Stages 1–3 are pure build work, no live-lab state change — safe to execute any time. Stage 4 is the one-way cutover (reversible only via rollback protocol). Stage 5 is deferred until Evaluator has validated the refactor over 3 phases.

---

## File map

### Created by this plan

| Path | Purpose |
|---|---|
| `tools/memory.py` | Anthropic-protocol tool + AGI extensions |
| `tools/migrate_to_memories.py` | One-time bootstrap script (deleted after migration validates) |
| `tests/test_memory.py` | pytest suite for `tools/memory.py` |
| `tests/test_migrate_to_memories.py` | pytest suite for bootstrap |
| `data/memories/` | Three-tier memory root (all subfiles seeded by bootstrap) |
| `data/infra/memory_telemetry.log` | Per-session memory metrics (append-only) |

### Modified by this plan

| Path | Change |
|---|---|
| `data/agents/director/procedural.md` | New session-start sequence |
| `data/agents/pi/procedural.md` | Memory usage for directives |
| `data/agents/evaluator/procedural.md` | 7 new discipline checks (items 12–18) |
| `data/agents/findings_curator/procedural.md` | KM closeout responsibilities |
| `run_agi_lab.sh` | `write_session_brief` function + telemetry line logging |
| `CLAUDE.md` | Path references updated |
| `tools/dashboard.py`, `tools/dashboard.html` | Path updates + tier-size panel |
| `tools/slack_bot.py` | Path updates for `/status`, `/phase`, etc. |
| `tools/lab_memory.py` | `--incremental` mode, explicit indexing of `data/memories/` |
| `Makefile` | `memory-audit`, `memory-index`, `memory-rotate-log` targets |

### Archived (moved, not deleted; breadcrumbs left)

Executed at Task 26 during migration:
- `data/state.md` → `data/memories/current.md`
- `data/decisions_recent.md` + `data/director_log.md` → `data/memories/log.md` (merged, rotated)
- `data/pi_notes.md`, `data/values.md`, `data/shared_knowledge.md`, `data/killed_ideas.md`, `data/procedures.md`, `data/index.md`, `data/mission_reframe_2026-04-18.md`, `data/checkpoints/ARCHIVED.md`, `programs/portfolio.md` → `data/memories/...`

---

# Stage 1: Build tool

## Task 1: Memory skeleton with path rooting and exceptions

**Files:**
- Create: `tools/memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory.py
"""Tests for tools/memory.py — Anthropic-protocol memory tool.

Covers the six protocol commands (view/create/str_replace/insert/delete/rename)
plus AGI extensions (search/snapshot) and admin commands (index/audit/rotate-log).

See spec: docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.memory import Memory, PathOutsideRootError  # noqa: E402


def test_memory_init_creates_root_if_missing(tmp_path):
    root = tmp_path / "memories"
    assert not root.exists()
    m = Memory(root=root)
    assert root.exists()
    assert m.root == root.resolve()


def test_memory_rejects_path_outside_root(tmp_path):
    m = Memory(root=tmp_path / "memories")
    with pytest.raises(PathOutsideRootError):
        m._resolve("../escape.md")
    with pytest.raises(PathOutsideRootError):
        m._resolve("/absolute/path.md")


def test_memory_resolves_relative_path(tmp_path):
    m = Memory(root=tmp_path / "memories")
    resolved = m._resolve("subdir/file.md")
    assert resolved == (tmp_path / "memories" / "subdir" / "file.md").resolve()
```

- [ ] **Step 2: Run test to verify it fails**

```
.venv/bin/pytest tests/test_memory.py -v
```

Expected: `ImportError: cannot import name 'Memory' from 'tools.memory'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/memory.py
#!/usr/bin/env python3
"""tools/memory.py -- Anthropic memory_20250818-protocol tool for AGI lab.

Implements the six protocol commands (view/create/str_replace/insert/delete/rename)
plus AGI extensions (search/snapshot) and admin commands (index/audit/rotate-log).

Spec: docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md

Deviation from Anthropic protocol: `delete` archives + breadcrumbs rather than
destroying. Values §4 archive-never-delete.
"""
from __future__ import annotations

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


class Memory:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        if Path(path).is_absolute():
            raise PathOutsideRootError(f"absolute path rejected: {path}")
        candidate = (self.root / path).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PathOutsideRootError(f"path escapes memory root: {path}")
        return candidate
```

- [ ] **Step 4: Run test to verify it passes**

```
.venv/bin/pytest tests/test_memory.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "$(cat <<'EOF'
feat(memory): Memory skeleton with path rooting

Add tools/memory.py with path-sandboxed resolver and exception types.
First piece of Anthropic memory_20250818 protocol implementation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: view command (file, range, directory listing)

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_memory.py`:

```python
def test_view_reads_full_file(tmp_path):
    m = Memory(root=tmp_path / "m")
    (m.root / "note.md").write_text("line1\nline2\nline3\n")
    assert m.view("note.md") == "line1\nline2\nline3\n"


def test_view_with_line_range(tmp_path):
    m = Memory(root=tmp_path / "m")
    (m.root / "note.md").write_text("a\nb\nc\nd\ne\n")
    assert m.view("note.md", view_range=(2, 4)) == "b\nc\nd\n"


def test_view_directory_lists_children(tmp_path):
    m = Memory(root=tmp_path / "m")
    (m.root / "a.md").write_text("x")
    (m.root / "sub").mkdir()
    (m.root / "sub" / "b.md").write_text("y")
    listing = m.view("")
    assert sorted(listing) == ["a.md", "sub/"]


def test_view_missing_file_raises(tmp_path):
    from tools.memory import MemoryFileNotFoundError
    m = Memory(root=tmp_path / "m")
    with pytest.raises(MemoryFileNotFoundError):
        m.view("nope.md")
```

- [ ] **Step 2: Run to verify failure**

```
.venv/bin/pytest tests/test_memory.py -v
```

Expected: 4 new tests fail (`AttributeError: 'Memory' object has no attribute 'view'`).

- [ ] **Step 3: Implement view**

Add to `tools/memory.py` in the `Memory` class:

```python
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
```

- [ ] **Step 4: Verify tests pass**

```
.venv/bin/pytest tests/test_memory.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): view command (file/range/dir listing)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: create command (new file and overwrite, atomic write)

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_create_writes_new_file(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("hello.md", "world")
    assert (m.root / "hello.md").read_text() == "world"


def test_create_overwrites_existing(tmp_path):
    m = Memory(root=tmp_path / "m")
    (m.root / "x.md").write_text("old")
    m.create("x.md", "new")
    assert (m.root / "x.md").read_text() == "new"


def test_create_makes_parent_directories(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("deep/nested/file.md", "ok")
    assert (m.root / "deep" / "nested" / "file.md").read_text() == "ok"


def test_create_is_atomic(tmp_path, monkeypatch):
    """If write fails mid-way, original file must be untouched."""
    m = Memory(root=tmp_path / "m")
    (m.root / "critical.md").write_text("original")

    def boom(*a, **kw):
        raise IOError("disk full")

    monkeypatch.setattr("pathlib.Path.rename", boom)
    with pytest.raises(IOError):
        m.create("critical.md", "replacement")
    assert (m.root / "critical.md").read_text() == "original"
```

- [ ] **Step 2: Verify failure**

```
.venv/bin/pytest tests/test_memory.py -v -k create
```

Expected: 4 fail.

- [ ] **Step 3: Implement create with atomic write**

```python
    def create(self, path: str, file_text: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(file_text)
        tmp.rename(target)
```

- [ ] **Step 4: Verify pass**

```
.venv/bin/pytest tests/test_memory.py -v -k create
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): create command with atomic write

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: str_replace with uniqueness check

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_str_replace_happy_path(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("f.md", "hello world\n")
    m.str_replace("f.md", "world", "earth")
    assert m.view("f.md") == "hello earth\n"


def test_str_replace_nonunique_raises(tmp_path):
    from tools.memory import NonUniqueMatchError
    m = Memory(root=tmp_path / "m")
    m.create("f.md", "aaa bbb aaa\n")
    with pytest.raises(NonUniqueMatchError):
        m.str_replace("f.md", "aaa", "ZZZ")


def test_str_replace_nomatch_raises(tmp_path):
    from tools.memory import NoMatchError
    m = Memory(root=tmp_path / "m")
    m.create("f.md", "hello\n")
    with pytest.raises(NoMatchError):
        m.str_replace("f.md", "missing", "anything")


def test_str_replace_preserves_rest_of_file(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("f.md", "line1\nline2\nline3\n")
    m.str_replace("f.md", "line2", "NEW")
    assert m.view("f.md") == "line1\nNEW\nline3\n"
```

- [ ] **Step 2: Verify failure**

```
.venv/bin/pytest tests/test_memory.py -v -k str_replace
```

Expected: 4 fail.

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Verify pass**

```
.venv/bin/pytest tests/test_memory.py -v -k str_replace
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): str_replace with uniqueness enforcement

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: insert command (line-based, 0 = prepend)

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_insert_middle_of_file(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("f.md", "line1\nline3\n")
    m.insert("f.md", insert_line=1, insert_text="line2\n")
    assert m.view("f.md") == "line1\nline2\nline3\n"


def test_insert_at_start_line_zero(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("f.md", "old_first\n")
    m.insert("f.md", insert_line=0, insert_text="new_first\n")
    assert m.view("f.md") == "new_first\nold_first\n"


def test_insert_at_end(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("f.md", "a\nb\n")
    m.insert("f.md", insert_line=2, insert_text="c\n")
    assert m.view("f.md") == "a\nb\nc\n"
```

- [ ] **Step 2: Verify failure**

```
.venv/bin/pytest tests/test_memory.py -v -k insert
```

- [ ] **Step 3: Implement**

```python
    def insert(self, path: str, insert_line: int, insert_text: str) -> None:
        target = self._resolve(path)
        if not target.exists():
            raise MemoryFileNotFoundError(str(target.relative_to(self.root)))
        lines = target.read_text().splitlines(keepends=True)
        if insert_line < 0 or insert_line > len(lines):
            raise ValueError(f"insert_line {insert_line} out of range")
        new_lines = lines[:insert_line] + [insert_text] + lines[insert_line:]
        self.create(path, "".join(new_lines))
```

- [ ] **Step 4: Verify pass**

```
.venv/bin/pytest tests/test_memory.py -v -k insert
```

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): insert command

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: rename command (moves within memory tree only)

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_rename_moves_file(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("a.md", "content")
    m.rename("a.md", "b.md")
    assert not (m.root / "a.md").exists()
    assert m.view("b.md") == "content"


def test_rename_into_subdirectory(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("a.md", "c")
    m.rename("a.md", "sub/a.md")
    assert m.view("sub/a.md") == "c"


def test_rename_destination_outside_root_rejected(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("a.md", "c")
    with pytest.raises(PathOutsideRootError):
        m.rename("a.md", "../escaped.md")
```

- [ ] **Step 2: Verify failure**

```
.venv/bin/pytest tests/test_memory.py -v -k rename
```

- [ ] **Step 3: Implement**

```python
    def rename(self, old_path: str, new_path: str) -> None:
        src = self._resolve(old_path)
        dst = self._resolve(new_path)
        if not src.exists():
            raise MemoryFileNotFoundError(str(src.relative_to(self.root)))
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
```

- [ ] **Step 4: Verify pass**

```
.venv/bin/pytest tests/test_memory.py -v -k rename
```

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): rename command

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: delete override (archive + breadcrumb, not destroy)

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_delete_moves_to_archive_not_destroys(tmp_path):
    m = Memory(root=tmp_path / "m")
    archive = tmp_path / "archives"
    m = Memory(root=tmp_path / "m", archive_root=archive)
    m.create("doomed.md", "content I care about")
    archive_path = m.delete("doomed.md")
    # Original is gone, but a breadcrumb sits at the path
    assert (m.root / "doomed.md").exists()  # breadcrumb
    assert "archived" in (m.root / "doomed.md").read_text()
    # Archive has full content
    assert archive_path.read_text() == "content I care about"


def test_delete_breadcrumb_points_to_archive(tmp_path):
    archive = tmp_path / "archives"
    m = Memory(root=tmp_path / "m", archive_root=archive)
    m.create("x.md", "hi")
    archive_path = m.delete("x.md")
    breadcrumb = (m.root / "x.md").read_text()
    assert str(archive_path) in breadcrumb or archive_path.name in breadcrumb


def test_delete_missing_file_raises(tmp_path):
    archive = tmp_path / "archives"
    m = Memory(root=tmp_path / "m", archive_root=archive)
    with pytest.raises(MemoryFileNotFoundError):
        m.delete("nope.md")
```

- [ ] **Step 2: Verify failure**

```
.venv/bin/pytest tests/test_memory.py -v -k delete
```

- [ ] **Step 3: Implement**

Update `__init__` and add `delete`:

```python
    def __init__(self, root: Path | str, archive_root: Path | str | None = None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if archive_root is None:
            # Default: data/archives/deletes/ sibling of root.parent
            archive_root = self.root.parent / "archives" / "deletes"
        self.archive_root = Path(archive_root).resolve()

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
```

- [ ] **Step 4: Verify pass**

```
.venv/bin/pytest tests/test_memory.py -v -k delete
```

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): delete override — archive + breadcrumb per Values §4

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: snapshot extension (defensive pre-edit copy)

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_snapshot_preserves_original(tmp_path):
    snap_root = tmp_path / "snaps"
    m = Memory(root=tmp_path / "m", snapshot_root=snap_root)
    m.create("important.md", "before-edit")
    snap_path = m.snapshot("important.md")
    # Original still there
    assert m.view("important.md") == "before-edit"
    # Snapshot has the content
    assert snap_path.read_text() == "before-edit"
    # Snapshot path is under snap_root
    assert snap_root in snap_path.parents


def test_snapshot_multiple_calls_stamped(tmp_path):
    import time
    snap_root = tmp_path / "snaps"
    m = Memory(root=tmp_path / "m", snapshot_root=snap_root)
    m.create("f.md", "v1")
    s1 = m.snapshot("f.md")
    time.sleep(0.01)
    m.create("f.md", "v2")
    s2 = m.snapshot("f.md")
    assert s1 != s2
    assert s1.read_text() == "v1"
    assert s2.read_text() == "v2"
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement**

Update `__init__` to accept `snapshot_root`, add method:

```python
    def __init__(self, root, archive_root=None, snapshot_root=None):
        # ... existing code ...
        if snapshot_root is None:
            snapshot_root = self.root.parent / "archives" / "snapshots"
        self.snapshot_root = Path(snapshot_root).resolve()

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
```

- [ ] **Step 4: Verify pass**

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): snapshot extension for defensive pre-edit copies

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: search + resolve extensions

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing search test**

Note on types: `tools.lab_memory.LabMemory.search` returns `list[Hit]` where `Hit` is a dataclass with fields `source_path`, `chunk_text`, `distance` (see `tools/lab_memory.py:82`). Our wrapper passes through.

```python
def test_search_delegates_to_lab_memory(tmp_path, monkeypatch):
    m = Memory(root=tmp_path / "m")

    called = {}

    class FakeHit:
        def __init__(self, source_path, chunk_text, distance):
            self.source_path = source_path
            self.chunk_text = chunk_text
            self.distance = distance

    class FakeLabMemory:
        def __init__(self, db_path):
            called["init"] = db_path
        def search(self, query, top_k=10, **_kw):
            called["query"] = query
            called["top_k"] = top_k
            return [FakeHit("data/memories/doc.md", "router equilibrium...", 0.09)]

    monkeypatch.setattr("tools.memory._get_lab_memory", lambda: FakeLabMemory("x"))
    results = m.search("router equilibrium", k=3)
    assert called["query"] == "router equilibrium"
    assert called["top_k"] == 3
    assert len(results) == 1
    assert results[0].source_path == "data/memories/doc.md"
```

- [ ] **Step 2: Verify search test fails**

```
.venv/bin/pytest tests/test_memory.py -v -k search
```

- [ ] **Step 3: Implement search**

```python
def _get_lab_memory():
    from tools.lab_memory import LabMemory
    lm = LabMemory("tools/lab_memory.db")
    return lm


class Memory:
    # ... existing ...

    def search(self, query: str, k: int = 5):
        """Semantic search via lab_memory.py. Returns list[lab_memory.Hit]."""
        lm = _get_lab_memory()
        return lm.search(query, top_k=k)
```

- [ ] **Step 4: Verify search test passes**

- [ ] **Step 5: Write the failing resolve tests**

Reference format per spec §9:
- `[log:YYYY-MM-DD#D-NNN]` → live `log.md` if entry exists, else `data/archives/<date>/log.md`
- `[session:timestamp]` → session_brief archive or `history.md` entry
- `[episodic:<role>/<file>]` → `data/agents/<role>/episodic/<file>.md`

```python
def test_resolve_episodic_reference(tmp_path):
    m = Memory(root=tmp_path / "m")
    # Create a fake episodic record in the repo root (memory root's parent.parent)
    agents_dir = tmp_path / "data" / "agents" / "pi" / "episodic"
    agents_dir.mkdir(parents=True)
    target = agents_dir / "2026-04-18_program_2_phase_1.md"
    target.write_text("x")
    m.repo_root = tmp_path  # override for test
    resolved = m.resolve("[episodic:pi/2026-04-18_program_2_phase_1]")
    assert resolved == target


def test_resolve_log_reference_live(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("log.md", "## D-117 header\nbody\n## D-116 earlier\nbody\n")
    resolved = m.resolve("[log:2026-04-19#D-117]")
    assert resolved == (m.root / "log.md").resolve()


def test_resolve_log_reference_archived(tmp_path):
    archive_root = tmp_path / "data" / "archives"
    m = Memory(root=tmp_path / "m", archive_root=archive_root / "deletes")
    m.repo_root = tmp_path
    arch_log = archive_root / "2026-03-01" / "log.md"
    arch_log.parent.mkdir(parents=True)
    arch_log.write_text("## D-099 old\nbody\n")
    # D-099 is not in live log; resolver should find it in archive
    m.create("log.md", "## D-117 header\nbody\n")
    resolved = m.resolve("[log:2026-03-01#D-099]")
    assert resolved == arch_log.resolve()


def test_resolve_unknown_reference_raises(tmp_path):
    from tools.memory import UnresolvedReferenceError
    m = Memory(root=tmp_path / "m")
    with pytest.raises(UnresolvedReferenceError):
        m.resolve("[log:2099-01-01#D-999]")
```

- [ ] **Step 6: Verify resolve tests fail**

- [ ] **Step 7: Implement resolve**

Add to the exceptions block at the top:

```python
class UnresolvedReferenceError(MemoryError):
    """Reference tag does not resolve to an existing path."""
```

Add to `Memory`:

```python
    def __init__(self, root, archive_root=None, snapshot_root=None):
        # ... existing ...
        # repo_root defaults to memory root's parent.parent (data/memories → repo root)
        self.repo_root = self.root.parent.parent

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
            if candidate.exists():
                return candidate.resolve()
            raise UnresolvedReferenceError(f"episodic not found: {candidate}")
        if tag == "log":
            # [log:YYYY-MM-DD#D-NNN] — try live log first, then archives
            if "#" in body:
                date_part, anchor = body.split("#", 1)
            else:
                date_part, anchor = body, None
            live = self.root / "log.md"
            if live.exists() and anchor and anchor in live.read_text():
                return live.resolve()
            archive = self.repo_root / "data" / "archives" / date_part / "log.md"
            if archive.exists():
                if anchor is None or anchor in archive.read_text():
                    return archive.resolve()
            raise UnresolvedReferenceError(f"log reference unresolved: {reference}")
        if tag == "session":
            hist = self.root / "history.md"
            if hist.exists() and body in hist.read_text():
                return hist.resolve()
            raise UnresolvedReferenceError(f"session reference unresolved: {reference}")
        raise UnresolvedReferenceError(f"unknown reference tag: {tag}")
```

- [ ] **Step 8: Verify resolve tests pass**

```
.venv/bin/pytest tests/test_memory.py -v -k "search or resolve"
```

- [ ] **Step 9: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): search + resolve extensions

resolve handles [log:date#D-NNN], [episodic:role/file], [session:ts]
reference formats per spec §9.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 10: Admin commands (index, audit, rotate-log)

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_audit_reports_hot_overflow(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("current.md", "x" * (41 * 1024))  # 41 KB, cap 40
    breaches = m.audit()
    assert any("current.md" in b and "hot" in b.lower() for b in breaches)


def test_audit_reports_log_overflow(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("log.md", "y" * (31 * 1024))
    breaches = m.audit()
    assert any("log.md" in b and "log" in b.lower() for b in breaches)


def test_audit_clean_state_returns_empty(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("current.md", "small")
    m.create("log.md", "small")
    assert m.audit() == []


def test_index_regenerates_INDEX_md(tmp_path):
    m = Memory(root=tmp_path / "m")
    m.create("mission.md", "Beat Opus 4.7")
    m.create("governance/values.md", "10 values")
    m.index()
    content = (m.root / "INDEX.md").read_text()
    assert "mission.md" in content
    assert "governance/values.md" in content


def test_rotate_log_moves_oldest_to_archive(tmp_path):
    archive_root = tmp_path / "archives"
    m = Memory(root=tmp_path / "m", archive_root=archive_root)
    # 40 KB log with dated entries at top (newest) and bottom (oldest)
    entries = []
    for i in range(500):
        entries.append(f"## 2026-04-{i % 30 + 1:02d} entry {i}\n" + "x" * 60 + "\n")
    m.create("log.md", "".join(entries))
    archive_path = m.rotate_log(cap_kb=30)
    # Live log under cap
    live_size = (m.root / "log.md").stat().st_size
    assert live_size <= 30 * 1024
    # Archive contains the overflow
    assert archive_path is not None
    assert archive_path.exists()
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement**

```python
    # Tier caps in bytes
    CAP_HOT = 40 * 1024
    CAP_WIKI_TOTAL = 50 * 1024
    CAP_LOG = 30 * 1024
    WIKI_PER_FILE_SOFT = 15 * 1024

    # Tier membership by filename (first-tier is the convention)
    HOT_FILES = {"current.md"}
    LOG_FILES = {"log.md"}

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
            # Grab first non-empty line as description
            desc = ""
            for line in f.read_text().splitlines():
                if line.strip() and not line.startswith("---"):
                    desc = line.lstrip("# ").strip()[:80]
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
```

- [ ] **Step 4: Verify pass**

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): admin commands — index, audit, rotate-log

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: CLI wrapper with argparse + --help

**Files:**
- Modify: `tools/memory.py`
- Modify: `tests/test_memory.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cli_help_documents_all_commands(tmp_path, capsys):
    import subprocess
    result = subprocess.run(
        [sys.executable, "tools/memory.py", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    for cmd in ["view", "create", "str-replace", "insert", "delete", "rename",
                "search", "snapshot", "resolve", "index", "audit", "rotate-log"]:
        assert cmd in result.stdout, f"{cmd} missing from --help"


def test_cli_view_reads_file(tmp_path):
    import subprocess
    # Create a memory tree via library, view via CLI
    m = Memory(root=tmp_path / "m")
    m.create("file.md", "hello cli")
    result = subprocess.run(
        [sys.executable, "tools/memory.py", "--root", str(m.root), "view", "file.md"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "hello cli" in result.stdout
```

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement**

Append to `tools/memory.py`:

```python
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
```

Add `import sys` at top if missing.

- [ ] **Step 4: Verify pass**

```
.venv/bin/pytest tests/test_memory.py -v
```

All Task 1–11 tests pass.

```bash
chmod +x tools/memory.py
python3 tools/memory.py --help
```

- [ ] **Step 5: Commit**

```bash
git add tools/memory.py tests/test_memory.py
git commit -m "feat(memory): CLI wrapper with full subcommand surface

Closes Stage 1 (tool build). tools/memory.py now implements full Anthropic
memory_20250818 protocol plus search/snapshot extensions and admin commands.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**STAGE 1 GATE:** Before advancing, verify: `.venv/bin/pytest tests/test_memory.py -v` shows all tests pass, and `python3 tools/memory.py --help` documents all 11 subcommands.

---

# Stage 2: Bootstrap & prompts

## Task 12: Migration script skeleton + fixture test

**Files:**
- Create: `tools/migrate_to_memories.py`
- Create: `tests/test_migrate_to_memories.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_migrate_to_memories.py
"""Tests for one-time bootstrap of data/memories/ from legacy paths."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_fixture_repo(tmp_path: Path) -> Path:
    """Build a minimal fake AGI repo fixture."""
    repo = tmp_path / "fake_repo"
    (repo / "data").mkdir(parents=True)
    (repo / "data" / "state.md").write_text("# state\nactive: prog_2\n")
    (repo / "data" / "pi_notes.md").write_text("# PI notes\n")
    (repo / "data" / "values.md").write_text("# Values\n")
    (repo / "data" / "shared_knowledge.md").write_text("# Shared\n")
    (repo / "data" / "killed_ideas.md").write_text("# Killed\n")
    (repo / "data" / "procedures.md").write_text("# Procedures\n")
    (repo / "data" / "index.md").write_text("# Index\n")
    (repo / "data" / "director_log.md").write_text(
        "## 2026-04-19 entry A\nalpha\n\n## 2026-04-18 entry B\nbeta\n"
    )
    (repo / "data" / "decisions_recent.md").write_text(
        "## D-116 phase close\ndetails\n\n## D-115 something\ndetails\n"
    )
    (repo / "data" / "mission_reframe_2026-04-18.md").write_text("# Reframe\n")
    (repo / "data" / "checkpoints").mkdir()
    (repo / "data" / "checkpoints" / "ARCHIVED.md").write_text("# archived\n")
    (repo / "programs").mkdir()
    (repo / "programs" / "portfolio.md").write_text("# portfolio\n")
    (repo / "CLAUDE.md").write_text("# AGI\n## Mission\ntext\n")
    return repo


def test_migration_creates_all_tier_files(tmp_path):
    from tools.migrate_to_memories import migrate
    repo = _make_fixture_repo(tmp_path)
    migrate(repo_root=repo)
    mem = repo / "data" / "memories"
    assert (mem / "current.md").exists()
    assert (mem / "log.md").exists()
    assert (mem / "governance" / "pi_notes.md").exists()
    assert (mem / "governance" / "values.md").exists()
    assert (mem / "procedures.md").exists()
    assert (mem / "shared.md").exists()
    assert (mem / "killed.md").exists()
    assert (mem / "mission.md").exists()
    assert (mem / "programs" / "portfolio.md").exists()
    assert (mem / "INDEX.md").exists()


def test_migration_merges_log_sources(tmp_path):
    from tools.migrate_to_memories import migrate
    repo = _make_fixture_repo(tmp_path)
    migrate(repo_root=repo)
    log = (repo / "data" / "memories" / "log.md").read_text()
    assert "D-116" in log
    assert "entry A" in log or "entry B" in log  # at least one merged in


def test_migration_passes_audit(tmp_path):
    from tools.migrate_to_memories import migrate
    from tools.memory import Memory
    repo = _make_fixture_repo(tmp_path)
    migrate(repo_root=repo)
    m = Memory(root=repo / "data" / "memories")
    breaches = m.audit()
    assert breaches == [], f"Post-migration audit failed: {breaches}"


def test_migration_snapshots_originals_before_moving(tmp_path):
    from tools.migrate_to_memories import migrate
    repo = _make_fixture_repo(tmp_path)
    migrate(repo_root=repo)
    snap = repo / "data" / "archives" / "snapshots" / "pre-memory-refactor"
    assert snap.exists()
    assert (snap / "state.md").exists()
    assert (snap / "pi_notes.md").exists()
    assert (snap / "decisions_recent.md").exists()
```

- [ ] **Step 2: Verify failure**

```
.venv/bin/pytest tests/test_migrate_to_memories.py -v
```

- [ ] **Step 3: Implement skeleton**

```python
# tools/migrate_to_memories.py
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


def migrate(repo_root: Path) -> None:
    repo_root = Path(repo_root).resolve()
    mem_root = repo_root / "data" / "memories"
    arch_root = repo_root / "data" / "archives" / "memories-deletes"
    snap_root = repo_root / "data" / "archives" / "memories-snapshots"

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
    args = p.parse_args()
    if args.dry_run:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp) / "repo"
            shutil.copytree(args.repo_root, tmp_root, symlinks=True,
                            ignore=shutil.ignore_patterns(".git", ".venv",
                                                          "__pycache__", "*.db"))
            migrate(tmp_root)
            print(f"Dry-run complete. Inspect: {tmp_root / 'data' / 'memories'}")
    else:
        migrate(Path(args.repo_root))
        print("Migration complete.")


if __name__ == "__main__":
    _cli()
```

- [ ] **Step 4: Verify pass**

```
.venv/bin/pytest tests/test_migrate_to_memories.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tools/migrate_to_memories.py tests/test_migrate_to_memories.py
git commit -m "feat(memory): bootstrap script for one-time migration

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Director procedural update — new session-start sequence

**Files:**
- Modify: `data/agents/director/procedural.md`

- [ ] **Step 1: Read current procedural**

```bash
wc -l data/agents/director/procedural.md
head -40 data/agents/director/procedural.md
```

- [ ] **Step 2: Write the new session-start preamble**

Prepend this section to `data/agents/director/procedural.md` (before existing content):

```markdown
# Director procedural — session start sequence (memory-tool era)

## First three tool calls, always

1. `memory.view session_brief.md`
   - If `generated_at` header timestamp is older than 10 minutes, treat brief as stale and jump to step 2 directly (skip using brief content).
2. `memory.view INDEX.md`
3. `memory.view <file>` for each file the brief or INDEX says is relevant to the next step (typically `current.md` + one more).

Target cumulative read: ≤ 10 KB. If exceeding, you are reading too much.

## Edits during session

- Phase transitions → `memory.str_replace current.md` on the state line. NEVER rewrite `current.md` wholesale.
- New decisions → `memory.insert log.md --line 0 --text "<decision block>"`. Prepend, newest-first.
- Wiki updates (rare) → `memory.str_replace` on the specific wiki file. `memory.snapshot <file>` before any edit touching more than 2 KB.
- Archival → `memory.delete <path>`. Values §4: this archives + breadcrumbs, it does NOT destroy.

## End-of-phase (not end-of-session)

Dispatch `findings_curator` with `KM-Closeout` instruction. KM runs the 6 closeout tasks (see `data/agents/findings_curator/procedural.md §KM-Closeout`) before Evaluator's phase gate.

## Do NOT

- Do NOT `memory.view data/state.md` or any legacy path — those are moved, breadcrumbed to new locations.
- Do NOT pass `memories/` content to subagents by directory reference. Dispatch with curated file excerpts in the prompt (unchanged pattern from prior sessions).
- Do NOT full-file rewrite any tier file. Evaluator check #16 watches git diff for this; a >80% line change on a tier file in one session blocks phase close.

## If the brief or INDEX is missing

Fall back: `memory.view` individually on `current.md`, `mission.md`, `governance/pi_notes.md`, `log.md` (first 20 lines). File an ORG_ADAPTATION note noting the runner failed to write session_brief.md.

---

(Remaining procedural content below — unchanged from prior version except where explicitly noted.)

```

- [ ] **Step 3: Preserve existing procedural content below the new preamble**

Open `data/agents/director/procedural.md`, prepend the block above, keep the rest.

- [ ] **Step 4: Verify file is well-formed**

```bash
head -60 data/agents/director/procedural.md
wc -l data/agents/director/procedural.md
```

Expected: new preamble is visible at top; file length increased by ~40 lines.

- [ ] **Step 5: Commit**

```bash
git add data/agents/director/procedural.md
git commit -m "feat(director): memory-tool session-start sequence

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: PI procedural update — directive writes via memory.insert

**Files:**
- Modify: `data/agents/pi/procedural.md`

- [ ] **Step 1: Write the addition**

Append this section to `data/agents/pi/procedural.md`:

```markdown

## Memory tool usage (memory-tool era, post-D-117)

Your directives live in `data/memories/governance/pi_notes.md`. Post-migration:

- Add a directive → `memory.insert governance/pi_notes.md --line 0 --text "<directive block>"`
  (newest-first; never rewrite the file)
- Reference an archived directive → use `[episodic:pi/<file>]` format (resolvable by `memory.py resolve`).
- To co-sign a Director decision → dispatch creates an episodic record at `data/agents/pi/episodic/<date>_<topic>.md` as before. That directory is NOT part of the memory tool surface — it is your private CoALA episodic memory.

When reviewing a Director's work:
- Read `memory.view session_brief.md` and `memory.view current.md` first.
- Read `memory.view log.md --lines 1-40` for recent decisions.
- Only then read specific program artifacts referenced in the brief.
```

- [ ] **Step 2: Verify**

```bash
tail -30 data/agents/pi/procedural.md
```

- [ ] **Step 3: Commit**

```bash
git add data/agents/pi/procedural.md
git commit -m "feat(pi): memory-tool directive writes via insert

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Evaluator procedural — add 7 memory discipline checks

**Files:**
- Modify: `data/agents/evaluator/procedural.md`

- [ ] **Step 1: Append memory_discipline section**

Append to `data/agents/evaluator/procedural.md`:

```markdown

## §memory_discipline (checks 12–18, post-D-117 migration)

Checklist extends from 11 to 18 items. In every evaluator report, include a
`§memory_discipline` block with PASS/FAIL for each:

| # | Check | Fail condition | Fix path |
|---|---|---|---|
| 12 | Hot tier size | `memory.audit` reports HOT breach | Dispatch `findings_curator` for KM closeout |
| 13 | Wiki tier total | `memory.audit` reports WIKI-TOTAL breach | KM splits largest wiki file |
| 14 | Log tier size | `memory.audit` reports LOG breach | KM runs `memory.py rotate-log` |
| 15 | INDEX + reference integrity | Any `INDEX.md` pointer resolves to missing file, OR any `memories/*.md` not reachable from INDEX, OR any `[log:...]` / `[episodic:...]` / `[session:...]` reference inside a wiki file fails `memory.py resolve` | `memory.py index` regenerates; broken refs investigated individually |
| 16 | No full-file rewrites | `git diff` on any tier file in last session shows >80% line change with no rename | Investigate; revert if accidental; use `str_replace`/`insert` instead |
| 17 | KM closeout ran | Any tier ≥90% cap at phase close AND no `data/agents/findings_curator/episodic/<date>_KM-Closeout_*.md` for this phase | Block phase close until KM dispatched |
| 18 | Session brief consumed | Director's first session tool call was NOT `memory.view session_brief.md` or `memory.view INDEX.md` (check via Director session log) | Note in report; 3 consecutive violations trigger ORG_ADAPTATION |

### How to run the checks

```bash
python3 tools/memory.py audit              # covers 12, 13, 14
python3 tools/memory.py index              # regenerates INDEX; diff vs prior run reveals orphans/missing (check 15)
grep -oE '\[(log|episodic|session):[^]]+\]' data/memories/**/*.md | \
    while IFS=: read -r _ ref; do \
        python3 tools/memory.py resolve "[${ref}" >/dev/null 2>&1 || echo "BROKEN: [${ref}"; \
    done                                    # covers 15 (reference integrity)
git log -1 --name-status data/memories/    # covers 16 (spot full-file writes)
ls data/agents/findings_curator/episodic/ | grep "$(date +%Y-%m)"  # covers 17
grep -m1 'memory.view' data/agents/director/episodic/<latest>.md   # covers 18
```

### Report format

Append to existing evaluator report template:

```markdown
## §memory_discipline
- [ ] 12 Hot tier size: PASS / FAIL (<current.md size>)
- [ ] 13 Wiki total: PASS / FAIL (<total>)
- [ ] 14 Log tier size: PASS / FAIL (<log size>)
- [ ] 15 INDEX integrity: PASS / FAIL (<broken pointers or orphans>)
- [ ] 16 No full-file rewrites: PASS / FAIL (<files flagged>)
- [ ] 17 KM closeout ran: PASS / FAIL (<reason>)
- [ ] 18 Session brief consumed: PASS / FAIL (<first tool call>)
```

Failures in 12–14 must be addressed before phase close. Failures in 15–18 are logged but phase may close; 3 consecutive violations trigger ORG_ADAPTATION.
```

- [ ] **Step 2: Verify**

```bash
grep -c "memory_discipline" data/agents/evaluator/procedural.md
```

Expected: ≥1.

- [ ] **Step 3: Commit**

```bash
git add data/agents/evaluator/procedural.md
git commit -m "feat(evaluator): 7 new memory-discipline checks (12–18)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: findings_curator procedural — KM-Closeout section

**Files:**
- Modify: `data/agents/findings_curator/procedural.md`

- [ ] **Step 1: Append §KM-Closeout**

Append to `data/agents/findings_curator/procedural.md`:

```markdown

## §KM-Closeout (post-D-117 migration)

Invoked by the Director at every phase close, BEFORE Evaluator's phase gate.

### Six tasks, in order

1. **Findings inbox.** Read `current.md` and the last 40 lines of `log.md`. Identify durable facts (principles, killed approaches, roster changes, resolved governance questions). Promote them to the appropriate wiki file via `memory.insert`. Do NOT promote ephemeral work-in-progress details.

2. **INDEX maintenance.** Run `python3 tools/memory.py index`. Then manually spot-check: every file in `memories/` should appear once; no file should link to a missing target.

3. **Cap enforcement.** Run `python3 tools/memory.py audit`.
   - Hot >36 KB (90% of cap): compress `current.md` — move completed phase items to `log.md` via `memory.insert`, compress verbose sections in `current.md` via `memory.str_replace`.
   - Wiki total >45 KB (90% of cap): split the largest wiki file along topic boundaries (e.g., `shared.md` → `shared/architecture.md`, `shared/data.md`, etc.). Update INDEX afterward.
   - Wiki per-file >15 KB (soft target): inspect for natural topic split. If a clean boundary exists, split. If the file is one cohesive topic (e.g., `governance/values.md` describing 10 values as a unit), leave it and note reasoning in the closeout record.
   - Log >27 KB (90% of cap): `python3 tools/memory.py rotate-log --cap-kb 30`. Verify `history.md` got its breadcrumb.

4. **Semantic reindex.** `python3 tools/lab_memory.py ingest --incremental`. Limits reindexing to files changed this phase.

5. **Breadcrumb audit.** For every archive move in this phase (visible via `git log --diff-filter=R --name-status data/memories/`), confirm a breadcrumb exists at the old path.

6. **Telemetry line.** Append to `data/infra/memory_telemetry.log`:

   ```
   <ISO8601> | startup_kb=<last-session> | brief_age_s=<last-session> | eval_checks_passed=<11-to-18> | log_kb=<new> | wiki_kb=<total> | hot_kb=<current> | km_closeout=<none|phase_close_only|cap_triggered>
   ```

### Episodic record

Write the closeout report to `data/agents/findings_curator/episodic/<YYYY-MM-DD>_KM-Closeout_<phase>.md` with:
- Tasks 1–6 completion status
- Any files split
- Any files rotated
- Breaches found and fixed
- Breaches unfixable (escalation to Evaluator)

### Failure modes

- If cap cannot be brought under 90% (e.g., wiki total 48 KB and no file splits cleanly): flag in episodic record, do NOT block phase close, report to Evaluator check #17. Lab_architect addresses structurally at next retro.
- If `lab_memory.py ingest` fails: continue (non-blocking); flag in closeout record.
```

- [ ] **Step 2: Verify**

```bash
grep -c "KM-Closeout" data/agents/findings_curator/procedural.md
```

- [ ] **Step 3: Commit**

```bash
git add data/agents/findings_curator/procedural.md
git commit -m "feat(findings_curator): KM-Closeout section (6 tasks)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: PI directive D-117 drafted and queued in pi_notes.md

**Files:**
- Modify: `data/pi_notes.md` (still at legacy path until migration — this directive rides legacy path, becomes `governance/pi_notes.md` at migration)

- [ ] **Step 1: Prepend DIRECTIVE D-117**

Add near the top of `data/pi_notes.md`, just after the DIRECTIVE D-114 block:

```markdown

## DIRECTIVE D-117 — Memory discipline is load-bearing (2026-04-19)

Effective at migration cutover (Program 2 Phase 2 close → Phase 3 open).

1. All Director-level orchestration state lives in `data/memories/` via `tools/memory.py`. Legacy `data/state.md`, `data/decisions_recent.md`, `data/director_log.md`, and others are archived with breadcrumbs (see `docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md` §10).

2. The Director's session-start sequence is: `memory.view session_brief.md` → `memory.view INDEX.md` → selective `memory.view` on indicated files. Target cumulative read: ≤10 KB.

3. Tier caps (hard unless noted):
   - Hot (`current.md`): 40 KB
   - Wiki (total across governance/, programs/, shared.md, killed.md, mission.md, procedures.md, etc.): 50 KB hard + 15 KB soft per-file
   - Log (`log.md`): 30 KB rolling; overflow → `data/archives/<YYYY-MM-DD>/log.md`

4. Evaluator's checklist grows from 11 to 18 items. Items 12–18 are memory discipline (see `data/agents/evaluator/procedural.md §memory_discipline`). Failures in 12–14 block phase close; 15–18 log and escalate after 3 consecutive.

5. KM closeout (`data/agents/findings_curator/procedural.md §KM-Closeout`) runs at every phase close BEFORE Evaluator's gate. Six tasks: findings inbox, INDEX maintenance, cap enforcement, semantic reindex, breadcrumb audit, telemetry line.

6. Values §4 archive-never-delete stands. `memory.delete` archives + breadcrumbs; it never destroys.

7. Rollback trigger: 3 consecutive phases with ≥2 evaluator memory-discipline failures, OR startup reads ≥20 KB for 5 consecutive sessions, OR `memory.py` corrupts an unrecoverable file → restore from `data/archives/snapshots/pre-memory-refactor/`, revert prompts, rename `data/memories/` → `data/memories_attempt_1/`, `lab_architect` runs ad-hoc retro.

8. Compaction bump (`CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` 50 → 70) is DEFERRED until 3 consecutive clean phases post-migration. Do not bump before validating.

This is a binding directive. Agents may not interpret it away. Lab_architect audits compliance at Program 3 open retro.
```

- [ ] **Step 2: Verify**

```bash
grep "D-117" data/pi_notes.md
```

- [ ] **Step 3: Commit**

```bash
git add data/pi_notes.md
git commit -m "feat(pi): DIRECTIVE D-117 — memory discipline is load-bearing

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**STAGE 2 GATE:** `.venv/bin/pytest tests/ -v -k "memory or migrate"` all pass; `data/agents/{director,pi,evaluator,findings_curator}/procedural.md` all carry the new sections; D-117 is in pi_notes.md.

---

# Stage 3: Runner & integration

## Task 18: Runner writes session_brief.md before each Director launch

**Files:**
- Modify: `run_agi_lab.sh`

- [ ] **Step 1: Find the Director-launch invocation**

```bash
grep -n "DIRECTOR_PROMPT\|claude.*--model" run_agi_lab.sh | head -10
```

Expected: locate the block inside the while-loop where Director is invoked.

- [ ] **Step 2: Add write_session_brief function**

Add near the top of `run_agi_lab.sh`, after other helper functions:

```bash
write_session_brief() {
    # Compose data/memories/session_brief.md from live state.
    # Atomic write (tmp + mv) so partial writes can't be consumed.
    local mem_root="data/memories"
    local brief_tmp="${mem_root}/session_brief.md.tmp"
    local brief_final="${mem_root}/session_brief.md"

    # If memories not yet bootstrapped, fall back gracefully (pre-migration).
    if [ ! -d "${mem_root}" ]; then
        return 0
    fi

    mkdir -p "${mem_root}"
    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local rate_limit_state="clean"
    if [ -f "data/infra/rate_limit_resets_at" ]; then
        rate_limit_state="waiting_until_$(cat data/infra/rate_limit_resets_at)"
    fi

    local last_telemetry="(none)"
    if [ -f "data/infra/memory_telemetry.log" ]; then
        last_telemetry=$(tail -n1 "data/infra/memory_telemetry.log")
    fi

    local last_decision="(unknown)"
    if [ -f "${mem_root}/log.md" ]; then
        last_decision=$(grep -m1 '^## D-' "${mem_root}/log.md" | sed 's/^## //' || echo "(none)")
    fi

    {
        echo "---"
        echo "generated_at: ${now}"
        echo "runner_pid: $$"
        echo "---"
        echo "# Session brief"
        echo ""
        echo "Last decision: ${last_decision}"
        echo "Rate-limit: ${rate_limit_state}"
        echo "Memory last session: ${last_telemetry}"
        echo ""
        echo "First tool call MUST be:"
        echo "  memory.view session_brief.md  (you are reading it now)"
        echo "Then:"
        echo "  memory.view INDEX.md"
        echo "Then selective views per INDEX."
    } > "${brief_tmp}"
    mv "${brief_tmp}" "${brief_final}"
}
```

- [ ] **Step 3: Call write_session_brief inside the Director-launch loop**

Locate the while-loop's Director invocation (around the `DIRECTOR_PROMPT=$(cat ...)` line) and add the call immediately before the `claude` invocation:

```bash
    write_session_brief
    DIRECTOR_PROMPT=$(cat data/agents/director_prompt.md)
    # ... existing claude invocation ...
```

- [ ] **Step 4: Test the function standalone**

```bash
mkdir -p data/memories
bash -c 'source run_agi_lab.sh; write_session_brief; cat data/memories/session_brief.md'
```

Expected: brief appears with generated_at header and body.

- [ ] **Step 5: Commit**

```bash
git add run_agi_lab.sh
git commit -m "feat(runner): write_session_brief.md before each Director launch

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Runner logs memory telemetry after each Director session

**Files:**
- Modify: `run_agi_lab.sh`

- [ ] **Step 1: Add log_memory_telemetry function**

```bash
log_memory_telemetry() {
    local mem_root="data/memories"
    local telem="data/infra/memory_telemetry.log"
    if [ ! -d "${mem_root}" ]; then
        return 0
    fi

    local now
    now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local hot_kb=0 log_kb=0 wiki_kb=0
    [ -f "${mem_root}/current.md" ] && hot_kb=$(( $(stat -f%z "${mem_root}/current.md" 2>/dev/null || echo 0) / 1024 ))
    [ -f "${mem_root}/log.md" ] && log_kb=$(( $(stat -f%z "${mem_root}/log.md" 2>/dev/null || echo 0) / 1024 ))
    # Wiki = everything in memories/ except hot/log/meta
    wiki_kb=$(find "${mem_root}" -name "*.md" \
        ! -name "current.md" ! -name "log.md" ! -name "INDEX.md" \
        ! -name "history.md" ! -name "session_brief.md" \
        -exec stat -f%z {} \; 2>/dev/null | awk '{s+=$1} END {print int(s/1024)}')
    wiki_kb=${wiki_kb:-0}

    mkdir -p data/infra
    echo "${now} | hot_kb=${hot_kb} | wiki_kb=${wiki_kb} | log_kb=${log_kb} | km_closeout=phase_close_only" \
        >> "${telem}"
}
```

- [ ] **Step 2: Call log_memory_telemetry after each Director session exits**

Find where the Director's `claude` invocation completes (typically just before the next loop iteration / checkpoint handling) and add:

```bash
    log_memory_telemetry
```

- [ ] **Step 3: Test standalone**

```bash
bash -c 'source run_agi_lab.sh; log_memory_telemetry; tail -1 data/infra/memory_telemetry.log'
```

Expected: a line matching the telemetry schema appears.

- [ ] **Step 4: Commit**

```bash
git add run_agi_lab.sh
git commit -m "feat(runner): log memory telemetry per session

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 20: Makefile targets for memory admin

**Files:**
- Modify: `Makefile`

- [ ] **Step 1: Inspect existing structure**

```bash
grep -n "^\.PHONY\|^lab-\|^hwmon" Makefile | head -20
```

- [ ] **Step 2: Append memory targets**

Append to `Makefile`:

```makefile

# ---- Memory tool (tools/memory.py) ----
.PHONY: memory-audit memory-index memory-rotate-log memory-test

memory-audit:
	.venv/bin/python3 tools/memory.py audit

memory-index:
	.venv/bin/python3 tools/memory.py index

memory-rotate-log:
	.venv/bin/python3 tools/memory.py rotate-log --cap-kb 30

memory-test:
	.venv/bin/pytest tests/test_memory.py tests/test_migrate_to_memories.py -v
```

- [ ] **Step 3: Verify targets work**

```bash
make memory-test
```

Expected: all tests pass.

```bash
mkdir -p data/memories && echo "# test" > data/memories/current.md
make memory-audit
```

Expected: "OK: no breaches" (small file).

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "build(make): memory-audit, memory-index, memory-rotate-log, memory-test

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 21: CLAUDE.md path references updated

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Find references to legacy paths**

```bash
grep -n "state\.md\|index\.md\|decisions_recent\.md\|director_log\.md\|shared_knowledge\.md\|killed_ideas\.md" CLAUDE.md
```

- [ ] **Step 2: Replace State Files section**

In `CLAUDE.md`, find the `## State Files` section and replace with:

```markdown
## State Files (post-D-117 memory refactor)

Primary entry (read first, always):
- `data/memories/session_brief.md` — runner-written pre-session snapshot
- `data/memories/INDEX.md` — directory of all memory files

Tiered memory content:
- `data/memories/current.md` — hot tier, active state (cap 40 KB)
- `data/memories/log.md` — log tier, recent decisions (cap 30 KB rolling; overflow archived)
- `data/memories/mission.md`, `governance/`, `shared.md`, `killed.md`, `procedures.md`, `programs/` — wiki tier (50 KB total cap, 15 KB soft per-file)

Tool: `tools/memory.py` implements Anthropic memory_20250818 protocol. See spec at `docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md`.

Agent and program state:
- `data/agents/<role>/{procedural.md,episodic/,semantic.md}` — per-role CoALA memory (NOT part of memory tiers)
- `programs/<program>/` — program deliverables, papers, closure memos (NOT part of memory tiers)

Legacy paths (archived with breadcrumbs at their old locations post-migration):
- `data/state.md`, `data/decisions_recent.md`, `data/director_log.md`, `data/pi_notes.md`, `data/shared_knowledge.md`, `data/killed_ideas.md`, `data/procedures.md`, `data/index.md`, `data/mission_reframe_2026-04-18.md`, `data/checkpoints/ARCHIVED.md`, `programs/portfolio.md`
```

- [ ] **Step 3: Verify**

```bash
grep "data/memories" CLAUDE.md
```

Expected: multiple matches.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(claude-md): update State Files for memory refactor

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 22: dashboard.py and dashboard.html path updates + tier panel

**Files:**
- Modify: `tools/dashboard.py`
- Modify: `tools/dashboard.html`

- [ ] **Step 1: Find state.md references**

```bash
grep -n "state\.md\|director_log\|decisions_recent" tools/dashboard.py tools/dashboard.html
```

- [ ] **Step 2: Update dashboard.py to prefer memories paths with fallback**

Replace paths with a helper that prefers the new location and falls back to legacy during dual-existence window. Add near the top of `dashboard.py`:

```python
def _memory_or_legacy(new_rel: str, legacy_rel: str) -> Path:
    new_path = Path(new_rel)
    legacy_path = Path(legacy_rel)
    if new_path.exists():
        return new_path
    return legacy_path  # breadcrumb or absent — caller handles
```

Replace every instance of hard-coded `data/state.md`:

```python
# Before
state_path = Path("data/state.md")

# After
state_path = _memory_or_legacy("data/memories/current.md", "data/state.md")
```

Same pattern for `data/decisions_recent.md` → `data/memories/log.md`, `data/director_log.md` → `data/memories/log.md`.

- [ ] **Step 3: Add tier-size panel**

Add a function that reads `memory.audit` output (or emulates it) and emits tier sizes. Add to dashboard.py's HTML generation:

```python
def _tier_sizes() -> dict:
    mem = Path("data/memories")
    if not mem.exists():
        return {"hot_kb": 0, "wiki_kb": 0, "log_kb": 0, "available": False}
    def kb(p):
        return p.stat().st_size // 1024 if p.exists() else 0
    hot_kb = kb(mem / "current.md")
    log_kb = kb(mem / "log.md")
    wiki_kb = sum(
        kb(p) for p in mem.rglob("*.md")
        if p.name not in {"current.md", "log.md", "INDEX.md",
                          "history.md", "session_brief.md"}
    )
    return {"hot_kb": hot_kb, "wiki_kb": wiki_kb, "log_kb": log_kb, "available": True}
```

Surface the result in the HTML with a panel titled "Memory tiers" showing hot/wiki/log vs caps (40/50/30 KB).

- [ ] **Step 4: Edit dashboard.html**

Add a panel section (or if HTML is generated from Python, add the template in Python):

```html
<section class="tier-panel">
    <h3>Memory tiers</h3>
    <div>Hot: <span id="hot_kb">0</span> / 40 KB</div>
    <div>Wiki: <span id="wiki_kb">0</span> / 50 KB</div>
    <div>Log: <span id="log_kb">0</span> / 30 KB</div>
</section>
```

- [ ] **Step 5: Smoke-test locally**

```bash
.venv/bin/python3 tools/dashboard.py --generate-static 2>&1 | head -20
```

Expected: no path-not-found errors; tier panel data emitted.

- [ ] **Step 6: Commit**

```bash
git add tools/dashboard.py tools/dashboard.html
git commit -m "feat(dashboard): tier-size panel + memories paths with legacy fallback

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 23: slack_bot.py path updates

**Files:**
- Modify: `tools/slack_bot.py`

- [ ] **Step 1: Find command handlers**

```bash
grep -n '"/status"\|"/phase"\|"/program"\|"/last"\|"/recent"\|state\.md\|director_log' tools/slack_bot.py
```

- [ ] **Step 2: Replace paths with memory-or-legacy helper**

Add near the top of `slack_bot.py`:

```python
def _mem_or_legacy(new_rel: str, legacy_rel: str):
    from pathlib import Path
    np = Path(new_rel)
    if np.exists():
        return np
    return Path(legacy_rel)
```

In each command handler that reads state, update:

```python
# /status
state = _mem_or_legacy("data/memories/current.md", "data/state.md").read_text()
# /recent or /last
log = _mem_or_legacy("data/memories/log.md", "data/decisions_recent.md").read_text()
# /phase (typically reads current.md for the "State" block)
# etc.
```

- [ ] **Step 3: Smoke-test command imports**

```bash
.venv/bin/python3 -c "import tools.slack_bot; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add tools/slack_bot.py
git commit -m "feat(slack): memories paths with legacy fallback for /status, /recent etc.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 24: lab_memory.py --incremental mode + explicit memories indexing

**Files:**
- Modify: `tools/lab_memory.py`

- [ ] **Step 1: Find the ingest entry point**

```bash
grep -n "def ingest\|def main\|--incremental" tools/lab_memory.py
```

- [ ] **Step 2: Add --incremental to the argparse CLI**

Locate the argparse section (likely near `def main()` or `if __name__ == "__main__"`). Add to the `ingest` subcommand:

```python
p_ingest.add_argument(
    "--incremental", action="store_true",
    help="only re-ingest files modified since last index timestamp stored in DB"
)
p_ingest.add_argument(
    "--include-memories", action="store_true", default=True,
    help="include data/memories/ in the ingest corpus (default: True)"
)
```

- [ ] **Step 3: Implement incremental filter**

In the ingest function, add a mtime filter:

```python
def _last_ingest_timestamp(self) -> float:
    row = self.conn.execute(
        "SELECT MAX(ingested_at) FROM chunks"
    ).fetchone()
    return row[0] if row and row[0] else 0.0

def ingest_incremental(self, root_paths: list[Path]) -> int:
    since = self._last_ingest_timestamp()
    count = 0
    for root in root_paths:
        for f in root.rglob("*.md"):
            if f.stat().st_mtime <= since:
                continue
            self.ingest(path=str(f), ...)  # existing signature
            count += 1
    return count
```

Surface it via `--incremental` flag. Default ingest corpus: `programs/`, `data/`, `data/memories/`, `docs/`.

- [ ] **Step 4: Test incremental**

```bash
# Touch nothing
.venv/bin/python3 tools/lab_memory.py ingest --incremental
# Expected: "0 files ingested (incremental)"

# Modify a file, then incremental should pick it up
touch data/memories/current.md  # if file exists
.venv/bin/python3 tools/lab_memory.py ingest --incremental
# Expected: non-zero count
```

(Smoke test only; `current.md` may not exist yet pre-migration — skip that half if so.)

- [ ] **Step 5: Commit**

```bash
git add tools/lab_memory.py
git commit -m "feat(lab_memory): --incremental mode using chunk mtime

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**STAGE 3 GATE:** `make memory-test` passes; `run_agi_lab.sh` contains `write_session_brief` and `log_memory_telemetry`; dashboard/slack/Makefile paths all updated. **No live lab-state changes yet.**

---

# Stage 4: Migration execution (GATED on Program 2 Phase 2 close)

**Do NOT start Stage 4 until Program 2 Phase 2 closes.** Check via:

```bash
grep -l "Phase 2.*close\|Phase 2 → Phase 3" data/director_log.md programs/program_2_dense_vs_moe_sub100m/meta/*.md 2>/dev/null
```

If that check is empty, wait. Migration on a mid-phase live lab would break the running session.

## Task 25: Pre-migration snapshot committed

**Files:**
- Modify: (none — creates `data/archives/snapshots/pre-memory-refactor/`)

- [ ] **Step 1: Verify legacy paths all exist**

```bash
for f in data/state.md data/pi_notes.md data/values.md data/shared_knowledge.md \
         data/killed_ideas.md data/procedures.md data/index.md data/director_log.md \
         data/decisions_recent.md data/mission_reframe_2026-04-18.md \
         data/checkpoints/ARCHIVED.md programs/portfolio.md; do
    [ -f "$f" ] && echo "OK: $f" || echo "MISSING: $f"
done
```

Expected: all OK (or explicit MISSING noted — non-blocking if file legitimately never existed).

- [ ] **Step 2: Run the snapshot function standalone**

```bash
.venv/bin/python3 -c "
from pathlib import Path
from tools.migrate_to_memories import _snapshot_all
snap = _snapshot_all(Path('.'))
print(f'Snapshot root: {snap}')
print(sorted(p.name for p in snap.iterdir()))
"
```

Expected: snapshot directory populated with 8–12 files.

- [ ] **Step 3: Commit snapshot**

```bash
git add data/archives/snapshots/pre-memory-refactor/
git commit -m "chore(memory): pre-migration snapshot of legacy paths

Archive-never-delete (Values §4). Referenced by rollback protocol §14.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 26: Run the migration and commit the new tree

**Files:**
- (all legacy files become breadcrumbs; `data/memories/` populated)

- [ ] **Step 1: Dry-run first**

```bash
.venv/bin/python3 tools/migrate_to_memories.py --dry-run --repo-root .
```

Expected: "Dry-run complete. Inspect: /tmp/..."
Walk the output path, confirm the memories tree looks sane.

- [ ] **Step 2: Real migration**

```bash
.venv/bin/python3 tools/migrate_to_memories.py --repo-root .
```

Expected: "Migration complete."

- [ ] **Step 3: Replace legacy files with breadcrumbs**

For each legacy file successfully migrated, replace its content with a breadcrumb pointing to the new path. Write a small helper:

```bash
for pair in \
    "data/state.md:data/memories/current.md" \
    "data/pi_notes.md:data/memories/governance/pi_notes.md" \
    "data/values.md:data/memories/governance/values.md" \
    "data/procedures.md:data/memories/procedures.md" \
    "data/shared_knowledge.md:data/memories/shared.md" \
    "data/killed_ideas.md:data/memories/killed.md" \
    "data/index.md:data/memories/INDEX.md" \
    "data/director_log.md:data/memories/log.md" \
    "data/decisions_recent.md:data/memories/log.md" \
    "data/mission_reframe_2026-04-18.md:data/memories/mission_reframe_2026-04-18.md"; do
    old="${pair%%:*}"; new="${pair##*:}"
    stamp=$(date -u +%Y-%m-%d)
    echo "<!-- archived ${stamp} → ${new} — see snapshot at data/archives/snapshots/pre-memory-refactor/ -->" > "${old}"
done
```

- [ ] **Step 4: Run audit**

```bash
make memory-audit
```

Expected: "OK: no breaches" (or exactly one soft-target breach for `shared.md` if not pre-split; if so, manually split per spec §10.2 before continuing).

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: no regressions in existing tests; memory tests still pass.

- [ ] **Step 6: Commit**

```bash
git add data/memories/ data/state.md data/pi_notes.md data/values.md \
    data/procedures.md data/shared_knowledge.md data/killed_ideas.md \
    data/index.md data/director_log.md data/decisions_recent.md \
    data/mission_reframe_2026-04-18.md data/checkpoints/ARCHIVED.md
git commit -m "refactor(memory): migrate to data/memories/ + breadcrumb legacy paths

D-117 cutover. Three-tier model live. Evaluator checks 12–18 active from
next phase close. See spec 2026-04-19-memory-tool-and-wiki-refactor.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 27: Semantic reindex post-migration

**Files:**
- (updates `tools/lab_memory.db`)

- [ ] **Step 1: Full reindex**

```bash
.venv/bin/python3 tools/lab_memory.py ingest --full \
    --roots programs data docs data/memories
```

Expected: chunk count increases; `data/memories/` files now searchable.

- [ ] **Step 2: Spot-check a search**

```bash
.venv/bin/python3 tools/lab_memory.py search "memory tier cap" -k 3
```

Expected: top hits include the spec + `current.md` / `governance/pi_notes.md`.

- [ ] **Step 3: Commit updated DB if tracked**

```bash
git add tools/lab_memory.db 2>/dev/null
git status tools/lab_memory.db
# If status shows a change, commit:
git commit -m "chore(lab_memory): reindex after memory migration

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>" || echo "DB not tracked — skip"
```

---

## Task 28: Cutover verification — first post-migration Director dry-run

**Files:**
- (read-only verification; no commits unless fix needed)

- [ ] **Step 1: Simulate runner pre-session hook**

```bash
bash -c 'source run_agi_lab.sh; write_session_brief; cat data/memories/session_brief.md'
```

Expected: brief well-formed, timestamp current.

- [ ] **Step 2: Simulate Director's first read**

```bash
wc -c data/memories/session_brief.md data/memories/INDEX.md data/memories/current.md
```

Expected: sum is <= 10 KB.

- [ ] **Step 3: Run a full audit + index + test**

```bash
make memory-audit
make memory-index
make memory-test
```

Expected: audit clean, index completes, tests pass.

- [ ] **Step 4: If anything fails, STOP and diagnose**

Either:
- Fix in a new commit and re-run from Step 1
- Or invoke rollback per spec §14

- [ ] **Step 5: Log cutover telemetry manually**

```bash
.venv/bin/python3 -c "
from datetime import datetime, timezone
from pathlib import Path
line = datetime.now(timezone.utc).isoformat() + ' | cutover=success | hot_kb=' \\
    + str(Path('data/memories/current.md').stat().st_size // 1024) \\
    + ' | wiki_kb=' + str(sum(
        p.stat().st_size for p in Path('data/memories').rglob('*.md')
        if p.name not in {'current.md','log.md','INDEX.md','history.md','session_brief.md'}
    ) // 1024) \\
    + ' | log_kb=' + str(Path('data/memories/log.md').stat().st_size // 1024)
Path('data/infra/memory_telemetry.log').parent.mkdir(parents=True, exist_ok=True)
with open('data/infra/memory_telemetry.log', 'a') as f:
    f.write(line + chr(10))
print(line)
"
```

- [ ] **Step 6: Commit telemetry baseline**

```bash
git add data/infra/memory_telemetry.log
git commit -m "chore(telemetry): baseline cutover line

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 29: Accountability ledger entry for D-117

**Files:**
- Modify: `data/accountability_ledger.md`

- [ ] **Step 1: Append D-117 entry**

Append to `data/accountability_ledger.md`:

```markdown

## D-117 — memory refactor cutover
**Date:** <fill: YYYY-MM-DD of cutover>
**Type:** structural change (not an incident)
**Summary:** Three-tier memory model (hot/wiki/log) with Anthropic memory_20250818-compatible tool live. Evaluator checks expanded 11→18. Rollback available from `data/archives/snapshots/pre-memory-refactor/` for 3 consecutive phases.
**Artifacts:**
- Spec: `docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md` (commit 0a199ba)
- Plan: `docs/superpowers/plans/2026-04-19-memory-tool-and-wiki-refactor.md`
- Tool: `tools/memory.py`
- Directive: `data/memories/governance/pi_notes.md §DIRECTIVE D-117`
**Validation window:** 3 consecutive clean phases from <cutover date>. At end of window, compaction bump 50 → 70 if clean; otherwise rollback.
```

- [ ] **Step 2: Commit**

```bash
git add data/accountability_ledger.md
git commit -m "chore(ledger): D-117 memory refactor cutover entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**STAGE 4 GATE (cutover):** At this point the lab is running under the new memory regime. **Resume the Director.** The validation window (3 clean phases) begins.

---

# Stage 5: Post-validation (GATED on 3 clean phases)

**Wait for evaluator reports from 3 consecutive phases to show zero memory-discipline failures (checks 12–18) before starting Stage 5.**

If any phase fails 2+ memory-discipline checks, invoke rollback per spec §14 instead of proceeding.

## Task 30: Bump CLAUDE_AUTOCOMPACT_PCT_OVERRIDE 50 → 70

**Files:**
- Modify: `run_agi_lab.sh`

- [ ] **Step 1: Find the override**

```bash
grep -n "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" run_agi_lab.sh
```

Expected: one or more matches.

- [ ] **Step 2: Change 50 → 70**

Edit the line:

```bash
# Before
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50
# After
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70
```

- [ ] **Step 3: Add a comment referencing D-117 validation**

```bash
# Bumped 50 → 70 on <DATE> after 3 consecutive clean phases post-D-117 migration.
# Rollback trigger still reverts this per spec §14.
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=70
```

- [ ] **Step 4: Commit**

```bash
git add run_agi_lab.sh
git commit -m "chore(runner): bump autocompact 50→70 post-D-117 validation

Three consecutive clean phases under the three-tier memory regime.
Startup reads consistently ≤10 KB. Safe to extend compaction window.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 31: Remove migration script and legacy-path fallbacks

**Files:**
- Delete: `tools/migrate_to_memories.py`, `tests/test_migrate_to_memories.py`
- Modify: `tools/dashboard.py`, `tools/slack_bot.py` (remove `_memory_or_legacy` / `_mem_or_legacy` fallbacks; direct `data/memories/...` paths only)

- [ ] **Step 1: Verify breadcrumbs still resolvable**

```bash
for f in data/state.md data/pi_notes.md data/values.md \
         data/shared_knowledge.md data/killed_ideas.md data/procedures.md \
         data/index.md data/director_log.md data/decisions_recent.md \
         data/mission_reframe_2026-04-18.md; do
    head -1 "$f" | grep -q "archived" && echo "OK breadcrumb: $f" || echo "MISSING breadcrumb: $f"
done
```

Expected: all OK.

- [ ] **Step 2: Remove legacy fallback in dashboard.py**

```python
# Before
state_path = _memory_or_legacy("data/memories/current.md", "data/state.md")
# After
state_path = Path("data/memories/current.md")
```

Remove the `_memory_or_legacy` helper entirely.

- [ ] **Step 3: Same for slack_bot.py**

Remove `_mem_or_legacy` helper and inline `Path("data/memories/...")` calls.

- [ ] **Step 4: Remove migration script**

```bash
git rm tools/migrate_to_memories.py tests/test_migrate_to_memories.py
```

- [ ] **Step 5: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: tests for `test_migrate_to_memories.py` no longer exist; everything else passes.

- [ ] **Step 6: Commit**

```bash
git add tools/dashboard.py tools/slack_bot.py
git commit -m "chore(memory): drop legacy fallbacks and migration script post-validation

Removed:
- tools/migrate_to_memories.py (one-time bootstrap, purpose served)
- tests/test_migrate_to_memories.py
- _memory_or_legacy helpers in dashboard.py and slack_bot.py

Legacy paths remain as breadcrumbs; git history preserves prior code.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

**STAGE 5 GATE (refactor fully adopted):** Memory tool is the only surface; legacy paths are breadcrumbs; compaction at 70; telemetry collecting; evaluator on 18-check cadence.

---

# Post-implementation tasks (not numbered, do as needed)

- Watch `data/infra/memory_telemetry.log` at each phase close; confirm hot/wiki/log sizes trending as spec expects.
- At Program 3 open, `lab_architect` retro incorporates memory refactor as a structural artifact.
- If native Anthropic Memory tool becomes available in Claude Code: configure it to mount at `data/memories/`, retire `tools/memory.py` (save a copy to `tools/archive/` for audit).
- `shared.md` may need splitting during first KM closeout if it came out of migration above the 15 KB soft target — this is expected, spec §10.3.

---

# Self-review checklist

Run through before advancing stages:

1. **Spec coverage:** every §-numbered requirement in `2026-04-19-memory-tool-and-wiki-refactor.md` has a task or is explicitly out-of-scope.
2. **Type consistency:** `Memory.view` returns `str | list[str]`, callers in tests and dashboard handle both.
3. **Path discipline:** all `_resolve` calls guard against escape; tests cover `../` and absolute paths.
4. **Archive-never-delete:** `delete` writes a breadcrumb; tests check that breadcrumb references archive path.
5. **TDD order:** each task writes test → fail → impl → pass → commit. No task skips the fail-verification step.
6. **No placeholders:** scan for "TBD", "TODO", "similar to" — fix any that appear.

---

*End of plan. Next step per `superpowers:subagent-driven-development`: dispatch implementer subagent for Task 1, then spec-reviewer, then code-quality-reviewer. Repeat through Task 11 for Stage 1, review with user, then Stage 2, etc.*
