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
from tools.memory import Memory, MemoryFileNotFoundError, PathOutsideRootError  # noqa: E402


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


def test_resolve_log_anchor_does_not_substring_match(tmp_path):
    """D-1 should not match D-11 or D-100 (anchor word-boundary)."""
    m = Memory(root=tmp_path / "m")
    m.create("log.md", "## D-11 header\nbody\n## D-100 bigger\nbody\n")
    # D-1 is NOT in the log; only D-11 and D-100 are. Must raise.
    from tools.memory import UnresolvedReferenceError
    with pytest.raises(UnresolvedReferenceError):
        m.resolve("[log:2026-04-19#D-1]")
    # Sanity: D-11 and D-100 should still resolve
    assert m.resolve("[log:2026-04-19#D-11]") == (m.root / "log.md").resolve()
    assert m.resolve("[log:2026-04-19#D-100]") == (m.root / "log.md").resolve()


def test_resolve_episodic_rejects_path_traversal(tmp_path):
    """Crafted episodic reference with .. must not escape data/agents/."""
    m = Memory(root=tmp_path / "m")
    m.repo_root = tmp_path
    # Create a file OUTSIDE data/agents/ to try to reach
    secret = tmp_path / "secret.md"
    secret.write_text("nope")
    from tools.memory import UnresolvedReferenceError
    with pytest.raises(UnresolvedReferenceError):
        m.resolve("[episodic:pi/../../secret]")


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


def test_index_skips_yaml_frontmatter_block(tmp_path):
    """index() must skip full --- ... --- frontmatter, not mistake yaml for description."""
    m = Memory(root=tmp_path / "m")
    m.create(
        "brief.md",
        "---\n"
        "title: Session brief\n"
        "generated_at: 2026-04-19\n"
        "---\n"
        "\n"
        "# Real heading\n"
        "Body text.\n",
    )
    m.index()
    idx = (m.root / "INDEX.md").read_text()
    # Description should be "Real heading", not "title: Session brief"
    assert "Real heading" in idx
    assert "title: Session brief" not in idx


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
