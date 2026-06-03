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


def test_migration_refuses_if_memories_already_exists_without_force(tmp_path):
    from tools.migrate_to_memories import migrate
    repo = _make_fixture_repo(tmp_path)
    # Pre-create a populated memories dir
    (repo / "data" / "memories").mkdir(parents=True)
    (repo / "data" / "memories" / "something.md").write_text("pre-existing")
    with pytest.raises(RuntimeError, match="memories.*already"):
        migrate(repo_root=repo)


def test_migration_force_overwrites_existing(tmp_path):
    from tools.migrate_to_memories import migrate
    repo = _make_fixture_repo(tmp_path)
    (repo / "data" / "memories").mkdir(parents=True)
    (repo / "data" / "memories" / "stale.md").write_text("stale")
    migrate(repo_root=repo, force=True)
    # Migration should have succeeded: current.md should exist
    assert (repo / "data" / "memories" / "current.md").exists()


def test_migration_allowed_if_memories_empty_dir(tmp_path):
    from tools.migrate_to_memories import migrate
    repo = _make_fixture_repo(tmp_path)
    (repo / "data" / "memories").mkdir(parents=True)  # empty dir is fine
    migrate(repo_root=repo)  # no force needed
    assert (repo / "data" / "memories" / "current.md").exists()
