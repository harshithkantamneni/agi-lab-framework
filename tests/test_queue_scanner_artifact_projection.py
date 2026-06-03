"""Tests for artifact_queue_projector integration into queue_scanner.scan()."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _setup_scanner_repo(tmp_path):
    """Build a sandbox repo skeleton for queue_scanner integration tests."""
    (tmp_path / "data/work_queue").mkdir(parents=True)
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/agents/_shared").mkdir(parents=True)
    (tmp_path / "data/memories/current.md").write_text(
        "# Current state\n\nactive_program: sample_program\n"
    )
    # Copy handler schema for orphan detector compatibility
    src_handlers = REPO / "data/agents/_shared/work_queue_handlers.md"
    if src_handlers.exists():
        (tmp_path / "data/agents/_shared/work_queue_handlers.md").write_text(src_handlers.read_text())
    return tmp_path


def _install_sample_program_schema(tmp_path):
    """Copy the artifact projector test fixture's sample_schema.yaml as the active program's schema."""
    prog_dir = tmp_path / "programs/sample_program"
    prog_dir.mkdir(parents=True)
    fixture = REPO / "tests/fixtures/artifact_projector/sample_schema.yaml"
    (prog_dir / "artifact_schema.yaml").write_text(fixture.read_text())


def test_scan_emits_artifact_projection_items(tmp_path, monkeypatch):
    """scan() includes _detect_artifact_projection items alongside other detectors."""
    from tools.queue_scanner import scan
    import tools.work_queue
    _setup_scanner_repo(tmp_path)
    _install_sample_program_schema(tmp_path)

    # Patch work_queue's _REPO so enqueue writes to tmp_path
    orig_repo = tools.work_queue._REPO
    tools.work_queue._REPO = tmp_path
    try:
        new_ids = scan(repo_root=tmp_path)
    finally:
        tools.work_queue._REPO = orig_repo

    # pending should include P1.foundation (leaf artifact with no prereqs missing)
    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    assert pending_path.exists()
    items = []
    for line in pending_path.read_text().splitlines():
        if line.strip():
            items.append(json.loads(line))
    artifact_items = [it for it in items if it.get("created_by") == "artifact_queue_projector"]
    assert len(artifact_items) == 1
    assert artifact_items[0]["payload"]["_artifact_id"] == "P1.foundation"


def test_scan_no_artifact_emit_when_active_program_missing(tmp_path, monkeypatch):
    """If current.md has no active_program OR program has no schema, no artifact items."""
    from tools.queue_scanner import scan
    import tools.work_queue
    _setup_scanner_repo(tmp_path)
    # Set active_program to one with no schema file
    (tmp_path / "data/memories/current.md").write_text(
        "# Current state\n\nactive_program: no_schema_program\n"
    )

    orig_repo = tools.work_queue._REPO
    tools.work_queue._REPO = tmp_path
    try:
        scan(repo_root=tmp_path)
    finally:
        tools.work_queue._REPO = orig_repo

    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    if pending_path.exists():
        items = [json.loads(l) for l in pending_path.read_text().splitlines() if l.strip()]
        artifact_items = [it for it in items if it.get("created_by") == "artifact_queue_projector"]
        assert artifact_items == []


def test_scan_artifact_projection_idempotent(tmp_path, monkeypatch):
    """Running scan twice does not duplicate artifact items (compute_id stable)."""
    from tools.queue_scanner import scan
    import tools.work_queue
    _setup_scanner_repo(tmp_path)
    _install_sample_program_schema(tmp_path)

    orig_repo = tools.work_queue._REPO
    tools.work_queue._REPO = tmp_path
    try:
        scan(repo_root=tmp_path)
        scan(repo_root=tmp_path)  # second call
    finally:
        tools.work_queue._REPO = orig_repo

    # Should still be exactly 1 artifact item (idempotent)
    pending_path = tmp_path / "data/work_queue/pending.jsonl"
    items = [json.loads(l) for l in pending_path.read_text().splitlines() if l.strip()]
    artifact_items = [it for it in items if it.get("created_by") == "artifact_queue_projector"]
    assert len(artifact_items) == 1


def test_scan_artifact_projection_failure_non_fatal(tmp_path, monkeypatch):
    """If projector raises, scan still completes and emits other detectors' items."""
    from tools.queue_scanner import scan
    import tools.work_queue
    _setup_scanner_repo(tmp_path)
    # Don't install schema → projector returns empty (not an error). For a real
    # exception test, mock project_artifacts to raise:
    import tools.artifact_queue_projector as ap
    original = ap.project_artifacts
    def raising_projector(program, repo_root=None):
        raise RuntimeError("simulated projector failure")
    monkeypatch.setattr(ap, "project_artifacts", raising_projector)

    orig_repo = tools.work_queue._REPO
    tools.work_queue._REPO = tmp_path
    try:
        # Should not raise
        new_ids = scan(repo_root=tmp_path)
    finally:
        tools.work_queue._REPO = orig_repo

    # Scan completed without crashing
    assert isinstance(new_ids, list)
