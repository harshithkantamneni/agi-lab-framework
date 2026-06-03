"""Validates tools/training_digest.py module skeleton."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def test_module_imports():
    import tools.training_digest as td
    assert hasattr(td, "main")
    assert hasattr(td, "REPO")
    assert hasattr(td, "SCHEMA_VERSION")
    assert td.SCHEMA_VERSION == "1.0"


def test_main_no_args_prints_usage_and_returns_nonzero(capsys):
    import tools.training_digest as td
    rc = td.main(argv=[])
    assert rc != 0


def test_main_with_cell_id_returns_zero_or_nonzero_no_crash(tmp_path, monkeypatch):
    """Skeleton: main with cell_id should not raise. Real logic comes in Task 11."""
    import tools.training_digest as td
    monkeypatch.setattr(td, "REPO", tmp_path)
    rc = td.main(argv=["A42"])
    # Skeleton may return 0 (no-op) or non-zero (cell not found) — either is acceptable
    assert isinstance(rc, int)


def _setup_full_repo(tmp_path):
    """Same as test_training_digest_parser._setup_digest_repo but importable here."""
    import json
    runs_dir = tmp_path / "data/runs/phase3_A42"
    runs_dir.mkdir(parents=True)
    fixture_log = REPO / "tests/fixtures/training_digest/phase3_A42_sample.log"
    (runs_dir / "stdout.log").write_text(fixture_log.read_text())

    ckpt_dir = tmp_path / "data/checkpoints/phase3_factorial"
    ckpt_dir.mkdir(parents=True)
    fixture_entry = json.loads((REPO / "tests/fixtures/training_digest/phase3_A42_run_index_entry.json").read_text())
    (ckpt_dir / "run_index.json").write_text(json.dumps({"A42": fixture_entry}, indent=2))
    return tmp_path


def test_main_writes_md_and_json_files(tmp_path, monkeypatch):
    """main A42 with explicit --phase writes both .md and .json sidecar."""
    import tools.training_digest as td
    monkeypatch.setattr(td, "REPO", _setup_full_repo(tmp_path))

    rc = td.main(argv=["A42", "--phase", "phase3_factorial"])
    assert rc == 0

    md_path = tmp_path / "data/digests/training/A42.md"
    json_path = tmp_path / "data/digests/training/A42.json"
    assert md_path.exists(), "expected markdown digest file"
    assert json_path.exists(), "expected JSON sidecar"

    md = md_path.read_text()
    assert "A42" in md
    assert "Outcome" in md


def test_main_auto_discovers_phase(tmp_path, monkeypatch):
    """If --phase is omitted, main scans data/checkpoints/*/run_index.json for the cell."""
    import tools.training_digest as td
    monkeypatch.setattr(td, "REPO", _setup_full_repo(tmp_path))

    rc = td.main(argv=["A42"])
    assert rc == 0
    assert (tmp_path / "data/digests/training/A42.json").exists()


def test_main_skips_when_digest_fresh(tmp_path, monkeypatch):
    """If digest exists and stdout_mtime <= digest_generated_at, skip regen.

    Run main once, capture digest_generated_at, run again, verify the timestamp didn't change.
    """
    import tools.training_digest as td
    import json as _json
    monkeypatch.setattr(td, "REPO", _setup_full_repo(tmp_path))

    rc = td.main(argv=["A42", "--phase", "phase3_factorial"])
    assert rc == 0
    json_path = tmp_path / "data/digests/training/A42.json"
    first_gen_at = _json.loads(json_path.read_text())["digest_generated_at"]

    # Run again immediately — should skip (mtime hasn't changed)
    rc = td.main(argv=["A42", "--phase", "phase3_factorial"])
    assert rc == 0
    second_gen_at = _json.loads(json_path.read_text())["digest_generated_at"]
    assert first_gen_at == second_gen_at, \
        f"expected idempotent skip, but digest_generated_at changed: {first_gen_at} -> {second_gen_at}"


def test_main_regenerates_when_stdout_newer(tmp_path, monkeypatch):
    """If stdout.log mtime is newer than digest_generated_at, regen automatically."""
    import tools.training_digest as td
    import json as _json
    import time
    monkeypatch.setattr(td, "REPO", _setup_full_repo(tmp_path))

    rc = td.main(argv=["A42", "--phase", "phase3_factorial"])
    assert rc == 0
    json_path = tmp_path / "data/digests/training/A42.json"
    first_gen_at = _json.loads(json_path.read_text())["digest_generated_at"]

    # Make stdout.log newer than the digest
    time.sleep(1.1)
    (tmp_path / "data/runs/phase3_A42/stdout.log").touch()

    rc = td.main(argv=["A42", "--phase", "phase3_factorial"])
    assert rc == 0
    second_gen_at = _json.loads(json_path.read_text())["digest_generated_at"]
    assert first_gen_at != second_gen_at, \
        "expected regen because stdout.log is newer"


def test_main_regen_flag_forces_rebuild(tmp_path, monkeypatch):
    """--regen forces regen even when digest is fresh."""
    import tools.training_digest as td
    import json as _json
    import time
    monkeypatch.setattr(td, "REPO", _setup_full_repo(tmp_path))

    rc = td.main(argv=["A42", "--phase", "phase3_factorial"])
    assert rc == 0
    json_path = tmp_path / "data/digests/training/A42.json"
    first_gen_at = _json.loads(json_path.read_text())["digest_generated_at"]

    time.sleep(1.1)  # ensure timestamp difference

    rc = td.main(argv=["A42", "--phase", "phase3_factorial", "--regen"])
    assert rc == 0
    second_gen_at = _json.loads(json_path.read_text())["digest_generated_at"]
    assert first_gen_at != second_gen_at


def test_main_returns_nonzero_when_cell_not_found(tmp_path, monkeypatch):
    """Cell ID with no matching stdout/run_index entry should return non-zero."""
    import tools.training_digest as td
    monkeypatch.setattr(td, "REPO", _setup_full_repo(tmp_path))
    rc = td.main(argv=["NONEXISTENT", "--phase", "phase3_factorial"])
    assert rc != 0
