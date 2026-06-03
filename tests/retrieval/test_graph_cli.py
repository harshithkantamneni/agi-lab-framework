"""CLI smoke tests for tools/retrieval/graph.py."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

def test_graph_build_cli(tmp_path):
    db = tmp_path / "test_graph.db"
    out = subprocess.run(
        [sys.executable, "-m", "tools.retrieval.graph", "build",
         "--db", str(db), "--root", str(REPO / "tests/retrieval/fixtures/sample_corpus")],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, f"stderr: {out.stderr}"
    assert "decisions" in out.stdout.lower() or "decisions" in out.stderr.lower()
    assert db.exists() and db.stat().st_size > 0


def test_graph_ppr_cli(tmp_path):
    db = tmp_path / "test_graph.db"
    subprocess.run(
        [sys.executable, "-m", "tools.retrieval.graph", "build",
         "--db", str(db), "--root", str(REPO / "tests/retrieval/fixtures/sample_corpus")],
        check=True, cwd=str(REPO),
    )
    out = subprocess.run(
        [sys.executable, "-m", "tools.retrieval.graph", "ppr",
         "--db", str(db), "--seed", "D-420", "--top-k", "5"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, f"stderr: {out.stderr}"
    assert "D-420" in out.stdout
