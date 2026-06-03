"""Integration test: run brief_assembler against real lab state via CLI, verify output."""
from __future__ import annotations
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_cli_writes_context_brief(tmp_path):
    out_path = tmp_path / "context_brief.md"
    result = subprocess.run(
        ["python3", "tools/brief_assembler.py", "--out", str(out_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()
    content = out_path.read_text()
    assert content.startswith("---\ngenerated_at:")
    assert "session_type:" in content
    # Real state should be valid markdown — we don't assert specific session_type
    # since lab state may be in any phase
    size_kb = len(content.encode("utf-8")) / 1024
    assert size_kb < 30, f"brief too large: {size_kb:.1f} KB"
