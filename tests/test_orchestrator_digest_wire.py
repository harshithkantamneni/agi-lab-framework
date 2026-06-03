"""Smoke test: orchestrator includes calls to training_digest tools."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ORCHESTRATOR = REPO / "tools/run_phase3_factorial.py"


def test_orchestrator_calls_per_cell_digester():
    """The per-cell digester invocation must appear inside the cell loop."""
    text = ORCHESTRATOR.read_text()
    assert "training_digest.py" in text, \
        "orchestrator must invoke tools/training_digest.py per cell"


def test_orchestrator_calls_phase_aggregator():
    """The phase-level aggregator invocation must appear after the loop."""
    text = ORCHESTRATOR.read_text()
    assert "training_digest_aggregate.py" in text, \
        "orchestrator must invoke tools/training_digest_aggregate.py at phase end"


def test_orchestrator_digest_calls_are_non_fatal():
    """Both invocations must be wrapped so a digester crash doesn't halt training."""
    text = ORCHESTRATOR.read_text()
    # Find the digester invocations and confirm they're inside try/except
    # OR they use subprocess with check=False (NOT check=True)
    digest_idx = text.find("training_digest.py")
    assert digest_idx > 0
    # Look at the ~300 chars around the call for try/except OR check=False
    block = text[max(0, digest_idx - 200): digest_idx + 200]
    assert ("try:" in block and "except" in text[digest_idx:digest_idx+800]) \
        or "check=False" in block, \
        f"per-cell digester invocation must be non-fatal (try/except or check=False); got: {block!r}"

    agg_idx = text.find("training_digest_aggregate.py")
    block = text[max(0, agg_idx - 200): agg_idx + 200]
    assert ("try:" in block and "except" in text[agg_idx:agg_idx+800]) \
        or "check=False" in block, \
        "aggregator invocation must be non-fatal"


def test_orchestrator_per_cell_digest_runs_after_run_index_write():
    """Per-cell digester must fire AFTER write_run_index, not before
    (so the digester sees fresh run_index data)."""
    text = ORCHESTRATOR.read_text()
    # Find the per-cell digester invocation and look at the preceding context
    idx = text.find("training_digest.py")
    assert idx > 0
    preceding = text[max(0, idx - 800): idx]
    # write_run_index call must be visible in the preceding window
    assert "write_run_index" in preceding, \
        "per-cell digester must fire AFTER write_run_index in the cell loop"


def test_orchestrator_python_syntax_valid():
    """The modified file must still parse as valid Python."""
    import ast
    text = ORCHESTRATOR.read_text()
    ast.parse(text)
