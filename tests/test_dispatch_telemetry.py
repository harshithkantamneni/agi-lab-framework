"""Tests for dispatch telemetry + rollup."""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_log_outcome_appends_to_telemetry(tmp_path, monkeypatch):
    """log_outcome should append a single JSON line to telemetry."""
    import tools.dispatch_helper as dh
    monkeypatch.setattr(dh, "_REPO", tmp_path)
    (tmp_path / "data/infra").mkdir(parents=True)

    dh.log_outcome(
        role="findings_curator",
        model="claude-sonnet-4-6",
        escalated=False,
        verifier_pass=None,
        task_class="archive",
    )

    log_path = tmp_path / "data/infra/dispatch_telemetry.jsonl"
    assert log_path.exists()
    line = log_path.read_text().strip()
    rec = json.loads(line)
    assert rec["role"] == "findings_curator"
    assert rec["model_dispatched"] == "claude-sonnet-4-6"
    assert rec["escalated"] is False
    assert rec["verifier_pass"] is None
    assert rec["task_class"] == "archive"
    assert "ts" in rec


def test_log_outcome_multiple_calls_each_appended(tmp_path, monkeypatch):
    import tools.dispatch_helper as dh
    monkeypatch.setattr(dh, "_REPO", tmp_path)
    (tmp_path / "data/infra").mkdir(parents=True)

    dh.log_outcome(role="a", model="claude-haiku-4-5", escalated=False, verifier_pass=None)
    dh.log_outcome(role="b", model="claude-sonnet-4-6", escalated=True, verifier_pass=None)
    dh.log_outcome(role="c", model="claude-opus-4-7", escalated=False, verifier_pass=False)

    log_path = tmp_path / "data/infra/dispatch_telemetry.jsonl"
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 3
    roles = [json.loads(l)["role"] for l in lines]
    assert roles == ["a", "b", "c"]


def test_rollup_summarizes_per_role_rates(tmp_path, monkeypatch):
    """dispatch_rollup.py reads telemetry and produces per-role rate table."""
    import tools.dispatch_rollup as dr
    monkeypatch.setattr(dr, "REPO", tmp_path)
    log_dir = tmp_path / "data/infra"
    log_dir.mkdir(parents=True)
    log = log_dir / "dispatch_telemetry.jsonl"
    log.write_text("\n".join([
        json.dumps({"ts": "2026-05-04T00:00:00Z", "role": "x", "model_dispatched": "claude-haiku-4-5",
                    "task_class": "", "escalated": True, "verifier_pass": None}),
        json.dumps({"ts": "2026-05-04T00:01:00Z", "role": "x", "model_dispatched": "claude-haiku-4-5",
                    "task_class": "", "escalated": False, "verifier_pass": None}),
        json.dumps({"ts": "2026-05-04T00:02:00Z", "role": "y", "model_dispatched": "claude-opus-4-7",
                    "task_class": "", "escalated": False, "verifier_pass": False}),
    ]) + "\n")

    dr.main()

    rollup_path = tmp_path / "data/infra/dispatch_rollup.md"
    assert rollup_path.exists()
    content = rollup_path.read_text()
    assert "| x |" in content
    assert "| y |" in content
    # x: 2 dispatches, 1 escalation = 50% escalation rate
    assert "50.0%" in content
    # y: 1 dispatch, 1 verifier_fail = 100% verifier_fail rate
    assert "100.0%" in content


def test_rollup_handles_missing_telemetry(tmp_path, monkeypatch):
    import tools.dispatch_rollup as dr
    monkeypatch.setattr(dr, "REPO", tmp_path)
    # No telemetry file — should not crash
    dr.main()  # prints "no telemetry yet" but doesn't raise
