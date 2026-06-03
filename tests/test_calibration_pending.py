"""Tests for tools/calibration_pending.py — the pending-calibration nudge.

All tests run against a tiny temp jsonl (tmp_path) via the repo_root param, so
they never touch the live data/infra/calibration_telemetry.jsonl. A guard test
confirms the module never writes the telemetry file.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from tools import calibration_pending as cp


def _write_telemetry(repo_root: Path, records: list[dict]) -> Path:
    path = repo_root / "data/infra/calibration_telemetry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _iso_days_ago(days: float) -> str:
    dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
    return dt.isoformat(timespec="seconds")


def test_returns_only_pending_records(tmp_path):
    _write_telemetry(
        tmp_path,
        [
            {"ts": _iso_days_ago(1), "decision_id": "D-001", "role": "pi",
             "gate_type": "phase_close", "claim": "pending one",
             "confidence": 0.9, "outcome": None, "outcome_known_at": None},
            {"ts": _iso_days_ago(2), "decision_id": "D-002", "role": "pi",
             "gate_type": "paper_approval", "claim": "resolved true",
             "confidence": 0.8, "outcome": True,
             "outcome_known_at": _iso_days_ago(0)},
            {"ts": _iso_days_ago(3), "decision_id": "D-003", "role": "pi",
             "gate_type": "sub_step", "claim": "resolved false",
             "confidence": 0.7, "outcome": False,
             "outcome_known_at": _iso_days_ago(0)},
        ],
    )
    claims = cp.list_pending_claims(repo_root=tmp_path)
    ids = {c["decision_id"] for c in claims}
    assert ids == {"D-001"}
    assert claims[0]["gate_type"] == "phase_close"
    assert claims[0]["confidence"] == 0.9


def test_age_days_computed_and_sorted_oldest_first(tmp_path):
    _write_telemetry(
        tmp_path,
        [
            {"ts": _iso_days_ago(2), "decision_id": "D-NEW", "role": "pi",
             "gate_type": "x", "claim": "newer", "confidence": 0.5,
             "outcome": None, "outcome_known_at": None},
            {"ts": _iso_days_ago(10), "decision_id": "D-OLD", "role": "pi",
             "gate_type": "x", "claim": "older", "confidence": 0.5,
             "outcome": None, "outcome_known_at": None},
        ],
    )
    claims = cp.list_pending_claims(repo_root=tmp_path)
    # Oldest first.
    assert [c["decision_id"] for c in claims] == ["D-OLD", "D-NEW"]
    # age_days is a positive float roughly matching the inputs.
    ages = {c["decision_id"]: c["age_days"] for c in claims}
    assert ages["D-OLD"] is not None and 9 < ages["D-OLD"] < 11
    assert ages["D-NEW"] is not None and 1 < ages["D-NEW"] < 3


def test_unparseable_ts_gives_none_age_and_sorts_last(tmp_path):
    _write_telemetry(
        tmp_path,
        [
            {"ts": "not-a-date", "decision_id": "D-BAD", "role": "pi",
             "gate_type": "x", "claim": "bad ts", "confidence": 0.5,
             "outcome": None, "outcome_known_at": None},
            {"ts": _iso_days_ago(5), "decision_id": "D-GOOD", "role": "pi",
             "gate_type": "x", "claim": "good ts", "confidence": 0.5,
             "outcome": None, "outcome_known_at": None},
        ],
    )
    claims = cp.list_pending_claims(repo_root=tmp_path)
    by_id = {c["decision_id"]: c for c in claims}
    assert by_id["D-BAD"]["age_days"] is None
    assert by_id["D-GOOD"]["age_days"] is not None
    # Unparseable-ts record sorts last.
    assert claims[-1]["decision_id"] == "D-BAD"


def test_format_includes_pending_omits_resolved(tmp_path):
    _write_telemetry(
        tmp_path,
        [
            {"ts": _iso_days_ago(1), "decision_id": "D-PENDING", "role": "pi",
             "gate_type": "phase_close", "claim": "should appear",
             "confidence": 0.9, "outcome": None, "outcome_known_at": None},
            {"ts": _iso_days_ago(2), "decision_id": "D-RESOLVED", "role": "pi",
             "gate_type": "paper_approval", "claim": "should NOT appear",
             "confidence": 0.8, "outcome": True,
             "outcome_known_at": _iso_days_ago(0)},
        ],
    )
    brief = cp.format_pending_brief(cp.list_pending_claims(repo_root=tmp_path))
    assert "### Pending calibration claims (1)" in brief
    assert "score_calibration()" in brief
    assert "D-PENDING" in brief
    assert "D-RESOLVED" not in brief


def test_format_truncates_long_claim():
    long_claim = "x" * 500
    claims = [
        {"decision_id": "D-LONG", "role": "pi", "gate_type": "g",
         "confidence": 0.5, "claim": long_claim, "ts": None, "age_days": None},
    ]
    brief = cp.format_pending_brief(claims)
    # The rendered claim portion is truncated to ~100 chars.
    assert long_claim not in brief
    assert "x" * 99 in brief


def test_empty_list_renders_empty_string():
    assert cp.format_pending_brief([]) == ""


def test_missing_file_returns_empty(tmp_path):
    # No telemetry file written under tmp_path.
    assert cp.list_pending_claims(repo_root=tmp_path) == []
    assert cp.format_pending_brief(cp.list_pending_claims(repo_root=tmp_path)) == ""


def test_blank_and_malformed_lines_skipped(tmp_path):
    path = tmp_path / "data/infra/calibration_telemetry.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    good = {"ts": _iso_days_ago(1), "decision_id": "D-OK", "role": "pi",
            "gate_type": "x", "claim": "ok", "confidence": 0.5,
            "outcome": None, "outcome_known_at": None}
    path.write_text("\n".join(["", "  ", "{not json", json.dumps(good), ""]) + "\n")
    claims = cp.list_pending_claims(repo_root=tmp_path)
    assert [c["decision_id"] for c in claims] == ["D-OK"]


def test_module_never_writes_the_telemetry_file(tmp_path):
    path = _write_telemetry(
        tmp_path,
        [
            {"ts": _iso_days_ago(1), "decision_id": "D-001", "role": "pi",
             "gate_type": "phase_close", "claim": "pending",
             "confidence": 0.9, "outcome": None, "outcome_known_at": None},
        ],
    )
    before_bytes = path.read_bytes()
    before_mtime = path.stat().st_mtime

    # Exercise the full read + format path multiple times.
    for _ in range(3):
        cp.format_pending_brief(cp.list_pending_claims(repo_root=tmp_path))

    assert path.read_bytes() == before_bytes
    assert path.stat().st_mtime == before_mtime
