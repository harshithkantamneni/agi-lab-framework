"""Validates PI self-correction infrastructure (Reflexion + Calibration + Budgets)."""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# === Stage P1: Reflexion retrospective ===

RETRO_TEMPLATE = REPO / "data/agents/_shared/program_retrospective_template.md"
PI_PROCEDURAL = REPO / "data/agents/pi/procedural.md"


def test_retrospective_template_exists():
    assert RETRO_TEMPLATE.exists(), f"missing: {RETRO_TEMPLATE}"
    text = RETRO_TEMPLATE.read_text()
    # Template must ask the 3 reflection questions explicitly
    assert "Q1" in text and "Q2" in text and "Q3" in text
    # Must require calibration on the answers
    assert "confidence" in text.lower()
    # Must specify verifier check
    assert "verifier" in text.lower() or "lab_architect" in text.lower()


def test_pi_procedural_references_retrospective_template():
    text = PI_PROCEDURAL.read_text()
    # Procedural must reference the template path
    assert "program_retrospective_template.md" in text, \
        "PI procedural must reference the retrospective template at program close"
    # Must require it at program close
    assert "program close" in text.lower() or "close — reflexion" in text.lower()
    # Must specify the output path pattern
    assert "pi/episodic" in text and "retrospective" in text.lower()


def test_retrospective_template_flags_rote_answers():
    """Template must explicitly flag generic 'it went fine' answers as insufficient."""
    text = RETRO_TEMPLATE.read_text()
    assert "rote" in text.lower() or "generic" in text.lower(), \
        "template must explicitly call out generic answers as verifier-flagged"


# === Stage P2: Hard cycle budgets ===

BUDGETS_PATH = REPO / "data/programs/budgets.json"


def test_budgets_file_exists_and_parses():
    assert BUDGETS_PATH.exists(), f"missing: {BUDGETS_PATH}"
    cfg = json.loads(BUDGETS_PATH.read_text())
    assert "default" in cfg
    d = cfg["default"]
    for f in ("max_days_without_close", "max_total_days", "warn_at_pct"):
        assert f in d, f"default budget missing {f}"
    assert 0 < d["warn_at_pct"] <= 1
    assert d["max_days_without_close"] > 0
    assert d["max_total_days"] >= d["max_days_without_close"]


def test_pi_procedural_describes_budget_tripwire_workflow():
    text = PI_PROCEDURAL.read_text()
    assert "Hard Budget" in text or "budget tripwire" in text.lower()
    # Must list the 3 response options
    for keyword in ("close", "pivot", "extension"):
        assert keyword in text.lower(), f"PI procedural missing tripwire option: {keyword}"
    # Must reference the override file mechanism
    assert "budget_overrides.md" in text, \
        "PI procedural must mention how to extend a budget (override file)"


def test_brief_assembler_surfaces_budget_when_overrun(tmp_path):
    """Brief assembler must inject a budget tripwire notice when active program is over budget."""
    import sys, time, os
    sys.path.insert(0, str(REPO))
    from tools.brief_assembler import _check_program_budget

    # Set up fake repo with active program and budgets config
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/current.md").write_text(
        "active_program: test_overrun_program\n"
    )
    (tmp_path / "data/programs").mkdir(parents=True)
    (tmp_path / "data/programs/budgets.json").write_text(json.dumps({
        "default": {"max_days_without_close": 1, "max_total_days": 1, "warn_at_pct": 0.5}
    }))
    prog_dir = tmp_path / "programs/test_overrun_program"
    prog_dir.mkdir(parents=True)
    memo = prog_dir / "program_open_memo.md"
    memo.write_text("opened")
    # Backdate the memo to 5 days ago — ensures over-budget
    old = time.time() - 5 * 86400
    os.utime(memo, (old, old))

    notice = _check_program_budget(tmp_path)
    assert notice, "expected a budget tripwire notice for overrun program"
    assert "TRIPWIRE" in notice or "WARNING" in notice
    assert "test_overrun_program" in notice


def test_brief_assembler_silent_when_within_budget(tmp_path):
    import sys
    sys.path.insert(0, str(REPO))
    from tools.brief_assembler import _check_program_budget

    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/current.md").write_text(
        "active_program: test_fresh_program\n"
    )
    (tmp_path / "data/programs").mkdir(parents=True)
    (tmp_path / "data/programs/budgets.json").write_text(json.dumps({
        "default": {"max_days_without_close": 14, "max_total_days": 60, "warn_at_pct": 0.75}
    }))
    prog_dir = tmp_path / "programs/test_fresh_program"
    prog_dir.mkdir(parents=True)
    (prog_dir / "program_open_memo.md").write_text("opened")  # mtime = now

    notice = _check_program_budget(tmp_path)
    assert notice == "", f"expected silent (within budget), got: {notice!r}"


def test_budget_override_suppresses_tripwire(tmp_path):
    """When budget_overrides.md exists, tripwire should not fire."""
    import sys, time, os
    sys.path.insert(0, str(REPO))
    from tools.brief_assembler import _check_program_budget

    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/current.md").write_text(
        "active_program: test_overridden\n"
    )
    (tmp_path / "data/programs").mkdir(parents=True)
    (tmp_path / "data/programs/budgets.json").write_text(json.dumps({
        "default": {"max_days_without_close": 1, "max_total_days": 1, "warn_at_pct": 0.5}
    }))
    prog_dir = tmp_path / "programs/test_overridden"
    prog_dir.mkdir(parents=True)
    memo = prog_dir / "program_open_memo.md"
    memo.write_text("opened")
    old = time.time() - 5 * 86400
    os.utime(memo, (old, old))
    # Add override
    (prog_dir / "budget_overrides.md").write_text("Extended for envelope paper completion.")

    notice = _check_program_budget(tmp_path)
    # Override suppresses TRIPWIRE but the warning may still fire
    assert "TRIPWIRE" not in notice, \
        f"override should suppress tripwire, got: {notice!r}"


# === Stage P3: Calibrated confidence ===

VERIFIER_PAIRS = REPO / "data/agents/_shared/verifier_pairs.json"
CALIB_LOGGER = REPO / "tools/calibration_logger.py"
CALIB_ROLLUP = REPO / "tools/calibration_rollup.py"


def test_calibration_logger_appends_record(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(REPO))
    import tools.calibration_logger as cl
    monkeypatch.setattr(cl, "_REPO", tmp_path)
    (tmp_path / "data/infra").mkdir(parents=True)

    cl.log_calibration(
        decision_id="D-100",
        claim="this paper is publishable at ICLR workshop",
        confidence=0.65,
        reasoning="solid positive section, novel framing, but small-scale",
        gate_type="paper_approval",
    )

    log = tmp_path / "data/infra/calibration_telemetry.jsonl"
    assert log.exists()
    rec = json.loads(log.read_text().strip())
    assert rec["decision_id"] == "D-100"
    assert rec["confidence"] == 0.65
    assert rec["gate_type"] == "paper_approval"
    assert rec["outcome"] is None  # not yet scored


def test_calibration_logger_rejects_invalid_confidence(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(REPO))
    import tools.calibration_logger as cl
    monkeypatch.setattr(cl, "_REPO", tmp_path)

    import pytest as _pt
    with _pt.raises(ValueError):
        cl.log_calibration(decision_id="D-X", claim="x", confidence=1.5, reasoning="x")
    with _pt.raises(ValueError):
        cl.log_calibration(decision_id="D-X", claim="x", confidence=-0.1, reasoning="x")


def test_score_calibration_updates_outcome(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(REPO))
    import tools.calibration_logger as cl
    monkeypatch.setattr(cl, "_REPO", tmp_path)
    (tmp_path / "data/infra").mkdir(parents=True)

    cl.log_calibration(decision_id="D-200", claim="x", confidence=0.7, reasoning="x")
    found = cl.score_calibration("D-200", outcome=True)
    assert found

    log = tmp_path / "data/infra/calibration_telemetry.jsonl"
    rec = json.loads(log.read_text().strip())
    assert rec["outcome"] is True
    assert rec["outcome_known_at"] is not None


def test_calibration_rollup_buckets_correctly(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, str(REPO))
    import tools.calibration_rollup as cr
    monkeypatch.setattr(cr, "REPO", tmp_path)
    (tmp_path / "data/infra").mkdir(parents=True)
    log = tmp_path / "data/infra/calibration_telemetry.jsonl"
    # 3 records at 70% confidence; 2 hit, 1 miss → actual hit rate 67%
    log.write_text("\n".join([
        json.dumps({"decision_id": f"D-{i}", "confidence": 0.7,
                    "outcome": (i < 2), "claim": "x", "reasoning": "x", "role": "pi", "gate_type": ""})
        for i in range(3)
    ]) + "\n")

    cr.main()

    rollup = tmp_path / "data/infra/calibration_rollup.md"
    assert rollup.exists()
    content = rollup.read_text()
    assert "60-80%" in content
    assert "66.7%" in content or "67" in content  # 2/3 hit rate


def test_pi_procedural_requires_calibration_on_verdicts():
    text = PI_PROCEDURAL.read_text()
    assert "Calibrated Confidence" in text or "calibrated confidence" in text.lower()
    assert "log_calibration" in text  # must reference the logger
    assert "CONFIDENCE" in text  # must show the format
    assert "CALIBRATION_REASONING" in text


def test_verifier_pair_for_pi_includes_calibration_check():
    pairs = json.loads(VERIFIER_PAIRS.read_text())
    pi_pair = pairs["pi"]
    assert "calibration_check" in pi_pair, \
        "pi verifier pair must include calibration_check field"
    assert "CONFIDENCE" in pi_pair["calibration_check"]
    assert "log_calibration" in pi_pair["calibration_check"]
