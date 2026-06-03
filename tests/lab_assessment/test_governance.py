from tools.lab_assessment.governance import compute, calibration_error
from tools.lab_assessment.sources import Sources


def test_calibration_error_perfect():
    # confidence 1.0 always correct, 0.0 always wrong -> error 0
    rows = [{"confidence": 1.0, "outcome": True}, {"confidence": 0.0, "outcome": False}]
    assert calibration_error(rows) == 0.0


def test_calibration_error_value():
    # one bin at 0.9 with 1/2 correct -> |0.9 - 0.5| = 0.4
    rows = [{"confidence": 0.9, "outcome": True}, {"confidence": 0.9, "outcome": False}]
    assert abs(calibration_error(rows) - 0.4) < 1e-9


def test_calibration_skips_unresolved_outcomes():
    # outcome=None rows (logged but not yet scored) are ignored — only resolved rows count
    rows = [{"confidence": 0.9, "outcome": None}, {"confidence": 0.9, "outcome": True},
            {"confidence": 0.9, "outcome": False}]
    assert abs(calibration_error(rows) - 0.4) < 1e-9


def test_governance_metrics(fake_repo):
    r = compute(Sources(fake_repo))
    assert r.dimension == "governance"
    assert r.metrics["org_adaptation_flags"] == 1   # one decision mentions it
    assert isinstance(r.metrics["calibration_mean_abs_error"], float)
    assert r.metrics["calibration_resolved_samples"] == 3
    assert r.metrics["evaluator_overall"] == "PASS_WITH_FLAGS"


def test_governance_all_outcomes_unresolved_is_unavailable(tmp_path):
    # mirrors the real lab TODAY: confidences logged, outcomes not yet scored
    d = tmp_path / "data/infra"
    d.mkdir(parents=True)
    (d / "calibration_telemetry.jsonl").write_text(
        '{"confidence": 0.9, "outcome": null}\n{"confidence": 0.8, "outcome": null}\n')
    r = compute(Sources(tmp_path))
    assert r.metrics["calibration_mean_abs_error"] == "unavailable"
    assert r.metrics["calibration_logged_samples"] == 2
    assert r.metrics["calibration_resolved_samples"] == 0
    assert any("resolved" in c.lower() for c in r.caveats)


def test_governance_degrades_no_calibration(tmp_path):
    r = compute(Sources(tmp_path))
    assert r.metrics["calibration_mean_abs_error"] == "unavailable"
