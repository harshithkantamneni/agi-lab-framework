from tools.lab_assessment.sources import Sources


def test_experiments_rows(fake_repo):
    src = Sources(fake_repo)
    rows = src.experiments()
    assert len(rows) == 2
    assert {r["status"] for r in rows} == {"proposed", "complete"}


def test_cost_rollup(fake_repo):
    src = Sources(fake_repo)
    cost = src.cost_rollup()
    assert cost["total_cost_usd"] == 42.5
    assert cost["session_count"] == 17


def test_scorecard_scores(fake_repo):
    src = Sources(fake_repo)
    scores = src.scorecard_scores()
    assert abs(scores["MMLU"]["accuracy"] - 0.2895) < 1e-9
    assert scores["MMLU"]["correct"] == 66 and scores["MMLU"]["total"] == 228
    assert "HellaSwag" in scores


def test_calibration_telemetry(fake_repo):
    src = Sources(fake_repo)
    rows = src.calibration_telemetry()
    assert len(rows) == 3
    assert rows[0]["confidence"] == 0.9


def test_decision_headers(fake_repo):
    src = Sources(fake_repo)
    decs = src.decision_headers()
    assert {d["id"] for d in decs} == {"D-10", "D-11"}
    assert any("ORG_ADAPTATION" in d["headline"] for d in decs)


def test_closure_memos(fake_repo):
    src = Sources(fake_repo)
    memos = src.closure_memos()
    assert len(memos) == 1
    assert "envelope" in memos[0]["text"]


def test_missing_source_returns_safe_empty(tmp_path):
    src = Sources(tmp_path)  # empty dir
    assert src.experiments() == []
    assert src.cost_rollup() is None
    assert src.scorecard_scores() == {}
    assert src.calibration_telemetry() == []
    assert src.decision_headers() == []
