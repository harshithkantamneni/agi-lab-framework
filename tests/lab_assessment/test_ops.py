from tools.lab_assessment.ops import compute
from tools.lab_assessment.sources import Sources


def test_ops_metrics(fake_repo):
    r = compute(Sources(fake_repo))
    assert r.dimension == "ops"
    assert r.metrics["total_cost_usd"] == 42.5
    assert r.metrics["session_count"] == 17
    assert abs(r.metrics["cost_per_session_usd"] - 42.5 / 17) < 1e-6
    assert r.metrics["decisions_tracked"] == 2
    assert r.verdict_level in ("Strong", "Solid", "Developing", "Weak", "N/A")


def test_ops_degrades_when_cost_missing(tmp_path):
    r = compute(Sources(tmp_path))
    assert r.metrics["total_cost_usd"] == "unavailable"
    assert any("cost" in c.lower() for c in r.caveats)
