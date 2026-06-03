from tools.lab_assessment.benchmarks import compute, RANDOM_BASELINE
from tools.lab_assessment.sources import Sources


def test_benchmarks_metrics(fake_repo):
    r = compute(Sources(fake_repo))
    assert r.dimension == "benchmarks"
    mmlu = r.metrics["MMLU"]
    assert abs(mmlu["accuracy"] - 0.2895) < 1e-9
    assert mmlu["random_baseline"] == 0.25
    assert mmlu["above_random"] is True   # 0.2895 > 0.25
    hs = r.metrics["HellaSwag"]
    assert hs["above_random"] is False    # 0.23 < 0.25
    assert r.verdict_level in ("Weak", "Developing", "N/A")


def test_benchmarks_degrades_when_scorecard_missing(tmp_path):
    r = compute(Sources(tmp_path))
    assert r.metrics == {} or all(v == "unavailable" for v in r.metrics.values())
    assert r.verdict_level == "N/A"
