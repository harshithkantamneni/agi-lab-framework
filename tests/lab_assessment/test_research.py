from tools.lab_assessment.research import compute, classify_output_type
from tools.lab_assessment.sources import Sources


def test_classify_output_type():
    assert classify_output_type("public_type: envelope\n...") == "envelope"
    assert classify_output_type("a methodology contribution") == "methodology"
    assert classify_output_type("nothing here") == "unclassified"


def test_research_metrics(fake_repo):
    r = compute(Sources(fake_repo))
    assert r.dimension == "research"
    assert r.metrics["programs_closed"] == 1
    assert r.metrics["output_type_mix"]["envelope"] == 1
    assert r.metrics["decisions_total"] == 2
    assert r.verdict_level in ("Strong", "Solid", "Developing", "Weak", "N/A")


def test_research_degrades_no_programs(tmp_path):
    r = compute(Sources(tmp_path))
    assert r.metrics["programs_closed"] == 0
