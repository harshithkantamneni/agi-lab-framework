from tools.lab_assessment.types import DimensionResult, VERDICT_LEVELS


def test_verdict_levels_are_the_five_we_agreed():
    assert VERDICT_LEVELS == ["Strong", "Solid", "Developing", "Weak", "N/A"]


def test_dimension_result_round_trips_to_dict():
    r = DimensionResult(
        dimension="ops",
        metrics={"cost_usd": 12.5, "throughput": "unavailable"},
        verdict_level="Solid",
        verdict_rationale="cheap per program",
        relative_to="Real-Mission cost goals",
        caveats=["throughput log missing"],
    )
    d = r.to_dict()
    assert d["dimension"] == "ops"
    assert d["verdict"]["level"] == "Solid"
    assert d["metrics"]["throughput"] == "unavailable"
    assert d["caveats"] == ["throughput log missing"]


def test_invalid_verdict_level_raises():
    import pytest
    with pytest.raises(ValueError):
        DimensionResult(dimension="x", metrics={}, verdict_level="Great",
                        verdict_rationale="", relative_to="", caveats=[])
