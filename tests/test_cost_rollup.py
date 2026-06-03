"""Validates tools/cost_rollup.py — jsonl session parser + pricing constants."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FIXTURE = REPO / "tests/fixtures/cost_rollup"


def test_module_imports():
    import tools.cost_rollup as cr
    assert hasattr(cr, "REPO")
    assert hasattr(cr, "PRICING")
    assert hasattr(cr, "SCHEMA_VERSION")
    assert hasattr(cr, "main")
    assert cr.SCHEMA_VERSION == "1.0"


def test_pricing_has_4_buckets_per_model():
    """Every PRICING entry must have input, cache_creation, cache_read, output."""
    import tools.cost_rollup as cr
    required = {"input", "cache_creation", "cache_read", "output"}
    for model, prices in cr.PRICING.items():
        missing = required - set(prices)
        assert not missing, f"{model}: missing {missing}"


def test_pricing_cache_creation_is_1_25x_input():
    """cache_creation must be 1.25x input rate per Anthropic 5-min cache premium."""
    import tools.cost_rollup as cr
    for model, prices in cr.PRICING.items():
        ratio = prices["cache_creation"] / prices["input"]
        assert abs(ratio - 1.25) < 0.01, \
            f"{model}: cache_creation/input ratio {ratio:.3f} != 1.25"


def test_pricing_opus_canonical_values():
    """Hardcoded sanity check on opus pricing."""
    import tools.cost_rollup as cr
    p = cr.PRICING["claude-opus-4-7"]
    assert p["input"] == 15.0
    assert p["cache_creation"] == 18.75
    assert p["cache_read"] == 1.50
    assert p["output"] == 75.0


def test_pricing_sonnet_3x_cheaper_than_opus_on_input():
    import tools.cost_rollup as cr
    assert cr.PRICING["claude-sonnet-4-6"]["input"] == 3.0
    assert cr.PRICING["claude-opus-4-7"]["input"] / cr.PRICING["claude-sonnet-4-6"]["input"] == 5.0


def test_main_no_args_returns_zero_or_nonzero_no_crash():
    import tools.cost_rollup as cr
    rc = cr.main(argv=[])
    assert isinstance(rc, int)


def test_parse_session_jsonl_canonical_opus():
    from tools.cost_rollup import parse_session_jsonl
    result = parse_session_jsonl(FIXTURE / "session_sample_opus.jsonl")
    assert result is not None
    assert result["session_id"] == "test-session-opus"
    assert result["model"] == "claude-opus-4-7"
    assert result["message_count"] == 3
    # Sum input/output/cache across all 3 messages
    assert result["total_input_tokens"] == 100 + 50 + 30
    assert result["total_output_tokens"] == 50 + 200 + 100
    assert result["total_cache_read_tokens"] == 10000 + 15000 + 15500
    assert result["total_cache_creation_tokens"] == 500
    # Timestamps captured
    assert result["start_ts"] == "2026-05-13T16:30:00.000Z"
    assert result["end_ts"] == "2026-05-13T16:32:00.000Z"


def test_parse_session_jsonl_canonical_sonnet():
    from tools.cost_rollup import parse_session_jsonl
    result = parse_session_jsonl(FIXTURE / "session_sample_sonnet.jsonl")
    assert result is not None
    assert result["model"] == "claude-sonnet-4-6"
    assert result["message_count"] == 2
    assert result["total_input_tokens"] == 300
    assert result["total_cache_creation_tokens"] == 300


def test_parse_session_jsonl_corrupt_lines_skipped():
    """Corrupt lines + non-assistant lines silently skipped. Missing fields default to 0."""
    from tools.cost_rollup import parse_session_jsonl
    result = parse_session_jsonl(FIXTURE / "session_sample_corrupt.jsonl")
    assert result is not None
    # 2 valid assistant messages should be counted
    assert result["message_count"] == 2
    assert result["total_input_tokens"] == 100 + 50
    # cache_read missing in 2nd record -> defaults to 0
    assert result["total_cache_read_tokens"] == 1000
    # cache_creation missing in 2nd -> defaults to 0
    assert result["total_cache_creation_tokens"] == 0


def test_parse_session_jsonl_missing_file_returns_none():
    from tools.cost_rollup import parse_session_jsonl
    result = parse_session_jsonl(FIXTURE / "nonexistent.jsonl")
    assert result is None


def test_parse_session_jsonl_zero_assistant_messages_returns_none():
    """If a jsonl has only user/system messages, no token data — return None."""
    import tempfile
    from tools.cost_rollup import parse_session_jsonl
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write('{"type":"user","timestamp":"2026-05-13T16:30:00.000Z","sessionId":"u-only"}\n')
        path = f.name
    try:
        result = parse_session_jsonl(Path(path))
        assert result is None
    finally:
        Path(path).unlink()


# === compute_cost tests ===

def test_compute_cost_opus_canonical():
    from tools.cost_rollup import compute_cost
    # 1M input + 1M cache_read + 1M output at opus pricing (no cache_creation)
    cost = compute_cost(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=0,
    )
    # $15 (input) + $1.50 (cache_read) + $75 (output) = $91.50
    assert abs(cost - 91.50) < 0.01


def test_compute_cost_sonnet_cheaper_than_opus():
    from tools.cost_rollup import compute_cost
    cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, 1_000_000, 0)
    # $3 (input) + $0.30 (cache_read) + $15 (output) = $18.30
    assert abs(cost - 18.30) < 0.01


def test_compute_cost_haiku_cheapest():
    from tools.cost_rollup import compute_cost
    cost = compute_cost("claude-haiku-4-5", 1_000_000, 1_000_000, 1_000_000, 0)
    # $1 (input) + $0.10 (cache_read) + $5 (output) = $6.10
    assert abs(cost - 6.10) < 0.01


def test_compute_cost_unknown_model_falls_back_to_default():
    """Unknown model uses DEFAULT_PRICING (opus rates) so we never under-bill."""
    from tools.cost_rollup import compute_cost
    cost_unknown = compute_cost("future-model-x", 1_000_000, 0, 0, 0)
    cost_opus = compute_cost("claude-opus-4-7", 1_000_000, 0, 0, 0)
    assert abs(cost_unknown - cost_opus) < 0.01


def test_compute_cost_cache_creation_priced_at_1_25x_input():
    """cache_creation is 1.25× input rate (Anthropic 5-min cache write premium)."""
    from tools.cost_rollup import compute_cost
    cost_creation = compute_cost("claude-opus-4-7", 0, 0, 0, 1_000_000)
    cost_input = compute_cost("claude-opus-4-7", 1_000_000, 0, 0, 0)
    # creation should be 1.25× input
    assert abs(cost_creation - 1.25 * cost_input) < 0.01
    assert abs(cost_creation - 18.75) < 0.01  # 1M × $18.75/M


def test_compute_cost_zero_tokens_returns_zero():
    from tools.cost_rollup import compute_cost
    assert compute_cost("claude-opus-4-7", 0, 0, 0, 0) == 0.0


def test_compute_cost_small_token_counts():
    """Tokens in the hundreds — typical for a single session message."""
    from tools.cost_rollup import compute_cost
    # 100 input + 50 output + 10000 cache_read + 500 cache_creation, opus pricing
    cost = compute_cost("claude-opus-4-7", 100, 50, 10000, 500)
    # input: 100/1M × $15 = $0.0015
    # output: 50/1M × $75 = $0.00375
    # cache_read: 10000/1M × $1.50 = $0.015
    # cache_creation: 500/1M × $18.75 = $0.009375
    # Total = $0.029625
    assert abs(cost - 0.029625) < 0.0001


def test_compute_cost_session_jsonl_integration():
    """End-to-end: parse a fixture, compute cost on the aggregated counts."""
    from tools.cost_rollup import parse_session_jsonl, compute_cost
    result = parse_session_jsonl(FIXTURE / "session_sample_opus.jsonl")
    cost = compute_cost(
        model=result["model"],
        input_tokens=result["total_input_tokens"],
        output_tokens=result["total_output_tokens"],
        cache_read_tokens=result["total_cache_read_tokens"],
        cache_creation_tokens=result["total_cache_creation_tokens"],
    )
    # Fixture totals: input=180, output=350, cache_read=40500, cache_creation=500
    # opus: 180/1M × 15 + 350/1M × 75 + 40500/1M × 1.50 + 500/1M × 18.75
    # = 0.0027 + 0.02625 + 0.06075 + 0.009375 = 0.099075
    assert abs(cost - 0.099075) < 0.0001


# === load_dispatch_telemetry tests ===

def test_load_dispatch_telemetry_canonical():
    from tools.cost_rollup import load_dispatch_telemetry
    records = load_dispatch_telemetry(FIXTURE / "dispatch_telemetry_sample.jsonl")
    assert len(records) == 3
    assert records[0]["role"] == "tooling_engineer"
    assert records[1]["role"] == "pi"
    assert records[2]["role"] == "findings_curator"


def test_load_dispatch_telemetry_missing_file_returns_empty():
    from tools.cost_rollup import load_dispatch_telemetry
    records = load_dispatch_telemetry(FIXTURE / "nonexistent.jsonl")
    assert records == []


def test_load_dispatch_telemetry_skips_corrupt_lines(tmp_path):
    from tools.cost_rollup import load_dispatch_telemetry
    p = tmp_path / "corrupt.jsonl"
    p.write_text(
        '{"ts": "2026-05-13T16:30:00+00:00", "role": "pi"}\n'
        'NOT JSON\n'
        '\n'
        '{"ts": "2026-05-13T16:35:00+00:00", "role": "evaluator"}\n'
    )
    records = load_dispatch_telemetry(p)
    assert len(records) == 2
    assert {r["role"] for r in records} == {"pi", "evaluator"}


# === map_session_to_role tests ===

def test_map_session_to_role_matches_by_proximity():
    """Session starts 15s after a dispatch → match."""
    from tools.cost_rollup import map_session_to_role
    dispatch_records = [
        {"ts": "2026-05-13T16:30:30+00:00", "role": "tooling_engineer"},
    ]
    session = {
        "session_id": "agent-abc123",
        "start_ts": "2026-05-13T16:30:45.000Z",
    }
    assert map_session_to_role(session, dispatch_records) == "tooling_engineer"


def test_map_session_to_role_picks_closest_when_multiple_in_window():
    """CRITICAL: multiple dispatches in ±60s → pick the closest by timestamp,
    NOT the first in iteration order."""
    from tools.cost_rollup import map_session_to_role
    # Three dispatches at 16:30:00, 16:30:50, 16:31:30
    # Session starts at 16:30:45 — closest is 16:30:50 (pi, 5s delta)
    # NOT the first (tooling_engineer, 45s delta) or last (evaluator, 45s delta)
    dispatch_records = [
        {"ts": "2026-05-13T16:30:00+00:00", "role": "tooling_engineer"},
        {"ts": "2026-05-13T16:30:50+00:00", "role": "pi"},
        {"ts": "2026-05-13T16:31:30+00:00", "role": "evaluator"},
    ]
    session = {
        "session_id": "agent-abc",
        "start_ts": "2026-05-13T16:30:45.000Z",
    }
    role = map_session_to_role(session, dispatch_records)
    assert role == "pi", f"expected closest match 'pi', got {role!r}"


def test_map_session_to_role_no_match_far_session():
    """Session outside ±60s window of any dispatch → subagent_unknown_role for agent-prefix."""
    from tools.cost_rollup import map_session_to_role
    dispatch_records = [{"ts": "2026-05-13T16:30:00+00:00", "role": "pi"}]
    session = {"session_id": "agent-xyz", "start_ts": "2026-05-13T17:00:00.000Z"}
    assert map_session_to_role(session, dispatch_records) == "subagent_unknown_role"


def test_map_session_to_role_director_when_no_agent_prefix_and_no_match():
    """Non-agent-prefixed UUIDs → director."""
    from tools.cost_rollup import map_session_to_role
    session = {"session_id": "fd20dc49-c00b-44be-98bd-e9f04650a57a", "start_ts": "2026-05-13T16:30:00.000Z"}
    assert map_session_to_role(session, []) == "director"


def test_map_session_to_role_handles_missing_start_ts():
    """Session with no start_ts → fallback to director or subagent_unknown_role based on prefix."""
    from tools.cost_rollup import map_session_to_role
    assert map_session_to_role({"session_id": "agent-x", "start_ts": None}, []) == "subagent_unknown_role"
    assert map_session_to_role({"session_id": "uuid-here", "start_ts": None}, []) == "director"


def test_map_session_to_role_handles_corrupt_dispatch_timestamps():
    """Dispatch records with malformed ts are skipped, not crashed on."""
    from tools.cost_rollup import map_session_to_role
    dispatch_records = [
        {"ts": "not-a-valid-iso", "role": "broken"},
        {"ts": "2026-05-13T16:30:30+00:00", "role": "tooling_engineer"},
    ]
    session = {"session_id": "agent-abc", "start_ts": "2026-05-13T16:30:45.000Z"}
    role = map_session_to_role(session, dispatch_records)
    assert role == "tooling_engineer"


# === aggregate_by_model tests ===

def test_aggregate_by_model_canonical():
    """Group by model; sum sessions, tokens, cost."""
    from tools.cost_rollup import aggregate_by_model
    enriched = [
        {"model": "claude-opus-4-7", "total_input_tokens": 1000, "total_output_tokens": 100,
         "total_cache_read_tokens": 50000, "total_cache_creation_tokens": 0,
         "cost_usd": 0.10, "role": "director"},
        {"model": "claude-opus-4-7", "total_input_tokens": 500, "total_output_tokens": 50,
         "total_cache_read_tokens": 20000, "total_cache_creation_tokens": 0,
         "cost_usd": 0.05, "role": "pi"},
        {"model": "claude-sonnet-4-6", "total_input_tokens": 2000, "total_output_tokens": 200,
         "total_cache_read_tokens": 30000, "total_cache_creation_tokens": 1000,
         "cost_usd": 0.03, "role": "tooling_engineer"},
    ]
    agg = aggregate_by_model(enriched)
    assert "claude-opus-4-7" in agg
    assert agg["claude-opus-4-7"]["sessions"] == 2
    assert agg["claude-opus-4-7"]["input_tokens"] == 1500
    assert agg["claude-opus-4-7"]["output_tokens"] == 150
    assert agg["claude-opus-4-7"]["cache_read_tokens"] == 70000
    assert agg["claude-opus-4-7"]["cache_creation_tokens"] == 0
    assert abs(agg["claude-opus-4-7"]["cost_usd"] - 0.15) < 0.001
    assert agg["claude-sonnet-4-6"]["sessions"] == 1
    assert agg["claude-sonnet-4-6"]["cache_creation_tokens"] == 1000


def test_aggregate_by_model_empty_input():
    from tools.cost_rollup import aggregate_by_model
    assert aggregate_by_model([]) == {}


def test_aggregate_by_model_handles_unknown_model_label():
    """Sessions with model=None or missing get bucketed under 'unknown'."""
    from tools.cost_rollup import aggregate_by_model
    enriched = [
        {"model": None, "total_input_tokens": 100, "total_output_tokens": 50,
         "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
         "cost_usd": 0.01, "role": "director"},
        {"model": "claude-opus-4-7", "total_input_tokens": 200, "total_output_tokens": 100,
         "total_cache_read_tokens": 0, "total_cache_creation_tokens": 0,
         "cost_usd": 0.02, "role": "director"},
    ]
    agg = aggregate_by_model(enriched)
    assert "unknown" in agg
    assert agg["unknown"]["sessions"] == 1


# === aggregate_by_role tests ===

def test_aggregate_by_role_canonical():
    from tools.cost_rollup import aggregate_by_role
    enriched = [
        {"role": "director", "cost_usd": 1.50, "model": "claude-opus-4-7"},
        {"role": "director", "cost_usd": 0.80, "model": "claude-opus-4-7"},
        {"role": "pi", "cost_usd": 2.50, "model": "claude-opus-4-7"},
        {"role": "tooling_engineer", "cost_usd": 0.50, "model": "claude-sonnet-4-6"},
    ]
    agg = aggregate_by_role(enriched)
    assert agg["director"]["dispatches"] == 2
    assert abs(agg["director"]["cost_usd"] - 2.30) < 0.001
    assert agg["pi"]["dispatches"] == 1
    assert agg["pi"]["model"] == "claude-opus-4-7"
    assert agg["tooling_engineer"]["dispatches"] == 1
    assert agg["tooling_engineer"]["model"] == "claude-sonnet-4-6"


def test_aggregate_by_role_empty_input():
    from tools.cost_rollup import aggregate_by_role
    assert aggregate_by_role([]) == {}


def test_aggregate_by_role_computes_avg_cost_per_dispatch():
    from tools.cost_rollup import aggregate_by_role
    enriched = [
        {"role": "pi", "cost_usd": 1.00, "model": "claude-opus-4-7"},
        {"role": "pi", "cost_usd": 3.00, "model": "claude-opus-4-7"},
    ]
    agg = aggregate_by_role(enriched)
    assert agg["pi"]["dispatches"] == 2
    assert abs(agg["pi"]["cost_usd"] - 4.00) < 0.001
    # Average $ per dispatch
    assert abs(agg["pi"]["avg_cost_per_dispatch"] - 2.00) < 0.001


# === detect_wastage_events tests ===

def test_detect_wastage_events_counts_all_classes():
    from tools.cost_rollup import detect_wastage_events
    dispatch = [
        {"ts": "2026-05-13T16:30:30+00:00", "role": "tooling_engineer", "escalated": True, "verifier_pass": None},
        {"ts": "2026-05-13T16:35:00+00:00", "role": "pi", "escalated": False, "verifier_pass": False},
        {"ts": "2026-05-13T17:00:00+00:00", "role": "evaluator", "escalated": False, "verifier_pass": True},
    ]
    events = detect_wastage_events(
        post_director_path=FIXTURE / "post_director_sample.jsonl",
        queue_telemetry_path=FIXTURE / "queue_telemetry_sample.jsonl",
        dispatch_telemetry=dispatch,
        since_ts="2026-05-13T00:00:00+00:00",
        until_ts="2026-05-13T23:59:59+00:00",
    )
    # silent_death (no error field): 1 (the 16:40 record)
    assert events["silent_death_recoveries"] == 1
    # post_director_errors (error field populated): 1 (the 16:50 record)
    assert events["post_director_errors"] == 1
    # failed_claims (action=fail): 2 (16:35 and 17:00)
    assert events["failed_claims"] == 2
    # escalated_dispatches: 1
    assert events["escalated_dispatches"] == 1
    # verifier_failures: 1
    assert events["verifier_failures"] == 1


def test_detect_wastage_events_respects_date_range():
    """Records outside [since_ts, until_ts] are excluded."""
    from tools.cost_rollup import detect_wastage_events
    # Use a range that EXCLUDES May 13's records but INCLUDES the May 12 record
    events = detect_wastage_events(
        post_director_path=FIXTURE / "post_director_sample.jsonl",
        queue_telemetry_path=FIXTURE / "queue_telemetry_sample.jsonl",
        dispatch_telemetry=[],
        since_ts="2026-05-12T00:00:00+00:00",
        until_ts="2026-05-12T23:59:59+00:00",
    )
    # Only the D-320 May 12 record is in range (no error, no silent_death)
    assert events["silent_death_recoveries"] == 0
    assert events["post_director_errors"] == 0
    # No queue records on May 12
    assert events["failed_claims"] == 0


def test_detect_wastage_events_empty_returns_zeros():
    from tools.cost_rollup import detect_wastage_events
    events = detect_wastage_events(
        post_director_path=FIXTURE / "nonexistent.jsonl",
        queue_telemetry_path=FIXTURE / "nonexistent.jsonl",
        dispatch_telemetry=[],
        since_ts="2026-05-13T00:00:00+00:00",
        until_ts="2026-05-13T23:59:59+00:00",
    )
    assert events["silent_death_recoveries"] == 0
    assert events["post_director_errors"] == 0
    assert events["failed_claims"] == 0
    assert events["escalated_dispatches"] == 0
    assert events["verifier_failures"] == 0
    assert events["holding_loop_count"] == 0


def test_detect_wastage_events_holding_loop_from_error_burst():
    """3+ post_director error records within range → holding_loop_count ≥ 1."""
    import tempfile
    from tools.cost_rollup import detect_wastage_events
    # Synthesize a burst of 6 error records → 2 holding loops (6 / 3)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(6):
            f.write('{"ts":"2026-05-13T16:%02d:00+00:00","branch_taken":"structured_success","session_id":"D-X","error":"E"}\n' % (10 + i * 5))
        path = f.name
    try:
        events = detect_wastage_events(
            post_director_path=Path(path),
            queue_telemetry_path=FIXTURE / "nonexistent.jsonl",
            dispatch_telemetry=[],
            since_ts="2026-05-13T00:00:00+00:00",
            until_ts="2026-05-13T23:59:59+00:00",
        )
        assert events["post_director_errors"] == 6
        # 6 errors / 3 per loop = 2 holding loops
        assert events["holding_loop_count"] == 2
    finally:
        Path(path).unlink()


# === find_outliers tests (MAD-based) ===

def test_find_outliers_uses_MAD_robust_to_extreme_values():
    """One extreme outlier should be flagged. MAD is robust where mean+stddev fails."""
    from tools.cost_rollup import find_outliers
    enriched = [
        {"session_id": "s1", "cost_usd": 0.10, "role": "director"},
        {"session_id": "s2", "cost_usd": 0.12, "role": "director"},
        {"session_id": "s3", "cost_usd": 0.11, "role": "director"},
        {"session_id": "s4", "cost_usd": 0.15, "role": "director"},
        {"session_id": "s5", "cost_usd": 5.00, "role": "measurement_theorist"},  # ~40× median
    ]
    outliers = find_outliers(enriched, mad_threshold=3.5)
    assert len(outliers) == 1
    assert outliers[0]["session_id"] == "s5"


def test_find_outliers_returns_empty_when_no_extremes():
    """Tight cluster → no outliers."""
    from tools.cost_rollup import find_outliers
    enriched = [
        {"session_id": f"s{i}", "cost_usd": 0.10 + i * 0.01, "role": "director"}
        for i in range(10)
    ]
    outliers = find_outliers(enriched, mad_threshold=3.5)
    assert outliers == []


def test_find_outliers_handles_lt_3_sessions():
    """With <3 sessions, MAD can't establish a baseline → return empty."""
    from tools.cost_rollup import find_outliers
    assert find_outliers([], mad_threshold=3.5) == []
    assert find_outliers([{"session_id": "s1", "cost_usd": 1.0, "role": "director"}],
                          mad_threshold=3.5) == []
    assert find_outliers(
        [{"session_id": "s1", "cost_usd": 1.0, "role": "director"},
         {"session_id": "s2", "cost_usd": 2.0, "role": "director"}],
        mad_threshold=3.5,
    ) == []


def test_find_outliers_zero_mad_does_not_crash():
    """All sessions have identical cost → MAD is 0; should not divide by zero."""
    from tools.cost_rollup import find_outliers
    enriched = [
        {"session_id": f"s{i}", "cost_usd": 1.0, "role": "director"}
        for i in range(5)
    ]
    outliers = find_outliers(enriched, mad_threshold=3.5)
    # No outlier — all identical, MAD == 0 is a degenerate case
    assert outliers == []


def test_find_outliers_sorted_by_cost_descending():
    """Multiple outliers returned sorted by cost descending."""
    from tools.cost_rollup import find_outliers
    enriched = [
        {"session_id": "tight1", "cost_usd": 0.10, "role": "director"},
        {"session_id": "tight2", "cost_usd": 0.11, "role": "director"},
        {"session_id": "tight3", "cost_usd": 0.10, "role": "director"},
        {"session_id": "tight4", "cost_usd": 0.12, "role": "director"},
        {"session_id": "high1", "cost_usd": 8.00, "role": "pi"},  # outlier
        {"session_id": "high2", "cost_usd": 5.00, "role": "measurement_theorist"},  # outlier
    ]
    outliers = find_outliers(enriched, mad_threshold=3.5)
    assert len(outliers) == 2
    # Sorted descending by cost
    assert outliers[0]["session_id"] == "high1"
    assert outliers[1]["session_id"] == "high2"


# === compute_week_delta tests ===

def test_compute_week_delta_basic():
    from tools.cost_rollup import compute_week_delta
    current = {
        "total_cost_usd": 50.0,
        "session_count": 30,
        "wastage_events": {"silent_death_recoveries": 0, "post_director_errors": 0,
                           "failed_claims": 2, "escalated_dispatches": 1,
                           "verifier_failures": 0, "holding_loop_count": 0},
    }
    previous = {
        "total_cost_usd": 80.0,
        "session_count": 40,
        "wastage_events": {"silent_death_recoveries": 3, "post_director_errors": 5,
                           "failed_claims": 5, "escalated_dispatches": 2,
                           "verifier_failures": 1, "holding_loop_count": 0},
    }
    delta = compute_week_delta(current, previous)
    # Cost: 50/80 - 1 = -37.5%
    assert abs(delta["cost_pct_change"] - (-37.5)) < 0.5
    # Sessions: 30/40 - 1 = -25%
    assert abs(delta["session_pct_change"] - (-25.0)) < 0.5
    # Wastage event sum: 3 (current) vs 16 (previous) → fewer
    assert delta["wastage_event_count_change"] < 0


def test_compute_week_delta_handles_no_previous():
    """Previous week None or empty → return empty dict (no delta computable)."""
    from tools.cost_rollup import compute_week_delta
    assert compute_week_delta({"total_cost_usd": 50.0}, None) == {}
    assert compute_week_delta({"total_cost_usd": 50.0}, {}) == {}


def test_compute_week_delta_handles_zero_previous_cost():
    """Previous cost 0 → cost_pct_change should be inf or handled gracefully."""
    from tools.cost_rollup import compute_week_delta
    delta = compute_week_delta(
        {"total_cost_usd": 50.0, "session_count": 10,
         "wastage_events": {"silent_death_recoveries": 0, "post_director_errors": 0,
                            "failed_claims": 0, "escalated_dispatches": 0,
                            "verifier_failures": 0, "holding_loop_count": 0}},
        {"total_cost_usd": 0.0, "session_count": 0,
         "wastage_events": {"silent_death_recoveries": 0, "post_director_errors": 0,
                            "failed_claims": 0, "escalated_dispatches": 0,
                            "verifier_failures": 0, "holding_loop_count": 0}},
    )
    # Either inf-marker or None — both acceptable; just shouldn't crash
    assert "cost_pct_change" in delta


# === render_markdown tests ===

def _sample_summary():
    """Reusable canonical summary dict for renderer tests."""
    return {
        "week": "2026-W20",
        "date_range": ["2026-05-11", "2026-05-17"],
        "total_cost_usd": 47.32,
        "total_input_tokens": 150_000,
        "total_output_tokens": 8_000,
        "total_cache_read_tokens": 12_000_000,
        "session_count": 42,
        "by_model": {
            "claude-opus-4-7": {"sessions": 12, "cost_usd": 38.50,
                                "input_tokens": 100_000, "output_tokens": 5_000,
                                "cache_read_tokens": 8_000_000, "cache_creation_tokens": 1000},
            "claude-sonnet-4-6": {"sessions": 25, "cost_usd": 6.20,
                                  "input_tokens": 50_000, "output_tokens": 3_000,
                                  "cache_read_tokens": 4_000_000, "cache_creation_tokens": 0},
        },
        "by_role": {
            "director": {"dispatches": 8, "cost_usd": 14.20,
                         "model": "claude-opus-4-7", "avg_cost_per_dispatch": 1.775},
            "measurement_theorist": {"dispatches": 1, "cost_usd": 12.00,
                                     "model": "claude-opus-4-7", "avg_cost_per_dispatch": 12.00},
            "tooling_engineer": {"dispatches": 5, "cost_usd": 2.50,
                                 "model": "claude-sonnet-4-6", "avg_cost_per_dispatch": 0.50},
        },
        "wastage_events": {
            "silent_death_recoveries": 0,
            "post_director_errors": 0,
            "failed_claims": 1,
            "escalated_dispatches": 2,
            "verifier_failures": 0,
            "holding_loop_count": 0,
        },
        "outlier_sessions": [
            {"session_id": "D-324", "role": "measurement_theorist", "cost_usd": 12.00},
        ],
        "delta_vs_previous_week": {
            "cost_pct_change": -22.5,
            "session_pct_change": 5.0,
            "wastage_event_count_change": -13,
        },
    }


def test_render_markdown_has_all_sections():
    from tools.cost_rollup import render_markdown
    md = render_markdown(_sample_summary())

    # Title with week + date range
    assert "2026-W20" in md
    assert "2026-05-11" in md and "2026-05-17" in md

    # Total + sessions
    assert "47.32" in md
    assert "42" in md

    # Δ vs prev week in header
    assert "-22.5" in md or "−22.5" in md

    # Sections
    assert "## By model" in md
    assert "## By role" in md
    assert "## Wastage events" in md
    assert "## Outliers" in md or "## Outlier sessions" in md
    assert "## Δ vs" in md or "## Delta vs" in md

    # Per-model rows
    assert "claude-opus-4-7" in md or "opus-4-7" in md
    assert "claude-sonnet-4-6" in md or "sonnet-4-6" in md
    assert "38.50" in md
    assert "6.20" in md or "6.2" in md

    # Per-role rows
    assert "director" in md
    assert "measurement_theorist" in md
    assert "tooling_engineer" in md
    assert "14.20" in md or "14.2" in md
    assert "1.78" in md  # avg_cost_per_dispatch for director

    # Wastage event labels
    assert "silent_death_recoveries" in md
    assert "post_director_errors" in md
    assert "failed_claims" in md

    # Outlier rendered
    assert "D-324" in md
    assert "12.00" in md or "12.0" in md

    # Footer cite
    assert "tools/cost_rollup.py" in md


def test_render_markdown_empty_summary_no_crash():
    """Empty summary (0 sessions) should still produce valid markdown — not crash."""
    from tools.cost_rollup import render_markdown
    empty = {
        "week": "2026-W20",
        "date_range": ["2026-05-11", "2026-05-17"],
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_read_tokens": 0,
        "session_count": 0,
        "by_model": {},
        "by_role": {},
        "wastage_events": {
            "silent_death_recoveries": 0, "post_director_errors": 0,
            "failed_claims": 0, "escalated_dispatches": 0,
            "verifier_failures": 0, "holding_loop_count": 0,
        },
        "outlier_sessions": [],
        "delta_vs_previous_week": {},
    }
    md = render_markdown(empty)
    assert isinstance(md, str)
    assert len(md) > 0
    assert "2026-W20" in md
    # Some signal that the period was empty
    assert ("No sessions" in md
            or "0 sessions" in md.lower()
            or "0.00" in md)


def test_render_markdown_under_5kb():
    """For realistic data, markdown stays under 5 KB."""
    from tools.cost_rollup import render_markdown
    md = render_markdown(_sample_summary())
    assert len(md) < 5000, f"markdown is {len(md)} bytes, expected <5000"


def test_render_markdown_handles_no_outliers():
    """No outliers → 'None.' or similar fallback in the outliers section."""
    from tools.cost_rollup import render_markdown
    summary = _sample_summary()
    summary["outlier_sessions"] = []
    md = render_markdown(summary)
    # Section header still present
    assert "## Outlier" in md
    # Some "none" indicator
    assert ("None" in md or "no outliers" in md.lower())


def test_render_markdown_handles_no_previous_week():
    """delta_vs_previous_week empty/None → no delta in header but valid output."""
    from tools.cost_rollup import render_markdown
    summary = _sample_summary()
    summary["delta_vs_previous_week"] = {}
    md = render_markdown(summary)
    # Should not crash; header shouldn't reference percent change
    assert isinstance(md, str)
    assert "2026-W20" in md


# === main() end-to-end tests ===

def _make_test_repo(tmp_path):
    """Build a sandbox repo + fake project jsonls dir."""
    (tmp_path / "data/infra").mkdir(parents=True)
    proj = tmp_path / "fake_project_jsonls"
    proj.mkdir()
    return tmp_path, proj


def test_main_writes_md_and_json(tmp_path, monkeypatch):
    """End-to-end: 2 session jsonls → both output files exist with valid content."""
    import tools.cost_rollup as cr
    import json as _json
    repo, proj = _make_test_repo(tmp_path)
    monkeypatch.setattr(cr, "REPO", repo)
    monkeypatch.setattr(cr, "PROJECT_JSONLS_DIR", proj)

    # Copy fixtures into the fake project
    import shutil
    shutil.copy(FIXTURE / "session_sample_opus.jsonl", proj / "test-session-opus.jsonl")
    shutil.copy(FIXTURE / "session_sample_sonnet.jsonl", proj / "test-session-sonnet.jsonl")

    rc = cr.main(argv=["--since", "2026-05-13", "--until", "2026-05-14"])
    assert rc == 0

    md_path = repo / "data/infra/cost_rollup.md"
    json_path = repo / "data/infra/cost_rollup.json"
    assert md_path.exists()
    assert json_path.exists()

    summary = _json.loads(json_path.read_text())
    assert summary["schema_version"] == "1.0"
    assert summary["session_count"] == 2  # opus + sonnet fixture
    assert "claude-opus-4-7" in summary["by_model"]
    assert "claude-sonnet-4-6" in summary["by_model"]


def test_main_atomic_write_no_tmp_residue(tmp_path, monkeypatch):
    """After main() exits, no .tmp files should remain."""
    import tools.cost_rollup as cr
    repo, proj = _make_test_repo(tmp_path)
    monkeypatch.setattr(cr, "REPO", repo)
    monkeypatch.setattr(cr, "PROJECT_JSONLS_DIR", proj)
    cr.main(argv=["--since", "2026-05-13", "--until", "2026-05-14"])
    # No .tmp files remaining
    assert not (repo / "data/infra/cost_rollup.md.tmp").exists()
    assert not (repo / "data/infra/cost_rollup.json.tmp").exists()
    # But final files do exist
    assert (repo / "data/infra/cost_rollup.md").exists()
    assert (repo / "data/infra/cost_rollup.json").exists()


def test_main_zero_sessions_in_range_produces_empty_rollup(tmp_path, monkeypatch):
    """Empty range → valid empty rollup, not crash."""
    import tools.cost_rollup as cr
    import json as _json
    repo, proj = _make_test_repo(tmp_path)
    monkeypatch.setattr(cr, "REPO", repo)
    monkeypatch.setattr(cr, "PROJECT_JSONLS_DIR", proj)
    # Date range with no sessions (1990 range)
    rc = cr.main(argv=["--since", "1990-01-01", "--until", "1990-01-02"])
    assert rc == 0
    summary = _json.loads((repo / "data/infra/cost_rollup.json").read_text())
    assert summary["total_cost_usd"] == 0.0
    assert summary["session_count"] == 0
    assert summary["by_model"] == {}
    assert summary["by_role"] == {}
    md = (repo / "data/infra/cost_rollup.md").read_text()
    assert "No sessions in range" in md or "0 sessions" in md.lower() or "$0.00" in md


def test_main_cross_aggregate_invariant(tmp_path, monkeypatch):
    """CRITICAL: total_cost_usd must equal sum(by_model[*].cost_usd)
    AND sum(by_role[*].cost_usd) within 0.01."""
    import tools.cost_rollup as cr
    import json as _json
    repo, proj = _make_test_repo(tmp_path)
    monkeypatch.setattr(cr, "REPO", repo)
    monkeypatch.setattr(cr, "PROJECT_JSONLS_DIR", proj)

    import shutil
    shutil.copy(FIXTURE / "session_sample_opus.jsonl", proj / "test-session-opus.jsonl")
    shutil.copy(FIXTURE / "session_sample_sonnet.jsonl", proj / "test-session-sonnet.jsonl")

    cr.main(argv=["--since", "2026-05-13", "--until", "2026-05-14"])
    summary = _json.loads((repo / "data/infra/cost_rollup.json").read_text())

    total = summary["total_cost_usd"]
    by_model_sum = sum(b["cost_usd"] for b in summary["by_model"].values())
    by_role_sum = sum(b["cost_usd"] for b in summary["by_role"].values())
    assert abs(total - by_model_sum) < 0.01, \
        f"by_model sum {by_model_sum} != total {total}"
    assert abs(total - by_role_sum) < 0.01, \
        f"by_role sum {by_role_sum} != total {total}"


def test_main_handles_corrupt_jsonl_in_proj_dir(tmp_path, monkeypatch):
    """A garbage jsonl in the project dir should be silently skipped, not crash main."""
    import tools.cost_rollup as cr
    repo, proj = _make_test_repo(tmp_path)
    monkeypatch.setattr(cr, "REPO", repo)
    monkeypatch.setattr(cr, "PROJECT_JSONLS_DIR", proj)

    # Add a valid + a corrupt jsonl
    import shutil
    shutil.copy(FIXTURE / "session_sample_opus.jsonl", proj / "valid.jsonl")
    (proj / "corrupt.jsonl").write_text("THIS IS NOT JSON\nGARBAGE\n")

    rc = cr.main(argv=["--since", "2026-05-13", "--until", "2026-05-14"])
    assert rc == 0


# === TODO-C5-1: last non-synthetic model ===

def test_parse_session_jsonl_prefers_last_real_model_over_synthetic(tmp_path):
    """A session whose last message is <synthetic> should still report the
    last REAL (non-synthetic) model in `out['model']`."""
    from tools.cost_rollup import parse_session_jsonl
    p = tmp_path / "real_then_synthetic.jsonl"
    p.write_text(
        '{"type":"assistant","timestamp":"2026-05-13T16:30:00.000Z","sessionId":"s1","message":{"model":"claude-opus-4-7","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":1000,"cache_creation_input_tokens":0}}}\n'
        '{"type":"assistant","timestamp":"2026-05-13T16:30:30.000Z","sessionId":"s1","message":{"model":"claude-opus-4-7","usage":{"input_tokens":50,"output_tokens":25,"cache_read_input_tokens":500,"cache_creation_input_tokens":0}}}\n'
        '{"type":"assistant","timestamp":"2026-05-13T16:31:00.000Z","sessionId":"s1","message":{"model":"<synthetic>","usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}\n'
    )
    result = parse_session_jsonl(p)
    # Should be opus-4-7, not <synthetic>
    assert result["model"] == "claude-opus-4-7", f"got {result['model']!r}"


def test_parse_session_jsonl_returns_synthetic_when_only_model_seen(tmp_path):
    """Degenerate case: the only model in the file is <synthetic> → return it."""
    from tools.cost_rollup import parse_session_jsonl
    p = tmp_path / "synthetic_only.jsonl"
    p.write_text(
        '{"type":"assistant","timestamp":"2026-05-13T16:30:00.000Z","sessionId":"s2","message":{"model":"<synthetic>","usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}\n'
    )
    result = parse_session_jsonl(p)
    assert result["model"] == "<synthetic>"


def test_parse_session_jsonl_skips_synthetic_in_middle(tmp_path):
    """If <synthetic> appears mid-stream, ignore it and keep the real model."""
    from tools.cost_rollup import parse_session_jsonl
    p = tmp_path / "synthetic_middle.jsonl"
    p.write_text(
        '{"type":"assistant","timestamp":"2026-05-13T16:30:00.000Z","sessionId":"s3","message":{"model":"claude-sonnet-4-6","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":1000,"cache_creation_input_tokens":0}}}\n'
        '{"type":"assistant","timestamp":"2026-05-13T16:30:30.000Z","sessionId":"s3","message":{"model":"<synthetic>","usage":{"input_tokens":0,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}\n'
        '{"type":"assistant","timestamp":"2026-05-13T16:31:00.000Z","sessionId":"s3","message":{"model":"claude-opus-4-7","usage":{"input_tokens":50,"output_tokens":25,"cache_read_input_tokens":500,"cache_creation_input_tokens":0}}}\n'
    )
    result = parse_session_jsonl(p)
    # Last real model is opus-4-7 (the synthetic in the middle is ignored)
    assert result["model"] == "claude-opus-4-7"


# === TODO-C5-2: dedup outliers by session_id ===

def test_find_outliers_dedups_by_session_id_summing_costs():
    """Same session_id appearing in multiple enriched entries → one outlier with summed cost."""
    from tools.cost_rollup import find_outliers
    enriched = [
        {"session_id": "tight1", "cost_usd": 0.10, "role": "director"},
        {"session_id": "tight2", "cost_usd": 0.11, "role": "director"},
        {"session_id": "tight3", "cost_usd": 0.10, "role": "director"},
        {"session_id": "tight4", "cost_usd": 0.12, "role": "director"},
        # Same session_id appears 3 times — represents parent + 2 subagent jsonls
        {"session_id": "fd20dc49", "cost_usd": 3.00, "role": "director"},
        {"session_id": "fd20dc49", "cost_usd": 2.50, "role": "director"},
        {"session_id": "fd20dc49", "cost_usd": 1.50, "role": "director"},
    ]
    outliers = find_outliers(enriched, mad_threshold=3.5)
    # Should be 1 deduplicated outlier with cost_usd = 3.00 + 2.50 + 1.50 = 7.00
    fd_outliers = [o for o in outliers if o["session_id"] == "fd20dc49"]
    assert len(fd_outliers) == 1
    assert abs(fd_outliers[0]["cost_usd"] - 7.00) < 0.01


def test_find_outliers_no_dedup_when_unique_session_ids():
    """Unique session_ids → no dedup, normal outlier flagging."""
    from tools.cost_rollup import find_outliers
    enriched = [
        {"session_id": "tight1", "cost_usd": 0.10, "role": "director"},
        {"session_id": "tight2", "cost_usd": 0.11, "role": "director"},
        {"session_id": "tight3", "cost_usd": 0.10, "role": "director"},
        {"session_id": "tight4", "cost_usd": 0.12, "role": "director"},
        {"session_id": "outlier1", "cost_usd": 5.00, "role": "pi"},
        {"session_id": "outlier2", "cost_usd": 3.00, "role": "measurement_theorist"},
    ]
    outliers = find_outliers(enriched, mad_threshold=3.5)
    session_ids = {o["session_id"] for o in outliers}
    assert "outlier1" in session_ids
    assert "outlier2" in session_ids
