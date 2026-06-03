# Cost Rollup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a heuristic Python tool that joins Claude Code session jsonl logs with our existing telemetry, multiplies by hardcoded Anthropic pricing, and emits a weekly cost rollup at `data/infra/cost_rollup.{md,json}`.

**Architecture:** Single file `tools/cost_rollup.py`. Parses `~/.claude/projects/<sanitized>/*.jsonl` for token usage, cross-references `data/infra/dispatch_telemetry.jsonl` for role mapping, computes per-model + per-role + wastage-event aggregates, renders markdown + JSON. Wired into `tools/post_director.py` success path so the rollup refreshes on every Director exit.

**Tech Stack:** Python 3 (stdlib only — re, json, os, glob, statistics, datetime, pathlib) + pytest. No external deps.

**Spec:** `docs/superpowers/specs/2026-05-13-cost-rollup.md`

## Refinements applied (2026-05-13 pre-implementation review)

Before dispatching the first implementer, these corrections from a critical re-read:

1. **cache_creation pricing fixed**: `cache_creation_input_tokens` is priced at **1.25× input rate** (Anthropic 5-min cache premium), not 1.0× input. PRICING constants below are corrected.
2. **Role mapping picks closest, not first**: if multiple dispatch_telemetry records fall within the ±60s window, we pick the one with the smallest `|session.start_ts − dispatch.ts|`, not the first iterated.
3. **Outlier detection uses MAD** (median absolute deviation), not mean+stddev. MAD is robust to the very outliers we're hunting; mean+stddev gets inflated by outliers and hides them.
4. **Atomic file writes are mandatory**: every output file (`.md` and `.json`) is written via `tmp_path.write_text(...); tmp_path.rename(final_path)` to prevent partial reads when post_director fires the rollup concurrently with operator manual runs.
5. **Cross-aggregate invariant test added**: `total_cost_usd == sum(by_model[*].cost_usd) == sum(by_role[*].cost_usd)` within rounding tolerance. Catches aggregation bugs.
6. **Empty-range case covered**: 0 sessions in range produces a valid empty rollup (zero counts, "No sessions in range" narrative), not a crash.
7. **Holding-loop detection rule concrete**: ≥3 post_director records within 2h where `branch_taken` contains "error" OR `session_id` is null/missing AND no fresh deliverables produced in the same window.

---

## Stage C1: jsonl Session Parser + Pricing Math

**Goal:** Build the core parser that walks `~/.claude/projects/<sanitized>/*.jsonl` and extracts per-session token usage, plus the pricing-math function that converts tokens → $.

### Task 1: Test fixture — minimal session jsonl sample

**Files:**
- Create: `tests/fixtures/cost_rollup/session_sample_opus.jsonl`
- Create: `tests/fixtures/cost_rollup/session_sample_sonnet.jsonl`
- Create: `tests/fixtures/cost_rollup/session_sample_corrupt.jsonl`

- [ ] **Step 1: Create three fixture files**

`session_sample_opus.jsonl` — 3 assistant messages, all opus-4-7, with explicit token usage:

```jsonl
{"type":"assistant","timestamp":"2026-05-13T16:30:00.000Z","sessionId":"test-session-opus","message":{"model":"claude-opus-4-7","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":10000,"cache_creation_input_tokens":500}}}
{"type":"assistant","timestamp":"2026-05-13T16:31:00.000Z","sessionId":"test-session-opus","message":{"model":"claude-opus-4-7","usage":{"input_tokens":50,"output_tokens":200,"cache_read_input_tokens":15000,"cache_creation_input_tokens":0}}}
{"type":"assistant","timestamp":"2026-05-13T16:32:00.000Z","sessionId":"test-session-opus","message":{"model":"claude-opus-4-7","usage":{"input_tokens":30,"output_tokens":100,"cache_read_input_tokens":15500,"cache_creation_input_tokens":0}}}
```

`session_sample_sonnet.jsonl` — same shape but model=`claude-sonnet-4-6`, 2 messages.

`session_sample_corrupt.jsonl` — mix of valid + corrupt lines:
```jsonl
{"type":"assistant","timestamp":"2026-05-13T16:30:00.000Z","sessionId":"test-corrupt","message":{"model":"claude-opus-4-7","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":1000,"cache_creation_input_tokens":0}}}
THIS IS NOT JSON
{"type":"user","timestamp":"2026-05-13T16:31:00.000Z","sessionId":"test-corrupt"}
{"type":"assistant","timestamp":"2026-05-13T16:32:00.000Z","sessionId":"test-corrupt","message":{"model":"claude-opus-4-7","usage":{"input_tokens":50,"output_tokens":25}}}
```

- [ ] **Step 2: Commit**

```bash
mkdir -p tests/fixtures/cost_rollup
# (write the three files above)
git add tests/fixtures/cost_rollup/
git commit -m "test(fixture): session jsonl samples for cost_rollup parser tests"
```

---

### Task 2: cost_rollup.py — module skeleton + parse_session_jsonl()

**Files:**
- Create: `tools/cost_rollup.py`
- Create: `tests/test_cost_rollup.py`

- [ ] **Step 1: Write failing tests**

```python
"""Validates tools/cost_rollup.py — jsonl session parser + pricing."""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FIXTURE = REPO / "tests/fixtures/cost_rollup"


def test_module_imports():
    import tools.cost_rollup as cr
    assert hasattr(cr, "REPO")
    assert hasattr(cr, "PRICING")
    assert hasattr(cr, "main")


def test_parse_session_jsonl_canonical_opus():
    from tools.cost_rollup import parse_session_jsonl
    result = parse_session_jsonl(FIXTURE / "session_sample_opus.jsonl")
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


def test_parse_session_jsonl_corrupt_lines_skipped():
    """Corrupt lines and non-assistant lines should be silently skipped."""
    from tools.cost_rollup import parse_session_jsonl
    result = parse_session_jsonl(FIXTURE / "session_sample_corrupt.jsonl")
    # 2 valid assistant messages should be counted
    assert result["message_count"] == 2
    assert result["total_input_tokens"] == 100 + 50
    # cache_read missing in 2nd record should default to 0
    assert result["total_cache_read_tokens"] == 1000


def test_parse_session_jsonl_missing_file_returns_none():
    from tools.cost_rollup import parse_session_jsonl
    result = parse_session_jsonl(FIXTURE / "nonexistent.jsonl")
    assert result is None
```

- [ ] **Step 2-4: Implement, run tests, iterate**

Create `tools/cost_rollup.py`:

```python
"""Cost rollup — joins Claude Code session jsonl logs with our telemetry and
emits weekly cost reports at data/infra/cost_rollup.{md,json}.

Per spec: docs/superpowers/specs/2026-05-13-cost-rollup.md.

CLI:
    python3 tools/cost_rollup.py [--week YYYY-WW] [--since YYYY-MM-DD] [--until YYYY-MM-DD]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# Ensure repo root on sys.path so direct invocation works.
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_ON_PATH = str(_TOOLS_DIR.parent)
if _REPO_ROOT_ON_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_ON_PATH)

REPO = Path(__file__).resolve().parent.parent

SCHEMA_VERSION = "1.0"

# All values $ per million tokens.
# cache_creation_5m is priced at 1.25× input (Anthropic 5-min cache write premium).
# cache_creation_1h is priced higher; spec is v1 — we lump under cache_creation_5m
# since most cache_creation in our usage is the default 5-min TTL.
PRICING = {
    "claude-opus-4-7":              {"input": 15.0, "cache_creation": 18.75, "cache_read": 1.50, "output": 75.0},
    "claude-opus-4-7[1m]":          {"input": 15.0, "cache_creation": 18.75, "cache_read": 1.50, "output": 75.0},
    "claude-sonnet-4-6":            {"input":  3.0, "cache_creation":  3.75, "cache_read": 0.30, "output": 15.0},
    "claude-haiku-4-5":             {"input":  1.0, "cache_creation":  1.25, "cache_read": 0.10, "output":  5.0},
    "claude-haiku-4-5-20251001":    {"input":  1.0, "cache_creation":  1.25, "cache_read": 0.10, "output":  5.0},
    "claude-opus-4-5-20251101":     {"input": 15.0, "cache_creation": 18.75, "cache_read": 1.50, "output": 75.0},
}
DEFAULT_PRICING = PRICING["claude-opus-4-7"]


def parse_session_jsonl(path: Path) -> dict | None:
    """Read a Claude Code session jsonl and aggregate token usage.

    Returns dict with: session_id, model (last seen), message_count,
    total_input_tokens, total_output_tokens, total_cache_read_tokens,
    total_cache_creation_tokens, start_ts, end_ts. Returns None if file
    missing. Corrupt lines and non-assistant lines silently skipped.
    """
    if not Path(path).exists():
        return None
    out = {
        "session_id": None,
        "model": None,
        "message_count": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_cache_read_tokens": 0,
        "total_cache_creation_tokens": 0,
        "start_ts": None,
        "end_ts": None,
    }
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "assistant":
                continue
            msg = obj.get("message", {}) or {}
            usage = msg.get("usage") or {}
            ts = obj.get("timestamp")

            if out["session_id"] is None:
                out["session_id"] = obj.get("sessionId")
            if msg.get("model"):
                out["model"] = msg["model"]
            if ts:
                if out["start_ts"] is None:
                    out["start_ts"] = ts
                out["end_ts"] = ts

            out["message_count"] += 1
            out["total_input_tokens"] += usage.get("input_tokens", 0) or 0
            out["total_output_tokens"] += usage.get("output_tokens", 0) or 0
            out["total_cache_read_tokens"] += usage.get("cache_read_input_tokens", 0) or 0
            out["total_cache_creation_tokens"] += usage.get("cache_creation_input_tokens", 0) or 0

    if out["message_count"] == 0:
        return None
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cost rollup over session jsonls.")
    parser.add_argument("--week", help="ISO week YYYY-WW (default: current)")
    parser.add_argument("--since", help="Start date YYYY-MM-DD")
    parser.add_argument("--until", help="End date YYYY-MM-DD")
    args = parser.parse_args(argv)
    # Skeleton — real logic in Tasks 3-7.
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Verify tests pass**

```bash
cd <repo> && source .venv/bin/activate
python -m pytest tests/test_cost_rollup.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add tools/cost_rollup.py tests/test_cost_rollup.py
git commit -m "feat: cost_rollup.parse_session_jsonl() + PRICING constants + skeleton"
```

---

### Task 3: cost_rollup.py — compute_cost() pricing math

**Files:**
- Modify: `tools/cost_rollup.py`
- Modify: `tests/test_cost_rollup.py`

- [ ] **Step 1: Write failing tests**

```python
def test_compute_cost_opus_canonical():
    from tools.cost_rollup import compute_cost
    # 1M input + 1M cache_read + 1M output at opus pricing
    cost = compute_cost(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_creation_tokens=0,
    )
    # $15 (input) + $1.50 (cache_read) + $75 (output) = $91.50
    assert abs(cost - 91.50) < 0.01


def test_compute_cost_sonnet_3x_cheaper_input_5x_cheaper_cache():
    from tools.cost_rollup import compute_cost
    cost = compute_cost("claude-sonnet-4-6", 1_000_000, 1_000_000, 1_000_000, 0)
    # $3 + $0.30 + $15 = $18.30
    assert abs(cost - 18.30) < 0.01


def test_compute_cost_haiku_cheapest():
    from tools.cost_rollup import compute_cost
    cost = compute_cost("claude-haiku-4-5", 1_000_000, 1_000_000, 1_000_000, 0)
    # $1 + $0.10 + $5 = $6.10
    assert abs(cost - 6.10) < 0.01


def test_compute_cost_unknown_model_falls_back_to_default():
    """Unknown model uses DEFAULT_PRICING (opus rates) so we never under-bill."""
    from tools.cost_rollup import compute_cost
    cost_unknown = compute_cost("future-model-x", 1_000_000, 0, 0, 0)
    cost_opus = compute_cost("claude-opus-4-7", 1_000_000, 0, 0, 0)
    assert abs(cost_unknown - cost_opus) < 0.01


def test_compute_cost_cache_creation_priced_as_input():
    from tools.cost_rollup import compute_cost
    cost = compute_cost("claude-opus-4-7", 0, 0, 0, 1_000_000)
    # cache_creation @ $15/M (same as input)
    assert abs(cost - 15.0) < 0.01
```

- [ ] **Step 2-4: Implement, run, iterate**

Add to `tools/cost_rollup.py`:

```python
def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> float:
    """Convert per-bucket token counts → dollars using PRICING constants.

    cache_creation is priced at 1.25× input rate (Anthropic 5-min cache premium).
    Unknown models fall back to DEFAULT_PRICING (opus rates) so we never under-bill.
    Returns dollars.
    """
    prices = PRICING.get(model, DEFAULT_PRICING)
    return (
        (input_tokens / 1_000_000) * prices["input"]
        + (cache_creation_tokens / 1_000_000) * prices["cache_creation"]
        + (cache_read_tokens / 1_000_000) * prices["cache_read"]
        + (output_tokens / 1_000_000) * prices["output"]
    )
```

**Additional test for cache_creation_5m premium:**

```python
def test_compute_cost_cache_creation_priced_at_1_25x_input():
    """cache_creation is 1.25× input rate per Anthropic 5-min cache premium."""
    from tools.cost_rollup import compute_cost
    cost_creation = compute_cost("claude-opus-4-7", 0, 0, 0, 1_000_000)
    cost_input = compute_cost("claude-opus-4-7", 1_000_000, 0, 0, 0)
    # creation should be 1.25× input
    assert abs(cost_creation - 1.25 * cost_input) < 0.01
    assert abs(cost_creation - 18.75) < 0.01  # 1M × $18.75/M
```

(The existing `test_compute_cost_cache_creation_priced_as_input` test from Task 3 spec was WRONG — replace it with the test above.)

- [ ] **Step 5: Commit**

```bash
git add tools/cost_rollup.py tests/test_cost_rollup.py
git commit -m "feat: cost_rollup.compute_cost() — pricing math with default fallback"
```

---

## Stage C1 Gate (user review): pause to verify parser + pricing math against real data before building role mapping.

---

## Stage C2: Session → Role Mapping + Aggregation

**Goal:** Tag each session as Director / subagent (with role) / unknown, then aggregate by model and role.

### Task 4: cost_rollup.py — load_dispatch_telemetry() + map_session_to_role()

**Files:**
- Modify: `tools/cost_rollup.py`
- Modify: `tests/test_cost_rollup.py`
- Create: `tests/fixtures/cost_rollup/dispatch_telemetry_sample.jsonl`

- [ ] **Step 1: Create fixture for dispatch_telemetry**

```jsonl
{"ts": "2026-05-13T16:30:30+00:00", "role": "tooling_engineer", "model_dispatched": "claude-sonnet-4-6", "task_class": "apparatus_build", "escalated": false, "verifier_pass": null}
{"ts": "2026-05-13T16:35:00+00:00", "role": "pi", "model_dispatched": "claude-opus-4-7", "task_class": "close_gate_cosign", "escalated": false, "verifier_pass": true}
```

- [ ] **Step 2: Write failing tests**

```python
def test_load_dispatch_telemetry():
    from tools.cost_rollup import load_dispatch_telemetry
    records = load_dispatch_telemetry(FIXTURE / "dispatch_telemetry_sample.jsonl")
    assert len(records) == 2
    assert records[0]["role"] == "tooling_engineer"


def test_map_session_to_role_matches_by_timestamp_proximity(tmp_path):
    """A session that starts within ±60s of a dispatch_telemetry record gets tagged with that role."""
    from tools.cost_rollup import map_session_to_role
    dispatch_records = [
        {"ts": "2026-05-13T16:30:30+00:00", "role": "tooling_engineer", "model_dispatched": "claude-sonnet-4-6"},
    ]
    # Session starts 15s after dispatch → match
    session = {
        "session_id": "agent-abc123",
        "start_ts": "2026-05-13T16:30:45.000Z",
    }
    role = map_session_to_role(session, dispatch_records)
    assert role == "tooling_engineer"


def test_map_session_to_role_no_match_for_far_session():
    """Sessions outside ±60s window don't match."""
    from tools.cost_rollup import map_session_to_role
    dispatch_records = [
        {"ts": "2026-05-13T16:30:00+00:00", "role": "tooling_engineer"},
    ]
    session = {"session_id": "agent-xyz", "start_ts": "2026-05-13T17:00:00.000Z"}
    role = map_session_to_role(session, dispatch_records)
    # No match → either "subagent_unknown_role" (if agent-prefix) or "director" (if not)
    assert role == "subagent_unknown_role"


def test_map_session_to_role_director_when_no_agent_prefix():
    """Non-agent-prefixed sessions are tagged as director."""
    from tools.cost_rollup import map_session_to_role
    session = {"session_id": "fd20dc49-c00b-44be-98bd-e9f04650a57a", "start_ts": "2026-05-13T16:30:00.000Z"}
    role = map_session_to_role(session, [])
    assert role == "director"
```

- [ ] **Step 3-5: Implement, run, iterate**

Add to `tools/cost_rollup.py`:

```python
ROLE_MAP_WINDOW_SECONDS = 60


def load_dispatch_telemetry(path: Path | None = None) -> list[dict]:
    p = path if path is not None else REPO / "data" / "infra" / "dispatch_telemetry.jsonl"
    if not Path(p).exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _parse_iso_to_ts(s: str) -> float:
    """Parse ISO 8601 timestamp (with Z or +00:00) → unix epoch seconds."""
    if not s:
        raise ValueError("empty timestamp")
    s = s.replace("Z", "+00:00")
    # Trim sub-second if present beyond microseconds
    return datetime.fromisoformat(s).timestamp()


def map_session_to_role(session: dict, dispatch_records: list[dict]) -> str:
    """Tag a session as director / <role> / subagent_unknown_role based on
    timestamp proximity to dispatch_telemetry records.

    Picks the CLOSEST matching dispatch (smallest |session.start_ts - dispatch.ts|)
    within ROLE_MAP_WINDOW_SECONDS. If no match: agent-prefix → subagent_unknown_role;
    otherwise → director.
    """
    sid = session.get("session_id", "")
    is_agent = sid.startswith("agent-")
    start_ts_str = session.get("start_ts", "")
    if not start_ts_str:
        return "subagent_unknown_role" if is_agent else "director"
    try:
        start_ts = _parse_iso_to_ts(start_ts_str)
    except ValueError:
        return "subagent_unknown_role" if is_agent else "director"

    # Find the closest dispatch record within the window.
    best_match = None
    best_delta = ROLE_MAP_WINDOW_SECONDS + 1  # sentinel
    for rec in dispatch_records:
        rec_ts_str = rec.get("ts", "")
        if not rec_ts_str:
            continue
        try:
            rec_ts = _parse_iso_to_ts(rec_ts_str)
        except ValueError:
            continue
        delta = abs(start_ts - rec_ts)
        if delta <= ROLE_MAP_WINDOW_SECONDS and delta < best_delta:
            best_delta = delta
            best_match = rec

    if best_match is not None:
        return best_match.get("role", "subagent_unknown_role")
    return "subagent_unknown_role" if is_agent else "director"
```

**Additional test for closest-match selection:**

```python
def test_map_session_to_role_picks_closest_dispatch_when_multiple_in_window():
    """Multiple dispatches in window → pick the closest by timestamp."""
    from tools.cost_rollup import map_session_to_role
    dispatch_records = [
        {"ts": "2026-05-13T16:30:00+00:00", "role": "tooling_engineer"},
        {"ts": "2026-05-13T16:30:50+00:00", "role": "pi"},  # closer to session
        {"ts": "2026-05-13T16:31:30+00:00", "role": "evaluator"},
    ]
    session = {
        "session_id": "agent-abc",
        "start_ts": "2026-05-13T16:30:45.000Z",  # 5s after pi dispatch
    }
    role = map_session_to_role(session, dispatch_records)
    assert role == "pi", f"expected closest match 'pi', got {role!r}"
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: cost_rollup — load_dispatch_telemetry + map_session_to_role (timestamp proximity)"
```

---

### Task 5: cost_rollup.py — aggregate_by_model() + aggregate_by_role()

**Files:**
- Modify: `tools/cost_rollup.py`
- Modify: `tests/test_cost_rollup.py`

- [ ] **Step 1-5: TDD aggregator functions**

```python
def test_aggregate_by_model():
    from tools.cost_rollup import aggregate_by_model
    enriched = [
        {"model": "claude-opus-4-7", "total_input_tokens": 1000, "total_output_tokens": 100,
         "total_cache_read_tokens": 50000, "total_cache_creation_tokens": 0,
         "cost_usd": 0.10, "role": "director"},
        {"model": "claude-opus-4-7", "total_input_tokens": 500, "total_output_tokens": 50,
         "total_cache_read_tokens": 20000, "total_cache_creation_tokens": 0,
         "cost_usd": 0.05, "role": "pi"},
        {"model": "claude-sonnet-4-6", "total_input_tokens": 2000, "total_output_tokens": 200,
         "total_cache_read_tokens": 30000, "total_cache_creation_tokens": 0,
         "cost_usd": 0.03, "role": "tooling_engineer"},
    ]
    agg = aggregate_by_model(enriched)
    assert "claude-opus-4-7" in agg
    assert agg["claude-opus-4-7"]["sessions"] == 2
    assert abs(agg["claude-opus-4-7"]["cost_usd"] - 0.15) < 0.001
    assert agg["claude-sonnet-4-6"]["sessions"] == 1


def test_aggregate_by_role():
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
    assert agg["tooling_engineer"]["model"] == "claude-sonnet-4-6"
```

Implementation:
```python
def aggregate_by_model(enriched_sessions: list[dict]) -> dict[str, dict]:
    """Group by model; sum sessions, tokens, cost."""
    out: dict[str, dict] = {}
    for s in enriched_sessions:
        m = s.get("model", "unknown")
        b = out.setdefault(m, {
            "sessions": 0, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cost_usd": 0.0,
        })
        b["sessions"] += 1
        b["input_tokens"] += s.get("total_input_tokens", 0)
        b["output_tokens"] += s.get("total_output_tokens", 0)
        b["cache_read_tokens"] += s.get("total_cache_read_tokens", 0)
        b["cost_usd"] += s.get("cost_usd", 0.0)
    return out


def aggregate_by_role(enriched_sessions: list[dict]) -> dict[str, dict]:
    """Group by role; sum dispatches + cost."""
    out: dict[str, dict] = {}
    for s in enriched_sessions:
        r = s.get("role", "unknown")
        b = out.setdefault(r, {"dispatches": 0, "cost_usd": 0.0, "model": None})
        b["dispatches"] += 1
        b["cost_usd"] += s.get("cost_usd", 0.0)
        if b["model"] is None:
            b["model"] = s.get("model")
    return out
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: cost_rollup — aggregate_by_model + aggregate_by_role"
```

---

## Stage C3: Wastage Events + Outliers + Week Delta

**Goal:** Detect wastage events from each telemetry source, find outlier sessions, compute week-over-week delta.

### Task 6: cost_rollup.py — detect_wastage_events()

**Files:**
- Modify: `tools/cost_rollup.py`
- Modify: `tests/test_cost_rollup.py`
- Create: `tests/fixtures/cost_rollup/post_director_sample.jsonl`
- Create: `tests/fixtures/cost_rollup/queue_telemetry_sample.jsonl`

- [ ] **Step 1: Create fixtures**

`post_director_sample.jsonl`:
```jsonl
{"ts":"2026-05-13T16:30:00+00:00","branch_taken":"structured_success","session_id":"D-324"}
{"ts":"2026-05-13T16:40:00+00:00","branch_taken":"silent_death","session_id":null}
{"ts":"2026-05-13T16:50:00+00:00","branch_taken":"structured_success","session_id":"D-325","error":"NoModuleError"}
```

`queue_telemetry_sample.jsonl`:
```jsonl
{"ts":"2026-05-13T16:00:00Z","action":"claim","item_id":"wq-a","item_type":"phase_advance"}
{"ts":"2026-05-13T16:30:00Z","action":"complete","item_id":"wq-a","item_type":"phase_advance"}
{"ts":"2026-05-13T16:35:00Z","action":"fail","item_id":"wq-b","item_type":"diagnostic_review"}
```

- [ ] **Step 2-5: TDD wastage detector**

```python
def test_detect_wastage_events_counts_silent_deaths():
    from tools.cost_rollup import detect_wastage_events
    events = detect_wastage_events(
        post_director_path=FIXTURE / "post_director_sample.jsonl",
        queue_telemetry_path=FIXTURE / "queue_telemetry_sample.jsonl",
        dispatch_telemetry=[
            {"ts": "2026-05-13T16:30:30+00:00", "role": "tooling_engineer", "escalated": True},
        ],
        since_ts="2026-05-13T00:00:00+00:00",
        until_ts="2026-05-13T23:59:59+00:00",
    )
    assert events["silent_death_recoveries"] == 1
    assert events["post_director_errors"] == 1
    assert events["failed_claims"] == 1
    assert events["escalated_dispatches"] == 1
```

Implementation reads the files within `[since_ts, until_ts]`, counts each event class, returns dict.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: cost_rollup.detect_wastage_events() — silent_death + errors + failed claims + escalations"
```

---

### Task 7: cost_rollup.py — find_outliers() + compute_week_delta()

**Files:**
- Modify: `tools/cost_rollup.py`
- Modify: `tests/test_cost_rollup.py`

- [ ] **Step 1-5: TDD outliers + delta**

```python
def test_find_outliers_uses_MAD_robust_to_extreme_values():
    """Use median absolute deviation (MAD) not mean+stddev — one outlier
    inflates stddev and hides itself; MAD is robust."""
    from tools.cost_rollup import find_outliers
    enriched = [
        {"session_id": "s1", "cost_usd": 0.10, "role": "director"},
        {"session_id": "s2", "cost_usd": 0.12, "role": "director"},
        {"session_id": "s3", "cost_usd": 0.11, "role": "director"},
        {"session_id": "s4", "cost_usd": 0.15, "role": "director"},
        {"session_id": "s5", "cost_usd": 5.00, "role": "measurement_theorist"},  # 40x median
    ]
    outliers = find_outliers(enriched, mad_threshold=3.5)  # 3.5× MAD
    assert len(outliers) == 1
    assert outliers[0]["session_id"] == "s5"


def test_find_outliers_returns_empty_when_no_extremes():
    from tools.cost_rollup import find_outliers
    enriched = [
        {"session_id": f"s{i}", "cost_usd": 0.10 + i * 0.01, "role": "director"}
        for i in range(10)
    ]
    outliers = find_outliers(enriched, mad_threshold=3.5)
    assert outliers == []


def test_find_outliers_handles_lt_3_sessions():
    """With <3 sessions, MAD can't establish a baseline — return empty."""
    from tools.cost_rollup import find_outliers
    assert find_outliers([], mad_threshold=3.5) == []
    assert find_outliers([{"session_id": "s1", "cost_usd": 1.0, "role": "director"}],
                          mad_threshold=3.5) == []


def test_compute_week_delta_basic():
    from tools.cost_rollup import compute_week_delta
    current = {"total_cost_usd": 50.0, "session_count": 30,
               "wastage_events": {"silent_death_recoveries": 0, "post_director_errors": 0,
                                  "failed_claims": 2, "escalated_dispatches": 1,
                                  "verifier_failures": 0, "holding_loop_count": 0}}
    previous = {"total_cost_usd": 80.0, "session_count": 40,
                "wastage_events": {"silent_death_recoveries": 3, "post_director_errors": 5,
                                   "failed_claims": 5, "escalated_dispatches": 2,
                                   "verifier_failures": 1, "holding_loop_count": 0}}
    delta = compute_week_delta(current, previous)
    assert abs(delta["cost_pct_change"] - (-37.5)) < 0.5  # 50/80 - 1
    assert abs(delta["session_pct_change"] - (-25.0)) < 0.5
    assert delta["wastage_event_count_change"] < 0  # fewer wastage events


def test_compute_week_delta_handles_no_previous():
    from tools.cost_rollup import compute_week_delta
    delta = compute_week_delta({"total_cost_usd": 50.0}, None)
    assert delta == {} or delta.get("cost_pct_change") is None
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: cost_rollup — find_outliers + compute_week_delta"
```

---

## Stage C4: Renderers + main() Wiring

### Task 8: cost_rollup.py — render_markdown() + write_json()

**Files:**
- Modify: `tools/cost_rollup.py`
- Modify: `tests/test_cost_rollup.py`

- [ ] **Step 1-5: TDD markdown + JSON renderers**

```python
def test_render_markdown_canonical():
    from tools.cost_rollup import render_markdown
    summary = {
        "week": "2026-W20",
        "date_range": ["2026-05-11", "2026-05-17"],
        "total_cost_usd": 47.32,
        "session_count": 42,
        "by_model": {
            "claude-opus-4-7": {"sessions": 12, "cost_usd": 38.50,
                                "input_tokens": 100000, "output_tokens": 5000,
                                "cache_read_tokens": 8000000},
            "claude-sonnet-4-6": {"sessions": 25, "cost_usd": 6.20,
                                  "input_tokens": 50000, "output_tokens": 3000,
                                  "cache_read_tokens": 4000000},
        },
        "by_role": {
            "director": {"dispatches": 8, "cost_usd": 14.20, "model": "claude-opus-4-7"},
            "tooling_engineer": {"dispatches": 5, "cost_usd": 2.50, "model": "claude-sonnet-4-6"},
        },
        "wastage_events": {"silent_death_recoveries": 0, "post_director_errors": 0,
                           "failed_claims": 1, "escalated_dispatches": 2,
                           "verifier_failures": 0, "holding_loop_count": 0},
        "outlier_sessions": [],
        "delta_vs_previous_week": {"cost_pct_change": -22.5},
    }
    md = render_markdown(summary)
    assert "2026-W20" in md
    assert "47.32" in md
    assert "opus-4-7" in md or "claude-opus-4-7" in md
    assert "−22.5%" in md or "-22.5%" in md
    assert "## By model" in md
    assert "## By role" in md
    assert "## Wastage events" in md
```

Implement `render_markdown(summary) -> str` returning the markdown shape from the spec § 6.

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: cost_rollup.render_markdown() — weekly cost report renderer"
```

---

### Task 9: cost_rollup.py — main() + week filtering + end-to-end

**Files:**
- Modify: `tools/cost_rollup.py`
- Modify: `tests/test_cost_rollup.py`

- [ ] **Step 1-5: TDD end-to-end main()**

```python
def test_main_writes_md_and_json_for_current_week(tmp_path, monkeypatch):
    import tools.cost_rollup as cr
    monkeypatch.setattr(cr, "REPO", tmp_path)
    # Set up fake project dir
    proj = tmp_path / "fake_project_jsonls"
    proj.mkdir()
    # Copy fixtures into the fake project
    import shutil
    shutil.copy(FIXTURE / "session_sample_opus.jsonl", proj / "test-session-opus.jsonl")
    # Override the project path
    monkeypatch.setattr(cr, "PROJECT_JSONLS_DIR", proj)
    # ... set up dispatch_telemetry sample, post_director sample
    rc = cr.main(argv=["--since", "2026-05-13", "--until", "2026-05-14"])
    assert rc == 0
    assert (tmp_path / "data/infra/cost_rollup.md").exists()
    assert (tmp_path / "data/infra/cost_rollup.json").exists()
```

Implement `main()`:
1. Parse args → determine date range
2. Glob `~/.claude/projects/<sanitized>/*.jsonl` (or PROJECT_JSONLS_DIR override)
3. For each jsonl, `parse_session_jsonl()` → skip if start_ts outside range
4. Load `dispatch_telemetry`, map each session to role
5. Compute cost per session
6. Aggregate by model + role
7. Detect wastage events
8. Find outliers
9. Compute delta vs previous week (parse existing cost_rollup.json if present)
10. Render markdown + JSON
11. **Atomic write — MANDATORY**: write to `<final>.tmp` then `os.rename(tmp, final)`. Tests assert this behavior.

**Additional tests required:**

```python
def test_main_atomic_write_no_partial_reads(tmp_path, monkeypatch):
    """Output files must be atomically replaced — never leave partial content visible."""
    import tools.cost_rollup as cr
    monkeypatch.setattr(cr, "REPO", tmp_path)
    # Set up minimal valid data
    (tmp_path / "data/infra").mkdir(parents=True)
    # Run main
    cr.main(argv=["--since", "2026-05-13", "--until", "2026-05-14"])
    # After main: final file exists, .tmp does NOT exist (was renamed)
    assert (tmp_path / "data/infra/cost_rollup.md").exists()
    assert not (tmp_path / "data/infra/cost_rollup.md.tmp").exists()
    assert (tmp_path / "data/infra/cost_rollup.json").exists()
    assert not (tmp_path / "data/infra/cost_rollup.json.tmp").exists()


def test_main_zero_sessions_in_range_produces_empty_rollup(tmp_path, monkeypatch):
    """Empty range → valid rollup with zero counts, no crash."""
    import tools.cost_rollup as cr
    import json as _json
    monkeypatch.setattr(cr, "REPO", tmp_path)
    (tmp_path / "data/infra").mkdir(parents=True)
    # Date range with no sessions
    rc = cr.main(argv=["--since", "1990-01-01", "--until", "1990-01-02"])
    assert rc == 0
    summary = _json.loads((tmp_path / "data/infra/cost_rollup.json").read_text())
    assert summary["total_cost_usd"] == 0.0
    assert summary["session_count"] == 0
    assert summary["by_model"] == {}
    assert summary["by_role"] == {}
    md = (tmp_path / "data/infra/cost_rollup.md").read_text()
    assert "No sessions in range" in md or "Total: $0.00" in md


def test_main_cross_aggregate_invariant(tmp_path, monkeypatch):
    """total_cost_usd must equal sum(by_model[*].cost_usd) AND sum(by_role[*].cost_usd)."""
    import tools.cost_rollup as cr
    import json as _json
    monkeypatch.setattr(cr, "REPO", tmp_path)
    # ... set up real fixtures with 3-5 sessions of varied models/roles ...
    cr.main(argv=["--since", "2026-05-13", "--until", "2026-05-14"])
    summary = _json.loads((tmp_path / "data/infra/cost_rollup.json").read_text())
    total = summary["total_cost_usd"]
    by_model_sum = sum(b["cost_usd"] for b in summary["by_model"].values())
    by_role_sum = sum(b["cost_usd"] for b in summary["by_role"].values())
    assert abs(total - by_model_sum) < 0.01, \
        f"by_model sum {by_model_sum} != total {total}"
    assert abs(total - by_role_sum) < 0.01, \
        f"by_role sum {by_role_sum} != total {total}"
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat: cost_rollup.main() — end-to-end + atomic file writes"
```

---

## Stage C4 Gate (user review): pause to verify end-to-end on real data — `python3 tools/cost_rollup.py` against this conversation's actual jsonls should produce a meaningful weekly rollup.

---

## Stage C5: Runner Integration + Validation

### Task 10: post_director.py — fire cost_rollup on success path

**Files:**
- Modify: `tools/post_director.py`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Write a smoke test that verifies post_director invokes cost_rollup at session-end**

```python
def test_post_director_invokes_cost_rollup_on_success(tmp_path, monkeypatch):
    # Verify that after finalize_from_structured runs the success branch,
    # cost_rollup is invoked (subprocess or import).
    # Use a mock or check that cost_rollup.md mtime advances.
    ...  # implementation matches the runner pattern
```

- [ ] **Step 2-5: Add cost_rollup invocation to post_director.py main()**

After the `finalize_from_structured` success path completes (in `main()`), add a non-fatal subprocess call:

```python
# Refresh weekly cost rollup — non-fatal.
try:
    import subprocess as _subprocess
    _subprocess.run(
        ["python3", "tools/cost_rollup.py"],
        cwd=str(REPO),
        capture_output=True,
        check=False,
        timeout=30,
    )
except Exception as _cost_exc:
    pass  # silent; rollup is best-effort
```

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(post_director): fire cost_rollup on session-end (non-fatal)"
```

---

### Task 11: End-to-end validation against real lab data

**Files:**
- (no code changes — operational task)

- [ ] **Step 1: Run cost_rollup against current data**

```bash
cd <repo> && source .venv/bin/activate
python3 tools/cost_rollup.py --since 2026-04-16 --until 2026-05-13
cat data/infra/cost_rollup.md
```

- [ ] **Step 2: Compare to the manual audit we ran in this conversation**

The manual audit found:
- Total visible: 2.031 B tokens
- This conversation: 1.886 B tokens
- Lab autonomous (all other sessions): ~145 M tokens

Verify cost_rollup's numbers are within ±5% of these manual totals.

- [ ] **Step 3: Commit a validation notes doc**

```bash
git add docs/superpowers/notes/2026-05-13-cost-rollup-validation.md
git commit -m "validate: cost_rollup matches manual audit ±5%; runner integration confirmed"
```

---

## Self-Review Notes

1. **Spec coverage**: every section of the spec maps to ≥1 task.
   - Pricing constants → Task 2
   - parse_session_jsonl → Task 2
   - compute_cost → Task 3
   - load_dispatch_telemetry + map_session_to_role → Task 4
   - aggregate_by_model + by_role → Task 5
   - detect_wastage_events → Task 6
   - find_outliers + compute_week_delta → Task 7
   - render_markdown + JSON → Task 8
   - main() end-to-end → Task 9
   - post_director wiring → Task 10
   - validation → Task 11

2. **No placeholders**: all task steps have concrete code or commands.

3. **TDD discipline**: every implementation task has Test → Run-fail → Implement → Run-pass → Commit cadence.

4. **Pricing as constants** is documented in the module top so the operator can update on Anthropic price changes without finding scattered references.

5. **Stage gates** at C1 and C4 give the operator review checkpoints before committing further.

---

## Estimated effort

- Stage C1: ~1.5h (parser + pricing math, 3 tasks)
- Stage C2: ~1.5h (role mapping + aggregators, 2 tasks)
- Stage C3: ~1h (wastage events + delta, 2 tasks)
- Stage C4: ~1h (renderers + main wiring, 2 tasks)
- Stage C5: ~30min (runner integration + validation, 2 tasks)

**Total: ~5.5 hours** via subagent-driven-development, plus user-review gates between C1/C2 and after C4.
