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

# Ensure repo root on sys.path so direct invocation works without pytest's auto-pathing.
# Mirrors the pattern in tools/queue_scanner.py and tools/post_director.py.
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_ON_PATH = str(_TOOLS_DIR.parent)
if _REPO_ROOT_ON_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_ON_PATH)

REPO = Path(__file__).resolve().parent.parent

PROJECT_JSONLS_DIR = Path.home() / ".claude" / "projects" / str(REPO).replace("/", "-")

SCHEMA_VERSION = "1.0"

# All values $ per million tokens.
# cache_creation is priced at 1.25x input (Anthropic 5-min cache write premium).
# cache_read is priced at 0.10x input (cache hit discount).
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
    missing OR no assistant messages. Corrupt lines and non-assistant
    lines silently skipped. Missing usage subfields default to 0.
    """
    if not Path(path).exists():
        return None
    out: dict[str, Any] = {
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
            m = msg.get("model")
            if m:
                # Prefer the last REAL (non-synthetic) model. Claude Code's orchestration
                # appends a "<synthetic>" stub at session close; we want the actual model
                # that did the work, not the orchestration label.
                if m != "<synthetic>" or out["model"] is None:
                    out["model"] = m
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


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_creation_tokens: int,
) -> float:
    """Convert per-bucket token counts → dollars using PRICING constants.

    Pricing buckets:
      - input: $/M for fresh input tokens
      - cache_creation: 1.25× input ($/M for cache writes, Anthropic 5-min premium)
      - cache_read: 0.1× input ($/M for cache hits)
      - output: $/M for generated tokens

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


# ---------------------------------------------------------------------------
# Session → role mapping
# ---------------------------------------------------------------------------

ROLE_MAP_WINDOW_SECONDS = 60


def load_dispatch_telemetry(path: Path | None = None) -> list[dict]:
    """Read data/infra/dispatch_telemetry.jsonl (or override path) into a
    list of dicts. Missing file → empty list. Corrupt lines silently skipped.
    """
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
    """Parse ISO 8601 timestamp (with Z or +00:00 suffix) → unix epoch seconds.
    Raises ValueError on malformed input."""
    if not s:
        raise ValueError("empty timestamp")
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def map_session_to_role(session: dict, dispatch_records: list[dict]) -> str:
    """Tag a session as director / <role> / subagent_unknown_role.

    Picks the CLOSEST matching dispatch (smallest |session.start_ts - dispatch.ts|)
    within ROLE_MAP_WINDOW_SECONDS (±60s). If no match within window:
      - session_id starts with "agent-" → "subagent_unknown_role"
      - otherwise → "director"

    Dispatch records with malformed timestamps are silently skipped.
    """
    sid = session.get("session_id", "") or ""
    is_agent = sid.startswith("agent-")
    fallback = "subagent_unknown_role" if is_agent else "director"

    start_ts_str = session.get("start_ts")
    if not start_ts_str:
        return fallback
    try:
        start_ts = _parse_iso_to_ts(start_ts_str)
    except ValueError:
        return fallback

    # Find the CLOSEST dispatch record within the window.
    best_match: dict | None = None
    best_delta = ROLE_MAP_WINDOW_SECONDS + 1.0  # sentinel: > window so nothing matches yet
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
    return fallback


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


def aggregate_by_model(enriched_sessions: list[dict]) -> dict[str, dict]:
    """Group enriched sessions by model. Returns model_name → bucket dict.

    Bucket fields: sessions, input_tokens, output_tokens, cache_read_tokens,
    cache_creation_tokens, cost_usd.

    Sessions with model=None / missing → bucketed under "unknown".
    """
    out: dict[str, dict] = {}
    for s in enriched_sessions:
        m = s.get("model") or "unknown"
        b = out.setdefault(m, {
            "sessions": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cost_usd": 0.0,
        })
        b["sessions"] += 1
        b["input_tokens"] += s.get("total_input_tokens", 0) or 0
        b["output_tokens"] += s.get("total_output_tokens", 0) or 0
        b["cache_read_tokens"] += s.get("total_cache_read_tokens", 0) or 0
        b["cache_creation_tokens"] += s.get("total_cache_creation_tokens", 0) or 0
        b["cost_usd"] += s.get("cost_usd", 0.0) or 0.0
    return out


def aggregate_by_role(enriched_sessions: list[dict]) -> dict[str, dict]:
    """Group enriched sessions by role. Returns role_name → bucket dict.

    Bucket fields: dispatches, cost_usd, model (last seen), avg_cost_per_dispatch.
    """
    out: dict[str, dict] = {}
    for s in enriched_sessions:
        r = s.get("role") or "unknown"
        b = out.setdefault(r, {"dispatches": 0, "cost_usd": 0.0, "model": None})
        b["dispatches"] += 1
        b["cost_usd"] += s.get("cost_usd", 0.0) or 0.0
        if b["model"] is None:
            b["model"] = s.get("model")
    # Compute avg per role
    for b in out.values():
        b["avg_cost_per_dispatch"] = (
            b["cost_usd"] / b["dispatches"] if b["dispatches"] > 0 else 0.0
        )
    return out


# ---------------------------------------------------------------------------
# Wastage event detection
# ---------------------------------------------------------------------------


def _ts_in_range(ts_str: str, since_ts: str, until_ts: str) -> bool:
    """Return True if ts_str (ISO) falls within [since_ts, until_ts]."""
    if not ts_str:
        return False
    try:
        ts = _parse_iso_to_ts(ts_str)
        s = _parse_iso_to_ts(since_ts)
        u = _parse_iso_to_ts(until_ts)
        return s <= ts <= u
    except ValueError:
        return False


def _read_jsonl(path: Path) -> list[dict]:
    """Read a jsonl file into a list of dicts. Missing file → empty. Corrupt lines skipped."""
    if not Path(path).exists():
        return []
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def detect_wastage_events(
    post_director_path: Path,
    queue_telemetry_path: Path,
    dispatch_telemetry: list[dict],
    since_ts: str,
    until_ts: str,
) -> dict[str, int]:
    """Count wastage events within [since_ts, until_ts] by joining 3 telemetry sources.

    Event classes:
      silent_death_recoveries: post_director branch_taken == "silent_death" AND error is null
      post_director_errors:    post_director records with error field populated
      failed_claims:           queue_telemetry action == "fail"
      escalated_dispatches:    dispatch_telemetry escalated == True
      verifier_failures:       dispatch_telemetry verifier_pass == False
      holding_loop_count:      post_director errors / 3 (integer division), modeling
                                "≥3 repeated failed attempts = one holding loop event"
    """
    pd_records = _read_jsonl(post_director_path)
    qt_records = _read_jsonl(queue_telemetry_path)

    events = {
        "silent_death_recoveries": 0,
        "post_director_errors": 0,
        "failed_claims": 0,
        "escalated_dispatches": 0,
        "verifier_failures": 0,
        "holding_loop_count": 0,
    }

    # post_director — silent_death and errors
    pd_error_count = 0
    for rec in pd_records:
        if not _ts_in_range(rec.get("ts", ""), since_ts, until_ts):
            continue
        if rec.get("branch_taken") == "silent_death" and not rec.get("error"):
            events["silent_death_recoveries"] += 1
        if rec.get("error"):
            events["post_director_errors"] += 1
            pd_error_count += 1

    # holding_loop_count: ≥3 errors = 1 holding loop
    events["holding_loop_count"] = pd_error_count // 3

    # queue_telemetry — failed claims
    for rec in qt_records:
        if not _ts_in_range(rec.get("ts", ""), since_ts, until_ts):
            continue
        if rec.get("action") == "fail":
            events["failed_claims"] += 1

    # dispatch_telemetry — escalations and verifier failures
    for rec in dispatch_telemetry:
        if not _ts_in_range(rec.get("ts", ""), since_ts, until_ts):
            continue
        if rec.get("escalated") is True:
            events["escalated_dispatches"] += 1
        if rec.get("verifier_pass") is False:
            events["verifier_failures"] += 1

    return events


# ---------------------------------------------------------------------------
# Outliers + week-over-week delta
# ---------------------------------------------------------------------------


def find_outliers(
    enriched_sessions: list[dict],
    mad_threshold: float = 3.5,
) -> list[dict]:
    """Flag sessions whose cost is > mad_threshold × MAD above the median.

    Uses MEDIAN ABSOLUTE DEVIATION (MAD) — robust to the very outliers we're
    hunting. Mean+stddev would be inflated by outliers and hide them.

    With <3 sessions or MAD==0 (all identical), returns empty list — no
    baseline to deviate from.

    DEDUP: same session_id appearing in multiple jsonl files (parent + subagent
    files) is summed into ONE outlier entry. Operator-facing view shows
    "this session cost $X total" rather than 4 separate rows.

    Returns deduped outlier session dicts sorted by cost descending.
    """
    if len(enriched_sessions) < 3:
        return []
    costs = [s.get("cost_usd", 0.0) or 0.0 for s in enriched_sessions]
    median_cost = statistics.median(costs)
    abs_devs = [abs(c - median_cost) for c in costs]
    mad = statistics.median(abs_devs)
    if mad == 0:
        return []
    threshold = median_cost + mad_threshold * mad

    # Filter to outliers
    flagged = [
        s for s in enriched_sessions
        if (s.get("cost_usd", 0.0) or 0.0) > threshold
    ]

    # Dedup by session_id, summing cost_usd across rows with the same id
    by_sid: dict[str, dict] = {}
    for s in flagged:
        sid = s.get("session_id") or "unknown"
        if sid not in by_sid:
            by_sid[sid] = dict(s)
        else:
            by_sid[sid]["cost_usd"] = (
                by_sid[sid].get("cost_usd", 0.0) + s.get("cost_usd", 0.0)
            )

    out = sorted(by_sid.values(), key=lambda s: s.get("cost_usd", 0.0) or 0.0, reverse=True)
    return out


def compute_week_delta(current: dict, previous: dict | None) -> dict:
    """Compute pct-change in cost, sessions, wastage between two weekly summaries.

    Returns dict with cost_pct_change, session_pct_change, wastage_event_count_change.
    Returns {} when previous is None or empty (no delta computable).

    Division-by-zero is guarded — if previous cost is 0, cost_pct_change is None.
    """
    if not previous:
        return {}

    def _pct(curr: float, prev: float) -> float | None:
        if prev == 0:
            return None
        return (curr - prev) / prev * 100.0

    cur_cost = current.get("total_cost_usd", 0.0) or 0.0
    prev_cost = previous.get("total_cost_usd", 0.0) or 0.0
    cur_sessions = current.get("session_count", 0) or 0
    prev_sessions = previous.get("session_count", 0) or 0

    cur_wastage = sum((current.get("wastage_events", {}) or {}).values())
    prev_wastage = sum((previous.get("wastage_events", {}) or {}).values())

    return {
        "cost_pct_change": _pct(cur_cost, prev_cost),
        "session_pct_change": _pct(cur_sessions, prev_sessions),
        "wastage_event_count_change": cur_wastage - prev_wastage,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def _human_tokens(n: int) -> str:
    """Format token count as compact string (123 / 12K / 12.3M / 1.2B)."""
    if n is None:
        return "?"
    n = int(n)
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}K"
    if n < 1_000_000_000:
        return f"{n / 1_000_000:.1f}M"
    return f"{n / 1_000_000_000:.2f}B"


def _human_dollars(amt: float) -> str:
    """Format $ amount using round-half-up to match expected display (e.g. $1.78 not $1.77)."""
    if amt is None:
        return "?"
    import math
    rounded = math.floor(amt * 100 + 0.5) / 100
    return f"${rounded:.2f}"


def _wastage_emoji(count: int) -> str:
    """🟢 if zero, 🟡 if 1-5, 🔴 if >5."""
    if count == 0:
        return "🟢"
    if count <= 5:
        return "🟡"
    return "🔴"


def render_markdown(summary: dict) -> str:
    """Render a weekly cost summary dict to markdown narrative (~2-3 KB)."""
    week = summary.get("week", "?")
    date_range = summary.get("date_range", ["?", "?"])
    total_cost = summary.get("total_cost_usd", 0.0)
    session_count = summary.get("session_count", 0)
    delta = summary.get("delta_vs_previous_week", {}) or {}
    cost_pct = delta.get("cost_pct_change")

    lines: list[str] = []

    # Header
    lines.append(f"# Cost Rollup — Week {week} ({date_range[0]} to {date_range[1]})")
    lines.append("")
    header_bits = [
        f"**Total:** {_human_dollars(total_cost)}",
        f"**Sessions:** {session_count}",
    ]
    if cost_pct is not None:
        sign = "−" if cost_pct < 0 else "+"
        header_bits.append(f"**Δ vs prev week:** {sign}{abs(cost_pct):.1f}%")
    lines.append(" · ".join(header_bits))
    lines.append("")

    # Empty-period short-circuit
    if session_count == 0:
        lines.append("_No sessions in range._")
        lines.append("")
        lines.append("---")
        lines.append(
            "*Generated by tools/cost_rollup.py. "
            "Pricing constants at module top — update on Anthropic price changes.*"
        )
        return "\n".join(lines) + "\n"

    # By model
    lines.append("## By model")
    lines.append("")
    lines.append("| model | sessions | tokens | $ |")
    lines.append("|---|---|---|---|")
    by_model = summary.get("by_model", {}) or {}
    for model, b in sorted(by_model.items(), key=lambda kv: -kv[1].get("cost_usd", 0)):
        total_tok = (b.get("input_tokens", 0) + b.get("output_tokens", 0)
                     + b.get("cache_read_tokens", 0) + b.get("cache_creation_tokens", 0))
        lines.append(
            f"| {model} | {b.get('sessions', 0)} | "
            f"{_human_tokens(total_tok)} | {_human_dollars(b.get('cost_usd', 0))} |"
        )
    lines.append("")

    # By role
    lines.append("## By role (top dispatchers)")
    lines.append("")
    lines.append("| role | dispatches | model | total $ | $/dispatch |")
    lines.append("|---|---|---|---|---|")
    by_role = summary.get("by_role", {}) or {}
    for role, b in sorted(by_role.items(), key=lambda kv: -kv[1].get("cost_usd", 0)):
        lines.append(
            f"| {role} | {b.get('dispatches', 0)} | {b.get('model', '?')} | "
            f"{_human_dollars(b.get('cost_usd', 0))} | "
            f"{_human_dollars(b.get('avg_cost_per_dispatch', 0))} |"
        )
    lines.append("")

    # Wastage events
    lines.append("## Wastage events")
    lines.append("")
    wastage = summary.get("wastage_events", {}) or {}
    event_order = [
        "silent_death_recoveries",
        "post_director_errors",
        "failed_claims",
        "escalated_dispatches",
        "verifier_failures",
        "holding_loop_count",
    ]
    for ev in event_order:
        n = wastage.get(ev, 0)
        lines.append(f"- {_wastage_emoji(n)} {ev}: {n}")
    lines.append("")

    # Outliers
    lines.append("## Outlier sessions (>3.5× MAD above median)")
    lines.append("")
    outliers = summary.get("outlier_sessions", []) or []
    if not outliers:
        lines.append("None.")
    else:
        for o in outliers:
            sid = o.get("session_id", "?")
            role = o.get("role", "?")
            cost = o.get("cost_usd", 0.0)
            lines.append(f"- {sid} — {role} — {_human_dollars(cost)}")
    lines.append("")

    # Δ vs previous week
    if delta:
        lines.append("## Δ vs previous week")
        lines.append("")
        if delta.get("cost_pct_change") is not None:
            pct = delta["cost_pct_change"]
            sign = "−" if pct < 0 else "+"
            lines.append(f"- Cost: {sign}{abs(pct):.1f}%")
        if delta.get("session_pct_change") is not None:
            pct = delta["session_pct_change"]
            sign = "−" if pct < 0 else "+"
            lines.append(f"- Sessions: {sign}{abs(pct):.1f}%")
        wec = delta.get("wastage_event_count_change")
        if wec is not None:
            sign = "" if wec >= 0 else "−"
            lines.append(f"- Wastage event count: {sign}{abs(wec)}")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        "*Generated by tools/cost_rollup.py. "
        "Pricing constants at module top — update on Anthropic price changes.*"
    )

    return "\n".join(lines) + "\n"


def _parse_date_range(args) -> tuple[str, str]:
    """Resolve --week or --since/--until → (since_iso, until_iso) tuple."""
    if args.since:
        since_str = args.since + "T00:00:00+00:00"
    else:
        # Default: 7 days back from today
        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).date()
        since_str = seven_days_ago.isoformat() + "T00:00:00+00:00"

    if args.until:
        until_str = args.until + "T23:59:59+00:00"
    else:
        until_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:59+00:00")

    return since_str, until_str


def _iso_week_label(ts_str: str) -> tuple[str, list[str]]:
    """Return ISO week label '2026-W20' + date range ['YYYY-MM-DD', 'YYYY-MM-DD']."""
    dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    year, week, _ = dt.isocalendar()
    # Find Monday + Sunday of that ISO week
    monday = dt - timedelta(days=dt.weekday())
    sunday = monday + timedelta(days=6)
    return f"{year}-W{week:02d}", [monday.date().isoformat(), sunday.date().isoformat()]


def _atomic_write(path: Path, content: str) -> None:
    """Write content to path atomically (write to .tmp then os.rename)."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)  # atomic on POSIX + Windows


def _read_previous_summary() -> dict | None:
    """Read the existing cost_rollup.json if present (for week-delta computation)."""
    p = REPO / "data" / "infra" / "cost_rollup.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cost rollup over session jsonls.")
    parser.add_argument("--week", help="ISO week YYYY-WW (default: current)")
    parser.add_argument("--since", help="Start date YYYY-MM-DD")
    parser.add_argument("--until", help="End date YYYY-MM-DD")
    args = parser.parse_args(argv)

    since_str, until_str = _parse_date_range(args)
    week, date_range = _iso_week_label(since_str)

    # Step 1: glob session jsonls
    proj_dir = PROJECT_JSONLS_DIR
    if not proj_dir.exists():
        # Empty range fallback — no session data available
        sessions: list[dict] = []
    else:
        session_paths = sorted(proj_dir.glob("*.jsonl"))
        sessions = []
        for path in session_paths:
            session = parse_session_jsonl(path)
            if session is None:
                continue
            # Filter by date range (use start_ts)
            if not session.get("start_ts"):
                continue
            if not _ts_in_range(session["start_ts"], since_str, until_str):
                continue
            sessions.append(session)

    # Step 2: load dispatch_telemetry + map session → role + compute cost
    dispatch_records = load_dispatch_telemetry()
    enriched: list[dict] = []
    for s in sessions:
        role = map_session_to_role(s, dispatch_records)
        cost = compute_cost(
            model=s.get("model") or "",
            input_tokens=s.get("total_input_tokens", 0),
            output_tokens=s.get("total_output_tokens", 0),
            cache_read_tokens=s.get("total_cache_read_tokens", 0),
            cache_creation_tokens=s.get("total_cache_creation_tokens", 0),
        )
        enriched.append({**s, "role": role, "cost_usd": cost})

    # Step 3: aggregate
    by_model = aggregate_by_model(enriched)
    by_role = aggregate_by_role(enriched)

    # Step 4: wastage events
    events = detect_wastage_events(
        post_director_path=REPO / "data" / "infra" / "post_director_telemetry.jsonl",
        queue_telemetry_path=REPO / "data" / "work_queue" / "queue_telemetry.jsonl",
        dispatch_telemetry=dispatch_records,
        since_ts=since_str,
        until_ts=until_str,
    )

    # Step 5: outliers
    outliers_raw = find_outliers(enriched)
    outliers = [
        {"session_id": o.get("session_id"), "role": o.get("role"), "cost_usd": o.get("cost_usd")}
        for o in outliers_raw
    ]

    # Step 6: build summary
    total_cost = sum(s.get("cost_usd", 0.0) for s in enriched)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "week": week,
        "date_range": date_range,
        "since_ts": since_str,
        "until_ts": until_str,
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": sum(s.get("total_input_tokens", 0) for s in enriched),
        "total_output_tokens": sum(s.get("total_output_tokens", 0) for s in enriched),
        "total_cache_read_tokens": sum(s.get("total_cache_read_tokens", 0) for s in enriched),
        "session_count": len(enriched),
        "by_model": by_model,
        "by_role": by_role,
        "wastage_events": events,
        "outlier_sessions": outliers,
    }

    # Step 7: week delta vs previous run
    previous = _read_previous_summary()
    summary["delta_vs_previous_week"] = compute_week_delta(summary, previous)

    # Step 8: atomic write both outputs
    md = render_markdown(summary)
    (REPO / "data" / "infra").mkdir(parents=True, exist_ok=True)
    _atomic_write(REPO / "data" / "infra" / "cost_rollup.md", md)
    _atomic_write(
        REPO / "data" / "infra" / "cost_rollup.json",
        json.dumps(summary, indent=2),
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
