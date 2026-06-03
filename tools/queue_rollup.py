"""Rollup script for work queue telemetry.

Reads data/work_queue/queue_telemetry.jsonl and produces a per-item-type
summary at data/infra/queue_rollup.md showing lifetime counts (enqueued,
completed, failed) plus median claim→complete latency.

Also includes per-type flagging when failed/(completed+failed) > 0.20.

Pattern matches dispatch_rollup.py and calibration_rollup.py: REPO at
module level for monkeypatch testing, idempotent overwrite of output.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

REPO = Path(__file__).resolve().parent.parent


def main():
    telem_path = REPO / "data" / "work_queue" / "queue_telemetry.jsonl"
    if not telem_path.exists():
        print("no telemetry yet")
        return

    # Count by item type: enqueued, completed, failed
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"enqueued": 0, "completed": 0, "failed": 0, "latencies": []}
    )

    # Track claim and complete events for latency computation
    claim_times: dict[str, float] = {}  # item_id -> ts (ISO string)
    complete_times: dict[str, float] = {}  # item_id -> ts (ISO string)

    for line in telem_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue

        action = rec.get("action", "?")
        item_id = rec.get("item_id", "?")
        item_type = rec.get("item_type", "?")

        counts[item_type]["enqueued"] += action == "enqueue"
        counts[item_type]["completed"] += action == "complete"
        counts[item_type]["failed"] += action == "fail"

        # Track timestamps for latency calculation
        ts = rec.get("ts")
        if ts:
            if action == "claim":
                claim_times[item_id] = ts
            elif action == "complete":
                complete_times[item_id] = ts

    # Compute latencies per item type
    for item_id, claim_ts in claim_times.items():
        if item_id not in complete_times:
            continue
        complete_ts = complete_times[item_id]
        # Parse ISO format: "2026-05-04T22:30:00Z"
        try:
            claim_dt = datetime.fromisoformat(claim_ts.replace("Z", "+00:00"))
            complete_dt = datetime.fromisoformat(complete_ts.replace("Z", "+00:00"))
            latency_secs = (complete_dt - claim_dt).total_seconds()
            if latency_secs >= 0:  # Only count positive latencies
                # Map item_id back to type by re-scanning telemetry
                # (This is not the most efficient, but correctness first)
                for line in telem_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get("item_id") == item_id and rec.get("action") == "claim":
                            item_type = rec.get("item_type", "?")
                            counts[item_type]["latencies"].append(latency_secs)
                            break
                    except json.JSONDecodeError:
                        continue
        except (ValueError, AttributeError):
            continue

    # Build report
    out_path = REPO / "data" / "infra" / "queue_rollup.md"
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "# Work queue rollup",
        "",
        f"Generated: {now_iso}",
        "",
    ]

    # Add active state from work_queue.summary() if available
    try:
        from tools.work_queue import summary

        state = summary()
        lines.extend(
            [
                "## Active state",
                f"- Pending: {state.get('pending', 0)}",
                f"- Claimed: {state.get('claimed', 0)}",
                f"- Completed today: {state.get('completed_today', 0)}",
                f"- Failed today: {state.get('failed_today', 0)}",
                "",
            ]
        )
    except (ImportError, Exception):
        pass

    # Per-type table
    lines.extend(
        [
            "## By item type (lifetime, from telemetry)",
            "| type | enqueued | completed | failed | median claim→complete (s) | flag |",
            "|---|---|---|---|---|---|",
        ]
    )

    for item_type in sorted(counts):
        c = counts[item_type]
        enqueued = c["enqueued"]
        completed = c["completed"]
        failed = c["failed"]
        latencies = c["latencies"]

        # Compute median latency
        median_latency = ""
        if latencies:
            med = median(latencies)
            median_latency = f"{med:.0f}"

        # Compute fail rate flag
        flag = ""
        total_resolved = completed + failed
        if total_resolved > 0:
            fail_rate = failed / total_resolved
            if fail_rate > 0.20:
                flag = "FAIL RATE HIGH"

        lines.append(
            f"| {item_type} | {enqueued} | {completed} | {failed} | {median_latency} | {flag} |"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"rollup: wrote {len(counts)} item types to {out_path}")


if __name__ == "__main__":
    main()
