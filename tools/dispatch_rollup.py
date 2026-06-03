"""Weekly rollup of dispatch telemetry.

Reads data/infra/dispatch_telemetry.jsonl and produces a per-role table at
data/infra/dispatch_rollup.md showing total dispatches, escalation rate,
and verifier-fail rate. Flags roles where escalation or verifier-fail rate
exceeds thresholds — these are signals to adjust the default tier in
agents.json.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    log_path = REPO / "data/infra/dispatch_telemetry.jsonl"
    if not log_path.exists():
        print("no telemetry yet")
        return

    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "escalated": 0, "verifier_fail": 0, "verifier_seen": 0}
    )
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = rec.get("role", "?")
        counts[role]["total"] += 1
        if rec.get("escalated"):
            counts[role]["escalated"] += 1
        vp = rec.get("verifier_pass")
        if vp is not None:
            counts[role]["verifier_seen"] += 1
            if vp is False:
                counts[role]["verifier_fail"] += 1

    out_path = REPO / "data/infra/dispatch_rollup.md"
    lines = [
        "# Dispatch rollup",
        "",
        "| role | total | escalation_rate | verifier_fail_rate | flag |",
        "|---|---|---|---|---|",
    ]
    for role in sorted(counts):
        c = counts[role]
        total = c["total"]
        esc_rate = c["escalated"] / total if total else 0.0
        vfail_rate = (c["verifier_fail"] / c["verifier_seen"]) if c["verifier_seen"] else 0.0

        flag = ""
        if esc_rate > 0.30:
            flag = "ESCALATION HIGH (consider tier upgrade)"
        elif vfail_rate > 0.20:
            flag = "VERIFIER FAIL HIGH (consider model change or procedural fix)"

        lines.append(
            f"| {role} | {total} | {esc_rate*100:.1f}% | {vfail_rate*100:.1f}% | {flag} |"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"rollup: wrote {len(counts)} rows to {out_path}")


if __name__ == "__main__":
    main()
