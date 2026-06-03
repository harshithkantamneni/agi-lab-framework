#!/usr/bin/env python3
"""Research Digest — Synthesize findings across all research streams for cross-agent knowledge sharing."""

import json
import os
import sqlite3
import time
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/experiments.db")
DIGEST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/digests")
SHARED_KNOWLEDGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/shared_knowledge.md")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_recent_experiments(hours=24):
    """Get experiments updated in the last N hours."""
    conn = get_db()
    cutoff = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
    rows = conn.execute(
        "SELECT * FROM experiments WHERE updated_at > ? ORDER BY updated_at DESC",
        (cutoff,)
    ).fetchall()

    # Get notes for these experiments
    results = []
    for row in rows:
        exp = dict(row)
        notes = conn.execute(
            "SELECT * FROM notes WHERE experiment_id = ? ORDER BY created_at",
            (exp["id"],)
        ).fetchall()
        exp["notes"] = [dict(n) for n in notes]
        results.append(exp)

    conn.close()
    return results


def get_stream_summary():
    """Get summary of each research stream."""
    conn = get_db()
    streams = conn.execute(
        "SELECT stream, COUNT(*) as total, "
        "SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as successes, "
        "SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failures, "
        "SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) as running, "
        "MAX(updated_at) as last_activity "
        "FROM experiments GROUP BY stream ORDER BY last_activity DESC"
    ).fetchall()
    conn.close()
    return [dict(s) for s in streams]


def get_key_findings():
    """Get successful experiments with results — these are confirmed findings."""
    conn = get_db()
    findings = conn.execute(
        "SELECT * FROM experiments WHERE status='success' AND result IS NOT NULL "
        "ORDER BY updated_at DESC LIMIT 50"
    ).fetchall()
    conn.close()
    return [dict(f) for f in findings]


def get_cross_references():
    """Find experiments that reference each other via parent_id."""
    conn = get_db()
    refs = conn.execute(
        "SELECT e1.id as child_id, e1.title as child_title, e1.stream as child_stream, "
        "e2.id as parent_id, e2.title as parent_title, e2.stream as parent_stream "
        "FROM experiments e1 JOIN experiments e2 ON e1.parent_id = e2.id "
        "WHERE e1.stream != e2.stream"
    ).fetchall()
    conn.close()
    return [dict(r) for r in refs]


def generate_digest():
    """Generate a comprehensive research digest."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    streams = get_stream_summary()
    recent = get_recent_experiments(hours=24)
    findings = get_key_findings()
    cross_refs = get_cross_references()

    lines = []
    lines.append(f"# Research Digest — {now}")
    lines.append("")

    # Stream overview
    lines.append("## Stream Status")
    lines.append("")
    for s in streams:
        lines.append(f"### {s['stream'].upper()}")
        lines.append(f"- Total: {s['total']} experiments | Success: {s['successes']} | Failed: {s['failures']} | Running: {s['running']}")
        lines.append(f"- Last activity: {s['last_activity']}")
        lines.append("")

    # Recent activity
    if recent:
        lines.append("## Recent Activity (24h)")
        lines.append("")
        for exp in recent:
            status_icon = {"proposed": "?", "running": ">>", "success": "OK", "failed": "XX"}.get(exp["status"], "  ")
            lines.append(f"- [{status_icon}] **#{exp['id']}** [{exp['stream']}] {exp['title']}")
            if exp.get("result"):
                lines.append(f"  - Result: {exp['result']}")
            for note in exp.get("notes", []):
                lines.append(f"  - Note: {note['content']}")
        lines.append("")

    # Key findings
    if findings:
        lines.append("## Key Findings (Confirmed)")
        lines.append("")
        for f in findings:
            lines.append(f"- **#{f['id']}** [{f['stream']}] {f['title']}")
            lines.append(f"  - {f['result']}")
            if f.get("metrics"):
                lines.append(f"  - Metrics: {f['metrics']}")
        lines.append("")

    # Cross-stream connections
    if cross_refs:
        lines.append("## Cross-Stream Connections")
        lines.append("")
        for ref in cross_refs:
            lines.append(f"- #{ref['child_id']} ({ref['child_stream']}) \"{ref['child_title']}\" builds on #{ref['parent_id']} ({ref['parent_stream']}) \"{ref['parent_title']}\"")
        lines.append("")

    # Potential connections (same tags across streams)
    lines.append("## Agent Instructions")
    lines.append("")
    lines.append("When starting research in any stream, READ THIS FILE FIRST.")
    lines.append("Check if findings from other streams affect your work.")
    lines.append("Log cross-references using --parent flag when building on another stream's finding.")
    lines.append("")

    digest_text = "\n".join(lines)

    # Save as shared knowledge
    with open(SHARED_KNOWLEDGE, "w") as f:
        f.write(digest_text)

    # Save timestamped digest
    os.makedirs(DIGEST_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest_path = os.path.join(DIGEST_DIR, f"digest_{ts}.md")
    with open(digest_path, "w") as f:
        f.write(digest_text)

    return digest_text, digest_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Research Digest Generator")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("generate", help="Generate new digest")
    sub.add_parser("show", help="Show current shared knowledge")

    s = sub.add_parser("streams", help="Show stream summaries")
    f = sub.add_parser("findings", help="Show key findings")

    args = parser.parse_args()

    if args.command == "generate":
        text, path = generate_digest()
        print(text)
        print(f"\n--- Saved to {path} and {SHARED_KNOWLEDGE} ---")

    elif args.command == "show":
        if os.path.exists(SHARED_KNOWLEDGE):
            with open(SHARED_KNOWLEDGE) as f:
                print(f.read())
        else:
            print("No shared knowledge yet. Run 'generate' first.")

    elif args.command == "streams":
        for s in get_stream_summary():
            print(f"{s['stream']:<15} total={s['total']} ok={s['successes']} fail={s['failures']} running={s['running']}")

    elif args.command == "findings":
        for f in get_key_findings():
            print(f"#{f['id']:<4} [{f['stream']}] {f['title']}: {f['result']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
