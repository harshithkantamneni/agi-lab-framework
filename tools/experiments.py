#!/usr/bin/env python3
"""Experiment Tracker — Log hypotheses, architectures, benchmarks, and results."""

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/experiments.db")


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            stream TEXT NOT NULL,
            type TEXT NOT NULL,
            title TEXT NOT NULL,
            description TEXT,
            status TEXT DEFAULT 'proposed',
            result TEXT,
            metrics TEXT,
            tags TEXT,
            parent_id INTEGER,
            FOREIGN KEY (parent_id) REFERENCES experiments(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments(id)
        )
    """)
    conn.commit()
    return conn


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_experiment(stream, exp_type, title, description=None, tags=None, parent_id=None):
    conn = get_db()
    ts = now()
    cur = conn.execute(
        "INSERT INTO experiments (created_at, updated_at, stream, type, title, description, tags, parent_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (ts, ts, stream, exp_type, title, description, json.dumps(tags) if tags else None, parent_id)
    )
    conn.commit()
    exp_id = cur.lastrowid
    conn.close()
    return exp_id


def update_status(exp_id, status, result=None, metrics=None):
    conn = get_db()
    ts = now()
    if metrics:
        metrics = json.dumps(metrics)
    conn.execute(
        "UPDATE experiments SET status=?, result=?, metrics=?, updated_at=? WHERE id=?",
        (status, result, metrics, ts, exp_id)
    )
    conn.commit()
    conn.close()


def add_note(exp_id, content):
    conn = get_db()
    ts = now()
    conn.execute(
        "INSERT INTO notes (experiment_id, created_at, content) VALUES (?, ?, ?)",
        (exp_id, ts, content)
    )
    conn.commit()
    conn.close()


def list_experiments(stream=None, status=None, exp_type=None, limit=20):
    conn = get_db()
    query = "SELECT * FROM experiments WHERE 1=1"
    params = []
    if stream:
        query += " AND stream = ?"
        params.append(stream)
    if status:
        query += " AND status = ?"
        params.append(status)
    if exp_type:
        query += " AND type = ?"
        params.append(exp_type)
    query += " ORDER BY updated_at DESC LIMIT ?"
    params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_experiment(exp_id):
    conn = get_db()
    exp = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
    notes = conn.execute("SELECT * FROM notes WHERE experiment_id = ? ORDER BY created_at", (exp_id,)).fetchall()
    conn.close()
    if exp:
        result = dict(exp)
        result["notes"] = [dict(n) for n in notes]
        return result
    return None


def get_stats():
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) as c FROM experiments").fetchone()["c"]
    by_status = conn.execute("SELECT status, COUNT(*) as c FROM experiments GROUP BY status").fetchall()
    by_stream = conn.execute("SELECT stream, COUNT(*) as c FROM experiments GROUP BY stream").fetchall()
    by_type = conn.execute("SELECT type, COUNT(*) as c FROM experiments GROUP BY type").fetchall()
    conn.close()
    return {
        "total": total,
        "by_status": {r["status"]: r["c"] for r in by_status},
        "by_stream": {r["stream"]: r["c"] for r in by_stream},
        "by_type": {r["type"]: r["c"] for r in by_type},
    }


def format_experiment(exp):
    status_icon = {"proposed": "?", "running": ">>", "success": "OK", "failed": "XX", "abandoned": "--"}.get(exp["status"], "  ")
    return f"[{status_icon}] #{exp['id']:>4d} | {exp['stream']:<15} | {exp['type']:<12} | {exp['title']}"


def main():
    parser = argparse.ArgumentParser(description="Experiment Tracker")
    sub = parser.add_subparsers(dest="command")

    # Log new experiment
    lg = sub.add_parser("log", help="Log a new experiment/hypothesis")
    lg.add_argument("title")
    lg.add_argument("--stream", required=True, help="Research stream (math, neuro, physics, arch, hardware, learning)")
    lg.add_argument("--type", required=True, dest="exp_type", help="Type (hypothesis, architecture, benchmark, observation, idea)")
    lg.add_argument("--desc", help="Description")
    lg.add_argument("--tags", nargs="+", help="Tags")
    lg.add_argument("--parent", type=int, help="Parent experiment ID")

    # Update status
    up = sub.add_parser("update", help="Update experiment status")
    up.add_argument("id", type=int)
    up.add_argument("--status", required=True, choices=["proposed", "running", "success", "failed", "abandoned"])
    up.add_argument("--result", help="Result description")
    up.add_argument("--metrics", help="JSON metrics string")

    # Add note
    nt = sub.add_parser("note", help="Add a note to an experiment")
    nt.add_argument("id", type=int)
    nt.add_argument("content")

    # List
    ls = sub.add_parser("list", help="List experiments")
    ls.add_argument("--stream", help="Filter by stream")
    ls.add_argument("--status", help="Filter by status")
    ls.add_argument("--type", help="Filter by type")
    ls.add_argument("-n", type=int, default=20, help="Max results")
    ls.add_argument("--json", action="store_true")

    # Get details
    gt = sub.add_parser("get", help="Get experiment details")
    gt.add_argument("id", type=int)
    gt.add_argument("--json", action="store_true")

    # Stats
    sub.add_parser("stats", help="Show statistics")

    args = parser.parse_args()

    if args.command == "log":
        exp_id = log_experiment(args.stream, args.exp_type, args.title, args.desc, args.tags, args.parent)
        print(f"Logged experiment #{exp_id}: {args.title}")

    elif args.command == "update":
        metrics = json.loads(args.metrics) if args.metrics else None
        update_status(args.id, args.status, args.result, metrics)
        print(f"Updated #{args.id} -> {args.status}")

    elif args.command == "note":
        add_note(args.id, args.content)
        print(f"Note added to #{args.id}")

    elif args.command == "list":
        exps = list_experiments(args.stream, args.status, args.type, args.n)
        if args.json:
            print(json.dumps(exps, indent=2))
        else:
            if not exps:
                print("No experiments found.")
            else:
                for e in exps:
                    print(format_experiment(e))

    elif args.command == "get":
        exp = get_experiment(args.id)
        if not exp:
            print(f"Experiment #{args.id} not found.")
            return
        if args.json:
            print(json.dumps(exp, indent=2))
        else:
            print(f"#{exp['id']} — {exp['title']}")
            print(f"Stream: {exp['stream']} | Type: {exp['type']} | Status: {exp['status']}")
            print(f"Created: {exp['created_at']} | Updated: {exp['updated_at']}")
            if exp['description']:
                print(f"\n{exp['description']}")
            if exp['result']:
                print(f"\nResult: {exp['result']}")
            if exp['metrics']:
                print(f"Metrics: {exp['metrics']}")
            if exp['tags']:
                print(f"Tags: {exp['tags']}")
            if exp['notes']:
                print(f"\nNotes ({len(exp['notes'])}):")
                for n in exp['notes']:
                    print(f"  [{n['created_at']}] {n['content']}")

    elif args.command == "stats":
        s = get_stats()
        print(f"Total experiments: {s['total']}")
        print(f"\nBy status:")
        for k, v in s["by_status"].items():
            print(f"  {k}: {v}")
        print(f"\nBy stream:")
        for k, v in s["by_stream"].items():
            print(f"  {k}: {v}")
        print(f"\nBy type:")
        for k, v in s["by_type"].items():
            print(f"  {k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
