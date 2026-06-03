#!/usr/bin/env python3
"""AGI Lab Dashboard — Live monitoring server.
Reads all state files and serves a real-time dashboard.
Usage: python tools/dashboard.py [--port 8420]
"""

import http.server
import json
import os
import re
import shutil
import sqlite3
import socketserver
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Health data will use cached logs instead of live system data.")

PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8420

# Response cache for SSE efficiency (thread-safe)
import threading
_cache: dict = {"data": None, "time": 0}
_cache_lock = threading.Lock()
CACHE_TTL = 2  # seconds
ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


def _memory_or_legacy(new_rel: str, legacy_rel: str) -> Path:
    """Prefer the memory-tier path; fall back to legacy during migration overlap."""
    new_path = Path(new_rel)
    legacy_path = Path(legacy_rel)
    if new_path.exists():
        return new_path
    return legacy_path


def _tier_sizes() -> dict:
    """Compute current hot/wiki/log tier sizes in KB.

    Returns {"hot_kb", "wiki_kb", "log_kb", "available"}.
    available is False pre-migration (when data/memories/ doesn't exist).
    """
    mem = Path("data/memories")
    if not mem.exists():
        return {"hot_kb": 0, "wiki_kb": 0, "log_kb": 0, "available": False}

    def kb(p: Path) -> int:
        return p.stat().st_size // 1024 if p.exists() else 0

    hot_kb = kb(mem / "current.md")
    log_kb = kb(mem / "log.md")
    exclude = {"current.md", "log.md", "INDEX.md", "history.md", "session_brief.md"}
    wiki_kb = sum(
        kb(p) for p in mem.rglob("*.md") if p.name not in exclude
    )
    return {"hot_kb": hot_kb, "wiki_kb": wiki_kb, "log_kb": log_kb, "available": True}


def read_file(path: str | Path) -> str:
    """Read a file, return empty string if missing."""
    try:
        return Path(path).read_text()
    except Exception:
        return ""


def get_session_state() -> dict:
    """Parse state.md (or legacy session_state.md) into structured data."""
    text = read_file(_memory_or_legacy("data/memories/current.md", str(DATA / "state.md")))
    if not text.strip():
        text = read_file(DATA / "session_state.md")
    state = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("## Current Cycle:"):
            state["cycle"] = line.split(":")[1].strip()
        elif line.startswith("## Phase:"):
            state["phase"] = line.split(":", 1)[1].strip()
        elif line.startswith("## Status:"):
            state["status"] = line.split(":", 1)[1].strip()
    # Active agents
    agents = []
    in_active = False
    for line in text.splitlines():
        if "Active Agents" in line or "Active agents" in line:
            in_active = True
            continue
        if in_active:
            if line.strip().startswith("- "):
                agents.append(line.strip().lstrip("- ").split(":")[0].strip())
            elif line.strip().startswith("##") or (line.strip() == "" and agents):
                in_active = False
    state["active_agents"] = agents
    state["raw"] = text
    return state


def get_roadmap() -> dict:
    """Parse state.md for milestones/inventory, fallback to roadmap.md."""
    text = read_file(_memory_or_legacy("data/memories/current.md", str(DATA / "state.md")))
    if not text.strip():
        text = read_file(DATA / "roadmap.md")
    milestones = []
    for line in text.splitlines():
        if line.strip().startswith("- [x]"):
            milestones.append({"text": line.strip()[6:].strip(), "done": True})
        elif line.strip().startswith("- [ ]"):
            milestones.append({"text": line.strip()[6:].strip(), "done": False})

    # Parse code inventory table
    code_inventory = []
    in_table = False
    for line in text.splitlines():
        if "Component" in line and "Files" in line and "Tests" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("|--") or line.strip().startswith("|-"):
                continue
            if line.strip().startswith("|"):
                cols = [c.strip() for c in line.split("|")[1:-1]]
                if len(cols) >= 4 and cols[0]:
                    code_inventory.append({
                        "component": cols[0],
                        "files": cols[1],
                        "tests": cols[2],
                        "status": cols[3]
                    })
            else:
                in_table = False

    # Extract total test count from code inventory (skip summary row)
    test_count = 0
    for item in code_inventory:
        if "Total" in item["component"] or "**" in item["component"]:
            m = re.search(r"(\d+)", item["tests"])
            if m:
                test_count = int(m.group(1))  # Use the total directly
                break
    if test_count == 0:
        for item in code_inventory:
            if "Total" not in item["component"] and "**" not in item["component"]:
                m = re.search(r"(\d+)\s*pass", item["tests"])
                if m:
                    test_count += int(m.group(1))

    return {
        "milestones": milestones,
        "code_inventory": code_inventory,
        "test_count": test_count,
        "raw": text,
    }


def get_benchmarks() -> list:
    """Parse benchmark_tracker.md."""
    text = read_file(DATA / "benchmark_tracker.md")
    benchmarks = []
    in_table = False
    for line in text.splitlines():
        if "Benchmark" in line and "Opus 4.6" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("|--") or line.strip().startswith("|-"):
                continue
            if line.strip().startswith("|"):
                cols = [c.strip() for c in line.split("|")[1:-1]]
                if len(cols) >= 4 and cols[0]:
                    target_str = cols[2] if len(cols) > 2 else ""
                    target = 0
                    m = re.search(r"(\d+\.?\d*)%", target_str)
                    if m:
                        target = float(m.group(1))
                    our_str = cols[3] if len(cols) > 3 else "—"
                    our_score = 0
                    m2 = re.search(r"(\d+\.?\d*)%?", our_str)
                    if m2 and our_str != "—" and our_str != "--":
                        our_score = float(m2.group(1))
                    benchmarks.append({
                        "name": cols[0],
                        "target": target,
                        "score": our_score,
                        "status": cols[-1] if len(cols) > 4 else ""
                    })
            else:
                in_table = False
    return benchmarks


def get_code_reviews() -> list:
    """Parse all code review files from data/engineering/reviews/."""
    reviews_dir = DATA / "engineering" / "reviews"
    reviews = []
    if not reviews_dir.exists():
        return reviews
    for f in sorted(reviews_dir.glob("review_*.md")):
        text = f.read_text()
        name = f.stem.replace("review_", "").replace("_", " ").title()
        review = {"name": name, "file": f.name, "status": "PENDING", "qa": "—"}
        for line in text.splitlines():
            # Normalize: strip markdown formatting
            clean = line.strip().lstrip("#").strip().lstrip("*").strip()
            clean_lower = clean.lower()
            # Match status lines: "Status: APPROVED", "## Status: NEEDS_FIXES", "**Status**: APPROVED"
            if clean_lower.startswith("status") and ":" in clean:
                val = clean.split(":", 1)[1].strip().strip("*").strip().upper()
                if "APPROVED" in val:
                    review["status"] = "APPROVED"
                elif "NEEDS" in val or "FIXES" in val:
                    review["status"] = "NEEDS_FIXES"
                elif "BLOCKED" in val:
                    review["status"] = "BLOCKED"
            # Match QA lines: "QA Status: PASS", "### QA Status: PENDING"
            if "qa" in clean_lower and "status" in clean_lower and ":" in clean:
                val = clean.split(":", 1)[1].strip().strip("*").strip().upper()
                if "PASS" in val:
                    review["qa"] = "PASS"
                elif "FAIL" in val:
                    review["qa"] = "FAIL"
        reviews.append(review)
    return reviews



def get_work_streams() -> list:
    """Parse state.md (or legacy cycle_queue.md) for active work."""
    text = read_file(_memory_or_legacy("data/memories/current.md", str(DATA / "state.md")))
    if not text.strip():
        text = read_file(DATA / "cycle_queue.md")
    streams = []
    current_priority = ""
    in_current = False
    # Find the last cycle section (highest cycle number)
    last_cycle_line = -1
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"##\s+Cycle\s+\d+", line.strip()) and "Completed" not in line:
            last_cycle_line = i
    for i, line in enumerate(lines):
        stripped = line.strip()
        if i == last_cycle_line:
            in_current = True
            continue
        # Stop at completed/backlog sections
        if in_current and (stripped.startswith("### Completed") or stripped.startswith("## Backlog") or stripped.startswith("## Code Quality")):
            break
        if not in_current:
            continue
        if stripped.startswith("### P0"):
            current_priority = "P0"
        elif stripped.startswith("### P1"):
            current_priority = "P1"
        elif stripped.startswith("### P2"):
            current_priority = "P2"
        elif stripped and stripped[0].isdigit() and "." in stripped[:3]:
            status = "pending"
            if "[IN_PROGRESS]" in stripped or "[IN PROGRESS]" in stripped:
                status = "active"
            elif "[DONE]" in stripped:
                status = "done"
            elif "[BLOCKED]" in stripped:
                status = "blocked"
            elif "[PENDING]" in stripped:
                status = "pending"
            clean = re.sub(r"\[.*?\]\s*", "", stripped)
            clean = re.sub(r"^\d+\.\s*", "", clean)
            streams.append({
                "text": clean.strip(),
                "status": status,
                "priority": current_priority,
            })
    return streams


def get_decisions() -> list:
    """Parse decisions_recent.md (or legacy director_log.md) for decisions."""
    text = read_file(_memory_or_legacy("data/memories/log.md", str(DATA / "decisions_recent.md")))
    if not text.strip():
        text = read_file(DATA / "director_log.md")
    decisions = []
    current = None
    for line in text.splitlines():
        if line.startswith("## Decision"):
            if current:
                decisions.append(current)
            title = line.replace("## Decision", "").strip().lstrip(":").strip()
            current = {"id": "", "title": title, "date": "", "what": ""}
        elif current:
            if line.startswith("Date:"):
                current["date"] = line.split(":", 1)[1].strip()
            elif line.startswith("Cycle:"):
                current["cycle"] = line.split(":", 1)[1].strip()
            elif line.startswith("### What"):
                pass
            elif current.get("title") and not current["what"] and line.strip() and not line.startswith("#"):
                current["what"] = line.strip()
    if current:
        decisions.append(current)
    return list(reversed(decisions))


def get_memory_budget() -> dict:
    """Parse memory_budget.md for actual allocation numbers."""
    text = read_file(DATA / "engineering" / "memory_budget.md")
    budget = {
        "total_mb": 18432,
        "allocations": [],
        "confirmed_used_mb": 0,
        "headroom_mb": 0,
    }

    in_table = False
    for line in text.splitlines():
        if "Component" in line and "Budget" in line:
            in_table = True
            continue
        if in_table:
            if line.strip().startswith("|--") or line.strip().startswith("|-"):
                continue
            if line.strip().startswith("|"):
                cols = [c.strip() for c in line.split("|")[1:-1]]
                if len(cols) >= 4 and cols[0]:
                    name = cols[0].strip("*").strip()
                    budget_val = re.search(r"[\d,]+", cols[1].replace(",", ""))
                    actual_val = re.search(r"[\d,]+", cols[2].replace(",", "").replace("~", ""))
                    status = cols[3] if len(cols) > 3 else ""
                    if budget_val and name not in ("Total Used", "Headroom"):
                        budget["allocations"].append({
                            "name": name,
                            "budget_mb": int(budget_val.group()),
                            "actual_mb": int(actual_val.group()) if actual_val else 0,
                            "status": status,
                        })
            else:
                in_table = False

    # Parse key numbers section
    for line in text.splitlines():
        if "headroom" in line.lower():
            # Match the number right before "headroom" (e.g., "5.65 GB headroom")
            m = re.search(r"(\d+\.?\d*)\s*GB\s*headroom", line)
            if m:
                budget["headroom_mb"] = int(float(m.group(1)) * 1024)
        if "confirmed fit" in line.lower():
            m = re.search(r"(\d+\.?\d*)\s*GB\s*for\s*model", line)
            if m:
                budget["confirmed_used_mb"] = int(float(m.group(1)) * 1024)

    # Fallback calculation
    if budget["confirmed_used_mb"] == 0:
        budget["confirmed_used_mb"] = sum(
            a["actual_mb"] or a["budget_mb"] for a in budget["allocations"]
        )
    if budget["headroom_mb"] == 0:
        budget["headroom_mb"] = budget["total_mb"] - budget["confirmed_used_mb"]

    return budget


def get_health() -> dict:
    """Get system health from thermal_log, disk_budget, and live data."""
    thermal = read_file(DATA / "infra" / "thermal_log.md")
    disk = read_file(DATA / "infra" / "disk_budget.md")
    build = read_file(DATA / "infra" / "build_status.md")

    health = {
        "cpu": "—", "gpu": "—", "thermal": "—", "disk_free": "—",
        "battery": "—", "memory_used": "—", "memory_avail": "—",
        "build": "—", "swap": "—",
    }

    # Live memory from psutil
    if HAS_PSUTIL:
        try:
            mem = psutil.virtual_memory()
            health["memory_used"] = f"{mem.used / (1024**3):.1f}GB"
            health["memory_avail"] = f"{mem.available / (1024**3):.1f}GB"
            swap = psutil.swap_memory()
            health["swap"] = f"{swap.used / (1024**3):.1f}GB"
            health["cpu"] = f"{psutil.cpu_percent(interval=0.3):.0f}%"
        except Exception:
            pass
    else:
        for line in thermal.splitlines():
            if "Memory used" in line and "|" in line:
                cols = [c.strip() for c in line.split("|")]
                for c in cols:
                    if "GB" in c and "/" in c:
                        health["memory_used"] = c
                        break

    # Thermal from log file (no sudo required)
    thermal_text = thermal.lower()
    if "throttle" in thermal_text and "none" not in thermal_text:
        health["thermal"] = "THROTTLED"
    else:
        health["thermal"] = "NORMAL"

    # Live GPU utilization
    try:
        gpu_out = subprocess.check_output(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            text=True, timeout=3
        )
        gpu_match = re.search(r'"Device Utilization %"=(\d+)', gpu_out)
        if gpu_match:
            health["gpu"] = f"{gpu_match.group(1)}%"
    except Exception:
        pass

    # Battery
    if HAS_PSUTIL:
        try:
            batt = psutil.sensors_battery()
            if batt:
                health["battery"] = f"{batt.percent:.0f}%"
                if batt.power_plugged:
                    health["battery"] += " ⚡"
        except Exception:
            pass

    # Disk free
    for line in disk.splitlines():
        if "free:" in line.lower():
            m = re.search(r"(\d+\.?\d*)\s*GB", line)
            if m:
                health["disk_free"] = f"{int(float(m.group(1)))}GB"

    # Live disk if not found in log
    if health["disk_free"] == "—":
        try:
            _, _, free = shutil.disk_usage("/")
            health["disk_free"] = f"{free // (1024**3)}GB"
        except Exception:
            pass

    # Test count — prefer state.md (or roadmap.md) over stale build_status.md
    roadmap_text = read_file(_memory_or_legacy("data/memories/current.md", str(DATA / "state.md")))
    if not roadmap_text.strip():
        roadmap_text = read_file(DATA / "roadmap.md")
    tm = re.search(r"\*\*Total C tests\*\*.*?(\d+)", roadmap_text)
    if tm:
        health["build"] = f"{tm.group(1)} pass"
    else:
        for line in build.splitlines():
            m = re.search(r"(\d+)/\d+\s+C\s+tests", line)
            if m:
                health["build"] = f"{m.group(1)} pass"
                break

    return health


def get_build_info() -> dict:
    """Parse build_status.md into structured data."""
    text = read_file(DATA / "infra" / "build_status.md")
    info = {
        "c_core": {"status": "—", "files": 0},
        "metal": {"status": "—", "shaders": 0},
        "swift": {"status": "—"},
        "tests": {"status": "—", "total": 0, "passing": 0},
        "last_build": "—",
        "components": [],
    }

    for line in text.splitlines():
        if line.startswith("## C Core:"):
            info["c_core"]["status"] = "PASS" if "PASS" in line else "FAIL"
            m = re.search(r"(\d+)\s+source", line)
            if m:
                info["c_core"]["files"] = int(m.group(1))
        elif line.startswith("## Metal Shaders:"):
            info["metal"]["status"] = "PASS" if "PASS" in line else "FAIL"
            m = re.search(r"(\d+)\s+shader", line)
            if m:
                info["metal"]["shaders"] = int(m.group(1))
        elif line.startswith("## Swift Bridge:"):
            info["swift"]["status"] = "NOT STARTED" if "NOT YET" in line else "PASS"
        elif line.startswith("## Tests:"):
            info["tests"]["status"] = "PASS" if "PASS" in line else "FAIL"
            m = re.search(r"(\d+)/(\d+)", line)
            if m:
                info["tests"]["passing"] = int(m.group(1))
                info["tests"]["total"] = int(m.group(2))
        elif line.startswith("## Last Successful Build:"):
            info["last_build"] = line.split(":", 1)[1].strip()
        elif line.strip().startswith("- test_"):
            m = re.search(r"(test_\w+):\s*(\d+)\s+tests", line)
            if m:
                info["components"].append({
                    "name": m.group(1),
                    "tests": int(m.group(2)),
                })

    return info


def get_research() -> dict:
    """Parse research strategy and hypothesis status."""
    strategy = read_file(DATA / "research" / "strategy.md")

    hypotheses = []
    current_team = ""
    for line in strategy.splitlines():
        if line.startswith("### ") and "Team" in line:
            current_team = line.replace("###", "").strip()
        elif line.strip().startswith("- H-"):
            h_id = line.strip().split(":")[0].lstrip("- ").strip()
            desc = line.strip().split(":", 1)[1].strip() if ":" in line else ""
            status = "theorized"
            if "CONFIRMED" in line.upper():
                status = "confirmed"
            elif "MODIFIED" in line.upper():
                status = "modified"
            elif "FALSIFIED" in line.upper() or "REJECTED" in line.upper():
                status = "dead"
            elif "TESTED" in line.upper():
                status = "tested"
            hypotheses.append({
                "id": h_id,
                "team": current_team,
                "desc": desc[:120],
                "status": status
            })

    # Cross-check with cycle_queue for completed hypotheses
    queue = read_file(_memory_or_legacy("data/memories/current.md", str(DATA / "state.md")))
    if not queue.strip():
        queue = read_file(DATA / "cycle_queue.md")
    for line in queue.splitlines():
        if "[DONE]" in line and "H-" in line:
            m = re.search(r"H-\w+", line)
            if m:
                h_id = m.group(0)
                status = "confirmed"
                if "MODIFIED" in line:
                    status = "modified"
                elif "FALSIFIED" in line or "REJECTED" in line:
                    status = "dead"
                for h in hypotheses:
                    if h["id"] == h_id:
                        h["status"] = status

    return {"hypotheses": hypotheses, "raw_strategy": strategy}


def get_session_count() -> int:
    """Count session logs."""
    log_dir = DATA / "infra" / "session_logs"
    if not log_dir.exists():
        return 0
    return len(list(log_dir.glob("*.log")))


def get_experiment_count() -> int:
    """Count experiments in DB."""
    db_path = DATA / "experiments.db"
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def get_bibliography() -> list:
    """Parse bibliography.md into structured data."""
    text = read_file(DATA / "bibliography.md")
    papers = []
    current_section = ""
    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("## arXiv"):
            current_section = line[3:].split("(")[0].strip()
        if line.strip().startswith("|") and not line.strip().startswith("|--") and not line.strip().startswith("|-") and not line.strip().startswith("| #"):
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if cols and cols[0].isdigit():
                papers.append({
                    "num": int(cols[0]),
                    "title": cols[1] if len(cols) > 1 else "",
                    "authors": cols[2][:60] if len(cols) > 2 else "",
                    "year": cols[3] if len(cols) > 3 else "",
                    "relevance": cols[4] if len(cols) > 4 else "",
                    "stream": cols[5] if len(cols) > 5 else "",
                    "section": current_section,
                })
    return papers


def get_risks() -> list:
    """Parse key risks from state.md (or legacy session_state.md)."""
    text = read_file(_memory_or_legacy("data/memories/current.md", str(DATA / "state.md")))
    if not text.strip():
        text = read_file(DATA / "session_state.md")
    risks = []
    in_risks = False
    for line in text.splitlines():
        if "Key Risks" in line or "## Key Risks" in line:
            in_risks = True
            continue
        if in_risks:
            if line.strip().startswith("##") and "Risk" not in line:
                break
            if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
                clean = re.sub(r"^\d+\.\s*", "", line.strip())
                # Extract severity from parenthetical at end: (CRITICAL), (HIGH), (MEDIUM)
                severity = "MEDIUM"
                sev_match = re.search(r"\((\w+)\)\s*$", clean)
                if sev_match:
                    sev = sev_match.group(1).upper()
                    if sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                        severity = sev
                elif "CRITICAL" in clean.upper():
                    severity = "CRITICAL"
                elif "HIGH" in clean.upper():
                    severity = "HIGH"
                # Extract text before the severity/detail
                text_clean = re.sub(r"\s*\(\w+\)\s*$", "", clean)
                # Split on " — " to get main text vs detail
                parts = text_clean.split(" — ", 1)
                risks.append({
                    "text": parts[0].strip()[:80],
                    "detail": parts[1].strip() if len(parts) > 1 else "",
                    "severity": severity,
                })
    return risks


def get_experiments() -> list:
    """Get experiments from DB with details."""
    db_path = DATA / "experiments.db"
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, stream, type, title, status, metrics, created_at FROM experiments ORDER BY id DESC LIMIT 20"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_training_progress() -> dict:
    """Parse the latest training log for live experiment data."""
    # Find the most recent training log
    exp_dir = DATA / "experiments"
    logs = list(exp_dir.rglob("training_log.txt")) if exp_dir.exists() else []
    logs += list(exp_dir.rglob("training.log")) if exp_dir.exists() else []
    logs += list(DATA.glob("experiments/*.log"))
    # Deduplicate
    logs = list({str(p): p for p in logs}.values())
    if not logs:
        return {"active": False}
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    log_path = logs[0]
    lines = read_file(log_path).splitlines()

    progress = _parse_training_log(lines, log_path)
    progress["max_steps"] = _detect_max_steps(lines)
    _compute_timing(lines, progress)
    progress["running"] = _is_experiment_running()
    return progress


def _parse_training_log(lines: list, log_path: Path) -> dict:
    """Parse summary and step lines from training log."""
    p = {
        "active": True, "log_file": str(log_path.relative_to(DATA)),
        "ppl_history": [], "loss_history": [], "entropy_history": [],
        "current_step": 0, "max_steps": 0, "best_ppl": 0,
        "latest_loss": 0, "latest_entropy": 0, "nan_count": 0,
        "tokens_trained": "", "epochs": 0,
    }
    for line in lines:
        m = re.search(
            r"Step\s+(\d+)\s+summary.*avg_lm=([\d.]+).*best_ppl=([\d.]+)"
            r".*epochs=(\d+).*tokens=(\S+).*NaN=(\d+)", line
        )
        if m:
            step, avg_lm, best_ppl = int(m.group(1)), float(m.group(2)), float(m.group(3))
            p["ppl_history"].append({"step": step, "ppl": best_ppl})
            p["loss_history"].append({"step": step, "loss": avg_lm})
            p["current_step"] = step
            p["best_ppl"] = best_ppl
            p["latest_loss"] = avg_lm
            p["tokens_trained"] = m.group(5)
            p["epochs"] = int(m.group(4))
            p["nan_count"] = int(m.group(6))

        if line.strip() and line.strip()[0].isdigit() and "|" in line:
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 6:
                try:
                    step = int(cols[0])
                    entropy = float(cols[4])
                    if step % 100 == 0:
                        p["entropy_history"].append({"step": step, "entropy": entropy})
                    p["latest_entropy"] = entropy
                    p["current_step"] = max(p["current_step"], step)
                except (ValueError, IndexError):
                    pass
    return p


def _detect_max_steps(lines: list) -> int:
    """Detect max_steps from log header or running process."""
    for line in lines[:30]:
        m = re.search(r"[Ss]teps[=:]\s*(\d+)", line)
        if m:
            return int(m.group(1))
        m = re.search(r"over\s+(\d+)\s+steps", line)
        if m:
            return int(m.group(1))
    try:
        ps = subprocess.check_output(["ps", "aux"], text=True, timeout=2)
        m = re.search(r"scale_experiment.*--steps\s+(\d+)", ps)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return 0


def _compute_timing(lines: list, progress: dict) -> None:
    """Compute steps/sec and ETA from step timing data.

    Log columns vary across experiments. The last column is always ms (milliseconds).
    We use that for timing, falling back to file timestamps if unavailable.
    """
    step_times = []
    for line in lines:
        if line.strip() and line.strip()[0].isdigit() and "|" in line:
            cols = [c.strip() for c in line.split("|")]
            if len(cols) >= 4:
                try:
                    step = int(cols[0])
                    # Last column is ms (milliseconds per step)
                    ms = float(cols[-1])
                    if 100 < ms < 600000:  # sanity: between 0.1s and 10min per step
                        step_times.append((step, ms / 1000.0))  # convert to seconds
                except (ValueError, IndexError):
                    pass
    if step_times and progress["current_step"] > 0:
        recent = step_times[-20:]
        avg = sum(t for _, t in recent) / len(recent)
        progress["steps_per_sec"] = round(1.0 / avg, 3) if avg > 0 else 0
        progress["elapsed_sec"] = int(progress["current_step"] * avg)
        if progress["max_steps"] > 0:
            remaining = progress["max_steps"] - progress["current_step"]
            progress["eta_sec"] = int(remaining * avg)


def _is_experiment_running() -> bool:
    """Check if a scale_experiment binary is running (not just mentioned in args)."""
    try:
        ps = subprocess.check_output(["pgrep", "-x", "scale_experi"], text=True, timeout=2)
        return bool(ps.strip())
    except Exception:
        try:
            ps = subprocess.check_output(["ps", "aux"], text=True, timeout=2)
            for line in ps.splitlines():
                if "./build/scale_experiment" in line and "grep" not in line:
                    return True
        except Exception:
            pass
        return False


def get_last_activity() -> dict:
    """Find most recently modified state file."""
    newest = 0
    newest_file = ""
    for p in DATA.rglob("*.md"):
        try:
            mtime = p.stat().st_mtime
            if mtime > newest:
                newest = mtime
                newest_file = str(p.relative_to(DATA))
        except Exception:
            pass
    if newest > 0:
        return {
            "timestamp": datetime.fromtimestamp(newest).isoformat(),
            "file": newest_file,
            "ago_seconds": int(time.time() - newest),
        }
    return {"timestamp": "", "file": "", "ago_seconds": -1}


def get_cycle_history() -> list:
    """Parse cycle milestones from decisions_recent.md (or legacy director_log.md)."""
    text = read_file(_memory_or_legacy("data/memories/log.md", str(DATA / "decisions_recent.md")))
    if not text.strip():
        text = read_file(DATA / "director_log.md")
    cycles = []
    for line in text.splitlines():
        if line.startswith("## Decision"):
            title = line.replace("## Decision", "").strip().lstrip(":").strip()
            cycles.append(title)
    return cycles


def build_api_response() -> dict:
    """Build the full dashboard data, with thread-safe TTL cache for SSE efficiency."""
    now = time.time()
    with _cache_lock:
        if _cache["data"] and (now - _cache["time"]) < CACHE_TTL:
            return _cache["data"]

    session = get_session_state()
    roadmap = get_roadmap()

    # Override stale build_status.md test count with live roadmap count
    build_info = get_build_info()
    if roadmap["test_count"] > 0:
        build_info["tests"]["passing"] = roadmap["test_count"]
        build_info["tests"]["total"] = roadmap["test_count"]
        build_info["tests"]["status"] = "PASS"

    result = {
        "timestamp": datetime.now().isoformat(),
        "session": session,
        "roadmap": roadmap,
        "benchmarks": get_benchmarks(),
        "code_reviews": get_code_reviews(),
        "streams": get_work_streams(),
        "decisions": get_decisions(),
        "health": get_health(),
        "memory_budget": get_memory_budget(),
        "build_info": build_info,
        "research": get_research(),
        "session_count": get_session_count(),
        "experiment_count": get_experiment_count(),
        "bibliography": get_bibliography(),
        "risks": get_risks(),
        "experiments": get_experiments(),
        "last_activity": get_last_activity(),
        "cycle_history": get_cycle_history(),
        "training": get_training_progress(),
        "memory_tiers": _tier_sizes(),
    }

    with _cache_lock:
        _cache["data"] = result
        _cache["time"] = now
    return result


DASHBOARD_HTML_PATH = ROOT / "tools" / "dashboard.html"


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == "/api/data":
            data = build_api_response()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif self.path == "/api/stream":
            # Server-Sent Events endpoint for real-time updates
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                while True:
                    data = build_api_response()
                    payload = json.dumps(data)
                    self.wfile.write(f"data: {payload}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(3)  # Push every 3 seconds
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML_PATH.read_bytes())
        else:
            self.send_response(404)
            self.end_headers()


class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    with ReusableTCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"AGI Lab Dashboard running at http://localhost:{PORT}")
        print(f"Reading state from: {DATA}")
        print("Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")


if __name__ == "__main__":
    main()
