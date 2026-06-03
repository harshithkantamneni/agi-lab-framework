#!/usr/bin/env python3
"""Pre-flight check — Verify system resources before training runs."""

import json
import os
import sys
import psutil


THRESHOLDS = {
    "min_memory_available_gb": 4.0,
    "max_memory_percent": 85.0,
    "min_disk_free_gb": 20.0,
    "max_swap_percent": 50.0,
    "min_battery_percent": 20,
    "require_plugged_in": False,
}


def check_memory():
    mem = psutil.virtual_memory()
    avail_gb = mem.available / (1024**3)
    issues = []

    if avail_gb < THRESHOLDS["min_memory_available_gb"]:
        issues.append(f"LOW MEMORY: {avail_gb:.1f}GB available (need {THRESHOLDS['min_memory_available_gb']}GB)")
    if mem.percent > THRESHOLDS["max_memory_percent"]:
        issues.append(f"HIGH MEMORY USAGE: {mem.percent}% (max {THRESHOLDS['max_memory_percent']}%)")

    swap = psutil.swap_memory()
    if swap.percent > THRESHOLDS["max_swap_percent"]:
        issues.append(f"HIGH SWAP: {swap.percent}% (max {THRESHOLDS['max_swap_percent']}%)")

    return {
        "available_gb": round(avail_gb, 2),
        "used_percent": mem.percent,
        "swap_percent": swap.percent,
        "issues": issues,
    }


def check_disk():
    disk = psutil.disk_usage("/")
    free_gb = disk.free / (1024**3)
    issues = []

    if free_gb < THRESHOLDS["min_disk_free_gb"]:
        issues.append(f"LOW DISK: {free_gb:.1f}GB free (need {THRESHOLDS['min_disk_free_gb']}GB)")

    return {"free_gb": round(free_gb, 1), "issues": issues}


def check_battery():
    bat = psutil.sensors_battery()
    issues = []

    if bat:
        if bat.percent < THRESHOLDS["min_battery_percent"] and not bat.power_plugged:
            issues.append(f"LOW BATTERY: {bat.percent}% (min {THRESHOLDS['min_battery_percent']}%, not plugged in)")
        if THRESHOLDS["require_plugged_in"] and not bat.power_plugged:
            issues.append("NOT PLUGGED IN: Training requires AC power")
        return {
            "percent": bat.percent,
            "plugged_in": bat.power_plugged,
            "issues": issues,
        }
    return {"percent": None, "plugged_in": None, "issues": []}


def check_processes():
    """Check for resource-heavy processes that might compete."""
    heavy = []
    for p in psutil.process_iter(["pid", "name", "memory_percent"]):
        try:
            mem = p.info.get("memory_percent") or 0
            if mem > 10.0:
                heavy.append({"name": p.info["name"], "memory_percent": round(mem, 1)})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    issues = []
    if heavy:
        names = ", ".join(f"{p['name']} ({p['memory_percent']}%)" for p in heavy)
        issues.append(f"HEAVY PROCESSES: {names}")
    return {"heavy_processes": heavy, "issues": issues}


def check_checkpoint_dir():
    cp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/checkpoints")
    issues = []
    if not os.path.exists(cp_dir):
        issues.append(f"CHECKPOINT DIR MISSING: {cp_dir}")
    return {"path": cp_dir, "exists": os.path.exists(cp_dir), "issues": issues}


def check_cloud_sync():
    """ADVISORY (not blocking): warn if the repo lives under a cloud file-sync
    root. Cloud sync (iCloud "Desktop & Documents", Dropbox, OneDrive) evicts
    files to the cloud -> a cold `import torch` re-downloads ~2,100 files (minutes
    vs <1s), and it conflict-renames symlinks -> breaks .venv/bin/python -> lab
    outage. Keep the repo + .venv on a non-synced path (~/code, ~/dev)."""
    repo = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    home = os.path.realpath(os.path.expanduser("~"))
    icloud = os.path.join(home, "Library", "Mobile Documents")
    under = lambda root: repo == root or repo.startswith(root + os.sep)
    sync_root = None
    if under(os.path.realpath(icloud)):
        sync_root = "iCloud Drive"
    elif os.path.isdir(icloud) and (under(os.path.join(home, "Desktop")) or under(os.path.join(home, "Documents"))):
        sync_root = "iCloud 'Desktop & Documents' sync"
    elif under(os.path.join(home, "Dropbox")) or "/Dropbox/" in repo + os.sep:
        sync_root = "Dropbox"
    elif "OneDrive" in repo:
        sync_root = "OneDrive"
    warnings = []
    if sync_root:
        warnings.append(
            f"CLOUD-SYNCED REPO: {repo} is under {sync_root}. This evicts files "
            "(slow torch imports) and conflict-renames symlinks (breaks .venv). "
            "Move the repo + .venv to a non-synced path (e.g. ~/code or ~/dev)."
        )
    return {"path": repo, "sync_root": sync_root, "warnings": warnings}


def run_preflight():
    checks = {
        "memory": check_memory(),
        "disk": check_disk(),
        "battery": check_battery(),
        "processes": check_processes(),
        "checkpoints": check_checkpoint_dir(),
        "cloud_sync": check_cloud_sync(),
    }

    all_issues = []
    all_warnings = []
    for name, result in checks.items():
        all_issues.extend(result.get("issues", []))
        all_warnings.extend(result.get("warnings", []))

    passed = len(all_issues) == 0

    print("=" * 50)
    print("PRE-FLIGHT CHECK")
    print("=" * 50)

    mem = checks["memory"]
    print(f"  Memory:     {mem['available_gb']:.1f}GB available ({mem['used_percent']}% used, swap {mem['swap_percent']}%)")

    dsk = checks["disk"]
    print(f"  Disk:       {dsk['free_gb']:.0f}GB free")

    bat = checks["battery"]
    if bat["percent"] is not None:
        plug = "plugged in" if bat["plugged_in"] else "on battery"
        print(f"  Battery:    {bat['percent']}% ({plug})")

    cp = checks["checkpoints"]
    print(f"  Checkpoints: {'OK' if cp['exists'] else 'MISSING'}")

    if checks["processes"]["heavy_processes"]:
        for p in checks["processes"]["heavy_processes"]:
            print(f"  Heavy proc: {p['name']} using {p['memory_percent']}% memory")

    cs = checks["cloud_sync"]
    if cs["sync_root"]:
        print(f"  Cloud sync: ⚠ repo under {cs['sync_root']}")

    print()
    if passed:
        print("STATUS: ALL CLEAR — safe to start training")
    else:
        print("STATUS: ISSUES FOUND")
        for issue in all_issues:
            print(f"  !! {issue}")

    # Advisory warnings (do NOT block — e.g. cloud-synced repo path).
    for w in all_warnings:
        print(f"  ⚠  {w}")

    print("=" * 50)
    return passed


if __name__ == "__main__":
    ok = run_preflight()
    sys.exit(0 if ok else 1)
