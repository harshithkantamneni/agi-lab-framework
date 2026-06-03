#!/usr/bin/env python3
"""Hardware Monitor — Real-time M3 Pro CPU/GPU/memory/thermal monitoring."""

import argparse
import json
import subprocess
import time
import psutil


def get_memory():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "percent": mem.percent,
        "swap_used_gb": round(swap.used / (1024**3), 2),
        "swap_percent": swap.percent,
    }


def get_cpu():
    freq = psutil.cpu_freq()
    per_core = psutil.cpu_percent(interval=0.5, percpu=True)
    p_cores = per_core[:6]  # M3 Pro: 6 performance cores
    e_cores = per_core[6:]  # M3 Pro: 6 efficiency cores
    return {
        "total_percent": psutil.cpu_percent(interval=0),
        "p_cores_avg": round(sum(p_cores) / len(p_cores), 1) if p_cores else 0,
        "e_cores_avg": round(sum(e_cores) / len(e_cores), 1) if e_cores else 0,
        "per_core": per_core,
        "freq_mhz": round(freq.current, 0) if freq else None,
    }


def get_gpu_and_thermal():
    """Use powermetrics for GPU and thermal data (requires sudo for full data)."""
    result = {"gpu_active": None, "thermal_pressure": None, "gpu_freq_mhz": None}
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-n", "AppleARMIODevice"],
            capture_output=True, text=True, timeout=5
        )
        # Thermal pressure from sysctl
        tp = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_level"],
            capture_output=True, text=True, timeout=5
        )
        if tp.returncode == 0:
            result["memory_pressure_level"] = tp.stdout.strip()
    except Exception:
        pass

    # GPU usage via ioreg
    try:
        out = subprocess.run(
            ["ioreg", "-r", "-n", "AGXAcceleratorM3"],
            capture_output=True, text=True, timeout=5
        )
        if "PerformanceStatistics" in out.stdout:
            result["gpu_active"] = True
        else:
            # Try generic name
            out2 = subprocess.run(
                ["ioreg", "-r", "-c", "AGXAccelerator"],
                capture_output=True, text=True, timeout=5
            )
            result["gpu_active"] = "PerformanceStatistics" in out2.stdout
    except Exception:
        pass

    return result


def get_disk():
    disk = psutil.disk_usage("/")
    return {
        "total_gb": round(disk.total / (1024**3), 1),
        "used_gb": round(disk.used / (1024**3), 1),
        "free_gb": round(disk.free / (1024**3), 1),
        "percent": round(disk.percent, 1),
    }


def get_battery():
    bat = psutil.sensors_battery()
    if bat:
        return {
            "percent": bat.percent,
            "plugged_in": bat.power_plugged,
            "secs_left": bat.secsleft if bat.secsleft != psutil.POWER_TIME_UNLIMITED else "unlimited",
        }
    return None


def get_processes_top(n=5):
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            info = p.info
            procs.append(info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: (x.get("cpu_percent") or 0), reverse=True)
    return procs[:n]


def snapshot():
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "memory": get_memory(),
        "cpu": get_cpu(),
        "gpu_thermal": get_gpu_and_thermal(),
        "disk": get_disk(),
        "battery": get_battery(),
        "top_processes": get_processes_top(),
    }


def format_readable(data):
    lines = []
    lines.append(f"=== Hardware Monitor — {data['timestamp']} ===\n")

    m = data["memory"]
    lines.append(f"MEMORY: {m['used_gb']:.1f}GB / {m['total_gb']:.1f}GB ({m['percent']}%) | Available: {m['available_gb']:.1f}GB | Swap: {m['swap_used_gb']:.1f}GB ({m['swap_percent']}%)")

    c = data["cpu"]
    lines.append(f"CPU: {c['total_percent']}% total | P-cores avg: {c['p_cores_avg']}% | E-cores avg: {c['e_cores_avg']}%")

    g = data["gpu_thermal"]
    lines.append(f"GPU: active={g.get('gpu_active', '?')} | Memory pressure: {g.get('memory_pressure_level', '?')}")

    d = data["disk"]
    lines.append(f"DISK: {d['used_gb']:.0f}GB / {d['total_gb']:.0f}GB ({d['percent']}%) | Free: {d['free_gb']:.0f}GB")

    b = data["battery"]
    if b:
        lines.append(f"BATTERY: {b['percent']}% | Plugged in: {b['plugged_in']}")

    lines.append("\nTOP PROCESSES:")
    for p in data["top_processes"]:
        name = p.get('name') or 'unknown'
        cpu = p.get('cpu_percent') or 0.0
        mem = p.get('memory_percent') or 0.0
        lines.append(f"  {name:<30} CPU: {cpu:>5.1f}%  MEM: {mem:>5.1f}%")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="M3 Pro Hardware Monitor")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", type=float, default=0, help="Refresh interval in seconds (0 = single snapshot)")
    parser.add_argument("--log", type=str, help="Append JSON snapshots to file")
    args = parser.parse_args()

    while True:
        data = snapshot()

        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(format_readable(data))

        if args.log:
            with open(args.log, "a") as f:
                f.write(json.dumps(data) + "\n")

        if args.watch <= 0:
            break
        time.sleep(args.watch)
        print()


if __name__ == "__main__":
    main()
