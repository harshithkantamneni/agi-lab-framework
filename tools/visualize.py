#!/usr/bin/env python3
"""Visualization tool — Training curves, architecture diagrams, benchmark plots."""

import argparse
import json
import os
import sys
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/plots")


def ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        sys.exit(1)


def plot_training_curve(log_path, output=None):
    """Plot loss curves from a training log (JSON lines format)."""
    plt = ensure_matplotlib()

    steps, losses, lr_values = [], [], []
    with open(log_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            steps.append(entry.get("step", len(steps)))
            losses.append(entry.get("loss", 0))
            if "lr" in entry:
                lr_values.append(entry["lr"])

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(steps, losses, "b-", alpha=0.3, label="loss (raw)")

    # Smoothed loss
    if len(losses) > 10:
        window = min(50, len(losses) // 5)
        smoothed = np.convolve(losses, np.ones(window)/window, mode="valid")
        ax1.plot(steps[window-1:], smoothed, "b-", linewidth=2, label="loss (smoothed)")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color="b")
    ax1.set_title("Training Curve")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    if lr_values:
        ax2 = ax1.twinx()
        ax2.plot(steps[:len(lr_values)], lr_values, "r--", alpha=0.5, label="learning rate")
        ax2.set_ylabel("Learning Rate", color="r")
        ax2.legend(loc="upper right")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = output or os.path.join(OUTPUT_DIR, "training_curve.png")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")


def plot_benchmark_comparison(baseline_path, current_path=None, output=None):
    """Plot benchmark comparison bar chart."""
    plt = ensure_matplotlib()

    with open(baseline_path) as f:
        baseline = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Matmul GFLOPS
    if "matmul" in baseline:
        sizes = [r["size"] for r in baseline["matmul"]]
        gflops = [r["gflops"] for r in baseline["matmul"]]
        axes[0].bar(sizes, gflops, color="steelblue", alpha=0.8)
        axes[0].set_title("Matrix Multiply Performance")
        axes[0].set_ylabel("GFLOPS")
        axes[0].set_xlabel("Matrix Size")
        axes[0].grid(True, alpha=0.3, axis="y")

    # Memory bandwidth
    if "memory_bandwidth" in baseline:
        sizes = [f"{r['size_mb']}MB" for r in baseline["memory_bandwidth"]]
        read_bw = [r["read_gbps"] for r in baseline["memory_bandwidth"]]
        write_bw = [r["write_gbps"] for r in baseline["memory_bandwidth"]]
        x = np.arange(len(sizes))
        axes[1].bar(x - 0.2, read_bw, 0.4, label="Read", color="steelblue", alpha=0.8)
        axes[1].bar(x + 0.2, write_bw, 0.4, label="Write", color="coral", alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(sizes)
        axes[1].set_title("Memory Bandwidth")
        axes[1].set_ylabel("GB/s")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis="y")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = output or os.path.join(OUTPUT_DIR, "benchmark.png")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")


def plot_experiment_dashboard(output=None):
    """Generate experiment tracker dashboard."""
    plt = ensure_matplotlib()
    import sqlite3

    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/experiments.db")
    if not os.path.exists(db_path):
        print("No experiments database found.")
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get data
    by_stream = conn.execute(
        "SELECT stream, status, COUNT(*) as c FROM experiments GROUP BY stream, status"
    ).fetchall()

    by_type = conn.execute(
        "SELECT type, COUNT(*) as c FROM experiments GROUP BY type ORDER BY c DESC"
    ).fetchall()

    timeline = conn.execute(
        "SELECT DATE(created_at) as day, COUNT(*) as c FROM experiments GROUP BY day ORDER BY day"
    ).fetchall()
    conn.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Stream breakdown
    streams = {}
    for row in by_stream:
        s, st, c = row["stream"], row["status"], row["c"]
        if s not in streams:
            streams[s] = {}
        streams[s][st] = c

    if streams:
        statuses = ["proposed", "running", "success", "failed", "abandoned"]
        colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c", "#7f8c8d"]
        stream_names = list(streams.keys())
        x = np.arange(len(stream_names))
        bottom = np.zeros(len(stream_names))

        for status, color in zip(statuses, colors):
            values = [streams[s].get(status, 0) for s in stream_names]
            if any(v > 0 for v in values):
                axes[0].bar(x, values, bottom=bottom, label=status, color=color)
                bottom += values

        axes[0].set_xticks(x)
        axes[0].set_xticklabels(stream_names, rotation=45, ha="right")
        axes[0].set_title("Experiments by Stream")
        axes[0].legend(fontsize=8)

    # Type breakdown
    if by_type:
        types = [r["type"] for r in by_type]
        counts = [r["c"] for r in by_type]
        axes[1].pie(counts, labels=types, autopct="%1.0f%%", startangle=90)
        axes[1].set_title("Experiment Types")

    # Timeline
    if timeline:
        days = [r["day"] for r in timeline]
        counts = [r["c"] for r in timeline]
        axes[2].bar(range(len(days)), counts, color="steelblue")
        axes[2].set_xticks(range(len(days)))
        axes[2].set_xticklabels(days, rotation=45, ha="right", fontsize=8)
        axes[2].set_title("Experiments Over Time")
        axes[2].set_ylabel("Count")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = output or os.path.join(OUTPUT_DIR, "experiment_dashboard.png")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved: {output}")


def generate_arch_dot(arch_json, output=None):
    """Generate a Graphviz DOT architecture diagram from JSON description."""
    with open(arch_json) as f:
        arch = json.load(f)

    dot_lines = ["digraph Architecture {", '  rankdir=TB;', '  node [shape=record, style=filled, fillcolor="#E8E8E8"];']

    for layer in arch.get("layers", []):
        label = f"{layer['name']}\\n{layer.get('type', '')}\\n{layer.get('params', '')}"
        color = {
            "input": "#AED6F1", "attention": "#F9E79F", "ffn": "#ABEBC6",
            "output": "#F5B7B1", "memory": "#D7BDE2", "norm": "#D5DBDB",
        }.get(layer.get("type", ""), "#E8E8E8")
        dot_lines.append(f'  {layer["name"]} [label="{label}", fillcolor="{color}"];')

    for conn in arch.get("connections", []):
        label = f' [label="{conn.get("label", "")}"]' if "label" in conn else ""
        dot_lines.append(f'  {conn["from"]} -> {conn["to"]}{label};')

    dot_lines.append("}")
    dot_text = "\n".join(dot_lines)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dot_path = os.path.join(OUTPUT_DIR, "architecture.dot")
    with open(dot_path, "w") as f:
        f.write(dot_text)

    # Try to render with graphviz
    output = output or os.path.join(OUTPUT_DIR, "architecture.png")
    try:
        import subprocess
        subprocess.run(["dot", "-Tpng", dot_path, "-o", output], check=True)
        print(f"Saved: {output}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(f"Graphviz not available. DOT file saved to: {dot_path}")
        print("Render manually: dot -Tpng architecture.dot -o architecture.png")


def main():
    parser = argparse.ArgumentParser(description="Visualization Tool")
    sub = parser.add_subparsers(dest="command")

    tc = sub.add_parser("training", help="Plot training curves")
    tc.add_argument("log_file", help="JSONL training log")
    tc.add_argument("-o", "--output", help="Output PNG path")

    bm = sub.add_parser("benchmark", help="Plot benchmark results")
    bm.add_argument("baseline", help="Baseline JSON file")
    bm.add_argument("-o", "--output", help="Output PNG path")

    sub.add_parser("dashboard", help="Experiment tracker dashboard")

    ar = sub.add_parser("architecture", help="Render architecture diagram from JSON")
    ar.add_argument("json_file", help="Architecture JSON file")
    ar.add_argument("-o", "--output", help="Output PNG path")

    args = parser.parse_args()

    if args.command == "training":
        plot_training_curve(args.log_file, args.output)
    elif args.command == "benchmark":
        plot_benchmark_comparison(args.baseline, output=args.output)
    elif args.command == "dashboard":
        plot_experiment_dashboard()
    elif args.command == "architecture":
        generate_arch_dot(args.json_file, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
