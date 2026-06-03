#!/usr/bin/env python3
"""Benchmark runner — Measure and compare performance of primitives and models."""

import argparse
import json
import os
import subprocess
import sys
import time
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/experiments.db")


def time_fn(fn, warmup=3, iterations=10):
    """Time a function with warmup and multiple iterations."""
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = fn()
        end = time.perf_counter_ns()
        times.append((end - start) / 1e6)  # ms

    return {
        "mean_ms": round(np.mean(times), 3),
        "std_ms": round(np.std(times), 3),
        "min_ms": round(np.min(times), 3),
        "max_ms": round(np.max(times), 3),
        "iterations": iterations,
    }


def bench_matmul(sizes=None):
    """Benchmark matrix multiplication at various sizes."""
    if sizes is None:
        sizes = [64, 128, 256, 512, 1024, 2048]

    results = []
    for n in sizes:
        a = np.random.randn(n, n).astype(np.float32)
        b = np.random.randn(n, n).astype(np.float32)

        timing = time_fn(lambda: a @ b, warmup=2, iterations=5)

        flops = 2 * n * n * n  # 2N^3 for matmul
        gflops = flops / (timing["mean_ms"] * 1e6)  # GFLOPS

        results.append({
            "size": f"{n}x{n}",
            "gflops": round(gflops, 2),
            **timing,
        })
        print(f"  matmul {n:>5}x{n:<5}: {timing['mean_ms']:>8.2f}ms  ({gflops:.2f} GFLOPS)")

    return results


def bench_memory_bandwidth():
    """Measure memory bandwidth (read/write throughput)."""
    sizes_mb = [1, 4, 16, 64, 256]
    results = []

    for size_mb in sizes_mb:
        n = size_mb * 1024 * 1024 // 4  # float32 elements
        data = np.random.randn(n).astype(np.float32)

        # Read bandwidth
        def read_test():
            return np.sum(data)
        r_timing = time_fn(read_test, warmup=2, iterations=5)
        r_gbps = (size_mb / 1024) / (r_timing["mean_ms"] / 1000)

        # Write bandwidth
        out = np.empty_like(data)
        def write_test():
            np.copyto(out, data)
        w_timing = time_fn(write_test, warmup=2, iterations=5)
        w_gbps = (size_mb / 1024) / (w_timing["mean_ms"] / 1000)

        results.append({
            "size_mb": size_mb,
            "read_gbps": round(r_gbps, 2),
            "write_gbps": round(w_gbps, 2),
        })
        print(f"  {size_mb:>4}MB: read {r_gbps:>6.2f} GB/s | write {w_gbps:>6.2f} GB/s")

    return results


def bench_activation_functions():
    """Benchmark common activation functions on large arrays."""
    n = 10_000_000
    x = np.random.randn(n).astype(np.float32)
    results = {}

    fns = {
        "relu": lambda: np.maximum(x, 0),
        "sigmoid": lambda: 1 / (1 + np.exp(-x)),
        "tanh": lambda: np.tanh(x),
        "gelu": lambda: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))),
        "softmax": lambda: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))),
    }

    for name, fn in fns.items():
        timing = time_fn(fn, warmup=2, iterations=5)
        throughput = (n * 4) / (timing["mean_ms"] * 1e6)  # GB/s
        results[name] = {**timing, "throughput_gbps": round(throughput, 2)}
        print(f"  {name:<10}: {timing['mean_ms']:>8.2f}ms  ({throughput:.2f} GB/s)")

    return results


def bench_c_binary(binary_path, args=None):
    """Benchmark an arbitrary C binary."""
    cmd = [binary_path] + (args or [])
    timing = time_fn(
        lambda: subprocess.run(cmd, capture_output=True, check=True),
        warmup=1, iterations=5
    )
    return timing


def run_baseline():
    """Run full baseline benchmark suite."""
    print("=" * 60)
    print("BASELINE BENCHMARKS — NumPy on M3 Pro (Accelerate backend)")
    print("=" * 60)

    print("\n--- Matrix Multiplication ---")
    matmul = bench_matmul()

    print("\n--- Memory Bandwidth ---")
    membw = bench_memory_bandwidth()

    print("\n--- Activation Functions (10M elements) ---")
    activations = bench_activation_functions()

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": "numpy+accelerate",
        "matmul": matmul,
        "memory_bandwidth": membw,
        "activations": activations,
    }

    # Save results
    os.makedirs("data", exist_ok=True)
    out_path = "data/baseline_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


def compare(baseline_path, current_path):
    """Compare two benchmark results."""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(current_path) as f:
        current = json.load(f)

    print("=" * 60)
    print(f"COMPARISON: {baseline.get('backend', '?')} vs {current.get('backend', '?')}")
    print("=" * 60)

    if "matmul" in baseline and "matmul" in current:
        print("\n--- Matrix Multiplication ---")
        for b, c in zip(baseline["matmul"], current["matmul"]):
            speedup = b["mean_ms"] / c["mean_ms"] if c["mean_ms"] > 0 else 0
            print(f"  {b['size']}: {b['mean_ms']:.2f}ms -> {c['mean_ms']:.2f}ms ({speedup:.2f}x)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Runner")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("baseline", help="Run full baseline suite (numpy/Accelerate)")

    mm = sub.add_parser("matmul", help="Benchmark matrix multiply")
    mm.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 256, 512, 1024, 2048])

    sub.add_parser("membw", help="Benchmark memory bandwidth")
    sub.add_parser("activations", help="Benchmark activation functions")

    cmp = sub.add_parser("compare", help="Compare two benchmark results")
    cmp.add_argument("baseline")
    cmp.add_argument("current")

    cb = sub.add_parser("binary", help="Benchmark a C binary")
    cb.add_argument("path")
    cb.add_argument("--args", nargs="*", default=[])

    args = parser.parse_args()

    if args.command == "baseline":
        run_baseline()
    elif args.command == "matmul":
        bench_matmul(args.sizes)
    elif args.command == "membw":
        bench_memory_bandwidth()
    elif args.command == "activations":
        bench_activation_functions()
    elif args.command == "compare":
        compare(args.baseline, args.current)
    elif args.command == "binary":
        result = bench_c_binary(args.path, args.args)
        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
