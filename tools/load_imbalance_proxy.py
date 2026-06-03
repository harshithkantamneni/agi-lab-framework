#!/usr/bin/env python3
"""load_imbalance_proxy.py -- Estimate per-expert load imbalance from training log.

Q2 deliverable for D-095 kill signal #3 monitoring.

The current scale_experiment log emits, per logged step:
    step | total_loss (pred + bal + lm) | ppl | gnorm | mean_entropy | vn_d | epoch | ms

Per-expert load is NOT emitted, but `mean_entropy` (averaged across L layers)
is a strong proxy under top-k routing with LFB:

  - At uniform load (1/K each), single-layer entropy = ln(K).
    For K=8: H_uniform = ln(8) ~= 2.0794.
  - At max-collapse (one expert wins all tokens), H -> 0.
  - General: for a probability vector p over K experts,
    if max/min ratio is r and the rest are uniform-ish, entropy decays
    monotonically with r.

We invert that relationship for a 2-bin coarse model that is
ROBUST (not exact): assume one "hot" expert has fraction p_hot,
the remaining K-1 share (1 - p_hot) uniformly. Then:
    H = -p_hot*log(p_hot) - (1 - p_hot)*log((1 - p_hot)/(K-1))
We solve for p_hot given observed H, then ratio = p_hot / ((1-p_hot)/(K-1)).

Caveats:
  - This is a LOWER BOUND on the true max/min ratio: real distributions
    have MORE skew than the 2-bin model implies, because multiple cold
    experts can be near-zero. Treat the estimate as conservative.
  - Entropy here is the router's pre-noise softmax entropy (mu_probs),
    not the post-top-k empirical assignment entropy. They correlate but
    are not identical; treat threshold crossings as warnings, not gates.

Usage:
    python tools/load_imbalance_proxy.py LOG_PATH [--K 8] [--threshold 5.0]

Exit 0 on parse success; exit 1 if any checkpoint estimate exceeds threshold.
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

# Match log lines like:
#   500  |   9.0696 ( 0.000 + 0.0000 +  9.070) |   8687.18 |  8.193 | 0.27454 | 0.0000 |  0 | 2187
LOG_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([-\d.]+)\s*\(\s*([-\d.]+)\s*\+\s*([-\d.]+)\s*\+\s*([-\d.]+)\s*\)"
    r"\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)"
)


def solve_phot_from_entropy(H: float, K: int) -> float:
    """Invert 2-bin model: find p_hot in [1/K, 1) s.t. observed entropy is H.
    Returns 1/K when H >= ln(K) (already uniform / no concentration)."""
    H_uniform = math.log(K)
    if H >= H_uniform - 1e-6:
        return 1.0 / K
    # Bisection on p_hot in (1/K, 1 - 1e-9).
    lo, hi = 1.0 / K, 1.0 - 1e-9

    def H_of(p):
        if p >= 1.0 - 1e-12:
            return 0.0
        rest = (1.0 - p) / (K - 1)
        eps = 1e-12
        return -p * math.log(p + eps) - (K - 1) * rest * math.log(rest + eps)

    # H is monotonically decreasing in p_hot for p_hot in [1/K, 1).
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if H_of(mid) > H:
            lo = mid  # need MORE concentration -> larger p
        else:
            hi = mid
    return 0.5 * (lo + hi)


def ratio_from_entropy(H: float, K: int) -> float:
    """Lower-bound estimate of max/min expert load ratio."""
    p_hot = solve_phot_from_entropy(H, K)
    p_cold = (1.0 - p_hot) / (K - 1)
    if p_cold <= 1e-12:
        return float("inf")
    return p_hot / p_cold


def parse_log(path: Path):
    """Yield (step, lm_loss, ppl, gnorm, entropy) tuples for matched lines."""
    with path.open("r", errors="replace") as f:
        for line in f:
            m = LOG_RE.match(line)
            if not m:
                continue
            step = int(m.group(1))
            lm = float(m.group(5))
            ppl = float(m.group(6))
            gnorm = float(m.group(7))
            entropy = float(m.group(8))
            yield step, lm, ppl, gnorm, entropy


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("log_path", type=Path, help="Training log file path.")
    ap.add_argument("--K", type=int, default=8,
                    help="Number of experts per layer (default 8).")
    ap.add_argument("--threshold", type=float, default=5.0,
                    help="Max allowed lower-bound max/min ratio "
                         "(D-095 gate=5.0; kill=10.0).")
    ap.add_argument("--checkpoint-stride", type=int, default=20,
                    help="Subsample stride; D-095 says 'every 20 steps'.")
    args = ap.parse_args(argv)

    if not args.log_path.exists():
        print(f"FATAL: log not found: {args.log_path}", file=sys.stderr)
        return 2

    H_uniform = math.log(args.K)
    print(f"# K={args.K}  H_uniform=ln(K)={H_uniform:.4f}  "
          f"gate_ratio={args.threshold}  stride={args.checkpoint_stride}")
    print("# step  entropy  ratio_lb  ratio_kill_signal_hit  pred_p_hot  "
          "gnorm  ppl")

    consecutive_above_kill = 0
    worst_ratio = 0.0
    worst_step = -1
    any_violation = False
    rows = []
    for (step, _lm, ppl, gnorm, ent) in parse_log(args.log_path):
        if step % args.checkpoint_stride != 0 and step > 0:
            continue
        # Skip the warmup region (D-095: "after step 500").
        rows.append((step, ent, ppl, gnorm))

    for step, ent, ppl, gnorm in rows:
        ratio = ratio_from_entropy(ent, args.K)
        p_hot = solve_phot_from_entropy(ent, args.K)
        kill_hit = ratio > 10.0  # D-095 kill threshold
        gate_hit = ratio > args.threshold and step >= 500
        if kill_hit and step >= 500:
            consecutive_above_kill += 1
        else:
            consecutive_above_kill = 0
        if ratio > worst_ratio:
            worst_ratio = ratio
            worst_step = step
        if gate_hit:
            any_violation = True
        flag = ""
        if consecutive_above_kill >= 3:
            flag = "  <-- KILL: 3 consecutive checkpoints > 10:1"
        elif kill_hit and step >= 500:
            flag = "  <-- KILL CANDIDATE (>10:1)"
        elif gate_hit:
            flag = "  <-- GATE FAIL (>5:1)"
        print(f"{step:5d}  {ent:7.4f}  {ratio:8.2f}  {str(kill_hit):>5s}        "
              f"{p_hot:6.4f}  {gnorm:6.3f}  {ppl:9.2f}{flag}")

    print(f"# WORST: step={worst_step} ratio_lb={worst_ratio:.2f}")
    print(f"# NOTE: ratio is a LOWER BOUND from a 2-bin entropy model. "
          f"True per-expert max/min may be higher.")
    return 1 if any_violation else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
