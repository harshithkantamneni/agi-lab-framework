#!/usr/bin/env python3
"""checkpoint_load_dump.py -- Read V1..V5 checkpoint and print per-(layer,expert)
load EMA + bias (Q1 deliverable for D-095 kill-signal #3 monitoring).

The V5 checkpoint format (see src/training/checkpoint.{c,h}):
    [4]  magic     = 0x48535041  ("HSPA")
    [4]  version   = 5
    [HSPAConfig]   - 56 bytes (11 int32 + float + 2 enum)
    [TrainConfig]  - version-gated size (V1=100, V2=104, V3/V4=116, V5=168)
    [CheckpointMeta] = 24 bytes (i32 step, i32 epoch, i64 tokens, f32 ppl, f32 loss)
    [1]  has_adam
    [WEIGHTS]      - large block, deterministic order (see write_weights)
    [ADAM]         - same order, 2x for m and v (if has_adam)
    [ROUTER_BIAS]  - per layer: K floats expert_bias, K floats expert_load_ema
    [1]  has_default_moe                              -- V5 only
    [DEFAULT_MOE_EMA]  - if has_default_moe: per layer L*K*D floats
    [1]  has_stream                                   -- V4+
    [StreamLoaderState] - if has_stream

V5 FIX NOTE (Cycle 30, devops subtask 1 per D-097):
    The previous tool had TCFG_SIZE_BY_VERSION[5] = 128. True V5 TrainConfig
    size is 168 bytes (Cycle 29 Rev-2 added 40 bytes: 5 floats/ints +
    3 bools+pad for entropy-penalty / τ-anneal fields). This caused the
    parser to read CheckpointMeta 40 bytes early, producing the observed
    'step=1 epoch=1041865114 tokens=1063675494' scrambled output and
    negative EMAs. The downstream EMA block read was also missing the
    V5-only DEFAULT_MOE_EMA gate between router_bias and stream_state.
    Both fixed below. Sanity guard added to loudly flag out-of-band EMAs.

What this tool does (READ-ONLY, no model rebuild required):
    1. Validate magic + version.
    2. Parse HSPAConfig fields we need: d_model, n_layers, n_heads, n_kv_heads,
       head_dim, n_experts, d_ff, vocab_size.
    3. Compute byte sizes of WEIGHTS and ADAM blocks analytically.
    4. fseek past WEIGHTS (and ADAM if present).
    5. Read ROUTER_BIAS block: per layer, [K floats expert_bias][K floats EMA].
    6. Print per-(layer, expert) bias + EMA, plus per-layer max/min(EMA) ratio,
       which is the closest in-checkpoint proxy for the D-095 kill signal #3
       gate (max_per_expert_load / min_per_expert_load).
    7. Sanity-guard: flag EMA values outside [0,1] or per-layer sums != ~1.0.

Why this works (and its limit):
    `expert_load_ema` is updated in router_update_bias() each training step
    using `actual_load = expert_counts[j] / total_decisions`. It is therefore
    a FRACTIONAL-LOAD EMA (decay rate `bias_update_rate` = 0.01 by default).
    It is NOT a cumulative call counter, but it is monotonically related to
    recent load -- and the D-095 gate is fundamentally a fractional measure
    (load ratio), so EMA is the right quantity, not raw counts.

    Caveat: at step 500 with alpha=0.01, the EMA has effective horizon
    1/alpha = 100 steps, so it represents recent-100-step routing only.
    For early-training kill, that's actually preferable (catches active
    collapse, not stale uniform initialization).

Usage:
    python tools/checkpoint_load_dump.py CKPT_PATH [--gate 5.0] [--kill 10.0]
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

CKPT_MAGIC = 0x48535041  # "HSPA"
SUPPORTED_VERSIONS = (1, 2, 3, 4, 5)

# HSPAConfig layout (src/model/hspa_config.h). All ints are int32. Two enums
# (DType) at the end -- typically int (4B). Padding is implementation-defined,
# but the struct here is all 4-byte fields, so on aarch64-darwin we expect
# tightly packed 13 ints = 52 bytes. To be safe we read the first 11 int32
# fields explicitly.
HSPA_CONFIG_FIELDS = [
    "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim",
    "n_experts", "n_active", "d_ff", "vocab_size", "max_seq_len",
    "ipc_iterations",
]

# TrainConfig sizes per checkpoint version (see src/training/checkpoint.h
# and src/training/train_config.h).
#   V1 = 100 (before balance_h_target)
#   V2 = 104 (added balance_h_target)
#   V3/V4 = 116 (+ use_loss_free_balance, lfb_bias_step, lfb_ema_rate)
#   V5 = 168 (Cycle 29 Rev-2 current).
#     IMPORTANT: V5 shipped with TWO incompatible TrainConfig sizes under the
#     SAME version byte:
#       - V5-early (Cycle 27 Plan B):  128 bytes (+ use_default_moe,
#         default_moe_alpha, default_moe_sigma_init = +12 bytes over V4).
#       - V5-late  (Cycle 29 Rev-2):   168 bytes (+ entropy penalty + tau-anneal
#         fields = +40 bytes over V5-early).
#     The C trainer did NOT bump CKPT_VERSION between the two. This means we
#     must distinguish them at parse time. We do so by computing the expected
#     file size for each candidate tcfg layout given the HSPAConfig, and
#     picking the tcfg size that matches the actual file size.
#     V1-V4 entries below are still canonical (single size per version).
TCFG_SIZE_BY_VERSION = {1: 100, 2: 104, 3: 116, 4: 116}
V5_TCFG_CANDIDATES = (168, 128)  # try current first, fall back to early
V5_TCFG_CURRENT = 168

# StreamLoaderState size when has_stream=1 (Cycle 24+ struct). 536 bytes
# including compiler alignment padding on aarch64-darwin.
STREAM_LOADER_STATE_SIZE = 536

# CheckpointMeta = i32 step, i32 epoch, i64 tokens_trained, f32 best_ppl, f32 best_loss
CHECKPOINT_META_FMT = "<iiqff"
CHECKPOINT_META_SIZE = struct.calcsize(CHECKPOINT_META_FMT)

# HSPAConfig struct: 11 int32 fields + float rms_norm_eps + 2 enum (4B each)
# = 11*4 + 4 + 2*4 = 56 bytes; we read it entirely as raw and parse the head.
HSPA_CONFIG_RAW_SIZE = 56

# StreamLoaderState: rng_state (u64) + rng_inc (u64) + n_sources (i32) +
#   byte_offsets[32] (i64) + epochs[32] (i64).
# = 8 + 8 + 4 + 32*8 + 32*8 = 532 bytes (+ 4 pad after n_sources on aarch64
# due to i64 alignment -> 536). We don't parse this; only the presence byte.

# Sanity tolerance for sum-of-EMA-per-layer ~ 1.0. EMA is a decaying average
# of fractional loads (sum is 1.0 per step); it may drift slightly during
# warmup before converging.
EMA_SUM_TOLERANCE = 1e-4


def adam_byte_size(cfg: dict) -> int:
    D = cfg["d_model"]
    V = cfg["vocab_size"]
    L = cfg["n_layers"]
    K = cfg["n_experts"]
    nh = cfg["n_heads"]
    nkv = cfg["n_kv_heads"]
    hd = cfg["head_dim"]
    dff = cfg["d_ff"]

    embed = V * D
    per_layer = (
        D
        + D * (nh * hd)
        + D * (nkv * hd)
        + D * (nkv * hd)
        + (nh * hd) * D
        + D
        + D * K
        + D * K
        + K * (D * dff + D * dff + dff * D)
    )
    final_norm = D
    total_params = embed + L * per_layer + final_norm
    return total_params * 2 * 4  # *2 for m & v, *4 bytes/float


def weights_byte_size(cfg: dict) -> int:
    # Same as adam, but 1x not 2x (no m,v duplication), and no /2.
    return adam_byte_size(cfg) // 2


def default_moe_ema_byte_size(cfg: dict) -> int:
    return cfg["n_layers"] * cfg["n_experts"] * cfg["d_model"] * 4


def router_bias_byte_size(cfg: dict) -> int:
    return cfg["n_layers"] * (2 * cfg["n_experts"] * 4)


def detect_v5_tcfg_size(file_size: int, cfg: dict,
                         has_adam: int = 1) -> int | None:
    """Auto-detect V5 TrainConfig size by matching expected file size.

    V5 shipped with two incompatible TrainConfig layouts under the same
    version byte (Cycle 27 Plan B = 128 bytes; Cycle 29 Rev-2 = 168 bytes).
    The only reliable distinguisher without parsing the whole payload is
    total file size, since all other blocks are deterministic given
    HSPAConfig + has_adam + the V5 presence flags. In practice Cycle 27+
    V5 checkpoints all have has_default_moe=1 and has_stream=1 (both are
    mandatory for the Phase B training path), so we enumerate those
    combinations against the two candidate tcfg sizes.

    Returns the matching tcfg size in bytes, or None if no layout fits.
    """
    header = 8
    hspa = HSPA_CONFIG_RAW_SIZE
    meta = CHECKPOINT_META_SIZE
    adam_flag = 1
    moe_flag = 1
    stream_flag = 1

    w = weights_byte_size(cfg)
    a = adam_byte_size(cfg) if has_adam else 0
    rb = router_bias_byte_size(cfg)
    moe = default_moe_ema_byte_size(cfg)
    sls = STREAM_LOADER_STATE_SIZE

    for tcfg in V5_TCFG_CANDIDATES:
        # Enumerate (has_default_moe, has_stream) presence combos.
        for has_moe in (1, 0):
            for has_stream in (1, 0):
                total = (
                    header + hspa + tcfg + meta + adam_flag
                    + w + a + rb
                    + moe_flag + (moe if has_moe else 0)
                    + stream_flag + (sls if has_stream else 0)
                )
                if total == file_size:
                    return tcfg
    return None


def sanity_check_ema(layer_ema: list[list[float]], K: int) -> list[str]:
    """Loud-warning sanity guard on the parsed EMA block.

    Returns a list of warning strings (empty if all OK). The caller prints
    them prominently at the top of the output so a broken parse cannot
    quietly mislead the Director's kill post-mortem.
    """
    warnings: list[str] = []
    for l, ema in enumerate(layer_ema):
        # Range [0, 1]: EMA of fractional load cannot be outside this.
        lo, hi = min(ema), max(ema)
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            warnings.append(
                f"layer {l}: EMA out of [0,1] (min={lo:.4f} max={hi:.4f}). "
                f"Likely a parse-offset bug; DO NOT trust this output."
            )
        s = sum(ema)
        # Per-layer EMA should sum to ~1.0 once converged. Tolerate some
        # drift during warmup (K*alpha typical up to ~8%), so use a wider
        # band for the warning (not fatal, just visible).
        # Actual contract is s ~= 1.0 +/- O(K*alpha); we require it to be
        # at least broadly in the [0.5, 1.5] band to consider the parse
        # plausible. The tighter +/- EMA_SUM_TOLERANCE is an info-only note.
        if s < 0.5 or s > 1.5:
            warnings.append(
                f"layer {l}: EMA sum = {s:.4f} (expected ~1.0). "
                f"Either EMA hasn't warmed up, or parse is broken."
            )
    return warnings


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("ckpt_path", type=Path)
    ap.add_argument("--gate", type=float, default=5.0,
                    help="D-095 revised gate ratio (max/min EMA).")
    ap.add_argument("--kill", type=float, default=10.0,
                    help="D-095 kill threshold ratio.")
    args = ap.parse_args(argv)

    if not args.ckpt_path.exists():
        print(f"FATAL: checkpoint not found: {args.ckpt_path}", file=sys.stderr)
        return 2

    file_size = args.ckpt_path.stat().st_size

    with args.ckpt_path.open("rb") as f:
        magic, version = struct.unpack("<II", f.read(8))
        if magic != CKPT_MAGIC:
            print(f"FATAL: bad magic 0x{magic:08x}, expected 0x{CKPT_MAGIC:08x}",
                  file=sys.stderr)
            return 2
        if version not in SUPPORTED_VERSIONS:
            print(f"FATAL: unsupported version {version}", file=sys.stderr)
            return 2

        # HSPAConfig: parse only the head fields (the rest are float/enum we
        # don't need for indexing).
        raw_cfg = f.read(HSPA_CONFIG_RAW_SIZE)
        ints = struct.unpack("<" + "i" * len(HSPA_CONFIG_FIELDS),
                              raw_cfg[: len(HSPA_CONFIG_FIELDS) * 4])
        cfg = dict(zip(HSPA_CONFIG_FIELDS, ints))

        # TrainConfig: version-specific size. For V5 we MUST auto-detect
        # between the two historical layouts (Cycle 27 = 128 bytes,
        # Cycle 29 = 168 bytes). See detect_v5_tcfg_size() docstring.
        if version == 5:
            tcfg_size = detect_v5_tcfg_size(file_size, cfg, has_adam=1)
            if tcfg_size is None:
                # Fall back to current (168) so downstream sanity guards
                # can still flag the mis-parse loudly.
                tcfg_size = V5_TCFG_CURRENT
                print(
                    f"# WARNING: could not auto-detect V5 TrainConfig size "
                    f"from file size {file_size}. Assuming {tcfg_size}; "
                    f"check sanity warnings.", file=sys.stderr)
            else:
                # Informational marker so a future reader can see which
                # sub-variant was detected.
                print(f"# V5 sub-variant: TrainConfig={tcfg_size} bytes "
                      f"(auto-detected from file size {file_size})")
        else:
            tcfg_size = TCFG_SIZE_BY_VERSION[version]
        f.seek(tcfg_size, 1)

        # CheckpointMeta
        meta_buf = f.read(CHECKPOINT_META_SIZE)
        step, epoch, tokens, best_ppl, best_loss = struct.unpack(
            CHECKPOINT_META_FMT, meta_buf)

        # has_adam flag
        has_adam = struct.unpack("<B", f.read(1))[0]

        L, K = cfg["n_layers"], cfg["n_experts"]
        D = cfg["d_model"]
        print(f"# checkpoint: {args.ckpt_path}")
        print(f"# version={version} step={step} epoch={epoch} tokens={tokens} "
              f"best_ppl={best_ppl:.2f} best_loss={best_loss:.4f}")
        print(f"# config: L={L} D={D} K={K} k={cfg['n_active']} "
              f"V={cfg['vocab_size']} d_ff={cfg['d_ff']}")
        print(f"# has_adam={has_adam}")

        # Skip weights
        wsz = weights_byte_size(cfg)
        f.seek(wsz, 1)
        # Skip Adam if present
        if has_adam:
            asz = adam_byte_size(cfg)
            f.seek(asz, 1)

        # ROUTER_BIAS block: per layer, K floats expert_bias, K floats EMA.
        layer_bias: list[list[float]] = [list() for _ in range(L)]
        layer_ema: list[list[float]] = [list() for _ in range(L)]
        for l in range(L):
            buf = f.read(K * 4)
            layer_bias[l] = list(struct.unpack(f"<{K}f", buf))
            buf = f.read(K * 4)
            layer_ema[l] = list(struct.unpack(f"<{K}f", buf))

        # V5: optional Default MoE EMA block (1-byte presence flag, then
        # per layer K*D floats). We do NOT dump this (large: 0.125 MB at 50M)
        # but we MUST consume the bytes if present so that any downstream
        # tooling that reads the stream block still works on the same file.
        has_default_moe = 0
        if version >= 5:
            has_default_moe = struct.unpack("<B", f.read(1))[0]
            if has_default_moe:
                # Skip L*K*D floats.
                skip = L * K * D * 4
                f.seek(skip, 1)

        # V4+: optional StreamLoaderState (informational; we skip the payload).
        has_stream = 0
        if version >= 4:
            has_stream = struct.unpack("<B", f.read(1))[0]
            # We don't parse the StreamLoaderState body here; leaving it
            # implicit preserves a single read path. The presence byte is
            # useful metadata to print.

    # ---- Regression guard (§5 subtask 1): loudly flag impossible EMA values ----
    warnings = sanity_check_ema(layer_ema, K)
    if warnings:
        bar = "!" * 72
        print()
        print(bar)
        print("SANITY WARNING: parsed EMA values look wrong. This tool is a")
        print("POST-MORTEM AUTHORITY for the D-095 kill gate -- do not trust")
        print("the numbers below until the warnings are explained:")
        for w in warnings:
            print(f"  - {w}")
        print(bar)
        print()

    # ---- Metadata banner tail ----
    print(f"# has_default_moe={has_default_moe} has_stream={has_stream}")

    # ---- Print per-(layer, expert) ----
    print()
    print("layer expert  bias        load_ema")
    for l in range(L):
        for e in range(K):
            print(f"{l:5d} {e:6d}  {layer_bias[l][e]:+.5f}  {layer_ema[l][e]:.6f}")
        print(f"{'-' * 50}")

    # ---- Per-layer max/min ratio + uniform target ----
    uniform = 1.0 / K
    print()
    print(f"layer  min_ema    max_ema    ratio   gate({args.gate}) "
          f"kill({args.kill})  uniform={uniform:.5f}")
    worst_ratio = 0.0
    n_gate_violations = 0
    n_kill_violations = 0
    for l in range(L):
        mn = min(layer_ema[l])
        mx = max(layer_ema[l])
        ratio = (mx / mn) if mn > 1e-12 else float("inf")
        gate_hit = ratio > args.gate
        kill_hit = ratio > args.kill
        if gate_hit:
            n_gate_violations += 1
        if kill_hit:
            n_kill_violations += 1
        worst_ratio = max(worst_ratio, ratio)
        flag = ""
        if kill_hit:
            flag = "  KILL"
        elif gate_hit:
            flag = "  GATE-FAIL"
        print(f"{l:5d}  {mn:.6f}   {mx:.6f}   {ratio:6.2f}   "
              f"{str(gate_hit):>5s}      {str(kill_hit):>5s}{flag}")

    print()
    print(f"# WORST layer ratio: {worst_ratio:.2f}")
    print(f"# Layers exceeding gate ({args.gate}): {n_gate_violations}/{L}")
    print(f"# Layers exceeding kill ({args.kill}): {n_kill_violations}/{L}")
    print()
    print("# NOTE: expert_load_ema has horizon 1/bias_update_rate steps "
          "(default 0.01 -> ~100 step EWMA). At step 500 it reflects steps "
          "~400-500. EMA is fractional load (in [0,1]), summing across K "
          "experts approximates 1.0 (modulo EMA lag).")
    # Print sum-of-layer-0 as the canonical sanity number.
    s0 = sum(layer_ema[0])
    print(f"# SUM of layer-0 EMA = {s0:.4f} (sanity: should be ~1.0 once "
          f"EMA converges; tolerance +/- {EMA_SUM_TOLERANCE} for perfectly "
          f"normalized, wider band acceptable during warmup)")

    return 0 if n_kill_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
