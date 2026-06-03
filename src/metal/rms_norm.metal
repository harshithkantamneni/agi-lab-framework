// rms_norm.metal — RMS Normalization for M3 Pro
//
// Computes: output_i = input_i * weight_i / sqrt(mean(input^2) + eps)
//
// Where mean(input^2) = (1/D) * sum(input_i^2) for i in [0, D)
//
// For d_model=4096:
//   - Single threadgroup per row: 256 threads
//   - Each thread reduces ~16 elements
//   - SIMD reduction within groups, then cross-SIMD in threadgroup memory
//   - Float32 accumulation for numerical stability, FP16 I/O
//
// Threadgroup: 256 threads = 8 SIMD groups of 32
// Threadgroup memory: 8 floats for cross-SIMD reduction

#include <metal_stdlib>
using namespace metal;

// Standard RMS Norm: one threadgroup per row
kernel void rms_norm_fp16(
    device const half* input     [[buffer(0)]],   // [rows, D]
    device const half* weight    [[buffer(1)]],   // [D] learnable scale
    device half* output          [[buffer(2)]],   // [rows, D]
    constant uint& D             [[buffer(3)]],
    constant uint& rows          [[buffer(4)]],
    constant float& eps          [[buffer(5)]],
    uint2 group_id               [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id.y;
    if (row >= rows) return;

    device const half* row_in = input + row * D;
    device half* row_out = output + row * D;

    // Shared memory for cross-SIMD reduction
    threadgroup float simd_partial[8];

    // --- Compute sum of squares ---
    float sum_sq = 0.0f;
    for (uint i = tid; i < D; i += 256) {
        float val = float(row_in[i]);
        sum_sq += val * val;
    }

    // SIMD reduction
    sum_sq = simd_sum(sum_sq);

    if (simd_lane == 0) {
        simd_partial[simd_idx] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SIMD reduction in first group
    if (simd_idx == 0) {
        float val = (simd_lane < 8) ? simd_partial[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            simd_partial[0] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = simd_partial[0];
    float rms = rsqrt(total_sum_sq / float(D) + eps);

    // --- Normalize and scale ---
    for (uint i = tid; i < D; i += 256) {
        float val = float(row_in[i]);
        float w = float(weight[i]);
        row_out[i] = half(val * rms * w);
    }
}

// Fused RMS Norm + Residual Add:
// output = rms_norm(input + residual) * weight
// Saves a memory round-trip vs separate add + norm
kernel void rms_norm_residual_fp16(
    device const half* input     [[buffer(0)]],   // [rows, D]
    device const half* residual  [[buffer(1)]],   // [rows, D]
    device const half* weight    [[buffer(2)]],   // [D]
    device half* output          [[buffer(3)]],   // [rows, D]
    device half* normed_input    [[buffer(4)]],   // [rows, D] store input+residual for backward
    constant uint& D             [[buffer(5)]],
    constant uint& rows          [[buffer(6)]],
    constant float& eps          [[buffer(7)]],
    uint2 group_id               [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id.y;
    if (row >= rows) return;

    device const half* row_in = input + row * D;
    device const half* row_res = residual + row * D;
    device half* row_out = output + row * D;
    device half* row_normed = normed_input + row * D;

    threadgroup float simd_partial[8];

    // First pass: add residual, compute sum of squares, store sum
    float sum_sq = 0.0f;
    for (uint i = tid; i < D; i += 256) {
        float val = float(row_in[i]) + float(row_res[i]);
        row_normed[i] = half(val); // store for later use
        sum_sq += val * val;
    }

    // SIMD reduction
    sum_sq = simd_sum(sum_sq);

    if (simd_lane == 0) {
        simd_partial[simd_idx] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        float val = (simd_lane < 8) ? simd_partial[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            simd_partial[0] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total_sum_sq = simd_partial[0];
    float rms = rsqrt(total_sum_sq / float(D) + eps);

    // Second pass: normalize and scale
    for (uint i = tid; i < D; i += 256) {
        float val = float(row_normed[i]);
        float w = float(weight[i]);
        row_out[i] = half(val * rms * w);
    }
}
