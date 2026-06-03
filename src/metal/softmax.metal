// softmax.metal — Numerically Stable Softmax for M3 Pro
//
// Computes: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
//
// Handles arbitrary sequence lengths for attention scores.
// Two-pass approach:
//   Pass 1: Find max and compute sum(exp(x - max)) via online reduction
//   Pass 2: Normalize each element
//
// For d_model=4096 or seq_len up to 8192:
//   - Use a single threadgroup with SIMD reductions
//   - Threadgroup size: 256 (8 SIMD groups of 32)
//   - Each thread handles ceil(D / 256) elements
//
// Occupancy: 256 threads = 8 SIMD groups per threadgroup.
// Threadgroup memory: 8 halfs for cross-SIMD reduction (negligible).

#include <metal_stdlib>
using namespace metal;

// Single-row softmax. Each threadgroup processes one row.
// Grid: (1, num_rows, 1) threadgroups of (256, 1, 1) threads.
kernel void softmax_fp16(
    device const half* input     [[buffer(0)]],   // [rows, D]
    device half* output          [[buffer(1)]],   // [rows, D]
    constant uint& D             [[buffer(2)]],   // dimension to softmax over
    constant uint& rows          [[buffer(3)]],   // number of rows
    uint2 group_id               [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id.y;
    if (row >= rows) return;

    device const half* row_in = input + row * D;
    device half* row_out = output + row * D;

    // Threadgroup shared memory for cross-SIMD reduction
    // 8 SIMD groups -> 8 partial results
    threadgroup float tg_max[8];
    threadgroup float tg_sum[8];

    // --- Pass 1a: Find maximum value ---
    float local_max = -INFINITY;
    for (uint i = tid; i < D; i += 256) {
        float val = float(row_in[i]);
        local_max = max(local_max, val);
    }

    // SIMD reduction for max within each SIMD group
    local_max = simd_max(local_max);

    // Write per-SIMD max to shared memory
    if (simd_lane == 0) {
        tg_max[simd_idx] = local_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // First SIMD group reduces across all SIMD groups
    if (simd_idx == 0) {
        float val = (simd_lane < 8) ? tg_max[simd_lane] : -INFINITY;
        val = simd_max(val);
        if (simd_lane == 0) {
            tg_max[0] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = tg_max[0];

    // --- Pass 1b: Compute sum of exp(x - max) ---
    float local_sum = 0.0f;
    for (uint i = tid; i < D; i += 256) {
        float val = float(row_in[i]) - row_max;
        local_sum += exp(val);
    }

    // SIMD reduction for sum
    local_sum = simd_sum(local_sum);

    if (simd_lane == 0) {
        tg_sum[simd_idx] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_idx == 0) {
        float val = (simd_lane < 8) ? tg_sum[simd_lane] : 0.0f;
        val = simd_sum(val);
        if (simd_lane == 0) {
            tg_sum[0] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_sum = tg_sum[0];

    // --- Pass 2: Normalize ---
    float inv_sum = 1.0f / row_sum;
    for (uint i = tid; i < D; i += 256) {
        float val = float(row_in[i]) - row_max;
        row_out[i] = half(exp(val) * inv_sum);
    }
}

// Fused online softmax: single-pass Milakov-Gimelshein algorithm.
// Computes max and sum simultaneously, then normalizes.
// Better for very long sequences where two passes over memory hurts.
kernel void softmax_online_fp16(
    device const half* input     [[buffer(0)]],
    device half* output          [[buffer(1)]],
    constant uint& D             [[buffer(2)]],
    constant uint& rows          [[buffer(3)]],
    uint2 group_id               [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]],
    uint simd_lane               [[thread_index_in_simdgroup]],
    uint simd_idx                [[simdgroup_index_in_threadgroup]]
) {
    uint row = group_id.y;
    if (row >= rows) return;

    device const half* row_in = input + row * D;
    device half* row_out = output + row * D;

    threadgroup float simd_max_buf[8];
    threadgroup float simd_sum_buf[8];

    // Online computation: track running max and corrected sum
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (uint i = tid; i < D; i += 256) {
        float val = float(row_in[i]);
        float old_max = local_max;
        local_max = max(local_max, val);
        // Correct the running sum for the new max
        local_sum = local_sum * exp(old_max - local_max) + exp(val - local_max);
    }

    // SIMD reduction: combine (max, sum) pairs
    // For combining: new_max = max(a.max, b.max)
    //                new_sum = a.sum * exp(a.max - new_max) + b.sum * exp(b.max - new_max)
    for (uint offset = 16; offset >= 1; offset >>= 1) {
        float other_max = simd_shuffle_down(local_max, offset);
        float other_sum = simd_shuffle_down(local_sum, offset);
        float new_max = max(local_max, other_max);
        local_sum = local_sum * exp(local_max - new_max) + other_sum * exp(other_max - new_max);
        local_max = new_max;
    }

    if (simd_lane == 0) {
        simd_max_buf[simd_idx] = local_max;
        simd_sum_buf[simd_idx] = local_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cross-SIMD reduction in first SIMD group
    if (simd_idx == 0 && simd_lane < 8) {
        local_max = simd_max_buf[simd_lane];
        local_sum = simd_sum_buf[simd_lane];

        for (uint offset = 4; offset >= 1; offset >>= 1) {
            float other_max = simd_shuffle_down(local_max, offset);
            float other_sum = simd_shuffle_down(local_sum, offset);
            float new_max = max(local_max, other_max);
            local_sum = local_sum * exp(local_max - new_max) + other_sum * exp(other_max - new_max);
            local_max = new_max;
        }

        if (simd_lane == 0) {
            simd_max_buf[0] = local_max;
            simd_sum_buf[0] = local_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float final_max = simd_max_buf[0];
    float final_sum = simd_sum_buf[0];
    float inv_sum = 1.0f / final_sum;

    // Normalize
    for (uint i = tid; i < D; i += 256) {
        float val = float(row_in[i]) - final_max;
        row_out[i] = half(exp(val) * inv_sum);
    }
}
