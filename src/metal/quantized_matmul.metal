// quantized_matmul.metal — 4-bit Quantized Matrix Multiplication for M3 Pro
//
// Computes: output[M,N] = input[M,K] x weights_q4[K,N]
//   input:   FP16 [M, K]
//   weights: INT4 packed, block-quantized (block_size=32)
//   output:  FP16 [M, N]
//
// Quantization format (per block of 32 values):
//   [scale: half (2B)] [zero: half (2B)] [16 bytes packed INT4 (2 values per byte)]
//   Total: 20 bytes per block of 32 weights
//   Dequant: value = (int4_val - zero) * scale
//
// Weight layout: weights[K/32][N][20]
//   For column n, block b: offset = (b * N + n) * 20
//
// Threadgroup tiling strategy for M3 Pro (18 GPU cores, SIMD width 32):
//   - Threadgroup: 32x8 = 256 threads
//   - Each threadgroup computes a TILE_M x TILE_N (32x32) output tile
//   - Threads iterate over K in blocks of 32 (matching quantization block size)
//   - Threadgroup memory caches input tile for reuse across N dimension
//   - SIMD reduction across K-block partial sums
//
// Occupancy rationale:
//   - 256 threads/threadgroup = 8 SIMD groups
//   - M3 Pro supports up to 1024 threads/threadgroup
//   - 256 chosen to balance occupancy with register pressure from accumulator state
//   - 18 GPU cores can run multiple threadgroups concurrently

#include <metal_stdlib>
using namespace metal;

// Tile dimensions
constant uint TILE_M = 32;
constant uint TILE_N = 32;
constant uint BLOCK_SIZE = 32;   // quantization block size
constant uint BLOCK_BYTES = 20;  // 2 (scale) + 2 (zero) + 16 (data) bytes per block

// Decode a single INT4 value from a packed byte.
// Even index (idx & 1 == 0): low nibble. Odd index: high nibble.
inline half dequantize_q4(uint8_t packed_byte, uint nibble_idx, half scale, half zero) {
    uint8_t val;
    if (nibble_idx & 1) {
        val = (packed_byte >> 4) & 0x0F;
    } else {
        val = packed_byte & 0x0F;
    }
    return (half(val) - zero) * scale;
}

// Main quantized matmul kernel.
// Grid: ceil(N/TILE_N) x ceil(M/TILE_M) threadgroups
// Threadgroup: 32 x 8 threads
//   - tid.x in [0..31] covers TILE_N columns (one per thread)
//   - tid.y in [0..7] covers TILE_M rows (each thread handles 4 rows)
kernel void matmul_q4_fp16(
    device const half* input        [[buffer(0)]],   // [M, K]
    device const uint8_t* weights   [[buffer(1)]],   // [K/32, N, 20] packed Q4
    device half* output             [[buffer(2)]],   // [M, N]
    constant uint& M                [[buffer(3)]],
    constant uint& K                [[buffer(4)]],
    constant uint& N                [[buffer(5)]],
    uint2 group_id                  [[threadgroup_position_in_grid]],
    uint2 tid                       [[thread_position_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]],
    uint simd_idx                   [[simdgroup_index_in_threadgroup]]
) {
    // Which output tile this threadgroup is responsible for
    uint tile_row_start = group_id.y * TILE_M;  // starting M row
    uint tile_col_start = group_id.x * TILE_N;  // starting N col

    // Each thread handles ROWS_PER_THREAD rows and 1 column
    // tid.x = column within tile [0..31], tid.y = row group [0..7]
    // Each of the 8 row-groups handles 4 rows -> 32 rows total
    const uint ROWS_PER_THREAD = 4;
    uint col = tile_col_start + tid.x;
    uint row_base = tile_row_start + tid.y * ROWS_PER_THREAD;

    // Accumulators for this thread's output values
    half acc[4] = {half(0), half(0), half(0), half(0)};

    // Threadgroup shared memory for input tile:
    // Cache BLOCK_SIZE (32) columns of input for TILE_M (32) rows
    threadgroup half input_tile[TILE_M * BLOCK_SIZE]; // 32 * 32 = 1024 halfs = 2KB

    // Number of quantization blocks along K
    uint num_k_blocks = K / BLOCK_SIZE;

    // Iterate over K in blocks of 32
    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint k_offset = kb * BLOCK_SIZE;

        // Collaboratively load input tile into threadgroup memory.
        // 256 threads loading 1024 elements = 4 elements per thread.
        uint flat_tid = tid.y * 32 + tid.x; // [0..255]
        for (uint i = 0; i < 4; i++) {
            uint elem_idx = flat_tid * 4 + i;
            uint r = elem_idx / BLOCK_SIZE; // row in [0..31]
            uint c = elem_idx % BLOCK_SIZE; // col in [0..31]
            uint global_row = tile_row_start + r;
            if (global_row < M) {
                input_tile[r * BLOCK_SIZE + c] = input[global_row * K + k_offset + c];
            } else {
                input_tile[r * BLOCK_SIZE + c] = half(0);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Early exit if this thread's column is out of bounds
        if (col < N) {
            // Fetch the quantized weight block for this column and K-block
            // Weight layout: block at (kb, col) -> offset = (kb * N + col) * BLOCK_BYTES
            uint block_offset = (kb * N + col) * BLOCK_BYTES;

            // Read scale and zero from block header
            // Scale is at offset 0-1, zero is at offset 2-3 (as half)
            half scale = *reinterpret_cast<device const half*>(weights + block_offset);
            half zero  = *reinterpret_cast<device const half*>(weights + block_offset + 2);

            // Pointer to the 16 packed INT4 bytes (32 values)
            device const uint8_t* packed = weights + block_offset + 4;

            // Dequantize all 32 weight values for this column's K-block
            half w_vals[32];
            for (uint i = 0; i < 16; i++) {
                uint8_t byte_val = packed[i];
                w_vals[i * 2]     = (half(byte_val & 0x0F) - zero) * scale;
                w_vals[i * 2 + 1] = (half((byte_val >> 4) & 0x0F) - zero) * scale;
            }

            // Compute dot product: accumulate over the 32 K-elements
            for (uint r = 0; r < ROWS_PER_THREAD; r++) {
                uint local_row = tid.y * ROWS_PER_THREAD + r;
                half dot = half(0);

                // Unrolled 8x for performance
                for (uint ki = 0; ki < 32; ki += 8) {
                    dot += input_tile[local_row * BLOCK_SIZE + ki]     * w_vals[ki];
                    dot += input_tile[local_row * BLOCK_SIZE + ki + 1] * w_vals[ki + 1];
                    dot += input_tile[local_row * BLOCK_SIZE + ki + 2] * w_vals[ki + 2];
                    dot += input_tile[local_row * BLOCK_SIZE + ki + 3] * w_vals[ki + 3];
                    dot += input_tile[local_row * BLOCK_SIZE + ki + 4] * w_vals[ki + 4];
                    dot += input_tile[local_row * BLOCK_SIZE + ki + 5] * w_vals[ki + 5];
                    dot += input_tile[local_row * BLOCK_SIZE + ki + 6] * w_vals[ki + 6];
                    dot += input_tile[local_row * BLOCK_SIZE + ki + 7] * w_vals[ki + 7];
                }
                acc[r] += dot;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Handle remainder K elements if K is not a multiple of BLOCK_SIZE
    uint k_remainder = K % BLOCK_SIZE;
    if (k_remainder > 0 && col < N) {
        uint kb = num_k_blocks;
        uint k_offset = kb * BLOCK_SIZE;
        uint block_offset = (kb * N + col) * BLOCK_BYTES;

        half scale = *reinterpret_cast<device const half*>(weights + block_offset);
        half zero  = *reinterpret_cast<device const half*>(weights + block_offset + 2);
        device const uint8_t* packed = weights + block_offset + 4;

        for (uint r = 0; r < ROWS_PER_THREAD; r++) {
            uint global_row = row_base + r;
            if (global_row < M) {
                half dot = half(0);
                for (uint ki = 0; ki < k_remainder; ki++) {
                    half in_val = input[global_row * K + k_offset + ki];
                    uint8_t byte_val = packed[ki / 2];
                    half w_val;
                    if (ki & 1) {
                        w_val = (half((byte_val >> 4) & 0x0F) - zero) * scale;
                    } else {
                        w_val = (half(byte_val & 0x0F) - zero) * scale;
                    }
                    dot += in_val * w_val;
                }
                acc[r] += dot;
            }
        }
    }

    // Write results to output
    if (col < N) {
        for (uint r = 0; r < ROWS_PER_THREAD; r++) {
            uint global_row = row_base + r;
            if (global_row < M) {
                output[global_row * N + col] = acc[r];
            }
        }
    }
}

// Batch=1 optimized variant: M=1, single-row vector-matrix multiply.
// This is the inference hot path. Purely memory-bound.
// Strategy: each threadgroup processes a chunk of N columns.
// No input tiling needed (single row fits in registers).
// Threadgroup: 256 threads, each handles 1 output column.
kernel void vecmat_q4_fp16(
    device const half* input        [[buffer(0)]],   // [1, K]
    device const uint8_t* weights   [[buffer(1)]],   // [K/32, N, 20] packed Q4
    device half* output             [[buffer(2)]],   // [1, N]
    constant uint& K                [[buffer(3)]],
    constant uint& N                [[buffer(4)]],
    uint gid                        [[thread_position_in_grid]]
) {
    if (gid >= N) return;

    uint num_k_blocks = K / BLOCK_SIZE;
    half acc = half(0);

    for (uint kb = 0; kb < num_k_blocks; kb++) {
        uint k_offset = kb * BLOCK_SIZE;
        uint block_offset = (kb * N + gid) * BLOCK_BYTES;

        half scale = *reinterpret_cast<device const half*>(weights + block_offset);
        half zero  = *reinterpret_cast<device const half*>(weights + block_offset + 2);
        device const uint8_t* packed = weights + block_offset + 4;

        // Accumulate dot product over 32-element block
        half dot = half(0);
        for (uint i = 0; i < 16; i++) {
            uint8_t byte_val = packed[i];
            half w0 = (half(byte_val & 0x0F) - zero) * scale;
            half w1 = (half((byte_val >> 4) & 0x0F) - zero) * scale;

            uint ki = i * 2;
            dot += input[k_offset + ki]     * w0;
            dot += input[k_offset + ki + 1] * w1;
        }
        acc += dot;
    }

    output[gid] = acc;
}
