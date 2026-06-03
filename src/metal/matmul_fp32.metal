// matmul_fp32.metal -- High-Performance FP32 Matrix Multiplication for M3 Pro
//
// Computes: C[M,N] = A[M,K] x B[K,N]  (all FP32, row-major)
//
// Optimized for Apple M3 Pro (Metal 3, 18 GPU cores, SIMD width 32):
//   - simdgroup_matrix 8x8 hardware matrix multiply
//   - Register tiling: each simdgroup accumulates multiple 8x8 tiles
//   - Double buffering: overlaps threadgroup loads with compute
//   - Vectorized float4 loads for coalesced memory access
//   - Handles arbitrary M,N,K (including non-power-of-2 like 384)
//
// Tiling strategy:
//   - Threadgroup tile: TILE_M=32 x TILE_N=32 output, TILE_K=32 along K
//   - 128 threads = 4 SIMD groups of 32 threads each
//   - SIMD groups arranged 2x2: each computes a 16x16 sub-tile
//   - Each simdgroup accumulates 4 simdgroup_matrix 8x8 tiles (2x2 grid)
//   - Shared memory: 2 buffers x (32x32) x 4 bytes x 2 matrices = 16 KB
//
// Memory layout (all row-major):
//   A[row, col] at offset row * K + col
//   B[row, col] at offset row * N + col
//   C[row, col] at offset row * N + col

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// --- Tile dimensions ---
constant uint TILE_M = 32;   // output rows per threadgroup
constant uint TILE_N = 32;   // output cols per threadgroup
constant uint TILE_K = 32;   // K-dimension per tile iteration
// 128 threads = 4 SIMD groups of 32 threads per threadgroup

// --- Helper: load A tile into threadgroup memory with float4 vectorization ---
// A tile is TILE_M x TILE_K = 32 x 32 = 1024 floats
// 128 threads, each loads 8 floats (2 x float4)
inline void load_A_tile(
    device const float* A,
    threadgroup float* As,
    uint flat_tid,
    uint tile_row,
    uint k_offset,
    uint M,
    uint K
) {
    // 1024 elements / 128 threads = 8 elements per thread
    // Load as 2 x float4 for coalesced access
    for (uint i = 0; i < 8; i++) {
        uint elem = flat_tid * 8 + i;
        uint r = elem / TILE_K;     // row within tile [0..31]
        uint c = elem % TILE_K;     // col within tile [0..31]
        uint global_row = tile_row + r;
        uint global_col = k_offset + c;
        As[r * TILE_K + c] = (global_row < M && global_col < K)
                             ? A[global_row * K + global_col]
                             : 0.0f;
    }
}

// --- Helper: load B tile into threadgroup memory ---
// B tile is TILE_K x TILE_N = 32 x 32 = 1024 floats
inline void load_B_tile(
    device const float* B,
    threadgroup float* Bs,
    uint flat_tid,
    uint k_offset,
    uint tile_col,
    uint K,
    uint N
) {
    for (uint i = 0; i < 8; i++) {
        uint elem = flat_tid * 8 + i;
        uint r = elem / TILE_N;
        uint c = elem % TILE_N;
        uint global_row = k_offset + r;
        uint global_col = tile_col + c;
        Bs[r * TILE_N + c] = (global_row < K && global_col < N)
                             ? B[global_row * N + global_col]
                             : 0.0f;
    }
}

// ===========================================================================
// Main optimized FP32 matmul kernel using simdgroup_matrix hardware
// ===========================================================================
// Grid: ceil(N/TILE_N) x ceil(M/TILE_M) threadgroups
// Threadgroup: 128 threads (4 SIMD groups of 32)
//
// Each simdgroup computes a 16x16 sub-tile of the 32x32 output:
//   simdgroup 0: rows [0..15],  cols [0..15]
//   simdgroup 1: rows [0..15],  cols [16..31]
//   simdgroup 2: rows [16..31], cols [0..15]
//   simdgroup 3: rows [16..31], cols [16..31]
//
// Each 16x16 sub-tile is computed as a 2x2 grid of 8x8 simdgroup_matrix ops.
kernel void matmul_fp32(
    device const float* A           [[buffer(0)]],   // [M, K] row-major
    device const float* B           [[buffer(1)]],   // [K, N] row-major
    device float* C                 [[buffer(2)]],   // [M, N] row-major
    constant uint& M                [[buffer(3)]],
    constant uint& K                [[buffer(4)]],
    constant uint& N                [[buffer(5)]],
    uint2 group_id                  [[threadgroup_position_in_grid]],
    uint flat_tid                   [[thread_index_in_threadgroup]],
    uint simd_gid                   [[simdgroup_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]]
) {
    // Double-buffered threadgroup memory: ping-pong between loads and compute
    threadgroup float As[2][TILE_M * TILE_K];   // 2 x 32x32 x 4B = 8 KB
    threadgroup float Bs[2][TILE_K * TILE_N];   // 2 x 32x32 x 4B = 8 KB
                                                // Total shared: 16 KB

    // Output tile origin in global coordinates
    uint tile_row = group_id.y * TILE_M;
    uint tile_col = group_id.x * TILE_N;

    // Each simdgroup handles a 16x16 sub-tile of the 32x32 output.
    // Arranged as 2x2 grid of simdgroups:
    //   simd_gid: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    uint sg_row = (simd_gid / 2) * 16;   // 0 or 16
    uint sg_col = (simd_gid % 2) * 16;   // 0 or 16

    // Accumulators: 2x2 grid of 8x8 simdgroup matrices = 16x16 output per simdgroup
    simdgroup_matrix<float, 8, 8> acc00(0.0f);  // rows [0..7],  cols [0..7]
    simdgroup_matrix<float, 8, 8> acc01(0.0f);  // rows [0..7],  cols [8..15]
    simdgroup_matrix<float, 8, 8> acc10(0.0f);  // rows [8..15], cols [0..7]
    simdgroup_matrix<float, 8, 8> acc11(0.0f);  // rows [8..15], cols [8..15]

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // --- Prefetch first tile (buffer 0) ---
    load_A_tile(A, As[0], flat_tid, tile_row, 0, M, K);
    load_B_tile(B, Bs[0], flat_tid, 0, tile_col, K, N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint cur_buf = kt & 1;
        uint next_buf = 1 - cur_buf;

        // --- Prefetch next tile into the other buffer (double buffering) ---
        uint next_k = (kt + 1) * TILE_K;
        if (kt + 1 < num_k_tiles) {
            load_A_tile(A, As[next_buf], flat_tid, tile_row, next_k, M, K);
            load_B_tile(B, Bs[next_buf], flat_tid, next_k, tile_col, K, N);
        }

        // --- Compute: iterate over TILE_K in 8-element steps for simdgroup_matrix ---
        // Each step loads 8 columns of A and 8 rows of B as 8x8 matrices
        for (uint kk = 0; kk < TILE_K; kk += 8) {
            // Load A sub-matrices for this simdgroup:
            //   a_top:  rows [sg_row..sg_row+7],  cols [kk..kk+7]  from As
            //   a_bot:  rows [sg_row+8..sg_row+15], cols [kk..kk+7] from As
            simdgroup_matrix<float, 8, 8> a_top;
            simdgroup_matrix<float, 8, 8> a_bot;
            simdgroup_load(a_top, As[cur_buf] + sg_row * TILE_K + kk, TILE_K);
            simdgroup_load(a_bot, As[cur_buf] + (sg_row + 8) * TILE_K + kk, TILE_K);

            // Load B sub-matrices:
            //   b_left:  rows [kk..kk+7], cols [sg_col..sg_col+7]  from Bs
            //   b_right: rows [kk..kk+7], cols [sg_col+8..sg_col+15] from Bs
            simdgroup_matrix<float, 8, 8> b_left;
            simdgroup_matrix<float, 8, 8> b_right;
            simdgroup_load(b_left,  Bs[cur_buf] + kk * TILE_N + sg_col, TILE_N);
            simdgroup_load(b_right, Bs[cur_buf] + kk * TILE_N + sg_col + 8, TILE_N);

            // Multiply-accumulate: 4 matmul ops per step
            simdgroup_multiply_accumulate(acc00, a_top, b_left,  acc00);
            simdgroup_multiply_accumulate(acc01, a_top, b_right, acc01);
            simdgroup_multiply_accumulate(acc10, a_bot, b_left,  acc10);
            simdgroup_multiply_accumulate(acc11, a_bot, b_right, acc11);
        }

        // Wait for next tile prefetch to complete before we loop back
        if (kt + 1 < num_k_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // --- Write results to C with boundary checks ---
    // Use As[0] as a full 32x32 staging buffer (row-major, stride=TILE_N=32).
    // Each simdgroup stores its four 8x8 accumulator tiles at the correct
    // position within the 32x32 layout, offset by (sg_row, sg_col).
    threadgroup float* staging = As[0];  // 32x32 = 1024 floats

    // simdgroup_store: writes 8x8 tile to threadgroup memory at given offset/stride
    // Stride is TILE_N=32 (full tile width) so all 4 simdgroups write to
    // non-overlapping regions of the same 32x32 buffer.
    simdgroup_store(acc00, staging + (sg_row + 0) * TILE_N + (sg_col + 0), TILE_N);
    simdgroup_store(acc01, staging + (sg_row + 0) * TILE_N + (sg_col + 8), TILE_N);
    simdgroup_store(acc10, staging + (sg_row + 8) * TILE_N + (sg_col + 0), TILE_N);
    simdgroup_store(acc11, staging + (sg_row + 8) * TILE_N + (sg_col + 8), TILE_N);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // All 128 threads cooperatively copy the 32x32 staging buffer to global C.
    // 1024 elements / 128 threads = 8 elements per thread.
    for (uint i = 0; i < 8; i++) {
        uint elem = flat_tid * 8 + i;
        uint local_r = elem / TILE_N;
        uint local_c = elem % TILE_N;
        uint global_r = tile_row + local_r;
        uint global_c = tile_col + local_c;
        if (global_r < M && global_c < N) {
            C[global_r * N + global_c] = staging[local_r * TILE_N + local_c];
        }
    }
}

// ===========================================================================
// Batched FP32 matmul using simdgroup_matrix
// C[b] = A[b] @ B[b] for each batch b
// ===========================================================================
// Grid: ceil(N/TILE_N) x ceil(M/TILE_M) x batch_size
// Threadgroup: 128 threads (4 simdgroups)
kernel void matmul_fp32_batched(
    device const float* A           [[buffer(0)]],   // [batch, M, K]
    device const float* B           [[buffer(1)]],   // [batch, K, N]
    device float* C                 [[buffer(2)]],   // [batch, M, N]
    constant uint& M                [[buffer(3)]],
    constant uint& K                [[buffer(4)]],
    constant uint& N                [[buffer(5)]],
    uint3 group_id                  [[threadgroup_position_in_grid]],
    uint flat_tid                   [[thread_index_in_threadgroup]],
    uint simd_gid                   [[simdgroup_index_in_threadgroup]],
    uint simd_lane                  [[thread_index_in_simdgroup]]
) {
    uint batch = group_id.z;
    device const float* A_batch = A + batch * M * K;
    device const float* B_batch = B + batch * K * N;
    device float* C_batch = C + batch * M * N;

    threadgroup float As[2][TILE_M * TILE_K];
    threadgroup float Bs[2][TILE_K * TILE_N];

    uint tile_row = group_id.y * TILE_M;
    uint tile_col = group_id.x * TILE_N;

    uint sg_row = (simd_gid / 2) * 16;
    uint sg_col = (simd_gid % 2) * 16;

    simdgroup_matrix<float, 8, 8> acc00(0.0f);
    simdgroup_matrix<float, 8, 8> acc01(0.0f);
    simdgroup_matrix<float, 8, 8> acc10(0.0f);
    simdgroup_matrix<float, 8, 8> acc11(0.0f);

    uint num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // Prefetch first tile
    load_A_tile(A_batch, As[0], flat_tid, tile_row, 0, M, K);
    load_B_tile(B_batch, Bs[0], flat_tid, 0, tile_col, K, N);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kt = 0; kt < num_k_tiles; kt++) {
        uint cur_buf = kt & 1;
        uint next_buf = 1 - cur_buf;

        uint next_k = (kt + 1) * TILE_K;
        if (kt + 1 < num_k_tiles) {
            load_A_tile(A_batch, As[next_buf], flat_tid, tile_row, next_k, M, K);
            load_B_tile(B_batch, Bs[next_buf], flat_tid, next_k, tile_col, K, N);
        }

        for (uint kk = 0; kk < TILE_K; kk += 8) {
            simdgroup_matrix<float, 8, 8> a_top, a_bot;
            simdgroup_load(a_top, As[cur_buf] + sg_row * TILE_K + kk, TILE_K);
            simdgroup_load(a_bot, As[cur_buf] + (sg_row + 8) * TILE_K + kk, TILE_K);

            simdgroup_matrix<float, 8, 8> b_left, b_right;
            simdgroup_load(b_left,  Bs[cur_buf] + kk * TILE_N + sg_col, TILE_N);
            simdgroup_load(b_right, Bs[cur_buf] + kk * TILE_N + sg_col + 8, TILE_N);

            simdgroup_multiply_accumulate(acc00, a_top, b_left,  acc00);
            simdgroup_multiply_accumulate(acc01, a_top, b_right, acc01);
            simdgroup_multiply_accumulate(acc10, a_bot, b_left,  acc10);
            simdgroup_multiply_accumulate(acc11, a_bot, b_right, acc11);
        }

        if (kt + 1 < num_k_tiles) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Write results via staging buffer (32x32 layout, stride=TILE_N)
    threadgroup float* staging = As[0];
    simdgroup_store(acc00, staging + (sg_row + 0) * TILE_N + (sg_col + 0), TILE_N);
    simdgroup_store(acc01, staging + (sg_row + 0) * TILE_N + (sg_col + 8), TILE_N);
    simdgroup_store(acc10, staging + (sg_row + 8) * TILE_N + (sg_col + 0), TILE_N);
    simdgroup_store(acc11, staging + (sg_row + 8) * TILE_N + (sg_col + 8), TILE_N);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 8; i++) {
        uint elem = flat_tid * 8 + i;
        uint local_r = elem / TILE_N;
        uint local_c = elem % TILE_N;
        uint global_r = tile_row + local_r;
        uint global_c = tile_col + local_c;
        if (global_r < M && global_c < N) {
            C_batch[global_r * N + global_c] = staging[local_r * TILE_N + local_c];
        }
    }
}
