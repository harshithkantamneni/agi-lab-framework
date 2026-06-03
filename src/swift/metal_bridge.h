/* metal_bridge.h -- C-callable interface to Metal GPU compute.
 *
 * This header declares the public API for the Swift-Metal bridge.
 * The implementation lives in MetalBridge.swift, compiled as a
 * dynamic library that links into C binaries.
 *
 * Thread safety: all functions are safe to call from any thread.
 * The bridge serializes GPU submissions internally.
 *
 * Memory: uses MTLStorageModeShared (unified memory) for zero-copy
 * CPU<->GPU data sharing on Apple Silicon.
 */

#ifndef METAL_BRIDGE_H
#define METAL_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the Metal device, command queue, and load compiled shaders.
 * Must be called before any other metal_* function.
 * Returns 0 on success, -1 on failure (no Metal device, shader not found, etc.).
 * Safe to call multiple times -- subsequent calls are no-ops returning 0. */
int metal_init(void);

/* Release all Metal resources (device, queue, pipeline states, buffers).
 * After this call, metal_init() must be called again before using other functions.
 * Safe to call even if not initialized. */
void metal_cleanup(void);

/* FP32 matrix multiplication on GPU: C = A @ B.
 *
 * A: [M, K] row-major FP32 input matrix.
 * B: [K, N] row-major FP32 input matrix.
 * C: [M, N] row-major FP32 output matrix.
 * M, N, K: matrix dimensions (must be > 0).
 *
 * The caller owns all buffers and must ensure they are large enough:
 *   A: at least M*K floats
 *   B: at least K*N floats
 *   C: at least M*N floats
 *
 * Uses MTLStorageModeShared buffers with nocopy where possible.
 * Falls back to a memcpy path for unaligned inputs. */
void metal_matmul(const float *A, const float *B, float *C,
                  int M, int N, int K);

/* Query whether Metal is initialized and ready for use.
 * Returns 1 if initialized, 0 otherwise. */
int metal_is_ready(void);

#ifdef __cplusplus
}
#endif

#endif /* METAL_BRIDGE_H */
