/* qat_context.h -- QATContext: per-step w_hat cache for 4-bit QAT.
 *
 * Program 3 Phase 7 apparatus. D-613, 2026-05-28.
 * Binding spec: p7_qat_coverage_architecture.md §2.1 + §7 (Option E).
 *
 * Purpose:
 *   QATContext caches the fake-quantized shadow weight (w_hat) for each
 *   master weight tensor W, keyed by the W pointer, for the duration of one
 *   training step.  A single forward pass and the subsequent backward STE
 *   both reference the SAME w_hat, which ensures:
 *     - iPC double-forward is bit-identical (both see the same quantized grid).
 *     - STE backward is consistent with the forward (dW accumulates on FP32
 *       master; w_hat_T used for dx computation is the same grid-snapped copy).
 *
 * Design (lazy first-encounter allocation):
 *   Shadow weight Tensor structs, their float data buffers, and their
 *   QuantGroup arrays are allocated LAZILY on the first call to
 *   qat_context_w_hat for each distinct master weight pointer W.  The hash
 *   map slot is reused for all subsequent steps (same W pointer →  same slot).
 *   ZERO per-step heap allocation after the first step warms up all 224 slots.
 *
 * Hash map:
 *   Open-addressing, pointer-keyed, capacity must be > n_weights (load < 1.0).
 *   Default capacity 256 > 224 program-3 in-scope weights.
 *
 * Step protocol:
 *   1. qat_context_begin_step(qc)   -- invalidate all cached entries
 *   2. For each weight W used in forward: w_hat = qat_context_w_hat(qc, W)
 *      -- first call: quantize+cache; subsequent: cache hit
 *   3. qat_context_covered_count(qc) -- check that all 224 weights fired
 *
 * Memory contract:
 *   qat_context_create/destroy own all internal storage.
 *   Returned const Tensor * pointers are valid until the next begin_step call.
 *   Caller does NOT free any returned pointer.
 *
 * Thread safety: none (single-threaded by design per CLAUDE.md).
 */

#ifndef QAT_CONTEXT_H
#define QAT_CONTEXT_H

#include "quantize.h"
#include "tensor.h"

#include <stdbool.h>
#include <stdint.h>

/* ---- Opaque context type (defined in qat_context.c) ---- */
typedef struct QATContext QATContext;

/* ---- Lifecycle ---- */

/* Create a QATContext.
 *
 * enabled:    if false, qat_context_w_hat returns W unchanged (pass-through).
 * group_size: number of elements per quantization group (typically 128).
 * capacity:   hash map capacity; must exceed the number of distinct weight
 *             tensors that will call qat_context_w_hat in one step.
 *             Use 256 for the full 119.6M MoE model (224 in-scope weights).
 *
 * Pre-allocates all internal shadow weight buffers.
 * Returns NULL on allocation failure. */
QATContext *qat_context_create(bool enabled, int group_size, int capacity);

/* Destroy the context and free all internal storage.
 * NULL is tolerated (no-op). */
void qat_context_destroy(QATContext *qc);

/* ---- Step protocol ---- */

/* Invalidate all cached w_hat entries for the upcoming step.
 * covered_count is reset to 0.
 * NULL is tolerated (no-op). */
void qat_context_begin_step(QATContext *qc);

/* Return the fake-quantized shadow weight for master weight W.
 *
 * Enabled context:
 *   First call with pointer W in this step: quantize W to INT4 grid then
 *   dequantize back to FP32, store result in a pre-allocated slot, cache
 *   the W->slot mapping.
 *   Subsequent calls with same W in same step: cache hit, return same pointer.
 *   covered_count increments by 1 on cache miss (first call only).
 *
 * Disabled context (or NULL qc):
 *   Returns W unchanged (same pointer, no copy, no alloc).
 *
 * The returned pointer is valid until the next qat_context_begin_step call.
 * Caller does NOT free the returned pointer. */
const Tensor *qat_context_w_hat(QATContext *qc, const Tensor *W);

/* ---- Queries ---- */

/* Return number of distinct master-weight pointers that called w_hat this step.
 * This is the §3.3 clause 1 gate: must equal 224 before the coverage assertion
 * passes for program 3's in-scope weight set. */
int qat_context_covered_count(const QATContext *qc);

/* Return number of w_hat calls that were cache HITS this step (same W pointer
 * seen again within the same step).  STE backward and iPC double-forward both
 * consume cached w_hat; this counter proves they are using the cache rather
 * than re-quantizing.  §3.3 Clause 4 gate: must be > 0 when QAT enabled. */
int qat_context_cache_hits(const QATContext *qc);

/* Return true if this context was created with enabled=true. */
bool qat_context_is_enabled(const QATContext *qc);

#endif /* QAT_CONTEXT_H */
