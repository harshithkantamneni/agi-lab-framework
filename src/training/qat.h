/* qat.h -- QAT fake-quantize + Straight-Through Estimator (STE) module.
 *
 * Program 3 Phase 7 apparatus: 4-bit Quantization-Aware Training.
 * Binding signature from program_open_memo §5.2 (D-608).
 *
 * Design:
 *   Forward  — fake_quantize_int4() forces weights onto the 16-level
 *              per-group asymmetric INT4 grid but keeps FP32 storage.
 *              Delegates to the existing PTQ primitives in src/core/quantize.c
 *              (quantize_fp32_to_int4 / dequantize_int4_to_fp32).
 *
 *   Backward — fake_quantize_int4_backward() implements the STE: gradients
 *              pass through the quantization boundary unchanged (identity map).
 *              Pattern precedented at backprop_train.c:901 (router top-k STE).
 *
 * Ownership / memory contract:
 *   - fake_quantize_int4() allocates the returned Tensor *w_hat on the heap
 *     (malloc for the Tensor struct + malloc for the float data).
 *     The QuantGroup *groups array is caller-allocated and caller-owned;
 *     the function WRITES the per-group scale/zero_point into groups[].
 *     Caller must free(w_hat->data) then free(w_hat) when done.
 *
 *   - fake_quantize_int4_backward() allocates and returns a new Tensor
 *     (same contract: caller must free(->data) then free(struct)).
 *
 *   - Neither function uses a MemoryPool — the returned tensors live on
 *     the C heap for explicit lifecycle control during the training loop.
 *     The call-site follow-on may wrap these into pool-backed temporaries.
 *
 * Thread safety: none (single-threaded by design per CLAUDE.md).
 */

#ifndef QAT_H
#define QAT_H

#include "quantize.h"
#include "tensor.h"

/* ---- Forward: fake-quantize FP32 → INT4 grid → FP32 ----
 *
 * Quantizes w_fp32 to per-group asymmetric INT4 using group_size elements
 * per group, then dequantizes back to FP32.  The result is a weight tensor
 * forced onto the 16-level INT4 grid while retaining FP32 storage — the
 * "fake quantization" trick that makes forward matmul simulate 4-bit
 * precision without changing the data type flowing through the graph.
 *
 * Parameters:
 *   w_fp32    — input weight tensor (DTYPE_FP32, any shape, contiguous).
 *   groups    — caller-allocated array of quantize_num_groups(w_fp32, group_size)
 *               QuantGroup entries.  On return, filled with per-group
 *               (scale, zero_point) metadata.
 *   group_size — number of consecutive elements per quantization group
 *               (typically 128 per program_open_memo §5.2).
 *               The last group may be smaller if numel % group_size != 0.
 *
 * Returns:
 *   Newly heap-allocated FP32 Tensor *w_hat of the same shape as w_fp32.
 *   Returns NULL on allocation failure.
 *
 * Caller responsibility:
 *   free(w_hat->data);
 *   free(w_hat);
 */
Tensor *fake_quantize_int4(Tensor *w_fp32, QuantGroup *groups, int group_size);

/* ---- Backward: STE gradient passthrough ----
 *
 * Implements the Straight-Through Estimator (STE) for the quantization
 * boundary:  dL/dw = dL/dw_hat  (identity, no clipping).
 *
 * This is the standard STE for fake-quantization (Bengio et al. 2013 §3).
 * The weight w_fp32 argument is accepted to match the call-site convention
 * (the backward function receives both the upstream gradient AND the
 * corresponding primal weight) but is not used — the STE is unconditional.
 * Precedented by the router top-k STE at backprop_train.c:901 which also
 * passes the gradient through unchanged.
 *
 * Parameters:
 *   grad_out  — upstream gradient tensor (DTYPE_FP32, same shape as w_fp32).
 *   w_fp32    — primal weight tensor (accepted but not read; kept for
 *               call-site symmetry).
 *
 * Returns:
 *   Newly heap-allocated FP32 Tensor *grad_in, element-wise equal to grad_out.
 *   Returns NULL on allocation failure.
 *
 * Caller responsibility: same as fake_quantize_int4 — free data then struct.
 */
Tensor *fake_quantize_int4_backward(const Tensor *grad_out,
                                    const Tensor *w_fp32);

#endif /* QAT_H */
