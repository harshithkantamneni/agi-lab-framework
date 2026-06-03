/* qat.c -- QAT fake-quantize + STE implementation.
 *
 * Program 3 Phase 7 apparatus.  See qat.h for full interface contract.
 *
 * Forward (fake_quantize_int4):
 *   1. Allocate a scratch INT4 tensor of the same shape.
 *   2. Call quantize_fp32_to_int4() to snap weights onto the INT4 grid
 *      and populate the caller-supplied QuantGroup metadata.
 *   3. Allocate the output FP32 tensor w_hat.
 *   4. Call dequantize_int4_to_fp32() to convert back to FP32.
 *   5. Free the scratch INT4 tensor.
 *   6. Return w_hat.
 *
 * Backward (fake_quantize_int4_backward — STE):
 *   Identity map: copy grad_out into a new FP32 tensor and return it.
 *   The quantization boundary is transparent to gradients (Bengio 2013 §3).
 *
 * Memory:
 *   All returned tensors are heap-allocated (malloc for struct + data).
 *   Scratch INT4 tensor is allocated and freed within fake_quantize_int4.
 *   No MemoryPool used — caller owns the returned tensors.
 */

#include "qat.h"
#include "quantize.h"
#include "tensor.h"

#include <stdlib.h>
#include <string.h>

/* ---- Internal helpers -------------------------------------------------- */

/* Allocate a bare heap Tensor with a separate malloc'd data buffer.
 * dtype and shape are copied from `src`.
 * Returns NULL on any allocation failure (both buffers are freed on fail). */
static Tensor *alloc_like(const Tensor *src, DType dtype) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;

    /* Copy the full struct to inherit shape, stride, ndim */
    *t = *src;
    t->dtype   = dtype;
    t->pool    = NULL;
    t->is_view = false;

    /* Compute storage size for the requested dtype */
    size_t nbytes;
    if (dtype == DTYPE_INT4) {
        /* INT4: packed 2 per byte; round up */
        size_t numel = (size_t)src->shape[0];
        for (int32_t d = 1; d < src->ndim; d++) numel *= (size_t)src->shape[d];
        nbytes = (numel + 1u) / 2u;
    } else {
        /* FP32 or INT8: use original tensor's numel * bytes per element */
        size_t numel = (size_t)src->shape[0];
        for (int32_t d = 1; d < src->ndim; d++) numel *= (size_t)src->shape[d];
        size_t bpe = (dtype == DTYPE_FP32) ? 4u : 1u;
        nbytes = numel * bpe;
    }

    t->data = malloc(nbytes);
    if (!t->data) {
        free(t);
        return NULL;
    }
    return t;
}

/* ---- Public API -------------------------------------------------------- */

/*
 * fake_quantize_int4 — forward pass of the QAT fake-quantize op.
 *
 * Ownership: returned tensor is heap-allocated; caller must:
 *   free(w_hat->data);
 *   free(w_hat);
 */
Tensor *fake_quantize_int4(Tensor *w_fp32, QuantGroup *groups, int group_size) {
    if (!w_fp32 || !groups || group_size <= 0) return NULL;
    if (w_fp32->dtype != DTYPE_FP32)           return NULL;

    /* Step 1: scratch INT4 tensor (same shape as w_fp32) */
    Tensor *q4 = alloc_like(w_fp32, DTYPE_INT4);
    if (!q4) return NULL;

    /* Step 2: quantize FP32 → INT4, filling groups[] */
    quantize_fp32_to_int4(q4, w_fp32, groups, (int32_t)group_size);

    /* Step 3: allocate output FP32 tensor */
    Tensor *w_hat = alloc_like(w_fp32, DTYPE_FP32);
    if (!w_hat) {
        free(q4->data);
        free(q4);
        return NULL;
    }

    /* Step 4: dequantize INT4 → FP32 using the groups written in step 2 */
    dequantize_int4_to_fp32(w_hat, q4, groups, (int32_t)group_size);

    /* Step 5: free scratch INT4 — its job is done */
    free(q4->data);
    free(q4);

    /* Step 6: return w_hat (caller owns it) */
    return w_hat;
}

/*
 * fake_quantize_int4_backward — STE gradient passthrough.
 *
 * dL/dw = dL/dw_hat  (identity).
 *
 * w_fp32 is not read; it is accepted to match the call-site convention.
 *
 * Ownership: returned tensor is heap-allocated; caller must:
 *   free(grad_in->data);
 *   free(grad_in);
 */
Tensor *fake_quantize_int4_backward(const Tensor *grad_out,
                                    const Tensor *w_fp32) {
    (void)w_fp32; /* STE: primal weight is not used in the backward pass */
    if (!grad_out) return NULL;
    if (grad_out->dtype != DTYPE_FP32) return NULL;

    /* Compute numel for the copy */
    size_t numel = (size_t)grad_out->shape[0];
    for (int32_t d = 1; d < grad_out->ndim; d++) numel *= (size_t)grad_out->shape[d];

    /* Allocate grad_in as a heap-owned copy of grad_out */
    Tensor *grad_in = (Tensor *)malloc(sizeof(Tensor));
    if (!grad_in) return NULL;

    *grad_in = *grad_out;        /* copy shape, stride, ndim, dtype */
    grad_in->pool    = NULL;
    grad_in->is_view = false;

    grad_in->data = malloc(numel * sizeof(float));
    if (!grad_in->data) {
        free(grad_in);
        return NULL;
    }

    /* STE: grad_in = grad_out (identity passthrough) */
    memcpy(grad_in->data, grad_out->data, numel * sizeof(float));

    return grad_in;
}
