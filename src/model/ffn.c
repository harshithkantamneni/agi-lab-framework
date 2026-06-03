/* ffn.c -- SwiGLU expert feed-forward network implementation.
 *
 * Forward pass:
 *   1. gate = x @ W_gate        [seq_len, d_ff]
 *   2. up   = x @ W_up          [seq_len, d_ff]
 *   3. h    = silu(gate) * up    [seq_len, d_ff]  (SwiGLU activation)
 *   4. out  = h @ W_down         [seq_len, d_model]
 *
 * Intermediate tensors (gate, up, h) are allocated from the scratch pool
 * and destroyed after use.  The scratch pool is reset before allocation
 * to guarantee a clean arena.
 *
 * Weights are allocated from the provided weight pool and zero-initialized.
 * Proper weight initialization (e.g., Kaiming, Xavier) is done by the
 * training setup, not here.
 */

#include "ffn.h"
#include "ops.h"
#include "qat_context.h"

#include <stdlib.h>

ExpertFFN *expert_ffn_create(MemoryPool *pool, const HSPAConfig *cfg) {
    if (!pool || !cfg) {
        return NULL;
    }

    ExpertFFN *ffn = (ExpertFFN *)calloc(1, sizeof(ExpertFFN));
    if (!ffn) {
        return NULL;
    }

    int32_t d_model = cfg->d_model;
    int32_t d_ff = cfg->d_ff;

    /* W_gate: [d_model, d_ff] */
    int32_t gate_shape[] = {d_model, d_ff};
    ffn->W_gate = tensor_create(pool, gate_shape, 2, DTYPE_FP32);
    if (!ffn->W_gate) {
        free(ffn);
        return NULL;
    }
    tensor_fill(ffn->W_gate, 0.0f);

    /* W_up: [d_model, d_ff] */
    int32_t up_shape[] = {d_model, d_ff};
    ffn->W_up = tensor_create(pool, up_shape, 2, DTYPE_FP32);
    if (!ffn->W_up) {
        tensor_destroy(ffn->W_gate);
        free(ffn);
        return NULL;
    }
    tensor_fill(ffn->W_up, 0.0f);

    /* W_down: [d_ff, d_model] */
    int32_t down_shape[] = {d_ff, d_model};
    ffn->W_down = tensor_create(pool, down_shape, 2, DTYPE_FP32);
    if (!ffn->W_down) {
        tensor_destroy(ffn->W_up);
        tensor_destroy(ffn->W_gate);
        free(ffn);
        return NULL;
    }
    tensor_fill(ffn->W_down, 0.0f);

    return ffn;
}

void expert_ffn_forward(Tensor *out, const ExpertFFN *ffn, const Tensor *x,
                        MemoryPool *scratch, QATContext *qat) {
    if (!out || !ffn || !x || !scratch) {
        return;
    }

    int32_t seq_len = x->shape[0];
    int32_t d_ff = ffn->W_gate->shape[1];

    /* Reset scratch pool for fresh intermediate allocations. */
    pool_reset(scratch);

    /* Allocate intermediates from scratch pool. */
    int32_t inter_shape[] = {seq_len, d_ff};

    Tensor *gate = tensor_create(scratch, inter_shape, 2, DTYPE_FP32);
    if (!gate) {
        return;
    }

    Tensor *up = tensor_create(scratch, inter_shape, 2, DTYPE_FP32);
    if (!up) {
        tensor_destroy(gate);
        return;
    }

    Tensor *h = tensor_create(scratch, inter_shape, 2, DTYPE_FP32);
    if (!h) {
        tensor_destroy(up);
        tensor_destroy(gate);
        return;
    }

    /* Step 1: gate = x @ W_gate   [seq_len, d_model] @ [d_model, d_ff] */
    op_matmul_qat(gate, x, ffn->W_gate, qat);

    /* Step 2: up = x @ W_up       [seq_len, d_model] @ [d_model, d_ff] */
    op_matmul_qat(up, x, ffn->W_up, qat);

    /* Step 3: h = silu(gate) * up  (element-wise SwiGLU) */
    op_swiglu(h, gate, up);

    /* Step 4: out = h @ W_down    [seq_len, d_ff] @ [d_ff, d_model] */
    op_matmul_qat(out, h, ffn->W_down, qat);

    /* Clean up intermediates. */
    tensor_destroy(h);
    tensor_destroy(up);
    tensor_destroy(gate);
}

void expert_ffn_destroy(ExpertFFN *ffn) {
    if (!ffn) {
        return;
    }

    if (ffn->W_gate) {
        tensor_destroy(ffn->W_gate);
    }
    if (ffn->W_up) {
        tensor_destroy(ffn->W_up);
    }
    if (ffn->W_down) {
        tensor_destroy(ffn->W_down);
    }

    free(ffn);
}
