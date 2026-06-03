/* ffn.h -- SwiGLU expert feed-forward network.
 *
 * Each expert uses a gated linear unit with SiLU activation:
 *   gate = x @ W_gate     [d_model -> d_ff]
 *   up   = x @ W_up       [d_model -> d_ff]
 *   h    = silu(gate) * up [d_ff]
 *   out  = h @ W_down      [d_ff -> d_model]
 *
 * With 16 experts per layer, each expert has ~28.2M parameters.
 * Expert d_ff ~ 1850 (derived from total param budget).
 */

#ifndef FFN_H
#define FFN_H

#include "hspa_config.h"
#include "memory_pool.h"
#include "qat_context.h"
#include "tensor.h"

typedef struct {
    Tensor *W_gate; /* Gate projection:   [d_model, d_ff] */
    Tensor *W_up;   /* Up projection:     [d_model, d_ff] */
    Tensor *W_down; /* Down projection:   [d_ff, d_model] */
} ExpertFFN;

/* Create one expert FFN, allocating weights from `pool`.
 * Returns NULL on failure. */
ExpertFFN *expert_ffn_create(MemoryPool *pool, const HSPAConfig *cfg);

/* Forward pass: SwiGLU FFN.
 * out: [seq_len, d_model] (pre-allocated)
 * x: [seq_len, d_model] input
 * scratch: temporary pool for intermediate activations
 * qat: QATContext for fake-quantizing weight matrices; NULL = plain matmul */
void expert_ffn_forward(Tensor *out, const ExpertFFN *ffn, const Tensor *x,
                        MemoryPool *scratch, QATContext *qat);

/* Destroy the expert FFN struct. */
void expert_ffn_destroy(ExpertFFN *ffn);

#endif /* FFN_H */
