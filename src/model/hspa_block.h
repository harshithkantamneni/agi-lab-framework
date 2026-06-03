/* hspa_block.h -- One HSPA transformer block.
 *
 * Each block contains:
 *   1. Pre-attention RMSNorm
 *   2. Grouped Query Attention (GQA)
 *   3. Residual connection
 *   4. Pre-FFN RMSNorm
 *   5. FEP Router -> top-k expert selection
 *   6. Mixture of Experts (n_experts SwiGLU FFNs, only n_active evaluated)
 *   7. Residual connection
 *
 * Expert count is configurable via cfg->n_experts (D-004).
 */

#ifndef HSPA_BLOCK_H
#define HSPA_BLOCK_H

#include "attention.h"
#include "ffn.h"
#include "hspa_config.h"
#include "memory_pool.h"
#include "qat_context.h"
#include "rmsnorm.h"
#include "router.h"
#include "tensor.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    RMSNorm *attn_norm;     /* Pre-attention normalization */
    Attention *attn;        /* Grouped Query Attention */
    RMSNorm *ffn_norm;      /* Pre-FFN normalization */
    FEPRouter *router;      /* FEP expert router */
    ExpertFFN **experts;    /* Array of n_experts SwiGLU FFNs */
    int32_t n_experts;      /* Number of experts (from cfg, not hardcoded) */
} HSPABlock;

/* Create one HSPA block with all sub-components.
 * Allocates n_experts FFNs (configurable via cfg->n_experts).
 * Returns NULL on failure. */
HSPABlock *hspa_block_create(MemoryPool *pool, const HSPAConfig *cfg);

/* Forward pass through one block.
 * out: [seq_len, d_model] (pre-allocated)
 * x: [seq_len, d_model] input (will be used as residual)
 * cache: KV cache for this layer's attention
 * pos: starting position in the sequence
 * scratch: temporary pool for intermediate activations
 * training: controls router exploration behavior
 * n_active_override: if > 0, use this as n_active instead of cfg->n_active
 *                    (for progressive sparsification). 0 = use default. */
void hspa_block_forward(Tensor *out, const HSPABlock *block, const Tensor *x,
                        KVCache *cache, int32_t pos, MemoryPool *scratch,
                        bool training, int32_t n_active_override,
                        QATContext *qat);

/* Destroy the block and all sub-components. */
void hspa_block_destroy(HSPABlock *block);

#endif /* HSPA_BLOCK_H */
