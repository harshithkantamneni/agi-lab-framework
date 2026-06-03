/* hspa_model.h -- Complete HSPA v1 model.
 *
 * Hierarchical Sparse Predictive Architecture:
 *   - Token embedding (weight-tied with lm_head)
 *   - 32 HSPA transformer blocks (each with MoE)
 *   - Final RMSNorm
 *   - Linear projection to vocab (tied embedding weights)
 *
 * Memory pools:
 *   - weight_pool: POOL_WEIGHTS for all model parameters
 *   - activation_pool: POOL_ACTIVATIONS for layer-streaming activations
 *   - scratch_pool: POOL_SCRATCH for temporary intermediates
 *
 * Total: 16B params, 3.36B active per token (top-2 of 16 experts).
 */

#ifndef HSPA_MODEL_H
#define HSPA_MODEL_H

#include "attention.h"
#include "embedding.h"
#include "hspa_block.h"
#include "hspa_config.h"
#include "memory_pool.h"
#include "rmsnorm.h"
#include "tensor.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    HSPAConfig cfg;             /* Model configuration */
    Embedding *embed;           /* Token embedding (shared with lm_head) */
    HSPABlock **layers;         /* Array of n_layers HSPA blocks */
    KVCache **kv_caches;        /* Per-layer KV caches: [n_layers] */
    RMSNorm *final_norm;        /* Final RMSNorm before lm_head */
    Tensor *lm_head;            /* [vocab_size, d_model] -- alias of embed->weight */
    MemoryPool *weight_pool;    /* Static pool for model weights */
    MemoryPool *activation_pool;/* Ring buffer for layer activations */
    MemoryPool *scratch_pool;   /* Temporary scratch space */
} HSPAModel;

/* Create the complete HSPA v1 model.
 * Allocates all pools, layers, weights, and KV caches.
 * Returns NULL on failure. */
HSPAModel *hspa_model_create(const HSPAConfig *cfg);

/* Full forward pass: tokens -> logits.
 * logits: [seq_len, vocab_size] (pre-allocated by caller, or allocated internally)
 * tokens: array of seq_len token IDs
 * seq_len: number of tokens in the input
 * training: controls router exploration and iPC behavior */
void hspa_model_forward(Tensor *logits, HSPAModel *model,
                        const int32_t *tokens, int32_t seq_len, bool training);

/* Full forward pass with optional QAT context (Program 3, Phase 7, D-613 T5).
 * Identical to hspa_model_forward but passes qat through to every
 * hspa_block_forward call.  When qat is NULL, behavior is bit-identical to
 * hspa_model_forward (regression invariant: NULL-qat path must produce the
 * same output as the plain forward call).
 *
 * qat: if non-NULL, must already have qat_context_begin_step called by
 *      the caller this step.  Destroyed/managed by the caller. */
void hspa_model_forward_qat(Tensor *logits, HSPAModel *model,
                            const int32_t *tokens, int32_t seq_len, bool training,
                            QATContext *qat);

/* Destroy the model and free all pools. */
void hspa_model_destroy(HSPAModel *model);

#endif /* HSPA_MODEL_H */
