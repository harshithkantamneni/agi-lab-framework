/* embedding.h -- Token embedding and unembedding (weight-tied).
 *
 * The embedding table maps token IDs to dense vectors of size d_model.
 * Weight is shared with the unembedding (lm_head) via pointer aliasing.
 *
 * Forward: out[i, :] = weight[token_ids[i], :] for each position i.
 */

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "hspa_config.h"
#include "memory_pool.h"
#include "tensor.h"

#include <stdint.h>

typedef struct {
    Tensor *weight; /* [vocab_size, d_model] -- shared with unembedding */
} Embedding;

/* Create an embedding table allocated from `pool`.
 * Weight shape: [cfg->vocab_size, cfg->d_model], dtype FP32.
 * Initialized to zero (caller should init with proper scheme).
 * Returns NULL on failure. */
Embedding *embedding_create(MemoryPool *pool, const HSPAConfig *cfg);

/* Embedding lookup: out[i, :] = weight[token_ids[i], :].
 * out must be pre-allocated with shape [seq_len, d_model].
 * token_ids: array of seq_len token indices in [0, vocab_size). */
void embedding_forward(Tensor *out, const Embedding *emb,
                       const int32_t *token_ids, int32_t seq_len);

/* Destroy the embedding struct (weight data owned by pool). */
void embedding_destroy(Embedding *emb);

#endif /* EMBEDDING_H */
