/* embedding.c -- Token embedding lookup.
 *
 * Simple table lookup: for each token ID, copy the corresponding row
 * from the weight matrix to the output.
 *
 * Weight shape: [vocab_size, d_model]
 * Output shape: [seq_len, d_model]
 *
 * The weight tensor is allocated from the pool and zero-initialized.
 * Proper weight initialization (e.g., N(0, 0.02)) is done by the
 * training setup, not here.
 */

#include "embedding.h"

#include <stdlib.h>
#include <string.h>

Embedding *embedding_create(MemoryPool *pool, const HSPAConfig *cfg) {
    if (!pool || !cfg || cfg->vocab_size <= 0 || cfg->d_model <= 0) {
        return NULL;
    }

    Embedding *emb = (Embedding *)calloc(1, sizeof(Embedding));
    if (!emb) {
        return NULL;
    }

    int32_t shape[] = {cfg->vocab_size, cfg->d_model};
    emb->weight = tensor_create(pool, shape, 2, DTYPE_FP32);
    if (!emb->weight) {
        free(emb);
        return NULL;
    }

    /* Zero-initialize. Real init comes from training setup. */
    tensor_fill(emb->weight, 0.0f);

    return emb;
}

void embedding_forward(Tensor *out, const Embedding *emb,
                       const int32_t *token_ids, int32_t seq_len) {
    if (!out || !emb || !emb->weight || !token_ids || seq_len <= 0) {
        return;
    }

    int32_t d_model = emb->weight->shape[1];

    /* For each token in the sequence, copy its embedding row to the output. */
    for (int32_t i = 0; i < seq_len; i++) {
        int32_t tok = token_ids[i];

        /* Bounds check: clamp to valid range. */
        if (tok < 0) tok = 0;
        if (tok >= emb->weight->shape[0]) tok = emb->weight->shape[0] - 1;

        /* Copy: out[i, :] = weight[tok, :] */
        for (int32_t j = 0; j < d_model; j++) {
            int32_t src_idx[] = {tok, j};
            int32_t dst_idx[] = {i, j};
            float val = tensor_get(emb->weight, src_idx);
            tensor_set(out, dst_idx, val);
        }
    }
}

void embedding_destroy(Embedding *emb) {
    if (!emb) {
        return;
    }

    if (emb->weight) {
        tensor_destroy(emb->weight);
    }

    free(emb);
}
