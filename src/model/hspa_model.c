/* hspa_model.c -- Complete HSPA v1 model (Cycle 4, Wave 2).
 *
 * Composes all components into a full forward pass:
 *   1. Token embedding lookup
 *   2. N HSPA transformer blocks (attention + MoE + residuals)
 *   3. Final RMSNorm
 *   4. Linear projection to logits (weight-tied with embedding)
 *
 * Memory architecture:
 *   - weight_pool (POOL_WEIGHTS):  all model parameters
 *   - activation_pool (POOL_ACTIVATIONS):  KV caches + layer activations
 *   - scratch_pool (POOL_SCRATCH):  temporary intermediates, reset per layer
 *
 * Weight tying: lm_head shares memory with embed->weight.
 * embed->weight is [vocab_size, d_model], so computing logits requires
 * h @ embed->weight^T which is done as a manual transposed matmul.
 */

#include "hspa_model.h"
#include "ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- Pool sizing helpers ---- */

/* Estimate weight pool size for a given config.
 * Accounts for: embedding, n_layers * (attention + 2 norms + router + n_experts FFN),
 * final norm. All FP32 for now. */
static size_t estimate_weight_pool_size(const HSPAConfig *cfg) {
    size_t elem = sizeof(float);

    /* Embedding: [vocab_size, d_model] */
    size_t embed_bytes = (size_t)cfg->vocab_size * (size_t)cfg->d_model * elem;

    /* Per-layer weights */
    size_t attn_bytes = (size_t)cfg->d_model * (size_t)(cfg->n_heads * cfg->head_dim) * elem * 2  /* W_q, W_o */
                      + (size_t)cfg->d_model * (size_t)(cfg->n_kv_heads * cfg->head_dim) * elem * 2; /* W_k, W_v */
    size_t norm_bytes = (size_t)cfg->d_model * elem * 2; /* attn_norm + ffn_norm */
    size_t router_bytes = (size_t)cfg->d_model * (size_t)cfg->n_experts * elem * 2; /* W_mu + W_sigma */
    size_t expert_bytes = (size_t)cfg->n_experts *
                          ((size_t)cfg->d_model * (size_t)cfg->d_ff * elem * 2 /* W_gate, W_up */
                         + (size_t)cfg->d_ff * (size_t)cfg->d_model * elem);    /* W_down */

    size_t per_layer = attn_bytes + norm_bytes + router_bytes + expert_bytes;

    /* Final norm: [d_model] */
    size_t final_norm_bytes = (size_t)cfg->d_model * elem;

    /* Total with 20% overhead for alignment padding */
    size_t total = embed_bytes + (size_t)cfg->n_layers * per_layer + final_norm_bytes;
    return total + total / 5 + 65536;
}

/* Estimate activation pool size.
 * Accounts for: KV caches for all layers + space for 2D activation tensors. */
static size_t estimate_activation_pool_size(const HSPAConfig *cfg) {
    size_t elem = sizeof(float);

    /* KV cache per layer: 2 * [max_seq_len, n_kv_heads, head_dim] */
    size_t kv_per_layer = 2 * (size_t)cfg->max_seq_len * (size_t)cfg->n_kv_heads
                        * (size_t)cfg->head_dim * elem;
    size_t kv_total = (size_t)cfg->n_layers * kv_per_layer;

    /* Activation tensors: embeddings output, per-layer h, logits */
    size_t act_bytes = (size_t)cfg->max_seq_len * (size_t)cfg->d_model * elem * 4;
    size_t logit_bytes = (size_t)cfg->max_seq_len * (size_t)cfg->vocab_size * elem;

    return kv_total + act_bytes + logit_bytes + 65536;
}

/* Estimate scratch pool size.
 * Must hold intermediates for one layer forward pass. */
static size_t estimate_scratch_pool_size(const HSPAConfig *cfg) {
    size_t elem = sizeof(float);

    /* Block forward needs: norm_x, attn_out, h, norm_h, moe_out, row buffers.
     * Each is [seq_len, d_model]. Plus per-token expert dispatch buffers. */
    size_t per_layer = (size_t)cfg->max_seq_len * (size_t)cfg->d_model * elem * 8
                     + (size_t)cfg->d_model * (size_t)cfg->d_ff * elem * 4;

    /* Minimum 4 MB for test flexibility */
    if (per_layer < 4 * 1024 * 1024) {
        per_layer = 4 * 1024 * 1024;
    }

    return per_layer + 65536;
}

/* ---- API: Create ---- */

HSPAModel *hspa_model_create(const HSPAConfig *cfg) {
    if (!cfg || cfg->d_model <= 0 || cfg->n_layers <= 0 ||
        cfg->vocab_size <= 0 || cfg->n_heads <= 0) {
        return NULL;
    }

    HSPAModel *model = (HSPAModel *)calloc(1, sizeof(HSPAModel));
    if (!model) {
        return NULL;
    }

    /* Store config copy */
    model->cfg = *cfg;

    /* Step 1: Create memory pools */
    size_t wpool_size = estimate_weight_pool_size(cfg);
    size_t apool_size = estimate_activation_pool_size(cfg);
    size_t spool_size = estimate_scratch_pool_size(cfg);

    model->weight_pool = pool_create(wpool_size, POOL_WEIGHTS);
    if (!model->weight_pool) {
        free(model);
        return NULL;
    }

    model->activation_pool = pool_create(apool_size, POOL_ACTIVATIONS);
    if (!model->activation_pool) {
        pool_destroy(model->weight_pool);
        free(model);
        return NULL;
    }

    model->scratch_pool = pool_create(spool_size, POOL_SCRATCH);
    if (!model->scratch_pool) {
        pool_destroy(model->activation_pool);
        pool_destroy(model->weight_pool);
        free(model);
        return NULL;
    }

    /* Step 2: Create embedding (allocated from weight pool) */
    model->embed = embedding_create(model->weight_pool, cfg);
    if (!model->embed) {
        pool_destroy(model->scratch_pool);
        pool_destroy(model->activation_pool);
        pool_destroy(model->weight_pool);
        free(model);
        return NULL;
    }

    /* Step 3: Weight tying -- lm_head points to embed->weight.
     * embed->weight is [vocab_size, d_model].
     * In the forward pass we compute h @ embed->weight^T to get logits. */
    model->lm_head = model->embed->weight;

    /* Step 4: Create HSPA blocks (layers) */
    model->layers = (HSPABlock **)calloc((size_t)cfg->n_layers, sizeof(HSPABlock *));
    if (!model->layers) {
        embedding_destroy(model->embed);
        pool_destroy(model->scratch_pool);
        pool_destroy(model->activation_pool);
        pool_destroy(model->weight_pool);
        free(model);
        return NULL;
    }

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        model->layers[l] = hspa_block_create(model->weight_pool, cfg);
        if (!model->layers[l]) {
            /* Cleanup already-created layers */
            for (int32_t j = 0; j < l; j++) {
                hspa_block_destroy(model->layers[j]);
            }
            free(model->layers);
            embedding_destroy(model->embed);
            pool_destroy(model->scratch_pool);
            pool_destroy(model->activation_pool);
            pool_destroy(model->weight_pool);
            free(model);
            return NULL;
        }
    }

    /* Step 5: Create per-layer KV caches (allocated from activation pool) */
    model->kv_caches = (KVCache **)calloc((size_t)cfg->n_layers, sizeof(KVCache *));
    if (!model->kv_caches) {
        for (int32_t l = 0; l < cfg->n_layers; l++) {
            hspa_block_destroy(model->layers[l]);
        }
        free(model->layers);
        embedding_destroy(model->embed);
        pool_destroy(model->scratch_pool);
        pool_destroy(model->activation_pool);
        pool_destroy(model->weight_pool);
        free(model);
        return NULL;
    }

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        model->kv_caches[l] = kv_cache_create(model->activation_pool, cfg);
        if (!model->kv_caches[l]) {
            /* Cleanup already-created caches */
            for (int32_t j = 0; j < l; j++) {
                kv_cache_destroy(model->kv_caches[j]);
            }
            free(model->kv_caches);
            for (int32_t j = 0; j < cfg->n_layers; j++) {
                hspa_block_destroy(model->layers[j]);
            }
            free(model->layers);
            embedding_destroy(model->embed);
            pool_destroy(model->scratch_pool);
            pool_destroy(model->activation_pool);
            pool_destroy(model->weight_pool);
            free(model);
            return NULL;
        }
    }

    /* Step 6: Create final RMSNorm (allocated from weight pool) */
    model->final_norm = rmsnorm_create(model->weight_pool, cfg->d_model,
                                       cfg->rms_norm_eps);
    if (!model->final_norm) {
        for (int32_t l = 0; l < cfg->n_layers; l++) {
            kv_cache_destroy(model->kv_caches[l]);
        }
        free(model->kv_caches);
        for (int32_t l = 0; l < cfg->n_layers; l++) {
            hspa_block_destroy(model->layers[l]);
        }
        free(model->layers);
        embedding_destroy(model->embed);
        pool_destroy(model->scratch_pool);
        pool_destroy(model->activation_pool);
        pool_destroy(model->weight_pool);
        free(model);
        return NULL;
    }

    return model;
}

/* ---- API: Forward ---- */

/* Thin wrapper: plain forward delegates to the QAT variant with qat=NULL.
 * NULL-qat path is bit-identical to the pre-T5 hspa_model_forward (regression
 * invariant: nothing in hspa_block_forward changes when qat==NULL). */
void hspa_model_forward(Tensor *logits, HSPAModel *model,
                        const int32_t *tokens, int32_t seq_len, bool training) {
    hspa_model_forward_qat(logits, model, tokens, seq_len, training, NULL);
}

void hspa_model_forward_qat(Tensor *logits, HSPAModel *model,
                             const int32_t *tokens, int32_t seq_len, bool training,
                             QATContext *qat) {
    if (!logits || !model || !tokens || seq_len <= 0) {
        return;
    }

    const HSPAConfig *cfg = &model->cfg;
    int32_t d_model = cfg->d_model;
    int32_t vocab_size = cfg->vocab_size;

    /* ================================================================
     * Step 1: Embed tokens -> h [seq_len, d_model]
     *
     * Allocate h from scratch pool. It persists across layers since
     * we swap h <-> layer_out each iteration.
     * ================================================================ */
    pool_reset(model->scratch_pool);

    int32_t h_shape[] = {seq_len, d_model};
    Tensor *h = tensor_create(model->scratch_pool, h_shape, 2, DTYPE_FP32);
    if (!h) {
        return;
    }

    embedding_forward(h, model->embed, tokens, seq_len);

    /* ================================================================
     * Step 2: Pass through each HSPA block
     *
     * For each layer:
     *   a. Reset scratch pool (but preserve h by copying to activation pool)
     *   b. Run hspa_block_forward(layer_out, layers[l], h, kv_cache, pos, scratch)
     *   c. h = layer_out (for next iteration)
     *
     * Since hspa_block_forward needs a scratch pool for its own intermediates,
     * and our h is also on the scratch pool, we need a strategy. We use a
     * dedicated layer scratch pool separate from h's storage.
     * ================================================================ */

    /* Allocate a separate scratch pool for layer forward passes.
     * This keeps h safe on model->scratch_pool. */
    size_t layer_scratch_size = estimate_scratch_pool_size(cfg);
    MemoryPool *layer_scratch = pool_create(layer_scratch_size, POOL_SCRATCH);
    if (!layer_scratch) {
        tensor_destroy(h);
        return;
    }

    /* Allocate layer output on model->scratch_pool alongside h */
    Tensor *layer_out = tensor_create(model->scratch_pool, h_shape, 2, DTYPE_FP32);
    if (!layer_out) {
        pool_destroy(layer_scratch);
        tensor_destroy(h);
        return;
    }

    /* Get starting position from first layer's KV cache */
    int32_t pos = model->kv_caches[0]->pos;

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        /* Reset the layer scratch pool for this layer's intermediates */
        pool_reset(layer_scratch);

        /* Zero the output buffer */
        tensor_fill(layer_out, 0.0f);

        /* QAT coverage pre-registration (§3.3 clause 1 guard, D-613 T5).
         * Ensures all K experts' W_gate/W_up/W_down are registered in the
         * QATContext this step — even experts that receive no tokens and would
         * otherwise be skipped by the router.  This mirrors the training-arm
         * warm-up (backprop_train.c §616-622) so eval covered_count == 224
         * (L*K*3 + L*4) regardless of routing decisions.
         * When qat is NULL or disabled, qat_context_w_hat is a no-op. */
        if (qat && qat_context_is_enabled(qat)) {
            HSPABlock *blk = model->layers[l];
            for (int32_t j = 0; j < cfg->n_experts; j++) {
                ExpertFFN *expert = blk->experts[j];
                (void)qat_context_w_hat(qat, expert->W_gate);
                (void)qat_context_w_hat(qat, expert->W_up);
                (void)qat_context_w_hat(qat, expert->W_down);
            }
        }

        /* Run the block forward pass.
         * Input: h [seq_len, d_model]
         * Output: layer_out [seq_len, d_model]
         * KV cache for this layer is updated in-place. */
        hspa_block_forward(layer_out, model->layers[l], h, model->kv_caches[l],
                           pos, layer_scratch, training, 0, qat);

        /* Swap h and layer_out: h now holds this layer's output,
         * layer_out becomes the buffer for the next layer's output. */
        Tensor *tmp = h;
        h = layer_out;
        layer_out = tmp;
    }

    pool_destroy(layer_scratch);

    /* h now points to the final layer's output [seq_len, d_model] */

    /* ================================================================
     * Step 3: Final RMSNorm (per-row, same pattern as hspa_block.c)
     *
     * RMSNorm operates on 1D [d_model] vectors, so we process each
     * row of h independently.
     * ================================================================ */
    {
        int32_t row_shape[] = {d_model};
        Tensor *row_in = tensor_create(model->scratch_pool, row_shape, 1, DTYPE_FP32);
        Tensor *row_out = tensor_create(model->scratch_pool, row_shape, 1, DTYPE_FP32);
        if (!row_in || !row_out) {
            if (row_in) tensor_destroy(row_in);
            if (row_out) tensor_destroy(row_out);
            tensor_destroy(layer_out);
            tensor_destroy(h);
            return;
        }

        for (int32_t s = 0; s < seq_len; s++) {
            /* Extract row s from h into row_in */
            for (int32_t d = 0; d < d_model; d++) {
                int32_t idx2d[] = {s, d};
                int32_t idx1d[] = {d};
                tensor_set(row_in, idx1d, tensor_get(h, idx2d));
            }

            /* Apply final RMSNorm */
            rmsnorm_forward(row_out, model->final_norm, row_in);

            /* Write normalized row back into h */
            for (int32_t d = 0; d < d_model; d++) {
                int32_t idx2d[] = {s, d};
                int32_t idx1d[] = {d};
                tensor_set(h, idx2d, tensor_get(row_out, idx1d));
            }
        }

        tensor_destroy(row_out);
        tensor_destroy(row_in);
    }

    /* ================================================================
     * Step 4: Logits = h @ lm_head^T -> [seq_len, vocab_size]
     *
     * lm_head is a pointer to embed->weight which is [vocab_size, d_model].
     * We need to compute h @ lm_head^T, i.e., for each position s and
     * vocab index v:
     *   logits[s, v] = sum_d( h[s, d] * lm_head[v, d] )
     *
     * This is a manual transposed matmul since op_matmul expects
     * a[M,K] @ b[K,N] but we have h[seq_len, d_model] and
     * lm_head[vocab_size, d_model]. We need lm_head^T[d_model, vocab_size].
     *
     * We build the transpose explicitly then use op_matmul.
     * ================================================================ */
    {
        /* Allocate transposed lm_head: [d_model, vocab_size] */
        int32_t lm_t_shape[] = {d_model, vocab_size};
        Tensor *lm_head_t = tensor_create(model->scratch_pool, lm_t_shape, 2, DTYPE_FP32);
        if (!lm_head_t) {
            tensor_destroy(layer_out);
            tensor_destroy(h);
            return;
        }

        /* Transpose: lm_head_t[d, v] = lm_head[v, d] */
        for (int32_t v = 0; v < vocab_size; v++) {
            for (int32_t d = 0; d < d_model; d++) {
                int32_t src_idx[] = {v, d};
                int32_t dst_idx[] = {d, v};
                tensor_set(lm_head_t, dst_idx, tensor_get(model->lm_head, src_idx));
            }
        }

        /* logits = h @ lm_head_t -> [seq_len, vocab_size] */
        op_matmul(logits, h, lm_head_t);

        tensor_destroy(lm_head_t);
    }

    /* Cleanup scratch tensors */
    tensor_destroy(layer_out);
    tensor_destroy(h);
}

/* ---- API: Destroy ---- */

void hspa_model_destroy(HSPAModel *model) {
    if (!model) {
        return;
    }

    /* Destroy final norm */
    if (model->final_norm) {
        rmsnorm_destroy(model->final_norm);
    }

    /* Destroy KV caches */
    if (model->kv_caches) {
        for (int32_t l = 0; l < model->cfg.n_layers; l++) {
            if (model->kv_caches[l]) {
                kv_cache_destroy(model->kv_caches[l]);
            }
        }
        free(model->kv_caches);
    }

    /* Destroy HSPA blocks */
    if (model->layers) {
        for (int32_t l = 0; l < model->cfg.n_layers; l++) {
            if (model->layers[l]) {
                hspa_block_destroy(model->layers[l]);
            }
        }
        free(model->layers);
    }

    /* Destroy embedding.
     * NOTE: lm_head is a pointer alias to embed->weight, so we do NOT
     * separately destroy lm_head. embedding_destroy handles the weight. */
    if (model->embed) {
        embedding_destroy(model->embed);
    }
    model->lm_head = NULL; /* Clear dangling alias */

    /* Destroy pools (frees all backing memory) */
    if (model->scratch_pool) {
        pool_destroy(model->scratch_pool);
    }
    if (model->activation_pool) {
        pool_destroy(model->activation_pool);
    }
    if (model->weight_pool) {
        pool_destroy(model->weight_pool);
    }

    free(model);
}
