/* hspa_block.c -- One HSPA transformer block (Cycle 4, Wave 2).
 *
 * Composes attention + MoE + residual connections into a single
 * transformer block. This is the core repeating unit of the HSPA model.
 *
 * Forward pass algorithm:
 *   1. norm_x = attn_norm(x)                        -- pre-attention RMSNorm
 *   2. attn_out = attention_forward(norm_x, cache, pos)  -- GQA attention
 *   3. h = x + attn_out                             -- residual connection
 *   4. norm_h = ffn_norm(h)                          -- pre-FFN RMSNorm
 *   5. For each token t in seq_len:
 *      a. rd = router_forward(norm_h[t], cfg, training) -- per-token routing
 *      b. moe_out[t] = sum(rd.weights[i] * expert_forward(norm_h[t], experts[rd.ids[i]]))
 *      c. free rd.expert_ids, rd.expert_weights
 *   6. out = h + moe_out                            -- residual connection
 *
 * Memory discipline:
 *   - Weight pool: model parameters (norms, attention, router, experts)
 *   - Scratch pool: all intermediate activations during forward pass
 *   - RoutingDecision arrays are malloc'd by router and freed here
 *
 * Dependencies:
 *   - attention.h/.c -- GQA attention (creates its own internal scratch)
 *   - ffn.h/.c -- SwiGLU expert FFN (uses scratch pool we pass in)
 *   - router.h/.c -- FEP routing (per-token, returns RoutingDecision)
 *   - rmsnorm.h/.c -- RMSNorm (operates on 1D [dim] vectors)
 *   - ops.h/.c -- op_add for residual connections
 */

#include "hspa_block.h"
#include "ops.h"
#include "qat_context.h"

#include <stdlib.h>
#include <string.h>

/* ---- Internal: config parameters for the forward pass ---- */

typedef struct {
    int32_t d_model;
    int32_t n_experts;
    int32_t n_active;
    int32_t d_ff;
    float   rms_norm_eps;
} BlockParams;

/* File-scope registry to associate HSPABlock* with BlockParams.
 * Same pattern used in attention.c for its AttnParams. */

#define MAX_BLOCK_INSTANCES 64

static struct {
    const HSPABlock *block;
    BlockParams params;
    HSPAConfig cfg; /* Full config copy for router_forward */
} _block_registry[MAX_BLOCK_INSTANCES];
static int _block_registry_count = 0;

static void block_register(const HSPABlock *block, const BlockParams *p,
                           const HSPAConfig *cfg) {
    if (_block_registry_count < MAX_BLOCK_INSTANCES) {
        _block_registry[_block_registry_count].block = block;
        _block_registry[_block_registry_count].params = *p;
        _block_registry[_block_registry_count].cfg = *cfg;
        _block_registry_count++;
    }
}

static const BlockParams *block_lookup(const HSPABlock *block) {
    for (int i = 0; i < _block_registry_count; i++) {
        if (_block_registry[i].block == block) {
            return &_block_registry[i].params;
        }
    }
    return NULL;
}

static const HSPAConfig *block_cfg_lookup(const HSPABlock *block) {
    for (int i = 0; i < _block_registry_count; i++) {
        if (_block_registry[i].block == block) {
            return &_block_registry[i].cfg;
        }
    }
    return NULL;
}

static void block_unregister(const HSPABlock *block) {
    for (int i = 0; i < _block_registry_count; i++) {
        if (_block_registry[i].block == block) {
            _block_registry[i] = _block_registry[_block_registry_count - 1];
            _block_registry_count--;
            return;
        }
    }
}

/* ---- API: Create ---- */

HSPABlock *hspa_block_create(MemoryPool *pool, const HSPAConfig *cfg) {
    if (!pool || !cfg) {
        return NULL;
    }

    HSPABlock *block = (HSPABlock *)calloc(1, sizeof(HSPABlock));
    if (!block) {
        return NULL;
    }

    /* Step 1: Pre-attention RMSNorm [d_model] */
    block->attn_norm = rmsnorm_create(pool, cfg->d_model, cfg->rms_norm_eps);
    if (!block->attn_norm) {
        free(block);
        return NULL;
    }

    /* Step 2: Grouped Query Attention */
    block->attn = attention_create(pool, cfg);
    if (!block->attn) {
        rmsnorm_destroy(block->attn_norm);
        free(block);
        return NULL;
    }

    /* Step 3: Pre-FFN RMSNorm [d_model] */
    block->ffn_norm = rmsnorm_create(pool, cfg->d_model, cfg->rms_norm_eps);
    if (!block->ffn_norm) {
        attention_destroy(block->attn);
        rmsnorm_destroy(block->attn_norm);
        free(block);
        return NULL;
    }

    /* Step 4: FEP Router */
    block->router = router_create(pool, cfg);
    if (!block->router) {
        rmsnorm_destroy(block->ffn_norm);
        attention_destroy(block->attn);
        rmsnorm_destroy(block->attn_norm);
        free(block);
        return NULL;
    }

    /* Step 5: Allocate n_experts SwiGLU FFNs (configurable via D-004) */
    block->n_experts = cfg->n_experts;
    block->experts = (ExpertFFN **)calloc((size_t)cfg->n_experts,
                                          sizeof(ExpertFFN *));
    if (!block->experts) {
        router_destroy(block->router);
        rmsnorm_destroy(block->ffn_norm);
        attention_destroy(block->attn);
        rmsnorm_destroy(block->attn_norm);
        free(block);
        return NULL;
    }

    for (int32_t i = 0; i < cfg->n_experts; i++) {
        block->experts[i] = expert_ffn_create(pool, cfg);
        if (!block->experts[i]) {
            /* Cleanup already-created experts */
            for (int32_t j = 0; j < i; j++) {
                expert_ffn_destroy(block->experts[j]);
            }
            free(block->experts);
            router_destroy(block->router);
            rmsnorm_destroy(block->ffn_norm);
            attention_destroy(block->attn);
            rmsnorm_destroy(block->attn_norm);
            free(block);
            return NULL;
        }
    }

    /* Register config params for the forward pass */
    BlockParams bp;
    bp.d_model      = cfg->d_model;
    bp.n_experts    = cfg->n_experts;
    bp.n_active     = cfg->n_active;
    bp.d_ff         = cfg->d_ff;
    bp.rms_norm_eps = cfg->rms_norm_eps;
    block_register(block, &bp, cfg);

    return block;
}

/* ---- API: Forward ---- */

void hspa_block_forward(Tensor *out, const HSPABlock *block, const Tensor *x,
                        KVCache *cache, int32_t pos, MemoryPool *scratch,
                        bool training, int32_t n_active_override,
                        QATContext *qat) {
    if (!out || !block || !x || !cache || !scratch) {
        return;
    }

    const BlockParams *bp = block_lookup(block);
    const HSPAConfig *cfg = block_cfg_lookup(block);
    if (!bp || !cfg) {
        return;
    }

    /* For progressive sparsification: use override n_active if provided.
     * Create a local config copy to pass to router_forward. */
    HSPAConfig step_cfg = *cfg;
    if (n_active_override > 0 && n_active_override <= cfg->n_experts) {
        step_cfg.n_active = n_active_override;
    }

    int32_t seq_len = x->shape[0];
    int32_t d_model = bp->d_model;

    /* ================================================================
     * Step 1: norm_x = attn_norm(x)
     *
     * RMSNorm operates on 1D [d_model] vectors. We process each token
     * (row of x) independently by creating 1D slices.
     * ================================================================ */
    pool_reset(scratch);

    /* Allocate 2D buffer for normalized x: [seq_len, d_model] */
    int32_t x2d_shape[] = {seq_len, d_model};
    Tensor *norm_x = tensor_create(scratch, x2d_shape, 2, DTYPE_FP32);
    if (!norm_x) {
        return;
    }

    /* Per-row RMSNorm: extract row -> norm -> write back */
    {
        int32_t row_shape[] = {d_model};
        Tensor *row_in = tensor_create(scratch, row_shape, 1, DTYPE_FP32);
        Tensor *row_out = tensor_create(scratch, row_shape, 1, DTYPE_FP32);
        if (!row_in || !row_out) {
            if (row_in) tensor_destroy(row_in);
            if (row_out) tensor_destroy(row_out);
            tensor_destroy(norm_x);
            return;
        }

        for (int32_t s = 0; s < seq_len; s++) {
            /* Copy row s from x into row_in */
            for (int32_t d = 0; d < d_model; d++) {
                int32_t idx2d[] = {s, d};
                int32_t idx1d[] = {d};
                tensor_set(row_in, idx1d, tensor_get(x, idx2d));
            }

            /* Apply RMSNorm */
            rmsnorm_forward(row_out, block->attn_norm, row_in);

            /* Copy result into norm_x row s */
            for (int32_t d = 0; d < d_model; d++) {
                int32_t idx2d[] = {s, d};
                int32_t idx1d[] = {d};
                tensor_set(norm_x, idx2d, tensor_get(row_out, idx1d));
            }
        }

        tensor_destroy(row_out);
        tensor_destroy(row_in);
    }

    /* ================================================================
     * Step 2: attn_out = attention_forward(norm_x, cache, pos)
     *
     * attention_forward allocates its own internal scratch pool,
     * so we just pass our 2D tensors directly.
     * ================================================================ */
    Tensor *attn_out = tensor_create(scratch, x2d_shape, 2, DTYPE_FP32);
    if (!attn_out) {
        tensor_destroy(norm_x);
        return;
    }
    tensor_fill(attn_out, 0.0f);

    attention_forward(attn_out, block->attn, norm_x, cache, pos, qat);

    /* ================================================================
     * Step 3: h = x + attn_out (first residual connection)
     * ================================================================ */
    Tensor *h = tensor_create(scratch, x2d_shape, 2, DTYPE_FP32);
    if (!h) {
        tensor_destroy(attn_out);
        tensor_destroy(norm_x);
        return;
    }

    op_add(h, x, attn_out);

    /* norm_x and attn_out no longer needed after this point.
     * We cannot free from scratch pool (arena), but they will be
     * reclaimed when pool is reset by caller. */
    tensor_destroy(attn_out);
    tensor_destroy(norm_x);

    /* ================================================================
     * Step 4: norm_h = ffn_norm(h)
     *
     * Same per-row RMSNorm approach as Step 1.
     * ================================================================ */
    Tensor *norm_h = tensor_create(scratch, x2d_shape, 2, DTYPE_FP32);
    if (!norm_h) {
        tensor_destroy(h);
        return;
    }

    {
        int32_t row_shape[] = {d_model};
        Tensor *row_in = tensor_create(scratch, row_shape, 1, DTYPE_FP32);
        Tensor *row_out = tensor_create(scratch, row_shape, 1, DTYPE_FP32);
        if (!row_in || !row_out) {
            if (row_in) tensor_destroy(row_in);
            if (row_out) tensor_destroy(row_out);
            tensor_destroy(norm_h);
            tensor_destroy(h);
            return;
        }

        for (int32_t s = 0; s < seq_len; s++) {
            for (int32_t d = 0; d < d_model; d++) {
                int32_t idx2d[] = {s, d};
                int32_t idx1d[] = {d};
                tensor_set(row_in, idx1d, tensor_get(h, idx2d));
            }

            rmsnorm_forward(row_out, block->ffn_norm, row_in);

            for (int32_t d = 0; d < d_model; d++) {
                int32_t idx2d[] = {s, d};
                int32_t idx1d[] = {d};
                tensor_set(norm_h, idx2d, tensor_get(row_out, idx1d));
            }
        }

        tensor_destroy(row_out);
        tensor_destroy(row_in);
    }

    /* ================================================================
     * Step 5: Mixture of Experts -- Expert-Batched Dispatch (D-032)
     *
     * Two-phase approach for efficient BLAS utilization:
     *   Phase 1: Route all tokens, build per-expert batch lists
     *   Phase 2: For each expert with assigned tokens:
     *            gather inputs -> single batched FFN -> scatter outputs
     *
     * Instead of S*k individual [1,D]x[D,d_ff] matmuls (tiny, CPU-only),
     * we get K batched [N_j,D]x[D,d_ff] matmuls (32x larger per call).
     * ================================================================ */

    /* Per-expert token assignment list for batched dispatch. */
    typedef struct {
        int32_t *token_indices;
        float   *route_weights;
        int32_t count;
    } ExpertBatch;

    Tensor *moe_out = tensor_create(scratch, x2d_shape, 2, DTYPE_FP32);
    if (!moe_out) {
        tensor_destroy(norm_h);
        tensor_destroy(h);
        return;
    }
    tensor_fill(moe_out, 0.0f);

    /* Create a dedicated scratch pool for expert FFN forward passes.
     * expert_ffn_forward resets its scratch pool internally, so we
     * need a separate one to avoid clobbering our intermediates.
     * Sized for batched [N_j, d_ff] intermediates (N_j up to seq_len). */
    size_t expert_scratch_size = (size_t)seq_len * (size_t)bp->d_ff * 3 * sizeof(float)
                               + (size_t)seq_len * (size_t)d_model * sizeof(float)
                               + 65536;
    if (expert_scratch_size < 1024 * 1024) {
        expert_scratch_size = 1024 * 1024;
    }
    MemoryPool *expert_scratch = pool_create(expert_scratch_size, POOL_SCRATCH);
    if (!expert_scratch) {
        tensor_destroy(moe_out);
        tensor_destroy(norm_h);
        tensor_destroy(h);
        return;
    }

    int32_t n_experts = bp->n_experts;
    int32_t n_active  = step_cfg.n_active;

    /* Guard: stack-allocated ExpertBatch array is fixed at 16 */
    if (n_experts > 16) {
        pool_destroy(expert_scratch);
        tensor_destroy(moe_out);
        tensor_destroy(norm_h);
        tensor_destroy(h);
        return;
    }

    /* Allocate per-expert token index and weight arrays from scratch pool.
     * Each expert can receive at most seq_len tokens (worst case). */
    int32_t *all_token_indices = (int32_t *)pool_alloc(
        scratch, (size_t)n_experts * (size_t)seq_len * sizeof(int32_t),
        POOL_MIN_ALIGNMENT);
    float *all_route_weights = (float *)pool_alloc(
        scratch, (size_t)n_experts * (size_t)seq_len * sizeof(float),
        POOL_MIN_ALIGNMENT);

    /* Gather and scatter buffers (shared across experts, reused sequentially) */
    float *gather_buf = (float *)pool_alloc(
        scratch, (size_t)seq_len * (size_t)d_model * sizeof(float),
        POOL_MIN_ALIGNMENT);
    float *expert_out_buf = (float *)pool_alloc(
        scratch, (size_t)seq_len * (size_t)d_model * sizeof(float),
        POOL_MIN_ALIGNMENT);

    if (!all_token_indices || !all_route_weights || !gather_buf || !expert_out_buf) {
        pool_destroy(expert_scratch);
        tensor_destroy(moe_out);
        tensor_destroy(norm_h);
        tensor_destroy(h);
        return;
    }

    /* Initialize ExpertBatch structs (stack-allocated, K <= 16) */
    ExpertBatch batches[16];
    for (int32_t j = 0; j < n_experts && j < 16; j++) {
        batches[j].token_indices = all_token_indices + j * seq_len;
        batches[j].route_weights = all_route_weights + j * seq_len;
        batches[j].count = 0;
    }

    /* Router input: 1D [d_model] for per-token routing */
    int32_t tok_1d_shape[] = {d_model};
    Tensor *tok_1d = tensor_create(scratch, tok_1d_shape, 1, DTYPE_FP32);
    if (!tok_1d) {
        pool_destroy(expert_scratch);
        tensor_destroy(moe_out);
        tensor_destroy(norm_h);
        tensor_destroy(h);
        return;
    }

    /* ==== PHASE 1: Route all tokens, build per-expert batches ==== */
    for (int32_t t = 0; t < seq_len; t++) {
        /* Extract norm_h[t] as 1D [d_model] for the router */
        for (int32_t d = 0; d < d_model; d++) {
            int32_t idx2d[] = {t, d};
            int32_t idx1d[] = {d};
            tensor_set(tok_1d, idx1d, tensor_get(norm_h, idx2d));
        }

        /* Route this token to top-k experts */
        RoutingDecision rd = router_forward(block->router, tok_1d, &step_cfg,
                                            training);

        /* If routing failed (NULL pointers), skip this token */
        if (!rd.expert_ids || !rd.expert_weights) {
            if (rd.expert_ids) free(rd.expert_ids);
            if (rd.expert_weights) free(rd.expert_weights);
            continue;
        }

        /* Record assignments into per-expert batch lists */
        for (int32_t i = 0; i < n_active; i++) {
            int32_t expert_id = rd.expert_ids[i];
            float weight = rd.expert_weights[i];

            if (expert_id < 0 || expert_id >= n_experts) {
                continue;
            }

            int32_t idx = batches[expert_id].count;
            batches[expert_id].token_indices[idx] = t;
            batches[expert_id].route_weights[idx] = weight;
            batches[expert_id].count++;
        }

        free(rd.expert_ids);
        free(rd.expert_weights);
    }

    tensor_destroy(tok_1d);

    /* ==== PHASE 2: Batched expert dispatch ==== */
    for (int32_t j = 0; j < n_experts; j++) {
        int32_t N_j = batches[j].count;
        if (N_j == 0) continue;

        /* GATHER: collect input rows for expert j */
        for (int32_t i = 0; i < N_j; i++) {
            int32_t t = batches[j].token_indices[i];
            memcpy(gather_buf + (size_t)i * d_model,
                   (const float *)norm_h->data + (size_t)t * d_model,
                   (size_t)d_model * sizeof(float));
        }

        /* Create stack-initialized Tensor views over the raw buffers.
         * is_view=true prevents tensor_destroy from freeing the data. */
        Tensor gather_t;
        gather_t.data      = gather_buf;
        gather_t.shape[0]  = N_j;
        gather_t.shape[1]  = d_model;
        gather_t.shape[2]  = 1;
        gather_t.shape[3]  = 1;
        gather_t.stride[0] = d_model;
        gather_t.stride[1] = 1;
        gather_t.stride[2] = 1;
        gather_t.stride[3] = 1;
        gather_t.ndim      = 2;
        gather_t.dtype     = DTYPE_FP32;
        gather_t.pool      = NULL;
        gather_t.is_view   = true;

        Tensor output_t;
        output_t.data      = expert_out_buf;
        output_t.shape[0]  = N_j;
        output_t.shape[1]  = d_model;
        output_t.shape[2]  = 1;
        output_t.shape[3]  = 1;
        output_t.stride[0] = d_model;
        output_t.stride[1] = 1;
        output_t.stride[2] = 1;
        output_t.stride[3] = 1;
        output_t.ndim      = 2;
        output_t.dtype     = DTYPE_FP32;
        output_t.pool      = NULL;
        output_t.is_view   = true;

        /* SINGLE batched FFN call: [N_j, D] -> [N_j, D] */
        expert_ffn_forward(&output_t, block->experts[j],
                           &gather_t, expert_scratch, qat);

        /* SCATTER: weighted accumulate back to correct token positions */
        float *moe_data = (float *)moe_out->data;
        for (int32_t i = 0; i < N_j; i++) {
            int32_t t = batches[j].token_indices[i];
            float w = batches[j].route_weights[i];
            const float *src = expert_out_buf + (size_t)i * d_model;
            float *dst = moe_data + (size_t)t * d_model;
            for (int32_t d = 0; d < d_model; d++) {
                dst[d] += w * src[d];
            }
        }
    }

    pool_destroy(expert_scratch);

    /* ================================================================
     * Step 6: out = h + moe_out (second residual connection)
     * ================================================================ */
    op_add(out, h, moe_out);

    /* Cleanup intermediate tensors */
    tensor_destroy(moe_out);
    tensor_destroy(norm_h);
    tensor_destroy(h);
}

/* ---- API: Destroy ---- */

void hspa_block_destroy(HSPABlock *block) {
    if (!block) {
        return;
    }

    block_unregister(block);

    /* Destroy experts array */
    if (block->experts) {
        for (int32_t i = 0; i < block->n_experts; i++) {
            if (block->experts[i]) {
                expert_ffn_destroy(block->experts[i]);
            }
        }
        free(block->experts);
    }

    /* Destroy router */
    if (block->router) {
        router_destroy(block->router);
    }

    /* Destroy norms */
    if (block->ffn_norm) {
        rmsnorm_destroy(block->ffn_norm);
    }
    if (block->attn_norm) {
        rmsnorm_destroy(block->attn_norm);
    }

    /* Destroy attention */
    if (block->attn) {
        attention_destroy(block->attn);
    }

    free(block);
}
