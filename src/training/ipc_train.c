/* ipc_train.c -- iPC training step for HSPA v1.
 *
 * Implements the incremental Predictive Coding (iPC) training loop:
 *   Phase 0: Zero gradients, reset state, seed RNG
 *   Phase 1: Forward pass to initialize value nodes v[0..L]
 *   Phase 2: T inference iterations refine value nodes and accumulate gradients
 *   Phase 3: Clip gradients, apply SGD, compute loss
 *
 * First implementation simplifications:
 *   - Attention backward: approximate dW_q/dW_k/dW_v via norm_in^T @ delta
 *   - dW_o uses the cached context (attention internal output before W_o)
 *   - Per-token per-expert MoE backward with full SwiGLU gradient
 *   - Router entropy gradient for load balancing
 *   - Identity Jacobian for value node updates
 */

#include "ipc_train.h"
#include "embedding.h"
#include "hspa_block.h"
#include "ipc_state.h"
#include "loss.h"
#include "ops.h"
#include "qat_context.h"
#include "rmsnorm.h"
#include "router.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- QAT coverage observability seam (Program 3, Phase 7, D-613) ----
 * Single-writer: ipc_train_step; single-reader: tests. Thread-safe by design
 * (single-threaded lab). */
static int s_last_ipc_qat_covered_count = 0;
static int s_last_ipc_qat_cache_hits    = 0;

int ipc_last_qat_covered_count(void) {
    return s_last_ipc_qat_covered_count;
}

int ipc_last_qat_cache_hits(void) {
    return s_last_ipc_qat_cache_hits;
}

/* ---- Micro-experiment config ---- */

HSPAConfig hspa_config_micro(void) {
    HSPAConfig cfg;
    cfg.n_layers       = 4;
    cfg.d_model        = 128;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = 32;
    cfg.d_ff           = 64;
    cfg.n_experts      = 4;
    cfg.n_active       = 1;
    cfg.vocab_size     = 256;
    cfg.max_seq_len    = 128;
    cfg.ipc_iterations = 5;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* ---- Small config (~10M params) ---- */

HSPAConfig hspa_config_small(void) {
    HSPAConfig cfg;
    cfg.n_layers       = 6;
    cfg.d_model        = 256;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = 64;
    cfg.d_ff           = 192;
    cfg.n_experts      = 8;
    cfg.n_active       = 2;
    cfg.vocab_size     = 4096;
    cfg.max_seq_len    = 512;
    cfg.ipc_iterations = 5;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* ---- Medium config (~50M params) ---- */

HSPAConfig hspa_config_medium(void) {
    HSPAConfig cfg;
    cfg.n_layers       = 8;
    cfg.d_model        = 512;
    cfg.n_heads        = 8;
    cfg.n_kv_heads     = 4;
    cfg.head_dim       = 64;
    cfg.d_ff           = 384;
    cfg.n_experts      = 8;
    cfg.n_active       = 2;
    /* P-MEDIUM-V4096-LATENT remediation (D-182, PI condition #3, 2026-04-24):
     * previous value 4096 was a dead-default masked by --stream runtime override
     * at scale_experiment.c:589. D-122 config_drift detector caught it externally,
     * but the internal source of truth should match. All Program 2+ workflows
     * use the 32K tokenizer (tokenizer_32k.bin); Phase-B (D-091) adopted V=32768. */
    cfg.vocab_size     = 32768;
    cfg.max_seq_len    = 512;
    cfg.ipc_iterations = 5;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* ---- Phase C 100M config (~119.6M params) ----
 * Per data/engineering/phase_c_design.md §1.7 (chief_scientist, 2026-04-16).
 * Single-variable deltas vs medium: D 512→768, H 8→12, n_kv_heads 4→6,
 * d_ff 384→512, V 4096→32768 (the Phase B tokenizer default).
 * Unchanged: L=8, head_dim=64, K=8, k=2, max_seq_len=512, FP32/FP32. */

HSPAConfig hspa_config_100m(void) {
    HSPAConfig cfg;
    cfg.n_layers       = 8;
    cfg.d_model        = 768;
    cfg.n_heads        = 12;
    cfg.n_kv_heads     = 6;
    cfg.head_dim       = 64;
    cfg.d_ff           = 512;
    cfg.n_experts      = 8;
    cfg.n_active       = 2;
    cfg.vocab_size     = 32768;
    cfg.max_seq_len    = 512;
    cfg.ipc_iterations = 5;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* ---- Dense-A compute-matched control (~34.62M params) ----
 * Per data/engineering/dense_50m_control_design.md §1.3 and
 * programs/program_2_dense_vs_moe_sub100m/question.md operational form.
 *
 * FLOPs-matched to Rev-2 MoE active path:
 *   MoE active FFN FLOPs ∝ k * d_ff_moe = 2 * 384 = 768
 *   Dense FFN FLOPs      ∝ 1 * d_ff_dense
 *   → d_ff_dense = 768 (2x medium's d_ff).
 *
 * Program 2 comparison: this Dense-A (K=1, d_ff=768) vs Rev-2 MoE (K=8, k=2,
 * d_ff=384). Total params: Dense-A ~34.62M, MoE ~62.99M total / ~34.7M active.
 *
 * NOTE: This factory exposes the n_experts=1 code path for the first time in
 * the codebase. Router at K=1 is structurally present (W_mu, W_sigma are
 * [D,1]) but informationally trivial (softmax of 1 element = 1.0, entropy =
 * log(1) = 0). MoE-specific features (loss-free-balance, default-moe aux loss,
 * entropy-penalty, temp-anneal) are DISALLOWED with this config and the CLI
 * (scale_experiment.c) enforces this as a guardrail. */

HSPAConfig hspa_config_dense_50m_a(void) {
    HSPAConfig cfg;
    cfg.n_layers       = 8;
    cfg.d_model        = 512;
    cfg.n_heads        = 8;
    cfg.n_kv_heads     = 4;
    cfg.head_dim       = 64;
    cfg.d_ff           = 768;    /* 2 * MoE d_ff=384; FLOPs-matched k=2 active */
    cfg.n_experts      = 1;      /* DENSE: single expert */
    cfg.n_active       = 1;      /* 100% activation */
    cfg.vocab_size     = 32768;  /* Program 2 benchmark tokenizer */
    cfg.max_seq_len    = 512;
    cfg.ipc_iterations = 5;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* ---- Helper: copy tensor data ---- */

static void tensor_copy(Tensor *dst, const Tensor *src) {
    size_t n = tensor_numel(src);
    const float *s = (const float *)src->data;
    float *d = (float *)dst->data;
    memcpy(d, s, n * sizeof(float));
}

/* ---- Helper: per-row RMSNorm for 2D tensors ---- */

static void rms_norm_2d(Tensor *out, const Tensor *x,
                        const RMSNorm *norm, int32_t seq_len,
                        int32_t d_model, MemoryPool *scratch) {
    int32_t row_shape[] = {d_model};
    Tensor *row_in = tensor_create(scratch, row_shape, 1, DTYPE_FP32);
    Tensor *row_out = tensor_create(scratch, row_shape, 1, DTYPE_FP32);
    if (!row_in || !row_out) {
        if (row_in) tensor_destroy(row_in);
        if (row_out) tensor_destroy(row_out);
        return;
    }

    for (int32_t s = 0; s < seq_len; s++) {
        for (int32_t d = 0; d < d_model; d++) {
            int32_t idx2d[] = {s, d};
            int32_t idx1d[] = {d};
            tensor_set(row_in, idx1d, tensor_get(x, idx2d));
        }

        rmsnorm_forward(row_out, norm, row_in);

        for (int32_t d = 0; d < d_model; d++) {
            int32_t idx2d[] = {s, d};
            int32_t idx1d[] = {d};
            tensor_set(out, idx2d, tensor_get(row_out, idx1d));
        }
    }

    tensor_destroy(row_out);
    tensor_destroy(row_in);
}

/* ---- Helper: accumulate matmul A^T @ B into dW ----
 * A: [S, M], B: [S, N], dW: [M, N]
 * dW += A^T @ B
 */
static void accum_matmul_tn(Tensor *dW, const Tensor *A, const Tensor *B,
                            int32_t S, int32_t M, int32_t N) {
    const float *a = (const float *)A->data;
    const float *b = (const float *)B->data;
    float *dw = (float *)dW->data;
    for (int32_t s = 0; s < S; s++) {
        const float *as = a + s * M;
        const float *bs = b + s * N;
        for (int32_t i = 0; i < M; i++) {
            float ai = as[i];
            for (int32_t j = 0; j < N; j++) {
                dw[i * N + j] += ai * bs[j];
            }
        }
    }
}

/* ---- iPC Training Step ---- */

TrainStepResult ipc_train_step(HSPAModel *model, ModelGrad *grads,
                               AdamState *adam,
                               IPCTrainState *state,
                               const int32_t *tokens, const int32_t *targets,
                               int32_t seq_len,
                               const HSPAConfig *cfg, const TrainConfig *tcfg,
                               int32_t step, int32_t micro_batch_idx) {
    TrainStepResult result;
    memset(&result, 0, sizeof(result));

    if (!model || !grads || !state || !tokens || !targets || !cfg || !tcfg) {
        return result;
    }

    int32_t L = cfg->n_layers;
    int32_t D = cfg->d_model;
    int32_t V = cfg->vocab_size;
    int32_t K = cfg->n_experts;
    int32_t k = cfg->n_active;
    int32_t d_ff = cfg->d_ff;
    int32_t T = tcfg->T;

    /* ==================================================================
     * Phase 0: Init -- zero grads (first micro-batch only), reset state
     * ================================================================== */
    int32_t accum_steps = tcfg->grad_accum_steps > 0 ? tcfg->grad_accum_steps : 1;
    if (micro_batch_idx == 0) {
        grad_zero(grads);
    }
    ipc_state_reset(state);
    srand((unsigned int)(step + 1 + micro_batch_idx));

    /* Allocate a working scratch pool for training intermediates.
     * Size estimate: enough for several [S,D], [S,V], and [S,d_ff] tensors
     * plus per-token expert intermediates. */
    size_t scratch_size = (size_t)seq_len * (size_t)D * sizeof(float) * 32
                        + (size_t)seq_len * (size_t)V * sizeof(float) * 2
                        + (size_t)seq_len * (size_t)d_ff * sizeof(float) * 4
                        + 4 * 1024 * 1024;
    MemoryPool *work = pool_create(scratch_size, POOL_SCRATCH);
    if (!work) {
        return result;
    }

    /* Pre-allocate reusable tensors on the work pool */
    int32_t sd_shape[] = {seq_len, D};
    int32_t sv_shape[] = {seq_len, V};

    Tensor *v_out      = tensor_create(work, sd_shape, 2, DTYPE_FP32);
    Tensor *f_pred     = tensor_create(work, sd_shape, 2, DTYPE_FP32);
    Tensor *delta      = tensor_create(work, sd_shape, 2, DTYPE_FP32);
    Tensor *grad_v     = tensor_create(work, sd_shape, 2, DTYPE_FP32);
    Tensor *normed     = tensor_create(work, sd_shape, 2, DTYPE_FP32);
    Tensor *logits     = tensor_create(work, sv_shape, 2, DTYPE_FP32);
    Tensor *grad_logits = tensor_create(work, sv_shape, 2, DTYPE_FP32);
    Tensor *delta_top  = tensor_create(work, sd_shape, 2, DTYPE_FP32);
    Tensor *tmp_sd     = tensor_create(work, sd_shape, 2, DTYPE_FP32);
    Tensor *norm_cache = tensor_create(work, sd_shape, 2, DTYPE_FP32);

    if (!v_out || !f_pred || !delta || !grad_v || !normed ||
        !logits || !grad_logits || !delta_top || !tmp_sd || !norm_cache) {
        pool_destroy(work);
        return result;
    }

    /* Scratch pool for block forward passes (separate to not clobber work pool) */
    size_t blk_scratch_size = (size_t)seq_len * (size_t)D * sizeof(float) * 16
                            + (size_t)D * (size_t)d_ff * sizeof(float) * 4
                            + 4 * 1024 * 1024;
    MemoryPool *blk_scratch = pool_create(blk_scratch_size, POOL_SCRATCH);
    if (!blk_scratch) {
        pool_destroy(work);
        return result;
    }

    /* Pre-allocate reusable scratch buffers for Phase 2 gradient computation.
     * Router prediction-error gradient uses per-token buffers;
     * FFN weight gradients use batched buffers allocated per-layer (D-032). */
    float *scratch_expert_out     = (float *)malloc((size_t)D * sizeof(float));
    float *scratch_dlogit         = (float *)malloc((size_t)K * sizeof(float));
    float *scratch_probs          = (float *)malloc((size_t)K * sizeof(float));
    float *scratch_dH_dlogit      = (float *)malloc((size_t)K * sizeof(float));

    if (!scratch_expert_out || !scratch_dlogit || !scratch_probs ||
        !scratch_dH_dlogit) {
        free(scratch_expert_out);
        free(scratch_dlogit); free(scratch_probs); free(scratch_dH_dlogit);
        pool_destroy(blk_scratch);
        pool_destroy(work);
        return result;
    }

    /* ==================================================================
     * QAT context: per-step w_hat cache (Program 3, Phase 7, D-613).
     * Created here (after all allocation-guard early returns) so early
     * exits above do NOT need to destroy it.  All early exits BELOW
     * (inner-loop allocation failures) must call qat_context_destroy(qat).
     * capacity 256 > 224 in-scope weights.
     * ================================================================== */
    QATContext *qat = qat_context_create(tcfg->use_qat, tcfg->qat_group_size, 256);
    /* qat_context_create returns NULL on failure.
     * NULL qat is fully safe: all qat_context_* calls tolerate NULL.
     * We continue even on failure (training proceeds without QAT). */

    /* Begin the per-step cache: invalidates all cached w_hat from last step. */
    qat_context_begin_step(qat);

    /* ==================================================================
     * Phase 1: Forward pass to initialize value nodes
     *
     * v[0] = embedding(tokens)
     * For l=0..L-1: v[l+1] = block_forward(v[l])
     * Also capture routing decisions at each layer.
     * ================================================================== */

    /* v[0] = embedding output */
    embedding_forward(state->vn.values[0], model->embed, tokens, seq_len);

    /* Forward through each block */
    for (int32_t l = 0; l < L; l++) {
        /* Reset KV cache position for each training step */
        model->kv_caches[l]->pos = 0;

        pool_reset(blk_scratch);
        tensor_fill(v_out, 0.0f);

        hspa_block_forward(v_out, model->layers[l], state->vn.values[l],
                           model->kv_caches[l], 0, blk_scratch, true,
                           cfg->n_active, qat);

        /* MoE pre-registration: force ALL K experts' W_gate/W_up/W_down into
         * the w_hat cache for this step, regardless of which experts are actually
         * routed.  Without this, only k=2 of K=8 experts per token fire in the
         * natural forward, leaving covered_count < 224 (D-613, §4.1). */
        if (qat && qat_context_is_enabled(qat)) {
            HSPABlock *prereg_block = model->layers[l];
            for (int32_t j = 0; j < K; j++) {
                ExpertFFN *expert = prereg_block->experts[j];
                (void)qat_context_w_hat(qat, expert->W_gate);
                (void)qat_context_w_hat(qat, expert->W_up);
                (void)qat_context_w_hat(qat, expert->W_down);
            }
        }

        /* Copy output into v[l+1] */
        tensor_copy(state->vn.values[l + 1], v_out);

        /* Capture routing decisions: re-run router per token to save them */
        RoutingState *rs = &state->routing[l];
        HSPABlock *block = model->layers[l];

        /* Compute pre-FFN norm for routing (same as in hspa_block_forward) */
        /* First compute h = v[l] + attn_out, then norm_h = ffn_norm(h).
         * But we don't have the intermediates from the forward pass.
         * Instead, re-compute the normalized hidden states for the router. */

        /* Compute pre-attn norm */
        pool_reset(blk_scratch);
        Tensor *norm_x = tensor_create(blk_scratch, sd_shape, 2, DTYPE_FP32);
        Tensor *attn_out_cache = tensor_create(blk_scratch, sd_shape, 2, DTYPE_FP32);
        Tensor *h_cache = tensor_create(blk_scratch, sd_shape, 2, DTYPE_FP32);
        Tensor *norm_h = tensor_create(blk_scratch, sd_shape, 2, DTYPE_FP32);
        if (!norm_x || !attn_out_cache || !h_cache || !norm_h) {
            qat_context_destroy(qat);
            pool_destroy(blk_scratch);
            pool_destroy(work);
            return result;
        }

        /* Reset KV cache to redo attention for caching intermediates */
        model->kv_caches[l]->pos = 0;

        rms_norm_2d(norm_x, state->vn.values[l], block->attn_norm,
                    seq_len, D, blk_scratch);

        /* Cache pre_attn_norm */
        tensor_copy(state->layer_caches[l].pre_attn_norm, norm_x);

        /* Attention forward to get attn_out.
         * Pass qat: W_q/W_k/W_v/W_o are already in cache from hspa_block_forward
         * above (cache hits, does not increment covered_count). */
        tensor_fill(attn_out_cache, 0.0f);
        attention_forward(attn_out_cache, block->attn, norm_x,
                          model->kv_caches[l], 0, qat);

        /* Cache attn_out */
        tensor_copy(state->layer_caches[l].attn_out, attn_out_cache);

        /* h = v[l] + attn_out (post-attention residual) */
        op_add(h_cache, state->vn.values[l], attn_out_cache);
        tensor_copy(state->layer_caches[l].post_attn_residual, h_cache);

        /* norm_h = ffn_norm(h) */
        rms_norm_2d(norm_h, h_cache, block->ffn_norm, seq_len, D, blk_scratch);
        tensor_copy(state->layer_caches[l].pre_ffn_norm, norm_h);

        /* ---- Expert-Batched Forward Cache (D-032) ----
         *
         * Phase 1: Route all tokens per-token (router is inherently per-token).
         * Phase 2: Batch gate/up projections per expert using op_matmul.
         *          Scatter results into per-token cache slots.
         */

        /* Per-expert token assignment list for batched dispatch. */
        typedef struct {
            int32_t *token_indices;
            int32_t *active_slots;   /* which active slot (0..k-1) each came from */
            int32_t count;
        } FwdExpertBatch;

        /* Allocate per-expert batch tracking arrays from blk_scratch */
        int32_t *fwd_all_indices = (int32_t *)pool_alloc(
            blk_scratch, (size_t)K * (size_t)seq_len * sizeof(int32_t),
            POOL_MIN_ALIGNMENT);
        int32_t *fwd_all_slots = (int32_t *)pool_alloc(
            blk_scratch, (size_t)K * (size_t)seq_len * sizeof(int32_t),
            POOL_MIN_ALIGNMENT);

        /* Gather buffer for batched matmul: [seq_len, D] (worst case) */
        float *fwd_gather = (float *)pool_alloc(
            blk_scratch, (size_t)seq_len * (size_t)D * sizeof(float),
            POOL_MIN_ALIGNMENT);
        /* Output buffers for gate and up: [seq_len, d_ff] each */
        float *fwd_gate_out = (float *)pool_alloc(
            blk_scratch, (size_t)seq_len * (size_t)d_ff * sizeof(float),
            POOL_MIN_ALIGNMENT);
        float *fwd_up_out = (float *)pool_alloc(
            blk_scratch, (size_t)seq_len * (size_t)d_ff * sizeof(float),
            POOL_MIN_ALIGNMENT);

        if (!fwd_all_indices || !fwd_all_slots || !fwd_gather ||
            !fwd_gate_out || !fwd_up_out) {
            qat_context_destroy(qat);
            pool_destroy(blk_scratch);
            pool_destroy(work);
            return result;
        }

        /* Guard: stack-allocated batch arrays are fixed at 16 */
        if (K > 16) {
            qat_context_destroy(qat);
            pool_destroy(blk_scratch);
            pool_destroy(work);
            return result;
        }

        FwdExpertBatch fwd_batches[16];
        for (int32_t j = 0; j < K && j < 16; j++) {
            fwd_batches[j].token_indices = fwd_all_indices + j * seq_len;
            fwd_batches[j].active_slots  = fwd_all_slots + j * seq_len;
            fwd_batches[j].count = 0;
        }

        /* Phase 1: Route all tokens, build per-expert batch lists */
        double entropy_sum = 0.0;
        int32_t tok_1d_shape[] = {D};
        Tensor *tok_1d = tensor_create(blk_scratch, tok_1d_shape, 1, DTYPE_FP32);
        if (!tok_1d) {
            qat_context_destroy(qat);
            pool_destroy(blk_scratch);
            pool_destroy(work);
            return result;
        }

        for (int32_t s = 0; s < seq_len; s++) {
            /* Extract norm_h[s] as 1D */
            for (int32_t d = 0; d < D; d++) {
                int32_t idx2d[] = {s, d};
                int32_t idx1d[] = {d};
                tensor_set(tok_1d, idx1d, tensor_get(norm_h, idx2d));
            }

            RoutingDecision rd = router_forward(block->router, tok_1d,
                                                cfg, true);
            if (!rd.expert_ids || !rd.expert_weights) {
                /* Fallback: select expert 0 with weight 1 */
                if (rd.expert_ids) free(rd.expert_ids);
                if (rd.expert_weights) free(rd.expert_weights);
                for (int32_t a = 0; a < k; a++) {
                    rs->expert_ids[s * k + a] = a % K;
                    rs->expert_weights[s * k + a] = 1.0f / (float)k;
                }
                rs->entropies[s] = 0.0f;
            } else {
                for (int32_t a = 0; a < k; a++) {
                    rs->expert_ids[s * k + a] = rd.expert_ids[a];
                    rs->expert_weights[s * k + a] = rd.expert_weights[a];
                }
                rs->entropies[s] = rd.entropy;
                entropy_sum += (double)rd.entropy;
                free(rd.expert_ids);
                free(rd.expert_weights);
            }

            /* Record which tokens go to which expert */
            for (int32_t a = 0; a < k; a++) {
                int32_t eid = rs->expert_ids[s * k + a];
                if (eid < 0 || eid >= K) continue;

                int32_t bi = fwd_batches[eid].count;
                fwd_batches[eid].token_indices[bi] = s;
                fwd_batches[eid].active_slots[bi] = a;
                fwd_batches[eid].count++;
            }
        }
        rs->mean_entropy = (float)(entropy_sum / (double)seq_len);
        rs->n_active = k;  /* Update for progressive sparsification (P2) */

        tensor_destroy(tok_1d);

        /* Phase 2: Batched gate/up projections per expert */
        for (int32_t j = 0; j < K; j++) {
            int32_t N_j = fwd_batches[j].count;
            if (N_j == 0) continue;

            ExpertFFN *expert = block->experts[j];
            const float *norm_h_data = (const float *)norm_h->data;

            /* GATHER: collect input rows for expert j */
            for (int32_t i = 0; i < N_j; i++) {
                int32_t s = fwd_batches[j].token_indices[i];
                memcpy(fwd_gather + (size_t)i * D,
                       norm_h_data + (size_t)s * D,
                       (size_t)D * sizeof(float));
            }

            /* Create stack-initialized Tensor views for batched matmul */
            Tensor gather_t;
            gather_t.data      = fwd_gather;
            gather_t.shape[0]  = N_j;
            gather_t.shape[1]  = D;
            gather_t.shape[2]  = 1;
            gather_t.shape[3]  = 1;
            gather_t.stride[0] = D;
            gather_t.stride[1] = 1;
            gather_t.stride[2] = 1;
            gather_t.stride[3] = 1;
            gather_t.ndim      = 2;
            gather_t.dtype     = DTYPE_FP32;
            gather_t.pool      = NULL;
            gather_t.is_view   = true;

            Tensor gate_out_t;
            gate_out_t.data      = fwd_gate_out;
            gate_out_t.shape[0]  = N_j;
            gate_out_t.shape[1]  = d_ff;
            gate_out_t.shape[2]  = 1;
            gate_out_t.shape[3]  = 1;
            gate_out_t.stride[0] = d_ff;
            gate_out_t.stride[1] = 1;
            gate_out_t.stride[2] = 1;
            gate_out_t.stride[3] = 1;
            gate_out_t.ndim      = 2;
            gate_out_t.dtype     = DTYPE_FP32;
            gate_out_t.pool      = NULL;
            gate_out_t.is_view   = true;

            Tensor up_out_t;
            up_out_t.data      = fwd_up_out;
            up_out_t.shape[0]  = N_j;
            up_out_t.shape[1]  = d_ff;
            up_out_t.shape[2]  = 1;
            up_out_t.shape[3]  = 1;
            up_out_t.stride[0] = d_ff;
            up_out_t.stride[1] = 1;
            up_out_t.stride[2] = 1;
            up_out_t.stride[3] = 1;
            up_out_t.ndim      = 2;
            up_out_t.dtype     = DTYPE_FP32;
            up_out_t.pool      = NULL;
            up_out_t.is_view   = true;

            /* Batched gate = gather @ W_gate (or w_hat): [N_j, D] @ [D, d_ff] = [N_j, d_ff].
             * Cache hit: W_gate already registered by hspa_block_forward + pre-registration
             * above; op_matmul_qat uses the same w_hat (STE consistency). */
            op_matmul_qat(&gate_out_t, &gather_t, expert->W_gate, qat);

            /* Batched up = gather @ W_up (or w_hat): same cache-hit logic. */
            op_matmul_qat(&up_out_t, &gather_t, expert->W_up, qat);

            /* SCATTER: write batched results into per-token cache slots */
            for (int32_t i = 0; i < N_j; i++) {
                int32_t s = fwd_batches[j].token_indices[i];
                int32_t a = fwd_batches[j].active_slots[i];
                int32_t cache_idx = s * k + a;

                float *gate_data = (float *)state->layer_caches[l]
                                       .expert_gate[cache_idx]->data;
                float *up_data = (float *)state->layer_caches[l]
                                     .expert_up[cache_idx]->data;

                memcpy(gate_data, fwd_gate_out + (size_t)i * d_ff,
                       (size_t)d_ff * sizeof(float));
                memcpy(up_data, fwd_up_out + (size_t)i * d_ff,
                       (size_t)d_ff * sizeof(float));
            }
        }
        tensor_destroy(norm_h);
        tensor_destroy(h_cache);
        tensor_destroy(attn_out_cache);
        tensor_destroy(norm_x);
    }

    /* ==================================================================
     * Phase 2: T inference iterations
     *
     * For t = 0..T-1:
     *   A. Compute prediction errors: epsilon[l] = v[l+1] - f_l(v[l])
     *   B. Compute output gradient (delta_top) from cross-entropy
     *   C. Update value nodes using prediction errors
     *   D. Compute weight gradients from prediction errors
     * ================================================================== */

    float max_vn_delta = 0.0f;

    /* Pre-allocate backward batch buffers once before the T loop.
     * Sizes depend on K, seq_len, D, d_ff — all constant per call.
     * Hoisted from the inner T*L loop to avoid 560 malloc/free per step. */
    typedef struct {
        int32_t *token_indices;
        int32_t *active_slots;
        float   *route_weights_bk;
        int32_t count;
    } BwdExpertBatch;

    int32_t *bk_indices = (int32_t *)malloc(
        (size_t)K * (size_t)seq_len * sizeof(int32_t));
    int32_t *bk_slots = (int32_t *)malloc(
        (size_t)K * (size_t)seq_len * sizeof(int32_t));
    float *bk_weights = (float *)malloc(
        (size_t)K * (size_t)seq_len * sizeof(float));
    float *bk_norm_buf      = (float *)malloc((size_t)seq_len * (size_t)D * sizeof(float));
    float *bk_delta_buf     = (float *)malloc((size_t)seq_len * (size_t)D * sizeof(float));
    float *bk_gate_buf      = (float *)malloc((size_t)seq_len * (size_t)d_ff * sizeof(float));
    float *bk_up_buf        = (float *)malloc((size_t)seq_len * (size_t)d_ff * sizeof(float));
    float *bk_gate_act_buf  = (float *)malloc((size_t)seq_len * (size_t)d_ff * sizeof(float));
    float *bk_hidden_buf    = (float *)malloc((size_t)seq_len * (size_t)d_ff * sizeof(float));
    float *bk_silu_grad_buf = (float *)malloc((size_t)seq_len * (size_t)d_ff * sizeof(float));
    float *bk_delta_hidden  = (float *)malloc((size_t)seq_len * (size_t)d_ff * sizeof(float));
    float *bk_delta_gate_pre = (float *)malloc((size_t)seq_len * (size_t)d_ff * sizeof(float));
    float *bk_delta_up_pre  = (float *)malloc((size_t)seq_len * (size_t)d_ff * sizeof(float));
    /* Temporary buffer for matmul_tn result before accumulating into gradients.
     * Shape is either [D, d_ff] or [d_ff, D] depending on which gradient. */
    float *bk_dW_tmp = (float *)malloc((size_t)D * (size_t)d_ff * sizeof(float));

    bool bk_alloc_ok = bk_indices && bk_slots && bk_weights &&
        bk_norm_buf && bk_delta_buf && bk_gate_buf && bk_up_buf &&
        bk_gate_act_buf && bk_hidden_buf && bk_silu_grad_buf &&
        bk_delta_hidden && bk_delta_gate_pre && bk_delta_up_pre &&
        bk_dW_tmp;

    for (int32_t t = 0; t < T; t++) {

        /* ---- Step A: Compute prediction errors ---- */
        /* Bottom-up prediction: f_l(v[l]) predicts v[l+1].
         *   epsilon[l] = v[l+1] - f_l(v[l])
         *
         * After forward init: v[l+1] = f_l(v[l]), so epsilon[l] = 0.
         * The CE gradient at v[L] drives non-zero errors that cascade down. */
        for (int32_t l = 0; l < L; l++) {
            model->kv_caches[l]->pos = 0;
            pool_reset(blk_scratch);

            tensor_fill(f_pred, 0.0f);
            /* Pass qat: cache hits for all 224 weights (already registered in Phase 1).
             * STE consistency: prediction-error forward uses same w_hat as Phase 1. */
            hspa_block_forward(f_pred, model->layers[l], state->vn.values[l],
                               model->kv_caches[l], 0, blk_scratch, false,
                               cfg->n_active, qat);

            /* epsilon[l] = v[l+1] - f_l(v[l]) */
            op_sub(state->pe.errors[l], state->vn.values[l + 1], f_pred);
        }

        /* ---- Step B: Compute output gradient ---- */
        /* Compute logits from v[L]: norm -> matmul with embed^T */
        rms_norm_2d(normed, state->vn.values[L], model->final_norm,
                    seq_len, D, work);

        /* logits = normed @ embed->weight^T : [S,D] @ [V,D]^T = [S,V] */
        op_matmul_nt(logits, normed, model->embed->weight);

        /* Cross-entropy gradient */
        float ce_loss = op_cross_entropy(logits, targets, seq_len, grad_logits);
        (void)ce_loss;

        /* delta_top = grad_logits @ embed->weight : [S,V] @ [V,D] = [S,D] */
        op_matmul(delta_top, grad_logits, model->embed->weight);

        /* Accumulate embedding gradient (output head):
         * logits = normed @ W_embed^T, so dW_embed += grad_logits^T @ normed
         * [V,S] @ [S,D] = [V,D] */
        accum_matmul_tn(grads->embed_grad.dweight, grad_logits, normed,
                        seq_len, cfg->vocab_size, D);

        /* ---- Step C: Update value nodes ---- */
        /* Bottom-up: epsilon[l] = v[l+1] - f_l(v[l]).
         * v[l] appears in eps[l-1] as the target and eps[l] as the input:
         *   eps[l-1] = v[l] - f_{l-1}(v[l-1])  →  d/dv[l] = +epsilon[l-1]
         *   eps[l]   = v[l+1] - f_l(v[l])       →  d/dv[l] = -Jf_l^T @ eps[l]
         *
         * With identity Jacobian (Jf ≈ I):
         *   dVFE/dv[l] = (1/sigma_{l-1}^2)*eps[l-1] - (1/sigma_l^2)*eps[l]
         *
         * Per-element normalization to keep scale-independent. */
        float norm_factor = 1.0f / (float)(seq_len * D);

        /* v[0] is CLAMPED. Update v[1]..v[L-1]. */
        for (int32_t l = 1; l < L; l++) {
            float inv_sigma_l = 1.0f;
            float inv_sigma_lm1 = 1.0f;
            if (state->pe.precision[l] > 0.0f) {
                inv_sigma_l = 1.0f / state->pe.precision[l];
            }
            if (state->pe.precision[l - 1] > 0.0f) {
                inv_sigma_lm1 = 1.0f / state->pe.precision[l - 1];
            }

            /* grad_v = norm * ((1/sigma_{l-1}^2)*eps[l-1] - (1/sigma_l^2)*eps[l]) */
            op_scale(grad_v, state->pe.errors[l - 1],
                     norm_factor * inv_sigma_lm1);
            op_add_scaled(grad_v, grad_v, state->pe.errors[l],
                          -norm_factor * inv_sigma_l);

            /* Track max change */
            size_t n = tensor_numel(grad_v);
            const float *gd = (const float *)grad_v->data;
            for (size_t i = 0; i < n; i++) {
                float abs_d = fabsf(tcfg->ipc_lr * gd[i]);
                if (abs_d > max_vn_delta) {
                    max_vn_delta = abs_d;
                }
            }

            /* v[l] -= alpha_v * grad_v */
            op_add_scaled(state->vn.values[l], state->vn.values[l],
                          grad_v, -tcfg->ipc_lr);
        }

        /* Update v[L] (top value node):
         * dVFE/dv[L] = (1/sigma_{L-1}^2) * epsilon[L-1] + lambda_out * delta_top
         * (v[L] is only the target of eps[L-1], no eps[L] exists) */
        {
            float inv_sigma_Lm1 = 1.0f;
            if (state->pe.precision[L - 1] > 0.0f) {
                inv_sigma_Lm1 = 1.0f / state->pe.precision[L - 1];
            }

            /* grad_v_top = norm*(1/sigma_{L-1}^2)*eps[L-1] + lambda_out*delta_top */
            op_scale(grad_v, state->pe.errors[L - 1],
                     norm_factor * inv_sigma_Lm1);
            op_add_scaled(grad_v, grad_v, delta_top, tcfg->lambda_out);

            op_add_scaled(state->vn.values[L], state->vn.values[L],
                          grad_v, -tcfg->ipc_lr);
        }

        /* ---- Step D: Compute weight gradients from prediction errors ---- */
        for (int32_t l = 0; l < L; l++) {
            HSPABlock *block = model->layers[l];
            BlockGrad *bg = &grads->block_grads[l];
            float sigma_sq = state->pe.precision[l];
            float inv_sigma_sq = (sigma_sq > 0.0f) ? 1.0f / sigma_sq : 1.0f;

            /* Error signal: delta = -(1/(S*D)) * (1/sigma_l^2) * epsilon[l]
             * Per-element normalization prevents gradient explosion. */
            op_scale(delta, state->pe.errors[l], -inv_sigma_sq * norm_factor);

            /* ---- Attention weight gradients (approximate) ----
             * We use the cached pre_attn_norm as the input signal.
             * Approximate: dW_q, dW_k, dW_v ~ norm_in^T @ delta
             * dW_o: use norm_in^T @ delta (same approximation for first impl)
             */
            Tensor *pre_attn = state->layer_caches[l].pre_attn_norm;
            accum_matmul_tn(bg->attn_grad.dW_q, pre_attn, delta, seq_len, D,
                            bg->attn_grad.dW_q->shape[1]);
            accum_matmul_tn(bg->attn_grad.dW_k, pre_attn, delta, seq_len, D,
                            bg->attn_grad.dW_k->shape[1]);
            accum_matmul_tn(bg->attn_grad.dW_v, pre_attn, delta, seq_len, D,
                            bg->attn_grad.dW_v->shape[1]);

            /* dW_o: approximate with pre_attn^T @ delta projected */
            {
                int32_t o_rows = bg->attn_grad.dW_o->shape[0];
                int32_t o_cols = bg->attn_grad.dW_o->shape[1];
                /* W_o is [n_heads*head_dim, d_model] so dW_o is same shape.
                 * Approximate: accumulate pre_attn^T @ delta but trim/pad
                 * to match dW_o shape. Since dW_o is [H*Dh, D] and
                 * pre_attn^T @ delta is [D, D], we use delta directly for
                 * the first D rows (H*Dh == D in this config, since
                 * n_heads * head_dim should equal d_model). */
                if (o_rows == D && o_cols == D) {
                    accum_matmul_tn(bg->attn_grad.dW_o, pre_attn, delta,
                                   seq_len, D, D);
                } else {
                    /* Shapes don't match exactly -- use zero-padded approach.
                     * Accumulate into what fits. */
                    int32_t rows = o_rows < D ? o_rows : D;
                    const float *pa = (const float *)pre_attn->data;
                    const float *dl = (const float *)delta->data;
                    float *dw = (float *)bg->attn_grad.dW_o->data;
                    for (int32_t s = 0; s < seq_len; s++) {
                        const float *pa_s = pa + s * D;
                        const float *dl_s = dl + s * D;
                        for (int32_t i = 0; i < rows; i++) {
                            float ai = pa_s[i < D ? i : 0];
                            for (int32_t j = 0; j < o_cols; j++) {
                                dw[i * o_cols + j] += ai * dl_s[j < D ? j : 0];
                            }
                        }
                    }
                }
            }

            /* ---- FFN/MoE weight gradients -- Expert-Batched (D-032) ----
             *
             * Instead of per-token outer products (accum_outer), gather all
             * tokens assigned to each expert and compute weight gradients via
             * batched matmuls: dW = X^T @ dY (op_matmul_tn).
             * This replaces S*k individual outer products with K matmul calls.
             */
            Tensor *pre_ffn = state->layer_caches[l].pre_ffn_norm;
            RoutingState *rs = &state->routing[l];

            /* Skip FFN gradients if backward batch buffers failed to allocate */
            if (!bk_alloc_ok) goto bwd_skip_ffn;

            /* Build per-expert batch lists from routing state
             * (buffers pre-allocated before T loop) */
            BwdExpertBatch bwd_batches[16];
            for (int32_t j = 0; j < K && j < 16; j++) {
                bwd_batches[j].token_indices   = bk_indices + j * seq_len;
                bwd_batches[j].active_slots    = bk_slots + j * seq_len;
                bwd_batches[j].route_weights_bk = bk_weights + j * seq_len;
                bwd_batches[j].count = 0;
            }

            for (int32_t s = 0; s < seq_len; s++) {
                for (int32_t a = 0; a < k; a++) {
                    int32_t eid = rs->expert_ids[s * k + a];
                    if (eid < 0 || eid >= K) continue;
                    int32_t bi = bwd_batches[eid].count;
                    bwd_batches[eid].token_indices[bi] = s;
                    bwd_batches[eid].active_slots[bi] = a;
                    bwd_batches[eid].route_weights_bk[bi] = rs->expert_weights[s * k + a];
                    bwd_batches[eid].count++;
                }
            }

            /* Process each expert's batch */
            for (int32_t j = 0; j < K; j++) {
                int32_t N_j = bwd_batches[j].count;
                if (N_j == 0) continue;

                ExpertFFN *expert = block->experts[j];
                ExpertFFNGrad *eg = &bg->expert_grads[j];

                /* GATHER: collect norm_h, delta, gate_pre, up_pre for this expert's tokens */
                for (int32_t i = 0; i < N_j; i++) {
                    int32_t s = bwd_batches[j].token_indices[i];
                    int32_t a = bwd_batches[j].active_slots[i];
                    float w = bwd_batches[j].route_weights_bk[i];
                    int32_t ci = s * k + a;

                    /* norm_batch[i] = pre_ffn[s] */
                    memcpy(bk_norm_buf + (size_t)i * D,
                           (const float *)pre_ffn->data + (size_t)s * D,
                           (size_t)D * sizeof(float));

                    /* delta_batch[i] = w * delta[s] (route-weight scaled) */
                    const float *delta_s = (const float *)delta->data + (size_t)s * D;
                    float *dst_d = bk_delta_buf + (size_t)i * D;
                    for (int32_t d = 0; d < D; d++) {
                        dst_d[d] = w * delta_s[d];
                    }

                    /* gate_batch[i] = cached gate_pre for this (token, slot) */
                    memcpy(bk_gate_buf + (size_t)i * d_ff,
                           (const float *)state->layer_caches[l].expert_gate[ci]->data,
                           (size_t)d_ff * sizeof(float));

                    /* up_batch[i] = cached up_pre for this (token, slot) */
                    memcpy(bk_up_buf + (size_t)i * d_ff,
                           (const float *)state->layer_caches[l].expert_up[ci]->data,
                           (size_t)d_ff * sizeof(float));
                }

                /* Batched SwiGLU intermediates (element-wise, memory-bound) */
                for (int32_t i = 0; i < N_j; i++) {
                    float *ga = bk_gate_buf + (size_t)i * d_ff;
                    float *up = bk_up_buf + (size_t)i * d_ff;
                    float *gate_act = bk_gate_act_buf + (size_t)i * d_ff;
                    float *hid = bk_hidden_buf + (size_t)i * d_ff;
                    float *sg = bk_silu_grad_buf + (size_t)i * d_ff;

                    for (int32_t f = 0; f < d_ff; f++) {
                        float x = ga[f];
                        float sig = 1.0f / (1.0f + expf(-x));
                        gate_act[f] = x * sig;
                        hid[f] = gate_act[f] * up[f];
                        sg[f] = sig * (1.0f + x * (1.0f - sig));
                    }
                }

                /* ---- dW_down += hidden_batch^T @ delta_batch ----
                 * hidden_batch: [N_j, d_ff], delta_batch: [N_j, D]
                 * Result: [d_ff, D] */
                {
                    Tensor hidden_t;
                    hidden_t.data      = bk_hidden_buf;
                    hidden_t.shape[0]  = N_j;
                    hidden_t.shape[1]  = d_ff;
                    hidden_t.shape[2]  = 1;
                    hidden_t.shape[3]  = 1;
                    hidden_t.stride[0] = d_ff;
                    hidden_t.stride[1] = 1;
                    hidden_t.stride[2] = 1;
                    hidden_t.stride[3] = 1;
                    hidden_t.ndim      = 2;
                    hidden_t.dtype     = DTYPE_FP32;
                    hidden_t.pool      = NULL;
                    hidden_t.is_view   = true;

                    Tensor delta_t;
                    delta_t.data      = bk_delta_buf;
                    delta_t.shape[0]  = N_j;
                    delta_t.shape[1]  = D;
                    delta_t.shape[2]  = 1;
                    delta_t.shape[3]  = 1;
                    delta_t.stride[0] = D;
                    delta_t.stride[1] = 1;
                    delta_t.stride[2] = 1;
                    delta_t.stride[3] = 1;
                    delta_t.ndim      = 2;
                    delta_t.dtype     = DTYPE_FP32;
                    delta_t.pool      = NULL;
                    delta_t.is_view   = true;

                    Tensor dW_tmp_t;
                    dW_tmp_t.data      = bk_dW_tmp;
                    dW_tmp_t.shape[0]  = d_ff;
                    dW_tmp_t.shape[1]  = D;
                    dW_tmp_t.shape[2]  = 1;
                    dW_tmp_t.shape[3]  = 1;
                    dW_tmp_t.stride[0] = D;
                    dW_tmp_t.stride[1] = 1;
                    dW_tmp_t.stride[2] = 1;
                    dW_tmp_t.stride[3] = 1;
                    dW_tmp_t.ndim      = 2;
                    dW_tmp_t.dtype     = DTYPE_FP32;
                    dW_tmp_t.pool      = NULL;
                    dW_tmp_t.is_view   = true;

                    /* op_matmul_tn overwrites dW_tmp; accumulate into gradient */
                    op_matmul_tn(&dW_tmp_t, &hidden_t, &delta_t);
                    {
                        float *dw = (float *)eg->dW_down->data;
                        int32_t n = d_ff * D;
                        for (int32_t i = 0; i < n; i++) {
                            dw[i] += bk_dW_tmp[i];
                        }
                    }

                    /* ---- delta_hidden = delta_batch @ W_down^T ----
                     * delta_batch: [N_j, D], W_down: [d_ff, D]
                     * Result: [N_j, d_ff] */
                    Tensor dh_t;
                    dh_t.data      = bk_delta_hidden;
                    dh_t.shape[0]  = N_j;
                    dh_t.shape[1]  = d_ff;
                    dh_t.shape[2]  = 1;
                    dh_t.shape[3]  = 1;
                    dh_t.stride[0] = d_ff;
                    dh_t.stride[1] = 1;
                    dh_t.stride[2] = 1;
                    dh_t.stride[3] = 1;
                    dh_t.ndim      = 2;
                    dh_t.dtype     = DTYPE_FP32;
                    dh_t.pool      = NULL;
                    dh_t.is_view   = true;

                    /* op_matmul_nt_qat: out = a @ b_hat^T (or b^T if QAT disabled).
                     * a: [N_j, D], b: [d_ff, D], out: [N_j, d_ff].
                     * STE consistency: dx computation uses same w_hat as forward.
                     * Cache hit: W_down already registered in Phase 1 pre-registration. */
                    op_matmul_nt_qat(&dh_t, &delta_t, expert->W_down, qat);
                }

                /* Backprop through SwiGLU (element-wise, batched) */
                for (int32_t i = 0; i < N_j; i++) {
                    float *dh   = bk_delta_hidden + (size_t)i * d_ff;
                    float *up   = bk_up_buf + (size_t)i * d_ff;
                    float *ga   = bk_gate_act_buf + (size_t)i * d_ff;
                    float *sg   = bk_silu_grad_buf + (size_t)i * d_ff;
                    float *dgp  = bk_delta_gate_pre + (size_t)i * d_ff;
                    float *dup  = bk_delta_up_pre + (size_t)i * d_ff;

                    for (int32_t f = 0; f < d_ff; f++) {
                        float delta_gate_act = dh[f] * up[f];
                        dup[f] = dh[f] * ga[f];
                        dgp[f] = delta_gate_act * sg[f];
                    }
                }

                /* ---- dW_gate += norm_batch^T @ delta_gate_pre_batch ----
                 * norm_batch: [N_j, D], delta_gate_pre_batch: [N_j, d_ff]
                 * Result: [D, d_ff] */
                {
                    Tensor norm_t;
                    norm_t.data      = bk_norm_buf;
                    norm_t.shape[0]  = N_j;
                    norm_t.shape[1]  = D;
                    norm_t.shape[2]  = 1;
                    norm_t.shape[3]  = 1;
                    norm_t.stride[0] = D;
                    norm_t.stride[1] = 1;
                    norm_t.stride[2] = 1;
                    norm_t.stride[3] = 1;
                    norm_t.ndim      = 2;
                    norm_t.dtype     = DTYPE_FP32;
                    norm_t.pool      = NULL;
                    norm_t.is_view   = true;

                    Tensor dgp_t;
                    dgp_t.data      = bk_delta_gate_pre;
                    dgp_t.shape[0]  = N_j;
                    dgp_t.shape[1]  = d_ff;
                    dgp_t.shape[2]  = 1;
                    dgp_t.shape[3]  = 1;
                    dgp_t.stride[0] = d_ff;
                    dgp_t.stride[1] = 1;
                    dgp_t.stride[2] = 1;
                    dgp_t.stride[3] = 1;
                    dgp_t.ndim      = 2;
                    dgp_t.dtype     = DTYPE_FP32;
                    dgp_t.pool      = NULL;
                    dgp_t.is_view   = true;

                    Tensor dup_t;
                    dup_t.data      = bk_delta_up_pre;
                    dup_t.shape[0]  = N_j;
                    dup_t.shape[1]  = d_ff;
                    dup_t.shape[2]  = 1;
                    dup_t.shape[3]  = 1;
                    dup_t.stride[0] = d_ff;
                    dup_t.stride[1] = 1;
                    dup_t.stride[2] = 1;
                    dup_t.stride[3] = 1;
                    dup_t.ndim      = 2;
                    dup_t.dtype     = DTYPE_FP32;
                    dup_t.pool      = NULL;
                    dup_t.is_view   = true;

                    Tensor dW_tmp_t;
                    dW_tmp_t.data      = bk_dW_tmp;
                    dW_tmp_t.shape[0]  = D;
                    dW_tmp_t.shape[1]  = d_ff;
                    dW_tmp_t.shape[2]  = 1;
                    dW_tmp_t.shape[3]  = 1;
                    dW_tmp_t.stride[0] = d_ff;
                    dW_tmp_t.stride[1] = 1;
                    dW_tmp_t.stride[2] = 1;
                    dW_tmp_t.stride[3] = 1;
                    dW_tmp_t.ndim      = 2;
                    dW_tmp_t.dtype     = DTYPE_FP32;
                    dW_tmp_t.pool      = NULL;
                    dW_tmp_t.is_view   = true;

                    /* dW_gate += norm^T @ delta_gate_pre */
                    op_matmul_tn(&dW_tmp_t, &norm_t, &dgp_t);
                    {
                        float *dw = (float *)eg->dW_gate->data;
                        int32_t n = D * d_ff;
                        for (int32_t i = 0; i < n; i++) {
                            dw[i] += bk_dW_tmp[i];
                        }
                    }

                    /* dW_up += norm^T @ delta_up_pre */
                    op_matmul_tn(&dW_tmp_t, &norm_t, &dup_t);
                    {
                        float *dw = (float *)eg->dW_up->data;
                        int32_t n = D * d_ff;
                        for (int32_t i = 0; i < n; i++) {
                            dw[i] += bk_dW_tmp[i];
                        }
                    }
                }
            }

            bwd_skip_ffn: ;

            /* ---- Router prediction-error gradient ----
             *
             * The entropy gradient above only pushes toward uniform routing.
             * This block adds dL_pred/dW_mu: the gradient of prediction error
             * loss w.r.t. router weights, so that experts which reduce error
             * get reinforced and experts which increase error get suppressed.
             *
             * For each token s with active experts {a_0..a_{k-1}}:
             *   1. Reconstruct expert_out_a = hidden_a @ W_down (D-dim output)
             *   2. score_a = dot(expert_out_a, delta_s)
             *      -- alignment of expert output with error signal
             *   3. For selected expert a: dlogit_a += score_a * w_a
             *      For non-selected j:    dlogit_j -= score_a * w_a / (K - k)
             *   4. dW_mu[d, j] += norm_h_s[d] * dlogit_j
             *
             * Positive score_a means expert_a output aligns with the error
             * gradient (helps reduce loss) -> reinforce routing to expert a.
             */
            {
                float *expert_out_buf = scratch_expert_out;
                float *dlogit_buf = scratch_dlogit;

                if (expert_out_buf && dlogit_buf) {
                    Tensor *pre_ffn_pe = state->layer_caches[l].pre_ffn_norm;

                    /* Entropy-adaptive scaling: prevent expert collapse.
                     * Scale pred gradient by entropy-adaptive ratio.
                     * When floor is active (balance_h_floor > 0), the floor
                     * provides collapse protection, so use full pred gradient
                     * (ent_ratio = 1.0). Without floor, use quadratic decay
                     * (H/ln(K))^2 to prevent runaway specialization. */
                    float ent_ratio;
                    if (tcfg->balance_h_floor > 0.0f) {
                        ent_ratio = 1.0f;
                    } else {
                        float ln_K = logf((float)K);
                        float ent_linear = (ln_K > 0.0f)
                            ? fminf(fmaxf(rs->mean_entropy / ln_K, 0.0f), 1.0f)
                            : 1.0f;
                        ent_ratio = ent_linear * ent_linear;
                    }

                    for (int32_t s = 0; s < seq_len; s++) {
                        const float *delta_pe_s =
                            (const float *)delta->data + s * D;
                        const float *norm_pe_s =
                            (const float *)pre_ffn_pe->data + s * D;

                        /* Zero per-token logit gradient accumulator */
                        memset(dlogit_buf, 0, (size_t)K * sizeof(float));

                        for (int32_t a = 0; a < k; a++) {
                            int32_t eid = rs->expert_ids[s * k + a];
                            float w_a = rs->expert_weights[s * k + a];

                            if (eid < 0 || eid >= K) continue;

                            ExpertFFN *expert_pe = block->experts[eid];
                            int32_t ci_pe = s * k + a;

                            /* Retrieve cached gate and up projections */
                            const float *gate_c =
                                (const float *)state->layer_caches[l]
                                    .expert_gate[ci_pe]->data;
                            const float *up_c =
                                (const float *)state->layer_caches[l]
                                    .expert_up[ci_pe]->data;

                            /* Recompute hidden = silu(gate) * up */
                            /* Then expert_out = hidden @ W_down */
                            const float *wd_pe =
                                (const float *)expert_pe->W_down->data;
                            memset(expert_out_buf, 0, (size_t)D * sizeof(float));

                            for (int32_t f = 0; f < d_ff; f++) {
                                float g = gate_c[f];
                                float sig = 1.0f / (1.0f + expf(-g));
                                float h = (g * sig) * up_c[f];
                                /* Accumulate h * W_down[f, :] into expert_out */
                                for (int32_t d = 0; d < D; d++) {
                                    expert_out_buf[d] += h * wd_pe[f * D + d];
                                }
                            }

                            /* score_a = tanh(dot(expert_out, delta_s) / sqrt(D))
                             * 1) Normalize by sqrt(D): proper variance scaling
                             *    (dot product variance ~ D, so /sqrt(D) gives
                             *    unit-variance scores regardless of hidden dim).
                             * 2) tanh squash: bounds to [-1,1], prevents the
                             *    positive feedback loop where preferred experts
                             *    get unbounded gradient reinforcement. */
                            float score_a = 0.0f;
                            for (int32_t d = 0; d < D; d++) {
                                score_a += expert_out_buf[d] * delta_pe_s[d];
                            }
                            score_a = tanhf(score_a / sqrtf((float)D));

                            /* Distribute gradient to logits:
                             * Only reinforce the selected expert. With uniform
                             * routing, push+pull cancels to zero (conservation).
                             * By only pushing, we break symmetry: experts whose
                             * outputs align with the error get stronger logits.
                             * The entropy gradient provides the counterbalance
                             * against total expert collapse.
                             *
                             * D-031 note: In loss-free mode the entropy gradient
                             * is disabled; the router bias mechanism
                             * (router_update_bias) serves as the counterbalance
                             * instead, adjusting per-expert bias to redistribute
                             * load without gradient interference. */
                            /* Anneal pred_scale: full strength early (establish
                             * specialization), decay to 10% late (let LM dominate).
                             * Prevents pred_error explosion from overwhelming
                             * LM gradient at high specialization. */
                            float pred_frac = (tcfg->max_steps > 0)
                                ? fminf((float)step / (float)tcfg->max_steps, 1.0f)
                                : 0.0f;
                            float pred_decay = 0.1f + 0.9f * 0.5f
                                * (1.0f + cosf(3.14159265f * pred_frac));
                            float effective_pred_scale =
                                tcfg->router_pred_scale * pred_decay;
                            float contrib = effective_pred_scale * score_a * w_a
                                          * ent_ratio;
                            /* Negative gradient: SGD does w -= lr*grad, so
                             * negative dlogit → SGD increases the logit →
                             * routes MORE to expert a when score_a > 0. */
                            dlogit_buf[eid] -= contrib;
                        }

                        /* Accumulate into dW_mu: dW_mu[d, j] += nh[d] * dlogit[j] */
                        float *dw_mu_pe =
                            (float *)bg->router_grad.dW_mu->data;
                        for (int32_t d = 0; d < D; d++) {
                            for (int32_t j = 0; j < K; j++) {
                                dw_mu_pe[d * K + j] +=
                                    norm_pe_s[d] * dlogit_buf[j];
                            }
                        }
                    }
                }

                /* scratch buffers freed at function cleanup */
            }

            /* ---- Router entropy gradient ----
             * dW_mu += norm_h[s] @ (-beta_balance * dH/d_logit)^T
             * where dH/d_logit_j = p[j] * (H - log(p[j]))
             * (simplified: use softmax of mu logits, not the full FEP G)
             *
             * D-031: When use_loss_free_balance is true, skip this entire
             * section. Router bias (router_update_bias) handles load balancing
             * without gradient interference. */
            if (!tcfg->use_loss_free_balance)
            {
                /* Three-regime balance control:
                 * 1. H < floor:   beta = +beta_balance (push entropy UP, prevent collapse)
                 * 2. floor <= H <= target: beta = 0 (free zone)
                 * 3. H > target:  beta < 0 (push entropy DOWN, proportional to excess)
                 * When h_target=0 or h_floor=0, falls back to simpler modes. */
                float beta = tcfg->beta_balance;
                if (tcfg->balance_h_floor > 0.0f) {
                    if (rs->mean_entropy < tcfg->balance_h_floor) {
                        /* Below floor: full balance force (push UP) */
                        beta = tcfg->beta_balance;
                    } else if (tcfg->balance_h_target > 0.0f &&
                               rs->mean_entropy > tcfg->balance_h_target) {
                        /* Above target: proportional specialization (push DOWN).
                         * 10x amplification so the gradient can compete with
                         * the pred gradient equilibrium at H~1.7. */
                        float excess = rs->mean_entropy - tcfg->balance_h_target;
                        beta = -10.0f * tcfg->beta_balance * excess;
                    } else {
                        /* Free zone: no balance force */
                        beta = 0.0f;
                    }
                }
                float r_eps = 1e-8f;

                float *probs = scratch_probs;
                float *dH_dlogit = scratch_dH_dlogit;

                {
                    Tensor *pre_ffn_norm = state->layer_caches[l].pre_ffn_norm;

                    for (int32_t s = 0; s < seq_len; s++) {
                        const float *nh = (const float *)pre_ffn_norm->data + s * D;

                        /* Compute mu logits: logit[j] = nh^T @ W_mu[:,j] */
                        const float *wmu = (const float *)block->router->W_mu->data;
                        float max_logit = -INFINITY;
                        for (int32_t j = 0; j < K; j++) {
                            float dot = 0.0f;
                            for (int32_t d = 0; d < D; d++) {
                                dot += nh[d] * wmu[d * K + j];
                            }
                            probs[j] = dot;
                            if (dot > max_logit) max_logit = dot;
                        }

                        /* Softmax */
                        float sum_exp = 0.0f;
                        for (int32_t j = 0; j < K; j++) {
                            probs[j] = expf(probs[j] - max_logit);
                            sum_exp += probs[j];
                        }
                        for (int32_t j = 0; j < K; j++) {
                            probs[j] /= sum_exp;
                        }

                        /* Entropy H = -sum(p * log(p)) */
                        float H = 0.0f;
                        for (int32_t j = 0; j < K; j++) {
                            if (probs[j] > r_eps) {
                                H -= probs[j] * logf(probs[j] + r_eps);
                            }
                        }

                        /* dH/d_logit[j] = p[j] * (H + log(p[j])) ...
                         * Actually: d(-H)/d_logit_j = p_j * (log(p_j) + 1 + H)
                         * For the VFE balance loss L_balance = (logK - H):
                         * dL_balance/d_logit = -dH/d_logit
                         * = p_j * (log(p_j) + 1 + H) ... but this is the chain
                         * through softmax.
                         *
                         * Simpler: gradient of H w.r.t. pre-softmax logits:
                         * dH/dz_j = p_j * (H - log(p_j) - 1) + p_j
                         * Simplification for first impl: just use
                         * dL_balance/dz_j = -beta * (dH/dz_j)
                         * where dH/dz_j = p_j * (H + log(p_j) + 1) (wrong sign)
                         *
                         * Correct derivation:
                         * H = -sum_i p_i log(p_i)
                         * dH/dz_j = sum_i [-dp_i/dz_j * (log p_i + 1)]
                         * dp_i/dz_j = p_i (delta_{ij} - p_j)
                         * dH/dz_j = -sum_i p_i(delta_{ij}-p_j)(log p_i + 1)
                         *         = -p_j(log p_j + 1) + p_j * sum_i p_i(log p_i + 1)
                         *         = -p_j(log p_j + 1) + p_j * (sum_i p_i log p_i + 1)
                         *         = -p_j(log p_j + 1) + p_j * (-H + 1)
                         *         = p_j * (-log p_j - 1 - H + 1)
                         *         = p_j * (-log p_j - H)
                         *         = -p_j * (log p_j + H)
                         *
                         * We want to minimize L_balance = logK - H, so:
                         * dL_balance/dz_j = -dH/dz_j = p_j * (log p_j + H)
                         *
                         * Then: dW_mu += -beta * nh @ (dL_balance/dz)^T
                         * (negative because we want to increase entropy)
                         * Wait: SGD does w -= lr * grad, so if we accumulate
                         * the gradient as dL/dW, SGD will decrease the loss.
                         * grad_W_mu = beta * nh^T @ [dL_balance/dz]
                         */
                        for (int32_t j = 0; j < K; j++) {
                            float log_p = logf(probs[j] + r_eps);
                            dH_dlogit[j] = beta * probs[j] * (log_p + H);
                        }

                        /* dW_mu += nh @ dH_dlogit^T : [D, K] */
                        float *dw_mu = (float *)bg->router_grad.dW_mu->data;
                        for (int32_t d = 0; d < D; d++) {
                            for (int32_t j = 0; j < K; j++) {
                                dw_mu[d * K + j] += nh[d] * dH_dlogit[j];
                            }
                        }
                    }

                    /* scratch buffers freed at function cleanup */
                }
            }
        }
    }

    /* Free backward batch buffers (allocated before T loop) */
    free(bk_indices); free(bk_slots); free(bk_weights);
    free(bk_norm_buf); free(bk_delta_buf);
    free(bk_gate_buf); free(bk_up_buf);
    free(bk_gate_act_buf); free(bk_hidden_buf);
    free(bk_silu_grad_buf); free(bk_delta_hidden);
    free(bk_delta_gate_pre); free(bk_delta_up_pre);
    free(bk_dW_tmp);

    /* ==================================================================
     * Phase 3: Clip gradients, apply optimizer, compute loss
     *          (only on the LAST micro-batch of an accumulation group)
     * ================================================================== */

    bool is_last_micro = (micro_batch_idx == accum_steps - 1);

    if (is_last_micro) {
        /* Average gradients across micro-batches */
        if (accum_steps > 1) {
            float inv_accum = 1.0f / (float)accum_steps;
            /* Scale all gradients by 1/accum_steps to average them. */
            {
                /* Scale embedding grad */
                int32_t n_embed = (int32_t)tensor_numel(grads->embed_grad.dweight);
                float *ed = grads->embed_grad.dweight->data;
                for (int32_t i = 0; i < n_embed; i++) ed[i] *= inv_accum;

                /* Scale final norm grad */
                int32_t n_fn = (int32_t)tensor_numel(grads->final_norm_grad.dweight);
                float *fnd = grads->final_norm_grad.dweight->data;
                for (int32_t i = 0; i < n_fn; i++) fnd[i] *= inv_accum;

                for (int32_t l2 = 0; l2 < grads->n_layers; l2++) {
                    BlockGrad *bg2 = &grads->block_grads[l2];
                    int32_t n;
                    float *d;

#define SCALE_TENSOR(t) do { \
    n = (int32_t)tensor_numel((t)); \
    d = (t)->data; \
    for (int32_t i2 = 0; i2 < n; i2++) d[i2] *= inv_accum; \
} while (0)

                    SCALE_TENSOR(bg2->attn_norm_grad.dweight);
                    SCALE_TENSOR(bg2->attn_grad.dW_q);
                    SCALE_TENSOR(bg2->attn_grad.dW_k);
                    SCALE_TENSOR(bg2->attn_grad.dW_v);
                    SCALE_TENSOR(bg2->attn_grad.dW_o);
                    SCALE_TENSOR(bg2->ffn_norm_grad.dweight);
                    SCALE_TENSOR(bg2->router_grad.dW_mu);
                    SCALE_TENSOR(bg2->router_grad.dW_sigma);

                    for (int32_t e = 0; e < bg2->n_experts; e++) {
                        SCALE_TENSOR(bg2->expert_grads[e].dW_gate);
                        SCALE_TENSOR(bg2->expert_grads[e].dW_up);
                        SCALE_TENSOR(bg2->expert_grads[e].dW_down);
                    }
#undef SCALE_TENSOR
                }
            }
        }

        /* Record gradient norm before clipping */
        result.grad_norm = grad_global_norm(grads);

        /* Clip gradients */
        grad_clip(grads, tcfg->grad_clip_norm);

        /* Apply optimizer with Depth-muP scaling */
        if (adam && tcfg->use_adam) {
            grad_apply_adam(model, grads, adam, cfg, tcfg, step);
        } else {
            grad_apply_sgd(model, grads, cfg, tcfg, state->pe.precision, step);
        }
    } else {
        /* Not the last micro-batch: just record current grad norm */
        result.grad_norm = grad_global_norm(grads);
    }

    /* Re-compute final prediction errors for loss (after value node convergence).
     * Bottom-up: epsilon[l] = v[l+1] - f_l(v[l]).
     * Pass qat: all 224 weights are already in cache (same step); pure cache hits. */
    for (int32_t l = 0; l < L; l++) {
        model->kv_caches[l]->pos = 0;
        pool_reset(blk_scratch);
        tensor_fill(f_pred, 0.0f);
        hspa_block_forward(f_pred, model->layers[l], state->vn.values[l],
                           model->kv_caches[l], 0, blk_scratch, false,
                           cfg->n_active, qat);
        op_sub(state->pe.errors[l], state->vn.values[l + 1], f_pred);
    }

    /* Compute logits for loss */
    rms_norm_2d(normed, state->vn.values[L], model->final_norm,
                seq_len, D, work);
    op_matmul_nt(state->logits, normed, model->embed->weight);

    /* Compute loss */
    result.loss = loss_compute(state, targets, cfg, tcfg);

    /* Mean routing entropy across all layers */
    float total_entropy = 0.0f;
    for (int32_t l = 0; l < L; l++) {
        total_entropy += state->routing[l].mean_entropy;
    }
    result.mean_entropy = total_entropy / (float)L;

    /* Value node convergence metric */
    result.vn_delta = max_vn_delta;

    /* Free pre-allocated scratch buffers */
    free(scratch_expert_out);
    free(scratch_dlogit); free(scratch_probs); free(scratch_dH_dlogit);

    /* Free tensor struct metadata before destroying pools.
     * pool_destroy frees pool data but not the calloc'd Tensor structs. */
    tensor_destroy(norm_cache); tensor_destroy(tmp_sd);
    tensor_destroy(delta_top); tensor_destroy(grad_logits);
    tensor_destroy(logits); tensor_destroy(normed);
    tensor_destroy(grad_v); tensor_destroy(delta);
    tensor_destroy(f_pred); tensor_destroy(v_out);

    /* QAT observability seam: record covered count + cache hits before destroying
     * context.  Test harness reads via ipc_last_qat_covered_count() and
     * ipc_last_qat_cache_hits().  NULL-safe API returns 0 when qat == NULL. */
    s_last_ipc_qat_covered_count = qat_context_covered_count(qat);
    s_last_ipc_qat_cache_hits    = qat_context_cache_hits(qat);

    /* Cleanup */
    qat_context_destroy(qat);
    pool_destroy(blk_scratch);
    pool_destroy(work);

    return result;
}
