/* backprop_train.c -- Standard backpropagation training step for HSPA.
 *
 * Implements a complete forward + backward pass with exact gradients,
 * as opposed to the approximate local learning rules in ipc_train.c.
 *
 * Forward pass saves all activations needed for backward:
 *   - Pre-norm inputs, Q/K/V projections, attention weights, contexts
 *   - Per-expert gate/up projections, SiLU activations
 *   - Routing decisions and weights
 *
 * Backward pass uses exact chain rule:
 *   - Exact attention backward (softmax Jacobian, proper GQA accumulation)
 *   - Exact RMSNorm backward
 *   - Exact SwiGLU backward through MoE routing
 *   - Straight-through estimator for router top-k
 *
 * Phase 3 (gradient averaging, clipping, optimizer) is identical to iPC.
 */

#include "backprop_train.h"
#include "embedding.h"
#include "hspa_block.h"
#include "ops.h"
#include "qat_context.h"
#include "rmsnorm.h"
#include "router.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- QAT coverage observability seam (Program 3, Phase 7, D-613) ----
 * Written by backprop_train_step at the end of each step.
 * Read by backprop_last_qat_covered_count() for test observability.
 * Single-threaded by design (CLAUDE.md); no synchronization needed. */
static int s_last_qat_covered_count = 0;

int backprop_last_qat_covered_count(void) {
    return s_last_qat_covered_count;
}

/* ============================================================
 * Cycle 29 Rev-2: Entropy-penalty + τ-anneal math helpers
 * Spec: data/engineering/entropy_penalty_temp_anneal_design.md §Rev-2
 * ============================================================ */

float entropy_penalty_shannon_H(const float *p, int32_t K) {
    if (!p || K <= 0) return 0.0f;
    float eps = 1e-8f;
    float H = 0.0f;
    for (int32_t j = 0; j < K; j++) {
        if (p[j] > eps) {
            H -= p[j] * logf(p[j] + eps);
        }
    }
    return H;
}

void entropy_penalty_grad_logit(const float *p, int32_t K,
                                float beta_H, float H_target,
                                float *out_grad) {
    if (!out_grad || K <= 0) return;
    for (int32_t j = 0; j < K; j++) out_grad[j] = 0.0f;
    if (!p || beta_H <= 0.0f) return;

    float H = entropy_penalty_shannon_H(p, K);
    float gap = (H_target > H) ? (H_target - H) : 0.0f;
    if (gap <= 0.0f) return; /* hinge satisfied: zero gradient (Cycle-13 immunity) */

    float scale = 2.0f * beta_H * gap;
    float eps = 1e-8f;
    for (int32_t j = 0; j < K; j++) {
        float lp = logf(p[j] + eps);
        out_grad[j] = scale * p[j] * (lp + H);
    }
}

float compute_router_temperature(float tau_max, float tau_min,
                                 int32_t step, int32_t S_anneal) {
    if (S_anneal <= 0) return tau_min;
    if (step <= 0) return tau_max;
    if (step >= S_anneal) return tau_min;
    float t = (float)step / (float)S_anneal;        /* 0..1 */
    float cos_phase = 0.5f * (1.0f + cosf((float)M_PI * t)); /* 1 → 0 */
    return tau_min + (tau_max - tau_min) * cos_phase;
}

float compute_beta_h_warmup(float beta_h_peak, int32_t step,
                            int32_t warmup_steps) {
    if (warmup_steps <= 0) return beta_h_peak;
    if (step <= 0) return 0.0f;
    if (step >= warmup_steps) return beta_h_peak;
    return beta_h_peak * ((float)step / (float)warmup_steps);
}

/* ---- Helper: per-row RMSNorm for 2D tensors ---- */

static void bp_rms_norm_2d(Tensor *out, const Tensor *x,
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

/* ---- Helper: RMSNorm backward for 2D tensors ----
 *
 * Given forward: y = (x / rms) * w, where rms = sqrt(mean(x^2) + eps)
 *
 * Backward:
 *   dx[i] = (w[i] / rms) * (dy[i] - x_normed[i] * dot(dy * w, x_normed) / D)
 *   dw[i] += sum_over_rows(dy * x_normed)
 *
 * Where x_normed = x / rms.
 *
 * x: [seq_len, d_model] -- original input to norm
 * dy: [seq_len, d_model] -- upstream gradient
 * norm: RMSNorm layer (for weight and eps)
 * dx: [seq_len, d_model] -- output: gradient w.r.t. input (WRITTEN, not accumulated)
 * dw: [d_model] -- gradient w.r.t. weight (ACCUMULATED)
 */
static void bp_rmsnorm_backward_2d(float *dx, float *dw,
                                   const float *x, const float *dy,
                                   const RMSNorm *norm,
                                   int32_t seq_len, int32_t D) {
    const float *w = (const float *)norm->weight->data;
    float eps = norm->eps;

    for (int32_t s = 0; s < seq_len; s++) {
        const float *xs = x + (size_t)s * D;
        const float *dys = dy + (size_t)s * D;
        float *dxs = dx + (size_t)s * D;

        /* Compute RMS for this row */
        float sum_sq = 0.0f;
        for (int32_t d = 0; d < D; d++) {
            sum_sq += xs[d] * xs[d];
        }
        float rms = sqrtf(sum_sq / (float)D + eps);
        float inv_rms = 1.0f / rms;

        /* dot(dy * w, x_normed) = dot(dy * w, x / rms) = (1/rms) * dot(dy * w, x) */
        float dot_dyw_x = 0.0f;
        for (int32_t d = 0; d < D; d++) {
            float x_normed = xs[d] * inv_rms;
            dot_dyw_x += dys[d] * w[d] * x_normed;
        }

        /* dx[i] = (w[i] / rms) * (dy[i] - x_normed[i] * dot / D)
         * Simplified: dx[i] = (w[i] * inv_rms) * dy[i] - (w[i] * xs[i] * inv_rms^2 * dot / D)
         * Actually:   dx[i] = inv_rms * (w[i] * dy[i] - (xs[i] * inv_rms) * dot / D * w[i])
         * Wait -- let me be more careful:
         *
         * Let x_n = x / rms. Then y = x_n * w.
         * dy/dx_n = w * dy (element-wise)
         * dx_n/dx: for row normalization, dx_n[i]/dx[j] = (delta_ij - x_n[i]*x_n[j]) / rms
         * Wait, actually:  x_n = x / rms(x)
         *   dx_n[i]/dx[j] = (1/rms) * (delta_ij - x[i]*x[j] / (D * rms^2))
         *
         * So: dx = (1/rms) * (dL/dx_n - x_n * <dL/dx_n, x_n> / D)
         * where dL/dx_n = dy * w.
         */
        float coeff = dot_dyw_x / (float)D;
        for (int32_t d = 0; d < D; d++) {
            float x_n = xs[d] * inv_rms;
            float dl_dxn = dys[d] * w[d];
            dxs[d] = inv_rms * (dl_dxn - x_n * coeff);
        }

        /* dw[i] += dy[i] * x_normed[i] for this row */
        for (int32_t d = 0; d < D; d++) {
            dw[d] += dys[d] * xs[d] * inv_rms;
        }
    }
}

/* ---- Helper: accumulate matmul A^T @ B into dW ----
 * A: [S, M], B: [S, N], dW: [M, N]
 * dW += A^T @ B
 */
static void bp_accum_matmul_tn(float *dw, const float *a, const float *b,
                               int32_t S, int32_t M, int32_t N) {
    for (int32_t s = 0; s < S; s++) {
        const float *as = a + (size_t)s * M;
        const float *bs = b + (size_t)s * N;
        for (int32_t i = 0; i < M; i++) {
            float ai = as[i];
            for (int32_t j = 0; j < N; j++) {
                dw[(size_t)i * N + j] += ai * bs[j];
            }
        }
    }
}

/* ---- Helper: initialize a stack Tensor view over raw float buffer ---- */

static void bp_init_tensor_view(Tensor *t, float *data,
                                int32_t rows, int32_t cols) {
    t->data      = data;
    t->shape[0]  = rows;
    t->shape[1]  = cols;
    t->shape[2]  = 1;
    t->shape[3]  = 1;
    t->stride[0] = cols;
    t->stride[1] = 1;
    t->stride[2] = 1;
    t->stride[3] = 1;
    t->ndim      = 2;
    t->dtype     = DTYPE_FP32;
    t->pool      = NULL;
    t->is_view   = true;
}

/* ====================================================================
 * Backprop Training Step
 * ==================================================================== */

TrainStepResult backprop_train_step(HSPAModel *model, ModelGrad *grads,
                                    AdamState *adam,
                                    const int32_t *tokens, const int32_t *targets,
                                    int32_t seq_len,
                                    const HSPAConfig *cfg, const TrainConfig *tcfg,
                                    int32_t step, int32_t micro_batch_idx) {
    TrainStepResult result;
    memset(&result, 0, sizeof(result));

    if (!model || !grads || !tokens || !targets || !cfg || !tcfg) {
        return result;
    }

    /* ---- QAT context (Program 3, Phase 7, D-613) ----
     * Constructed once per step.  When use_qat=false the context is created
     * with enabled=false so qat_context_w_hat() returns W unchanged —
     * making all op_matmul_qat / op_matmul_nt_qat calls bit-identical to
     * the plain op_matmul / op_matmul_nt they replaced.
     * Capacity 256 > 224 in-scope weights (load < 1.0). */
    QATContext *qat = qat_context_create(tcfg->use_qat,
                                         /*group_size=*/tcfg->qat_group_size,
                                         /*capacity=*/256);
    /* NULL means allocation failed; we continue with NULL so that
     * op_matmul_qat / op_matmul_nt_qat degrade to plain matmul. */

    int32_t L = cfg->n_layers;
    int32_t D = cfg->d_model;
    int32_t V = cfg->vocab_size;
    int32_t K = cfg->n_experts;
    int32_t k_active = cfg->n_active;
    int32_t d_ff = cfg->d_ff;
    int32_t n_heads = cfg->n_heads;
    int32_t n_kv_heads = cfg->n_kv_heads;
    int32_t head_dim = cfg->head_dim;
    int32_t q_dim = n_heads * head_dim;
    int32_t kv_dim = n_kv_heads * head_dim;
    int32_t heads_per_group = n_heads / n_kv_heads;

    /* Phase 0: Zero gradients on first micro-batch */
    int32_t accum_steps = tcfg->grad_accum_steps > 0 ? tcfg->grad_accum_steps : 1;
    if (micro_batch_idx == 0) {
        grad_zero(grads);
    }
    srand((unsigned int)(step + 1 + micro_batch_idx));

    /* ================================================================
     * Allocate scratch memory for forward activations cache
     * ================================================================ */

    /* Size estimate for all cached activations:
     * Per layer: 7 * S*D + n_heads*S*S + S*k_active*d_ff*4
     * Plus: S*V logits, S*V grad_logits, S*D for residuals, etc.
     */
    /* ================================================================
     * Allocate forward activation caches (long-lived, span entire fn)
     * All caches are calloc'd individually and freed at cleanup.
     * ================================================================ */

    /* v[0..L]: layer boundary activations, [seq_len, D] each */
    float **v_cache = (float **)calloc((size_t)(L + 1), sizeof(float *));
    if (!v_cache) { return result; }

    for (int32_t l = 0; l <= L; l++) {
        v_cache[l] = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
        if (!v_cache[l]) { goto cleanup; }
    }

    /* Per-layer caches */
    float **norm_attn_cache = (float **)calloc((size_t)L, sizeof(float *));
    float **q_cache   = (float **)calloc((size_t)L, sizeof(float *));
    float **k_cache   = (float **)calloc((size_t)L, sizeof(float *));
    float **vv_cache  = (float **)calloc((size_t)L, sizeof(float *)); /* V projections */
    float **attn_w_cache = (float **)calloc((size_t)L, sizeof(float *)); /* [n_heads, S, S] */
    float **context_cache = (float **)calloc((size_t)L, sizeof(float *)); /* [S, q_dim] */
    float **h_cache   = (float **)calloc((size_t)L, sizeof(float *)); /* post-attn residual */
    float **norm_ffn_cache = (float **)calloc((size_t)L, sizeof(float *));

    /* Per-layer per-expert caches */
    float ***gate_pre_cache = (float ***)calloc((size_t)L, sizeof(float **));
    float ***up_pre_cache   = (float ***)calloc((size_t)L, sizeof(float **));
    float ***gate_act_cache = (float ***)calloc((size_t)L, sizeof(float **));
    float ***hidden_cache   = (float ***)calloc((size_t)L, sizeof(float **));

    /* Per-layer routing caches */
    int32_t **route_expert_ids = (int32_t **)calloc((size_t)L, sizeof(int32_t *));
    float   **route_weights    = (float **)calloc((size_t)L, sizeof(float *));
    float   *route_entropies   = (float *)calloc((size_t)L, sizeof(float));

    /* Per-layer per-expert: which tokens assigned */
    typedef struct {
        int32_t *token_indices;
        int32_t *active_slots;
        int32_t count;
    } ExpertBatch;
    ExpertBatch **expert_batches = (ExpertBatch **)calloc((size_t)L, sizeof(ExpertBatch *));

    if (!norm_attn_cache || !q_cache || !k_cache || !vv_cache ||
        !attn_w_cache || !context_cache || !h_cache || !norm_ffn_cache ||
        !gate_pre_cache || !up_pre_cache || !gate_act_cache || !hidden_cache ||
        !route_expert_ids || !route_weights || !route_entropies ||
        !expert_batches) {
        goto cleanup;
    }

    for (int32_t l = 0; l < L; l++) {
        norm_attn_cache[l] = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
        q_cache[l]   = (float *)calloc((size_t)seq_len * (size_t)q_dim, sizeof(float));
        k_cache[l]   = (float *)calloc((size_t)seq_len * (size_t)kv_dim, sizeof(float));
        vv_cache[l]  = (float *)calloc((size_t)seq_len * (size_t)kv_dim, sizeof(float));
        attn_w_cache[l] = (float *)calloc((size_t)n_heads * (size_t)seq_len * (size_t)seq_len,
                                          sizeof(float));
        context_cache[l] = (float *)calloc((size_t)seq_len * (size_t)q_dim, sizeof(float));
        h_cache[l]   = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
        norm_ffn_cache[l] = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));

        route_expert_ids[l] = (int32_t *)calloc((size_t)seq_len * (size_t)k_active,
                                                sizeof(int32_t));
        route_weights[l]    = (float *)calloc((size_t)seq_len * (size_t)k_active,
                                             sizeof(float));

        /* Per-expert caches: allocate K arrays of max seq_len entries */
        gate_pre_cache[l] = (float **)calloc((size_t)K, sizeof(float *));
        up_pre_cache[l]   = (float **)calloc((size_t)K, sizeof(float *));
        gate_act_cache[l] = (float **)calloc((size_t)K, sizeof(float *));
        hidden_cache[l]   = (float **)calloc((size_t)K, sizeof(float *));
        expert_batches[l] = (ExpertBatch *)calloc((size_t)K, sizeof(ExpertBatch));

        if (!norm_attn_cache[l] || !q_cache[l] || !k_cache[l] || !vv_cache[l] ||
            !attn_w_cache[l] || !context_cache[l] || !h_cache[l] ||
            !norm_ffn_cache[l] || !route_expert_ids[l] || !route_weights[l] ||
            !gate_pre_cache[l] || !up_pre_cache[l] || !gate_act_cache[l] ||
            !hidden_cache[l] || !expert_batches[l]) {
            goto cleanup;
        }

        for (int32_t j = 0; j < K; j++) {
            gate_pre_cache[l][j] = (float *)calloc((size_t)seq_len * (size_t)d_ff,
                                                   sizeof(float));
            up_pre_cache[l][j]   = (float *)calloc((size_t)seq_len * (size_t)d_ff,
                                                   sizeof(float));
            gate_act_cache[l][j] = (float *)calloc((size_t)seq_len * (size_t)d_ff,
                                                   sizeof(float));
            hidden_cache[l][j]   = (float *)calloc((size_t)seq_len * (size_t)d_ff,
                                                   sizeof(float));
            expert_batches[l][j].token_indices = (int32_t *)calloc((size_t)seq_len,
                                                                   sizeof(int32_t));
            expert_batches[l][j].active_slots  = (int32_t *)calloc((size_t)seq_len,
                                                                   sizeof(int32_t));
            expert_batches[l][j].count = 0;

            if (!gate_pre_cache[l][j] || !up_pre_cache[l][j] ||
                !gate_act_cache[l][j] || !hidden_cache[l][j] ||
                !expert_batches[l][j].token_indices ||
                !expert_batches[l][j].active_slots) {
                goto cleanup;
            }
        }
    }

    /* Scratch pool for temporary computations */
    size_t scratch_size = (size_t)seq_len * (size_t)D * sizeof(float) * 16
                        + (size_t)seq_len * (size_t)V * sizeof(float) * 2
                        + 4 * 1024 * 1024;
    MemoryPool *scratch = pool_create(scratch_size, POOL_SCRATCH);
    if (!scratch) { goto cleanup; }

    /* ================================================================
     * FORWARD PASS
     * ================================================================ */

    /* Invalidate QAT w_hat cache for this step (MUST precede first matmul). */
    qat_context_begin_step(qat);

    /* v[0] = embedding(tokens) */
    {
        int32_t sd_shape[] = {seq_len, D};
        Tensor *embed_out = tensor_create(scratch, sd_shape, 2, DTYPE_FP32);
        if (!embed_out) { pool_destroy(scratch); goto cleanup; }
        embedding_forward(embed_out, model->embed, tokens, seq_len);
        memcpy(v_cache[0], embed_out->data, (size_t)seq_len * (size_t)D * sizeof(float));
        tensor_destroy(embed_out);
        pool_reset(scratch);
    }

    /* Process each layer */
    for (int32_t l = 0; l < L; l++) {
        HSPABlock *block = model->layers[l];
        float *v_in = v_cache[l];

        /* ---- Pre-attention RMSNorm ---- */
        {
            int32_t sd_shape[] = {seq_len, D};
            Tensor v_in_t, norm_out_t;
            bp_init_tensor_view(&v_in_t, v_in, seq_len, D);
            bp_init_tensor_view(&norm_out_t, norm_attn_cache[l], seq_len, D);
            Tensor *tmp_out = tensor_create(scratch, sd_shape, 2, DTYPE_FP32);
            if (!tmp_out) { pool_destroy(scratch); goto cleanup; }
            bp_init_tensor_view(&v_in_t, v_in, seq_len, D);
            bp_rms_norm_2d(tmp_out, &v_in_t, block->attn_norm, seq_len, D, scratch);
            memcpy(norm_attn_cache[l], tmp_out->data,
                   (size_t)seq_len * (size_t)D * sizeof(float));
            tensor_destroy(tmp_out);
            pool_reset(scratch);
        }

        /* ---- Q, K, V projections ---- */
        {
            Tensor na_t, q_t, k_t, vv_t;
            bp_init_tensor_view(&na_t, norm_attn_cache[l], seq_len, D);
            bp_init_tensor_view(&q_t, q_cache[l], seq_len, q_dim);
            bp_init_tensor_view(&k_t, k_cache[l], seq_len, kv_dim);
            bp_init_tensor_view(&vv_t, vv_cache[l], seq_len, kv_dim);

            /* Q = norm_attn @ W_q : [S,D] @ [D, q_dim] = [S, q_dim] */
            op_matmul_qat(&q_t, &na_t, block->attn->W_q, qat);
            /* K = norm_attn @ W_k : [S,D] @ [D, kv_dim] = [S, kv_dim] */
            op_matmul_qat(&k_t, &na_t, block->attn->W_k, qat);
            /* V = norm_attn @ W_v : [S,D] @ [D, kv_dim] = [S, kv_dim] */
            op_matmul_qat(&vv_t, &na_t, block->attn->W_v, qat);
        }

        /* ---- Multi-head attention with causal mask ---- */
        {
            float scale = 1.0f / sqrtf((float)head_dim);

            for (int32_t h = 0; h < n_heads; h++) {
                int32_t kv_h = h / heads_per_group; /* GQA: which KV head */
                float *aw = attn_w_cache[l] + (size_t)h * (size_t)seq_len * (size_t)seq_len;
                /* context_cache layout: [S, q_dim], head h at offset h*head_dim */
                /* Note: context_cache layout is [S, n_heads*head_dim] so head h's
                 * data for position s is at offset s*q_dim + h*head_dim */

                /* Compute scores: Q_h @ K_kv^T / sqrt(head_dim), with causal mask */
                for (int32_t i = 0; i < seq_len; i++) {
                    float *q_row = q_cache[l] + (size_t)i * (size_t)q_dim + (size_t)h * (size_t)head_dim;

                    for (int32_t j = 0; j < seq_len; j++) {
                        if (j > i) {
                            aw[i * seq_len + j] = -INFINITY;
                            continue;
                        }
                        float *k_row = k_cache[l] + (size_t)j * (size_t)kv_dim + (size_t)kv_h * (size_t)head_dim;
                        float dot = 0.0f;
                        for (int32_t d = 0; d < head_dim; d++) {
                            dot += q_row[d] * k_row[d];
                        }
                        aw[i * seq_len + j] = dot * scale;
                    }

                    /* Softmax over row i (only j <= i are finite) */
                    float max_val = -INFINITY;
                    for (int32_t j = 0; j <= i; j++) {
                        if (aw[i * seq_len + j] > max_val)
                            max_val = aw[i * seq_len + j];
                    }
                    float sum_exp = 0.0f;
                    for (int32_t j = 0; j <= i; j++) {
                        float e = expf(aw[i * seq_len + j] - max_val);
                        aw[i * seq_len + j] = e;
                        sum_exp += e;
                    }
                    float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
                    for (int32_t j = 0; j <= i; j++) {
                        aw[i * seq_len + j] *= inv_sum;
                    }
                    /* j > i entries are set to 0 (they were -inf, exp(-inf) = 0) */
                    for (int32_t j = i + 1; j < seq_len; j++) {
                        aw[i * seq_len + j] = 0.0f;
                    }

                    /* context[i, h] = sum_j attn_w[i,j] * V_kv[j] */
                    float *ctx_row = context_cache[l] + (size_t)i * (size_t)q_dim
                                   + (size_t)h * (size_t)head_dim;
                    memset(ctx_row, 0, (size_t)head_dim * sizeof(float));
                    for (int32_t j = 0; j <= i; j++) {
                        float w = aw[i * seq_len + j];
                        float *v_row = vv_cache[l] + (size_t)j * (size_t)kv_dim
                                     + (size_t)kv_h * (size_t)head_dim;
                        for (int32_t d = 0; d < head_dim; d++) {
                            ctx_row[d] += w * v_row[d];
                        }
                    }
                }
            }
        }

        /* attn_out = context_flat @ W_o : [S, q_dim] @ [q_dim, D] = [S, D] */
        float *attn_out = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
        if (!attn_out) { pool_destroy(scratch); goto cleanup; }

        {
            Tensor ctx_t, ao_t;
            bp_init_tensor_view(&ctx_t, context_cache[l], seq_len, q_dim);
            bp_init_tensor_view(&ao_t, attn_out, seq_len, D);
            op_matmul_qat(&ao_t, &ctx_t, block->attn->W_o, qat);
        }

        /* h[l] = v[l] + attn_out (residual connection) */
        for (int32_t i = 0; i < seq_len * D; i++) {
            h_cache[l][i] = v_in[i] + attn_out[i];
        }
        free(attn_out);

        /* ---- Pre-FFN RMSNorm ---- */
        {
            int32_t sd_shape[] = {seq_len, D};
            Tensor h_t;
            bp_init_tensor_view(&h_t, h_cache[l], seq_len, D);
            Tensor *tmp_out = tensor_create(scratch, sd_shape, 2, DTYPE_FP32);
            if (!tmp_out) { pool_destroy(scratch); goto cleanup; }
            bp_rms_norm_2d(tmp_out, &h_t, block->ffn_norm, seq_len, D, scratch);
            memcpy(norm_ffn_cache[l], tmp_out->data,
                   (size_t)seq_len * (size_t)D * sizeof(float));
            tensor_destroy(tmp_out);
            pool_reset(scratch);
        }

        /* ---- Router: route each token to top-k experts ---- */
        {
            int32_t tok_shape[] = {D};
            Tensor *tok_1d = tensor_create(scratch, tok_shape, 1, DTYPE_FP32);
            if (!tok_1d) { pool_destroy(scratch); goto cleanup; }

            double entropy_sum = 0.0;
            for (int32_t s = 0; s < seq_len; s++) {
                for (int32_t d = 0; d < D; d++) {
                    int32_t idx[] = {d};
                    tensor_set(tok_1d, idx, norm_ffn_cache[l][(size_t)s * D + d]);
                }

                RoutingDecision rd = router_forward(block->router, tok_1d,
                                                    cfg, true);
                if (!rd.expert_ids || !rd.expert_weights) {
                    if (rd.expert_ids) free(rd.expert_ids);
                    if (rd.expert_weights) free(rd.expert_weights);
                    for (int32_t a = 0; a < k_active; a++) {
                        route_expert_ids[l][(size_t)s * k_active + a] = a % K;
                        route_weights[l][(size_t)s * k_active + a] = 1.0f / (float)k_active;
                    }
                } else {
                    for (int32_t a = 0; a < k_active; a++) {
                        route_expert_ids[l][(size_t)s * k_active + a] = rd.expert_ids[a];
                        route_weights[l][(size_t)s * k_active + a] = rd.expert_weights[a];
                    }
                    entropy_sum += (double)rd.entropy;
                    free(rd.expert_ids);
                    free(rd.expert_weights);
                }

                /* Record batch assignments */
                for (int32_t a = 0; a < k_active; a++) {
                    int32_t eid = route_expert_ids[l][(size_t)s * k_active + a];
                    if (eid < 0 || eid >= K) continue;
                    int32_t bi = expert_batches[l][eid].count;
                    expert_batches[l][eid].token_indices[bi] = s;
                    expert_batches[l][eid].active_slots[bi] = a;
                    expert_batches[l][eid].count++;
                }
            }
            route_entropies[l] = (float)(entropy_sum / (double)seq_len);
            tensor_destroy(tok_1d);
            pool_reset(scratch);
        }

        /* ---- Expert FFN forward (batched per expert) ---- */
        /* Accumulate FFN output into v[l+1] = h[l] + sum(w_a * expert_a(norm_ffn)) */
        memcpy(v_cache[l + 1], h_cache[l], (size_t)seq_len * (size_t)D * sizeof(float));

        /* QAT coverage pre-registration (§3.3 clause 1 guard, D-613).
         * Ensure all K experts' W_gate/W_up/W_down are registered in the QATContext
         * this step — even experts that receive no tokens (N_j == 0) and would
         * otherwise be skipped.  qat_context_w_hat is a no-op when disabled, and a
         * pure cache-warmup when enabled (the main loop below gets cache hits).
         * This ensures covered_count == L*K*3 + L*4 == 224 regardless of routing. */
        if (qat && qat_context_is_enabled(qat)) {
            for (int32_t j = 0; j < K; j++) {
                ExpertFFN *expert = block->experts[j];
                (void)qat_context_w_hat(qat, expert->W_gate);
                (void)qat_context_w_hat(qat, expert->W_up);
                (void)qat_context_w_hat(qat, expert->W_down);
            }
        }

        for (int32_t j = 0; j < K; j++) {
            int32_t N_j = expert_batches[l][j].count;
            if (N_j == 0) continue;

            ExpertFFN *expert = block->experts[j];

            /* Gather input rows */
            float *gather_buf = (float *)calloc((size_t)N_j * (size_t)D, sizeof(float));
            if (!gather_buf) { pool_destroy(scratch); goto cleanup; }

            for (int32_t i = 0; i < N_j; i++) {
                int32_t s = expert_batches[l][j].token_indices[i];
                memcpy(gather_buf + (size_t)i * D,
                       norm_ffn_cache[l] + (size_t)s * D,
                       (size_t)D * sizeof(float));
            }

            /* gate = gather @ W_gate : [N_j, D] @ [D, d_ff] = [N_j, d_ff] */
            Tensor g_t, gate_t, up_t;
            bp_init_tensor_view(&g_t, gather_buf, N_j, D);
            bp_init_tensor_view(&gate_t, gate_pre_cache[l][j], N_j, d_ff);
            bp_init_tensor_view(&up_t, up_pre_cache[l][j], N_j, d_ff);

            op_matmul_qat(&gate_t, &g_t, expert->W_gate, qat);
            op_matmul_qat(&up_t, &g_t, expert->W_up, qat);

            free(gather_buf);

            /* SiLU + elementwise multiply */
            for (int32_t i = 0; i < N_j; i++) {
                float *gp = gate_pre_cache[l][j] + (size_t)i * d_ff;
                float *up = up_pre_cache[l][j] + (size_t)i * d_ff;
                float *ga = gate_act_cache[l][j] + (size_t)i * d_ff;
                float *hid = hidden_cache[l][j] + (size_t)i * d_ff;

                for (int32_t f = 0; f < d_ff; f++) {
                    float sig = 1.0f / (1.0f + expf(-gp[f]));
                    ga[f] = gp[f] * sig; /* silu(gate) */
                    hid[f] = ga[f] * up[f]; /* silu(gate) * up */
                }
            }

            /* expert_out = hidden @ W_down : [N_j, d_ff] @ [d_ff, D] = [N_j, D] */
            float *expert_out = (float *)calloc((size_t)N_j * (size_t)D, sizeof(float));
            if (!expert_out) { pool_destroy(scratch); goto cleanup; }

            {
                Tensor hid_t, eo_t;
                bp_init_tensor_view(&hid_t, hidden_cache[l][j], N_j, d_ff);
                bp_init_tensor_view(&eo_t, expert_out, N_j, D);
                op_matmul_qat(&eo_t, &hid_t, expert->W_down, qat);
            }

            /* Scatter: v[l+1][s] += w_a * expert_out[i] */
            for (int32_t i = 0; i < N_j; i++) {
                int32_t s = expert_batches[l][j].token_indices[i];
                int32_t a = expert_batches[l][j].active_slots[i];
                float w = route_weights[l][(size_t)s * k_active + a];

                float *dst = v_cache[l + 1] + (size_t)s * D;
                float *src = expert_out + (size_t)i * D;
                for (int32_t d = 0; d < D; d++) {
                    dst[d] += w * src[d];
                }
            }
            free(expert_out);
        }
    }

    /* ---- Final RMSNorm + logits ---- */
    float *norm_final = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
    float *logits_buf = (float *)calloc((size_t)seq_len * (size_t)V, sizeof(float));
    float *grad_logits_buf = (float *)calloc((size_t)seq_len * (size_t)V, sizeof(float));

    if (!norm_final || !logits_buf || !grad_logits_buf) {
        free(norm_final); free(logits_buf); free(grad_logits_buf);
        pool_destroy(scratch);
        goto cleanup;
    }

    {
        int32_t sd_shape[] = {seq_len, D};
        Tensor v_top_t;
        bp_init_tensor_view(&v_top_t, v_cache[L], seq_len, D);
        Tensor *tmp_out = tensor_create(scratch, sd_shape, 2, DTYPE_FP32);
        if (!tmp_out) {
            free(norm_final); free(logits_buf); free(grad_logits_buf);
            pool_destroy(scratch); goto cleanup;
        }
        bp_rms_norm_2d(tmp_out, &v_top_t, model->final_norm, seq_len, D, scratch);
        memcpy(norm_final, tmp_out->data, (size_t)seq_len * (size_t)D * sizeof(float));
        tensor_destroy(tmp_out);
        pool_reset(scratch);
    }

    /* logits = norm_final @ W_embed^T : [S,D] @ [V,D]^T = [S,V] */
    {
        Tensor nf_t, lg_t;
        bp_init_tensor_view(&nf_t, norm_final, seq_len, D);
        bp_init_tensor_view(&lg_t, logits_buf, seq_len, V);
        op_matmul_nt(&lg_t, &nf_t, model->embed->weight);
    }

    /* Cross-entropy loss + gradient */
    float ce_loss;
    {
        Tensor lg_t, glg_t;
        bp_init_tensor_view(&lg_t, logits_buf, seq_len, V);
        bp_init_tensor_view(&glg_t, grad_logits_buf, seq_len, V);
        ce_loss = op_cross_entropy(&lg_t, targets, seq_len, &glg_t);
    }

    /* ================================================================
     * BACKWARD PASS
     * ================================================================ */

    /* ---- Output head backward ---- */
    /* dW_embed += norm_final^T @ grad_logits : [D,S] @ [S,V] = [D,V]
     * Wait, W_embed is [V,D], so dW_embed is [V,D].
     * logits = norm_final @ W_embed^T
     * dW_embed += grad_logits^T @ norm_final : [V,S] @ [S,D] = [V,D]
     */
    bp_accum_matmul_tn((float *)grads->embed_grad.dweight->data,
                       grad_logits_buf, norm_final, seq_len, V, D);

    /* dnorm_final = grad_logits @ W_embed : [S,V] @ [V,D] = [S,D] */
    float *dnorm_final = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
    if (!dnorm_final) {
        free(norm_final); free(logits_buf); free(grad_logits_buf);
        pool_destroy(scratch); goto cleanup;
    }
    {
        Tensor glg_t, dnf_t;
        bp_init_tensor_view(&glg_t, grad_logits_buf, seq_len, V);
        bp_init_tensor_view(&dnf_t, dnorm_final, seq_len, D);
        op_matmul(&dnf_t, &glg_t, model->embed->weight);
    }

    /* dv[L] = rmsnorm_backward(dnorm_final, v[L], final_norm) */
    float *dv_cur = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
    if (!dv_cur) {
        free(dnorm_final); free(norm_final); free(logits_buf); free(grad_logits_buf);
        pool_destroy(scratch); goto cleanup;
    }
    bp_rmsnorm_backward_2d(dv_cur,
                           (float *)grads->final_norm_grad.dweight->data,
                           v_cache[L], dnorm_final,
                           model->final_norm, seq_len, D);
    free(dnorm_final);
    free(logits_buf);
    free(grad_logits_buf);

    /* ---- Layer backward: L-1 down to 0 ---- */
    /* Working buffers for backward */
    float *dh = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
    float *dffn_out = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
    float *dnorm_ffn = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
    float *dnorm_attn = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
    float *dattn_out = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
    float *dcontext = (float *)calloc((size_t)seq_len * (size_t)q_dim, sizeof(float));
    float *dQ = (float *)calloc((size_t)seq_len * (size_t)q_dim, sizeof(float));
    float *dK = (float *)calloc((size_t)seq_len * (size_t)kv_dim, sizeof(float));
    float *dV = (float *)calloc((size_t)seq_len * (size_t)kv_dim, sizeof(float));
    float *dv_next = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));

    if (!dh || !dffn_out || !dnorm_ffn || !dnorm_attn || !dattn_out ||
        !dcontext || !dQ || !dK || !dV || !dv_next) {
        free(dh); free(dffn_out); free(dnorm_ffn); free(dnorm_attn);
        free(dattn_out); free(dcontext); free(dQ); free(dK); free(dV);
        free(dv_next); free(dv_cur); free(norm_final);
        pool_destroy(scratch); goto cleanup;
    }

    for (int32_t l = L - 1; l >= 0; l--) {
        HSPABlock *block = model->layers[l];
        BlockGrad *bg = &grads->block_grads[l];

        /* dv[l+1] comes from dv_cur (gradient flowing from above) */
        /* Residual split: v[l+1] = h[l] + ffn_out
         * So dh = dv[l+1] and dffn_out = dv[l+1] */
        memcpy(dh, dv_cur, (size_t)seq_len * (size_t)D * sizeof(float));
        memcpy(dffn_out, dv_cur, (size_t)seq_len * (size_t)D * sizeof(float));

        /* ---- FFN/MoE backward ---- */
        memset(dnorm_ffn, 0, (size_t)seq_len * (size_t)D * sizeof(float));

        for (int32_t j = 0; j < K; j++) {
            int32_t N_j = expert_batches[l][j].count;
            if (N_j == 0) continue;

            ExpertFFN *expert = block->experts[j];
            ExpertFFNGrad *eg = &bg->expert_grads[j];

            /* Allocate per-expert backward buffers */
            float *dexpert_out = (float *)calloc((size_t)N_j * (size_t)D, sizeof(float));
            float *dhidden = (float *)calloc((size_t)N_j * (size_t)d_ff, sizeof(float));
            float *dgate_pre = (float *)calloc((size_t)N_j * (size_t)d_ff, sizeof(float));
            float *dup_pre = (float *)calloc((size_t)N_j * (size_t)d_ff, sizeof(float));
            float *norm_gather = (float *)calloc((size_t)N_j * (size_t)D, sizeof(float));

            if (!dexpert_out || !dhidden || !dgate_pre || !dup_pre || !norm_gather) {
                free(dexpert_out); free(dhidden); free(dgate_pre);
                free(dup_pre); free(norm_gather);
                continue;
            }

            /* Gather dffn_out rows scaled by routing weights, and norm inputs */
            for (int32_t i = 0; i < N_j; i++) {
                int32_t s = expert_batches[l][j].token_indices[i];
                int32_t a = expert_batches[l][j].active_slots[i];
                float w = route_weights[l][(size_t)s * k_active + a];

                float *dst = dexpert_out + (size_t)i * D;
                float *src_d = dffn_out + (size_t)s * D;
                for (int32_t d = 0; d < D; d++) {
                    dst[d] = w * src_d[d];
                }

                memcpy(norm_gather + (size_t)i * D,
                       norm_ffn_cache[l] + (size_t)s * D,
                       (size_t)D * sizeof(float));
            }

            /* dhidden = dexpert_out @ W_down^T : [N_j, D] @ [d_ff, D]^T = [N_j, d_ff] */
            {
                Tensor deo_t, dh_t;
                bp_init_tensor_view(&deo_t, dexpert_out, N_j, D);
                bp_init_tensor_view(&dh_t, dhidden, N_j, d_ff);
                op_matmul_nt_qat(&dh_t, &deo_t, expert->W_down, qat);
            }

            /* dW_down += hidden^T @ dexpert_out : [d_ff, N_j] @ [N_j, D] = [d_ff, D] */
            bp_accum_matmul_tn((float *)eg->dW_down->data,
                               hidden_cache[l][j], dexpert_out,
                               N_j, d_ff, D);

            /* Backprop through SwiGLU */
            for (int32_t i = 0; i < N_j; i++) {
                float *dhi = dhidden + (size_t)i * d_ff;
                float *up = up_pre_cache[l][j] + (size_t)i * d_ff;
                float *ga = gate_act_cache[l][j] + (size_t)i * d_ff;
                float *gp = gate_pre_cache[l][j] + (size_t)i * d_ff;
                float *dgp = dgate_pre + (size_t)i * d_ff;
                float *dup = dup_pre + (size_t)i * d_ff;

                for (int32_t f = 0; f < d_ff; f++) {
                    /* d(silu(gate)*up)/d(up) = silu(gate) = gate_act */
                    dup[f] = dhi[f] * ga[f];
                    /* d(silu(gate)*up)/d(gate_pre) = up * silu'(gate_pre)
                     * silu'(x) = sigmoid(x) * (1 + x*(1-sigmoid(x))) */
                    float sig = 1.0f / (1.0f + expf(-gp[f]));
                    float silu_grad = sig * (1.0f + gp[f] * (1.0f - sig));
                    dgp[f] = dhi[f] * up[f] * silu_grad;
                }
            }

            /* dW_gate += norm_gather^T @ dgate_pre */
            bp_accum_matmul_tn((float *)eg->dW_gate->data,
                               norm_gather, dgate_pre,
                               N_j, D, d_ff);

            /* dW_up += norm_gather^T @ dup_pre */
            bp_accum_matmul_tn((float *)eg->dW_up->data,
                               norm_gather, dup_pre,
                               N_j, D, d_ff);

            /* dnorm_ffn += dgate_pre @ W_gate^T + dup_pre @ W_up^T */
            /* Scatter back to full sequence positions */
            {
                float *dnorm_expert = (float *)calloc((size_t)N_j * (size_t)D, sizeof(float));
                if (dnorm_expert) {
                    Tensor dgp_t, dup_t, dne_t;
                    bp_init_tensor_view(&dgp_t, dgate_pre, N_j, d_ff);
                    bp_init_tensor_view(&dup_t, dup_pre, N_j, d_ff);
                    bp_init_tensor_view(&dne_t, dnorm_expert, N_j, D);

                    /* dnorm_expert = dgate_pre @ W_gate^T */
                    op_matmul_nt_qat(&dne_t, &dgp_t, expert->W_gate, qat);

                    /* dnorm_expert += dup_pre @ W_up^T */
                    {
                        float *tmp_ne = (float *)calloc((size_t)N_j * (size_t)D, sizeof(float));
                        if (tmp_ne) {
                            Tensor tmp_t;
                            bp_init_tensor_view(&tmp_t, tmp_ne, N_j, D);
                            op_matmul_nt_qat(&tmp_t, &dup_t, expert->W_up, qat);
                            for (int32_t i = 0; i < N_j * D; i++) {
                                dnorm_expert[i] += tmp_ne[i];
                            }
                            free(tmp_ne);
                        }
                    }

                    /* Scatter dnorm_expert[i] back to dnorm_ffn[s].
                     * No w-scaling here: dexpert_out already has the routing
                     * weight w baked in (line 757), so the full chain
                     * dL/d(norm_ffn) = w * d(expert)/d(input) is already
                     * captured in dnorm_expert. Adding w again would give w^2. */
                    for (int32_t i = 0; i < N_j; i++) {
                        int32_t s = expert_batches[l][j].token_indices[i];
                        float *dst = dnorm_ffn + (size_t)s * D;
                        float *src_ne = dnorm_expert + (size_t)i * D;
                        for (int32_t d = 0; d < D; d++) {
                            dst[d] += src_ne[d];
                        }
                    }
                    free(dnorm_expert);
                }
            }

            free(dexpert_out);
            free(dhidden);
            free(dgate_pre);
            free(dup_pre);
            free(norm_gather);
        }

        /* ---- Router backward (straight-through estimator) ----
         *
         * Chain rule: v[l+1] = h + sum_a(w_a * expert_a(norm_ffn))
         * dL/dw_a = dot(dL/dv[l+1], expert_a(norm_ffn))
         *         = dot(dffn_out, expert_out_a)
         *
         * Then backprop through softmax of routing weights:
         * d_logit_a = w_a * (dL/dw_a - sum_b(w_b * dL/dw_b))
         *
         * Finally: dW_mu += norm_ffn^T @ d_logits (per selected expert)
         *
         * dW_sigma is intentionally frozen in backprop mode — sigma
         * controls uncertainty in the FEP router and gets its gradient
         * from KL divergence in iPC. Without a KL term in the backprop
         * loss, sigma has no meaningful gradient signal.
         *
         * D-Plan-B (arXiv 2504.12463): when tcfg->use_default_moe is true
         * AND the router has a default_moe_ema tensor, take a DENSE path
         * that substitutes the EMA of expert outputs for unselected
         * experts. Result: every expert receives router gradient, not
         * just the top-k selected ones.
         */
        if (tcfg->use_default_moe && block->router->default_moe_ema) {
            float *dw_mu   = (float *)bg->router_grad.dW_mu->data;
            float *ema_all = block->router->default_moe_ema;   /* [K * D] */
            bool is_last_micro_local =
                (micro_batch_idx == (tcfg->grad_accum_steps > 0
                                         ? tcfg->grad_accum_steps - 1
                                         : 0));
            float ema_alpha = tcfg->default_moe_alpha > 0.0f
                                  ? tcfg->default_moe_alpha
                                  : block->router->default_moe_alpha;

            /* Scratch buffers (K*D for expert_out_per_expert_mean, K*D
             * pointer table is avoided; we store only the mean needed
             * by the EMA update, and a per-token K d_w table). */
            float *expert_mean_out =
                (float *)calloc((size_t)K * (size_t)D, sizeof(float));
            int32_t *expert_N = (int32_t *)calloc((size_t)K, sizeof(int32_t));
            float *d_w_full =
                (float *)calloc((size_t)seq_len * (size_t)K, sizeof(float));
            float *w_full =
                (float *)calloc((size_t)seq_len * (size_t)K, sizeof(float));
            if (!expert_mean_out || !expert_N || !d_w_full || !w_full) {
                free(expert_mean_out);
                free(expert_N);
                free(d_w_full);
                free(w_full);
                /* Fall through: dense path aborted on OOM; leaves W_mu
                 * grad unchanged for this layer. */
            } else {
                /* Step 1a: For each selected expert j, recompute
                 * expert_out_j = hidden @ W_down and use it both for
                 * d_w_full[s, j] (selected tokens) and for the EMA update. */
                for (int32_t j = 0; j < K; j++) {
                    int32_t N_j = expert_batches[l][j].count;
                    if (N_j == 0) continue;
                    ExpertFFN *expert = block->experts[j];

                    float *expert_out_j = (float *)calloc(
                        (size_t)N_j * (size_t)D, sizeof(float));
                    if (!expert_out_j) continue;

                    {
                        Tensor h_t, eo_t;
                        bp_init_tensor_view(&h_t, hidden_cache[l][j], N_j, d_ff);
                        bp_init_tensor_view(&eo_t, expert_out_j, N_j, D);
                        /* Router recompute (dense/default_moe branch): use same
                         * cached w_hat as the forward pass (STE consistency). */
                        op_matmul_qat(&eo_t, &h_t, expert->W_down, qat);
                    }

                    /* d_w_full[s, j] = dot(dffn_out[s], expert_out_j[i])
                     * for tokens s that selected expert j. */
                    for (int32_t i = 0; i < N_j; i++) {
                        int32_t s = expert_batches[l][j].token_indices[i];
                        float *df = dffn_out + (size_t)s * D;
                        float *eo = expert_out_j + (size_t)i * D;
                        float dw_val = 0.0f;
                        for (int32_t d = 0; d < D; d++) {
                            dw_val += df[d] * eo[d];
                        }
                        d_w_full[(size_t)s * K + j] = dw_val;
                    }

                    /* Accumulate mean for EMA update (sum now, divide
                     * later when we actually update the EMA). */
                    if (is_last_micro_local && ema_alpha > 0.0f) {
                        float *dst = expert_mean_out + (size_t)j * D;
                        for (int32_t i = 0; i < N_j; i++) {
                            float *src = expert_out_j + (size_t)i * D;
                            for (int32_t d = 0; d < D; d++) {
                                dst[d] += src[d];
                            }
                        }
                        expert_N[j] = N_j;
                    }
                    free(expert_out_j);
                }

                /* Step 1b: For unselected (s, j) pairs, fill d_w_full[s, j]
                 * with dot(dffn_out[s], EMA[j]). Selected slots already set. */
                for (int32_t s = 0; s < seq_len; s++) {
                    /* Build a K-bit mask of "was j selected for token s?" */
                    bool sel[64];  /* hard limit 64 experts */
                    if (K > 64) {
                        /* For K>64 we'd need a bitmap; AGI config never exceeds 64. */
                        goto dense_backward_cleanup;
                    }
                    for (int32_t j = 0; j < K; j++) sel[j] = false;
                    for (int32_t a = 0; a < k_active; a++) {
                        int32_t eid = route_expert_ids[l][(size_t)s * k_active + a];
                        if (eid >= 0 && eid < K) sel[eid] = true;
                    }
                    float *df = dffn_out + (size_t)s * D;
                    for (int32_t j = 0; j < K; j++) {
                        if (sel[j]) continue;
                        float *ema_j = ema_all + (size_t)j * D;
                        float dw_val = 0.0f;
                        for (int32_t d = 0; d < D; d++) {
                            dw_val += df[d] * ema_j[d];
                        }
                        d_w_full[(size_t)s * K + j] = dw_val;
                    }
                }

                /* Step 2: Per-token full-K softmax over (W_mu @ norm_ffn
                 * + expert_bias). This differs from the router's internal
                 * free-energy path -- the dense backward follows the
                 * paper's §2.5 formulation (plain softmax over logits,
                 * with the LFB bias shift already applied).
                 *
                 * Cycle 29 Rev-2: apply router temperature τ as logit
                 * divisor AFTER bias, BEFORE softmax. Chain rule scales
                 * d_logit by 1/τ to preserve gradient magnitude.
                 * Entropy penalty §1.3 adds a per-token hinge term when
                 * H[t] < H_target (zero above; Cycle-13 immune). */
                float *W_mu_data = (float *)block->router->W_mu->data;
                float *bias = block->router->expert_bias;  /* may be NULL pre-LFB */
                float tau_now = (tcfg->use_temp_anneal
                                 && block->router->router_temperature > 0.0f)
                                ? block->router->router_temperature : 1.0f;
                float inv_tau = 1.0f / tau_now;
                /* β_H warmup: linear ramp over entropy_beta_h_warmup_steps, then
                 * gated-off before entropy_warmup_steps. */
                float beta_h_effective = 0.0f;
                if (tcfg->use_entropy_penalty
                    && step >= tcfg->entropy_warmup_steps) {
                    beta_h_effective = compute_beta_h_warmup(
                        tcfg->entropy_beta_h, step,
                        tcfg->entropy_beta_h_warmup_steps);
                }
                float H_target = tcfg->entropy_h_target;
                for (int32_t s = 0; s < seq_len; s++) {
                    float *nh = norm_ffn_cache[l] + (size_t)s * D;
                    float *wf = w_full + (size_t)s * K;

                    /* logits[j] = (nh @ W_mu[:, j] + bias[j]) / τ */
                    float max_logit = -INFINITY;
                    for (int32_t j = 0; j < K; j++) {
                        float dot = 0.0f;
                        for (int32_t d = 0; d < D; d++) {
                            dot += nh[d] * W_mu_data[d * K + j];
                        }
                        if (bias) dot += bias[j];
                        dot *= inv_tau;
                        wf[j] = dot;
                        if (dot > max_logit) max_logit = dot;
                    }
                    float sum_exp = 0.0f;
                    for (int32_t j = 0; j < K; j++) {
                        wf[j] = expf(wf[j] - max_logit);
                        sum_exp += wf[j];
                    }
                    if (sum_exp <= 0.0f) sum_exp = 1.0f;
                    for (int32_t j = 0; j < K; j++) wf[j] /= sum_exp;

                    /* Entropy-penalty: compute H_t + gap + penalty_scale. */
                    float penalty_scale = 0.0f;
                    float H_t = 0.0f;
                    if (beta_h_effective > 0.0f) {
                        H_t = entropy_penalty_shannon_H(wf, K);
                        float gap = (H_target > H_t) ? (H_target - H_t) : 0.0f;
                        if (gap > 0.0f) {
                            penalty_scale = 2.0f * beta_h_effective * gap;
                        }
                    }

                    /* Softmax backward over all K:
                     *   w_sum_dw = sum_j w_full[j] * d_w_full[j]
                     *   d_logit[j] = w_full[j] * (d_w_full[j] - w_sum_dw)
                     *              + penalty_scale * w_full[j] * (log w_full[j] + H_t)
                     *   d_logit[j] *= 1/τ (chain rule for logits_eff = logits_raw/τ)
                     *   dW_mu[:, j] += nh[d] * d_logit[j] */
                    float w_sum_dw = 0.0f;
                    for (int32_t j = 0; j < K; j++) {
                        w_sum_dw += wf[j] * d_w_full[(size_t)s * K + j];
                    }
                    for (int32_t j = 0; j < K; j++) {
                        float d_logit = wf[j]
                            * (d_w_full[(size_t)s * K + j] - w_sum_dw);
                        if (penalty_scale > 0.0f) {
                            d_logit += penalty_scale * wf[j]
                                       * (logf(wf[j] + 1e-8f) + H_t);
                        }
                        d_logit *= inv_tau;  /* τ-anneal chain rule */
                        for (int32_t d = 0; d < D; d++) {
                            dw_mu[d * K + j] += nh[d] * d_logit;
                        }
                    }
                }

                /* Step 3: EMA update -- once per optimizer step, across
                 * all tokens routed to each expert in the micro-batch.
                 *   E[l][j] <- (1 - α) * E[l][j] + α * mean(expert_out_j)
                 * where mean is over the N_j tokens routed to expert j.
                 * Skipped when N_j == 0 (paper §2.4 conservative variant). */
                if (is_last_micro_local && ema_alpha > 0.0f) {
                    for (int32_t j = 0; j < K; j++) {
                        if (expert_N[j] == 0) continue;
                        float inv_N = 1.0f / (float)expert_N[j];
                        float *E_j   = ema_all + (size_t)j * D;
                        float *mean_j = expert_mean_out + (size_t)j * D;
                        for (int32_t d = 0; d < D; d++) {
                            float h_bar = mean_j[d] * inv_N;
                            E_j[d] = (1.0f - ema_alpha) * E_j[d]
                                   + ema_alpha * h_bar;
                        }
                    }
                }
dense_backward_cleanup:
                free(expert_mean_out);
                free(expert_N);
                free(d_w_full);
                free(w_full);
            }
        } else {
            float *dw_mu = (float *)bg->router_grad.dW_mu->data;

            /* Step 1: Compute expert_out per expert batch (hidden @ W_down)
             * and store d_w per token per active slot */
            float *d_w_per_token = (float *)calloc(
                (size_t)seq_len * (size_t)k_active, sizeof(float));
            if (d_w_per_token) {
                for (int32_t j = 0; j < K; j++) {
                    int32_t N_j = expert_batches[l][j].count;
                    if (N_j == 0) continue;
                    ExpertFFN *expert = block->experts[j];

                    /* expert_out_j = hidden_cache[l][j] @ W_down : [N_j,d_ff]@[d_ff,D]=[N_j,D] */
                    float *expert_out_j = (float *)calloc(
                        (size_t)N_j * (size_t)D, sizeof(float));
                    if (!expert_out_j) continue;

                    {
                        Tensor h_t, eo_t;
                        bp_init_tensor_view(&h_t, hidden_cache[l][j], N_j, d_ff);
                        bp_init_tensor_view(&eo_t, expert_out_j, N_j, D);
                        /* Router recompute (sparse/STE branch): use same cached
                         * w_hat as the forward pass (STE consistency). */
                        op_matmul_qat(&eo_t, &h_t, expert->W_down, qat);
                    }

                    /* d_w[s,a] = dot(dffn_out[s], expert_out[i]) */
                    for (int32_t i = 0; i < N_j; i++) {
                        int32_t s = expert_batches[l][j].token_indices[i];
                        int32_t a = expert_batches[l][j].active_slots[i];
                        float *df = dffn_out + (size_t)s * D;
                        float *eo = expert_out_j + (size_t)i * D;
                        float dw_val = 0.0f;
                        for (int32_t d = 0; d < D; d++) {
                            dw_val += df[d] * eo[d];
                        }
                        d_w_per_token[(size_t)s * k_active + a] = dw_val;
                    }
                    free(expert_out_j);
                }

                /* Step 2: Softmax backward for routing weights per token,
                 * then accumulate dW_mu */
                for (int32_t s = 0; s < seq_len; s++) {
                    float *nh = norm_ffn_cache[l] + (size_t)s * D;

                    /* Compute weighted sum: sum_b(w_b * d_w_b) */
                    float w_sum_dw = 0.0f;
                    for (int32_t a = 0; a < k_active; a++) {
                        float w = route_weights[l][(size_t)s * k_active + a];
                        float dw_val = d_w_per_token[(size_t)s * k_active + a];
                        w_sum_dw += w * dw_val;
                    }

                    /* Softmax backward: d_logit_a = w_a * (d_w_a - sum(w*d_w)) */
                    for (int32_t a = 0; a < k_active; a++) {
                        int32_t eid = route_expert_ids[l][(size_t)s * k_active + a];
                        if (eid < 0 || eid >= K) continue;
                        float w = route_weights[l][(size_t)s * k_active + a];
                        float dw_val = d_w_per_token[(size_t)s * k_active + a];
                        float d_logit = w * (dw_val - w_sum_dw);

                        /* dW_mu[:, eid] += norm_ffn[s, :] * d_logit */
                        for (int32_t d = 0; d < D; d++) {
                            dw_mu[d * K + eid] += nh[d] * d_logit;
                        }
                    }
                }
                free(d_w_per_token);
            }
        }

        /* ---- FFN norm backward ---- */
        {
            float *dh_from_ffn_norm = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
            if (dh_from_ffn_norm) {
                bp_rmsnorm_backward_2d(dh_from_ffn_norm,
                                       (float *)bg->ffn_norm_grad.dweight->data,
                                       h_cache[l], dnorm_ffn,
                                       block->ffn_norm, seq_len, D);
                /* Accumulate into dh */
                for (int32_t i = 0; i < seq_len * D; i++) {
                    dh[i] += dh_from_ffn_norm[i];
                }
                free(dh_from_ffn_norm);
            }
        }

        /* ---- Attention backward (EXACT) ---- */
        /* dattn_out = dh (gradient from residual) */
        memcpy(dattn_out, dh, (size_t)seq_len * (size_t)D * sizeof(float));

        /* dcontext = dattn_out @ W_o^T : [S,D] @ [q_dim,D]^T = [S, q_dim] */
        memset(dcontext, 0, (size_t)seq_len * (size_t)q_dim * sizeof(float));
        {
            Tensor dao_t, dc_t;
            bp_init_tensor_view(&dao_t, dattn_out, seq_len, D);
            bp_init_tensor_view(&dc_t, dcontext, seq_len, q_dim);
            op_matmul_nt_qat(&dc_t, &dao_t, block->attn->W_o, qat);
        }

        /* dW_o += context^T @ dattn_out : [q_dim, S] @ [S, D] = [q_dim, D] */
        bp_accum_matmul_tn((float *)bg->attn_grad.dW_o->data,
                           context_cache[l], dattn_out,
                           seq_len, q_dim, D);

        /* Per-head attention backward */
        memset(dQ, 0, (size_t)seq_len * (size_t)q_dim * sizeof(float));
        memset(dK, 0, (size_t)seq_len * (size_t)kv_dim * sizeof(float));
        memset(dV, 0, (size_t)seq_len * (size_t)kv_dim * sizeof(float));

        {
            float scale = 1.0f / sqrtf((float)head_dim);
            /* Pre-allocate scratch for dattn_w_row (M3 fix: avoid per-position malloc) */
            float *dattn_w_row = (float *)calloc((size_t)seq_len, sizeof(float));
            if (!dattn_w_row) goto skip_attn_bwd;

            for (int32_t h = 0; h < n_heads; h++) {
                int32_t kv_h = h / heads_per_group;
                float *aw = attn_w_cache[l] + (size_t)h * (size_t)seq_len * (size_t)seq_len;

                for (int32_t i = 0; i < seq_len; i++) {
                    float *dc_row = dcontext + (size_t)i * q_dim + (size_t)h * head_dim;

                    /* Zero the reusable scratch buffer */
                    memset(dattn_w_row, 0, (size_t)seq_len * sizeof(float));

                    /* dattn_weights[i,j] = dcontext[i] @ V[j]^T (for this head) */
                    /* dV[j] += attn_weights[i,j] * dcontext[i] */
                    for (int32_t j = 0; j <= i; j++) {
                        float *v_row = vv_cache[l] + (size_t)j * kv_dim
                                     + (size_t)kv_h * head_dim;
                        float dot = 0.0f;
                        for (int32_t d = 0; d < head_dim; d++) {
                            dot += dc_row[d] * v_row[d];
                        }
                        dattn_w_row[j] = dot;

                        /* dV accumulation */
                        float aw_ij = aw[i * seq_len + j];
                        float *dv_row = dV + (size_t)j * kv_dim + (size_t)kv_h * head_dim;
                        for (int32_t d = 0; d < head_dim; d++) {
                            dv_row[d] += aw_ij * dc_row[d];
                        }
                    }

                    /* Softmax backward:
                     * dscores[i,j] = attn_w[i,j] * (dattn_w[i,j] - sum_m(dattn_w[i,m] * attn_w[i,m]))
                     */
                    float dot_sum = 0.0f;
                    for (int32_t j = 0; j <= i; j++) {
                        dot_sum += dattn_w_row[j] * aw[i * seq_len + j];
                    }

                    for (int32_t j = 0; j <= i; j++) {
                        float ds = aw[i * seq_len + j] * (dattn_w_row[j] - dot_sum);
                        ds *= scale; /* divide by sqrt(head_dim) */

                        /* dQ[i] += ds * K[j] */
                        float *q_grad = dQ + (size_t)i * q_dim + (size_t)h * head_dim;
                        float *k_row = k_cache[l] + (size_t)j * kv_dim
                                     + (size_t)kv_h * head_dim;
                        for (int32_t d = 0; d < head_dim; d++) {
                            q_grad[d] += ds * k_row[d];
                        }

                        /* dK[j] += ds * Q[i] */
                        float *k_grad = dK + (size_t)j * kv_dim + (size_t)kv_h * head_dim;
                        float *q_row = q_cache[l] + (size_t)i * q_dim
                                     + (size_t)h * head_dim;
                        for (int32_t d = 0; d < head_dim; d++) {
                            k_grad[d] += ds * q_row[d];
                        }
                    }
                }
            }
            free(dattn_w_row);
            skip_attn_bwd:;
        }

        /* Project back: dnorm_attn = dQ @ W_q^T + dK @ W_k^T + dV @ W_v^T */
        memset(dnorm_attn, 0, (size_t)seq_len * (size_t)D * sizeof(float));
        {
            float *tmp_na = (float *)calloc((size_t)seq_len * (size_t)D, sizeof(float));
            if (tmp_na) {
                Tensor dq_t, dk_t, dvv_t, dna_t, tmp_t;
                bp_init_tensor_view(&dq_t, dQ, seq_len, q_dim);
                bp_init_tensor_view(&dna_t, dnorm_attn, seq_len, D);
                bp_init_tensor_view(&tmp_t, tmp_na, seq_len, D);

                /* dnorm_attn = dQ @ W_q^T : [S, q_dim] @ [D, q_dim]^T = [S, D] */
                op_matmul_nt_qat(&dna_t, &dq_t, block->attn->W_q, qat);

                /* dnorm_attn += dK @ W_k^T */
                bp_init_tensor_view(&dk_t, dK, seq_len, kv_dim);
                op_matmul_nt_qat(&tmp_t, &dk_t, block->attn->W_k, qat);
                for (int32_t i = 0; i < seq_len * D; i++) {
                    dnorm_attn[i] += tmp_na[i];
                }

                /* dnorm_attn += dV @ W_v^T */
                bp_init_tensor_view(&dvv_t, dV, seq_len, kv_dim);
                op_matmul_nt_qat(&tmp_t, &dvv_t, block->attn->W_v, qat);
                for (int32_t i = 0; i < seq_len * D; i++) {
                    dnorm_attn[i] += tmp_na[i];
                }

                free(tmp_na);
            }
        }

        /* Accumulate weight gradients */
        /* dW_q += norm_attn^T @ dQ : [D, S] @ [S, q_dim] = [D, q_dim] */
        bp_accum_matmul_tn((float *)bg->attn_grad.dW_q->data,
                           norm_attn_cache[l], dQ,
                           seq_len, D, q_dim);
        /* dW_k += norm_attn^T @ dK */
        bp_accum_matmul_tn((float *)bg->attn_grad.dW_k->data,
                           norm_attn_cache[l], dK,
                           seq_len, D, kv_dim);
        /* dW_v += norm_attn^T @ dV */
        bp_accum_matmul_tn((float *)bg->attn_grad.dW_v->data,
                           norm_attn_cache[l], dV,
                           seq_len, D, kv_dim);

        /* ---- Attention norm backward ---- */
        /* dv[l] = dh (from residual: h[l] = v[l] + attn_out)
         * dv[l] += rmsnorm_backward(dnorm_attn, v[l], attn_norm) */
        memset(dv_next, 0, (size_t)seq_len * (size_t)D * sizeof(float));
        bp_rmsnorm_backward_2d(dv_next,
                               (float *)bg->attn_norm_grad.dweight->data,
                               v_cache[l], dnorm_attn,
                               block->attn_norm, seq_len, D);

        /* dv[l] = dh + dv_from_norm */
        for (int32_t i = 0; i < seq_len * D; i++) {
            dv_next[i] += dh[i];
        }

        /* Move to next layer */
        memcpy(dv_cur, dv_next, (size_t)seq_len * (size_t)D * sizeof(float));
    }

    /* Embedding backward: dW_embed already accumulated in output head.
     * Also need: dv[0] flows into embedding gradient for the input embeddings.
     * dW_embed[token_id] += dv_cur[s] for each position s.
     */
    {
        float *dw_embed = (float *)grads->embed_grad.dweight->data;
        for (int32_t s = 0; s < seq_len; s++) {
            int32_t tid = tokens[s];
            if (tid >= 0 && tid < V) {
                for (int32_t d = 0; d < D; d++) {
                    dw_embed[(size_t)tid * D + d] += dv_cur[(size_t)s * D + d];
                }
            }
        }
    }

    /* Free backward working buffers */
    free(dh); free(dffn_out); free(dnorm_ffn); free(dnorm_attn);
    free(dattn_out); free(dcontext); free(dQ); free(dK); free(dV);
    free(dv_next); free(dv_cur);

    /* ================================================================
     * Phase 3: Clip gradients, apply optimizer
     * (identical to ipc_train.c lines 1290-1407)
     * ================================================================ */

    bool is_last_micro = (micro_batch_idx == accum_steps - 1);

    if (is_last_micro) {
        /* Average gradients across micro-batches */
        if (accum_steps > 1) {
            float inv_accum = 1.0f / (float)accum_steps;

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

        /* Record gradient norm before clipping */
        result.grad_norm = grad_global_norm(grads);

        /* Clip gradients */
        grad_clip(grads, tcfg->grad_clip_norm);

        /* Apply optimizer with Depth-muP scaling */
        if (adam && tcfg->use_adam) {
            grad_apply_adam(model, grads, adam, cfg, tcfg, step);
        } else {
            /* For backprop SGD, pass NULL precision (no iPC precision schedule) */
            float *unity_precision = (float *)calloc((size_t)L, sizeof(float));
            if (unity_precision) {
                for (int32_t l2 = 0; l2 < L; l2++) {
                    unity_precision[l2] = 1.0f;
                }
                grad_apply_sgd(model, grads, cfg, tcfg, unity_precision, step);
                free(unity_precision);
            }
        }
    } else {
        result.grad_norm = grad_global_norm(grads);
    }

    /* ================================================================
     * D-082: Update router biases for Loss-Free Balancing.
     * Only on last micro-batch when LFB is enabled.
     * Mirrors the bias update in scale_experiment.c for iPC mode.
     * ================================================================ */
    if (is_last_micro && tcfg->use_loss_free_balance) {
        int32_t *expert_counts = (int32_t *)calloc((size_t)K, sizeof(int32_t));
        if (expert_counts) {
            for (int32_t l = 0; l < L; l++) {
                memset(expert_counts, 0, (size_t)K * sizeof(int32_t));
                for (int32_t s = 0; s < seq_len; s++) {
                    for (int32_t a = 0; a < k_active; a++) {
                        int32_t eid = route_expert_ids[l][(size_t)s * k_active + a];
                        if (eid >= 0 && eid < K) {
                            expert_counts[eid]++;
                        }
                    }
                }
                int32_t total_decisions = seq_len * k_active;
                router_update_bias(model->layers[l]->router,
                                   expert_counts, total_decisions, K);
            }
            free(expert_counts);
        }
    }

    /* ================================================================
     * Compute result
     * ================================================================ */

    /* Fill loss components */
    result.loss.lm = ce_loss;
    result.loss.pred_error = 0.0f; /* No prediction errors in backprop */
    result.loss.balance = 0.0f;
    result.loss.total = ce_loss;
    result.loss.perplexity = expf(ce_loss);

    /* Mean routing entropy */
    float total_entropy = 0.0f;
    for (int32_t l = 0; l < L; l++) {
        total_entropy += route_entropies[l];
    }
    result.mean_entropy = total_entropy / (float)L;

    /* No value nodes in backprop */
    result.vn_delta = 0.0f;

    /* Record QAT coverage count for test observability (clause 1). */
    s_last_qat_covered_count = qat_context_covered_count(qat);

    /* Cleanup */
    free(norm_final);
    pool_destroy(scratch);

cleanup:
    /* Destroy QAT context (owns all w_hat shadow-weight storage). */
    qat_context_destroy(qat);
    /* Free all cached activations */
    if (v_cache) {
        for (int32_t l = 0; l <= L; l++) {
            free(v_cache[l]);
        }
        free(v_cache);
    }

    for (int32_t l = 0; l < L; l++) {
        if (norm_attn_cache) free(norm_attn_cache[l]);
        if (q_cache) free(q_cache[l]);
        if (k_cache) free(k_cache[l]);
        if (vv_cache) free(vv_cache[l]);
        if (attn_w_cache) free(attn_w_cache[l]);
        if (context_cache) free(context_cache[l]);
        if (h_cache) free(h_cache[l]);
        if (norm_ffn_cache) free(norm_ffn_cache[l]);
        if (route_expert_ids) free(route_expert_ids[l]);
        if (route_weights) free(route_weights[l]);

        if (gate_pre_cache && gate_pre_cache[l]) {
            for (int32_t j = 0; j < K; j++) free(gate_pre_cache[l][j]);
            free(gate_pre_cache[l]);
        }
        if (up_pre_cache && up_pre_cache[l]) {
            for (int32_t j = 0; j < K; j++) free(up_pre_cache[l][j]);
            free(up_pre_cache[l]);
        }
        if (gate_act_cache && gate_act_cache[l]) {
            for (int32_t j = 0; j < K; j++) free(gate_act_cache[l][j]);
            free(gate_act_cache[l]);
        }
        if (hidden_cache && hidden_cache[l]) {
            for (int32_t j = 0; j < K; j++) free(hidden_cache[l][j]);
            free(hidden_cache[l]);
        }
        if (expert_batches && expert_batches[l]) {
            for (int32_t j = 0; j < K; j++) {
                free(expert_batches[l][j].token_indices);
                free(expert_batches[l][j].active_slots);
            }
            free(expert_batches[l]);
        }
    }

    free(norm_attn_cache); free(q_cache); free(k_cache); free(vv_cache);
    free(attn_w_cache); free(context_cache); free(h_cache); free(norm_ffn_cache);
    free(gate_pre_cache); free(up_pre_cache); free(gate_act_cache); free(hidden_cache);
    free(route_expert_ids); free(route_weights); free(route_entropies);
    free(expert_batches);

    return result;
}
