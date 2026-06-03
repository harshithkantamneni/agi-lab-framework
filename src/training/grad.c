/* grad.c -- Gradient accumulator allocation, zeroing, clipping, SGD, and AdamW. */

#include "grad.h"
#include "ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Compute learning rate: linear warmup + cosine decay. */
static float compute_lr(const TrainConfig *tcfg, int32_t step) {
    float lr = tcfg->base_lr;
    if (step < tcfg->lr_warmup_steps && tcfg->lr_warmup_steps > 0) {
        lr = tcfg->base_lr * ((float)(step + 1) / (float)tcfg->lr_warmup_steps);
    } else if (tcfg->max_steps > tcfg->lr_warmup_steps) {
        float progress = (float)(step - tcfg->lr_warmup_steps)
                       / (float)(tcfg->max_steps - tcfg->lr_warmup_steps);
        if (progress > 1.0f) progress = 1.0f;
        lr = tcfg->lr_min + 0.5f * (tcfg->base_lr - tcfg->lr_min)
             * (1.0f + cosf((float)M_PI * progress));
    }
    if (lr < tcfg->lr_min) lr = tcfg->lr_min;
    return lr;
}

/* Estimate total bytes for all gradient tensors. */
static size_t estimate_grad_pool(const HSPAConfig *cfg) {
    int32_t D = cfg->d_model;
    int32_t L = cfg->n_layers;
    int32_t H = cfg->n_heads;
    int32_t H_kv = cfg->n_kv_heads;
    int32_t D_h = cfg->head_dim;
    int32_t K = cfg->n_experts;
    int32_t d_ff = cfg->d_ff;
    int32_t V = cfg->vocab_size;

    /* Embedding: V * D */
    size_t embed = (size_t)V * D * 4;

    /* Per layer: attn (4 weight matrices) + router (2) + K experts (3 each) + 2 norms */
    size_t per_layer =
        (size_t)D * H * D_h * 4 +           /* W_q */
        (size_t)D * H_kv * D_h * 4 +        /* W_k */
        (size_t)D * H_kv * D_h * 4 +        /* W_v */
        (size_t)H * D_h * D * 4 +           /* W_o */
        (size_t)D * K * 4 +                 /* W_mu */
        (size_t)D * K * 4 +                 /* W_sigma */
        (size_t)K * (D * d_ff * 4 * 2 + d_ff * D * 4) + /* experts */
        (size_t)D * 4 * 2;                  /* 2 norm weights */

    /* Final norm */
    size_t final = (size_t)D * 4;

    size_t total = embed + per_layer * L + final;
    return total + total / 4; /* 25% alignment overhead */
}

/* Helper: add squared elements of tensor to accumulator. */
static void accum_sq_norm(const Tensor *t, double *acc) {
    if (!t) return;
    int32_t n = tensor_numel(t);
    const float *d = t->data;
    double s = 0.0;
    for (int32_t i = 0; i < n; i++) {
        s += (double)d[i] * (double)d[i];
    }
    *acc += s;
}

/* Helper: scale all elements of a tensor in-place. */
static void tensor_scale_inplace(Tensor *t, float s) {
    if (!t) return;
    int32_t n = tensor_numel(t);
    float *d = t->data;
    for (int32_t i = 0; i < n; i++) {
        d[i] *= s;
    }
}

/* Helper: apply SGD update to a weight tensor: w -= lr * grad. */
static void sgd_update(Tensor *weight, const Tensor *grad, float lr) {
    if (!weight || !grad) return;
    int32_t n = tensor_numel(weight);
    float *w = weight->data;
    const float *g = grad->data;
    for (int32_t i = 0; i < n; i++) {
        w[i] -= lr * g[i];
    }
}

ModelGrad *grad_create(const HSPAConfig *cfg) {
    ModelGrad *grads = calloc(1, sizeof(ModelGrad));
    if (!grads) return NULL;

    int32_t L = cfg->n_layers;
    int32_t D = cfg->d_model;
    int32_t K = cfg->n_experts;

    grads->n_layers = L;

    size_t pool_size = estimate_grad_pool(cfg);
    grads->grad_pool = pool_create(pool_size, POOL_WEIGHTS);
    if (!grads->grad_pool) { free(grads); return NULL; }

    MemoryPool *p = grads->grad_pool;

    /* Embedding gradient */
    grads->embed_grad.dweight = tensor_create(p,
        (int32_t[]){cfg->vocab_size, D}, 2, DTYPE_FP32);
    if (!grads->embed_grad.dweight) goto fail;

    /* Block gradients */
    grads->block_grads = calloc((size_t)L, sizeof(BlockGrad));
    if (!grads->block_grads) goto fail;

    for (int32_t l = 0; l < L; l++) {
        BlockGrad *bg = &grads->block_grads[l];
        bg->n_experts = K;

        /* Attention norm */
        bg->attn_norm_grad.dweight = tensor_create(p,
            (int32_t[]){D}, 1, DTYPE_FP32);

        /* Attention */
        int32_t qd = cfg->n_heads * cfg->head_dim;
        int32_t kvd = cfg->n_kv_heads * cfg->head_dim;
        bg->attn_grad.dW_q = tensor_create(p, (int32_t[]){D, qd}, 2, DTYPE_FP32);
        bg->attn_grad.dW_k = tensor_create(p, (int32_t[]){D, kvd}, 2, DTYPE_FP32);
        bg->attn_grad.dW_v = tensor_create(p, (int32_t[]){D, kvd}, 2, DTYPE_FP32);
        bg->attn_grad.dW_o = tensor_create(p, (int32_t[]){qd, D}, 2, DTYPE_FP32);

        /* FFN norm */
        bg->ffn_norm_grad.dweight = tensor_create(p,
            (int32_t[]){D}, 1, DTYPE_FP32);

        /* Router */
        bg->router_grad.dW_mu = tensor_create(p,
            (int32_t[]){D, K}, 2, DTYPE_FP32);
        bg->router_grad.dW_sigma = tensor_create(p,
            (int32_t[]){D, K}, 2, DTYPE_FP32);

        /* Expert gradients */
        bg->expert_grads = calloc((size_t)K, sizeof(ExpertFFNGrad));
        if (!bg->expert_grads) goto fail;

        for (int32_t e = 0; e < K; e++) {
            bg->expert_grads[e].dW_gate = tensor_create(p,
                (int32_t[]){D, cfg->d_ff}, 2, DTYPE_FP32);
            bg->expert_grads[e].dW_up = tensor_create(p,
                (int32_t[]){D, cfg->d_ff}, 2, DTYPE_FP32);
            bg->expert_grads[e].dW_down = tensor_create(p,
                (int32_t[]){cfg->d_ff, D}, 2, DTYPE_FP32);
        }

        /* Verify all allocations */
        if (!bg->attn_norm_grad.dweight || !bg->ffn_norm_grad.dweight ||
            !bg->attn_grad.dW_q || !bg->attn_grad.dW_k ||
            !bg->attn_grad.dW_v || !bg->attn_grad.dW_o ||
            !bg->router_grad.dW_mu || !bg->router_grad.dW_sigma)
            goto fail;

        for (int32_t e = 0; e < K; e++) {
            if (!bg->expert_grads[e].dW_gate ||
                !bg->expert_grads[e].dW_up ||
                !bg->expert_grads[e].dW_down) goto fail;
        }
    }

    /* Final norm gradient */
    grads->final_norm_grad.dweight = tensor_create(p,
        (int32_t[]){D}, 1, DTYPE_FP32);
    if (!grads->final_norm_grad.dweight) goto fail;

    /* Zero everything */
    grad_zero(grads);
    return grads;

fail:
    grad_destroy(grads);
    return NULL;
}

/* Helper: zero a single tensor. */
static void tensor_zero(Tensor *t) {
    if (!t) return;
    tensor_fill(t, 0.0f);
}

void grad_zero(ModelGrad *grads) {
    if (!grads) return;

    tensor_zero(grads->embed_grad.dweight);
    tensor_zero(grads->final_norm_grad.dweight);

    for (int32_t l = 0; l < grads->n_layers; l++) {
        BlockGrad *bg = &grads->block_grads[l];

        tensor_zero(bg->attn_norm_grad.dweight);
        tensor_zero(bg->attn_grad.dW_q);
        tensor_zero(bg->attn_grad.dW_k);
        tensor_zero(bg->attn_grad.dW_v);
        tensor_zero(bg->attn_grad.dW_o);
        tensor_zero(bg->ffn_norm_grad.dweight);
        tensor_zero(bg->router_grad.dW_mu);
        tensor_zero(bg->router_grad.dW_sigma);

        for (int32_t e = 0; e < bg->n_experts; e++) {
            tensor_zero(bg->expert_grads[e].dW_gate);
            tensor_zero(bg->expert_grads[e].dW_up);
            tensor_zero(bg->expert_grads[e].dW_down);
        }
    }
}

float grad_global_norm(const ModelGrad *grads) {
    if (!grads) return 0.0f;

    double acc = 0.0;

    accum_sq_norm(grads->embed_grad.dweight, &acc);
    accum_sq_norm(grads->final_norm_grad.dweight, &acc);

    for (int32_t l = 0; l < grads->n_layers; l++) {
        const BlockGrad *bg = &grads->block_grads[l];

        accum_sq_norm(bg->attn_norm_grad.dweight, &acc);
        accum_sq_norm(bg->attn_grad.dW_q, &acc);
        accum_sq_norm(bg->attn_grad.dW_k, &acc);
        accum_sq_norm(bg->attn_grad.dW_v, &acc);
        accum_sq_norm(bg->attn_grad.dW_o, &acc);
        accum_sq_norm(bg->ffn_norm_grad.dweight, &acc);
        accum_sq_norm(bg->router_grad.dW_mu, &acc);
        accum_sq_norm(bg->router_grad.dW_sigma, &acc);

        for (int32_t e = 0; e < bg->n_experts; e++) {
            accum_sq_norm(bg->expert_grads[e].dW_gate, &acc);
            accum_sq_norm(bg->expert_grads[e].dW_up, &acc);
            accum_sq_norm(bg->expert_grads[e].dW_down, &acc);
        }
    }

    return (float)sqrt(acc);
}

void grad_clip(ModelGrad *grads, float max_norm) {
    if (!grads || max_norm <= 0.0f) return;

    float norm = grad_global_norm(grads);
    if (norm <= max_norm) return;

    float scale = max_norm / norm;

    tensor_scale_inplace(grads->embed_grad.dweight, scale);
    tensor_scale_inplace(grads->final_norm_grad.dweight, scale);

    for (int32_t l = 0; l < grads->n_layers; l++) {
        BlockGrad *bg = &grads->block_grads[l];

        tensor_scale_inplace(bg->attn_norm_grad.dweight, scale);
        tensor_scale_inplace(bg->attn_grad.dW_q, scale);
        tensor_scale_inplace(bg->attn_grad.dW_k, scale);
        tensor_scale_inplace(bg->attn_grad.dW_v, scale);
        tensor_scale_inplace(bg->attn_grad.dW_o, scale);
        tensor_scale_inplace(bg->ffn_norm_grad.dweight, scale);
        tensor_scale_inplace(bg->router_grad.dW_mu, scale);
        tensor_scale_inplace(bg->router_grad.dW_sigma, scale);

        for (int32_t e = 0; e < bg->n_experts; e++) {
            tensor_scale_inplace(bg->expert_grads[e].dW_gate, scale);
            tensor_scale_inplace(bg->expert_grads[e].dW_up, scale);
            tensor_scale_inplace(bg->expert_grads[e].dW_down, scale);
        }
    }
}

void grad_apply_sgd(HSPAModel *model, const ModelGrad *grads,
                    const HSPAConfig *cfg, const TrainConfig *tcfg,
                    const float *precision, int32_t step) {
    if (!model || !grads || !cfg || !tcfg) return;
    (void)precision; /* Precision scaling is in the gradient, not the LR */

    int32_t D = cfg->d_model;
    int32_t L = cfg->n_layers;

    float lr = compute_lr(tcfg, step);

    /* muP width scaling factors */
    float eta_width = 1.0f / (float)D;  /* muP: scale down with width */

    /* Embedding: no width scaling, no depth scaling */
    sgd_update(model->embed->weight, grads->embed_grad.dweight, lr);

    /* Per-layer updates with Depth-muP scaling */
    for (int32_t l = 0; l < L; l++) {
        const BlockGrad *bg = &grads->block_grads[l];
        HSPABlock *block = model->layers[l];

        /* muP width scaling only. Precision scaling is already in the
         * gradient (Step D scales by 1/sigma^2), so do NOT double-count. */
        float layer_lr = lr * eta_width;

        /* Attention weights */
        sgd_update(block->attn->W_q, bg->attn_grad.dW_q, layer_lr);
        sgd_update(block->attn->W_k, bg->attn_grad.dW_k, layer_lr);
        sgd_update(block->attn->W_v, bg->attn_grad.dW_v, layer_lr);
        sgd_update(block->attn->W_o, bg->attn_grad.dW_o, layer_lr);

        /* Norm weights: no width scaling */
        sgd_update(block->attn_norm->weight, bg->attn_norm_grad.dweight, lr);
        sgd_update(block->ffn_norm->weight, bg->ffn_norm_grad.dweight, lr);

        /* Router weights: no muP width scaling.
         * W_mu is [D, K] where K is fixed (not scaling with D).
         * muP 1/D scaling is for layers where BOTH dims scale with D. */
        sgd_update(block->router->W_mu, bg->router_grad.dW_mu, lr);
        sgd_update(block->router->W_sigma, bg->router_grad.dW_sigma, lr);

        /* Expert FFN weights */
        for (int32_t e = 0; e < bg->n_experts; e++) {
            sgd_update(block->experts[e]->W_gate,
                       bg->expert_grads[e].dW_gate, layer_lr);
            sgd_update(block->experts[e]->W_up,
                       bg->expert_grads[e].dW_up, layer_lr);
            sgd_update(block->experts[e]->W_down,
                       bg->expert_grads[e].dW_down, layer_lr);
        }
    }

    /* Final norm: no width or depth scaling */
    sgd_update(model->final_norm->weight, grads->final_norm_grad.dweight, lr);
}

void grad_destroy(ModelGrad *grads) {
    if (!grads) return;

    if (grads->block_grads) {
        for (int32_t l = 0; l < grads->n_layers; l++) {
            free(grads->block_grads[l].expert_grads);
        }
        free(grads->block_grads);
    }

    if (grads->grad_pool)
        pool_destroy(grads->grad_pool);

    free(grads);
}

/* ========================================================================
 * AdamW Optimizer
 * ======================================================================== */

/* Estimate total bytes for Adam moment tensors (2x the gradient pool:
 * one for first moments m, one for second moments v). */
static size_t estimate_adam_pool(const HSPAConfig *cfg) {
    /* Adam needs exactly 2x the gradient storage (m and v per parameter). */
    size_t grad_est = estimate_grad_pool(cfg);
    return grad_est * 2 + grad_est / 2; /* 2x + extra alignment headroom */
}

/* Helper: create an AdamMoment pair (m and v) with the same shape as `ref`.
 * Both tensors are zero-initialized. Returns false on failure. */
static bool adam_moment_create(AdamMoment *am, MemoryPool *pool,
                               const int32_t *shape, int32_t ndim) {
    am->m = tensor_create(pool, shape, ndim, DTYPE_FP32);
    am->v = tensor_create(pool, shape, ndim, DTYPE_FP32);
    if (!am->m || !am->v) return false;
    tensor_fill(am->m, 0.0f);
    tensor_fill(am->v, 0.0f);
    return true;
}

AdamState *adam_create(const HSPAConfig *cfg) {
    AdamState *adam = calloc(1, sizeof(AdamState));
    if (!adam) return NULL;

    int32_t L = cfg->n_layers;
    int32_t D = cfg->d_model;
    int32_t K = cfg->n_experts;

    adam->n_layers = L;

    size_t pool_size = estimate_adam_pool(cfg);
    adam->adam_pool = pool_create(pool_size, POOL_WEIGHTS);
    if (!adam->adam_pool) { free(adam); return NULL; }

    MemoryPool *p = adam->adam_pool;

    /* Embedding moments */
    if (!adam_moment_create(&adam->embed_m, p,
            (int32_t[]){cfg->vocab_size, D}, 2))
        goto fail;

    /* Per-layer moments */
    adam->layers = calloc((size_t)L, sizeof(struct AdamLayerState));
    if (!adam->layers) goto fail;

    for (int32_t l = 0; l < L; l++) {
        struct AdamLayerState *ls = &adam->layers[l];
        ls->n_experts = K;

        /* Attention norm */
        if (!adam_moment_create(&ls->attn_norm_m, p,
                (int32_t[]){D}, 1))
            goto fail;

        /* Attention weights */
        int32_t qd = cfg->n_heads * cfg->head_dim;
        int32_t kvd = cfg->n_kv_heads * cfg->head_dim;

        if (!adam_moment_create(&ls->attn_q_m, p,
                (int32_t[]){D, qd}, 2))
            goto fail;
        if (!adam_moment_create(&ls->attn_k_m, p,
                (int32_t[]){D, kvd}, 2))
            goto fail;
        if (!adam_moment_create(&ls->attn_v_m, p,
                (int32_t[]){D, kvd}, 2))
            goto fail;
        if (!adam_moment_create(&ls->attn_o_m, p,
                (int32_t[]){qd, D}, 2))
            goto fail;

        /* FFN norm */
        if (!adam_moment_create(&ls->ffn_norm_m, p,
                (int32_t[]){D}, 1))
            goto fail;

        /* Router */
        if (!adam_moment_create(&ls->router_mu_m, p,
                (int32_t[]){D, K}, 2))
            goto fail;
        if (!adam_moment_create(&ls->router_sigma_m, p,
                (int32_t[]){D, K}, 2))
            goto fail;

        /* Expert moments */
        ls->expert_gate_m = calloc((size_t)K, sizeof(AdamMoment));
        ls->expert_up_m   = calloc((size_t)K, sizeof(AdamMoment));
        ls->expert_down_m = calloc((size_t)K, sizeof(AdamMoment));
        if (!ls->expert_gate_m || !ls->expert_up_m || !ls->expert_down_m)
            goto fail;

        for (int32_t e = 0; e < K; e++) {
            if (!adam_moment_create(&ls->expert_gate_m[e], p,
                    (int32_t[]){D, cfg->d_ff}, 2))
                goto fail;
            if (!adam_moment_create(&ls->expert_up_m[e], p,
                    (int32_t[]){D, cfg->d_ff}, 2))
                goto fail;
            if (!adam_moment_create(&ls->expert_down_m[e], p,
                    (int32_t[]){cfg->d_ff, D}, 2))
                goto fail;
        }
    }

    /* Final norm moments */
    if (!adam_moment_create(&adam->final_norm_m, p,
            (int32_t[]){D}, 1))
        goto fail;

    return adam;

fail:
    adam_destroy(adam);
    return NULL;
}

/* AdamW update for a single parameter tensor.
 *
 * m = beta1 * m + (1 - beta1) * g
 * v = beta2 * v + (1 - beta2) * g^2
 * m_hat = m / (1 - beta1^t)
 * v_hat = v / (1 - beta2^t)
 * w = w * (1 - lr * weight_decay)     [decoupled weight decay]
 * w -= lr * m_hat / (sqrt(v_hat) + eps)
 *
 * Uses double precision for bias correction to avoid float underflow
 * at early steps where beta^t is close to 1.
 */
static void adamw_update(Tensor *weight, const Tensor *grad,
                         AdamMoment *am, float lr,
                         float beta1, float beta2, float eps,
                         float wd, int32_t step) {
    if (!weight || !grad || !am->m || !am->v) return;

    int32_t n = (int32_t)tensor_numel(weight);
    float *w = weight->data;
    const float *g = grad->data;
    float *m = am->m->data;
    float *v = am->v->data;

    /* Bias correction factors (double precision for numerical safety).
     * Use pow() instead of iterative multiplication — O(1) vs O(step). */
    double b1t = pow((double)beta1, (double)step);
    double b2t = pow((double)beta2, (double)step);
    double bc1 = 1.0 / (1.0 - b1t);
    double bc2 = 1.0 / (1.0 - b2t);

    float bc1f = (float)bc1;
    float bc2f = (float)bc2;

    for (int32_t i = 0; i < n; i++) {
        /* Update moments */
        m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];

        /* Bias-corrected estimates */
        float m_hat = m[i] * bc1f;
        float v_hat = v[i] * bc2f;

        /* Decoupled weight decay (AdamW) */
        w[i] *= (1.0f - lr * wd);

        /* Parameter update */
        w[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/* AdamW update without weight decay (for norm weights). */
static void adam_update_no_wd(Tensor *weight, const Tensor *grad,
                              AdamMoment *am, float lr,
                              float beta1, float beta2, float eps,
                              int32_t step) {
    adamw_update(weight, grad, am, lr, beta1, beta2, eps, 0.0f, step);
}

void grad_apply_adam(HSPAModel *model, const ModelGrad *grads,
                     AdamState *adam, const HSPAConfig *cfg,
                     const TrainConfig *tcfg, int32_t step) {
    if (!model || !grads || !adam || !cfg || !tcfg) return;

    int32_t D = cfg->d_model;
    int32_t L = cfg->n_layers;

    float lr = compute_lr(tcfg, step);

    /* muP width scaling factor */
    float eta_width = 1.0f / (float)D;

    float beta1 = tcfg->adam_beta1;
    float beta2 = tcfg->adam_beta2;
    float eps   = tcfg->adam_eps;
    float wd    = tcfg->weight_decay;

    /* muP weight decay scaling (Wang et al. ICML 2025):
     * wd_scaled = wd_base * (D / D_base) for width-scaled tensors.
     * Embedding, norm, and router weights use unscaled wd. */
    int32_t mup_base = tcfg->mup_base_width > 0 ? tcfg->mup_base_width : D;
    float wd_scaled = wd * (float)D / (float)mup_base;

    /* Adam step is 1-indexed for bias correction */
    int32_t adam_step = step + 1;

    /* Embedding: no width scaling, unscaled weight decay */
    adamw_update(model->embed->weight, grads->embed_grad.dweight,
                 &adam->embed_m, lr, beta1, beta2, eps, wd, adam_step);

    /* Per-layer updates with Depth-muP width scaling */
    for (int32_t l = 0; l < L; l++) {
        const BlockGrad *bg = &grads->block_grads[l];
        HSPABlock *block = model->layers[l];
        struct AdamLayerState *ls = &adam->layers[l];

        float layer_lr = lr * eta_width;

        /* Attention weights: width scaling + muP-scaled weight decay */
        adamw_update(block->attn->W_q, bg->attn_grad.dW_q,
                     &ls->attn_q_m, layer_lr, beta1, beta2, eps, wd_scaled,
                     adam_step);
        adamw_update(block->attn->W_k, bg->attn_grad.dW_k,
                     &ls->attn_k_m, layer_lr, beta1, beta2, eps, wd_scaled,
                     adam_step);
        adamw_update(block->attn->W_v, bg->attn_grad.dW_v,
                     &ls->attn_v_m, layer_lr, beta1, beta2, eps, wd_scaled,
                     adam_step);
        adamw_update(block->attn->W_o, bg->attn_grad.dW_o,
                     &ls->attn_o_m, layer_lr, beta1, beta2, eps, wd_scaled,
                     adam_step);

        /* Norm weights: no width scaling, no weight decay */
        adam_update_no_wd(block->attn_norm->weight,
                         bg->attn_norm_grad.dweight,
                         &ls->attn_norm_m, lr, beta1, beta2, eps,
                         adam_step);
        adam_update_no_wd(block->ffn_norm->weight,
                         bg->ffn_norm_grad.dweight,
                         &ls->ffn_norm_m, lr, beta1, beta2, eps,
                         adam_step);

        /* Router weights: no muP width scaling, unscaled weight decay
         * (K is fixed, not scaling with D) */
        adamw_update(block->router->W_mu, bg->router_grad.dW_mu,
                     &ls->router_mu_m, lr, beta1, beta2, eps, wd,
                     adam_step);
        adamw_update(block->router->W_sigma, bg->router_grad.dW_sigma,
                     &ls->router_sigma_m, lr, beta1, beta2, eps, wd,
                     adam_step);

        /* Expert FFN weights: width scaling + muP-scaled weight decay */
        for (int32_t e = 0; e < bg->n_experts; e++) {
            adamw_update(block->experts[e]->W_gate,
                         bg->expert_grads[e].dW_gate,
                         &ls->expert_gate_m[e], layer_lr,
                         beta1, beta2, eps, wd_scaled, adam_step);
            adamw_update(block->experts[e]->W_up,
                         bg->expert_grads[e].dW_up,
                         &ls->expert_up_m[e], layer_lr,
                         beta1, beta2, eps, wd_scaled, adam_step);
            adamw_update(block->experts[e]->W_down,
                         bg->expert_grads[e].dW_down,
                         &ls->expert_down_m[e], layer_lr,
                         beta1, beta2, eps, wd_scaled, adam_step);
        }
    }

    /* Final norm: no width scaling, no weight decay */
    adam_update_no_wd(model->final_norm->weight,
                     grads->final_norm_grad.dweight,
                     &adam->final_norm_m, lr, beta1, beta2, eps,
                     adam_step);
}

void adam_destroy(AdamState *adam) {
    if (!adam) return;

    if (adam->layers) {
        for (int32_t l = 0; l < adam->n_layers; l++) {
            struct AdamLayerState *ls = &adam->layers[l];
            free(ls->expert_gate_m);
            free(ls->expert_up_m);
            free(ls->expert_down_m);
        }
        free(adam->layers);
    }

    if (adam->adam_pool)
        pool_destroy(adam->adam_pool);

    free(adam);
}
