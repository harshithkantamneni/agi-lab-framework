/* weight_init.c -- Depth-muP weight initialization. */

#include "weight_init.h"
#include "ops.h"

#include <math.h>

/* Fill tensor with N(0, scale^2) using Box-Muller transform on uniform data. */
static void init_normal(Tensor *t, float scale) {
    if (!t || scale == 0.0f) return;
    /* First fill with uniform [0,1], then transform to normal */
    op_rand_uniform(t, 0.0f, 1.0f);

    int32_t n = tensor_numel(t);
    float *d = t->data;

    /* Box-Muller: pairs of uniform samples -> pairs of normal samples */
    int32_t i;
    for (i = 0; i + 1 < n; i += 2) {
        float u1 = d[i];
        float u2 = d[i + 1];
        /* Clamp to avoid log(0) */
        if (u1 < 1e-7f) u1 = 1e-7f;
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * (float)M_PI * u2;
        d[i]     = scale * r * cosf(theta);
        d[i + 1] = scale * r * sinf(theta);
    }
    /* Handle odd element */
    if (i < n) {
        float u1 = d[i];
        if (u1 < 1e-7f) u1 = 1e-7f;
        /* Use a fixed second uniform for the last element */
        float r = sqrtf(-2.0f * logf(u1));
        d[i] = scale * r;
    }
}

void weight_init_depth_mup(HSPAModel *model, const HSPAConfig *cfg) {
    if (!model || !cfg) return;

    int32_t D = cfg->d_model;
    int32_t L = cfg->n_layers;
    int32_t d_ff = cfg->d_ff;

    /* Width factor (standard muP): 1/sqrt(D) */
    float alpha_width = 1.0f / sqrtf((float)D);

    /* Depth factor (muPC extension): 1/sqrt(L) */
    float alpha_depth = 1.0f / sqrtf((float)L);

    /* Combined scale for attention and FFN input projections */
    float attn_scale = alpha_width * alpha_depth;

    /* Down projection: fan_in is d_ff, not D */
    float down_scale = (1.0f / sqrtf((float)d_ff)) * alpha_depth;

    /* Embedding: standard 1/sqrt(D), no depth factor */
    float embed_scale = 1.0f / sqrtf((float)D);

    /* Router: small init for near-uniform routing */
    float router_mu_scale = alpha_width * 0.01f;

    /* Initialize embedding */
    init_normal(model->embed->weight, embed_scale);

    /* Initialize each layer */
    for (int32_t l = 0; l < L; l++) {
        HSPABlock *block = model->layers[l];

        /* Attention weights */
        init_normal(block->attn->W_q, attn_scale);
        init_normal(block->attn->W_k, attn_scale);
        init_normal(block->attn->W_v, attn_scale);
        init_normal(block->attn->W_o, attn_scale);

        /* RMSNorm: all ones */
        tensor_fill(block->attn_norm->weight, 1.0f);
        tensor_fill(block->ffn_norm->weight, 1.0f);

        /* Router */
        init_normal(block->router->W_mu, router_mu_scale);
        tensor_fill(block->router->W_sigma, 0.0f); /* softplus(0) = ln(2) */

        /* Expert FFN weights */
        for (int32_t e = 0; e < cfg->n_experts; e++) {
            init_normal(block->experts[e]->W_gate, attn_scale);
            init_normal(block->experts[e]->W_up, attn_scale);
            init_normal(block->experts[e]->W_down, down_scale);
        }
    }

    /* Final norm: all ones */
    tensor_fill(model->final_norm->weight, 1.0f);
}
