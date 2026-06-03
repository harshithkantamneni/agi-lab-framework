/* ipc_state.c -- iPC training state allocation and management. */

#include "ipc_state.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Estimate total bytes needed for all training state tensors. */
static size_t estimate_pool_size(const HSPAConfig *cfg, int32_t seq_len) {
    int32_t L = cfg->n_layers;
    int32_t D = cfg->d_model;
    int32_t V = cfg->vocab_size;
    int32_t H = cfg->n_heads;
    int32_t H_kv = cfg->n_kv_heads;
    int32_t D_h = cfg->head_dim;
    int32_t k = cfg->n_active;
    int32_t d_ff = cfg->d_ff;
    int32_t S = seq_len;

    size_t vn_bytes = (size_t)(L + 1) * S * D * 4;          /* value nodes */
    size_t pe_bytes = (size_t)L * S * D * 4;                 /* prediction errors */
    size_t logit_bytes = (size_t)S * V * 4;                  /* logits */

    /* Per-layer cache */
    size_t per_layer_cache =
        (size_t)S * D * 4 * 5 +                              /* 5 activation tensors */
        (size_t)S * H * D_h * 4 +                            /* Q */
        (size_t)S * H_kv * D_h * 4 * 2 +                    /* K_proj, V_proj */
        (size_t)H * S * S * 4 +                              /* attn_weights */
        (size_t)S * k * d_ff * 4 * 2;                        /* expert gate, up */
    size_t cache_bytes = (size_t)L * per_layer_cache;

    /* Routing state */
    size_t routing_bytes = (size_t)L * S * k * (4 + 4) +     /* ids + weights */
                           (size_t)L * S * 4;                 /* entropies */

    size_t total = vn_bytes + pe_bytes + logit_bytes + cache_bytes + routing_bytes;
    return total + total / 4; /* 25% overhead for alignment */
}

IPCTrainState *ipc_state_create(const HSPAConfig *cfg, int32_t seq_len,
                                float sigma_bottom, float sigma_top) {
    IPCTrainState *state = calloc(1, sizeof(IPCTrainState));
    if (!state) return NULL;

    int32_t L = cfg->n_layers;
    int32_t D = cfg->d_model;
    int32_t V = cfg->vocab_size;
    int32_t k = cfg->n_active;

    state->n_layers = L;
    state->seq_len = seq_len;

    /* Create dedicated pool for training state */
    size_t pool_size = estimate_pool_size(cfg, seq_len);
    state->train_pool = pool_create(pool_size, POOL_WEIGHTS);
    if (!state->train_pool) { free(state); return NULL; }

    MemoryPool *p = state->train_pool;

    /* --- Value nodes: L+1 tensors of [seq_len, d_model] --- */
    state->vn.count = L + 1;
    state->vn.seq_len = seq_len;
    state->vn.d_model = D;
    state->vn.values = calloc((size_t)(L + 1), sizeof(Tensor *));
    if (!state->vn.values) goto fail;

    for (int32_t i = 0; i <= L; i++) {
        state->vn.values[i] = tensor_create(p, (int32_t[]){seq_len, D}, 2,
                                            DTYPE_FP32);
        if (!state->vn.values[i]) goto fail;
        tensor_fill(state->vn.values[i], 0.0f);
    }

    /* --- Prediction errors: L tensors of [seq_len, d_model] --- */
    state->pe.count = L;
    state->pe.errors = calloc((size_t)L, sizeof(Tensor *));
    state->pe.precision = calloc((size_t)L, sizeof(float));
    if (!state->pe.errors || !state->pe.precision) goto fail;

    for (int32_t l = 0; l < L; l++) {
        state->pe.errors[l] = tensor_create(p, (int32_t[]){seq_len, D}, 2,
                                            DTYPE_FP32);
        if (!state->pe.errors[l]) goto fail;
        tensor_fill(state->pe.errors[l], 0.0f);
    }

    /* Fixed geometric precision schedule (Qi et al. 2025):
     * sigma^2[l] = exp(log(sigma_bottom) + l * (log(sigma_top) - log(sigma_bottom)) / (L-1))
     * Bottom layer: sigma_bottom (1.0), top layer: sigma_top (0.01). */
    if (L == 1) {
        state->pe.precision[0] = sigma_bottom;
    } else {
        float log_bot = logf(sigma_bottom);
        float log_top = logf(sigma_top);
        for (int32_t l = 0; l < L; l++) {
            float t = (float)l / (float)(L - 1);
            state->pe.precision[l] = expf(log_bot + t * (log_top - log_bot));
        }
    }

    /* --- Routing state: L layers --- */
    state->routing = calloc((size_t)L, sizeof(RoutingState));
    if (!state->routing) goto fail;

    for (int32_t l = 0; l < L; l++) {
        state->routing[l].seq_len = seq_len;
        state->routing[l].n_active = k;
        state->routing[l].expert_ids = calloc((size_t)seq_len * k,
                                              sizeof(int32_t));
        state->routing[l].expert_weights = calloc((size_t)seq_len * k,
                                                  sizeof(float));
        state->routing[l].entropies = calloc((size_t)seq_len, sizeof(float));
        if (!state->routing[l].expert_ids ||
            !state->routing[l].expert_weights ||
            !state->routing[l].entropies) goto fail;
    }

    /* --- Layer caches: L layers --- */
    state->layer_caches = calloc((size_t)L, sizeof(LayerCache));
    if (!state->layer_caches) goto fail;

    for (int32_t l = 0; l < L; l++) {
        LayerCache *lc = &state->layer_caches[l];

        lc->pre_attn_norm = tensor_create(p, (int32_t[]){seq_len, D}, 2,
                                          DTYPE_FP32);
        lc->attn_out = tensor_create(p, (int32_t[]){seq_len, D}, 2,
                                     DTYPE_FP32);
        lc->post_attn_residual = tensor_create(p, (int32_t[]){seq_len, D}, 2,
                                               DTYPE_FP32);
        lc->pre_ffn_norm = tensor_create(p, (int32_t[]){seq_len, D}, 2,
                                         DTYPE_FP32);
        lc->moe_out = tensor_create(p, (int32_t[]){seq_len, D}, 2,
                                    DTYPE_FP32);

        if (!lc->pre_attn_norm || !lc->attn_out || !lc->post_attn_residual ||
            !lc->pre_ffn_norm || !lc->moe_out) goto fail;

        /* Attention internals */
        int32_t qd = cfg->n_heads * cfg->head_dim;
        int32_t kvd = cfg->n_kv_heads * cfg->head_dim;
        lc->Q = tensor_create(p, (int32_t[]){seq_len, qd}, 2, DTYPE_FP32);
        lc->K_proj = tensor_create(p, (int32_t[]){seq_len, kvd}, 2, DTYPE_FP32);
        lc->V_proj = tensor_create(p, (int32_t[]){seq_len, kvd}, 2, DTYPE_FP32);
        lc->attn_weights = tensor_create(p,
            (int32_t[]){cfg->n_heads, seq_len, seq_len}, 3, DTYPE_FP32);

        if (!lc->Q || !lc->K_proj || !lc->V_proj || !lc->attn_weights)
            goto fail;

        /* Expert intermediates */
        lc->n_active_total = seq_len * k;
        lc->expert_gate = calloc((size_t)lc->n_active_total, sizeof(Tensor *));
        lc->expert_up = calloc((size_t)lc->n_active_total, sizeof(Tensor *));
        if (!lc->expert_gate || !lc->expert_up) goto fail;

        for (int32_t idx = 0; idx < lc->n_active_total; idx++) {
            lc->expert_gate[idx] = tensor_create(p,
                (int32_t[]){cfg->d_ff}, 1, DTYPE_FP32);
            lc->expert_up[idx] = tensor_create(p,
                (int32_t[]){cfg->d_ff}, 1, DTYPE_FP32);
            if (!lc->expert_gate[idx] || !lc->expert_up[idx]) goto fail;
        }
    }

    /* --- Logits: [seq_len, vocab_size] --- */
    state->logits = tensor_create(p, (int32_t[]){seq_len, V}, 2, DTYPE_FP32);
    if (!state->logits) goto fail;

    return state;

fail:
    ipc_state_destroy(state);
    return NULL;
}

void ipc_state_reset(IPCTrainState *state) {
    if (!state) return;

    /* Zero value nodes */
    for (int32_t i = 0; i < state->vn.count; i++) {
        if (state->vn.values[i])
            tensor_fill(state->vn.values[i], 0.0f);
    }

    /* Zero prediction errors */
    for (int32_t l = 0; l < state->pe.count; l++) {
        if (state->pe.errors[l])
            tensor_fill(state->pe.errors[l], 0.0f);
    }

    /* Zero routing state */
    for (int32_t l = 0; l < state->n_layers; l++) {
        RoutingState *rs = &state->routing[l];
        memset(rs->expert_ids, 0,
               (size_t)rs->seq_len * rs->n_active * sizeof(int32_t));
        memset(rs->expert_weights, 0,
               (size_t)rs->seq_len * rs->n_active * sizeof(float));
        memset(rs->entropies, 0, (size_t)rs->seq_len * sizeof(float));
        rs->mean_entropy = 0.0f;
    }

    /* Zero logits */
    if (state->logits)
        tensor_fill(state->logits, 0.0f);
}

void ipc_state_destroy(IPCTrainState *state) {
    if (!state) return;

    /* Value node array (tensors owned by pool) */
    free(state->vn.values);

    /* Prediction errors */
    free(state->pe.errors);
    free(state->pe.precision);

    /* Routing state */
    if (state->routing) {
        for (int32_t l = 0; l < state->n_layers; l++) {
            free(state->routing[l].expert_ids);
            free(state->routing[l].expert_weights);
            free(state->routing[l].entropies);
        }
        free(state->routing);
    }

    /* Layer caches (tensor data owned by pool, but pointer arrays are heap) */
    if (state->layer_caches) {
        for (int32_t l = 0; l < state->n_layers; l++) {
            free(state->layer_caches[l].expert_gate);
            free(state->layer_caches[l].expert_up);
        }
        free(state->layer_caches);
    }

    /* Pool owns all tensor data */
    if (state->train_pool)
        pool_destroy(state->train_pool);

    free(state);
}
