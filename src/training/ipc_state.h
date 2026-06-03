/* ipc_state.h -- iPC value nodes, prediction errors, and training state.
 *
 * In incremental Predictive Coding (iPC):
 *   - v[l] is the value node at layer boundary l (L+1 total).
 *   - epsilon[l] = v[l+1] - f_l(v[l]) is the prediction error at layer l
 *     (bottom-up: f_l maps v[l] upward to predict v[l+1]).
 *   - Value nodes are initialized by a forward pass, then refined by
 *     T inference iterations to minimize variational free energy.
 *   - After convergence, prediction errors drive LOCAL weight updates.
 *
 * For the micro-experiment (L=4, S=128, D=128), all state fits in memory.
 * For the full model (L=32), a sliding window is used (future work).
 */

#ifndef IPC_STATE_H
#define IPC_STATE_H

#include "hspa_config.h"
#include "memory_pool.h"
#include "tensor.h"

#include <stdint.h>

/* Value nodes for all layer boundaries. */
typedef struct {
    Tensor **values;    /* v[l]: [n_layers+1] array of [seq_len, d_model].
                         * values[0] = embedding output (CLAMPED).
                         * values[l+1] = output of layer l. */
    int32_t count;      /* n_layers + 1 */
    int32_t seq_len;
    int32_t d_model;
} IPCValueNodes;

/* Prediction errors for all layers. */
typedef struct {
    Tensor **errors;    /* epsilon[l]: [n_layers] array of [seq_len, d_model].
                         * errors[l] = v[l] - f_l(v[l+1]). */
    float  *precision;  /* sigma_l^2: [n_layers] per-layer precision weights.
                         * Geometric schedule: sigma_bottom..sigma_top. */
    int32_t count;      /* n_layers */
} IPCPredErrors;

/* Per-layer routing decisions saved during forward pass. */
typedef struct {
    int32_t *expert_ids;    /* [seq_len * n_active] flattened expert indices */
    float   *expert_weights;/* [seq_len * n_active] flattened routing weights */
    float   *entropies;     /* [seq_len] per-token routing entropies */
    float    mean_entropy;  /* Mean entropy across all tokens */
    int32_t  seq_len;
    int32_t  n_active;
} RoutingState;

/* Per-layer saved activations needed for backward pass. */
typedef struct {
    Tensor *pre_attn_norm;      /* [seq_len, d_model] after attn_norm */
    Tensor *attn_out;           /* [seq_len, d_model] attention output */
    Tensor *post_attn_residual; /* [seq_len, d_model] x + attn_out */
    Tensor *pre_ffn_norm;       /* [seq_len, d_model] after ffn_norm */
    Tensor *moe_out;            /* [seq_len, d_model] weighted expert sum */

    /* Attention internals */
    Tensor *Q;                  /* [seq_len, n_heads * head_dim] */
    Tensor *K_proj;             /* [seq_len, n_kv_heads * head_dim] */
    Tensor *V_proj;             /* [seq_len, n_kv_heads * head_dim] */
    Tensor *attn_weights;       /* [n_heads, seq_len, seq_len] */

    /* Per-expert intermediates for active tokens */
    Tensor **expert_gate;       /* [seq_len * n_active] gate projections */
    Tensor **expert_up;         /* [seq_len * n_active] up projections */
    int32_t n_active_total;     /* seq_len * n_active */
} LayerCache;

/* Full iPC training state for one sample. */
typedef struct {
    IPCValueNodes  vn;          /* Value nodes for all layer boundaries */
    IPCPredErrors  pe;          /* Prediction errors for all layers */
    RoutingState  *routing;     /* [n_layers] per-layer routing decisions */
    LayerCache    *layer_caches;/* [n_layers] saved activations */
    Tensor        *logits;      /* [seq_len, vocab_size] model output */

    int32_t n_layers;
    int32_t seq_len;
    MemoryPool *train_pool;     /* Dedicated pool for all training state */
} IPCTrainState;

/* Create full iPC training state for a model config and sequence length.
 * Allocates all value nodes, prediction errors, routing state, layer caches.
 * precision is set to fixed geometric schedule from train_config.
 * Returns NULL on failure. */
IPCTrainState *ipc_state_create(const HSPAConfig *cfg, int32_t seq_len,
                                float sigma_bottom, float sigma_top);

/* Reset training state for a new sample.
 * Zeroes value nodes and prediction errors, frees routing data. */
void ipc_state_reset(IPCTrainState *state);

/* Destroy training state and free all memory. */
void ipc_state_destroy(IPCTrainState *state);

#endif /* IPC_STATE_H */
