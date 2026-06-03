/* grad.h -- Per-parameter gradient accumulators for iPC training.
 *
 * Mirrors the model weight structure exactly. Each weight tensor in
 * the model has a corresponding gradient tensor of the same shape.
 *
 * Gradients are accumulated across the iPC inference loop and then
 * applied via SGD or AdamW with Depth-muP learning rate scaling.
 *
 * AdamW maintains first-moment (m) and second-moment (v) estimates
 * for every model parameter, with bias-corrected updates and
 * decoupled weight decay (Loshchilov & Hutter, 2019).
 */

#ifndef GRAD_H
#define GRAD_H

#include "hspa_config.h"
#include "hspa_model.h"
#include "memory_pool.h"
#include "tensor.h"
#include "train_config.h"

#include <stdint.h>

/* Gradients for one attention layer. */
typedef struct {
    Tensor *dW_q;   /* [d_model, n_heads * head_dim] */
    Tensor *dW_k;   /* [d_model, n_kv_heads * head_dim] */
    Tensor *dW_v;   /* [d_model, n_kv_heads * head_dim] */
    Tensor *dW_o;   /* [n_heads * head_dim, d_model] */
} AttentionGrad;

/* Gradients for one expert FFN. */
typedef struct {
    Tensor *dW_gate;/* [d_model, d_ff] */
    Tensor *dW_up;  /* [d_model, d_ff] */
    Tensor *dW_down;/* [d_ff, d_model] */
} ExpertFFNGrad;

/* Gradients for the FEP router. */
typedef struct {
    Tensor *dW_mu;      /* [d_model, n_experts] */
    Tensor *dW_sigma;   /* [d_model, n_experts] */
} RouterGrad;

/* Gradients for RMSNorm. */
typedef struct {
    Tensor *dweight;    /* [d_model] */
} RMSNormGrad;

/* Gradients for one HSPA block. */
typedef struct {
    RMSNormGrad   attn_norm_grad;
    AttentionGrad attn_grad;
    RMSNormGrad   ffn_norm_grad;
    RouterGrad    router_grad;
    ExpertFFNGrad *expert_grads; /* [n_experts] -- all experts have slots */
    int32_t       n_experts;
} BlockGrad;

/* Gradients for the embedding (weight-tied with lm_head). */
typedef struct {
    Tensor *dweight;    /* [vocab_size, d_model] */
} EmbeddingGrad;

/* Complete model gradients. */
typedef struct {
    EmbeddingGrad  embed_grad;
    BlockGrad     *block_grads;     /* [n_layers] */
    RMSNormGrad    final_norm_grad;
    int32_t        n_layers;
    MemoryPool    *grad_pool;       /* Dedicated pool for all gradient tensors */
} ModelGrad;

/* Create gradient accumulators matching a model config.
 * All gradients initialized to zero. Returns NULL on failure. */
ModelGrad *grad_create(const HSPAConfig *cfg);

/* Zero all gradient accumulators (call before each training step). */
void grad_zero(ModelGrad *grads);

/* Compute global gradient L2 norm. */
float grad_global_norm(const ModelGrad *grads);

/* Clip gradients in-place to max_norm. */
void grad_clip(ModelGrad *grads, float max_norm);

/* Apply gradients to model weights via SGD with Depth-muP LR scaling.
 * precision: [n_layers] per-layer sigma^2 values for LR scaling.
 * step: current step (for LR warmup). */
void grad_apply_sgd(HSPAModel *model, const ModelGrad *grads,
                    const HSPAConfig *cfg, const TrainConfig *tcfg,
                    const float *precision, int32_t step);

/* Destroy all gradient accumulators. */
void grad_destroy(ModelGrad *grads);

/* ========================================================================
 * AdamW Optimizer State
 * ======================================================================== */

/* Per-tensor Adam moment state. */
typedef struct {
    Tensor *m;  /* First moment estimate (mean of gradients) */
    Tensor *v;  /* Second moment estimate (mean of squared gradients) */
} AdamMoment;

/* Complete Adam state matching model structure. */
typedef struct {
    /* Embedding */
    AdamMoment embed_m;

    /* Per-layer */
    struct AdamLayerState {
        AdamMoment attn_norm_m;
        AdamMoment attn_q_m, attn_k_m, attn_v_m, attn_o_m;
        AdamMoment ffn_norm_m;
        AdamMoment router_mu_m, router_sigma_m;
        AdamMoment *expert_gate_m;  /* [n_experts] */
        AdamMoment *expert_up_m;    /* [n_experts] */
        AdamMoment *expert_down_m;  /* [n_experts] */
        int32_t n_experts;
    } *layers;  /* [n_layers] */

    /* Final norm */
    AdamMoment final_norm_m;

    int32_t n_layers;
    MemoryPool *adam_pool;  /* Dedicated pool for moment tensors */
} AdamState;

/* Create Adam state matching a model config.
 * All moments initialized to zero. Returns NULL on failure. */
AdamState *adam_create(const HSPAConfig *cfg);

/* Apply AdamW update to all model weights.
 * step: 1-indexed training step (for bias correction). */
void grad_apply_adam(HSPAModel *model, const ModelGrad *grads,
                     AdamState *adam, const HSPAConfig *cfg,
                     const TrainConfig *tcfg, int32_t step);

/* Destroy Adam state and free all moment tensors. */
void adam_destroy(AdamState *adam);

#endif /* GRAD_H */
