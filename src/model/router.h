/* router.h -- FEP (Free Energy Principle) based expert router.
 *
 * Routes tokens to experts using active inference principles:
 *   - W_mu produces mean routing logits (standard gating)
 *   - W_sigma produces uncertainty estimates (epistemic exploration)
 *   - During training: sample from N(mu, sigma) for exploration
 *   - During inference: use mu directly (exploit mode)
 *
 * The router also computes routing entropy H(g) for the VFE loss:
 *   L_balance = beta_1 * (log K - H(g))
 * This replaces the traditional auxiliary load-balancing loss.
 *
 * Source: H-N1 (Neuroscience team FEP routing design)
 */

#ifndef ROUTER_H
#define ROUTER_H

#include "hspa_config.h"
#include "memory_pool.h"
#include "tensor.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    Tensor *W_mu;       /* Mean routing logits: [d_model, n_experts] */
    Tensor *W_sigma;    /* Uncertainty (log-sigma): [d_model, n_experts] */
    float sigma_prior;  /* Prior std dev for KL regularization: 1.0 */
    /* Auxiliary-loss-free load balancing (DeepSeek-style, arXiv 2408.15664) */
    float *expert_bias;     /* [n_experts] — added to routing logits before top-k */
    float *expert_load_ema; /* [n_experts] — EMA of expert load fraction */
    float bias_update_rate; /* EMA decay rate, e.g. 0.01 */
    float bias_step_size;   /* How much to adjust bias per update, e.g. 0.001 */
    int32_t n_experts;      /* Cached expert count for bias array sizing */
    /* Plan B: Default MoE / Dense Backprop (arXiv 2504.12463, D-Plan-B).
     * EMA of each expert's output; used to substitute for unselected
     * experts during the router backward pass. NULL when disabled. */
    float *default_moe_ema;     /* [n_experts * d_model] row-major, NULL=off */
    int32_t default_moe_dim;    /* cached d_model for sizing / save-load */
    float   default_moe_alpha;  /* EMA update rate; 0 when disabled */
    /* Cycle 29 Rev-2: router temperature τ applied as logit divisor in the
     * dense-backward softmax. 1.0 = no-op; training loop sets per-step from
     * cosine schedule (compute_router_temperature). Not persisted (derived
     * from step); no checkpoint format change. */
    float   router_temperature; /* [1..τ_max]; 1.0 = no-op */
} FEPRouter;

typedef struct {
    int32_t *expert_ids;    /* Selected expert indices: [n_active] */
    float *expert_weights;  /* Routing weights (softmax normalized): [n_active] */
    float entropy;          /* H(g) -- routing entropy for VFE loss */
} RoutingDecision;

/* Create a FEP router, allocating weights from `pool`.
 * Returns NULL on failure. */
FEPRouter *router_create(MemoryPool *pool, const HSPAConfig *cfg);

/* Route one token to its top-k experts.
 * x: [d_model] input hidden state for one token
 * cfg: model config (for n_experts, n_active)
 * training: if true, sample from N(mu, sigma); if false, use mu only
 * Returns a RoutingDecision with selected experts and weights.
 * NOTE: caller must free expert_ids and expert_weights when done. */
RoutingDecision router_forward(const FEPRouter *router, const Tensor *x,
                               const HSPAConfig *cfg, bool training);

/* Update expert biases based on observed load (auxiliary-loss-free balancing).
 * expert_counts: [n_experts] -- number of tokens routed to each expert this batch
 * total_tokens: total routing decisions in batch (e.g. seq_len * batch_size)
 * n_experts: number of experts
 * Call after each training step. Does NOT inject any gradient signal. */
void router_update_bias(FEPRouter *router, const int32_t *expert_counts,
                        int32_t total_tokens, int32_t n_experts);

/* Plan B: Allocate + init EMA tensor for Default MoE.
 * Initializes `default_moe_ema[n_experts * d_model]` to N(0, sigma_init).
 * sigma_init MUST be > 0 (paper §5 / design §9-R2: zero-init causes
 * degenerate re-inforcing loop for never-selected experts).
 * seed: deterministic RNG seed (pseudo-random Gaussian via Box-Muller).
 * Returns 0 on success, -1 on failure or already initialized.
 * Safe to call multiple times only after router_destroy_default_moe(). */
int router_init_default_moe(FEPRouter *router, int32_t d_model,
                            float alpha, float sigma_init, uint32_t seed);

/* Free default_moe_ema. Sets pointer to NULL. No-op if already freed. */
void router_destroy_default_moe(FEPRouter *router);

/* Destroy the router struct. */
void router_destroy(FEPRouter *router);

#endif /* ROUTER_H */
