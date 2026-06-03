/* train_config.h -- All hyperparameters for iPC training.
 *
 * Separates training config from model config (hspa_config.h).
 * Model config defines architecture; this defines how to train it.
 *
 * Header-only: default configs are static inline to avoid linking issues.
 */

#ifndef TRAIN_CONFIG_H
#define TRAIN_CONFIG_H

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    /* --- iPC inference loop --- */
    int32_t T;              /* iPC inference iterations per sample: 5 */
    float   ipc_lr;         /* Value node update rate (alpha_v): 0.1 */

    /* --- Weight update (SGD) --- */
    float   base_lr;        /* Base learning rate: 0.001 */
    int32_t lr_warmup_steps;/* Linear warmup steps: 100 */
    float   lr_min;         /* Minimum learning rate: 1e-5 */

    /* --- VFE loss coefficients --- */
    float   beta_balance;   /* Weight for load-balancing entropy: 0.01 */
    float   balance_h_floor;/* Floor-only mode: beta active only when H < floor. 0=always */
    float   router_pred_scale; /* Scale for prediction-error router gradient: 0.05 */
    float   lambda_out;     /* Output loss coupling strength: 1.0 */

    /* --- Precision schedule (Qi et al. 2025) --- */
    float   sigma_top;      /* sigma^2 at top layer (near output): 0.01 */
    float   sigma_bottom;   /* sigma^2 at bottom layer (near data): 1.0 */

    /* --- Optimizer selection --- */
    bool    use_adam;        /* If true, use AdamW; if false, use SGD */
    float   adam_beta1;      /* First moment decay: 0.9 */
    float   adam_beta2;      /* Second moment decay: 0.999 */
    float   adam_eps;        /* Numerical stability: 1e-8 */
    float   weight_decay;    /* AdamW decoupled weight decay: 0.01 */

    /* --- Gradient clipping --- */
    float   grad_clip_norm; /* Max gradient L2 norm: 1.0 */

    /* --- Gradient accumulation --- */
    int32_t grad_accum_steps; /* Micro-batches per effective batch: 1 (no accumulation) */

    /* --- muP weight decay scaling --- */
    int32_t mup_base_width; /* Reference hidden dim for wd scaling: 128 (micro) */

    /* --- Progressive sparsification (Sigma-MoE-Tiny, arXiv 2512.16248) --- */
    int32_t n_active_start;    /* Starting top-K (e.g., 4 for top-4-of-8). 0 = disabled */
    int32_t sparsify_steps;    /* Steps over which to anneal from n_active_start to cfg.n_active */

    /* --- Batching --- */
    int32_t batch_size;     /* Tokens per batch: 1 (micro-experiment) */
    int32_t max_steps;      /* Maximum training steps */

    /* --- Logging / checkpointing --- */
    int32_t log_every;      /* Print loss every N steps: 10 */
    int32_t checkpoint_every;/* Save every N steps: 100 */

    /* --- Entropy target (D-027, added after checkpoint v1) ---
     * Placed at end of struct for checkpoint backward compatibility. */
    float   balance_h_target;/* Entropy target: proportional push toward this H. 0=disabled */

    /* --- Loss-Free Balancing (D-031, arXiv 2408.15664) ---
     * When enabled: entropy gradient is DISABLED, balance loss excluded from
     * total loss. Router bias mechanism (router.c) handles load balancing
     * without gradient interference. Three-regime controller becomes
     * monitoring-only (entropy logged but no gradient generated). */
    bool    use_loss_free_balance; /* If true, disable entropy gradient + balance loss */
    float   lfb_bias_step;  /* Per-expert bias step size (router bias_step_size). 0=use default */
    float   lfb_ema_rate;   /* EMA decay rate for expert load tracking. 0=use default */

    /* --- Plan B: Default MoE / Dense Backprop (D-Plan-B, arXiv 2504.12463) ---
     * When enabled under --backprop, the router receives DENSE gradient signal
     * for all K experts (not just the top-k selected) by substituting each
     * unselected expert's output with its historical EMA. Fixes the entropy
     * collapse observed in Cycle 26 (killed_ideas.md D-094). Stacks on LFB.
     * Memory cost: L*K*d_model floats (e.g. 0.125 MB at 50M). */
    bool    use_default_moe;        /* If true, enable dense router backward via EMA proxy */
    float   default_moe_alpha;      /* EMA update rate; 0.01 (paper default) */
    float   default_moe_sigma_init; /* EMA Gaussian init std dev; 0.01 (NOT zero, §9-R2) */

    /* --- Entropy penalty + cosine τ-anneal (Cycle 29 Rev-2 escalation) ---
     * Spec: data/engineering/entropy_penalty_temp_anneal_design.md §Revision 2.
     * One-sided quadratic hinge L_H = β_H · max(0, H_target - H)² + cosine
     * router temperature anneal τ: τ_max → 1.0 over S_anneal steps.
     * Activates below H_target; zero gradient above (Cycle-13 immune). */
    bool    use_entropy_penalty;         /* Master flag for the §1 hinge penalty */
    float   entropy_beta_h;              /* β_H peak coefficient; Rev-2 default 0.15 (HARD CAP 0.30) */
    float   entropy_h_target;            /* H_target hinge floor; Rev-2 default 0.90 (HARD CAP 1.0) */
    int32_t entropy_warmup_steps;        /* Suppress penalty before this step; Rev-2 default 0 */
    int32_t entropy_beta_h_warmup_steps; /* Linear ramp β_H: 0 → peak; Rev-2 default 200 */

    bool    use_temp_anneal;             /* Master flag for the §2 cosine τ-anneal */
    float   temp_anneal_max;             /* τ_max at step 0; Rev-2 default 1.4 (HARD CAP 1.5) */
    float   temp_anneal_min;             /* τ_min at step >= S_anneal; default 1.0 */
    int32_t temp_anneal_steps;           /* S_anneal cosine span; Rev-2 default 500 */
    bool    use_temp_anneal_restoration; /* Rev-2 §R2.5 restoration band: pause decay if H<0.55 */

    /* --- Quantization-Aware Training (Program 3 Phase 7, D-613) ---
     * When true: Arm A (backprop) applies fake-quantize-INT4 to all in-scope
     * weight tensors (32 attention + 192 FFN = 224 total) via QATContext.
     * Placed at end for checkpoint backward compatibility. */
    bool    use_qat;          /* If true, enable 4-bit QAT for in-scope weights (Arm A) */
    int     qat_group_size;   /* INT4 quantization group size; default 128.
                               * Threaded through to qat_context_create in all three
                               * training arms (backprop, iPC, LocalFB).  Replaces the
                               * hardcoded 128 in each arm's QATContext construction call.
                               * Program-3 binding spec: 128 for all 21 factorial cells. */
} TrainConfig;

/* Default training config for micro-experiment. */
static inline TrainConfig train_config_micro(void) {
    return (TrainConfig){
        .T               = 5,
        .ipc_lr          = 0.1f,
        .base_lr         = 0.001f,
        .lr_warmup_steps = 100,
        .lr_min          = 1e-5f,
        .beta_balance    = 0.01f,
        .balance_h_floor = 0.0f,
        .router_pred_scale = 0.05f,
        .lambda_out      = 1.0f,
        .sigma_top       = 0.01f,
        .sigma_bottom    = 1.0f,
        .use_adam         = true,
        .adam_beta1      = 0.9f,
        .adam_beta2      = 0.999f,
        .adam_eps         = 1e-8f,
        .weight_decay    = 0.01f,
        .grad_clip_norm  = 1.0f,
        .grad_accum_steps = 1,
        .mup_base_width  = 128,
        .n_active_start  = 0,
        .sparsify_steps  = 0,
        .batch_size      = 1,
        .max_steps       = 1000,
        .log_every       = 10,
        .checkpoint_every = 100,
        .balance_h_target = 0.0f,
        .use_loss_free_balance = false,
        .lfb_bias_step   = 0.0f,
        .lfb_ema_rate    = 0.0f,
        .use_default_moe = false,
        .default_moe_alpha = 0.01f,
        .default_moe_sigma_init = 0.01f,
        .use_entropy_penalty         = false,
        .entropy_beta_h              = 0.15f,
        .entropy_h_target            = 0.90f,
        .entropy_warmup_steps        = 0,
        .entropy_beta_h_warmup_steps = 200,
        .use_temp_anneal             = false,
        .temp_anneal_max             = 1.4f,
        .temp_anneal_min             = 1.0f,
        .temp_anneal_steps           = 500,
        .use_temp_anneal_restoration = true,
        .use_qat             = false,
        .qat_group_size      = 128,
    };
}

#endif /* TRAIN_CONFIG_H */
