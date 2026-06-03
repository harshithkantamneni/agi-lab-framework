/* hspa_config.c -- HSPA v1 default configuration.
 *
 * Derives d_ff from the total parameter budget:
 *   total_params = 16B
 *   shared_params = embedding + attention + norms (per layer)
 *   expert_params = (total_params - shared_params) / (n_experts * n_layers)
 *   d_ff = expert_params / (3 * d_model)  [3 matrices: W_gate, W_up, W_down]
 *
 * With n_experts=16, n_layers=32, d_model=4096:
 *   Shared per layer: attn ~67M (W_q:16M + W_k:4M + W_v:4M + W_o:16M + norms + router)
 *   Total shared: 32 * ~67M + embedding(50k*4096) ~ 2.35B
 *   Expert budget: (16B - 2.35B) / (16 * 32) = ~26.7M per expert
 *   d_ff = 26.7M / (3 * 4096) ~ 2171
 *
 *   Using engineering plan value: d_ff = 1850 (conservative, from 28.2M/expert).
 */

#include "hspa_config.h"

HSPAConfig hspa_config_default(void) {
    HSPAConfig cfg;

    cfg.d_model        = 4096;
    cfg.n_layers       = 32;
    cfg.n_heads        = 32;
    cfg.n_kv_heads     = 8;
    cfg.head_dim       = 128;   /* d_model / n_heads = 4096 / 32 */
    cfg.n_experts      = 16;    /* CONFIGURABLE per D-004 */
    cfg.n_active       = 2;     /* top-2 routing */
    cfg.d_ff           = 1850;  /* derived: ~28.2M / (3 * 4096), per engineering plan */
    cfg.vocab_size     = 50000;
    cfg.max_seq_len    = 2048;
    cfg.ipc_iterations = 5;
    cfg.rms_norm_eps   = 1e-6f;
    cfg.storage_dtype  = DTYPE_INT4;
    cfg.compute_dtype  = DTYPE_FP32; /* Start with FP32, optimize to FP16 later */

    return cfg;
}
