/* ipc_train.h -- iPC training step: forward init, T inference iterations, weight updates.
 *
 * Core iPC algorithm (Salvatori et al. 2022, adapted for HSPA):
 *   1. Forward pass initializes value nodes at layer boundaries
 *   2. T inference iterations refine value nodes to minimize prediction errors
 *   3. At each iteration, weights are updated using LOCAL prediction errors
 *
 * First implementation uses:
 *   - Identity Jacobian approximation for value node updates
 *   - Full per-layer weight gradients (attention, FFN, router, norm)
 *   - Frozen KV caches and routing decisions during iterations
 *   - Fixed geometric precision schedule (Qi et al. 2025)
 */

#ifndef IPC_TRAIN_H
#define IPC_TRAIN_H

#include "grad.h"
#include "hspa_model.h"
#include "ipc_state.h"
#include "loss.h"
#include "train_config.h"

#include <stdint.h>

/* Result from one training step. */
typedef struct {
    LossComponents loss;        /* Loss breakdown */
    float          grad_norm;   /* Gradient L2 norm (before clipping) */
    float          mean_entropy;/* Mean routing entropy across all layers */
    float          vn_delta;    /* Max relative change in value nodes (convergence) */
} TrainStepResult;

/* Run one complete iPC training step (with gradient accumulation support).
 *
 * Steps:
 *   1. Forward pass to initialize value nodes (v[0]..v[L])
 *   2. Cache routing decisions and KV state (frozen for all T iterations)
 *   3. For t = 0..T-1:
 *      a. Compute prediction errors: epsilon[l] = v[l+1] - f_l(v[l])
 *      b. Compute output gradient: delta_top = E^T @ (softmax(logits) - target)
 *      c. Update value nodes using prediction errors (identity Jacobian)
 *      d. Compute per-layer weight gradients from prediction errors
 *      e. Accumulate gradients into ModelGrad
 *   4. Clip gradients and apply optimizer (Adam or SGD) with Depth-muP LR scaling
 *   5. Compute and return VFE loss
 *
 * model: the HSPA model (weights are modified in-place)
 * grads: pre-allocated gradient accumulators (zeroed internally on first micro-batch)
 * adam: Adam optimizer state (NULL to use SGD fallback)
 * state: pre-allocated iPC training state (reset internally)
 * tokens: input token IDs [seq_len]
 * targets: target token IDs [seq_len] (next-token prediction)
 * seq_len: number of tokens
 * cfg: model config
 * tcfg: training config (includes grad_accum_steps)
 * step: current global training step (for LR warmup / bias correction)
 * micro_batch_idx: index of current micro-batch [0..grad_accum_steps-1].
 *   - When micro_batch_idx == 0: zero gradients (Phase 0).
 *   - When micro_batch_idx == grad_accum_steps - 1: divide grads by
 *     grad_accum_steps and run Phase 3 (clip + apply optimizer).
 *   - Otherwise: accumulate gradients only (skip Phase 0 zero and Phase 3).
 */
TrainStepResult ipc_train_step(HSPAModel *model, ModelGrad *grads,
                               AdamState *adam,
                               IPCTrainState *state,
                               const int32_t *tokens, const int32_t *targets,
                               int32_t seq_len,
                               const HSPAConfig *cfg, const TrainConfig *tcfg,
                               int32_t step, int32_t micro_batch_idx);

/* Create a micro-experiment config (tiny HSPA for validation).
 * L=4, D=128, K=4, k=1, V=256, H=4, H_kv=2, D_h=32, D_ff=64.
 * ~1M total params. Fits easily in memory. */
HSPAConfig hspa_config_micro(void);

/* Small HSPA config for scale-up experiments.
 * L=6, D=256, K=8, k=2, V=4096, H=4, H_kv=2, D_h=64, D_ff=192.
 * Target: 8-12M total params. */
HSPAConfig hspa_config_small(void);

/* Medium HSPA config for scale-up experiments.
 * L=8, D=512, K=8, k=2, V=4096, H=8, H_kv=4, D_h=64, D_ff=384.
 * Target: 40-60M total params. */
HSPAConfig hspa_config_medium(void);

/* Phase C 100M HSPA config (data/engineering/phase_c_design.md §1.7).
 * L=8, D=768, K=8, k=2, V=32768, H=12, H_kv=6, D_h=64, D_ff=512.
 * Target: 119,649,024 params (~119.6M). */
HSPAConfig hspa_config_100m(void);

/* Dense-A compute-matched control for Program 2 (dense_vs_moe_sub100m).
 * data/engineering/dense_50m_control_design.md §1.3 + question.md locked form.
 * L=8, D=512, H=8, H_kv=4, D_h=64, D_ff=768, V=32768, K=1, k=1, S=512.
 * Target: 34,619,904 params (~34.62M) — matches MoE Rev-2 active path
 * (d_ff_dense = 2 * d_ff_moe = 2 * 384 = 768, FLOPs-matched for k=2 active).
 * Single-expert dense baseline; NO MoE features (entropy penalty, temp anneal,
 * loss-free balance, default MoE aux loss) may be combined with this config —
 * scale_experiment enforces this as a CLI guardrail. */
HSPAConfig hspa_config_dense_50m_a(void);

/* ---- QAT coverage observability seam (Program 3, Phase 7, D-613) ----
 * Returns the qat_context_covered_count() from the most recent call to
 * ipc_train_step().  Returns 0 when use_qat=false (disabled) or
 * before any step has run in this process.
 * Single-threaded (single-writer: ipc_train_step; single-reader: tests). */
int ipc_last_qat_covered_count(void);

/* Returns the qat_context_cache_hits() from the most recent call to
 * ipc_train_step().  Returns 0 when use_qat=false (disabled) or
 * before any step has run.  §3.3 Clause 4 gate: must be > 0 when enabled,
 * proving the STE backward consumed the same step's cached w_hat rather
 * than re-quantizing. */
int ipc_last_qat_cache_hits(void);

#endif /* IPC_TRAIN_H */
