/* backprop_train.h -- Standard backpropagation training step for HSPA.
 *
 * Drop-in replacement for ipc_train_step that uses standard backprop
 * (forward + backward pass) instead of iterative Predictive Coding.
 *
 * Used for A/B comparison: iPC vs backprop at micro scale (1M params)
 * to quantify the quality gap documented in the literature.
 *
 * Key differences from iPC:
 *   - Single forward+backward pass (no T inference iterations)
 *   - EXACT attention gradients (iPC uses approximate dW_q/dW_k/dW_v)
 *   - No value nodes or prediction errors
 *   - Same Phase 3 optimizer (Adam/SGD with Depth-muP)
 *   - Same gradient structures (ModelGrad, AdamState)
 */

#ifndef BACKPROP_TRAIN_H
#define BACKPROP_TRAIN_H

#include "grad.h"
#include "hspa_model.h"
#include "ipc_train.h"  /* for TrainStepResult, config functions */
#include "train_config.h"

#include <stdint.h>

/* Backprop training step -- drop-in replacement for ipc_train_step.
 * Same signature minus IPCTrainState (not needed for backprop).
 *
 * Steps:
 *   1. Forward pass: embed -> L blocks -> norm -> logits (saving activations)
 *   2. Cross-entropy loss + dlogits
 *   3. Backward pass: chain rule from output back through all layers
 *   4. Clip gradients and apply optimizer (Adam or SGD) with Depth-muP
 *
 * model: the HSPA model (weights modified in-place on last micro-batch)
 * grads: pre-allocated gradient accumulators (zeroed on micro_batch_idx==0)
 * adam: Adam optimizer state (NULL to use SGD fallback)
 * tokens: input token IDs [seq_len]
 * targets: target token IDs [seq_len] (next-token prediction)
 * seq_len: number of tokens
 * cfg: model config
 * tcfg: training config
 * step: current global training step (for LR warmup / bias correction)
 * micro_batch_idx: index within gradient accumulation group
 */
TrainStepResult backprop_train_step(HSPAModel *model, ModelGrad *grads,
                                    AdamState *adam,
                                    const int32_t *tokens, const int32_t *targets,
                                    int32_t seq_len,
                                    const HSPAConfig *cfg, const TrainConfig *tcfg,
                                    int32_t step, int32_t micro_batch_idx);

/* ---- QAT coverage observability seam (Program 3, Phase 7, D-613) ----
 * Returns the qat_context_covered_count() from the most recent call to
 * backprop_train_step().  Returns 0 when use_qat=false (disabled) or
 * before any step has run in this process.
 * Single-threaded (single-writer: backprop_train_step; single-reader: tests). */
int backprop_last_qat_covered_count(void);

/* ---- Cycle 29 Rev-2: Entropy-penalty + τ-anneal math helpers ----
 * Externally visible so unit tests (tests/test_entropy_penalty.c) can
 * exercise the math without spinning up a full model. */

/* Shannon entropy H = -sum_j p[j] * log(p[j] + 1e-8). K experts. */
float entropy_penalty_shannon_H(const float *p, int32_t K);

/* Per-logit gradient of the one-sided-quadratic hinge penalty.
 *   d_logit_H[j] = 2 * beta_H * max(0, H_target - H) * p[j] * (log p[j] + H)
 * Zero-fills out_grad when H >= H_target (Cycle-13 immune). */
void entropy_penalty_grad_logit(const float *p, int32_t K,
                                float beta_H, float H_target,
                                float *out_grad);

/* Cosine τ schedule: τ_max → τ_min over S_anneal steps; τ_min after.
 * τ(0)=τ_max, τ(S/2)=midpoint, τ(S)=τ_min, τ(step > S)=τ_min. */
float compute_router_temperature(float tau_max, float tau_min,
                                 int32_t step, int32_t S_anneal);

/* Linear β_H warmup: 0 at step 0, β_H_peak at step >= warmup_steps. */
float compute_beta_h_warmup(float beta_h_peak, int32_t step,
                            int32_t warmup_steps);

#endif /* BACKPROP_TRAIN_H */
