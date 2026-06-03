/* local_feedback_train.h -- LocalFB (Local Feedback Alignment) training arm.
 *
 * Program 3, Phase 7, increment-4a.  2026-05-28.
 * D-613: QAT coverage seam — arm C of four-arm QAT apparatus.
 *
 * Citations (LOCKED, per dispatch):
 *   Nøkland 2016, arXiv 1609.01596 — Direct Feedback Alignment
 *   Lee et al. 2015, arXiv 1412.7525 — Difference Target Propagation
 *
 * Spec: programs/program_3_alt_grad_qat_100m/p7_qat_coverage_architecture.md
 *       §6 (LocalFB forward-compat), §3 (224-coverage proof), §8 (gate clauses).
 *
 * Naming note (P-D432-LFB-DEPRECATION-INVENTORY): the prefix "lfb_" is owned
 * by the Loss-Free-Balance router subsystem (router.c / train_config.h).
 * All symbols in this file use the "localfb_" or "local_feedback_" prefix to
 * avoid collisions.
 *
 * Increment-4a scope: forward pass + QAT coverage + B_l init.
 * Backward pass (local-target error propagation, dW accumulation, optimizer)
 * is increment-4b and is NOT implemented here.
 *
 * B_l out of QAT scope (§6, p7_qat_coverage_architecture.md):
 *   The L feedback matrices B_l are the alternative-gradient mechanism's own
 *   parameters, NOT the 224 in-scope forward weights (32 attention + 192 FFN).
 *   B_l are NEVER routed through op_matmul_qat.  covered_count must equal 224,
 *   not 224 + L*D*D/group_size.
 */

#ifndef LOCAL_FEEDBACK_TRAIN_H
#define LOCAL_FEEDBACK_TRAIN_H

#include "grad.h"
#include "hspa_model.h"
#include "ipc_train.h" /* for TrainStepResult */
#include "tensor.h"
#include "train_config.h"

#include <stdint.h>

/* ---- LocalFBState ---------------------------------------------------------
 * Holds:
 *   - L fixed-random feedback matrices B_l, shape [D, D] FP32.
 *     Initialized with a seeded Gaussian distribution (Nøkland 2016 §3.2
 *     minimum-viable form: fixed random, not learned).
 *   - Per-layer activation scratch for local targets (used in increment-4b).
 *
 * Ownership: all tensor data is malloc'd internally; localfb_state_destroy
 * frees everything.  Caller does not free any sub-pointers.
 * --------------------------------------------------------------------------*/
typedef struct LocalFBState LocalFBState;

/* Create a LocalFBState for the given model config.
 *
 * cfg:  model config (n_layers and d_model determine B_l shapes).
 * seed: deterministic PRNG seed for B_l initialization.
 *       Same seed → bit-identical B_l matrices across processes.
 *
 * Returns NULL on allocation failure. */
LocalFBState *localfb_state_create(const HSPAConfig *cfg, uint32_t seed);

/* Destroy the state and free all internal storage.
 * NULL is tolerated (no-op). */
void localfb_state_destroy(LocalFBState *state);

/* Return a pointer to B_l for layer `layer` (0-indexed).
 * The returned Tensor has shape [d_model, d_model], dtype FP32.
 * Valid until localfb_state_destroy is called.
 * Returns NULL if layer is out of range or state is NULL. */
const Tensor *localfb_state_get_B(const LocalFBState *state, int32_t layer);

/* Return a pointer to B_ffn_l for layer `layer` (0-indexed).
 * The returned Tensor has shape [d_model, d_ff], dtype FP32.
 * Used to project the global output error to the FFN feedback signal:
 *   delta_ff_l = delta_top @ B_ffn_l  →  shape [seq_len, d_ff].
 * B_ffn_l is NOT in QAT scope (same rule as B_l).
 * Valid until localfb_state_destroy is called.
 * Returns NULL if layer is out of range, state is NULL, or B_ffn not
 * yet allocated (e.g., called on a pre-increment-4b-fix state). */
const Tensor *localfb_state_get_B_ffn(const LocalFBState *state, int32_t layer);

/* ---- Training step --------------------------------------------------------
 * Run one complete LocalFB training step (forward-only in increment-4a).
 *
 * Steps (increment-4a):
 *   1. Create QATContext (capacity 256 > 224 in-scope weights).
 *   2. qat_context_begin_step: invalidate stale w_hat entries.
 *   3. embed → per-layer hspa_block_forward(..., qat) with MoE pre-registration
 *      loop (force all K experts' W_gate/W_up/W_down into w_hat cache so
 *      covered_count reaches 224 regardless of top-k routing).
 *   4. Capture per-layer representations into LocalFBState scratch.
 *   5. final RMSNorm → logits → cross-entropy loss.
 *   6. Record covered_count + cache_hits → destroy QATContext.
 *   7. Backward + optimizer: STUBBED (increment-4b).
 *   8. Return TrainStepResult.
 *
 * model:          HSPA model (weights modified in-place on last micro-batch
 *                 once backward is implemented in increment-4b; no-op here).
 * grads:          pre-allocated gradient accumulators (zeroed on micro_batch_idx==0).
 * adam:           Adam optimizer state (NULL → SGD; not used in increment-4a).
 * state:          LocalFBState (B_l matrices + activation scratch).
 * tokens:         input token IDs [seq_len].
 * targets:        target token IDs [seq_len] (next-token prediction).
 * seq_len:        number of tokens.
 * cfg:            model config.
 * tcfg:           training config (use_qat flag, grad_accum_steps).
 * step:           current global training step.
 * micro_batch_idx: index within gradient accumulation group [0..grad_accum_steps-1].
 * --------------------------------------------------------------------------*/
TrainStepResult localfb_train_step(HSPAModel *model, ModelGrad *grads, AdamState *adam,
                                   LocalFBState *state, const int32_t *tokens,
                                   const int32_t *targets, int32_t seq_len, const HSPAConfig *cfg,
                                   const TrainConfig *tcfg, int32_t step, int32_t micro_batch_idx);

/* ---- QAT coverage observability seam (Program 3, Phase 7, D-613) ---------
 * Returns the qat_context_covered_count() from the most recent call to
 * localfb_train_step().  Returns 0 when use_qat=false (disabled) or before
 * any step has run in this process.
 * §3.3 Clause 1 acceptance gate: must equal 224 for the full 8L/8K config.
 * Single-threaded (single-writer: localfb_train_step; single-reader: tests). */
int localfb_last_qat_covered_count(void);

/* Returns qat_context_cache_hits() from the most recent localfb_train_step().
 * §3.3 Clause 4 gate: must be > 0 when QAT enabled.
 * Cache hits > 0 proves the STE backward (increment-4b) will consume the same
 * step's cached w_hat rather than re-quantizing. */
int localfb_last_qat_cache_hits(void);

#endif /* LOCAL_FEEDBACK_TRAIN_H */
