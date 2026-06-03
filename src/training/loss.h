/* loss.h -- Unified Variational Free Energy loss.
 *
 * L_VFE = L_pred + beta_balance * L_balance + lambda_out * L_output
 *
 * L_pred    = sum_l ||epsilon_l||^2 / (2 * sigma_l^2)
 * L_balance = sum_l (log K - H(g_l))  (routing entropy gap)
 * L_output  = (1/S) * sum_s CE(softmax(logits[s]), target[s])
 */

#ifndef LOSS_H
#define LOSS_H

#include "hspa_config.h"
#include "ipc_state.h"
#include "train_config.h"

#include <stdint.h>

/* Individual loss components for logging. */
typedef struct {
    float total;        /* Combined VFE loss */
    float pred_error;   /* L_pred: precision-weighted prediction errors */
    float balance;      /* L_balance: routing entropy penalty */
    float lm;           /* L_output: cross-entropy language modeling loss */
    float perplexity;   /* exp(L_output) */
} LossComponents;

/* Compute the full VFE loss from converged iPC state.
 * targets: [seq_len] target token IDs.
 * Returns LossComponents with all terms. */
LossComponents loss_compute(const IPCTrainState *state,
                            const int32_t *targets,
                            const HSPAConfig *cfg,
                            const TrainConfig *tcfg);

#endif /* LOSS_H */
