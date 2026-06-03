/* loss.c -- Unified Variational Free Energy loss computation.
 *
 * L_VFE = L_pred + beta_balance * L_balance + lambda_out * L_output
 *
 * L_pred    = sum_l ||epsilon_l||^2 / (2 * sigma_l^2)
 * L_balance = sum_l (log K - H(g_l))
 * L_output  = (1/S) * sum_s CE(softmax(logits[s]), target[s])
 *
 * Uses op_cross_entropy from ops.h for CE loss (handles softmax internally).
 * Perplexity = exp(L_output).
 */

#include "loss.h"
#include "ops.h"
#include "tensor.h"

#include <math.h>
#include <stddef.h>

LossComponents loss_compute(const IPCTrainState *state,
                            const int32_t *targets,
                            const HSPAConfig *cfg,
                            const TrainConfig *tcfg) {
    LossComponents lc;
    lc.total      = 0.0f;
    lc.pred_error = 0.0f;
    lc.balance    = 0.0f;
    lc.lm         = 0.0f;
    lc.perplexity = 0.0f;

    if (!state || !targets || !cfg || !tcfg) {
        return lc;
    }

    int32_t L = cfg->n_layers;
    int32_t K = cfg->n_experts;
    int32_t S = state->seq_len;

    /* ----------------------------------------------------------------
     * L_pred = sum_l ||epsilon[l]||^2 / (2 * sigma_l^2)
     *
     * For each layer l, compute the squared L2 norm of the prediction
     * error tensor epsilon[l] (shape [seq_len, d_model]) and weight by
     * the precision 1/(2*sigma_l^2).
     * ---------------------------------------------------------------- */
    int32_t D = cfg->d_model;
    double l_pred = 0.0;
    double elem_norm = 1.0 / ((double)S * (double)D); /* per-element normalization */
    for (int32_t l = 0; l < L; l++) {
        const Tensor *eps = state->pe.errors[l];
        float sigma_sq = state->pe.precision[l];

        /* Compute mean(epsilon[l]^2) — per-element squared error. */
        size_t n = tensor_numel(eps);
        const float *data = (const float *)eps->data;
        double sq_norm = 0.0;
        for (size_t i = 0; i < n; i++) {
            sq_norm += (double)data[i] * (double)data[i];
        }

        /* Precision-weighted: mean(eps^2) / (2 * sigma^2) */
        if (sigma_sq > 0.0f) {
            l_pred += (sq_norm * elem_norm) / (2.0 * (double)sigma_sq);
        }
    }
    lc.pred_error = (float)l_pred;

    /* ----------------------------------------------------------------
     * L_balance = sum_l (log(K) - H(g_l))
     *
     * H(g_l) is the mean routing entropy across all tokens at layer l.
     * Stored in state->routing[l].mean_entropy.
     * log(K) is the maximum entropy (uniform distribution over K experts).
     * ---------------------------------------------------------------- */
    float log_K = logf((float)K);
    double l_balance = 0.0;
    for (int32_t l = 0; l < L; l++) {
        float H_l = state->routing[l].mean_entropy;
        l_balance += (double)(log_K - H_l);
    }
    lc.balance = (float)l_balance;

    /* ----------------------------------------------------------------
     * L_output = (1/S) * sum_s CE(softmax(logits[s]), target[s])
     *
     * Uses op_cross_entropy which returns the average CE over the seq.
     * It handles softmax internally.
     * ---------------------------------------------------------------- */
    float l_output = op_cross_entropy(state->logits, targets, S, NULL);
    lc.lm = l_output;

    /* ----------------------------------------------------------------
     * Perplexity = exp(L_output)
     *
     * Clamp to avoid overflow for very large losses.
     * ---------------------------------------------------------------- */
    if (l_output < 80.0f) {
        lc.perplexity = expf(l_output);
    } else {
        lc.perplexity = INFINITY;
    }

    /* ----------------------------------------------------------------
     * Total VFE loss
     *
     * D-031: When use_loss_free_balance is true, balance is computed for
     * monitoring but excluded from total loss. Router bias handles load
     * balancing without gradient interference.
     * ---------------------------------------------------------------- */
    float balance_weight = tcfg->use_loss_free_balance ? 0.0f : tcfg->beta_balance;
    lc.total = lc.pred_error
             + balance_weight * lc.balance
             + tcfg->lambda_out * lc.lm;

    return lc;
}
