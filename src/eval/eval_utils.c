/* eval_utils.c -- Log-probability computation for HSPA evaluation.
 *
 * Implements numerically stable log-softmax:
 *   log_softmax(x)[j] = x[j] - max(x) - log(sum(exp(x - max(x))))
 *
 * This avoids overflow in exp() by shifting all values by -max.
 */

#include "eval_utils.h"

#include <float.h>
#include <math.h>

float compute_logprob(const Tensor *logits, const int32_t *tokens,
                      int32_t seq_len, int32_t start) {
    if (!logits || !tokens || seq_len < 2 || start < 0 ||
        start >= seq_len - 1) {
        return 0.0f;
    }

    /* logits shape: [seq_len, vocab_size] */
    int32_t vocab_size = logits->shape[1];
    float *data = (float *)logits->data;

    float sum_logprob = 0.0f;

    for (int32_t i = start; i < seq_len - 1; i++) {
        float *row = data + (int64_t)i * vocab_size;
        int32_t target = tokens[i + 1];

        /* Safety: clamp target to valid range */
        if (target < 0 || target >= vocab_size) {
            continue;
        }

        /* Step 1: find max for numerical stability */
        float max_val = -FLT_MAX;
        for (int32_t v = 0; v < vocab_size; v++) {
            if (row[v] > max_val) {
                max_val = row[v];
            }
        }

        /* Step 2: compute log(sum(exp(x - max))) */
        double sum_exp = 0.0;
        for (int32_t v = 0; v < vocab_size; v++) {
            sum_exp += exp((double)(row[v] - max_val));
        }
        float log_sum_exp = max_val + (float)log(sum_exp);

        /* Step 3: log_softmax[target] = logits[target] - log_sum_exp */
        float log_prob = row[target] - log_sum_exp;
        sum_logprob += log_prob;
    }

    return sum_logprob;
}
