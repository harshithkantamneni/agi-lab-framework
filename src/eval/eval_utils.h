/* eval_utils.h -- Evaluation utilities for HSPA model scoring.
 *
 * Provides log-probability computation from model logits, used for:
 *   - Perplexity measurement (average NLL over tokens)
 *   - Multiple-choice scoring (sum log-prob of each choice's tokens)
 *
 * The core function compute_logprob operates on the output of
 * hspa_model_forward: logits[i] predicts token[i+1].
 */

#ifndef EVAL_UTILS_H
#define EVAL_UTILS_H

#include "tensor.h"
#include <stdint.h>

/* Compute the sum of log-probabilities for tokens[start+1..seq_len-1].
 *
 * For each position i in [start, seq_len-2]:
 *   - logits row i (logits->data + i * vocab_size) predicts tokens[i+1]
 *   - Computes log_softmax(logits[i])[tokens[i+1]]
 *   - Uses numerically stable computation (subtract max first)
 *
 * logits:   [seq_len, vocab_size] tensor from hspa_model_forward
 * tokens:   input token IDs (length seq_len)
 * seq_len:  number of tokens
 * start:    first position to include (log-prob of token[start+1])
 *
 * Returns: sum of log-probabilities (always <= 0). */
float compute_logprob(const Tensor *logits, const int32_t *tokens,
                      int32_t seq_len, int32_t start);

#endif /* EVAL_UTILS_H */
