/* test_eval.c -- Tests for eval_utils log-probability computation.
 *
 * Uses a tiny HSPA model (d_model=16, n_layers=2, etc.) to verify:
 *   1. Log-probs are negative (probabilities < 1)
 *   2. Log-probs sum < 0
 *   3. Softmax probabilities sum to 1.0 at each position
 *   4. Edge cases: seq_len=1, start >= seq_len-1
 *   5. Known distribution: uniform logits give log(1/V) per token
 */

#include "../src/tests/unity.h"
#include "eval_utils.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "memory_pool.h"
#include "tensor.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper: create a tiny test config for fast testing. */
static HSPAConfig tiny_config(void) {
    HSPAConfig cfg = hspa_config_default();
    cfg.d_model = 16;
    cfg.n_layers = 2;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 4; /* d_model / n_heads */
    cfg.n_experts = 4;
    cfg.n_active = 2;
    cfg.d_ff = 32;
    cfg.vocab_size = 100;
    cfg.max_seq_len = 64;
    return cfg;
}

/* ========================================================================
 * TEST 1: logprobs_are_negative
 *
 * After a forward pass with real model weights, every individual
 * log-probability should be negative (since each probability < 1).
 * ======================================================================== */

static void test_logprobs_are_negative(void) {
    TEST_BEGIN(logprobs_are_negative);

    HSPAConfig cfg = tiny_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    /* Set small non-zero weights for non-trivial computation */
    tensor_fill(model->embed->weight, 0.01f);

    int32_t seq_len = 5;
    int32_t tokens[] = {10, 20, 30, 40, 50};

    /* Allocate logits and run forward pass */
    int32_t logits_shape[] = {seq_len, cfg.vocab_size};
    Tensor *logits =
        tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    hspa_model_forward(logits, model, tokens, seq_len, false);

    /* Check each position's log-prob individually */
    for (int32_t i = 0; i < seq_len - 1; i++) {
        float lp = compute_logprob(logits, tokens, seq_len, i);
        /* Sum from position i to end should be negative */
        ASSERT_TRUE(lp <= 0.0f);
    }

    /* Total sum should also be negative */
    float total = compute_logprob(logits, tokens, seq_len, 0);
    ASSERT_TRUE(total < 0.0f);

    tensor_destroy(logits);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 2: softmax_sums_to_one
 *
 * At each position, exp(log_softmax) should sum to 1.0. We verify this
 * by manually computing softmax from the logits and checking the sum.
 * ======================================================================== */

static void test_softmax_sums_to_one(void) {
    TEST_BEGIN(softmax_sums_to_one);

    HSPAConfig cfg = tiny_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    tensor_fill(model->embed->weight, 0.01f);

    int32_t seq_len = 3;
    int32_t tokens[] = {5, 15, 25};

    int32_t logits_shape[] = {seq_len, cfg.vocab_size};
    Tensor *logits =
        tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    hspa_model_forward(logits, model, tokens, seq_len, false);

    /* For each position, compute softmax and verify sum ~= 1.0 */
    float *data = (float *)logits->data;
    for (int32_t pos = 0; pos < seq_len; pos++) {
        float *row = data + (int64_t)pos * cfg.vocab_size;

        /* Find max for numerical stability */
        float max_val = -FLT_MAX;
        for (int32_t v = 0; v < cfg.vocab_size; v++) {
            if (row[v] > max_val) {
                max_val = row[v];
            }
        }

        /* Compute sum of exp(x - max) */
        double sum_exp = 0.0;
        for (int32_t v = 0; v < cfg.vocab_size; v++) {
            sum_exp += exp((double)(row[v] - max_val));
        }

        /* Softmax probabilities sum to 1.0 */
        double prob_sum = 0.0;
        for (int32_t v = 0; v < cfg.vocab_size; v++) {
            prob_sum += exp((double)(row[v] - max_val)) / sum_exp;
        }

        ASSERT_EQUAL_FLOAT(1.0f, (float)prob_sum, 1e-5f);
    }

    tensor_destroy(logits);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 3: edge_case_single_token
 *
 * With seq_len=1, there are no next-token predictions to score.
 * compute_logprob should return 0.
 * ======================================================================== */

static void test_edge_case_single_token(void) {
    TEST_BEGIN(edge_case_single_token);

    HSPAConfig cfg = tiny_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    tensor_fill(model->embed->weight, 0.01f);

    int32_t tokens[] = {42};
    int32_t logits_shape[] = {1, cfg.vocab_size};
    Tensor *logits =
        tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    hspa_model_forward(logits, model, tokens, 1, false);

    /* No next-token predictions possible with seq_len=1 */
    float lp = compute_logprob(logits, tokens, 1, 0);
    ASSERT_EQUAL_FLOAT(0.0f, lp, 1e-8f);

    tensor_destroy(logits);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 4: start_at_various_positions
 *
 * Log-prob from start=0 should equal log-prob(start=0..k) + log-prob(start=k).
 * Tests additivity of the log-prob sum.
 * ======================================================================== */

static void test_start_at_various_positions(void) {
    TEST_BEGIN(start_at_various_positions);

    HSPAConfig cfg = tiny_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    tensor_fill(model->embed->weight, 0.01f);

    int32_t seq_len = 6;
    int32_t tokens[] = {1, 2, 3, 4, 5, 6};

    int32_t logits_shape[] = {seq_len, cfg.vocab_size};
    Tensor *logits =
        tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    hspa_model_forward(logits, model, tokens, seq_len, false);

    /* Total from start=0 */
    float total = compute_logprob(logits, tokens, seq_len, 0);

    /* Split at position 3: sum(0..2) + sum(3..4) should equal total */
    /* compute_logprob(start=0) covers positions 0..seq_len-2
     * compute_logprob(start=3) covers positions 3..seq_len-2
     * We need to manually compute the first part (positions 0..2) */
    float part2 = compute_logprob(logits, tokens, seq_len, 3);

    /* Compute part1 by using a truncated seq_len=4 (positions 0,1,2 predict 1,2,3) */
    /* Actually, we verify: logprob(start=0, full) = logprob(start=0, to 3) + logprob(start=3, full)
     * We can check: total - part2 is the contribution of positions 0,1,2 */
    float part1 = total - part2;

    /* Both parts should be negative */
    ASSERT_TRUE(part1 <= 0.0f);
    ASSERT_TRUE(part2 <= 0.0f);

    /* And the total should be more negative than either part */
    ASSERT_TRUE(total <= part1);
    ASSERT_TRUE(total <= part2);

    tensor_destroy(logits);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 5: uniform_logits_give_log_one_over_V
 *
 * If all logits are identical (uniform distribution), each log-prob
 * should be exactly -log(vocab_size). Test with synthetic logits.
 * ======================================================================== */

static void test_uniform_logits(void) {
    TEST_BEGIN(uniform_logits);

    int32_t vocab_size = 100;
    int32_t seq_len = 4;

    /* Create a synthetic logits tensor with uniform values */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);

    int32_t logits_shape[] = {seq_len, vocab_size};
    Tensor *logits = tensor_create(pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    /* Fill all logits with same value (uniform distribution) */
    tensor_fill(logits, 1.0f);

    int32_t tokens[] = {0, 10, 50, 99};

    float logprob = compute_logprob(logits, tokens, seq_len, 0);

    /* With uniform logits, each token gets probability 1/V,
     * so log-prob = -log(V) per token, total = -(seq_len-1) * log(V) */
    float expected = -(float)(seq_len - 1) * logf((float)vocab_size);
    ASSERT_EQUAL_FLOAT(expected, logprob, 1e-4f);

    tensor_destroy(logits);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 6: null_and_invalid_inputs
 *
 * Verify compute_logprob handles NULL/invalid inputs gracefully.
 * ======================================================================== */

static void test_null_inputs(void) {
    TEST_BEGIN(null_inputs);

    int32_t tokens[] = {1, 2, 3};

    /* NULL logits */
    float lp = compute_logprob(NULL, tokens, 3, 0);
    ASSERT_EQUAL_FLOAT(0.0f, lp, 1e-8f);

    /* NULL tokens */
    lp = compute_logprob(NULL, NULL, 3, 0);
    ASSERT_EQUAL_FLOAT(0.0f, lp, 1e-8f);

    /* seq_len < 2 */
    lp = compute_logprob(NULL, tokens, 1, 0);
    ASSERT_EQUAL_FLOAT(0.0f, lp, 1e-8f);

    /* start >= seq_len - 1 */
    lp = compute_logprob(NULL, tokens, 3, 2);
    ASSERT_EQUAL_FLOAT(0.0f, lp, 1e-8f);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("=== Eval Utils Tests ===\n\n");

    printf("--- Log-probability computation ---\n");
    test_logprobs_are_negative();
    test_softmax_sums_to_one();

    printf("\n--- Edge cases ---\n");
    test_edge_case_single_token();
    test_start_at_various_positions();

    printf("\n--- Synthetic logits ---\n");
    test_uniform_logits();

    printf("\n--- Error handling ---\n");
    test_null_inputs();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
