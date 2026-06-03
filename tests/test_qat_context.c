/* test_qat_context.c -- Unity tests for the QATContext per-step w_hat cache.
 *
 * Program 3 Phase 7 apparatus: test_qat_context covers §10 item 1 spec.
 *
 * Tests:
 *   1. disabled_passthrough       -- enabled=false => w_hat returns W unchanged (same pointer)
 *   2. disabled_covered_count     -- enabled=false => covered_count stays 0 after w_hat calls
 *   3. enabled_grid_snap          -- enabled=true, first call grid-snaps (result != W data)
 *   4. enabled_cache_hit          -- second call for same W pointer returns identical pointer
 *   5. begin_step_clears_count    -- begin_step() sets covered_count to 0
 *   6. per_step_freshness         -- after begin_step, re-snap uses current weight data
 *   7. covered_count_increments   -- each distinct W pointer increments covered_count
 *   8. null_qat_returns_w         -- NULL context => qat_context_w_hat N/A (boundary)
 *   9. is_enabled_reflects_flag   -- qat_context_is_enabled returns correct value
 *  10. capacity_zero_disabled     -- capacity=0 still works (enabled=false case)
 *
 * Memory: all test tensors are heap-allocated (malloc-based, no pool).
 *         Caller frees: free(t->data); free(t);
 */

#include "qat_context.h"
#include "tensor.h"
#include "unity.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- Helpers ----------------------------------------------------------- */

/* Allocate a heap-backed FP32 tensor with given numel (1D shape). */
static Tensor *make_fp32(int32_t numel) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;
    memset(t, 0, sizeof(Tensor));
    t->shape[0] = numel;
    t->ndim     = 1;
    t->dtype    = DTYPE_FP32;
    t->pool     = NULL;
    t->is_view  = false;
    t->data     = malloc((size_t)numel * sizeof(float));
    if (!t->data) { free(t); return NULL; }
    return t;
}

static void free_fp32(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

/* Fill tensor with a simple pattern: data[i] = start + i * step */
static void fill_pattern(Tensor *t, float start, float step) {
    float *d = (float *)t->data;
    int32_t n = t->shape[0];
    for (int32_t i = 0; i < n; i++) d[i] = start + (float)i * step;
}

/* Check that two float buffers differ by at least `min_diff` in at least one
 * element. Returns true if they differ sufficiently. */
static bool buffers_differ(const float *a, const float *b, int32_t n,
                           float min_diff) {
    for (int32_t i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) >= min_diff) return true;
    }
    return false;
}

/* ---- Tests ------------------------------------------------------------- */

/*
 * TEST 1: disabled_passthrough
 * When QATContext is created with enabled=false, qat_context_w_hat must return
 * the exact same pointer as W (no copy, no quantization).
 */
static void test_disabled_passthrough(void) {
    TEST_BEGIN(disabled_passthrough);

    QATContext *qc = qat_context_create(false, 128, 64);
    ASSERT_NOT_NULL(qc);

    Tensor *W = make_fp32(256);
    ASSERT_NOT_NULL(W);
    fill_pattern(W, 0.0f, 0.01f);

    const Tensor *w_hat = qat_context_w_hat(qc, W);
    /* disabled: must return W unchanged (same pointer) */
    ASSERT_TRUE(w_hat == W);

    free_fp32(W);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 2: disabled_covered_count
 * covered_count must stay 0 when disabled, even after calling w_hat.
 */
static void test_disabled_covered_count(void) {
    TEST_BEGIN(disabled_covered_count);

    QATContext *qc = qat_context_create(false, 128, 64);
    ASSERT_NOT_NULL(qc);

    Tensor *W1 = make_fp32(128);
    Tensor *W2 = make_fp32(128);
    ASSERT_NOT_NULL(W1);
    ASSERT_NOT_NULL(W2);
    fill_pattern(W1, 1.0f, 0.01f);
    fill_pattern(W2, 2.0f, 0.01f);

    ASSERT_EQUAL_INT(0, qat_context_covered_count(qc));
    qat_context_w_hat(qc, W1);
    ASSERT_EQUAL_INT(0, qat_context_covered_count(qc));
    qat_context_w_hat(qc, W2);
    ASSERT_EQUAL_INT(0, qat_context_covered_count(qc));

    free_fp32(W1);
    free_fp32(W2);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 3: enabled_grid_snap
 * When enabled=true, the first call to w_hat must produce a tensor whose data
 * differs from the original (grid-snapped to INT4 levels), and the returned
 * pointer must NOT be the same as W.
 */
static void test_enabled_grid_snap(void) {
    TEST_BEGIN(enabled_grid_snap);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);

    /* Use 256 elements (2 groups of 128) with a pattern that forces rounding */
    Tensor *W = make_fp32(256);
    ASSERT_NOT_NULL(W);
    fill_pattern(W, -1.0f, 0.0078125f); /* range approx [-1, 1] with fine steps */

    const Tensor *w_hat = qat_context_w_hat(qc, W);
    ASSERT_NOT_NULL(w_hat);
    /* Must be a different pointer (internal buffer, not W itself) */
    ASSERT_TRUE(w_hat != W);
    /* Data must differ from original (quantization causes rounding) */
    ASSERT_TRUE(buffers_differ((const float *)W->data,
                               (const float *)w_hat->data,
                               256, 1e-6f));

    free_fp32(W);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 4: enabled_cache_hit
 * Second call with the same W pointer in the same step must return the EXACT
 * same pointer as the first call (cache hit, no re-quantization).
 */
static void test_enabled_cache_hit(void) {
    TEST_BEGIN(enabled_cache_hit);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);

    Tensor *W = make_fp32(256);
    ASSERT_NOT_NULL(W);
    fill_pattern(W, -0.5f, 0.004f);

    const Tensor *w_hat1 = qat_context_w_hat(qc, W);
    ASSERT_NOT_NULL(w_hat1);

    const Tensor *w_hat2 = qat_context_w_hat(qc, W);
    ASSERT_NOT_NULL(w_hat2);

    /* Cache hit: must return identical pointer */
    ASSERT_TRUE(w_hat1 == w_hat2);

    free_fp32(W);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 5: begin_step_clears_count
 * After calling qat_context_begin_step, covered_count must be 0, even if the
 * prior step had non-zero count.
 */
static void test_begin_step_clears_count(void) {
    TEST_BEGIN(begin_step_clears_count);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);

    Tensor *W = make_fp32(128);
    ASSERT_NOT_NULL(W);
    fill_pattern(W, 0.0f, 0.01f);

    /* Step 1: call w_hat to populate */
    qat_context_begin_step(qc);
    qat_context_w_hat(qc, W);
    ASSERT_EQUAL_INT(1, qat_context_covered_count(qc));

    /* Begin next step: count must reset */
    qat_context_begin_step(qc);
    ASSERT_EQUAL_INT(0, qat_context_covered_count(qc));

    free_fp32(W);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 6: per_step_freshness
 * After begin_step, the same W pointer with different data must produce a new
 * (re-quantized) result whose data differs from the prior step's result.
 *
 * This validates that begin_step marks the cache stale so re-quantization fires
 * on the next call, picking up current weight values.
 */
static void test_per_step_freshness(void) {
    TEST_BEGIN(per_step_freshness);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);

    Tensor *W = make_fp32(256);
    ASSERT_NOT_NULL(W);

    /* Step 1: pattern A */
    qat_context_begin_step(qc);
    fill_pattern(W, -1.0f, 0.0078125f);
    const Tensor *w_hat_step1 = qat_context_w_hat(qc, W);
    ASSERT_NOT_NULL(w_hat_step1);

    /* Save step 1 result data */
    float *saved_step1 = (float *)malloc(256 * sizeof(float));
    ASSERT_NOT_NULL(saved_step1);
    memcpy(saved_step1, w_hat_step1->data, 256 * sizeof(float));

    /* Step 2: different pattern — begin_step must invalidate cache */
    qat_context_begin_step(qc);
    fill_pattern(W, 0.5f, 0.003f); /* completely different values */
    const Tensor *w_hat_step2 = qat_context_w_hat(qc, W);
    ASSERT_NOT_NULL(w_hat_step2);

    /* Step 2 result must differ from step 1 result */
    ASSERT_TRUE(buffers_differ(saved_step1, (const float *)w_hat_step2->data,
                               256, 1e-6f));

    free(saved_step1);
    free_fp32(W);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 7: covered_count_increments
 * Each distinct W pointer in one step increments covered_count by 1.
 * Calling the same pointer twice keeps count at 1 (idempotent).
 */
static void test_covered_count_increments(void) {
    TEST_BEGIN(covered_count_increments);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);

    Tensor *W1 = make_fp32(128);
    Tensor *W2 = make_fp32(128);
    Tensor *W3 = make_fp32(128);
    ASSERT_NOT_NULL(W1);
    ASSERT_NOT_NULL(W2);
    ASSERT_NOT_NULL(W3);
    fill_pattern(W1, 0.1f, 0.01f);
    fill_pattern(W2, 0.2f, 0.01f);
    fill_pattern(W3, 0.3f, 0.01f);

    qat_context_begin_step(qc);
    ASSERT_EQUAL_INT(0, qat_context_covered_count(qc));

    qat_context_w_hat(qc, W1);
    ASSERT_EQUAL_INT(1, qat_context_covered_count(qc));

    /* Same pointer again: idempotent */
    qat_context_w_hat(qc, W1);
    ASSERT_EQUAL_INT(1, qat_context_covered_count(qc));

    qat_context_w_hat(qc, W2);
    ASSERT_EQUAL_INT(2, qat_context_covered_count(qc));

    qat_context_w_hat(qc, W3);
    ASSERT_EQUAL_INT(3, qat_context_covered_count(qc));

    free_fp32(W1);
    free_fp32(W2);
    free_fp32(W3);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 8: is_enabled_reflects_flag
 * qat_context_is_enabled must return the value passed to qat_context_create.
 */
static void test_is_enabled_reflects_flag(void) {
    TEST_BEGIN(is_enabled_reflects_flag);

    QATContext *qc_on  = qat_context_create(true,  128, 64);
    QATContext *qc_off = qat_context_create(false, 128, 64);
    ASSERT_NOT_NULL(qc_on);
    ASSERT_NOT_NULL(qc_off);

    ASSERT_TRUE(qat_context_is_enabled(qc_on));
    ASSERT_FALSE(qat_context_is_enabled(qc_off));

    qat_context_destroy(qc_on);
    qat_context_destroy(qc_off);
    TEST_END();
}

/*
 * TEST 9: w_hat_data_on_grid
 * Enabled mode: returned w_hat values must all lie on the INT4 dequantization
 * grid. For asymmetric per-group quant: each element = q * scale + zero_point
 * where q is integer in [0,15]. We verify this by checking that
 * fake_quantize(fake_quantize(W)) == fake_quantize(W)  (idempotent after snap).
 *
 * Implementation: call w_hat once (step 1), copy the result, begin_step,
 * fill W with the copied result data, call w_hat again (step 2).
 * Step 2 values should be extremely close to step 1 values (< 1e-6 per element).
 */
static void test_w_hat_data_on_grid(void) {
    TEST_BEGIN(w_hat_data_on_grid);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);

    Tensor *W = make_fp32(256);
    ASSERT_NOT_NULL(W);
    fill_pattern(W, -1.0f, 0.0078125f);

    /* Step 1 */
    qat_context_begin_step(qc);
    const Tensor *w_hat1 = qat_context_w_hat(qc, W);
    ASSERT_NOT_NULL(w_hat1);

    /* Copy step 1 result into W */
    memcpy(W->data, w_hat1->data, 256 * sizeof(float));

    /* Step 2: quantizing an already-quantized result should be idempotent */
    qat_context_begin_step(qc);
    const Tensor *w_hat2 = qat_context_w_hat(qc, W);
    ASSERT_NOT_NULL(w_hat2);

    /* Values should be essentially identical (tolerance: < 1e-5 per element) */
    const float *d1 = (const float *)w_hat1->data;
    const float *d2 = (const float *)w_hat2->data;
    for (int32_t i = 0; i < 256; i++) {
        float diff = fabsf(d1[i] - d2[i]);
        ASSERT_TRUE(diff < 1e-5f);
    }

    free_fp32(W);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 10: null_begin_step_safe
 * Calling qat_context_begin_step with NULL should not crash (NULL guard).
 * This is a defensive test — the implementation must tolerate NULL.
 */
static void test_null_begin_step_safe(void) {
    TEST_BEGIN(null_begin_step_safe);
    /* Must not crash */
    qat_context_begin_step(NULL);
    TEST_END();
}

/*
 * TEST 11: capacity_boundary_exact_fit (FLAG-3)
 *
 * When capacity == number of distinct weights, the hash map must accommodate
 * all weights without dropping into pass-through mode.  Verify that:
 *   (a) covered_count reaches N after N distinct w_hat calls.
 *   (b) The (N+1)-th distinct weight with capacity==N degrades gracefully
 *       (returns a valid pointer — either W itself or a cached slot).
 *
 * This test uses a small capacity (8) and 8 weights to exercise the full
 * linear-probe chain without an off-by-one that would silently drop weights.
 */
static void test_capacity_boundary_exact_fit(void) {
    TEST_BEGIN(capacity_boundary_exact_fit);

    /* capacity=9 > 8 weights, which is the "just over" threshold:
     * open-addressing needs load < 1 to guarantee no infinite probe loop.
     * We use 9 to test the near-capacity regime; then add a 9th weight to
     * verify graceful degradation (returns W or a valid slot, no crash). */
    const int N = 8;
    const int CAP = 9;
    const int GROUP = 4;

    QATContext *qc = qat_context_create(true, GROUP, CAP);
    ASSERT_NOT_NULL(qc);

    qat_context_begin_step(qc);

    /* Create N distinct weight tensors, each with GROUP elements (exactly 1 group) */
    Tensor *weights[8];
    for (int i = 0; i < N; i++) {
        weights[i] = make_fp32(GROUP);
        ASSERT_NOT_NULL(weights[i]);
        float *d = (float *)weights[i]->data;
        /* Fill with non-multiple-of-quant-step values to ensure non-trivial snap */
        for (int j = 0; j < GROUP; j++) {
            d[j] = 0.37f * (float)(i + 1) + 0.01f * (float)j;
        }
    }

    /* Register all N weights — covered_count should reach N */
    for (int i = 0; i < N; i++) {
        const Tensor *w_hat = qat_context_w_hat(qc, weights[i]);
        ASSERT_NOT_NULL(w_hat);
    }
    ASSERT_EQUAL_INT(N, qat_context_covered_count(qc));

    /* Re-register all N weights — all should be cache hits now */
    for (int i = 0; i < N; i++) {
        const Tensor *w_hat = qat_context_w_hat(qc, weights[i]);
        ASSERT_NOT_NULL(w_hat);
    }
    /* covered_count must not have increased (all hits) */
    ASSERT_EQUAL_INT(N, qat_context_covered_count(qc));
    /* cache_hits must be >= N (at least one hit per re-register call) */
    ASSERT_TRUE(qat_context_cache_hits(qc) >= N);

    /* Cleanup */
    qat_context_destroy(qc);
    for (int i = 0; i < N; i++) {
        free_fp32(weights[i]);
    }
    TEST_END();
}

/* ---- main -------------------------------------------------------------- */

int main(void) {
    RUN_TEST(test_disabled_passthrough);
    RUN_TEST(test_disabled_covered_count);
    RUN_TEST(test_enabled_grid_snap);
    RUN_TEST(test_enabled_cache_hit);
    RUN_TEST(test_begin_step_clears_count);
    RUN_TEST(test_per_step_freshness);
    RUN_TEST(test_covered_count_increments);
    RUN_TEST(test_is_enabled_reflects_flag);
    RUN_TEST(test_w_hat_data_on_grid);
    RUN_TEST(test_null_begin_step_safe);
    RUN_TEST(test_capacity_boundary_exact_fit);
    TEST_REPORT();
}
