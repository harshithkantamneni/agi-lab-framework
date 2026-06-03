/* test_op_matmul_qat.c -- Unity tests for op_matmul_qat / op_matmul_nt_qat.
 *
 * Program 3 Phase 7 apparatus: §10 item 2 spec.
 *
 * Tests:
 *   1. null_qat_equals_plain_matmul    -- NULL qat => bit-identical to op_matmul
 *   2. null_qat_nt_equals_plain_nt     -- NULL qat => bit-identical to op_matmul_nt
 *   3. enabled_qat_matmul_finite       -- enabled qat => output is finite (no NaN/Inf)
 *   4. enabled_qat_nt_finite           -- enabled qat nt => output is finite
 *   5. enabled_qat_matmul_differs      -- enabled qat => output differs from unquantized
 *   6. same_step_cache_consistency     -- op_matmul_qat and op_matmul_nt_qat hit same cache
 *
 * Memory: heap-allocated tensors (malloc, no pool).
 */

#include "ops.h"
#include "qat_context.h"
#include "tensor.h"
#include "unity.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- Helpers ----------------------------------------------------------- */

/* Allocate heap-backed FP32 tensor, 2D shape [r, c]. */
static Tensor *make_fp32_2d(int32_t r, int32_t c) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;
    memset(t, 0, sizeof(Tensor));
    t->shape[0] = r;
    t->shape[1] = c;
    t->ndim     = 2;
    t->dtype    = DTYPE_FP32;
    /* Row-major strides in elements */
    t->stride[0] = c;
    t->stride[1] = 1;
    t->pool     = NULL;
    t->is_view  = false;
    t->data     = malloc((size_t)r * (size_t)c * sizeof(float));
    if (!t->data) { free(t); return NULL; }
    return t;
}

static void free_fp32(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

static void fill_linspace(Tensor *t, float start, float step) {
    float *d = (float *)t->data;
    int32_t n = t->shape[0] * t->shape[1];
    for (int32_t i = 0; i < n; i++) d[i] = start + (float)i * step;
}

static void fill_const(Tensor *t, float v) {
    float *d = (float *)t->data;
    int32_t n = t->shape[0] * t->shape[1];
    for (int32_t i = 0; i < n; i++) d[i] = v;
}

/* Return true if all elements are finite. */
static bool all_finite(const Tensor *t) {
    const float *d = (const float *)t->data;
    int32_t n = t->shape[0] * t->shape[1];
    for (int32_t i = 0; i < n; i++) {
        if (!isfinite(d[i])) return false;
    }
    return true;
}

/* Return true if any element differs by > tol. */
static bool any_differs(const Tensor *a, const Tensor *b, float tol) {
    const float *da = (const float *)a->data;
    const float *db = (const float *)b->data;
    int32_t n = a->shape[0] * a->shape[1];
    for (int32_t i = 0; i < n; i++) {
        if (fabsf(da[i] - db[i]) > tol) return true;
    }
    return false;
}

/* ---- Tests ------------------------------------------------------------- */

/*
 * TEST 1: null_qat_equals_plain_matmul
 * op_matmul_qat(out, in, W, NULL) must produce bit-identical result to
 * op_matmul(out, in, W).
 */
static void test_null_qat_equals_plain_matmul(void) {
    TEST_BEGIN(null_qat_equals_plain_matmul);

    /* in: [4, 8], W: [8, 6], out: [4, 6] */
    Tensor *in   = make_fp32_2d(4, 8);
    Tensor *W    = make_fp32_2d(8, 6);
    Tensor *out1 = make_fp32_2d(4, 6);
    Tensor *out2 = make_fp32_2d(4, 6);
    ASSERT_NOT_NULL(in);
    ASSERT_NOT_NULL(W);
    ASSERT_NOT_NULL(out1);
    ASSERT_NOT_NULL(out2);

    fill_linspace(in, -1.0f, 0.05f);
    fill_linspace(W,   0.1f, 0.02f);
    fill_const(out1, 0.0f);
    fill_const(out2, 0.0f);

    op_matmul(out1, in, W);
    op_matmul_qat(out2, in, W, NULL);

    /* Bit-identical when qat == NULL */
    const float *d1 = (const float *)out1->data;
    const float *d2 = (const float *)out2->data;
    int32_t n = 4 * 6;
    for (int32_t i = 0; i < n; i++) {
        ASSERT_EQUAL_FLOAT(d1[i], d2[i], 0.0f);
    }

    free_fp32(in);
    free_fp32(W);
    free_fp32(out1);
    free_fp32(out2);
    TEST_END();
}

/*
 * TEST 2: null_qat_nt_equals_plain_nt
 * op_matmul_nt_qat(out, in, W, NULL) must produce bit-identical result to
 * op_matmul_nt(out, in, W).
 */
static void test_null_qat_nt_equals_plain_nt(void) {
    TEST_BEGIN(null_qat_nt_equals_plain_nt);

    /* in: [4, 8], W: [6, 8] (transposed second arg), out: [4, 6] */
    Tensor *in   = make_fp32_2d(4, 8);
    Tensor *W    = make_fp32_2d(6, 8);
    Tensor *out1 = make_fp32_2d(4, 6);
    Tensor *out2 = make_fp32_2d(4, 6);
    ASSERT_NOT_NULL(in);
    ASSERT_NOT_NULL(W);
    ASSERT_NOT_NULL(out1);
    ASSERT_NOT_NULL(out2);

    fill_linspace(in, -1.0f, 0.05f);
    fill_linspace(W,   0.1f, 0.02f);
    fill_const(out1, 0.0f);
    fill_const(out2, 0.0f);

    op_matmul_nt(out1, in, W);
    op_matmul_nt_qat(out2, in, W, NULL);

    const float *d1 = (const float *)out1->data;
    const float *d2 = (const float *)out2->data;
    int32_t n = 4 * 6;
    for (int32_t i = 0; i < n; i++) {
        ASSERT_EQUAL_FLOAT(d1[i], d2[i], 0.0f);
    }

    free_fp32(in);
    free_fp32(W);
    free_fp32(out1);
    free_fp32(out2);
    TEST_END();
}

/*
 * TEST 3: enabled_qat_matmul_finite
 * With a live QATContext, op_matmul_qat must produce finite output.
 */
static void test_enabled_qat_matmul_finite(void) {
    TEST_BEGIN(enabled_qat_matmul_finite);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    /* Use a W large enough for one group of 128: [128, 4] = 512 elements */
    Tensor *in  = make_fp32_2d(2, 128);
    Tensor *W   = make_fp32_2d(128, 4);
    Tensor *out = make_fp32_2d(2, 4);
    ASSERT_NOT_NULL(in);
    ASSERT_NOT_NULL(W);
    ASSERT_NOT_NULL(out);

    fill_linspace(in, -1.0f, 0.01f);
    fill_linspace(W,  -0.5f, 0.002f);
    fill_const(out, 0.0f);

    op_matmul_qat(out, in, W, qc);
    ASSERT_TRUE(all_finite(out));

    free_fp32(in);
    free_fp32(W);
    free_fp32(out);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 4: enabled_qat_nt_finite
 * With a live QATContext, op_matmul_nt_qat must produce finite output.
 */
static void test_enabled_qat_nt_finite(void) {
    TEST_BEGIN(enabled_qat_nt_finite);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    /* in: [2, 128], W: [4, 128] (NT layout), out: [2, 4] */
    Tensor *in  = make_fp32_2d(2, 128);
    Tensor *W   = make_fp32_2d(4, 128);
    Tensor *out = make_fp32_2d(2, 4);
    ASSERT_NOT_NULL(in);
    ASSERT_NOT_NULL(W);
    ASSERT_NOT_NULL(out);

    fill_linspace(in, -1.0f, 0.01f);
    fill_linspace(W,  -0.5f, 0.002f);
    fill_const(out, 0.0f);

    op_matmul_nt_qat(out, in, W, qc);
    ASSERT_TRUE(all_finite(out));

    free_fp32(in);
    free_fp32(W);
    free_fp32(out);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 5: enabled_qat_matmul_differs_from_plain
 * With QAT enabled, op_matmul_qat output must differ from op_matmul output
 * (because weights are snapped to the INT4 grid, introducing rounding error).
 * Use weights with fine gradations so the rounding error is visible.
 */
static void test_enabled_qat_matmul_differs_from_plain(void) {
    TEST_BEGIN(enabled_qat_matmul_differs_from_plain);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    /* in: [1, 128], W: [128, 4] */
    Tensor *in       = make_fp32_2d(1, 128);
    Tensor *W        = make_fp32_2d(128, 4);
    Tensor *out_plain = make_fp32_2d(1, 4);
    Tensor *out_qat   = make_fp32_2d(1, 4);
    ASSERT_NOT_NULL(in);
    ASSERT_NOT_NULL(W);
    ASSERT_NOT_NULL(out_plain);
    ASSERT_NOT_NULL(out_qat);

    fill_linspace(in, -1.0f, 0.015625f);   /* fine steps to drive rounding */
    fill_linspace(W,  -0.5f, 0.001953125f);
    fill_const(out_plain, 0.0f);
    fill_const(out_qat,   0.0f);

    op_matmul(out_plain, in, W);
    op_matmul_qat(out_qat, in, W, qc);

    /* Quantization must introduce visible difference */
    ASSERT_TRUE(any_differs(out_plain, out_qat, 1e-6f));

    free_fp32(in);
    free_fp32(W);
    free_fp32(out_plain);
    free_fp32(out_qat);
    qat_context_destroy(qc);
    TEST_END();
}

/*
 * TEST 6: same_step_cache_consistency
 * In a single step, op_matmul_qat(W) and op_matmul_nt_qat(W_T) must use the
 * same cached w_hat for the same master W pointer.
 *
 * Verification: call op_matmul_qat, then check that covered_count is 1,
 * then call op_matmul_nt_qat with the same W — covered_count must still be 1
 * (cache hit, not a new entry).
 *
 * Note: op_matmul_nt_qat(in, W, qat) uses the transpose of W's rows/cols.
 * The cache key is the W pointer itself, so both lookups use the same key.
 */
static void test_same_step_cache_consistency(void) {
    TEST_BEGIN(same_step_cache_consistency);

    QATContext *qc = qat_context_create(true, 128, 64);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    /* For both ops to be valid: W must be [128, 4] for matmul, [4, 128] for nt.
     * We use the same W pointer for both calls (testing cache key identity),
     * even though the shape interpretation differs between the two ops.
     * The cache must key on the pointer, not the shape semantics. */
    Tensor *in1  = make_fp32_2d(2, 128);
    Tensor *in2  = make_fp32_2d(2, 4);
    Tensor *W    = make_fp32_2d(128, 4);
    Tensor *out1 = make_fp32_2d(2, 4);
    Tensor *out2 = make_fp32_2d(2, 128);
    ASSERT_NOT_NULL(in1);
    ASSERT_NOT_NULL(in2);
    ASSERT_NOT_NULL(W);
    ASSERT_NOT_NULL(out1);
    ASSERT_NOT_NULL(out2);

    fill_linspace(in1, -1.0f, 0.01f);
    fill_linspace(in2, -1.0f, 0.1f);
    fill_linspace(W,   -0.5f, 0.002f);
    fill_const(out1, 0.0f);
    fill_const(out2, 0.0f);

    ASSERT_EQUAL_INT(0, qat_context_covered_count(qc));

    op_matmul_qat(out1, in1, W, qc);
    ASSERT_EQUAL_INT(1, qat_context_covered_count(qc));

    /* Same W pointer: must be a cache hit */
    op_matmul_nt_qat(out2, in2, W, qc);
    ASSERT_EQUAL_INT(1, qat_context_covered_count(qc));

    free_fp32(in1);
    free_fp32(in2);
    free_fp32(W);
    free_fp32(out1);
    free_fp32(out2);
    qat_context_destroy(qc);
    TEST_END();
}

/* ---- main -------------------------------------------------------------- */

int main(void) {
    RUN_TEST(test_null_qat_equals_plain_matmul);
    RUN_TEST(test_null_qat_nt_equals_plain_nt);
    RUN_TEST(test_enabled_qat_matmul_finite);
    RUN_TEST(test_enabled_qat_nt_finite);
    RUN_TEST(test_enabled_qat_matmul_differs_from_plain);
    RUN_TEST(test_same_step_cache_consistency);
    TEST_REPORT();
}
