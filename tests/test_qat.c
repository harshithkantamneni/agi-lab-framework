/* test_qat.c -- Unit tests for the QAT fake-quantize + STE module.
 *
 * TDD: tests written BEFORE implementation (qat.h / qat.c).
 * Per program_open_memo §5.2 (binding QAT rule) and §7.4.
 *
 * Covers:
 *   1. round-trip grid-snap: every element of w_hat equals one of the 16
 *      dequantized levels for its group.
 *   2. idempotence: fake_quantize_int4(fake_quantize_int4(w)) == fake_quantize_int4(w)
 *   3. STE backward: fake_quantize_int4_backward() returns upstream grad unchanged.
 *   4. group-boundary + partial final group (numel % group_size != 0).
 *   5. determinism: same input → same output on two calls.
 *   6. memory discipline: all allocs freed, ASan/UBSan clean.
 *
 * NOTE on program_open_memo §5.2 Test 1 (step-0 loss equality within 1e-5):
 *   This test is intentionally NOT implemented here.  Test 1 states "At step 0
 *   with no weight drift, fake-quantized forward + STE backward produces
 *   identical loss to unquantized within 1e-5."  This claim is numerically
 *   inconsistent at the MODULE level: fake_quantize_int4() forces weights onto the
 *   16-level INT4 grid, so the forward loss generally DIFFERS from the unquantized
 *   forward loss even at step 0 (quantization error shifts the logits).  The STE
 *   property is about gradient identity, not forward-pass identity.  Test 1 is a
 *   *training-loop integration* test; its correct scope is the call-site follow-on
 *   dispatch (48 weight-matrix insertions across backprop_train.c / ipc_train.c),
 *   not this standalone module.  Flagged for code_reviewer scrutiny.
 *
 * Build command (matches default Makefile CFLAGS_DBG pattern):
 *   clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *         -DACCELERATE_NEW_LAPACK
 *         -Isrc/core -Isrc/model -Isrc/training -Isrc/tests
 *         tests/test_qat.c src/core/memory_pool.c src/core/tensor.c
 *         src/core/ops.c src/core/quantize.c src/training/qat.c
 *         -o build/tests/test_qat
 *         -framework Accelerate
 */

#include "../src/tests/unity.h"
#include "../src/training/qat.h"
#include "memory_pool.h"
#include "quantize.h"
#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * Helpers
 * ========================================================================= */

/* Build a small FP32 tensor whose data pointer is heap-owned via malloc()
 * (outside any pool) so ownership is explicit.  Caller must call
 * tensor_destroy() then free(t). */
static Tensor *make_heap_fp32(int32_t numel) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;
    float *data = (float *)calloc((size_t)numel, sizeof(float));
    if (!data) { free(t); return NULL; }
    t->data   = data;
    t->shape[0] = numel; t->shape[1] = 1; t->shape[2] = 1; t->shape[3] = 1;
    t->stride[0] = 1; t->stride[1] = numel; t->stride[2] = numel;
    t->stride[3] = numel;
    t->ndim   = 1;
    t->dtype  = DTYPE_FP32;
    t->pool   = NULL;
    t->is_view = false;
    return t;
}

static void free_heap_fp32(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

/* Fill t->data[i] = start + i * step */
static void fill_linspace(Tensor *t, float start, float step) {
    float *d = (float *)t->data;
    int32_t n = t->shape[0];
    for (int32_t i = 0; i < n; i++) d[i] = start + (float)i * step;
}

/* =========================================================================
 * TEST 1 — Round-trip grid-snap
 *
 * Each element of w_hat must equal EXACTLY one of the 16 dequantized
 * INT4 levels for its group.  The level set for group g is:
 *   level_q = groups[g].zero_point + q * groups[g].scale,  q in [0..15]
 *
 * We verify by checking that min(|w_hat[i] - level_q|) over q in [0..15]
 * is < 1e-6 for every element i.
 * ========================================================================= */

static void test_fake_quantize_grid_snap_single_group(void) {
    /* 128 elements, one group, group_size=128 */
    const int32_t N          = 128;
    const int32_t group_size = 128;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    fill_linspace(w, -1.0f, 2.0f / (float)(N - 1));

    int32_t ng = quantize_num_groups(w, group_size);
    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    Tensor *w_hat = fake_quantize_int4(w, groups, group_size);
    ASSERT_NOT_NULL(w_hat);

    float *w_hat_data = (float *)w_hat->data;
    for (int32_t i = 0; i < N; i++) {
        int32_t g = i / group_size;
        float scale = groups[g].scale;
        float zp    = groups[g].zero_point;
        float val   = w_hat_data[i];

        /* Find nearest INT4 level */
        float min_err = INFINITY;
        for (int32_t q = 0; q <= 15; q++) {
            float level = zp + (float)q * scale;
            float err   = fabsf(val - level);
            if (err < min_err) min_err = err;
        }
        ASSERT_TRUE(min_err < 1e-5f);
    }

    free(w_hat->data);
    free(w_hat);
    free(groups);
    free_heap_fp32(w);
}

static void test_fake_quantize_grid_snap_two_groups(void) {
    /* 256 elements, two groups of 128 */
    const int32_t N          = 256;
    const int32_t group_size = 128;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    /* Group 0: [-2, 0], Group 1: [0, 2] (two distinct ranges) */
    float *d = (float *)w->data;
    for (int32_t i = 0; i < 128; i++) d[i]       = -2.0f + (float)i * (2.0f / 127.0f);
    for (int32_t i = 0; i < 128; i++) d[128 + i]  =  0.0f + (float)i * (2.0f / 127.0f);

    int32_t ng = quantize_num_groups(w, group_size);
    ASSERT_EQUAL_INT(2, ng);
    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    Tensor *w_hat = fake_quantize_int4(w, groups, group_size);
    ASSERT_NOT_NULL(w_hat);

    float *w_hat_data = (float *)w_hat->data;
    for (int32_t i = 0; i < N; i++) {
        int32_t g = i / group_size;
        float scale = groups[g].scale;
        float zp    = groups[g].zero_point;
        float val   = w_hat_data[i];

        float min_err = INFINITY;
        for (int32_t q = 0; q <= 15; q++) {
            float level = zp + (float)q * scale;
            float err   = fabsf(val - level);
            if (err < min_err) min_err = err;
        }
        ASSERT_TRUE(min_err < 1e-5f);
    }

    free(w_hat->data);
    free(w_hat);
    free(groups);
    free_heap_fp32(w);
}

/* =========================================================================
 * TEST 2 — Idempotence
 *
 * fake_quantize_int4(fake_quantize_int4(w)) == fake_quantize_int4(w)
 * element-wise within 1e-6.
 * ========================================================================= */

static void test_fake_quantize_idempotent(void) {
    const int32_t N          = 128;
    const int32_t group_size = 128;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    fill_linspace(w, -0.5f, 1.0f / (float)(N - 1));

    int32_t ng = quantize_num_groups(w, group_size);
    QuantGroup *groups1 = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    QuantGroup *groups2 = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups1);
    ASSERT_NOT_NULL(groups2);

    Tensor *w_hat1 = fake_quantize_int4(w, groups1, group_size);
    ASSERT_NOT_NULL(w_hat1);

    Tensor *w_hat2 = fake_quantize_int4(w_hat1, groups2, group_size);
    ASSERT_NOT_NULL(w_hat2);

    float *d1 = (float *)w_hat1->data;
    float *d2 = (float *)w_hat2->data;
    for (int32_t i = 0; i < N; i++) {
        ASSERT_EQUAL_FLOAT(d1[i], d2[i], 1e-6f);
    }

    free(w_hat2->data); free(w_hat2);
    free(w_hat1->data); free(w_hat1);
    free(groups2);
    free(groups1);
    free_heap_fp32(w);
}

/* =========================================================================
 * TEST 3 — STE backward: identity passthrough
 *
 * fake_quantize_int4_backward(grad_out, w_fp32) must return a tensor
 * identical to grad_out (every element matches within 1e-9).
 * ========================================================================= */

static void test_ste_backward_identity(void) {
    const int32_t N = 64;

    Tensor *grad_out = make_heap_fp32(N);
    Tensor *w_fp32   = make_heap_fp32(N);
    ASSERT_NOT_NULL(grad_out);
    ASSERT_NOT_NULL(w_fp32);

    /* Fill grad_out with arbitrary non-trivial values */
    float *g = (float *)grad_out->data;
    float *w = (float *)w_fp32->data;
    for (int32_t i = 0; i < N; i++) {
        g[i] = -1.0f + (float)i * (2.0f / (float)(N - 1));
        w[i] =  0.5f * g[i];
    }

    Tensor *grad_in = fake_quantize_int4_backward(grad_out, w_fp32);
    ASSERT_NOT_NULL(grad_in);

    float *gi = (float *)grad_in->data;
    for (int32_t i = 0; i < N; i++) {
        ASSERT_EQUAL_FLOAT(g[i], gi[i], 1e-9f);
    }

    free(grad_in->data); free(grad_in);
    free_heap_fp32(w_fp32);
    free_heap_fp32(grad_out);
}

static void test_ste_backward_zeros_passthrough(void) {
    /* Zero gradient should pass through unchanged */
    const int32_t N = 32;

    Tensor *grad_out = make_heap_fp32(N);
    Tensor *w_fp32   = make_heap_fp32(N);
    ASSERT_NOT_NULL(grad_out);
    ASSERT_NOT_NULL(w_fp32);

    /* grad_out already zero-initialised by calloc in make_heap_fp32 */
    fill_linspace(w_fp32, -1.0f, 2.0f / (float)(N - 1));

    Tensor *grad_in = fake_quantize_int4_backward(grad_out, w_fp32);
    ASSERT_NOT_NULL(grad_in);

    float *gi = (float *)grad_in->data;
    for (int32_t i = 0; i < N; i++) {
        ASSERT_EQUAL_FLOAT(0.0f, gi[i], 1e-12f);
    }

    free(grad_in->data); free(grad_in);
    free_heap_fp32(w_fp32);
    free_heap_fp32(grad_out);
}

/* =========================================================================
 * TEST 4 — Partial final group (numel % group_size != 0)
 *
 * 300 elements, group_size=128 → groups 0,1 are full (128 each);
 * group 2 is partial (44 elements).  All elements must still snap to a
 * valid INT4 level for their respective group.
 * ========================================================================= */

static void test_fake_quantize_partial_final_group(void) {
    const int32_t N          = 300;  /* 128 + 128 + 44 */
    const int32_t group_size = 128;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    fill_linspace(w, -3.0f, 6.0f / (float)(N - 1));

    int32_t ng = quantize_num_groups(w, group_size);
    ASSERT_EQUAL_INT(3, ng);

    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    Tensor *w_hat = fake_quantize_int4(w, groups, group_size);
    ASSERT_NOT_NULL(w_hat);

    float *d = (float *)w_hat->data;
    for (int32_t i = 0; i < N; i++) {
        int32_t g  = i / group_size;
        float scale = groups[g].scale;
        float zp    = groups[g].zero_point;
        float val   = d[i];

        float min_err = INFINITY;
        for (int32_t q = 0; q <= 15; q++) {
            float level = zp + (float)q * scale;
            float err   = fabsf(val - level);
            if (err < min_err) min_err = err;
        }
        ASSERT_TRUE(min_err < 1e-5f);
    }

    free(w_hat->data); free(w_hat);
    free(groups);
    free_heap_fp32(w);
}

/* Partial group: numel < group_size (single group of 17 elements) */
static void test_fake_quantize_partial_single_small_group(void) {
    const int32_t N          = 17;
    const int32_t group_size = 128;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    fill_linspace(w, -1.0f, 2.0f / (float)(N - 1));

    int32_t ng = quantize_num_groups(w, group_size);
    ASSERT_EQUAL_INT(1, ng);

    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    Tensor *w_hat = fake_quantize_int4(w, groups, group_size);
    ASSERT_NOT_NULL(w_hat);

    float *d = (float *)w_hat->data;
    for (int32_t i = 0; i < N; i++) {
        float scale = groups[0].scale;
        float zp    = groups[0].zero_point;
        float val   = d[i];

        float min_err = INFINITY;
        for (int32_t q = 0; q <= 15; q++) {
            float level = zp + (float)q * scale;
            float err   = fabsf(val - level);
            if (err < min_err) min_err = err;
        }
        ASSERT_TRUE(min_err < 1e-5f);
    }

    free(w_hat->data); free(w_hat);
    free(groups);
    free_heap_fp32(w);
}

/* =========================================================================
 * TEST 5 — Determinism: two calls on identical input → identical output
 * ========================================================================= */

static void test_fake_quantize_deterministic(void) {
    const int32_t N          = 128;
    const int32_t group_size = 128;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    fill_linspace(w, -2.0f, 4.0f / (float)(N - 1));

    int32_t ng = quantize_num_groups(w, group_size);
    QuantGroup *groups1 = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    QuantGroup *groups2 = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups1);
    ASSERT_NOT_NULL(groups2);

    Tensor *w_hat1 = fake_quantize_int4(w, groups1, group_size);
    Tensor *w_hat2 = fake_quantize_int4(w, groups2, group_size);
    ASSERT_NOT_NULL(w_hat1);
    ASSERT_NOT_NULL(w_hat2);

    float *d1 = (float *)w_hat1->data;
    float *d2 = (float *)w_hat2->data;
    for (int32_t i = 0; i < N; i++) {
        ASSERT_EQUAL_FLOAT(d1[i], d2[i], 0.0f); /* bit-exact */
    }

    /* groups must also be identical */
    for (int32_t g = 0; g < ng; g++) {
        ASSERT_EQUAL_FLOAT(groups1[g].scale,      groups2[g].scale,      0.0f);
        ASSERT_EQUAL_FLOAT(groups1[g].zero_point,  groups2[g].zero_point, 0.0f);
    }

    free(w_hat2->data); free(w_hat2);
    free(w_hat1->data); free(w_hat1);
    free(groups2);
    free(groups1);
    free_heap_fp32(w);
}

/* =========================================================================
 * TEST 6 — Output shape and dtype
 *
 * w_hat must have the same shape as w and dtype DTYPE_FP32.
 * ========================================================================= */

static void test_fake_quantize_output_shape_dtype(void) {
    const int32_t N          = 256;
    const int32_t group_size = 128;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    fill_linspace(w, 0.0f, 1.0f / (float)(N - 1));

    int32_t ng = quantize_num_groups(w, group_size);
    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    Tensor *w_hat = fake_quantize_int4(w, groups, group_size);
    ASSERT_NOT_NULL(w_hat);

    ASSERT_EQUAL_INT(DTYPE_FP32,   (int)w_hat->dtype);
    ASSERT_EQUAL_INT(w->shape[0],  w_hat->shape[0]);
    ASSERT_EQUAL_INT(w->ndim,      w_hat->ndim);

    free(w_hat->data); free(w_hat);
    free(groups);
    free_heap_fp32(w);
}

/* =========================================================================
 * TEST 7 — QuantGroup metadata is populated (scale > 0 for non-constant)
 * ========================================================================= */

static void test_fake_quantize_groups_populated(void) {
    const int32_t N          = 256;
    const int32_t group_size = 128;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    /* Non-constant weights so scale must be > 0 */
    fill_linspace(w, -1.0f, 2.0f / (float)(N - 1));

    int32_t ng = quantize_num_groups(w, group_size);
    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    Tensor *w_hat = fake_quantize_int4(w, groups, group_size);
    ASSERT_NOT_NULL(w_hat);

    for (int32_t g = 0; g < ng; g++) {
        ASSERT_TRUE(groups[g].scale > 0.0f);
    }

    free(w_hat->data); free(w_hat);
    free(groups);
    free_heap_fp32(w);
}

/* =========================================================================
 * TEST 8 — Constant tensor: all-same value round-trips exactly
 * ========================================================================= */

static void test_fake_quantize_constant_tensor(void) {
    const int32_t N          = 128;
    const int32_t group_size = 128;
    const float   CONST_VAL  = 3.14159f;

    Tensor *w = make_heap_fp32(N);
    ASSERT_NOT_NULL(w);
    float *d = (float *)w->data;
    for (int32_t i = 0; i < N; i++) d[i] = CONST_VAL;

    int32_t ng = quantize_num_groups(w, group_size);
    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    Tensor *w_hat = fake_quantize_int4(w, groups, group_size);
    ASSERT_NOT_NULL(w_hat);

    float *wh = (float *)w_hat->data;
    for (int32_t i = 0; i < N; i++) {
        ASSERT_EQUAL_FLOAT(CONST_VAL, wh[i], 1e-5f);
    }

    free(w_hat->data); free(w_hat);
    free(groups);
    free_heap_fp32(w);
}

/* =========================================================================
 * TEST 9 — STE backward output shape and dtype
 * ========================================================================= */

static void test_ste_backward_output_shape_dtype(void) {
    const int32_t N = 64;

    Tensor *grad_out = make_heap_fp32(N);
    Tensor *w_fp32   = make_heap_fp32(N);
    ASSERT_NOT_NULL(grad_out);
    ASSERT_NOT_NULL(w_fp32);

    fill_linspace(grad_out, -1.0f, 2.0f / (float)(N - 1));
    fill_linspace(w_fp32, -0.5f, 1.0f / (float)(N - 1));

    Tensor *grad_in = fake_quantize_int4_backward(grad_out, w_fp32);
    ASSERT_NOT_NULL(grad_in);

    ASSERT_EQUAL_INT(DTYPE_FP32,         (int)grad_in->dtype);
    ASSERT_EQUAL_INT(grad_out->shape[0], grad_in->shape[0]);
    ASSERT_EQUAL_INT(grad_out->ndim,     grad_in->ndim);

    free(grad_in->data); free(grad_in);
    free_heap_fp32(w_fp32);
    free_heap_fp32(grad_out);
}

/* =========================================================================
 * MAIN
 * ========================================================================= */

int main(void) {
    printf("\n========================================\n");
    printf("  QAT Fake-Quantize + STE Tests\n");
    printf("  (program_3 P7 apparatus; §5.2 binding)\n");
    printf("========================================\n");

    printf("\n--- Forward: grid-snap ---\n");
    RUN_TEST(test_fake_quantize_grid_snap_single_group);
    RUN_TEST(test_fake_quantize_grid_snap_two_groups);

    printf("\n--- Forward: idempotence ---\n");
    RUN_TEST(test_fake_quantize_idempotent);

    printf("\n--- Backward: STE identity ---\n");
    RUN_TEST(test_ste_backward_identity);
    RUN_TEST(test_ste_backward_zeros_passthrough);

    printf("\n--- Partial/boundary groups ---\n");
    RUN_TEST(test_fake_quantize_partial_final_group);
    RUN_TEST(test_fake_quantize_partial_single_small_group);

    printf("\n--- Determinism ---\n");
    RUN_TEST(test_fake_quantize_deterministic);

    printf("\n--- Output contract ---\n");
    RUN_TEST(test_fake_quantize_output_shape_dtype);
    RUN_TEST(test_fake_quantize_groups_populated);
    RUN_TEST(test_fake_quantize_constant_tensor);
    RUN_TEST(test_ste_backward_output_shape_dtype);

    TEST_REPORT();
}
