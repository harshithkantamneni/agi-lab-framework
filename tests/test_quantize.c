/* test_quantize.c -- Tests for group quantization module.
 *
 * Tests are organized in groups:
 *   1. Utility: quantize_num_groups, quantize_error_mse
 *   2. INT4 group quantization: round-trip, accuracy, edge cases
 *   3. INT8 group quantization: round-trip, accuracy
 *   4. Per-group correctness: verify scale/zero_point per group
 *   5. Edge cases: constant tensors, single element, group_size > numel
 *
 * Built with: clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *             -DACCELERATE_NEW_LAPACK -Isrc/core -Isrc/tests
 *             tests/test_quantize.c src/core/memory_pool.c src/core/tensor.c
 *             src/core/ops.c src/core/quantize.c
 *             -o build/tests/test_quantize
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "quantize.h"
#include "tensor.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * UTILITY TESTS
 * ======================================================================== */

static void test_quantize_num_groups_exact(void) {
    /* 256 elements / group_size 128 = 2 groups */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {256};
    Tensor *t = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(t);

    int32_t ng = quantize_num_groups(t, 128);
    ASSERT_EQUAL_INT(2, ng);

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_quantize_num_groups_remainder(void) {
    /* 300 elements / group_size 128 = 3 groups (128 + 128 + 44) */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {300};
    Tensor *t = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(t);

    int32_t ng = quantize_num_groups(t, 128);
    ASSERT_EQUAL_INT(3, ng);

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_quantize_num_groups_smaller_than_group(void) {
    /* 64 elements / group_size 128 = 1 group (partial) */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {64};
    Tensor *t = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(t);

    int32_t ng = quantize_num_groups(t, 128);
    ASSERT_EQUAL_INT(1, ng);

    tensor_destroy(t);
    pool_destroy(pool);
}

static void test_quantize_error_mse_identical(void) {
    /* MSE of two identical tensors should be 0. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {8};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *b = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    for (int i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, (float)i * 0.5f);
        tensor_set(b, idx, (float)i * 0.5f);
    }

    float mse = quantize_error_mse(a, b);
    ASSERT_EQUAL_FLOAT(0.0f, mse, 1e-8f);

    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);
}

static void test_quantize_error_mse_known(void) {
    /* MSE of [1,2,3,4] vs [1.5, 2.5, 3.5, 4.5] = mean(0.25*4) = 0.25 */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *b = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);

    float a_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b_vals[] = {1.5f, 2.5f, 3.5f, 4.5f};
    for (int i = 0; i < 4; i++) {
        int32_t idx[] = {i};
        tensor_set(a, idx, a_vals[i]);
        tensor_set(b, idx, b_vals[i]);
    }

    float mse = quantize_error_mse(a, b);
    ASSERT_EQUAL_FLOAT(0.25f, mse, 1e-6f);

    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);
}

/* ========================================================================
 * INT4 GROUP QUANTIZATION TESTS
 * ======================================================================== */

static void test_int4_group_roundtrip_small(void) {
    /* Quantize 8 FP32 values to INT4, dequantize back, check MSE is low. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {8};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    /* Fill with values spanning a range */
    float vals[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f};
    for (int i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(src, idx, vals[i]);
    }

    int32_t group_size = 8; /* single group for 8 elements */
    int32_t ng = quantize_num_groups(src, group_size);
    ASSERT_EQUAL_INT(1, ng);

    QuantGroup groups[1];
    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    float mse = quantize_error_mse(src, recon);
    /* INT4 has 16 levels over the range [0, 3.5]. Scale = 3.5/15 = 0.233.
     * Max quantization error is ~0.5 * scale = ~0.117.
     * MSE should be much less than scale^2 = ~0.054. */
    ASSERT_TRUE(mse < 0.06f);

    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int4_group_roundtrip_256(void) {
    /* Larger test: 256 elements with group_size=128 (2 groups). */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {256};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    /* Fill with a linearly spaced range [-1.0, 1.0] */
    for (int i = 0; i < 256; i++) {
        int32_t idx[] = {i};
        float val = -1.0f + 2.0f * (float)i / 255.0f;
        tensor_set(src, idx, val);
    }

    int32_t group_size = 128;
    int32_t ng = quantize_num_groups(src, group_size);
    ASSERT_EQUAL_INT(2, ng);

    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    float mse = quantize_error_mse(src, recon);
    /* For range [-1, 1], scale ~ 2.0/15 ~ 0.133.
     * Max error ~ 0.5 * scale ~ 0.067.
     * MSE should be well under 0.01. */
    ASSERT_TRUE(mse < 0.01f);

    free(groups);
    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int4_group_max_error_bounded(void) {
    /* Verify that no single element has error > 1.5 * scale for its group. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {128};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    /* Uniform-ish distribution */
    for (int i = 0; i < 128; i++) {
        int32_t idx[] = {i};
        float val = -5.0f + 10.0f * (float)i / 127.0f;
        tensor_set(src, idx, val);
    }

    int32_t group_size = 128;
    QuantGroup groups[1];
    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    /* Check max error is bounded by 1.5 * scale.
     * (Generous bound: rounding error should be <= 0.5 * scale.) */
    float scale = groups[0].scale;
    ASSERT_TRUE(scale > 0.0f);

    float max_err = 0.0f;
    for (int i = 0; i < 128; i++) {
        int32_t idx[] = {i};
        float orig = tensor_get(src, idx);
        float rec = tensor_get(recon, idx);
        float err = fabsf(orig - rec);
        if (err > max_err) max_err = err;
    }

    /* max rounding error should be <= 0.5 * scale, but use 1.5 for margin */
    ASSERT_TRUE(max_err <= 1.5f * scale);

    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int4_group_constant_tensor(void) {
    /* All-same-value tensor should quantize and dequantize exactly. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {16};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    /* Fill with constant value */
    tensor_fill(src, 3.14f);

    int32_t group_size = 16;
    QuantGroup groups[1];
    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    /* For a constant tensor, min == max, so scale should be 0 (or tiny).
     * All dequantized values should equal the constant. */
    for (int i = 0; i < 16; i++) {
        int32_t idx[] = {i};
        ASSERT_EQUAL_FLOAT(3.14f, tensor_get(recon, idx), 1e-5f);
    }

    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int4_group_single_element(void) {
    /* Single-element tensor: edge case. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {1};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    int32_t idx0[] = {0};
    tensor_set(src, idx0, 2.718f);

    int32_t group_size = 128; /* group_size > numel */
    int32_t ng = quantize_num_groups(src, group_size);
    ASSERT_EQUAL_INT(1, ng);

    QuantGroup groups[1];
    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    /* Single constant element should round-trip exactly */
    ASSERT_EQUAL_FLOAT(2.718f, tensor_get(recon, idx0), 1e-5f);

    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int4_group_size_larger_than_tensor(void) {
    /* group_size=128 but tensor has 10 elements: should still work. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {10};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    for (int i = 0; i < 10; i++) {
        int32_t idx[] = {i};
        tensor_set(src, idx, (float)i * 1.1f);
    }

    int32_t group_size = 128;
    int32_t ng = quantize_num_groups(src, group_size);
    ASSERT_EQUAL_INT(1, ng);

    QuantGroup groups[1];
    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    float mse = quantize_error_mse(src, recon);
    /* range [0, 9.9], scale ~ 9.9/15 ~ 0.66, MSE should be well bounded */
    ASSERT_TRUE(mse < 0.15f);

    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

/* ========================================================================
 * INT4 PER-GROUP CORRECTNESS TESTS
 * ======================================================================== */

static void test_int4_per_group_scale_zero_point(void) {
    /* Verify scale and zero_point are correct for each group. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {16};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);

    /* Group 0: elements [0..7] with values in [0, 7] */
    for (int i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(src, idx, (float)i);
    }
    /* Group 1: elements [8..15] with values in [10, 17] */
    for (int i = 8; i < 16; i++) {
        int32_t idx[] = {i};
        tensor_set(src, idx, 10.0f + (float)(i - 8));
    }

    int32_t group_size = 8;
    QuantGroup groups[2];
    quantize_fp32_to_int4(q4, src, groups, group_size);

    /* Group 0: min=0, max=7, scale = 7/15 = 0.4667, zero_point = 0 */
    ASSERT_EQUAL_FLOAT(0.0f, groups[0].zero_point, 1e-5f);
    ASSERT_EQUAL_FLOAT(7.0f / 15.0f, groups[0].scale, 1e-4f);

    /* Group 1: min=10, max=17, scale = 7/15 = 0.4667, zero_point = 10 */
    ASSERT_EQUAL_FLOAT(10.0f, groups[1].zero_point, 1e-5f);
    ASSERT_EQUAL_FLOAT(7.0f / 15.0f, groups[1].scale, 1e-4f);

    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int4_group_negative_values(void) {
    /* Test with mixed positive and negative values. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {8};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    float vals[] = {-3.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(src, idx, vals[i]);
    }

    int32_t group_size = 8;
    QuantGroup groups[1];
    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    /* zero_point should be -3.0 (min), scale = 7/15 = 0.4667 */
    ASSERT_EQUAL_FLOAT(-3.0f, groups[0].zero_point, 1e-5f);

    float mse = quantize_error_mse(src, recon);
    ASSERT_TRUE(mse < 0.08f);

    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int4_group_2d_tensor(void) {
    /* Test quantization of a 2D tensor (flattened for quantization). */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {4, 8}; /* 32 elements */

    Tensor *src = tensor_create(pool, shape, 2, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 2, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    /* Fill with values */
    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 8; c++) {
            int32_t idx[] = {r, c};
            tensor_set(src, idx, (float)(r * 8 + c) * 0.1f);
        }
    }

    int32_t group_size = 16; /* 2 groups of 16 */
    int32_t ng = quantize_num_groups(src, group_size);
    ASSERT_EQUAL_INT(2, ng);

    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    float mse = quantize_error_mse(src, recon);
    ASSERT_TRUE(mse < 0.01f);

    free(groups);
    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

/* ========================================================================
 * INT8 GROUP QUANTIZATION TESTS
 * ======================================================================== */

static void test_int8_group_roundtrip_small(void) {
    /* INT8 should be much more accurate than INT4. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {16};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q8 = tensor_create(pool, shape, 1, DTYPE_INT8);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q8);
    ASSERT_NOT_NULL(recon);

    for (int i = 0; i < 16; i++) {
        int32_t idx[] = {i};
        tensor_set(src, idx, -2.0f + 4.0f * (float)i / 15.0f);
    }

    int32_t group_size = 16;
    QuantGroup groups[1];
    quantize_fp32_to_int8(q8, src, groups, group_size);
    dequantize_int8_to_fp32(recon, q8, groups, group_size);

    float mse = quantize_error_mse(src, recon);
    /* INT8 has 256 levels. For range 4.0, scale = 4/255 ~ 0.016.
     * MSE should be tiny, well under 0.001. */
    ASSERT_TRUE(mse < 0.001f);

    tensor_destroy(recon);
    tensor_destroy(q8);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int8_group_roundtrip_256(void) {
    /* 256 elements with group_size=128. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {256};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q8 = tensor_create(pool, shape, 1, DTYPE_INT8);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q8);
    ASSERT_NOT_NULL(recon);

    for (int i = 0; i < 256; i++) {
        int32_t idx[] = {i};
        tensor_set(src, idx, -1.0f + 2.0f * (float)i / 255.0f);
    }

    int32_t group_size = 128;
    int32_t ng = quantize_num_groups(src, group_size);
    ASSERT_EQUAL_INT(2, ng);

    QuantGroup *groups = (QuantGroup *)malloc((size_t)ng * sizeof(QuantGroup));
    ASSERT_NOT_NULL(groups);

    quantize_fp32_to_int8(q8, src, groups, group_size);
    dequantize_int8_to_fp32(recon, q8, groups, group_size);

    float mse = quantize_error_mse(src, recon);
    /* INT8 much better than INT4 */
    ASSERT_TRUE(mse < 0.0001f);

    free(groups);
    tensor_destroy(recon);
    tensor_destroy(q8);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int8_much_better_than_int4(void) {
    /* Compare INT4 vs INT8 MSE for same data: INT8 must be strictly better. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {128};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *q8 = tensor_create(pool, shape, 1, DTYPE_INT8);
    Tensor *recon4 = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *recon8 = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(q8);
    ASSERT_NOT_NULL(recon4);
    ASSERT_NOT_NULL(recon8);

    for (int i = 0; i < 128; i++) {
        int32_t idx[] = {i};
        tensor_set(src, idx, -3.0f + 6.0f * (float)i / 127.0f);
    }

    int32_t group_size = 128;
    QuantGroup groups4[1], groups8[1];

    quantize_fp32_to_int4(q4, src, groups4, group_size);
    dequantize_int4_to_fp32(recon4, q4, groups4, group_size);
    float mse4 = quantize_error_mse(src, recon4);

    quantize_fp32_to_int8(q8, src, groups8, group_size);
    dequantize_int8_to_fp32(recon8, q8, groups8, group_size);
    float mse8 = quantize_error_mse(src, recon8);

    /* INT8 should have at least 10x less error than INT4 */
    ASSERT_TRUE(mse8 < mse4 / 10.0f);

    tensor_destroy(recon8);
    tensor_destroy(recon4);
    tensor_destroy(q8);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

static void test_int8_group_constant_tensor(void) {
    /* Constant INT8 tensor round-trip should be exact. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {32};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q8 = tensor_create(pool, shape, 1, DTYPE_INT8);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q8);
    ASSERT_NOT_NULL(recon);

    tensor_fill(src, -1.5f);

    int32_t group_size = 32;
    QuantGroup groups[1];
    quantize_fp32_to_int8(q8, src, groups, group_size);
    dequantize_int8_to_fp32(recon, q8, groups, group_size);

    for (int i = 0; i < 32; i++) {
        int32_t idx[] = {i};
        ASSERT_EQUAL_FLOAT(-1.5f, tensor_get(recon, idx), 1e-5f);
    }

    tensor_destroy(recon);
    tensor_destroy(q8);
    tensor_destroy(src);
    pool_destroy(pool);
}

/* ========================================================================
 * INT4 ZERO TENSOR TEST
 * ======================================================================== */

static void test_int4_group_all_zeros(void) {
    /* All-zero tensor: scale should be 0, dequantized values should be 0. */
    MemoryPool *pool = pool_create(1024 * 1024, POOL_SCRATCH);
    int32_t shape[] = {8};

    Tensor *src = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *q4 = tensor_create(pool, shape, 1, DTYPE_INT4);
    Tensor *recon = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(src);
    ASSERT_NOT_NULL(q4);
    ASSERT_NOT_NULL(recon);

    tensor_fill(src, 0.0f);

    int32_t group_size = 8;
    QuantGroup groups[1];
    quantize_fp32_to_int4(q4, src, groups, group_size);
    dequantize_int4_to_fp32(recon, q4, groups, group_size);

    for (int i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        ASSERT_EQUAL_FLOAT(0.0f, tensor_get(recon, idx), 1e-6f);
    }

    tensor_destroy(recon);
    tensor_destroy(q4);
    tensor_destroy(src);
    pool_destroy(pool);
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("\n========================================\n");
    printf("  Group Quantization Tests\n");
    printf("========================================\n");

    /* Utility */
    printf("--- Utility ---\n");
    RUN_TEST(test_quantize_num_groups_exact);
    RUN_TEST(test_quantize_num_groups_remainder);
    RUN_TEST(test_quantize_num_groups_smaller_than_group);
    RUN_TEST(test_quantize_error_mse_identical);
    RUN_TEST(test_quantize_error_mse_known);

    /* INT4 Group Quantization */
    printf("\n--- INT4 Group Quantization ---\n");
    RUN_TEST(test_int4_group_roundtrip_small);
    RUN_TEST(test_int4_group_roundtrip_256);
    RUN_TEST(test_int4_group_max_error_bounded);
    RUN_TEST(test_int4_group_constant_tensor);
    RUN_TEST(test_int4_group_single_element);
    RUN_TEST(test_int4_group_size_larger_than_tensor);
    RUN_TEST(test_int4_group_all_zeros);

    /* Per-Group Correctness */
    printf("\n--- Per-Group Correctness ---\n");
    RUN_TEST(test_int4_per_group_scale_zero_point);
    RUN_TEST(test_int4_group_negative_values);
    RUN_TEST(test_int4_group_2d_tensor);

    /* INT8 Group Quantization */
    printf("\n--- INT8 Group Quantization ---\n");
    RUN_TEST(test_int8_group_roundtrip_small);
    RUN_TEST(test_int8_group_roundtrip_256);
    RUN_TEST(test_int8_much_better_than_int4);
    RUN_TEST(test_int8_group_constant_tensor);

    TEST_REPORT();
}
