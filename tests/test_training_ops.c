/* test_training_ops.c -- Tests for training-specific tensor operations.
 *
 * Tests:
 *   1. op_sub                  -- element-wise subtraction
 *   2. op_add_scaled           -- fused add-scale
 *   3. op_matmul_tn            -- A^T @ B
 *   4. op_matmul_nt            -- A @ B^T
 *   5. op_silu                 -- SiLU activation
 *   6. op_silu_backward        -- SiLU derivative
 *   7. op_cross_entropy        -- loss + gradient
 *   8. op_cross_entropy_grad   -- gradient correctness
 *   9. op_rand_uniform         -- random in range
 *  10. op_init_xavier          -- Xavier initialization
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include <math.h>
#include <stdlib.h>

/* ---- TEST 1: op_sub ---- */
static void test_op_sub(void) {
    TEST_BEGIN(op_sub);

    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);

    int32_t shape[] = {3};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *b = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(a);
    ASSERT_NOT_NULL(b);
    ASSERT_NOT_NULL(out);

    /* a = [3, 5, 7], b = [1, 2, 3] -> out = [2, 3, 4] */
    int32_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    tensor_set(a, i0, 3.0f); tensor_set(a, i1, 5.0f); tensor_set(a, i2, 7.0f);
    tensor_set(b, i0, 1.0f); tensor_set(b, i1, 2.0f); tensor_set(b, i2, 3.0f);

    op_sub(out, a, b);

    ASSERT_EQUAL_FLOAT(2.0f, tensor_get(out, i0), 1e-6f);
    ASSERT_EQUAL_FLOAT(3.0f, tensor_get(out, i1), 1e-6f);
    ASSERT_EQUAL_FLOAT(4.0f, tensor_get(out, i2), 1e-6f);

    tensor_destroy(out);
    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 2: op_add_scaled ---- */
static void test_op_add_scaled(void) {
    TEST_BEGIN(op_add_scaled);

    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    int32_t shape[] = {3};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *b = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    /* a = [1, 2, 3], b = [4, 5, 6], s = -0.1 -> out = [0.6, 1.5, 2.4] */
    int32_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    tensor_set(a, i0, 1.0f); tensor_set(a, i1, 2.0f); tensor_set(a, i2, 3.0f);
    tensor_set(b, i0, 4.0f); tensor_set(b, i1, 5.0f); tensor_set(b, i2, 6.0f);

    op_add_scaled(out, a, b, -0.1f);

    ASSERT_EQUAL_FLOAT(0.6f, tensor_get(out, i0), 1e-5f);
    ASSERT_EQUAL_FLOAT(1.5f, tensor_get(out, i1), 1e-5f);
    ASSERT_EQUAL_FLOAT(2.4f, tensor_get(out, i2), 1e-5f);

    tensor_destroy(out);
    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 3: op_matmul_tn ---- */
static void test_op_matmul_tn(void) {
    TEST_BEGIN(op_matmul_tn);

    MemoryPool *pool = pool_create(65536, POOL_SCRATCH);

    /* a = [[1, 2],    (3x2)   a^T = [[1, 3, 5],   (2x3)
     *      [3, 4],                    [2, 4, 6]]
     *      [5, 6]]
     * b = [[7, 8, 9],  (3x3)
     *      [10,11,12],
     *      [13,14,15]]
     * a^T @ b = [[1*7+3*10+5*13, 1*8+3*11+5*14, 1*9+3*12+5*15],   (2x3)
     *            [2*7+4*10+6*13, 2*8+4*11+6*14, 2*9+4*12+6*15]]
     *         = [[102, 111, 120],
     *            [124, 135, 146]] */
    int32_t a_shape[] = {3, 2};
    int32_t b_shape[] = {3, 3};
    int32_t out_shape[] = {2, 3};

    Tensor *a = tensor_create(pool, a_shape, 2, DTYPE_FP32);
    Tensor *b = tensor_create(pool, b_shape, 2, DTYPE_FP32);
    Tensor *out = tensor_create(pool, out_shape, 2, DTYPE_FP32);

    int32_t idx[2];
    float a_vals[] = {1, 2, 3, 4, 5, 6};
    float b_vals[] = {7, 8, 9, 10, 11, 12, 13, 14, 15};
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 2; c++) {
            idx[0] = r; idx[1] = c;
            tensor_set(a, idx, a_vals[r * 2 + c]);
        }
    }
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            idx[0] = r; idx[1] = c;
            tensor_set(b, idx, b_vals[r * 3 + c]);
        }
    }

    op_matmul_tn(out, a, b);

    idx[0] = 0; idx[1] = 0; ASSERT_EQUAL_FLOAT(102.0f, tensor_get(out, idx), 1e-4f);
    idx[0] = 0; idx[1] = 1; ASSERT_EQUAL_FLOAT(111.0f, tensor_get(out, idx), 1e-4f);
    idx[0] = 0; idx[1] = 2; ASSERT_EQUAL_FLOAT(120.0f, tensor_get(out, idx), 1e-4f);
    /* [1,0]: 2*7 + 4*10 + 6*13 = 14+40+78 = 132 */
    idx[0] = 1; idx[1] = 0; ASSERT_EQUAL_FLOAT(132.0f, tensor_get(out, idx), 1e-4f);
    /* [1,1]: 2*8 + 4*11 + 6*14 = 16+44+84 = 144 */
    idx[0] = 1; idx[1] = 1; ASSERT_EQUAL_FLOAT(144.0f, tensor_get(out, idx), 1e-4f);
    /* [1,2]: 2*9 + 4*12 + 6*15 = 18+48+90 = 156 */
    idx[0] = 1; idx[1] = 2; ASSERT_EQUAL_FLOAT(156.0f, tensor_get(out, idx), 1e-4f);

    tensor_destroy(out);
    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 4: op_matmul_nt ---- */
static void test_op_matmul_nt(void) {
    TEST_BEGIN(op_matmul_nt);

    MemoryPool *pool = pool_create(65536, POOL_SCRATCH);

    /* a = [[1, 2, 3],  (2x3)
     *      [4, 5, 6]]
     * b = [[7, 8, 9],  (2x3)  b^T = [[7, 10],  (3x2)
     *      [10,11,12]]                [8, 11],
     *                                 [9, 12]]
     * a @ b^T = [[1*7+2*8+3*9, 1*10+2*11+3*12],   (2x2)
     *            [4*7+5*8+6*9, 4*10+5*11+6*12]]
     *         = [[50, 68],
     *            [122,167]] */
    int32_t a_shape[] = {2, 3};
    int32_t b_shape[] = {2, 3};
    int32_t out_shape[] = {2, 2};

    Tensor *a = tensor_create(pool, a_shape, 2, DTYPE_FP32);
    Tensor *b = tensor_create(pool, b_shape, 2, DTYPE_FP32);
    Tensor *out = tensor_create(pool, out_shape, 2, DTYPE_FP32);

    int32_t idx[2];
    float a_vals[] = {1, 2, 3, 4, 5, 6};
    float b_vals[] = {7, 8, 9, 10, 11, 12};
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 3; c++) {
            idx[0] = r; idx[1] = c;
            tensor_set(a, idx, a_vals[r * 3 + c]);
            tensor_set(b, idx, b_vals[r * 3 + c]);
        }
    }

    op_matmul_nt(out, a, b);

    idx[0] = 0; idx[1] = 0; ASSERT_EQUAL_FLOAT(50.0f, tensor_get(out, idx), 1e-4f);
    idx[0] = 0; idx[1] = 1; ASSERT_EQUAL_FLOAT(68.0f, tensor_get(out, idx), 1e-4f);
    idx[0] = 1; idx[1] = 0; ASSERT_EQUAL_FLOAT(122.0f, tensor_get(out, idx), 1e-4f);
    idx[0] = 1; idx[1] = 1; ASSERT_EQUAL_FLOAT(167.0f, tensor_get(out, idx), 1e-4f);

    tensor_destroy(out);
    tensor_destroy(b);
    tensor_destroy(a);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 5: op_silu ---- */
static void test_op_silu(void) {
    TEST_BEGIN(op_silu);

    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    int32_t shape[] = {4};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    /* silu(0) = 0, silu(1) = 1*sigmoid(1) ≈ 0.7311, silu(-1) ≈ -0.2689 */
    int32_t i0[] = {0}, i1[] = {1}, i2[] = {2}, i3[] = {3};
    tensor_set(a, i0, 0.0f);
    tensor_set(a, i1, 1.0f);
    tensor_set(a, i2, -1.0f);
    tensor_set(a, i3, 5.0f);

    op_silu(out, a);

    ASSERT_EQUAL_FLOAT(0.0f, tensor_get(out, i0), 1e-5f);
    ASSERT_EQUAL_FLOAT(0.7311f, tensor_get(out, i1), 1e-3f);
    ASSERT_EQUAL_FLOAT(-0.2689f, tensor_get(out, i2), 1e-3f);
    /* silu(5) = 5 * sigmoid(5) ≈ 5 * 0.9933 = 4.9665 */
    ASSERT_EQUAL_FLOAT(4.9665f, tensor_get(out, i3), 1e-3f);

    tensor_destroy(out);
    tensor_destroy(a);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 6: op_silu_backward ---- */
static void test_op_silu_backward(void) {
    TEST_BEGIN(op_silu_backward);

    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    int32_t shape[] = {3};
    Tensor *a = tensor_create(pool, shape, 1, DTYPE_FP32);
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    /* silu'(0) = sigmoid(0) * (1 + 0*(1-sigmoid(0))) = 0.5
     * silu'(1) = sigmoid(1) * (1 + 1*(1-sigmoid(1))) ≈ 0.9277
     * silu'(-1) = sigmoid(-1) * (1 + (-1)*(1-sigmoid(-1))) ≈ 0.0723 */
    int32_t i0[] = {0}, i1[] = {1}, i2[] = {2};
    tensor_set(a, i0, 0.0f);
    tensor_set(a, i1, 1.0f);
    tensor_set(a, i2, -1.0f);

    op_silu_backward(out, a);

    ASSERT_EQUAL_FLOAT(0.5f, tensor_get(out, i0), 1e-4f);
    /* Numerical: sig(1) ≈ 0.7311, silu'(1) = 0.7311*(1+1*0.2689) = 0.9277 */
    ASSERT_EQUAL_FLOAT(0.9277f, tensor_get(out, i1), 1e-3f);

    tensor_destroy(out);
    tensor_destroy(a);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 7: op_cross_entropy ---- */
static void test_op_cross_entropy(void) {
    TEST_BEGIN(op_cross_entropy);

    MemoryPool *pool = pool_create(65536, POOL_SCRATCH);

    /* logits for 2 tokens, vocab_size=4 */
    int32_t shape[] = {2, 4};
    Tensor *logits = tensor_create(pool, shape, 2, DTYPE_FP32);

    /* Token 0: logits=[1, 2, 3, 4], target=2 (correct answer has logit 3)
     * Token 1: logits=[4, 3, 2, 1], target=0 (correct answer has logit 4) */
    int32_t idx[2];
    float vals[] = {1, 2, 3, 4, 4, 3, 2, 1};
    for (int r = 0; r < 2; r++) {
        for (int c = 0; c < 4; c++) {
            idx[0] = r; idx[1] = c;
            tensor_set(logits, idx, vals[r * 4 + c]);
        }
    }

    int32_t targets[] = {2, 0};
    float loss = op_cross_entropy(logits, targets, 2, NULL);

    /* Manual: for token 0, logits=[1,2,3,4], target=2
     * log_sum_exp = 4 + log(e^(-3)+e^(-2)+e^(-1)+e^0) = 4 + log(1+e^-1+e^-2+e^-3)
     * ≈ 4 + log(1.5820) ≈ 4.4586
     * loss_0 = -3 + 4.4586 = 1.4586
     * Same computation for token 1 (symmetric): loss_1 = 1.4586
     * avg = 1.4586 */
    /* Token 0: loss=-3+4.44=1.44, Token 1: loss=-4+4.44=0.44, avg=0.94 */
    ASSERT_TRUE(loss > 0.5f);
    ASSERT_TRUE(loss < 2.0f);
    ASSERT_TRUE(!isnan(loss));

    tensor_destroy(logits);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 8: op_cross_entropy_grad ---- */
static void test_op_cross_entropy_grad(void) {
    TEST_BEGIN(op_cross_entropy_grad);

    MemoryPool *pool = pool_create(65536, POOL_SCRATCH);

    int32_t shape[] = {1, 3};
    Tensor *logits = tensor_create(pool, shape, 2, DTYPE_FP32);
    Tensor *grad = tensor_create(pool, shape, 2, DTYPE_FP32);

    /* logits = [0, 0, 0], target = 1
     * softmax = [1/3, 1/3, 1/3]
     * grad = [1/3, 1/3-1, 1/3] = [1/3, -2/3, 1/3] (before /seq_len)
     * With seq_len=1: grad is same */
    int32_t idx[2];
    idx[0] = 0; idx[1] = 0; tensor_set(logits, idx, 0.0f);
    idx[0] = 0; idx[1] = 1; tensor_set(logits, idx, 0.0f);
    idx[0] = 0; idx[1] = 2; tensor_set(logits, idx, 0.0f);

    int32_t targets[] = {1};
    float loss = op_cross_entropy(logits, targets, 1, grad);

    /* loss = -log(1/3) = log(3) ≈ 1.0986 */
    ASSERT_EQUAL_FLOAT(logf(3.0f), loss, 1e-4f);

    /* grad[0,0] = 1/3, grad[0,1] = 1/3-1 = -2/3, grad[0,2] = 1/3 */
    idx[0] = 0; idx[1] = 0;
    ASSERT_EQUAL_FLOAT(1.0f / 3.0f, tensor_get(grad, idx), 1e-4f);
    idx[0] = 0; idx[1] = 1;
    ASSERT_EQUAL_FLOAT(-2.0f / 3.0f, tensor_get(grad, idx), 1e-4f);
    idx[0] = 0; idx[1] = 2;
    ASSERT_EQUAL_FLOAT(1.0f / 3.0f, tensor_get(grad, idx), 1e-4f);

    tensor_destroy(grad);
    tensor_destroy(logits);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 9: op_rand_uniform ---- */
static void test_op_rand_uniform(void) {
    TEST_BEGIN(op_rand_uniform);

    srand(42);
    MemoryPool *pool = pool_create(4096, POOL_SCRATCH);
    int32_t shape[] = {100};
    Tensor *out = tensor_create(pool, shape, 1, DTYPE_FP32);

    op_rand_uniform(out, -1.0f, 1.0f);

    /* All values should be in [-1, 1] */
    for (int32_t i = 0; i < 100; i++) {
        int32_t idx[] = {i};
        float val = tensor_get(out, idx);
        ASSERT_TRUE(val >= -1.0f && val <= 1.0f);
    }

    /* Mean should be approximately 0 for uniform(-1,1) */
    float sum = op_sum(out);
    float mean = sum / 100.0f;
    ASSERT_TRUE(fabsf(mean) < 0.3f); /* loose bound for 100 samples */

    tensor_destroy(out);
    pool_destroy(pool);

    TEST_END();
}

/* ---- TEST 10: op_init_xavier ---- */
static void test_op_init_xavier(void) {
    TEST_BEGIN(op_init_xavier);

    srand(42);
    MemoryPool *pool = pool_create(65536, POOL_SCRATCH);
    int32_t shape[] = {64, 64};
    Tensor *out = tensor_create(pool, shape, 2, DTYPE_FP32);

    op_init_xavier(out, 64, 64);

    /* Xavier limit = sqrt(6/(64+64)) = sqrt(6/128) ≈ 0.2165 */
    float limit = sqrtf(6.0f / 128.0f);
    size_t n = tensor_numel(out);

    for (size_t i = 0; i < n; i++) {
        int32_t idx[2] = {(int32_t)(i / 64), (int32_t)(i % 64)};
        float val = tensor_get(out, idx);
        ASSERT_TRUE(val >= -limit - 1e-6f && val <= limit + 1e-6f);
    }

    tensor_destroy(out);
    pool_destroy(pool);

    TEST_END();
}

int main(void) {
    printf("=== Training Ops Tests ===\n\n");

    printf("--- Element-wise ---\n");
    test_op_sub();
    test_op_add_scaled();

    printf("\n--- Transposed Matmul ---\n");
    test_op_matmul_tn();
    test_op_matmul_nt();

    printf("\n--- Activations ---\n");
    test_op_silu();
    test_op_silu_backward();

    printf("\n--- Loss ---\n");
    test_op_cross_entropy();
    test_op_cross_entropy_grad();

    printf("\n--- Initialization ---\n");
    test_op_rand_uniform();
    test_op_init_xavier();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
