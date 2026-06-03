/* test_ffn.c -- Tests for SwiGLU Expert Feed-Forward Network.
 *
 * Tests:
 *   1. test_ffn_create          -- verify weight shapes
 *   2. test_ffn_create_null_checks -- NULL inputs return NULL
 *   3. test_ffn_forward_identity   -- identity-like weights, verify output
 *   4. test_ffn_forward_numerical  -- hand-computed SwiGLU, exact comparison
 *   5. test_ffn_forward_zero_input -- zero input produces zero output
 *   6. test_ffn_destroy_null       -- destroying NULL must not crash
 *
 * Built with:
 *   clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *         -Isrc/core -Isrc/model -Isrc/tests
 *         tests/test_ffn.c src/model/ffn.c src/core/ops.c ...
 *         -framework Accelerate -o build/tests/test_ffn
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include "hspa_config.h"
#include "ffn.h"

#include <math.h>
#include <stdlib.h>

/* ========================================================================
 * Helper: silu(x) = x / (1 + exp(-x))
 * ======================================================================== */
static float silu(float x) {
    return x / (1.0f + expf(-x));
}

/* ========================================================================
 * TEST 1: expert_ffn_create -- verify weight shapes
 * ======================================================================== */
static void test_ffn_create(void) {
    TEST_BEGIN(ffn_create);

    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    /* Use small dims for testing */
    HSPAConfig cfg = hspa_config_default();
    cfg.d_model = 8;
    cfg.d_ff = 4;

    ExpertFFN *ffn = expert_ffn_create(pool, &cfg);
    ASSERT_NOT_NULL(ffn);

    /* W_gate: [d_model, d_ff] = [8, 4] */
    ASSERT_NOT_NULL(ffn->W_gate);
    ASSERT_EQUAL_INT(2, ffn->W_gate->ndim);
    ASSERT_EQUAL_INT(8, ffn->W_gate->shape[0]);
    ASSERT_EQUAL_INT(4, ffn->W_gate->shape[1]);

    /* W_up: [d_model, d_ff] = [8, 4] */
    ASSERT_NOT_NULL(ffn->W_up);
    ASSERT_EQUAL_INT(2, ffn->W_up->ndim);
    ASSERT_EQUAL_INT(8, ffn->W_up->shape[0]);
    ASSERT_EQUAL_INT(4, ffn->W_up->shape[1]);

    /* W_down: [d_ff, d_model] = [4, 8] */
    ASSERT_NOT_NULL(ffn->W_down);
    ASSERT_EQUAL_INT(2, ffn->W_down->ndim);
    ASSERT_EQUAL_INT(4, ffn->W_down->shape[0]);
    ASSERT_EQUAL_INT(8, ffn->W_down->shape[1]);

    expert_ffn_destroy(ffn);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 2: expert_ffn_create -- NULL checks
 * ======================================================================== */
static void test_ffn_create_null_checks(void) {
    TEST_BEGIN(ffn_create_null_checks);

    MemoryPool *pool = pool_create(1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    HSPAConfig cfg = hspa_config_default();
    cfg.d_model = 8;
    cfg.d_ff = 4;

    /* NULL pool should return NULL */
    ExpertFFN *r1 = expert_ffn_create(NULL, &cfg);
    ASSERT_NULL(r1);

    /* NULL config should return NULL */
    ExpertFFN *r2 = expert_ffn_create(pool, NULL);
    ASSERT_NULL(r2);

    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 3: expert_ffn_forward -- identity-like weights
 *
 * Set up so that we can verify the output has the correct shape and
 * plausible values.  Use W_gate = W_up = identity-padded, W_down = identity.
 * With d_model=4, d_ff=4 (square), x = [1,2,3,4]:
 *   gate = x @ I = x,  up = x @ I = x
 *   h = silu(x) * x
 *   out = h @ I = h = silu(x) * x
 * ======================================================================== */
static void test_ffn_forward_identity(void) {
    TEST_BEGIN(ffn_forward_identity);

    /* Use d_model=4, d_ff=4 (square) so identity matrices work cleanly */
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    MemoryPool *scratch = pool_create(4 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);
    ASSERT_NOT_NULL(scratch);

    HSPAConfig cfg = hspa_config_default();
    cfg.d_model = 4;
    cfg.d_ff = 4;

    ExpertFFN *ffn = expert_ffn_create(wpool, &cfg);
    ASSERT_NOT_NULL(ffn);

    /* Set W_gate, W_up, W_down to identity matrices */
    for (int32_t i = 0; i < 4; i++) {
        for (int32_t j = 0; j < 4; j++) {
            float v = (i == j) ? 1.0f : 0.0f;
            int32_t idx[] = {i, j};
            tensor_set(ffn->W_gate, idx, v);
            tensor_set(ffn->W_up, idx, v);
            tensor_set(ffn->W_down, idx, v);
        }
    }

    /* Input: x = [[1, 2, 3, 4]] (seq_len=1, d_model=4) */
    int32_t x_shape[] = {1, 4};
    Tensor *x = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    float x_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int32_t j = 0; j < 4; j++) {
        int32_t idx[] = {0, j};
        tensor_set(x, idx, x_vals[j]);
    }

    /* Output: out = [[?, ?, ?, ?]] (seq_len=1, d_model=4) */
    int32_t out_shape[] = {1, 4};
    Tensor *out = tensor_create(apool, out_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);

    expert_ffn_forward(out, ffn, x, scratch, NULL);

    /* With identity weights:
     *   gate = x, up = x, h = silu(x) * x, out = h
     * So out[j] = silu(x[j]) * x[j] */
    for (int32_t j = 0; j < 4; j++) {
        int32_t idx[] = {0, j};
        float actual = tensor_get(out, idx);
        float expected = silu(x_vals[j]) * x_vals[j];
        ASSERT_EQUAL_FLOAT(expected, actual, 1e-4f);
    }

    tensor_destroy(out);
    tensor_destroy(x);
    expert_ffn_destroy(ffn);
    pool_destroy(scratch);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 4: expert_ffn_forward -- full numerical verification
 *
 * d_model=4, d_ff=2, seq_len=1.
 * x = [1.0, 0.5, -1.0, 2.0]
 *
 * W_gate = [[0.1, 0.2],    (4x2)
 *           [0.3, 0.4],
 *           [0.5, 0.6],
 *           [0.7, 0.8]]
 *
 * W_up = [[0.2, 0.1],      (4x2)
 *         [0.4, 0.3],
 *         [0.6, 0.5],
 *         [0.8, 0.7]]
 *
 * W_down = [[1.0, 0.5, 0.0, -0.5],   (2x4)
 *           [0.5, 1.0, 0.5,  0.0]]
 *
 * Manual computation:
 *   gate = x @ W_gate
 *     gate[0] = 1*0.1 + 0.5*0.3 + (-1)*0.5 + 2*0.7 = 0.1+0.15-0.5+1.4 = 1.15
 *     gate[1] = 1*0.2 + 0.5*0.4 + (-1)*0.6 + 2*0.8 = 0.2+0.2-0.6+1.6  = 1.40
 *
 *   up = x @ W_up
 *     up[0] = 1*0.2 + 0.5*0.4 + (-1)*0.6 + 2*0.8 = 0.2+0.2-0.6+1.6 = 1.40
 *     up[1] = 1*0.1 + 0.5*0.3 + (-1)*0.5 + 2*0.7 = 0.1+0.15-0.5+1.4 = 1.15
 *
 *   h = silu(gate) * up
 *     silu(1.15) = 1.15 / (1 + exp(-1.15))
 *     silu(1.40) = 1.40 / (1 + exp(-1.40))
 *     h[0] = silu(1.15) * 1.40
 *     h[1] = silu(1.40) * 1.15
 *
 *   out = h @ W_down (h is [1,2], W_down is [2,4], out is [1,4])
 *     out[j] = h[0]*W_down[0,j] + h[1]*W_down[1,j]
 * ======================================================================== */
static void test_ffn_forward_numerical(void) {
    TEST_BEGIN(ffn_forward_numerical);

    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    MemoryPool *scratch = pool_create(4 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);
    ASSERT_NOT_NULL(scratch);

    int32_t d_model = 4;
    int32_t d_ff = 2;

    HSPAConfig cfg = hspa_config_default();
    cfg.d_model = d_model;
    cfg.d_ff = d_ff;

    ExpertFFN *ffn = expert_ffn_create(wpool, &cfg);
    ASSERT_NOT_NULL(ffn);

    /* Set W_gate [4,2] */
    float w_gate_vals[4][2] = {
        {0.1f, 0.2f},
        {0.3f, 0.4f},
        {0.5f, 0.6f},
        {0.7f, 0.8f}
    };
    for (int32_t i = 0; i < d_model; i++) {
        for (int32_t j = 0; j < d_ff; j++) {
            int32_t idx[] = {i, j};
            tensor_set(ffn->W_gate, idx, w_gate_vals[i][j]);
        }
    }

    /* Set W_up [4,2] */
    float w_up_vals[4][2] = {
        {0.2f, 0.1f},
        {0.4f, 0.3f},
        {0.6f, 0.5f},
        {0.8f, 0.7f}
    };
    for (int32_t i = 0; i < d_model; i++) {
        for (int32_t j = 0; j < d_ff; j++) {
            int32_t idx[] = {i, j};
            tensor_set(ffn->W_up, idx, w_up_vals[i][j]);
        }
    }

    /* Set W_down [2,4] */
    float w_down_vals[2][4] = {
        { 1.0f, 0.5f, 0.0f, -0.5f},
        { 0.5f, 1.0f, 0.5f,  0.0f}
    };
    for (int32_t i = 0; i < d_ff; i++) {
        for (int32_t j = 0; j < d_model; j++) {
            int32_t idx[] = {i, j};
            tensor_set(ffn->W_down, idx, w_down_vals[i][j]);
        }
    }

    /* Input: x = [[1.0, 0.5, -1.0, 2.0]] */
    float x_vals[] = {1.0f, 0.5f, -1.0f, 2.0f};
    int32_t x_shape[] = {1, d_model};
    Tensor *x = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    for (int32_t j = 0; j < d_model; j++) {
        int32_t idx[] = {0, j};
        tensor_set(x, idx, x_vals[j]);
    }

    /* Allocate output [1, d_model] */
    int32_t out_shape[] = {1, d_model};
    Tensor *out = tensor_create(apool, out_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);

    expert_ffn_forward(out, ffn, x, scratch, NULL);

    /* --- Manual expected computation --- */

    /* gate = x @ W_gate: [1,4] @ [4,2] = [1,2] */
    float gate[2];
    gate[0] = 1.0f*0.1f + 0.5f*0.3f + (-1.0f)*0.5f + 2.0f*0.7f; /* 1.15 */
    gate[1] = 1.0f*0.2f + 0.5f*0.4f + (-1.0f)*0.6f + 2.0f*0.8f; /* 1.40 */

    /* up = x @ W_up: [1,4] @ [4,2] = [1,2] */
    float up[2];
    up[0] = 1.0f*0.2f + 0.5f*0.4f + (-1.0f)*0.6f + 2.0f*0.8f;   /* 1.40 */
    up[1] = 1.0f*0.1f + 0.5f*0.3f + (-1.0f)*0.5f + 2.0f*0.7f;   /* 1.15 */

    /* h = silu(gate) * up */
    float h[2];
    h[0] = silu(gate[0]) * up[0];
    h[1] = silu(gate[1]) * up[1];

    /* out = h @ W_down: [1,2] @ [2,4] = [1,4] */
    float expected[4];
    for (int32_t j = 0; j < d_model; j++) {
        expected[j] = h[0] * w_down_vals[0][j] + h[1] * w_down_vals[1][j];
    }

    /* Verify each output element */
    for (int32_t j = 0; j < d_model; j++) {
        int32_t idx[] = {0, j};
        float actual = tensor_get(out, idx);
        ASSERT_EQUAL_FLOAT(expected[j], actual, 1e-4f);
    }

    tensor_destroy(out);
    tensor_destroy(x);
    expert_ffn_destroy(ffn);
    pool_destroy(scratch);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 5: expert_ffn_forward -- zero input produces zero output
 *
 * silu(0) = 0 / (1 + exp(0)) = 0 / 2 = 0
 * So gate = 0 @ W = 0, up = 0 @ W = 0, h = silu(0) * 0 = 0, out = 0 @ W = 0.
 * ======================================================================== */
static void test_ffn_forward_zero_input(void) {
    TEST_BEGIN(ffn_forward_zero_input);

    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    MemoryPool *scratch = pool_create(4 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);
    ASSERT_NOT_NULL(scratch);

    HSPAConfig cfg = hspa_config_default();
    cfg.d_model = 8;
    cfg.d_ff = 4;

    ExpertFFN *ffn = expert_ffn_create(wpool, &cfg);
    ASSERT_NOT_NULL(ffn);

    /* Fill weights with non-zero values so only the zero input matters */
    tensor_fill(ffn->W_gate, 0.5f);
    tensor_fill(ffn->W_up, 0.3f);
    tensor_fill(ffn->W_down, 0.7f);

    /* Input: all zeros, seq_len=2 */
    int32_t x_shape[] = {2, 8};
    Tensor *x = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    tensor_fill(x, 0.0f);

    int32_t out_shape[] = {2, 8};
    Tensor *out = tensor_create(apool, out_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);
    tensor_fill(out, 999.0f); /* pre-fill with garbage to prove forward overwrites */

    expert_ffn_forward(out, ffn, x, scratch, NULL);

    /* Every output element should be 0.0 */
    for (int32_t i = 0; i < 2; i++) {
        for (int32_t j = 0; j < 8; j++) {
            int32_t idx[] = {i, j};
            float val = tensor_get(out, idx);
            ASSERT_EQUAL_FLOAT(0.0f, val, 1e-6f);
        }
    }

    tensor_destroy(out);
    tensor_destroy(x);
    expert_ffn_destroy(ffn);
    pool_destroy(scratch);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 6: expert_ffn_destroy -- NULL safety
 * ======================================================================== */
static void test_ffn_destroy_null(void) {
    TEST_BEGIN(ffn_destroy_null);

    /* Must not crash */
    expert_ffn_destroy(NULL);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */
int main(void) {
    printf("=== Expert FFN (SwiGLU) Tests ===\n\n");

    test_ffn_create();
    test_ffn_create_null_checks();
    test_ffn_forward_identity();
    test_ffn_forward_numerical();
    test_ffn_forward_zero_input();
    test_ffn_destroy_null();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
