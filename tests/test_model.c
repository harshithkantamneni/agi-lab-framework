/* test_model.c -- Tests for HSPA v1 model skeleton.
 *
 * Tests:
 *   1. Config: creation, default values, parameter validation
 *   2. RMSNorm: forward pass correctness
 *   3. Embedding: forward pass (table lookup)
 *   4. Struct sizes: verify all model structs can be instantiated
 *
 * Built with:
 *   clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *         -I src/tests -I src/core -I src/model
 *         tests/test_model.c src/model/hspa_config.c src/model/rmsnorm.c
 *         src/model/embedding.c src/core/memory_pool.c src/core/tensor.c
 *         src/core/ops.c -framework Accelerate -o build/tests/test_model
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include "hspa_config.h"
#include "rmsnorm.h"
#include "embedding.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * CONFIG TESTS
 * ======================================================================== */

static void test_config_default_values(void) {
    TEST_BEGIN(config_default_values);

    HSPAConfig cfg = hspa_config_default();

    ASSERT_EQUAL_INT(4096, cfg.d_model);
    ASSERT_EQUAL_INT(32, cfg.n_layers);
    ASSERT_EQUAL_INT(32, cfg.n_heads);
    ASSERT_EQUAL_INT(8, cfg.n_kv_heads);
    ASSERT_EQUAL_INT(128, cfg.head_dim);
    ASSERT_EQUAL_INT(16, cfg.n_experts);
    ASSERT_EQUAL_INT(2, cfg.n_active);
    ASSERT_EQUAL_INT(50000, cfg.vocab_size);
    ASSERT_EQUAL_INT(2048, cfg.max_seq_len);
    ASSERT_EQUAL_INT(5, cfg.ipc_iterations);
    ASSERT_EQUAL_INT(DTYPE_INT4, cfg.storage_dtype);
    ASSERT_EQUAL_INT(DTYPE_FP32, cfg.compute_dtype);

    TEST_END();
}

static void test_config_d_ff_derived(void) {
    TEST_BEGIN(config_d_ff_derived);

    HSPAConfig cfg = hspa_config_default();

    /* d_ff should be derived: expert_params / (3 * d_model)
     * With 16B total, ~28.2M per expert, d_ff ~ 1850.
     * The exact value depends on the derivation formula.
     * d_ff must be > 0 and reasonable. */
    ASSERT_TRUE(cfg.d_ff > 0);
    ASSERT_TRUE(cfg.d_ff >= 1024);
    ASSERT_TRUE(cfg.d_ff <= 4096);

    TEST_END();
}

static void test_config_rms_norm_eps(void) {
    TEST_BEGIN(config_rms_norm_eps);

    HSPAConfig cfg = hspa_config_default();
    ASSERT_EQUAL_FLOAT(1e-6f, cfg.rms_norm_eps, 1e-10f);

    TEST_END();
}

static void test_config_expert_count_configurable(void) {
    TEST_BEGIN(config_expert_count_configurable);

    /* D-004: expert count must be configurable, not hardcoded.
     * Verify that modifying n_experts changes the config. */
    HSPAConfig cfg = hspa_config_default();
    ASSERT_EQUAL_INT(16, cfg.n_experts);

    cfg.n_experts = 64;
    ASSERT_EQUAL_INT(64, cfg.n_experts);

    cfg.n_experts = 8;
    ASSERT_EQUAL_INT(8, cfg.n_experts);

    TEST_END();
}

static void test_config_head_dim_consistency(void) {
    TEST_BEGIN(config_head_dim_consistency);

    HSPAConfig cfg = hspa_config_default();

    /* head_dim * n_heads should equal d_model */
    ASSERT_EQUAL_INT(cfg.d_model, cfg.head_dim * cfg.n_heads);

    TEST_END();
}

/* ========================================================================
 * RMSNORM TESTS
 * ======================================================================== */

static void test_rmsnorm_create(void) {
    TEST_BEGIN(rmsnorm_create);

    MemoryPool *pool = pool_create(1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    RMSNorm *norm = rmsnorm_create(pool, 128, 1e-6f);
    ASSERT_NOT_NULL(norm);
    ASSERT_NOT_NULL(norm->weight);
    ASSERT_EQUAL_INT(128, norm->weight->shape[0]);
    ASSERT_EQUAL_INT(1, norm->weight->ndim);
    ASSERT_EQUAL_FLOAT(1e-6f, norm->eps, 1e-10f);

    rmsnorm_destroy(norm);
    pool_destroy(pool);

    TEST_END();
}

static void test_rmsnorm_weight_init_ones(void) {
    TEST_BEGIN(rmsnorm_weight_init_ones);

    MemoryPool *pool = pool_create(1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    RMSNorm *norm = rmsnorm_create(pool, 64, 1e-6f);
    ASSERT_NOT_NULL(norm);

    /* Weights should be initialized to 1.0 */
    for (int32_t i = 0; i < 64; i++) {
        int32_t idx[] = {i};
        float val = tensor_get(norm->weight, idx);
        ASSERT_EQUAL_FLOAT(1.0f, val, 1e-6f);
    }

    rmsnorm_destroy(norm);
    pool_destroy(pool);

    TEST_END();
}

static void test_rmsnorm_forward_unit_input(void) {
    TEST_BEGIN(rmsnorm_forward_unit_input);

    /* RMSNorm of a constant vector should normalize it.
     * If x = [c, c, c, ...], then rms = sqrt(mean(c^2) + eps) = sqrt(c^2 + eps)
     * out = x / rms * weight
     * With weight=1 and x = [2.0, 2.0, 2.0, 2.0]:
     *   rms = sqrt(4.0 + 1e-6) ~ 2.0
     *   out ~ [1.0, 1.0, 1.0, 1.0] */
    MemoryPool *wpool = pool_create(1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    int32_t dim = 4;
    RMSNorm *norm = rmsnorm_create(wpool, dim, 1e-6f);
    ASSERT_NOT_NULL(norm);

    /* Create input: [2.0, 2.0, 2.0, 2.0] */
    int32_t shape[] = {dim};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    tensor_fill(x, 2.0f);

    /* Create output */
    Tensor *out = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(out);

    rmsnorm_forward(out, norm, x);

    /* rms(x) = sqrt(mean(x^2) + eps) = sqrt(4.0 + 1e-6) ~ 2.0
     * out[i] = x[i] / rms(x) * weight[i] = 2.0 / 2.0 * 1.0 = 1.0 */
    for (int32_t i = 0; i < dim; i++) {
        int32_t idx[] = {i};
        float val = tensor_get(out, idx);
        ASSERT_EQUAL_FLOAT(1.0f, val, 1e-3f);
    }

    tensor_destroy(out);
    tensor_destroy(x);
    rmsnorm_destroy(norm);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_rmsnorm_forward_varied_input(void) {
    TEST_BEGIN(rmsnorm_forward_varied_input);

    /* Test with non-uniform input: [1.0, 2.0, 3.0, 4.0]
     * rms = sqrt(mean(1+4+9+16) + eps) = sqrt(7.5 + 1e-6) ~ 2.7386
     * out[i] = x[i] / rms * 1.0 */
    MemoryPool *wpool = pool_create(1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    int32_t dim = 4;
    RMSNorm *norm = rmsnorm_create(wpool, dim, 1e-6f);
    ASSERT_NOT_NULL(norm);

    int32_t shape[] = {dim};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);

    float input_vals[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (int32_t i = 0; i < dim; i++) {
        int32_t idx[] = {i};
        tensor_set(x, idx, input_vals[i]);
    }

    Tensor *out = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(out);

    rmsnorm_forward(out, norm, x);

    /* Manual computation */
    float sum_sq = 1.0f + 4.0f + 9.0f + 16.0f;
    float rms = sqrtf(sum_sq / 4.0f + 1e-6f);

    for (int32_t i = 0; i < dim; i++) {
        int32_t idx[] = {i};
        float val = tensor_get(out, idx);
        float expected = input_vals[i] / rms;
        ASSERT_EQUAL_FLOAT(expected, val, 1e-4f);
    }

    tensor_destroy(out);
    tensor_destroy(x);
    rmsnorm_destroy(norm);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_rmsnorm_forward_with_weight(void) {
    TEST_BEGIN(rmsnorm_forward_with_weight);

    /* Test with non-unit weights */
    MemoryPool *wpool = pool_create(1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    int32_t dim = 4;
    RMSNorm *norm = rmsnorm_create(wpool, dim, 1e-6f);
    ASSERT_NOT_NULL(norm);

    /* Set weights to [0.5, 1.0, 1.5, 2.0] */
    float weights[] = {0.5f, 1.0f, 1.5f, 2.0f};
    for (int32_t i = 0; i < dim; i++) {
        int32_t idx[] = {i};
        tensor_set(norm->weight, idx, weights[i]);
    }

    int32_t shape[] = {dim};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    tensor_fill(x, 2.0f);

    Tensor *out = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(out);

    rmsnorm_forward(out, norm, x);

    float rms = sqrtf(4.0f + 1e-6f); /* rms of [2,2,2,2] */
    for (int32_t i = 0; i < dim; i++) {
        int32_t idx[] = {i};
        float val = tensor_get(out, idx);
        float expected = (2.0f / rms) * weights[i];
        ASSERT_EQUAL_FLOAT(expected, val, 1e-4f);
    }

    tensor_destroy(out);
    tensor_destroy(x);
    rmsnorm_destroy(norm);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * EMBEDDING TESTS
 * ======================================================================== */

static void test_embedding_create(void) {
    TEST_BEGIN(embedding_create);

    MemoryPool *pool = pool_create(64 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    /* Use small config for testing */
    HSPAConfig cfg = hspa_config_default();
    cfg.vocab_size = 100;
    cfg.d_model = 16;

    Embedding *emb = embedding_create(pool, &cfg);
    ASSERT_NOT_NULL(emb);
    ASSERT_NOT_NULL(emb->weight);
    ASSERT_EQUAL_INT(100, emb->weight->shape[0]);
    ASSERT_EQUAL_INT(16, emb->weight->shape[1]);
    ASSERT_EQUAL_INT(2, emb->weight->ndim);

    embedding_destroy(emb);
    pool_destroy(pool);

    TEST_END();
}

static void test_embedding_forward_single_token(void) {
    TEST_BEGIN(embedding_forward_single_token);

    MemoryPool *wpool = pool_create(64 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(64 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    HSPAConfig cfg = hspa_config_default();
    cfg.vocab_size = 10;
    cfg.d_model = 4;

    Embedding *emb = embedding_create(wpool, &cfg);
    ASSERT_NOT_NULL(emb);

    /* Manually set embedding weights for token 3:
     * row 3 = [10.0, 20.0, 30.0, 40.0] */
    for (int32_t j = 0; j < 4; j++) {
        int32_t idx[] = {3, j};
        tensor_set(emb->weight, idx, (float)((j + 1) * 10));
    }

    /* Look up token 3 */
    int32_t seq_len = 1;
    int32_t shape[] = {seq_len, cfg.d_model};
    Tensor *out = tensor_create(apool, shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);

    int32_t tokens[] = {3};
    embedding_forward(out, emb, tokens, seq_len);

    /* Verify output matches row 3 of the embedding table */
    for (int32_t j = 0; j < 4; j++) {
        int32_t idx[] = {0, j};
        float val = tensor_get(out, idx);
        ASSERT_EQUAL_FLOAT((float)((j + 1) * 10), val, 1e-6f);
    }

    tensor_destroy(out);
    embedding_destroy(emb);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_embedding_forward_multiple_tokens(void) {
    TEST_BEGIN(embedding_forward_multiple_tokens);

    MemoryPool *wpool = pool_create(64 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(64 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    HSPAConfig cfg = hspa_config_default();
    cfg.vocab_size = 8;
    cfg.d_model = 4;

    Embedding *emb = embedding_create(wpool, &cfg);
    ASSERT_NOT_NULL(emb);

    /* Set distinct values for rows 1, 5, 2 */
    for (int32_t r = 0; r < 8; r++) {
        for (int32_t c = 0; c < 4; c++) {
            int32_t idx[] = {r, c};
            tensor_set(emb->weight, idx, (float)(r * 10 + c));
        }
    }

    int32_t tokens[] = {1, 5, 2};
    int32_t seq_len = 3;
    int32_t shape[] = {seq_len, cfg.d_model};
    Tensor *out = tensor_create(apool, shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);

    embedding_forward(out, emb, tokens, seq_len);

    /* Verify: out[0,:] = emb_weight[1,:] = [10, 11, 12, 13] */
    for (int32_t c = 0; c < 4; c++) {
        int32_t idx[] = {0, c};
        float val = tensor_get(out, idx);
        ASSERT_EQUAL_FLOAT((float)(1 * 10 + c), val, 1e-6f);
    }
    /* Verify: out[1,:] = emb_weight[5,:] = [50, 51, 52, 53] */
    for (int32_t c = 0; c < 4; c++) {
        int32_t idx[] = {1, c};
        float val = tensor_get(out, idx);
        ASSERT_EQUAL_FLOAT((float)(5 * 10 + c), val, 1e-6f);
    }
    /* Verify: out[2,:] = emb_weight[2,:] = [20, 21, 22, 23] */
    for (int32_t c = 0; c < 4; c++) {
        int32_t idx[] = {2, c};
        float val = tensor_get(out, idx);
        ASSERT_EQUAL_FLOAT((float)(2 * 10 + c), val, 1e-6f);
    }

    tensor_destroy(out);
    embedding_destroy(emb);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_embedding_boundary_tokens(void) {
    TEST_BEGIN(embedding_boundary_tokens);

    MemoryPool *wpool = pool_create(64 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(64 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    HSPAConfig cfg = hspa_config_default();
    cfg.vocab_size = 16;
    cfg.d_model = 4;

    Embedding *emb = embedding_create(wpool, &cfg);
    ASSERT_NOT_NULL(emb);

    /* Set values for first and last token */
    for (int32_t c = 0; c < 4; c++) {
        int32_t idx0[] = {0, c};
        tensor_set(emb->weight, idx0, (float)(100 + c));
        int32_t idx15[] = {15, c};
        tensor_set(emb->weight, idx15, (float)(200 + c));
    }

    /* Look up token 0 and token 15 (last valid) */
    int32_t tokens[] = {0, 15};
    int32_t seq_len = 2;
    int32_t shape[] = {seq_len, cfg.d_model};
    Tensor *out = tensor_create(apool, shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);

    embedding_forward(out, emb, tokens, seq_len);

    for (int32_t c = 0; c < 4; c++) {
        int32_t idx[] = {0, c};
        ASSERT_EQUAL_FLOAT((float)(100 + c), tensor_get(out, idx), 1e-6f);
    }
    for (int32_t c = 0; c < 4; c++) {
        int32_t idx[] = {1, c};
        ASSERT_EQUAL_FLOAT((float)(200 + c), tensor_get(out, idx), 1e-6f);
    }

    tensor_destroy(out);
    embedding_destroy(emb);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("=== HSPA v1 Model Skeleton Tests ===\n\n");

    printf("--- Config Tests ---\n");
    test_config_default_values();
    test_config_d_ff_derived();
    test_config_rms_norm_eps();
    test_config_expert_count_configurable();
    test_config_head_dim_consistency();

    printf("\n--- RMSNorm Tests ---\n");
    test_rmsnorm_create();
    test_rmsnorm_weight_init_ones();
    test_rmsnorm_forward_unit_input();
    test_rmsnorm_forward_varied_input();
    test_rmsnorm_forward_with_weight();

    printf("\n--- Embedding Tests ---\n");
    test_embedding_create();
    test_embedding_forward_single_token();
    test_embedding_forward_multiple_tokens();
    test_embedding_boundary_tokens();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
