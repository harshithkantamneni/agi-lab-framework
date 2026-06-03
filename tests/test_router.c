/* test_router.c -- Tests for FEP (Free Energy Principle) expert router.
 *
 * Tests:
 *   1. router_create: struct allocation, tensor shapes, sigma_prior
 *   2. router_create_null_checks: NULL pool returns NULL
 *   3. router_forward_returns_valid: n_active experts returned, weights sum ~1.0
 *   4. router_forward_deterministic_inference: same input + training=false => same output
 *   5. router_forward_expert_ids_valid: all expert_ids in [0, n_experts)
 *   6. router_forward_weights_sum_to_one: weights sum to 1.0 within tolerance
 *   7. router_entropy_range: entropy in [0, log(n_experts)]
 *   8. router_forward_biased_weights: strongly biased W_mu => preferred expert selected
 *
 * Uses SMALL dimensions: d_model=8, n_experts=4, n_active=2
 *
 * Built with:
 *   clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *         -Isrc/core -Isrc/model -Isrc/tests
 *         tests/test_router.c src/model/router.c src/model/hspa_config.c
 *         src/core/memory_pool.c src/core/tensor.c src/core/ops.c
 *         -framework Accelerate -o build/tests/test_router
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include "hspa_config.h"
#include "router.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================
 * HELPER: Create a small test config (d_model=8, n_experts=4, n_active=2)
 * ======================================================================== */

static HSPAConfig test_config(void) {
    HSPAConfig cfg = hspa_config_default();
    cfg.d_model   = 8;
    cfg.n_experts = 4;
    cfg.n_active  = 2;
    cfg.d_ff      = 16;
    return cfg;
}

/* ========================================================================
 * ROUTER CREATE TESTS
 * ======================================================================== */

static void test_router_create(void) {
    TEST_BEGIN(router_create);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    FEPRouter *router = router_create(pool, &cfg);
    ASSERT_NOT_NULL(router);

    /* W_mu shape: [d_model, n_experts] = [8, 4] */
    ASSERT_NOT_NULL(router->W_mu);
    ASSERT_EQUAL_INT(2, router->W_mu->ndim);
    ASSERT_EQUAL_INT(8, router->W_mu->shape[0]);
    ASSERT_EQUAL_INT(4, router->W_mu->shape[1]);

    /* W_sigma shape: [d_model, n_experts] = [8, 4] */
    ASSERT_NOT_NULL(router->W_sigma);
    ASSERT_EQUAL_INT(2, router->W_sigma->ndim);
    ASSERT_EQUAL_INT(8, router->W_sigma->shape[0]);
    ASSERT_EQUAL_INT(4, router->W_sigma->shape[1]);

    /* sigma_prior = 1.0 */
    ASSERT_EQUAL_FLOAT(1.0f, router->sigma_prior, 1e-6f);

    router_destroy(router);
    pool_destroy(pool);

    TEST_END();
}

static void test_router_create_null_checks(void) {
    TEST_BEGIN(router_create_null_checks);

    HSPAConfig cfg = test_config();

    /* NULL pool should return NULL */
    FEPRouter *router = router_create(NULL, &cfg);
    ASSERT_NULL(router);

    /* NULL config should return NULL */
    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    router = router_create(pool, NULL);
    ASSERT_NULL(router);

    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * ROUTER FORWARD TESTS
 * ======================================================================== */

static void test_router_forward_returns_valid(void) {
    TEST_BEGIN(router_forward_returns_valid);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    FEPRouter *router = router_create(wpool, &cfg);
    ASSERT_NOT_NULL(router);

    /* Create input x [d_model=8] */
    int32_t shape[] = {8};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    /* Fill with some non-zero values */
    for (int32_t i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(x, idx, (float)(i + 1) * 0.1f);
    }

    RoutingDecision rd = router_forward(router, x, &cfg, false);

    /* Should have n_active=2 experts selected */
    ASSERT_NOT_NULL(rd.expert_ids);
    ASSERT_NOT_NULL(rd.expert_weights);

    /* Weights should sum to ~1.0 */
    float wsum = 0.0f;
    for (int32_t i = 0; i < cfg.n_active; i++) {
        wsum += rd.expert_weights[i];
    }
    ASSERT_EQUAL_FLOAT(1.0f, wsum, 1e-4f);

    free(rd.expert_ids);
    free(rd.expert_weights);
    tensor_destroy(x);
    router_destroy(router);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_router_forward_deterministic_inference(void) {
    TEST_BEGIN(router_forward_deterministic_inference);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    FEPRouter *router = router_create(wpool, &cfg);
    ASSERT_NOT_NULL(router);

    /* Set W_mu to deterministic values so the router picks consistent experts */
    for (int32_t i = 0; i < 8; i++) {
        for (int32_t j = 0; j < 4; j++) {
            int32_t idx[] = {i, j};
            tensor_set(router->W_mu, idx, (float)(i * 4 + j) * 0.01f);
        }
    }

    /* Create input x */
    int32_t shape[] = {8};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    for (int32_t i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(x, idx, (float)(i + 1) * 0.5f);
    }

    /* Run twice with training=false */
    RoutingDecision rd1 = router_forward(router, x, &cfg, false);
    RoutingDecision rd2 = router_forward(router, x, &cfg, false);

    ASSERT_NOT_NULL(rd1.expert_ids);
    ASSERT_NOT_NULL(rd2.expert_ids);

    /* Same expert IDs */
    for (int32_t i = 0; i < cfg.n_active; i++) {
        ASSERT_EQUAL_INT(rd1.expert_ids[i], rd2.expert_ids[i]);
    }

    /* Same weights */
    for (int32_t i = 0; i < cfg.n_active; i++) {
        ASSERT_EQUAL_FLOAT(rd1.expert_weights[i], rd2.expert_weights[i], 1e-6f);
    }

    /* Same entropy */
    ASSERT_EQUAL_FLOAT(rd1.entropy, rd2.entropy, 1e-6f);

    free(rd1.expert_ids);
    free(rd1.expert_weights);
    free(rd2.expert_ids);
    free(rd2.expert_weights);
    tensor_destroy(x);
    router_destroy(router);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_router_forward_expert_ids_valid(void) {
    TEST_BEGIN(router_forward_expert_ids_valid);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    FEPRouter *router = router_create(wpool, &cfg);
    ASSERT_NOT_NULL(router);

    int32_t shape[] = {8};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    for (int32_t i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(x, idx, (float)(i + 1) * 0.3f);
    }

    RoutingDecision rd = router_forward(router, x, &cfg, false);
    ASSERT_NOT_NULL(rd.expert_ids);

    /* All expert IDs must be in [0, n_experts) */
    for (int32_t i = 0; i < cfg.n_active; i++) {
        ASSERT_TRUE(rd.expert_ids[i] >= 0);
        ASSERT_TRUE(rd.expert_ids[i] < cfg.n_experts);
    }

    /* Expert IDs must be unique */
    for (int32_t i = 0; i < cfg.n_active; i++) {
        for (int32_t j = i + 1; j < cfg.n_active; j++) {
            ASSERT_TRUE(rd.expert_ids[i] != rd.expert_ids[j]);
        }
    }

    free(rd.expert_ids);
    free(rd.expert_weights);
    tensor_destroy(x);
    router_destroy(router);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_router_forward_weights_sum_to_one(void) {
    TEST_BEGIN(router_forward_weights_sum_to_one);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    FEPRouter *router = router_create(wpool, &cfg);
    ASSERT_NOT_NULL(router);

    /* Try several different inputs */
    float test_inputs[3][8] = {
        {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
        {-1.0f, 2.0f, -0.5f, 0.3f, 1.5f, -0.1f, 0.9f, -0.7f}
    };

    int32_t shape[] = {8};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);

    for (int t = 0; t < 3; t++) {
        for (int32_t i = 0; i < 8; i++) {
            int32_t idx[] = {i};
            tensor_set(x, idx, test_inputs[t][i]);
        }

        RoutingDecision rd = router_forward(router, x, &cfg, false);
        ASSERT_NOT_NULL(rd.expert_weights);

        float wsum = 0.0f;
        for (int32_t i = 0; i < cfg.n_active; i++) {
            wsum += rd.expert_weights[i];
            /* Each weight should be positive */
            ASSERT_TRUE(rd.expert_weights[i] > 0.0f);
        }
        ASSERT_EQUAL_FLOAT(1.0f, wsum, 1e-4f);

        free(rd.expert_ids);
        free(rd.expert_weights);
    }

    tensor_destroy(x);
    router_destroy(router);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_router_entropy_range(void) {
    TEST_BEGIN(router_entropy_range);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    FEPRouter *router = router_create(wpool, &cfg);
    ASSERT_NOT_NULL(router);

    int32_t shape[] = {8};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    for (int32_t i = 0; i < 8; i++) {
        int32_t idx[] = {i};
        tensor_set(x, idx, (float)(i + 1) * 0.2f);
    }

    RoutingDecision rd = router_forward(router, x, &cfg, false);

    /* Entropy must be in [0, log(n_experts)] = [0, log(4)] = [0, 1.386...] */
    float max_entropy = logf((float)cfg.n_experts);
    ASSERT_TRUE(rd.entropy >= 0.0f);
    ASSERT_TRUE(rd.entropy <= max_entropy + 1e-4f);

    free(rd.expert_ids);
    free(rd.expert_weights);
    tensor_destroy(x);
    router_destroy(router);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

static void test_router_forward_biased_weights(void) {
    TEST_BEGIN(router_forward_biased_weights);

    /* FEP routing prefers experts with moderate uncertainty (epistemic
     * exploration): info_gain > ambiguity when sigma is near the sweet
     * spot (~1.0 with sigma_prior=1.0).  We bias W_sigma so expert 2
     * has sigma ~1.0 while others have sigma ~0.  With uniform logits,
     * the only G differentiator is (ambiguity - info_gain), which is
     * negative for expert 2 and ~0 for the rest.  Expert 2 should be
     * selected. */
    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    FEPRouter *router = router_create(wpool, &cfg);
    ASSERT_NOT_NULL(router);

    /* W_mu: all zero → uniform routing logits → mu_probs ≈ 0.25 each
     * → complexity terms cancel out. */
    tensor_fill(router->W_mu, 0.0f);

    /* W_sigma[:, 2] = 0.07 → raw = 8*0.07 = 0.56 → sigma = softplus(0.56) ≈ 1.01
     * W_sigma[:, others] = -1.0 → raw = -8 → sigma = softplus(-8) ≈ 0
     *
     * For expert 2:  ambiguity = 0.51,  info_gain = 0.70  → net = -0.19
     * For others:    ambiguity ≈ 0,     info_gain ≈ 0     → net =  0
     * Expert 2 has LOWEST G → selected. */
    tensor_fill(router->W_sigma, -1.0f);
    for (int32_t i = 0; i < 8; i++) {
        int32_t idx[] = {i, 2};
        tensor_set(router->W_sigma, idx, 0.07f);
    }

    int32_t shape[] = {8};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    tensor_fill(x, 1.0f);

    RoutingDecision rd = router_forward(router, x, &cfg, false);
    ASSERT_NOT_NULL(rd.expert_ids);

    /* Expert 2 should be one of the selected experts */
    bool found_expert_2 = false;
    for (int32_t i = 0; i < cfg.n_active; i++) {
        if (rd.expert_ids[i] == 2) {
            found_expert_2 = true;
            break;
        }
    }
    ASSERT_TRUE(found_expert_2);

    free(rd.expert_ids);
    free(rd.expert_weights);
    tensor_destroy(x);
    router_destroy(router);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("=== FEP Router Tests ===\n\n");

    printf("--- Router Create Tests ---\n");
    test_router_create();
    test_router_create_null_checks();

    printf("\n--- Router Forward Tests ---\n");
    test_router_forward_returns_valid();
    test_router_forward_deterministic_inference();
    test_router_forward_expert_ids_valid();
    test_router_forward_weights_sum_to_one();
    test_router_entropy_range();
    test_router_forward_biased_weights();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
