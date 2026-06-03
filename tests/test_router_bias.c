/* test_router_bias.c -- Tests for auxiliary-loss-free expert load balancing.
 *
 * Tests:
 *   1. test_bias_init_zero: biases start at 0, EMA at 1/n_experts
 *   2. test_bias_update_overloaded: overloaded expert's bias decreases
 *   3. test_bias_update_underloaded: underloaded expert's bias increases
 *   4. test_bias_affects_routing: positive bias shifts routing toward that expert
 *   5. test_bias_convergence: repeated updates push biases toward balance
 *
 * Uses SMALL dimensions: d_model=8, n_experts=4, n_active=2
 *
 * Built with:
 *   clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *         -Isrc/core -Isrc/model -Isrc/tests
 *         tests/test_router_bias.c src/model/router.c src/model/hspa_config.c
 *         src/core/memory_pool.c src/core/tensor.c src/core/ops.c
 *         -framework Accelerate -o build/tests/test_router_bias
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
 * TEST 1: Biases initialize to zero, EMA to uniform
 * ======================================================================== */

static void test_bias_init_zero(void) {
    TEST_BEGIN(bias_init_zero);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    FEPRouter *router = router_create(pool, &cfg);
    ASSERT_NOT_NULL(router);

    /* All expert biases should be 0.0 */
    ASSERT_NOT_NULL(router->expert_bias);
    for (int32_t j = 0; j < cfg.n_experts; j++) {
        ASSERT_EQUAL_FLOAT(0.0f, router->expert_bias[j], 1e-8f);
    }

    /* EMA should be 1/n_experts = 0.25 for 4 experts */
    ASSERT_NOT_NULL(router->expert_load_ema);
    float expected_ema = 1.0f / (float)cfg.n_experts;
    for (int32_t j = 0; j < cfg.n_experts; j++) {
        ASSERT_EQUAL_FLOAT(expected_ema, router->expert_load_ema[j], 1e-8f);
    }

    /* Check default hyperparameters */
    ASSERT_EQUAL_FLOAT(0.01f, router->bias_update_rate, 1e-8f);
    ASSERT_EQUAL_FLOAT(0.001f, router->bias_step_size, 1e-8f);

    /* Cached n_experts */
    ASSERT_EQUAL_INT(cfg.n_experts, router->n_experts);

    router_destroy(router);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 2: Overloaded expert's bias decreases
 * ======================================================================== */

static void test_bias_update_overloaded(void) {
    TEST_BEGIN(bias_update_overloaded);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    FEPRouter *router = router_create(pool, &cfg);
    ASSERT_NOT_NULL(router);

    /* Expert 0 gets ALL tokens, others get none */
    int32_t counts[] = {100, 0, 0, 0};
    int32_t total = 100;

    float bias_before = router->expert_bias[0];
    router_update_bias(router, counts, total, cfg.n_experts);
    float bias_after = router->expert_bias[0];

    /* Expert 0 is overloaded: its bias should decrease */
    ASSERT_TRUE(bias_after < bias_before);

    /* Verify EMA moved toward actual load for expert 0 */
    /* EMA = (1 - 0.01) * 0.25 + 0.01 * 1.0 = 0.2475 + 0.01 = 0.2575 */
    float expected_ema = (1.0f - 0.01f) * 0.25f + 0.01f * 1.0f;
    ASSERT_EQUAL_FLOAT(expected_ema, router->expert_load_ema[0], 1e-6f);

    router_destroy(router);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 3: Underloaded expert's bias increases
 * ======================================================================== */

static void test_bias_update_underloaded(void) {
    TEST_BEGIN(bias_update_underloaded);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    FEPRouter *router = router_create(pool, &cfg);
    ASSERT_NOT_NULL(router);

    /* Expert 0 gets NO tokens */
    int32_t counts[] = {0, 40, 30, 30};
    int32_t total = 100;

    float bias_before = router->expert_bias[0];
    router_update_bias(router, counts, total, cfg.n_experts);
    float bias_after = router->expert_bias[0];

    /* Expert 0 is underloaded: its bias should increase */
    ASSERT_TRUE(bias_after > bias_before);

    /* Verify EMA moved toward actual load for expert 0 */
    /* EMA = (1 - 0.01) * 0.25 + 0.01 * 0.0 = 0.2475 */
    float expected_ema = (1.0f - 0.01f) * 0.25f + 0.01f * 0.0f;
    ASSERT_EQUAL_FLOAT(expected_ema, router->expert_load_ema[0], 1e-6f);

    router_destroy(router);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 4: Bias actually affects routing decisions
 * ======================================================================== */

static void test_bias_affects_routing(void) {
    TEST_BEGIN(bias_affects_routing);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool1 = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *wpool2 = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool  = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool1);
    ASSERT_NOT_NULL(wpool2);
    ASSERT_NOT_NULL(apool);

    FEPRouter *router1 = router_create(wpool1, &cfg);
    FEPRouter *router2 = router_create(wpool2, &cfg);
    ASSERT_NOT_NULL(router1);
    ASSERT_NOT_NULL(router2);

    /* Set identical W_mu weights for both routers:
     * Make expert 3 the natural favorite by giving it high logit. */
    for (int32_t i = 0; i < cfg.d_model; i++) {
        for (int32_t j = 0; j < cfg.n_experts; j++) {
            int32_t idx[] = {i, j};
            float val = (j == 3) ? 1.0f : 0.0f;
            tensor_set(router1->W_mu, idx, val);
            tensor_set(router2->W_mu, idx, val);
        }
    }

    /* Give router2 a LARGE positive bias on expert 0 and large negative on expert 3.
     * This should override the natural preference for expert 3. */
    router2->expert_bias[0] =  100.0f;  /* strongly encourage expert 0 */
    router2->expert_bias[3] = -100.0f;  /* strongly discourage expert 3 */

    /* Create input */
    int32_t shape[] = {cfg.d_model};
    Tensor *x = tensor_create(apool, shape, 1, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    for (int32_t i = 0; i < cfg.d_model; i++) {
        int32_t idx[] = {i};
        tensor_set(x, idx, 1.0f);
    }

    /* Router 1 (no bias): should prefer expert 3 (highest logit) */
    RoutingDecision rd1 = router_forward(router1, x, &cfg, false);
    ASSERT_NOT_NULL(rd1.expert_ids);

    /* Router 2 (biased): should prefer expert 0 due to large positive bias */
    RoutingDecision rd2 = router_forward(router2, x, &cfg, false);
    ASSERT_NOT_NULL(rd2.expert_ids);

    /* Check that router2 selected expert 0 */
    bool router2_has_expert0 = false;
    for (int32_t i = 0; i < cfg.n_active; i++) {
        if (rd2.expert_ids[i] == 0) {
            router2_has_expert0 = true;
            break;
        }
    }
    ASSERT_TRUE(router2_has_expert0);

    /* Check that router2 did NOT select expert 3 (strongly discouraged) */
    bool router2_has_expert3 = false;
    for (int32_t i = 0; i < cfg.n_active; i++) {
        if (rd2.expert_ids[i] == 3) {
            router2_has_expert3 = true;
            break;
        }
    }
    ASSERT_FALSE(router2_has_expert3);

    free(rd1.expert_ids);
    free(rd1.expert_weights);
    free(rd2.expert_ids);
    free(rd2.expert_weights);
    tensor_destroy(x);
    router_destroy(router1);
    router_destroy(router2);
    pool_destroy(apool);
    pool_destroy(wpool1);
    pool_destroy(wpool2);

    TEST_END();
}

/* ========================================================================
 * TEST 5: Biases converge toward balancing under skewed load
 * ======================================================================== */

static void test_bias_convergence(void) {
    TEST_BEGIN(bias_convergence);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    FEPRouter *router = router_create(pool, &cfg);
    ASSERT_NOT_NULL(router);

    /* Use a larger step size to see convergence in 100 steps */
    router->bias_step_size = 0.01f;
    router->bias_update_rate = 0.1f;

    /* Simulate consistently skewed load: expert 0 gets 70%, expert 1 gets 30%,
     * experts 2 and 3 get nothing */
    int32_t counts[] = {70, 30, 0, 0};
    int32_t total = 100;

    for (int step = 0; step < 100; step++) {
        router_update_bias(router, counts, total, cfg.n_experts);
    }

    /* After 100 steps of consistent skew:
     * - Expert 0 (overloaded, 70% vs target 25%) should have NEGATIVE bias
     * - Expert 1 (slightly overloaded, 30% vs target 25%) should have NEGATIVE bias
     * - Experts 2 and 3 (underloaded, 0% vs target 25%) should have POSITIVE bias */
    ASSERT_TRUE(router->expert_bias[0] < 0.0f);
    ASSERT_TRUE(router->expert_bias[1] < 0.0f);
    ASSERT_TRUE(router->expert_bias[2] > 0.0f);
    ASSERT_TRUE(router->expert_bias[3] > 0.0f);

    /* Both overloaded experts have EMA > target for all steps, so with
     * a fixed step size their biases decrease equally each step. */
    ASSERT_EQUAL_FLOAT(router->expert_bias[0], router->expert_bias[1], 1e-6f);

    /* Underloaded experts should have equal positive bias (both got 0 load) */
    ASSERT_EQUAL_FLOAT(router->expert_bias[2], router->expert_bias[3], 1e-6f);

    /* Overloaded bias magnitude should equal underloaded bias magnitude
     * (same number of steps, same step size, just opposite direction) */
    ASSERT_EQUAL_FLOAT(-router->expert_bias[0], router->expert_bias[2], 1e-6f);

    router_destroy(router);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("=== Router Bias (Auxiliary-Loss-Free Balancing) Tests ===\n\n");

    test_bias_init_zero();
    test_bias_update_overloaded();
    test_bias_update_underloaded();
    test_bias_affects_routing();
    test_bias_convergence();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
