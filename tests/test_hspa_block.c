/* test_hspa_block.c -- Tests for HSPA transformer block.
 *
 * Small dimensions for testing:
 *   d_model=8, n_heads=4, n_kv_heads=2, head_dim=2,
 *   n_experts=4, n_active=2, d_ff=4,
 *   max_seq_len=16, vocab_size=32
 *
 * Tests:
 *   1. hspa_block_create -- all sub-components allocated
 *   2. hspa_block_create_expert_count -- n_experts experts created
 *   3. hspa_block_forward_shape -- output [seq_len, d_model] matches input
 *   4. hspa_block_forward_finite -- all outputs are finite (no NaN/Inf)
 *   5. hspa_block_forward_residual -- zero weights => output approx input
 *   6. hspa_block_destroy_null -- destroying NULL does not crash
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include "attention.h"
#include "ffn.h"
#include "hspa_block.h"
#include "hspa_config.h"
#include "rmsnorm.h"
#include "router.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper: create a small test config suitable for block tests. */
static HSPAConfig test_config(void) {
    HSPAConfig cfg = hspa_config_default();
    cfg.d_model     = 8;
    cfg.n_heads     = 4;
    cfg.n_kv_heads  = 2;
    cfg.head_dim    = 2;   /* d_model / n_heads = 8 / 4 */
    cfg.n_experts   = 4;
    cfg.n_active    = 2;
    cfg.d_ff        = 4;
    cfg.max_seq_len = 16;
    cfg.vocab_size  = 32;
    return cfg;
}

/* ========================================================================
 * TEST 1: hspa_block_create -- verify all sub-components allocated
 * ======================================================================== */

static void test_hspa_block_create(void) {
    TEST_BEGIN(hspa_block_create);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(8 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    HSPABlock *block = hspa_block_create(pool, &cfg);
    ASSERT_NOT_NULL(block);

    /* Verify all sub-components exist */
    ASSERT_NOT_NULL(block->attn_norm);
    ASSERT_NOT_NULL(block->attn);
    ASSERT_NOT_NULL(block->ffn_norm);
    ASSERT_NOT_NULL(block->router);
    ASSERT_NOT_NULL(block->experts);

    /* Verify attention weight shapes */
    ASSERT_NOT_NULL(block->attn->W_q);
    ASSERT_EQUAL_INT(8, block->attn->W_q->shape[0]);
    ASSERT_EQUAL_INT(8, block->attn->W_q->shape[1]); /* 4 heads * 2 head_dim */

    /* Verify norm weight dimension */
    ASSERT_NOT_NULL(block->attn_norm->weight);
    ASSERT_EQUAL_INT(1, block->attn_norm->weight->ndim);
    ASSERT_EQUAL_INT(8, block->attn_norm->weight->shape[0]);

    ASSERT_NOT_NULL(block->ffn_norm->weight);
    ASSERT_EQUAL_INT(1, block->ffn_norm->weight->ndim);
    ASSERT_EQUAL_INT(8, block->ffn_norm->weight->shape[0]);

    /* Verify router weight shapes */
    ASSERT_NOT_NULL(block->router->W_mu);
    ASSERT_EQUAL_INT(8, block->router->W_mu->shape[0]);
    ASSERT_EQUAL_INT(4, block->router->W_mu->shape[1]); /* n_experts */

    hspa_block_destroy(block);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 2: hspa_block_create_expert_count -- verify correct expert count
 * ======================================================================== */

static void test_hspa_block_create_expert_count(void) {
    TEST_BEGIN(hspa_block_create_expert_count);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(8 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    HSPABlock *block = hspa_block_create(pool, &cfg);
    ASSERT_NOT_NULL(block);

    /* n_experts should match config */
    ASSERT_EQUAL_INT(cfg.n_experts, block->n_experts);
    ASSERT_EQUAL_INT(4, block->n_experts);

    /* Each expert should be non-NULL and have correct weight shapes */
    for (int32_t i = 0; i < block->n_experts; i++) {
        ASSERT_NOT_NULL(block->experts[i]);
        ASSERT_NOT_NULL(block->experts[i]->W_gate);
        ASSERT_EQUAL_INT(8, block->experts[i]->W_gate->shape[0]); /* d_model */
        ASSERT_EQUAL_INT(4, block->experts[i]->W_gate->shape[1]); /* d_ff */
        ASSERT_NOT_NULL(block->experts[i]->W_down);
        ASSERT_EQUAL_INT(4, block->experts[i]->W_down->shape[0]); /* d_ff */
        ASSERT_EQUAL_INT(8, block->experts[i]->W_down->shape[1]); /* d_model */
    }

    hspa_block_destroy(block);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 3: hspa_block_forward_shape -- output shape matches input
 * ======================================================================== */

static void test_hspa_block_forward_shape(void) {
    TEST_BEGIN(hspa_block_forward_shape);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(8 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(8 * 1024 * 1024, POOL_ACTIVATIONS);
    MemoryPool *scratch = pool_create(8 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);
    ASSERT_NOT_NULL(scratch);

    HSPABlock *block = hspa_block_create(wpool, &cfg);
    ASSERT_NOT_NULL(block);

    KVCache *cache = kv_cache_create(apool, &cfg);
    ASSERT_NOT_NULL(cache);

    /* Set non-zero weights to get non-trivial behavior */
    tensor_fill(block->attn->W_q, 0.01f);
    tensor_fill(block->attn->W_k, 0.01f);
    tensor_fill(block->attn->W_v, 0.01f);
    tensor_fill(block->attn->W_o, 0.01f);

    /* Input: [4, 8] (4 tokens, d_model=8) */
    int32_t seq_len = 4;
    int32_t x_shape[] = {seq_len, cfg.d_model};
    Tensor *x = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    tensor_fill(x, 1.0f);

    /* Output: [4, 8] */
    int32_t out_shape[] = {seq_len, cfg.d_model};
    Tensor *out = tensor_create(apool, out_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);
    tensor_fill(out, 0.0f);

    hspa_block_forward(out, block, x, cache, 0, scratch, false, 0, NULL);

    /* Verify output shape matches */
    ASSERT_EQUAL_INT(2, out->ndim);
    ASSERT_EQUAL_INT(seq_len, out->shape[0]);
    ASSERT_EQUAL_INT(cfg.d_model, out->shape[1]);

    tensor_destroy(out);
    tensor_destroy(x);
    kv_cache_destroy(cache);
    hspa_block_destroy(block);
    pool_destroy(scratch);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 4: hspa_block_forward_finite -- all outputs finite (no NaN/Inf)
 * ======================================================================== */

static void test_hspa_block_forward_finite(void) {
    TEST_BEGIN(hspa_block_forward_finite);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(8 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(8 * 1024 * 1024, POOL_ACTIVATIONS);
    MemoryPool *scratch = pool_create(8 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);
    ASSERT_NOT_NULL(scratch);

    HSPABlock *block = hspa_block_create(wpool, &cfg);
    ASSERT_NOT_NULL(block);

    KVCache *cache = kv_cache_create(apool, &cfg);
    ASSERT_NOT_NULL(cache);

    /* Give weights small non-zero values for realistic behavior */
    tensor_fill(block->attn->W_q, 0.05f);
    tensor_fill(block->attn->W_k, 0.05f);
    tensor_fill(block->attn->W_v, 0.05f);
    tensor_fill(block->attn->W_o, 0.05f);
    tensor_fill(block->router->W_mu, 0.01f);

    /* Set expert weights to small values */
    for (int32_t e = 0; e < block->n_experts; e++) {
        tensor_fill(block->experts[e]->W_gate, 0.1f);
        tensor_fill(block->experts[e]->W_up, 0.1f);
        tensor_fill(block->experts[e]->W_down, 0.1f);
    }

    /* Input: [2, 8] */
    int32_t seq_len = 2;
    int32_t x_shape[] = {seq_len, cfg.d_model};
    Tensor *x = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    tensor_fill(x, 0.5f);

    /* Output: [2, 8] */
    int32_t out_shape[] = {seq_len, cfg.d_model};
    Tensor *out = tensor_create(apool, out_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);
    tensor_fill(out, 0.0f);

    hspa_block_forward(out, block, x, cache, 0, scratch, false, 0, NULL);

    /* Verify all outputs are finite */
    for (int32_t s = 0; s < seq_len; s++) {
        for (int32_t d = 0; d < cfg.d_model; d++) {
            int32_t idx[] = {s, d};
            float val = tensor_get(out, idx);
            ASSERT_TRUE(!isnan(val));
            ASSERT_TRUE(!isinf(val));
        }
    }

    tensor_destroy(out);
    tensor_destroy(x);
    kv_cache_destroy(cache);
    hspa_block_destroy(block);
    pool_destroy(scratch);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 5: hspa_block_forward_residual -- zero weights => output approx input
 *
 * When all projection weights (attention, FFN) are zero, the attention
 * and MoE branches contribute nothing. The residual connections pass
 * the input through unchanged: out = x + 0 + 0 = x.
 *
 * Note: RMSNorm weights default to 1.0, which normalizes the input,
 * but the linear projections are zero so the branch output is zero.
 * The residual adds zero to x, so output should equal input.
 * ======================================================================== */

static void test_hspa_block_forward_residual(void) {
    TEST_BEGIN(hspa_block_forward_residual);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(8 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(8 * 1024 * 1024, POOL_ACTIVATIONS);
    MemoryPool *scratch = pool_create(8 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);
    ASSERT_NOT_NULL(scratch);

    HSPABlock *block = hspa_block_create(wpool, &cfg);
    ASSERT_NOT_NULL(block);

    KVCache *cache = kv_cache_create(apool, &cfg);
    ASSERT_NOT_NULL(cache);

    /* All weights are zero-initialized by default (from create functions).
     * Attention W_q/W_k/W_v/W_o = 0 => attention output = 0.
     * Router W_mu/W_sigma = 0 => uniform routing, but expert FFN weights = 0
     * => expert output = 0. So both branches contribute zero.
     * Residual: out = x + 0 + 0 = x. */

    /* Input: [2, 8] with a distinctive pattern */
    int32_t seq_len = 2;
    int32_t x_shape[] = {seq_len, cfg.d_model};
    Tensor *x = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);

    /* Fill with recognizable non-uniform values */
    for (int32_t s = 0; s < seq_len; s++) {
        for (int32_t d = 0; d < cfg.d_model; d++) {
            int32_t idx[] = {s, d};
            tensor_set(x, idx, 0.1f * (float)(s * cfg.d_model + d + 1));
        }
    }

    /* Output: [2, 8] */
    int32_t out_shape[] = {seq_len, cfg.d_model};
    Tensor *out = tensor_create(apool, out_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);
    tensor_fill(out, 999.0f); /* sentinel */

    hspa_block_forward(out, block, x, cache, 0, scratch, false, 0, NULL);

    /* With zero weights, output should equal input (residual passthrough) */
    for (int32_t s = 0; s < seq_len; s++) {
        for (int32_t d = 0; d < cfg.d_model; d++) {
            int32_t idx[] = {s, d};
            float expected = tensor_get(x, idx);
            float actual = tensor_get(out, idx);
            ASSERT_EQUAL_FLOAT(expected, actual, 1e-4f);
        }
    }

    tensor_destroy(out);
    tensor_destroy(x);
    kv_cache_destroy(cache);
    hspa_block_destroy(block);
    pool_destroy(scratch);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 6: hspa_block_destroy_null -- destroying NULL does not crash
 * ======================================================================== */

static void test_hspa_block_destroy_null(void) {
    TEST_BEGIN(hspa_block_destroy_null);

    /* Should not crash or assert */
    hspa_block_destroy(NULL);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("=== HSPA Block Tests ===\n\n");

    printf("--- Block Creation ---\n");
    test_hspa_block_create();
    test_hspa_block_create_expert_count();

    printf("\n--- Forward Pass ---\n");
    test_hspa_block_forward_shape();
    test_hspa_block_forward_finite();

    printf("\n--- Residual Passthrough ---\n");
    test_hspa_block_forward_residual();

    printf("\n--- Null Safety ---\n");
    test_hspa_block_destroy_null();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
