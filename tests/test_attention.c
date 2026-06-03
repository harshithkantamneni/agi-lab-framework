/* test_attention.c -- Tests for GQA Attention + KV Cache.
 *
 * Small dimensions for testing:
 *   d_model=8, n_heads=4, n_kv_heads=2, head_dim=2,
 *   max_seq_len=16, vocab_size=32
 *
 * Tests:
 *   1. attention_create -- verify weight tensor shapes
 *   2. kv_cache_create -- verify cache shapes and pos=0
 *   3. attention_forward_single_token -- single token output shape
 *   4. attention_forward_kv_cache_update -- cache pos increments
 *   5. gqa_grouping -- 4 query heads share 2 KV heads correctly
 *   6. attention_forward_causal_mask -- future positions masked
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include "attention.h"
#include "hspa_config.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper: create a small test config. */
static HSPAConfig test_config(void) {
    HSPAConfig cfg = hspa_config_default();
    cfg.d_model     = 8;
    cfg.n_heads     = 4;
    cfg.n_kv_heads  = 2;
    cfg.head_dim    = 2;  /* d_model / n_heads = 8 / 4 */
    cfg.max_seq_len = 16;
    cfg.vocab_size  = 32;
    return cfg;
}

/* ========================================================================
 * TEST 1: attention_create -- verify weight shapes
 * ======================================================================== */

static void test_attention_create(void) {
    TEST_BEGIN(attention_create);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(pool);

    Attention *attn = attention_create(pool, &cfg);
    ASSERT_NOT_NULL(attn);

    /* W_q: [d_model, n_heads * head_dim] = [8, 8] */
    ASSERT_NOT_NULL(attn->W_q);
    ASSERT_EQUAL_INT(2, attn->W_q->ndim);
    ASSERT_EQUAL_INT(8, attn->W_q->shape[0]);
    ASSERT_EQUAL_INT(8, attn->W_q->shape[1]); /* 4 * 2 = 8 */

    /* W_k: [d_model, n_kv_heads * head_dim] = [8, 4] */
    ASSERT_NOT_NULL(attn->W_k);
    ASSERT_EQUAL_INT(2, attn->W_k->ndim);
    ASSERT_EQUAL_INT(8, attn->W_k->shape[0]);
    ASSERT_EQUAL_INT(4, attn->W_k->shape[1]); /* 2 * 2 = 4 */

    /* W_v: [d_model, n_kv_heads * head_dim] = [8, 4] */
    ASSERT_NOT_NULL(attn->W_v);
    ASSERT_EQUAL_INT(2, attn->W_v->ndim);
    ASSERT_EQUAL_INT(8, attn->W_v->shape[0]);
    ASSERT_EQUAL_INT(4, attn->W_v->shape[1]); /* 2 * 2 = 4 */

    /* W_o: [n_heads * head_dim, d_model] = [8, 8] */
    ASSERT_NOT_NULL(attn->W_o);
    ASSERT_EQUAL_INT(2, attn->W_o->ndim);
    ASSERT_EQUAL_INT(8, attn->W_o->shape[0]);
    ASSERT_EQUAL_INT(8, attn->W_o->shape[1]);

    attention_destroy(attn);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 2: kv_cache_create -- verify cache shapes and pos=0
 * ======================================================================== */

static void test_kv_cache_create(void) {
    TEST_BEGIN(kv_cache_create);

    HSPAConfig cfg = test_config();
    MemoryPool *pool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(pool);

    KVCache *cache = kv_cache_create(pool, &cfg);
    ASSERT_NOT_NULL(cache);

    /* k_cache: [max_seq_len, n_kv_heads, head_dim] = [16, 2, 2] */
    ASSERT_NOT_NULL(cache->k_cache);
    ASSERT_EQUAL_INT(3, cache->k_cache->ndim);
    ASSERT_EQUAL_INT(16, cache->k_cache->shape[0]);
    ASSERT_EQUAL_INT(2, cache->k_cache->shape[1]);
    ASSERT_EQUAL_INT(2, cache->k_cache->shape[2]);

    /* v_cache: [max_seq_len, n_kv_heads, head_dim] = [16, 2, 2] */
    ASSERT_NOT_NULL(cache->v_cache);
    ASSERT_EQUAL_INT(3, cache->v_cache->ndim);
    ASSERT_EQUAL_INT(16, cache->v_cache->shape[0]);
    ASSERT_EQUAL_INT(2, cache->v_cache->shape[1]);
    ASSERT_EQUAL_INT(2, cache->v_cache->shape[2]);

    /* Position starts at 0 */
    ASSERT_EQUAL_INT(0, cache->pos);

    kv_cache_destroy(cache);
    pool_destroy(pool);

    TEST_END();
}

/* ========================================================================
 * TEST 3: attention_forward_single_token -- output shape and non-NaN
 * ======================================================================== */

static void test_attention_forward_single_token(void) {
    TEST_BEGIN(attention_forward_single_token);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    Attention *attn = attention_create(wpool, &cfg);
    ASSERT_NOT_NULL(attn);

    KVCache *cache = kv_cache_create(apool, &cfg);
    ASSERT_NOT_NULL(cache);

    /* Set W_q to identity-like values so output is deterministic.
     * Use small non-zero values to avoid degenerate softmax. */
    tensor_fill(attn->W_q, 0.1f);
    tensor_fill(attn->W_k, 0.1f);
    tensor_fill(attn->W_v, 0.1f);
    tensor_fill(attn->W_o, 0.1f);

    /* Input: single token, [1, 8] */
    int32_t x_shape[] = {1, cfg.d_model};
    Tensor *x = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    tensor_fill(x, 1.0f);

    /* Output: [1, 8] */
    int32_t out_shape[] = {1, cfg.d_model};
    Tensor *out = tensor_create(apool, out_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out);
    tensor_fill(out, 0.0f);

    attention_forward(out, attn, x, cache, 0, NULL);

    /* Verify output is [1, 8] and values are finite (not NaN or Inf) */
    ASSERT_EQUAL_INT(2, out->ndim);
    ASSERT_EQUAL_INT(1, out->shape[0]);
    ASSERT_EQUAL_INT(8, out->shape[1]);

    for (int32_t j = 0; j < cfg.d_model; j++) {
        int32_t idx[] = {0, j};
        float val = tensor_get(out, idx);
        ASSERT_TRUE(!isnan(val));
        ASSERT_TRUE(!isinf(val));
    }

    tensor_destroy(out);
    tensor_destroy(x);
    kv_cache_destroy(cache);
    attention_destroy(attn);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 4: attention_forward_kv_cache_update -- cache pos increments
 * ======================================================================== */

static void test_attention_forward_kv_cache_update(void) {
    TEST_BEGIN(attention_forward_kv_cache_update);

    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    Attention *attn = attention_create(wpool, &cfg);
    ASSERT_NOT_NULL(attn);

    KVCache *cache = kv_cache_create(apool, &cfg);
    ASSERT_NOT_NULL(cache);

    tensor_fill(attn->W_q, 0.1f);
    tensor_fill(attn->W_k, 0.1f);
    tensor_fill(attn->W_v, 0.1f);
    tensor_fill(attn->W_o, 0.1f);

    /* First forward: 1 token at pos 0 */
    int32_t x_shape1[] = {1, cfg.d_model};
    Tensor *x1 = tensor_create(apool, x_shape1, 2, DTYPE_FP32);
    Tensor *out1 = tensor_create(apool, x_shape1, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x1);
    ASSERT_NOT_NULL(out1);
    tensor_fill(x1, 1.0f);

    ASSERT_EQUAL_INT(0, cache->pos);
    attention_forward(out1, attn, x1, cache, 0, NULL);
    ASSERT_EQUAL_INT(1, cache->pos);

    /* Second forward: 1 token at pos 1 */
    Tensor *x2 = tensor_create(apool, x_shape1, 2, DTYPE_FP32);
    Tensor *out2 = tensor_create(apool, x_shape1, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x2);
    ASSERT_NOT_NULL(out2);
    tensor_fill(x2, 0.5f);

    attention_forward(out2, attn, x2, cache, 1, NULL);
    ASSERT_EQUAL_INT(2, cache->pos);

    /* Third forward: 2 tokens at pos 2 (prefill-like) */
    int32_t x_shape3[] = {2, cfg.d_model};
    Tensor *x3 = tensor_create(apool, x_shape3, 2, DTYPE_FP32);
    int32_t out_shape3[] = {2, cfg.d_model};
    Tensor *out3 = tensor_create(apool, out_shape3, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x3);
    ASSERT_NOT_NULL(out3);
    tensor_fill(x3, 0.3f);

    attention_forward(out3, attn, x3, cache, 2, NULL);
    ASSERT_EQUAL_INT(4, cache->pos);

    tensor_destroy(out3);
    tensor_destroy(x3);
    tensor_destroy(out2);
    tensor_destroy(x2);
    tensor_destroy(out1);
    tensor_destroy(x1);
    kv_cache_destroy(cache);
    attention_destroy(attn);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 5: gqa_grouping -- 4 query heads map to 2 KV heads
 * ======================================================================== */

static void test_gqa_grouping(void) {
    TEST_BEGIN(gqa_grouping);

    /* With n_heads=4, n_kv_heads=2:
     * query heads 0,1 share KV head 0
     * query heads 2,3 share KV head 1
     *
     * Strategy: Set W_k such that KV head 0 and KV head 1 produce
     * different key vectors. Then set W_q such that query heads in
     * the same group produce the same Q. Verify that query heads
     * sharing a KV head produce the same output, and different groups
     * produce different output. */
    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool);

    Attention *attn = attention_create(wpool, &cfg);
    ASSERT_NOT_NULL(attn);

    KVCache *cache = kv_cache_create(apool, &cfg);
    ASSERT_NOT_NULL(cache);

    /* Set all weights to zero first */
    tensor_fill(attn->W_q, 0.0f);
    tensor_fill(attn->W_k, 0.0f);
    tensor_fill(attn->W_v, 0.0f);
    tensor_fill(attn->W_o, 0.0f);

    /* W_k: [8, 4] -- KV head 0 gets columns 0-1, KV head 1 gets columns 2-3.
     * Make KV head 0 keys different from KV head 1 keys.
     * Set W_k so that KV head 0 produces [1,0] and KV head 1 produces [0,1]
     * for a uniform input of 1.0. */
    {
        int32_t idx[] = {0, 0};
        tensor_set(attn->W_k, idx, 0.125f); /* col 0 = KV head 0, dim 0 */
    }
    {
        int32_t idx[] = {0, 3};
        tensor_set(attn->W_k, idx, 0.125f); /* col 3 = KV head 1, dim 1 */
    }

    /* W_v: [8, 4] -- make V differ per KV head too */
    {
        int32_t idx[] = {0, 0};
        tensor_set(attn->W_v, idx, 1.0f);
    }
    {
        int32_t idx[] = {0, 3};
        tensor_set(attn->W_v, idx, 2.0f);
    }

    /* W_q: [8, 8] -- heads 0,1 (cols 0-3) same, heads 2,3 (cols 4-7) same
     * but different from heads 0,1.
     * Set heads 0,1 to produce q=[1,0] and heads 2,3 to produce q=[0,1]. */
    {
        int32_t idx[] = {0, 0};
        tensor_set(attn->W_q, idx, 0.125f); /* head 0, dim 0 */
    }
    {
        int32_t idx[] = {0, 2};
        tensor_set(attn->W_q, idx, 0.125f); /* head 1, dim 0 */
    }
    {
        int32_t idx[] = {0, 5};
        tensor_set(attn->W_q, idx, 0.125f); /* head 2, dim 1 */
    }
    {
        int32_t idx[] = {0, 7};
        tensor_set(attn->W_q, idx, 0.125f); /* head 3, dim 1 */
    }

    /* W_o: identity-like [8, 8] */
    for (int32_t i = 0; i < 8; i++) {
        int32_t idx[] = {i, i};
        tensor_set(attn->W_o, idx, 1.0f);
    }

    /* Input: 1 token, [1, 8], all 1.0 */
    int32_t x_shape[] = {1, 8};
    Tensor *x = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    Tensor *out = tensor_create(apool, x_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    tensor_fill(x, 1.0f);
    tensor_fill(out, 0.0f);

    attention_forward(out, attn, x, cache, 0, NULL);

    /* After attention with W_o = identity, the output columns correspond to
     * head outputs. Heads 0 and 1 share KV head 0, so their outputs
     * (out[0,0:2] and out[0,2:4]) should be equal.
     * Heads 2 and 3 share KV head 1, so their outputs
     * (out[0,4:6] and out[0,6:8]) should be equal. */

    /* Head 0 output == Head 1 output (both use KV head 0) */
    for (int32_t d = 0; d < cfg.head_dim; d++) {
        int32_t idx0[] = {0, 0 * cfg.head_dim + d}; /* head 0 */
        int32_t idx1[] = {0, 1 * cfg.head_dim + d}; /* head 1 */
        float v0 = tensor_get(out, idx0);
        float v1 = tensor_get(out, idx1);
        ASSERT_EQUAL_FLOAT(v0, v1, 1e-5f);
    }

    /* Head 2 output == Head 3 output (both use KV head 1) */
    for (int32_t d = 0; d < cfg.head_dim; d++) {
        int32_t idx2[] = {0, 2 * cfg.head_dim + d}; /* head 2 */
        int32_t idx3[] = {0, 3 * cfg.head_dim + d}; /* head 3 */
        float v2 = tensor_get(out, idx2);
        float v3 = tensor_get(out, idx3);
        ASSERT_EQUAL_FLOAT(v2, v3, 1e-5f);
    }

    tensor_destroy(out);
    tensor_destroy(x);
    kv_cache_destroy(cache);
    attention_destroy(attn);
    pool_destroy(apool);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * TEST 6: attention_forward_causal_mask
 *   Token at pos 0 cannot attend to pos 1 (future).
 * ======================================================================== */

static void test_attention_forward_causal_mask(void) {
    TEST_BEGIN(attention_forward_causal_mask);

    /* Strategy: process 2 tokens at once. The output for token 0 should
     * only depend on token 0's KV (not token 1's). We verify by running:
     *   (A) 2 tokens together
     *   (B) 1 token alone (token 0 only)
     * The output for token 0 in both cases should be identical if the
     * causal mask is working. */
    HSPAConfig cfg = test_config();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *apool_a = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    MemoryPool *apool_b = pool_create(4 * 1024 * 1024, POOL_ACTIVATIONS);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(apool_a);
    ASSERT_NOT_NULL(apool_b);

    Attention *attn = attention_create(wpool, &cfg);
    ASSERT_NOT_NULL(attn);

    /* Use non-trivial weights */
    tensor_fill(attn->W_q, 0.1f);
    tensor_fill(attn->W_k, 0.1f);
    tensor_fill(attn->W_v, 0.1f);
    tensor_fill(attn->W_o, 0.1f);

    /* ---- Run A: 2 tokens at once ---- */
    KVCache *cache_a = kv_cache_create(apool_a, &cfg);
    ASSERT_NOT_NULL(cache_a);

    int32_t x_shape_a[] = {2, cfg.d_model};
    Tensor *x_a = tensor_create(apool_a, x_shape_a, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x_a);

    /* Token 0: all 1.0, Token 1: all 2.0 (different so mask matters) */
    for (int32_t j = 0; j < cfg.d_model; j++) {
        int32_t idx0[] = {0, j};
        tensor_set(x_a, idx0, 1.0f);
        int32_t idx1[] = {1, j};
        tensor_set(x_a, idx1, 2.0f);
    }

    int32_t out_shape_a[] = {2, cfg.d_model};
    Tensor *out_a = tensor_create(apool_a, out_shape_a, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out_a);
    tensor_fill(out_a, 0.0f);

    attention_forward(out_a, attn, x_a, cache_a, 0, NULL);

    /* ---- Run B: 1 token only (token 0) ---- */
    KVCache *cache_b = kv_cache_create(apool_b, &cfg);
    ASSERT_NOT_NULL(cache_b);

    int32_t x_shape_b[] = {1, cfg.d_model};
    Tensor *x_b = tensor_create(apool_b, x_shape_b, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x_b);
    tensor_fill(x_b, 1.0f); /* same as token 0 in run A */

    int32_t out_shape_b[] = {1, cfg.d_model};
    Tensor *out_b = tensor_create(apool_b, out_shape_b, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(out_b);
    tensor_fill(out_b, 0.0f);

    attention_forward(out_b, attn, x_b, cache_b, 0, NULL);

    /* ---- Compare: token 0's output should be identical in both runs ---- */
    for (int32_t j = 0; j < cfg.d_model; j++) {
        int32_t idx_a[] = {0, j};
        int32_t idx_b[] = {0, j};
        float val_a = tensor_get(out_a, idx_a);
        float val_b = tensor_get(out_b, idx_b);
        ASSERT_EQUAL_FLOAT(val_b, val_a, 1e-5f);
    }

    tensor_destroy(out_b);
    tensor_destroy(x_b);
    kv_cache_destroy(cache_b);

    tensor_destroy(out_a);
    tensor_destroy(x_a);
    kv_cache_destroy(cache_a);

    attention_destroy(attn);
    pool_destroy(apool_b);
    pool_destroy(apool_a);
    pool_destroy(wpool);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("=== GQA Attention + KV Cache Tests ===\n\n");

    printf("--- Attention Creation ---\n");
    test_attention_create();

    printf("\n--- KV Cache Creation ---\n");
    test_kv_cache_create();

    printf("\n--- Forward Pass ---\n");
    test_attention_forward_single_token();
    test_attention_forward_kv_cache_update();

    printf("\n--- GQA Grouping ---\n");
    test_gqa_grouping();

    printf("\n--- Causal Mask ---\n");
    test_attention_forward_causal_mask();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
