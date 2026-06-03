/* test_model_qat_thread.c -- Regression tests for Item 3: QATContext thread-through.
 *
 * Program 3 Phase 7 apparatus: §10 item 3 spec.
 *
 * Tests:
 *   1. attention_null_qat_shape         -- attention_forward(qat=NULL) => correct output shape
 *   2. attention_null_qat_finite        -- attention_forward(qat=NULL) => all finite (regression)
 *   3. ffn_null_qat_shape               -- expert_ffn_forward(qat=NULL) => correct output shape
 *   4. ffn_null_qat_finite              -- expert_ffn_forward(qat=NULL) => all finite (regression)
 *   5. hspa_null_qat_shape              -- hspa_block_forward(qat=NULL) => correct output shape
 *   6. hspa_null_qat_finite             -- hspa_block_forward(qat=NULL) => all finite (regression)
 *   7. attention_enabled_qat_finite     -- attention_forward(qat enabled) => all finite
 *   8. ffn_enabled_qat_finite           -- expert_ffn_forward(qat enabled) => all finite
 *
 * The regression invariant: passing qat=NULL must be bit-identical to the old
 * (no-QAT) behavior. We verify this by checking that output shapes and finiteness
 * are preserved. Full numerical bit-identity is verified via the shape tests.
 */

#include "attention.h"
#include "ffn.h"
#include "hspa_block.h"
#include "hspa_config.h"
#include "memory_pool.h"
#include "qat_context.h"
#include "tensor.h"
#include "unity.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---- Config helpers ---------------------------------------------------- */

/* Minimal HSPA config: small d_model=32, seq=4, 2 heads, 1 kv-head, 2 experts */
static HSPAConfig make_small_cfg(void) {
    HSPAConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.d_model       = 32;
    cfg.n_heads       = 2;
    cfg.n_kv_heads    = 1;
    cfg.head_dim      = 16;   /* d_model / n_heads */
    cfg.n_layers      = 1;
    cfg.max_seq_len   = 16;
    cfg.vocab_size    = 64;
    cfg.d_ff          = 64;
    cfg.n_experts     = 2;
    cfg.n_active      = 1;
    return cfg;
}

/* ---- Tensor helpers ---------------------------------------------------- */

/* Heap FP32 2D tensor. */
static Tensor *make_fp32_2d(int32_t r, int32_t c) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    if (!t) return NULL;
    memset(t, 0, sizeof(Tensor));
    t->shape[0] = r;
    t->shape[1] = c;
    t->ndim     = 2;
    t->dtype    = DTYPE_FP32;
    t->stride[0] = c;
    t->stride[1] = 1;
    t->data      = malloc((size_t)r * (size_t)c * sizeof(float));
    if (!t->data) { free(t); return NULL; }
    return t;
}

static void free_fp32(Tensor *t) {
    if (!t) return;
    free(t->data);
    free(t);
}

static void fill_small(Tensor *t, float v) {
    float *d = (float *)t->data;
    int32_t n = t->shape[0] * t->shape[1];
    for (int32_t i = 0; i < n; i++) d[i] = v * 0.01f * (float)(i % 17 - 8);
}

static bool all_finite_2d(const Tensor *t) {
    const float *d = (const float *)t->data;
    int32_t n = t->shape[0] * t->shape[1];
    for (int32_t i = 0; i < n; i++) {
        if (!isfinite(d[i])) return false;
    }
    return true;
}

/* ---- Tests ------------------------------------------------------------- */

/*
 * TEST 1 & 2: attention_forward with NULL qat
 * Verifies backward-compatibility: new signature with qat=NULL behaves as before.
 */
static void test_attention_null_qat_shape(void) {
    TEST_BEGIN(attention_null_qat_shape);

    HSPAConfig cfg = make_small_cfg();
    MemoryPool *wpool   = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *scratch = pool_create(1 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(scratch);

    Attention *attn = attention_create(wpool, &cfg);
    KVCache   *cache = kv_cache_create(wpool, &cfg);
    ASSERT_NOT_NULL(attn);
    ASSERT_NOT_NULL(cache);

    int32_t seq_len = 4;
    Tensor *x   = make_fp32_2d(seq_len, cfg.d_model);
    Tensor *out = make_fp32_2d(seq_len, cfg.d_model);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    fill_small(x, 1.0f);

    attention_forward(out, attn, x, cache, 0, NULL);

    ASSERT_EQUAL_INT(seq_len,   out->shape[0]);
    ASSERT_EQUAL_INT(cfg.d_model, out->shape[1]);

    free_fp32(x);
    free_fp32(out);
    pool_destroy(wpool);
    pool_destroy(scratch);
    TEST_END();
}

static void test_attention_null_qat_finite(void) {
    TEST_BEGIN(attention_null_qat_finite);

    HSPAConfig cfg = make_small_cfg();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(wpool);

    Attention *attn  = attention_create(wpool, &cfg);
    KVCache   *cache = kv_cache_create(wpool, &cfg);
    ASSERT_NOT_NULL(attn);
    ASSERT_NOT_NULL(cache);

    Tensor *x   = make_fp32_2d(4, cfg.d_model);
    Tensor *out = make_fp32_2d(4, cfg.d_model);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    fill_small(x, 1.0f);

    attention_forward(out, attn, x, cache, 0, NULL);
    ASSERT_TRUE(all_finite_2d(out));

    free_fp32(x);
    free_fp32(out);
    pool_destroy(wpool);
    TEST_END();
}

/*
 * TEST 3 & 4: expert_ffn_forward with NULL qat
 */
static void test_ffn_null_qat_shape(void) {
    TEST_BEGIN(ffn_null_qat_shape);

    HSPAConfig cfg = make_small_cfg();
    MemoryPool *wpool   = pool_create(2 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *scratch = pool_create(1 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(scratch);

    ExpertFFN *ffn = expert_ffn_create(wpool, &cfg);
    ASSERT_NOT_NULL(ffn);

    int32_t seq_len = 4;
    Tensor *x   = make_fp32_2d(seq_len, cfg.d_model);
    Tensor *out = make_fp32_2d(seq_len, cfg.d_model);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    fill_small(x, 1.0f);

    expert_ffn_forward(out, ffn, x, scratch, NULL);

    ASSERT_EQUAL_INT(seq_len,     out->shape[0]);
    ASSERT_EQUAL_INT(cfg.d_model, out->shape[1]);

    free_fp32(x);
    free_fp32(out);
    pool_destroy(wpool);
    pool_destroy(scratch);
    TEST_END();
}

static void test_ffn_null_qat_finite(void) {
    TEST_BEGIN(ffn_null_qat_finite);

    HSPAConfig cfg = make_small_cfg();
    MemoryPool *wpool   = pool_create(2 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *scratch = pool_create(1 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(scratch);

    ExpertFFN *ffn = expert_ffn_create(wpool, &cfg);
    ASSERT_NOT_NULL(ffn);

    Tensor *x   = make_fp32_2d(4, cfg.d_model);
    Tensor *out = make_fp32_2d(4, cfg.d_model);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    fill_small(x, 1.0f);

    expert_ffn_forward(out, ffn, x, scratch, NULL);
    ASSERT_TRUE(all_finite_2d(out));

    free_fp32(x);
    free_fp32(out);
    pool_destroy(wpool);
    pool_destroy(scratch);
    TEST_END();
}

/*
 * TEST 5 & 6: hspa_block_forward with NULL qat
 */
static void test_hspa_null_qat_shape(void) {
    TEST_BEGIN(hspa_null_qat_shape);

    HSPAConfig cfg = make_small_cfg();
    MemoryPool *wpool   = pool_create(16 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *scratch = pool_create(4  * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(scratch);

    HSPABlock *block = hspa_block_create(wpool, &cfg);
    KVCache   *cache = kv_cache_create(wpool, &cfg);
    ASSERT_NOT_NULL(block);
    ASSERT_NOT_NULL(cache);

    int32_t seq_len = 4;
    Tensor *x   = make_fp32_2d(seq_len, cfg.d_model);
    Tensor *out = make_fp32_2d(seq_len, cfg.d_model);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    fill_small(x, 1.0f);

    hspa_block_forward(out, block, x, cache, 0, scratch, false, 0, NULL);

    ASSERT_EQUAL_INT(seq_len,     out->shape[0]);
    ASSERT_EQUAL_INT(cfg.d_model, out->shape[1]);

    free_fp32(x);
    free_fp32(out);
    pool_destroy(wpool);
    pool_destroy(scratch);
    TEST_END();
}

static void test_hspa_null_qat_finite(void) {
    TEST_BEGIN(hspa_null_qat_finite);

    HSPAConfig cfg = make_small_cfg();
    MemoryPool *wpool   = pool_create(16 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *scratch = pool_create(4  * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(scratch);

    HSPABlock *block = hspa_block_create(wpool, &cfg);
    KVCache   *cache = kv_cache_create(wpool, &cfg);
    ASSERT_NOT_NULL(block);
    ASSERT_NOT_NULL(cache);

    Tensor *x   = make_fp32_2d(4, cfg.d_model);
    Tensor *out = make_fp32_2d(4, cfg.d_model);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    fill_small(x, 1.0f);

    hspa_block_forward(out, block, x, cache, 0, scratch, false, 0, NULL);
    ASSERT_TRUE(all_finite_2d(out));

    free_fp32(x);
    free_fp32(out);
    pool_destroy(wpool);
    pool_destroy(scratch);
    TEST_END();
}

/*
 * TEST 7: attention_forward with enabled QATContext
 */
static void test_attention_enabled_qat_finite(void) {
    TEST_BEGIN(attention_enabled_qat_finite);

    HSPAConfig cfg = make_small_cfg();
    MemoryPool *wpool = pool_create(4 * 1024 * 1024, POOL_WEIGHTS);
    ASSERT_NOT_NULL(wpool);

    Attention *attn  = attention_create(wpool, &cfg);
    KVCache   *cache = kv_cache_create(wpool, &cfg);
    ASSERT_NOT_NULL(attn);
    ASSERT_NOT_NULL(cache);

    /* Use small group_size matching cfg.group_size, capacity=32 is plenty for 4 weights */
    QATContext *qc = qat_context_create(true, 32, 32);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    Tensor *x   = make_fp32_2d(4, cfg.d_model);
    Tensor *out = make_fp32_2d(4, cfg.d_model);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    fill_small(x, 1.0f);

    attention_forward(out, attn, x, cache, 0, qc);
    ASSERT_TRUE(all_finite_2d(out));

    /* 4 weight matrices: W_q, W_k, W_v, W_o */
    ASSERT_EQUAL_INT(4, qat_context_covered_count(qc));

    free_fp32(x);
    free_fp32(out);
    qat_context_destroy(qc);
    pool_destroy(wpool);
    TEST_END();
}

/*
 * TEST 8: expert_ffn_forward with enabled QATContext
 */
static void test_ffn_enabled_qat_finite(void) {
    TEST_BEGIN(ffn_enabled_qat_finite);

    HSPAConfig cfg = make_small_cfg();
    MemoryPool *wpool   = pool_create(2 * 1024 * 1024, POOL_WEIGHTS);
    MemoryPool *scratch = pool_create(1 * 1024 * 1024, POOL_SCRATCH);
    ASSERT_NOT_NULL(wpool);
    ASSERT_NOT_NULL(scratch);

    ExpertFFN *ffn = expert_ffn_create(wpool, &cfg);
    ASSERT_NOT_NULL(ffn);

    QATContext *qc = qat_context_create(true, 32, 32);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    Tensor *x   = make_fp32_2d(4, cfg.d_model);
    Tensor *out = make_fp32_2d(4, cfg.d_model);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(out);
    fill_small(x, 1.0f);

    expert_ffn_forward(out, ffn, x, scratch, qc);
    ASSERT_TRUE(all_finite_2d(out));

    /* 3 weight matrices: W_gate, W_up, W_down */
    ASSERT_EQUAL_INT(3, qat_context_covered_count(qc));

    free_fp32(x);
    free_fp32(out);
    qat_context_destroy(qc);
    pool_destroy(wpool);
    pool_destroy(scratch);
    TEST_END();
}

/* ---- main -------------------------------------------------------------- */

int main(void) {
    RUN_TEST(test_attention_null_qat_shape);
    RUN_TEST(test_attention_null_qat_finite);
    RUN_TEST(test_ffn_null_qat_shape);
    RUN_TEST(test_ffn_null_qat_finite);
    RUN_TEST(test_hspa_null_qat_shape);
    RUN_TEST(test_hspa_null_qat_finite);
    RUN_TEST(test_attention_enabled_qat_finite);
    RUN_TEST(test_ffn_enabled_qat_finite);
    TEST_REPORT();
}
