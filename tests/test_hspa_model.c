/* test_hspa_model.c -- Tests for the complete HSPA v1 model.
 *
 * Uses a tiny config for fast, low-memory testing:
 *   d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=4,
 *   n_experts=4, n_active=2, d_ff=32, vocab_size=100, max_seq_len=64
 *
 * Tests:
 *   1. model_create_destroy      -- lifecycle: create, verify non-NULL, destroy
 *   2. model_forward_single_token -- 1 token forward, logits shape + finite
 *   3. model_forward_multiple_tokens -- 3 token forward
 *   4. model_forward_training_mode -- forward with training=true
 *   5. model_weight_tying        -- lm_head aliases embed->weight
 *   6. model_kv_cache_update     -- KV cache positions advance after forward
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include "attention.h"
#include "embedding.h"
#include "hspa_block.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "rmsnorm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper: create a tiny test config suitable for model tests. */
static HSPAConfig test_config(void) {
    HSPAConfig cfg = hspa_config_default();
    cfg.d_model     = 16;
    cfg.n_layers    = 2;
    cfg.n_heads     = 4;
    cfg.n_kv_heads  = 2;
    cfg.head_dim    = 4;   /* d_model / n_heads = 16 / 4 */
    cfg.n_experts   = 4;
    cfg.n_active    = 2;
    cfg.d_ff        = 32;
    cfg.vocab_size  = 100;
    cfg.max_seq_len = 64;
    return cfg;
}

/* ========================================================================
 * TEST 1: model_create_destroy -- lifecycle test
 *
 * Create a model with the tiny config, verify all fields are non-NULL,
 * then destroy without leaking.
 * ======================================================================== */

static void test_model_create_destroy(void) {
    TEST_BEGIN(model_create_destroy);

    HSPAConfig cfg = test_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    /* Verify config was copied */
    ASSERT_EQUAL_INT(cfg.d_model, model->cfg.d_model);
    ASSERT_EQUAL_INT(cfg.n_layers, model->cfg.n_layers);
    ASSERT_EQUAL_INT(cfg.vocab_size, model->cfg.vocab_size);

    /* Verify all components are non-NULL */
    ASSERT_NOT_NULL(model->embed);
    ASSERT_NOT_NULL(model->layers);
    ASSERT_NOT_NULL(model->kv_caches);
    ASSERT_NOT_NULL(model->final_norm);
    ASSERT_NOT_NULL(model->lm_head);
    ASSERT_NOT_NULL(model->weight_pool);
    ASSERT_NOT_NULL(model->activation_pool);
    ASSERT_NOT_NULL(model->scratch_pool);

    /* Verify each layer and KV cache exists */
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        ASSERT_NOT_NULL(model->layers[l]);
        ASSERT_NOT_NULL(model->kv_caches[l]);
    }

    /* Verify embedding shape */
    ASSERT_NOT_NULL(model->embed->weight);
    ASSERT_EQUAL_INT(cfg.vocab_size, model->embed->weight->shape[0]);
    ASSERT_EQUAL_INT(cfg.d_model, model->embed->weight->shape[1]);

    /* Verify final norm dimension */
    ASSERT_NOT_NULL(model->final_norm->weight);
    ASSERT_EQUAL_INT(1, model->final_norm->weight->ndim);
    ASSERT_EQUAL_INT(cfg.d_model, model->final_norm->weight->shape[0]);

    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 2: model_forward_single_token -- 1 token forward pass
 *
 * Feed one token, verify logits are [1, vocab_size] and all finite.
 * With zero-initialized weights, the logits will be all zeros (since
 * embed output is zero, all projections produce zero, and the lm_head
 * matmul on zero h gives zero logits).
 * ======================================================================== */

static void test_model_forward_single_token(void) {
    TEST_BEGIN(model_forward_single_token);

    HSPAConfig cfg = test_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    /* Give embedding some non-zero values so the forward pass is non-trivial */
    tensor_fill(model->embed->weight, 0.01f);

    /* Allocate logits: [1, vocab_size] */
    int32_t logits_shape[] = {1, cfg.vocab_size};
    Tensor *logits = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);
    tensor_fill(logits, -999.0f); /* sentinel to verify it was written */

    int32_t tokens[] = {42};
    hspa_model_forward(logits, model, tokens, 1, false);

    /* Verify logits shape is correct */
    ASSERT_EQUAL_INT(2, logits->ndim);
    ASSERT_EQUAL_INT(1, logits->shape[0]);
    ASSERT_EQUAL_INT(cfg.vocab_size, logits->shape[1]);

    /* Verify all logits are finite (no NaN or Inf) */
    for (int32_t v = 0; v < cfg.vocab_size; v++) {
        int32_t idx[] = {0, v};
        float val = tensor_get(logits, idx);
        ASSERT_TRUE(!isnan(val));
        ASSERT_TRUE(!isinf(val));
    }

    tensor_destroy(logits);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 3: model_forward_multiple_tokens -- 3 token forward pass
 *
 * Feed 3 tokens, verify logits are [3, vocab_size] and all finite.
 * ======================================================================== */

static void test_model_forward_multiple_tokens(void) {
    TEST_BEGIN(model_forward_multiple_tokens);

    HSPAConfig cfg = test_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    /* Set small non-zero weights for non-trivial computation */
    tensor_fill(model->embed->weight, 0.01f);

    /* Allocate logits: [3, vocab_size] */
    int32_t seq_len = 3;
    int32_t logits_shape[] = {seq_len, cfg.vocab_size};
    Tensor *logits = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);
    tensor_fill(logits, -999.0f);

    int32_t tokens[] = {10, 50, 99};
    hspa_model_forward(logits, model, tokens, seq_len, false);

    /* Verify logits shape */
    ASSERT_EQUAL_INT(2, logits->ndim);
    ASSERT_EQUAL_INT(seq_len, logits->shape[0]);
    ASSERT_EQUAL_INT(cfg.vocab_size, logits->shape[1]);

    /* Verify all logits are finite */
    for (int32_t s = 0; s < seq_len; s++) {
        for (int32_t v = 0; v < cfg.vocab_size; v++) {
            int32_t idx[] = {s, v};
            float val = tensor_get(logits, idx);
            ASSERT_TRUE(!isnan(val));
            ASSERT_TRUE(!isinf(val));
        }
    }

    tensor_destroy(logits);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 4: model_forward_training_mode -- forward with training=true
 *
 * When training=true, the router samples from N(mu, sigma) instead
 * of using mu directly. This test verifies the forward pass completes
 * without errors in training mode.
 * ======================================================================== */

static void test_model_forward_training_mode(void) {
    TEST_BEGIN(model_forward_training_mode);

    HSPAConfig cfg = test_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    /* Set small weights for non-trivial behavior */
    tensor_fill(model->embed->weight, 0.01f);
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        tensor_fill(model->layers[l]->attn->W_q, 0.01f);
        tensor_fill(model->layers[l]->attn->W_k, 0.01f);
        tensor_fill(model->layers[l]->attn->W_v, 0.01f);
        tensor_fill(model->layers[l]->attn->W_o, 0.01f);
        tensor_fill(model->layers[l]->router->W_mu, 0.01f);
        for (int32_t e = 0; e < model->layers[l]->n_experts; e++) {
            tensor_fill(model->layers[l]->experts[e]->W_gate, 0.1f);
            tensor_fill(model->layers[l]->experts[e]->W_up, 0.1f);
            tensor_fill(model->layers[l]->experts[e]->W_down, 0.1f);
        }
    }

    /* Allocate logits: [2, vocab_size] */
    int32_t seq_len = 2;
    int32_t logits_shape[] = {seq_len, cfg.vocab_size};
    Tensor *logits = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    int32_t tokens[] = {5, 15};
    hspa_model_forward(logits, model, tokens, seq_len, true);

    /* Verify all logits are finite in training mode */
    for (int32_t s = 0; s < seq_len; s++) {
        for (int32_t v = 0; v < cfg.vocab_size; v++) {
            int32_t idx[] = {s, v};
            float val = tensor_get(logits, idx);
            ASSERT_TRUE(!isnan(val));
            ASSERT_TRUE(!isinf(val));
        }
    }

    tensor_destroy(logits);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 5: model_weight_tying -- lm_head aliases embed->weight
 *
 * Verify that model->lm_head points to the same data as
 * model->embed->weight, ensuring weight tying is in effect.
 * ======================================================================== */

static void test_model_weight_tying(void) {
    TEST_BEGIN(model_weight_tying);

    HSPAConfig cfg = test_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    /* lm_head should be the exact same pointer as embed->weight */
    ASSERT_TRUE(model->lm_head == model->embed->weight);

    /* Verify they share the same data pointer */
    ASSERT_TRUE(model->lm_head->data == model->embed->weight->data);

    /* Verify shapes match: both should be [vocab_size, d_model] */
    ASSERT_EQUAL_INT(model->embed->weight->shape[0], model->lm_head->shape[0]);
    ASSERT_EQUAL_INT(model->embed->weight->shape[1], model->lm_head->shape[1]);
    ASSERT_EQUAL_INT(cfg.vocab_size, model->lm_head->shape[0]);
    ASSERT_EQUAL_INT(cfg.d_model, model->lm_head->shape[1]);

    /* Writing to embed->weight should be visible through lm_head */
    int32_t idx[] = {7, 3};
    tensor_set(model->embed->weight, idx, 42.0f);
    float val = tensor_get(model->lm_head, idx);
    ASSERT_EQUAL_FLOAT(42.0f, val, 1e-6f);

    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 6: model_kv_cache_update -- KV cache positions advance
 *
 * After a forward pass with seq_len tokens, each layer's KV cache
 * position should advance by seq_len.
 * ======================================================================== */

static void test_model_kv_cache_update(void) {
    TEST_BEGIN(model_kv_cache_update);

    HSPAConfig cfg = test_config();
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    /* Verify initial cache positions are 0 */
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        ASSERT_EQUAL_INT(0, model->kv_caches[l]->pos);
    }

    /* Give embedding non-zero values */
    tensor_fill(model->embed->weight, 0.01f);

    /* Run forward pass with 3 tokens */
    int32_t seq_len = 3;
    int32_t logits_shape[] = {seq_len, cfg.vocab_size};
    Tensor *logits = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    int32_t tokens[] = {1, 2, 3};
    hspa_model_forward(logits, model, tokens, seq_len, false);

    /* After processing 3 tokens, each KV cache position should be 3 */
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        ASSERT_EQUAL_INT(seq_len, model->kv_caches[l]->pos);
    }

    tensor_destroy(logits);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("=== HSPA v1 Model Tests ===\n\n");

    printf("--- Lifecycle ---\n");
    test_model_create_destroy();

    printf("\n--- Forward Pass ---\n");
    test_model_forward_single_token();
    test_model_forward_multiple_tokens();
    test_model_forward_training_mode();

    printf("\n--- Weight Tying ---\n");
    test_model_weight_tying();

    printf("\n--- KV Cache ---\n");
    test_model_kv_cache_update();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
