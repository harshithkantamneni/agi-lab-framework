/* test_eval_model_qat.c -- Acceptance tests for Program 3 P7 increment-5b T5.
 *
 * Tests the QAT eval wiring via the new hspa_model_forward_qat() API:
 *   1. Null-QAT regression: forward with qat=NULL is bit-identical to
 *      hspa_model_forward (existing callers unaffected).
 *   2. Honesty check: --qat-eval (enabled QATContext) produces DIFFERENT
 *      logits from no-QAT forward on at least one position. Proves the
 *      quantizer is active at eval time (§4.2 assertion-6 inflation guard).
 *   3. Coverage check: QATContext created for eval covers the same 224-weight
 *      census as training arms (Arm A/B/C gate clauses).
 *
 * These tests are RED (expected to FAIL) before the following changes:
 *   - hspa_model.h: add hspa_model_forward_qat() declaration
 *   - hspa_model.c: implement _qat variant; wrap existing forward as _qat(NULL)
 *
 * NOTE: eval_model.c has a main() and is not linked. We test the
 * hspa_model_forward_qat() API directly, which is what eval_model.c will call.
 *
 * Program 3, Phase 7, increment-5b (T5). D-613, 2026-05-28.
 *
 * Build (matches Makefile CFLAGS_DBG pattern):
 *   make build/tests/test_eval_model_qat
 */

#include "../src/tests/unity.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "qat_context.h"
#include "tensor.h"
#include "weight_init.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * Shared micro model dimensions.
 * Same as gate harness: D=64, L=8, K=8, k=2, V=256, DFF=128, SEQ=4.
 * Census: 8×4 attn=32, 8×8×3 FFN=192, total=224.
 * ==========================================================================*/

#define EVAL_D    64
#define EVAL_L    8
#define EVAL_K    8
#define EVAL_k    2
#define EVAL_V    256
#define EVAL_DFF  128
#define EVAL_SEQ  4

#define EVAL_EXPECTED_COVERED  224

static HSPAConfig eval_micro_config(void) {
    HSPAConfig cfg;
    cfg.d_model        = EVAL_D;
    cfg.n_layers       = EVAL_L;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = EVAL_D / 4;
    cfg.n_experts      = EVAL_K;
    cfg.n_active       = EVAL_k;
    cfg.d_ff           = EVAL_DFF;
    cfg.vocab_size     = EVAL_V;
    cfg.max_seq_len    = EVAL_SEQ;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

static void eval_fill_tokens(int32_t *tokens, int32_t seq_len,
                              int32_t vocab_size, uint32_t seed) {
    srand(seed);
    for (int32_t i = 0; i < seq_len; i++) {
        tokens[i] = (int32_t)(rand() % vocab_size);
    }
}

/* ==========================================================================
 * T5-TEST-1: hspa_model_forward_qat(qc=NULL) is bit-identical to
 *            hspa_model_forward (NULL-qat regression).
 *
 * The existing callers (eval_utils.h, test_eval.c, test_hspa_model.c) all
 * call hspa_model_forward, which after T5 becomes a thin wrapper calling
 * hspa_model_forward_qat(logits, model, tokens, seq_len, training, NULL).
 * This test proves the wrapper identity.
 *
 * RED: compile error "undefined reference to hspa_model_forward_qat" until
 *      the function is added to hspa_model.h / hspa_model.c.
 * ==========================================================================*/

static void test_t5_forward_qat_null_bit_identical_to_forward(void) {
    HSPAConfig cfg = eval_micro_config();

    srand(0xE001u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    int32_t tokens[EVAL_SEQ];
    eval_fill_tokens(tokens, EVAL_SEQ, EVAL_V, 0xE002u);

    int32_t logits_shape[] = {EVAL_SEQ, EVAL_V};

    /* Run 1: hspa_model_forward (current API — becomes thin wrapper) */
    Tensor *logits_a = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits_a);
    for (int32_t l = 0; l < cfg.n_layers; l++) model->kv_caches[l]->pos = 0;
    pool_reset(model->activation_pool);
    Tensor *logits_a2 = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits_a2);
    hspa_model_forward(logits_a2, model, tokens, EVAL_SEQ, false);

    /* Reset model state for second run */
    for (int32_t l = 0; l < cfg.n_layers; l++) model->kv_caches[l]->pos = 0;
    pool_reset(model->activation_pool);

    /* Run 2: hspa_model_forward_qat with qc=NULL */
    Tensor *logits_b = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits_b);
    /* This call is the new API — compile error until hspa_model.h is updated. */
    hspa_model_forward_qat(logits_b, model, tokens, EVAL_SEQ, false, NULL);

    /* Both must produce bit-identical results. */
    int32_t n_elems = EVAL_SEQ * EVAL_V;
    const float *da = (const float *)logits_a2->data;
    const float *db = (const float *)logits_b->data;
    for (int32_t i = 0; i < n_elems; i++) {
        if (da[i] != db[i]) {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "logit mismatch at index %d: forward=%.8f forward_qat(NULL)=%.8f",
                     (int)i, (double)da[i], (double)db[i]);
            TEST_FAIL(buf);
        }
    }

    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T5-TEST-2: QAT honesty check — enabled QATContext produces DIFFERENT logits.
 *
 * §4.2 assertion-6: a 4-bit-QAT cell's checkpoint stores FP32 master weights.
 * Scoring raw FP32 without quantization overstates quantized-arm accuracy.
 * eval_model.c with --qat-eval must apply the same fake_quantize_int4 to the
 * 224 in-scope weights. This test proves the quantizer fires at eval time.
 *
 * We run the same model+tokens with:
 *   A: qc=NULL (no quantization — raw FP32)
 *   B: qc=enabled QATContext (quantized w_hat)
 *
 * At least one logit must differ between A and B. If they are identical,
 * the quantizer is a silent no-op at eval time (inflation bug).
 *
 * RED: compile error until hspa_model_forward_qat is defined.
 * ==========================================================================*/

static void test_t5_qat_eval_logits_differ_from_fp32(void) {
    HSPAConfig cfg = eval_micro_config();

    srand(0xE003u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    int32_t tokens[EVAL_SEQ];
    eval_fill_tokens(tokens, EVAL_SEQ, EVAL_V, 0xE004u);

    int32_t logits_shape[] = {EVAL_SEQ, EVAL_V};

    /* Run A: no QAT (FP32 raw) */
    for (int32_t l = 0; l < cfg.n_layers; l++) model->kv_caches[l]->pos = 0;
    pool_reset(model->activation_pool);
    Tensor *logits_a = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits_a);
    hspa_model_forward_qat(logits_a, model, tokens, EVAL_SEQ, false, NULL);

    /* Save logits_a values before pool reset */
    float saved_a[EVAL_SEQ * EVAL_V];
    memcpy(saved_a, logits_a->data, (size_t)(EVAL_SEQ * EVAL_V) * sizeof(float));

    /* Run B: with QAT enabled */
    for (int32_t l = 0; l < cfg.n_layers; l++) model->kv_caches[l]->pos = 0;
    pool_reset(model->activation_pool);
    Tensor *logits_b = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits_b);

    /* Create enabled QATContext (group_size=128, capacity=256 per P3 binding spec) */
    QATContext *qc = qat_context_create(true, /*group_size=*/128, /*capacity=*/256);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    hspa_model_forward_qat(logits_b, model, tokens, EVAL_SEQ, false, qc);

    /* At least one logit must differ (proves quantizer is active) */
    int32_t n_elems = EVAL_SEQ * EVAL_V;
    const float *db = (const float *)logits_b->data;
    int found_diff = 0;
    for (int32_t i = 0; i < n_elems; i++) {
        if (saved_a[i] != db[i]) {
            found_diff = 1;
            break;
        }
    }
    ASSERT_TRUE(found_diff); /* If zero differences: QAT is a silent no-op */

    qat_context_destroy(qc);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T5-TEST-3: QAT eval covered_count == 224.
 *
 * A QATContext passed to hspa_model_forward_qat must cover the same 224-weight
 * census as the training arms. The pre-registration loop in hspa_block_forward
 * drives all K experts' W_gate/W_up/W_down into the cache. Attn weights
 * (W_q, W_k, W_v, W_o) fire during the forward pass itself.
 *
 * covered_count == 224 validates that eval-time QAT has complete coverage
 * (same as §3.3 Clause 1 for training arms).
 *
 * RED: compile error until hspa_model_forward_qat is defined.
 * ==========================================================================*/

static void test_t5_qat_eval_covered_224(void) {
    HSPAConfig cfg = eval_micro_config();

    srand(0xE005u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    int32_t tokens[EVAL_SEQ];
    eval_fill_tokens(tokens, EVAL_SEQ, EVAL_V, 0xE006u);

    int32_t logits_shape[] = {EVAL_SEQ, EVAL_V};

    /* Create enabled QATContext */
    QATContext *qc = qat_context_create(true, /*group_size=*/128, /*capacity=*/256);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    pool_reset(model->activation_pool);
    Tensor *logits = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    hspa_model_forward_qat(logits, model, tokens, EVAL_SEQ, false, qc);

    int covered = qat_context_covered_count(qc);
    ASSERT_EQUAL_INT(EVAL_EXPECTED_COVERED, covered);

    qat_context_destroy(qc);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T5-TEST-4: QAT eval logits are finite (no NaN/Inf inflation).
 *
 * Quantization should not produce NaN or Inf in the eval logits.
 * This is a sanity guard for the §3.3 Clause 3 analogue at eval time.
 * ==========================================================================*/

static void test_t5_qat_eval_logits_finite(void) {
    HSPAConfig cfg = eval_micro_config();

    srand(0xE007u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    int32_t tokens[EVAL_SEQ];
    eval_fill_tokens(tokens, EVAL_SEQ, EVAL_V, 0xE008u);

    int32_t logits_shape[] = {EVAL_SEQ, EVAL_V};

    QATContext *qc = qat_context_create(true, 128, 256);
    ASSERT_NOT_NULL(qc);
    qat_context_begin_step(qc);

    pool_reset(model->activation_pool);
    Tensor *logits = tensor_create(model->activation_pool, logits_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(logits);

    hspa_model_forward_qat(logits, model, tokens, EVAL_SEQ, false, qc);

    const float *data = (const float *)logits->data;
    int32_t n_elems = EVAL_SEQ * EVAL_V;
    for (int32_t i = 0; i < n_elems; i++) {
        ASSERT_TRUE(isfinite(data[i]));
    }

    qat_context_destroy(qc);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * main
 * ==========================================================================*/

int main(void) {
    printf("=== test_eval_model_qat (Program 3 P7 increment-5b T5) ===\n\n");
    printf("EVAL QAT WIRING: hspa_model_forward_qat API\n\n");

    RUN_TEST(test_t5_forward_qat_null_bit_identical_to_forward);
    RUN_TEST(test_t5_qat_eval_logits_differ_from_fp32);
    RUN_TEST(test_t5_qat_eval_covered_224);
    RUN_TEST(test_t5_qat_eval_logits_finite);

    TEST_REPORT();
}
