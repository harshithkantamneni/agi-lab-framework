/* test_ipc_qat.c -- QAT coverage + toggle integration tests for iPC.
 *
 * Program 3, Phase 7, increment-3c-ii.  D-613, 2026-05-28.
 *
 * Spec: programs/program_3_alt_grad_qat_100m/p7_qat_coverage_architecture.md
 *       §3 (224-coverage proof), §5 (iPC double-forward), §8 (acceptance gates).
 *
 * TDD: tests written BEFORE implementation in ipc_train.c (TDD Red phase).
 *
 * Tests:
 *
 *   TEST 1 — Clause 1 (§3.3 gate): Running one iPC step with
 *     tcfg.use_qat=true must result in exactly 224 distinct weight pointers
 *     registered in the QATContext.  Exposed via
 *     ipc_last_qat_covered_count() — a read-only seam added to
 *     ipc_train.c / ipc_train.h for test observability.
 *
 *     Decomposition: 8 layers × 4 attention weights (W_q, W_k, W_v, W_o)
 *     = 32 attention; 8 layers × 8 experts × 3 FFN weights (W_gate, W_up,
 *     W_down) = 192 FFN.  32 + 192 = 224.
 *
 *   TEST 2 — Clause 2 (toggle): Running a step with tcfg.use_qat=false
 *     produces deterministic, finite loss AND covered_count == 0
 *     (the disabled path must not register any weights).
 *     Two identical runs (same model state, same tokens) produce bit-exact
 *     equal loss, confirming the disabled path is a no-op pass-through.
 *
 *   TEST 3 — Structural decomposition: covered_count from an enabled step
 *     equals 32 (attn) + 192 (FFN) = 224 using the known formula.
 *
 *   TEST 4 — Multi-step stability: Two consecutive steps with use_qat=true
 *     each produce covered_count == 224 (begin_step correctly invalidates
 *     the cache between steps).
 *
 *   TEST 5 — Disabled step always gives covered_count == 0.
 *
 *   TEST 6 — iPC double-forward identity (§5): under enabled=true, the
 *     inline gate/up projections at L565/568 (Expert-Batched Forward Cache)
 *     consume the SAME w_hat as model-side gate/up projections in
 *     hspa_block_forward.  Verified indirectly: covered_count stays 224 (not
 *     448) even with Phase-1 double-forward.  The shared QATContext cache
 *     makes the second call a cache hit — no new distinct pointers registered.
 *
 * Build command (matches Makefile CFLAGS_DBG pattern):
 *   clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *         -DACCELERATE_NEW_LAPACK
 *         -Isrc/core -Isrc/model -Isrc/training -Isrc/tests
 *         tests/test_ipc_qat.c <C_SRCS>
 *         -o build/tests/test_ipc_qat
 *         -framework Accelerate -framework Metal -framework Foundation
 */

#include "../src/tests/unity.h"
#include "ipc_train.h"
#include "ipc_state.h"
#include "grad.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "router.h"
#include "train_config.h"
#include "weight_init.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * Micro model dimensions.
 * QAT scope: 8L × 4 attn + 8L × 8K × 3 FFN = 32 + 192 = 224.
 * Uses same dimensions as test_backprop_qat.c for cross-arm consistency.
 * ==========================================================================*/

#define QAT_IPC_D    64
#define QAT_IPC_L    8   /* 8 layers: matches spec 224-weight proof */
#define QAT_IPC_K    8   /* 8 experts: matches spec 192 FFN proof */
#define QAT_IPC_k    2
#define QAT_IPC_V    256
#define QAT_IPC_DFF  128
#define QAT_IPC_SEQ  4   /* Short sequence — we care about covered_count, not accuracy */

#define QAT_IPC_EXPECTED_ATTN  (QAT_IPC_L * 4)               /* 32 */
#define QAT_IPC_EXPECTED_FFN   (QAT_IPC_L * QAT_IPC_K * 3)   /* 192 */
#define QAT_IPC_EXPECTED_TOTAL (QAT_IPC_EXPECTED_ATTN + QAT_IPC_EXPECTED_FFN) /* 224 */

/* ==========================================================================
 * Helpers
 * ==========================================================================*/

static HSPAConfig ipc_qat_micro_config(void) {
    HSPAConfig cfg;
    cfg.d_model        = QAT_IPC_D;
    cfg.n_layers       = QAT_IPC_L;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = QAT_IPC_D / 4;   /* 16 */
    cfg.n_experts      = QAT_IPC_K;
    cfg.n_active       = QAT_IPC_k;
    cfg.d_ff           = QAT_IPC_DFF;
    cfg.vocab_size     = QAT_IPC_V;
    cfg.max_seq_len    = QAT_IPC_SEQ;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

static TrainConfig ipc_qat_tcfg_enabled(void) {
    TrainConfig tcfg = train_config_micro();
    tcfg.use_adam         = false;   /* SGD: simpler, deterministic */
    tcfg.base_lr          = 0.001f;
    tcfg.lr_warmup_steps  = 0;
    tcfg.grad_accum_steps = 1;
    tcfg.mup_base_width   = QAT_IPC_D;
    tcfg.use_qat          = true;
    return tcfg;
}

static TrainConfig ipc_qat_tcfg_disabled(void) {
    TrainConfig tcfg = ipc_qat_tcfg_enabled();
    tcfg.use_qat = false;
    return tcfg;
}

static void fill_ipc_tokens(int32_t *tokens, int32_t *targets,
                             int32_t seq_len, int32_t vocab_size, uint32_t seed) {
    srand(seed);
    for (int32_t i = 0; i < seq_len; i++) {
        tokens[i] = (int32_t)(rand() % vocab_size);
    }
    for (int32_t i = 0; i < seq_len - 1; i++) {
        targets[i] = tokens[i + 1];
    }
    targets[seq_len - 1] = tokens[0];
}

/* Deep-copy all in-scope weight tensors from src to dst.
 * Both models must have been created with identical configs.
 * Also copies router weights so routing decisions are identical across clones. */
static void copy_ipc_model_weights(HSPAModel *dst, const HSPAModel *src,
                                   const HSPAConfig *cfg) {
    int32_t D   = cfg->d_model;
    int32_t V   = cfg->vocab_size;
    int32_t dff = cfg->d_ff;
    int32_t q_dim  = cfg->n_heads * cfg->head_dim;
    int32_t kv_dim = cfg->n_kv_heads * cfg->head_dim;

    memcpy(dst->embed->weight->data, src->embed->weight->data,
           (size_t)(V * D) * sizeof(float));

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        const HSPABlock *sb = src->layers[l];
        HSPABlock *db = dst->layers[l];

        memcpy(db->attn->W_q->data, sb->attn->W_q->data,
               (size_t)(D * q_dim) * sizeof(float));
        memcpy(db->attn->W_k->data, sb->attn->W_k->data,
               (size_t)(D * kv_dim) * sizeof(float));
        memcpy(db->attn->W_v->data, sb->attn->W_v->data,
               (size_t)(D * kv_dim) * sizeof(float));
        memcpy(db->attn->W_o->data, sb->attn->W_o->data,
               (size_t)(q_dim * D) * sizeof(float));

        memcpy(db->attn_norm->weight->data, sb->attn_norm->weight->data,
               (size_t)D * sizeof(float));
        memcpy(db->ffn_norm->weight->data, sb->ffn_norm->weight->data,
               (size_t)D * sizeof(float));

        for (int32_t j = 0; j < cfg->n_experts; j++) {
            memcpy(db->experts[j]->W_gate->data, sb->experts[j]->W_gate->data,
                   (size_t)(D * dff) * sizeof(float));
            memcpy(db->experts[j]->W_up->data, sb->experts[j]->W_up->data,
                   (size_t)(D * dff) * sizeof(float));
            memcpy(db->experts[j]->W_down->data, sb->experts[j]->W_down->data,
                   (size_t)(dff * D) * sizeof(float));
        }
    }

    if (src->final_norm && dst->final_norm) {
        memcpy(dst->final_norm->weight->data, src->final_norm->weight->data,
               (size_t)D * sizeof(float));
    }

    /* Copy router weights so routing decisions are identical across clones. */
    int32_t K = cfg->n_experts;
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        const FEPRouter *sr = src->layers[l]->router;
        FEPRouter       *dr = dst->layers[l]->router;
        if (sr && dr) {
            memcpy(dr->W_mu->data,    sr->W_mu->data,
                   (size_t)(D * K) * sizeof(float));
            memcpy(dr->W_sigma->data, sr->W_sigma->data,
                   (size_t)(D * K) * sizeof(float));
            if (sr->expert_bias && dr->expert_bias) {
                memcpy(dr->expert_bias, sr->expert_bias,
                       (size_t)K * sizeof(float));
            }
        }
    }
}

/* ==========================================================================
 * TEST 1 — Clause 1: covered_count == 224 after one enabled iPC step.
 *
 * After ipc_train_step with use_qat=true, the QATContext must have seen
 * all 224 in-scope weight tensors: 8L×4 attn + 8L×8K×3 FFN.
 * ==========================================================================*/

static void test_ipc_qat_covered_count_224_one_step(void) {
    HSPAConfig cfg = ipc_qat_micro_config();
    TrainConfig tcfg = ipc_qat_tcfg_enabled();

    srand(0xD613u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    IPCTrainState *state = ipc_state_create(&cfg, QAT_IPC_SEQ,
                                            tcfg.sigma_bottom, tcfg.sigma_top);
    ASSERT_NOT_NULL(state);

    int32_t tokens[QAT_IPC_SEQ], targets[QAT_IPC_SEQ];
    fill_ipc_tokens(tokens, targets, QAT_IPC_SEQ, QAT_IPC_V, 0xABCDu);

    TrainStepResult r = ipc_train_step(model, grads, NULL, state,
                                       tokens, targets, QAT_IPC_SEQ,
                                       &cfg, &tcfg, 0, 0);

    ASSERT_TRUE(!isnan(r.loss.total));
    ASSERT_TRUE(!isinf(r.loss.total));

    int covered = ipc_last_qat_covered_count();
    ASSERT_EQUAL_INT(QAT_IPC_EXPECTED_TOTAL, covered);

    ipc_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ==========================================================================
 * TEST 2 — Clause 2: use_qat=false gives covered_count==0 and deterministic
 *   bit-identical loss across two runs from the same model state.
 * ==========================================================================*/

static void test_ipc_qat_disabled_deterministic_loss(void) {
    HSPAConfig cfg = ipc_qat_micro_config();
    TrainConfig tcfg = ipc_qat_tcfg_disabled();

    srand(0xD613u);
    HSPAModel *model_a = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model_a);
    weight_init_depth_mup(model_a, &cfg);

    HSPAModel *model_b = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model_b);
    copy_ipc_model_weights(model_b, model_a, &cfg);

    int32_t tokens[QAT_IPC_SEQ], targets[QAT_IPC_SEQ];
    fill_ipc_tokens(tokens, targets, QAT_IPC_SEQ, QAT_IPC_V, 0xABCDu);

    ModelGrad *grads_a = grad_create(&cfg);
    ModelGrad *grads_b = grad_create(&cfg);
    ASSERT_NOT_NULL(grads_a);
    ASSERT_NOT_NULL(grads_b);

    IPCTrainState *state_a = ipc_state_create(&cfg, QAT_IPC_SEQ,
                                              tcfg.sigma_bottom, tcfg.sigma_top);
    IPCTrainState *state_b = ipc_state_create(&cfg, QAT_IPC_SEQ,
                                              tcfg.sigma_bottom, tcfg.sigma_top);
    ASSERT_NOT_NULL(state_a);
    ASSERT_NOT_NULL(state_b);

    srand(1);
    TrainStepResult r_a = ipc_train_step(model_a, grads_a, NULL, state_a,
                                          tokens, targets, QAT_IPC_SEQ,
                                          &cfg, &tcfg, 0, 0);
    ASSERT_EQUAL_INT(0, ipc_last_qat_covered_count());

    srand(1);
    TrainStepResult r_b = ipc_train_step(model_b, grads_b, NULL, state_b,
                                          tokens, targets, QAT_IPC_SEQ,
                                          &cfg, &tcfg, 0, 0);
    ASSERT_EQUAL_INT(0, ipc_last_qat_covered_count());

    ASSERT_TRUE(!isnan(r_a.loss.total));
    ASSERT_TRUE(!isnan(r_b.loss.total));
    /* Clause 2: bit-identical (tolerance 0.0 = exact equality). */
    ASSERT_EQUAL_FLOAT(r_a.loss.total, r_b.loss.total, 0.0f);

    ipc_state_destroy(state_a);
    ipc_state_destroy(state_b);
    grad_destroy(grads_a);
    grad_destroy(grads_b);
    hspa_model_destroy(model_a);
    hspa_model_destroy(model_b);
}

/* ==========================================================================
 * TEST 3 — Structural decomposition: formula check + live coverage.
 * ==========================================================================*/

static void test_ipc_qat_coverage_decomposition(void) {
    ASSERT_EQUAL_INT(32,  QAT_IPC_EXPECTED_ATTN);
    ASSERT_EQUAL_INT(192, QAT_IPC_EXPECTED_FFN);
    ASSERT_EQUAL_INT(224, QAT_IPC_EXPECTED_TOTAL);

    HSPAConfig cfg = ipc_qat_micro_config();
    TrainConfig tcfg = ipc_qat_tcfg_enabled();

    srand(0xC0FEu);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    IPCTrainState *state = ipc_state_create(&cfg, QAT_IPC_SEQ,
                                            tcfg.sigma_bottom, tcfg.sigma_top);
    ASSERT_NOT_NULL(state);

    int32_t tokens[QAT_IPC_SEQ], targets[QAT_IPC_SEQ];
    fill_ipc_tokens(tokens, targets, QAT_IPC_SEQ, QAT_IPC_V, 0x1234u);

    ipc_train_step(model, grads, NULL, state,
                   tokens, targets, QAT_IPC_SEQ,
                   &cfg, &tcfg, 0, 0);

    ASSERT_EQUAL_INT(QAT_IPC_EXPECTED_TOTAL, ipc_last_qat_covered_count());

    ipc_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ==========================================================================
 * TEST 4 — Multi-step stability: covered_count == 224 on each of two
 *   consecutive steps (begin_step correctly invalidates cache per step).
 * ==========================================================================*/

static void test_ipc_qat_covered_count_stable_multi_step(void) {
    HSPAConfig cfg = ipc_qat_micro_config();
    TrainConfig tcfg = ipc_qat_tcfg_enabled();
    tcfg.base_lr = 0.0f;   /* Freeze weights so both steps see same model. */

    srand(0xFACEu);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    IPCTrainState *state = ipc_state_create(&cfg, QAT_IPC_SEQ,
                                            tcfg.sigma_bottom, tcfg.sigma_top);
    ASSERT_NOT_NULL(state);

    int32_t tokens[QAT_IPC_SEQ], targets[QAT_IPC_SEQ];
    fill_ipc_tokens(tokens, targets, QAT_IPC_SEQ, QAT_IPC_V, 0xBEEFu);

    ipc_train_step(model, grads, NULL, state,
                   tokens, targets, QAT_IPC_SEQ,
                   &cfg, &tcfg, 0, 0);
    ASSERT_EQUAL_INT(QAT_IPC_EXPECTED_TOTAL, ipc_last_qat_covered_count());

    ipc_train_step(model, grads, NULL, state,
                   tokens, targets, QAT_IPC_SEQ,
                   &cfg, &tcfg, 1, 0);
    ASSERT_EQUAL_INT(QAT_IPC_EXPECTED_TOTAL, ipc_last_qat_covered_count());

    ipc_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ==========================================================================
 * TEST 5 — Disabled always gives covered_count == 0.
 * ==========================================================================*/

static void test_ipc_qat_disabled_count_zero(void) {
    HSPAConfig cfg = ipc_qat_micro_config();
    TrainConfig tcfg = ipc_qat_tcfg_disabled();

    srand(0x9999u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    IPCTrainState *state = ipc_state_create(&cfg, QAT_IPC_SEQ,
                                            tcfg.sigma_bottom, tcfg.sigma_top);
    ASSERT_NOT_NULL(state);

    int32_t tokens[QAT_IPC_SEQ], targets[QAT_IPC_SEQ];
    fill_ipc_tokens(tokens, targets, QAT_IPC_SEQ, QAT_IPC_V, 0x5555u);

    ipc_train_step(model, grads, NULL, state,
                   tokens, targets, QAT_IPC_SEQ,
                   &cfg, &tcfg, 0, 0);
    ASSERT_EQUAL_INT(0, ipc_last_qat_covered_count());

    ipc_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ==========================================================================
 * TEST 6 — iPC double-forward identity (§5):
 *
 * Under enabled=true, covered_count must be exactly 224 — NOT 448.
 * This confirms the QATContext shared cache is functioning: the
 * Expert-Batched Forward Cache (inline gate/up at L565/568) calls
 * op_matmul_qat with the same qat context as hspa_block_forward.
 * The second encounter of each W_gate/W_up pointer is a cache hit
 * (returns same w_hat pointer, does NOT increment covered_count).
 *
 * If the double-forward were using a separate w_hat allocation or a
 * fresh context, covered_count would drift above 224.  Pinning to 224
 * is the observable test of §5 correctness.
 * ==========================================================================*/

static void test_ipc_qat_double_forward_identity(void) {
    HSPAConfig cfg = ipc_qat_micro_config();
    TrainConfig tcfg = ipc_qat_tcfg_enabled();

    srand(0xD0B1u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    IPCTrainState *state = ipc_state_create(&cfg, QAT_IPC_SEQ,
                                            tcfg.sigma_bottom, tcfg.sigma_top);
    ASSERT_NOT_NULL(state);

    int32_t tokens[QAT_IPC_SEQ], targets[QAT_IPC_SEQ];
    fill_ipc_tokens(tokens, targets, QAT_IPC_SEQ, QAT_IPC_V, 0xD0D0u);

    ipc_train_step(model, grads, NULL, state,
                   tokens, targets, QAT_IPC_SEQ,
                   &cfg, &tcfg, 0, 0);

    /* If double-forward double-counts: covered > 224.
     * If cache hits work correctly: covered == 224. */
    int covered = ipc_last_qat_covered_count();
    ASSERT_EQUAL_INT(QAT_IPC_EXPECTED_TOTAL, covered);

    ipc_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ==========================================================================
 * Main / test runner
 * ==========================================================================*/

int main(void) {
    printf("========================================\n");
    printf("  iPC QAT Coverage Integration Tests\n");
    printf("  (Program 3 P7 increment-3c-ii, D-613)\n");
    printf("========================================\n");

    printf("\n--- Clause 1: 224-coverage ---\n");
    RUN_TEST(test_ipc_qat_covered_count_224_one_step);
    RUN_TEST(test_ipc_qat_coverage_decomposition);

    printf("\n--- Clause 2: disabled toggle ---\n");
    RUN_TEST(test_ipc_qat_disabled_deterministic_loss);
    RUN_TEST(test_ipc_qat_disabled_count_zero);

    printf("\n--- Multi-step stability ---\n");
    RUN_TEST(test_ipc_qat_covered_count_stable_multi_step);

    printf("\n--- iPC double-forward identity (S5) ---\n");
    RUN_TEST(test_ipc_qat_double_forward_identity);

    TEST_REPORT();
}
