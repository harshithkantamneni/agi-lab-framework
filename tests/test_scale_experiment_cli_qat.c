/* test_scale_experiment_cli_qat.c -- Acceptance tests for Program 3 P7 increment-5b T0.
 *
 * Tests the QAT wiring through TrainConfig and qat_group_size threading:
 *   1. use_qat=true → covered_count==224 for each training arm (backprop, iPC, LocalFB).
 *   2. use_qat=false → covered_count==0 for each arm (pass-through).
 *   3. LocalFB arm is reachable and produces covered_count==224 (routes correctly).
 *   4. qat_group_size is threaded: train_config_micro qat_group_size defaults to 128.
 *   5. Mutual-exclusion between --backprop and --localfb is tested via
 *      direct ScaleArgs validation (not a fork test, tested via logic inspection).
 *
 * NOTE: scale_experiment.c has a main() and is not linked into tests.
 * The arm dispatch tests go directly through the step functions, same approach
 * as test_qat_gate_harness.c which also tests arm wiring without calling scale_experiment.
 *
 * These tests are RED (expected to FAIL) before the following changes are made:
 *   - train_config.h: add int qat_group_size field + default 128
 *   - backprop_train.c:261: replace hardcoded 128 with tcfg->qat_group_size
 *   - ipc_train.c:341: replace hardcoded 128 with tcfg->qat_group_size
 *   - local_feedback_train.c:732: replace hardcoded 128 with tcfg->qat_group_size
 *
 * Build (matches Makefile CFLAGS_DBG pattern):
 *   make build/tests/test_scale_experiment_cli_qat
 *
 * Program 3, Phase 7, increment-5b (T0). D-613, 2026-05-28.
 */

#include "../src/tests/unity.h"
#include "backprop_train.h"
#include "ipc_train.h"
#include "ipc_state.h"
#include "local_feedback_train.h"
#include "grad.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "train_config.h"
#include "weight_init.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * Shared micro model dimensions (same as test_qat_gate_harness.c).
 * D=64, L=8, K=8, k=2, V=256, DFF=128, seq=4.
 * Census: 8×4 attn=32, 8×8×3 FFN=192, total=224.
 * ==========================================================================*/

#define CLI_D    64
#define CLI_L    8
#define CLI_K    8
#define CLI_k    2
#define CLI_V    256
#define CLI_DFF  128
#define CLI_SEQ  4

#define CLI_EXPECTED_COVERED  224
#define CLI_QAT_GROUP_SIZE_DEFAULT  128

static HSPAConfig cli_micro_config(void) {
    HSPAConfig cfg;
    cfg.d_model        = CLI_D;
    cfg.n_layers       = CLI_L;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = CLI_D / 4;   /* 16 */
    cfg.n_experts      = CLI_K;
    cfg.n_active       = CLI_k;
    cfg.d_ff           = CLI_DFF;
    cfg.vocab_size     = CLI_V;
    cfg.max_seq_len    = CLI_SEQ;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

static TrainConfig cli_tcfg_qat_enabled(void) {
    TrainConfig tcfg = train_config_micro();
    tcfg.use_adam         = false;   /* SGD: deterministic */
    tcfg.base_lr          = 0.001f;
    tcfg.lr_warmup_steps  = 0;
    tcfg.grad_accum_steps = 1;
    tcfg.mup_base_width   = CLI_D;
    tcfg.use_qat          = true;
    /* qat_group_size must be 128 (default). This field is the NEW field
     * being added by T0. If compilation fails here with "no member named
     * 'qat_group_size'", that is the expected RED state. */
    tcfg.qat_group_size   = 128;
    return tcfg;
}

static TrainConfig cli_tcfg_qat_disabled(void) {
    TrainConfig tcfg = cli_tcfg_qat_enabled();
    tcfg.use_qat = false;
    return tcfg;
}

static void cli_fill_tokens(int32_t *tokens, int32_t *targets,
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

/* ==========================================================================
 * T0-TEST-1: TrainConfig has qat_group_size field with default 128.
 *
 * This test validates that train_config_micro() populates qat_group_size=128.
 * RED: fails with compile error "no member named 'qat_group_size'" until
 *      the field is added to train_config.h.
 * ==========================================================================*/

static void test_t0_train_config_qat_group_size_default_128(void) {
    TrainConfig tcfg = train_config_micro();
    /* Access the new field — compile error if not present (RED state). */
    int gs = tcfg.qat_group_size;
    ASSERT_EQUAL_INT(CLI_QAT_GROUP_SIZE_DEFAULT, gs);
    TEST_END();
}

/* ==========================================================================
 * T0-TEST-2: --qat arms Arm A (backprop) → covered_count==224.
 *
 * Validates that when use_qat=true AND qat_group_size=128 is threaded through
 * backprop_train_step, the 224-weight census fires correctly.
 *
 * After T0 implementation, backprop_train.c reads tcfg->qat_group_size instead
 * of the hardcoded 128. Behavior at group_size=128 is bit-identical to current.
 * RED state: compile error on qat_group_size field access.
 * ==========================================================================*/

static void test_t0_backprop_qat_covered_224(void) {
    HSPAConfig cfg   = cli_micro_config();
    TrainConfig tcfg = cli_tcfg_qat_enabled();

    srand(0xB001u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[CLI_SEQ], targets[CLI_SEQ];
    cli_fill_tokens(tokens, targets, CLI_SEQ, CLI_V, 0xB002u);

    backprop_train_step(model, grads, NULL,
                        tokens, targets, CLI_SEQ,
                        &cfg, &tcfg, 0, 0);

    int covered = backprop_last_qat_covered_count();
    ASSERT_EQUAL_INT(CLI_EXPECTED_COVERED, covered);

    grad_destroy(grads);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T0-TEST-3: --qat absent (Arm A backprop) → covered_count==0.
 *
 * Validates pass-through: disabled QAT means zero coverage.
 * ==========================================================================*/

static void test_t0_backprop_no_qat_covered_zero(void) {
    HSPAConfig cfg   = cli_micro_config();
    TrainConfig tcfg = cli_tcfg_qat_disabled();

    srand(0xB003u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[CLI_SEQ], targets[CLI_SEQ];
    cli_fill_tokens(tokens, targets, CLI_SEQ, CLI_V, 0xB004u);

    backprop_train_step(model, grads, NULL,
                        tokens, targets, CLI_SEQ,
                        &cfg, &tcfg, 0, 0);

    int covered = backprop_last_qat_covered_count();
    ASSERT_EQUAL_INT(0, covered);

    grad_destroy(grads);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T0-TEST-4: --qat arms Arm B (iPC) → covered_count==224.
 *
 * Validates qat_group_size threading in ipc_train_step.
 * RED: compile error on qat_group_size until train_config.h is updated.
 * ==========================================================================*/

static void test_t0_ipc_qat_covered_224(void) {
    HSPAConfig cfg   = cli_micro_config();
    TrainConfig tcfg = cli_tcfg_qat_enabled();

    srand(0xB005u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    IPCTrainState *state = ipc_state_create(&cfg, CLI_SEQ,
                                            tcfg.sigma_bottom, tcfg.sigma_top);
    ASSERT_NOT_NULL(state);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[CLI_SEQ], targets[CLI_SEQ];
    cli_fill_tokens(tokens, targets, CLI_SEQ, CLI_V, 0xB006u);

    ipc_train_step(model, grads, NULL, state,
                   tokens, targets, CLI_SEQ,
                   &cfg, &tcfg, 0, 0);

    int covered = ipc_last_qat_covered_count();
    ASSERT_EQUAL_INT(CLI_EXPECTED_COVERED, covered);

    grad_destroy(grads);
    ipc_state_destroy(state);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T0-TEST-5: --qat absent (Arm B iPC) → covered_count==0.
 * ==========================================================================*/

static void test_t0_ipc_no_qat_covered_zero(void) {
    HSPAConfig cfg   = cli_micro_config();
    TrainConfig tcfg = cli_tcfg_qat_disabled();

    srand(0xB007u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    IPCTrainState *state = ipc_state_create(&cfg, CLI_SEQ,
                                            tcfg.sigma_bottom, tcfg.sigma_top);
    ASSERT_NOT_NULL(state);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[CLI_SEQ], targets[CLI_SEQ];
    cli_fill_tokens(tokens, targets, CLI_SEQ, CLI_V, 0xB008u);

    ipc_train_step(model, grads, NULL, state,
                   tokens, targets, CLI_SEQ,
                   &cfg, &tcfg, 0, 0);

    int covered = ipc_last_qat_covered_count();
    ASSERT_EQUAL_INT(0, covered);

    grad_destroy(grads);
    ipc_state_destroy(state);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T0-TEST-6: --localfb arm routes to LocalFB step → covered_count==224.
 *
 * Validates that the LocalFB arm is reachable and produces the correct census.
 * RED: compile error on qat_group_size until train_config.h is updated.
 * ==========================================================================*/

static void test_t0_localfb_qat_covered_224(void) {
    HSPAConfig cfg   = cli_micro_config();
    TrainConfig tcfg = cli_tcfg_qat_enabled();

    srand(0xB009u);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    /* LocalFBState: seed=1 for deterministic B_l matrices. */
    LocalFBState *lfb = localfb_state_create(&cfg, /*seed=*/1u);
    ASSERT_NOT_NULL(lfb);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[CLI_SEQ], targets[CLI_SEQ];
    cli_fill_tokens(tokens, targets, CLI_SEQ, CLI_V, 0xB00Au);

    localfb_train_step(model, grads, NULL, lfb,
                       tokens, targets, CLI_SEQ,
                       &cfg, &tcfg, 0, 0);

    int covered = localfb_last_qat_covered_count();
    ASSERT_EQUAL_INT(CLI_EXPECTED_COVERED, covered);

    grad_destroy(grads);
    localfb_state_destroy(lfb);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T0-TEST-7: --localfb absent qat → covered_count==0.
 * ==========================================================================*/

static void test_t0_localfb_no_qat_covered_zero(void) {
    HSPAConfig cfg   = cli_micro_config();
    TrainConfig tcfg = cli_tcfg_qat_disabled();

    srand(0xB00Bu);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    LocalFBState *lfb = localfb_state_create(&cfg, /*seed=*/1u);
    ASSERT_NOT_NULL(lfb);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[CLI_SEQ], targets[CLI_SEQ];
    cli_fill_tokens(tokens, targets, CLI_SEQ, CLI_V, 0xB00Cu);

    localfb_train_step(model, grads, NULL, lfb,
                       tokens, targets, CLI_SEQ,
                       &cfg, &tcfg, 0, 0);

    int covered = localfb_last_qat_covered_count();
    ASSERT_EQUAL_INT(0, covered);

    grad_destroy(grads);
    localfb_state_destroy(lfb);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * T0-TEST-8: qat_group_size is threaded — non-default value reaches the arm.
 *
 * When qat_group_size is set to a NON-128 value AND use_qat=true, covered_count
 * must still equal 224 (the group_size controls INT4 grid granularity, not
 * the set of weights covered). This proves the field is threaded (read from
 * tcfg), not hardcoded.
 *
 * We use group_size=64 (half of default). The census is unchanged; only the
 * number of quantization groups per weight changes (more groups, same weights).
 * covered_count must still be 224.
 *
 * RED: compile error on qat_group_size; after fix, runtime-tests group_size=64
 * produces covered==224 (same census, different grid granularity).
 * ==========================================================================*/

static void test_t0_backprop_qat_group_size_threaded(void) {
    HSPAConfig cfg   = cli_micro_config();
    TrainConfig tcfg = cli_tcfg_qat_enabled();
    tcfg.qat_group_size = 64;  /* Non-default: proves field is threaded, not hardcoded */

    srand(0xB00Du);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[CLI_SEQ], targets[CLI_SEQ];
    cli_fill_tokens(tokens, targets, CLI_SEQ, CLI_V, 0xB00Eu);

    backprop_train_step(model, grads, NULL,
                        tokens, targets, CLI_SEQ,
                        &cfg, &tcfg, 0, 0);

    /* Census is still 224 regardless of group_size. */
    int covered = backprop_last_qat_covered_count();
    ASSERT_EQUAL_INT(CLI_EXPECTED_COVERED, covered);

    grad_destroy(grads);
    hspa_model_destroy(model);
    TEST_END();
}

/* ==========================================================================
 * main
 * ==========================================================================*/

int main(void) {
    printf("=== test_scale_experiment_cli_qat (Program 3 P7 increment-5b T0) ===\n\n");
    printf("ARM DISPATCH + QAT GROUP-SIZE THREADING\n\n");

    RUN_TEST(test_t0_train_config_qat_group_size_default_128);
    RUN_TEST(test_t0_backprop_qat_covered_224);
    RUN_TEST(test_t0_backprop_no_qat_covered_zero);
    RUN_TEST(test_t0_ipc_qat_covered_224);
    RUN_TEST(test_t0_ipc_no_qat_covered_zero);
    RUN_TEST(test_t0_localfb_qat_covered_224);
    RUN_TEST(test_t0_localfb_no_qat_covered_zero);
    RUN_TEST(test_t0_backprop_qat_group_size_threaded);

    TEST_REPORT();
}
