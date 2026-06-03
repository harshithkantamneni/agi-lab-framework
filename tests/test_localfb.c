/* test_localfb.c -- LocalFB (local feedback alignment) forward + coverage tests.
 *
 * Program 3, Phase 7, increment-4a.  2026-05-28.
 *
 * Citations (P-D432-LFB-DEPRECATION-INVENTORY: these symbols use localfb_ prefix
 * to avoid collision with Loss-Free-Balance "lfb_" router symbols):
 *   Nøkland 2016, arXiv 1609.01596 — Direct Feedback Alignment
 *   Lee et al. 2015, arXiv 1412.7525 — Difference Target Propagation
 *
 * Spec: programs/program_3_alt_grad_qat_100m/p7_qat_coverage_architecture.md
 *       §6 (LocalFB forward-compat), §3 (224-coverage proof), §8 (gate clauses).
 *       programs/program_3_alt_grad_qat_100m/program_open_memo.md §7.3.
 *
 * TDD: tests written FIRST (red), then local_feedback_train.c makes them green.
 *
 * Tests:
 *
 *   TEST 1 — Clause 1 acceptance gate (==224 covered, full 8L/8K config):
 *     Build a model with L=8, K=8 (matches hspa_config_100m shape except
 *     smaller D/V for speed). Run ONE localfb_train_step with use_qat=true.
 *     Assert localfb_last_qat_covered_count() == 224.
 *     32 attention (8L × {W_q,W_k,W_v,W_o}) + 192 FFN (8L × 8K × {W_gate,W_up,W_down}).
 *     B_l feedback matrices are NOT in QAT scope (§6, p7_coverage_architecture.md).
 *
 *   TEST 2 — Disabled pass-through (use_qat=false):
 *     covered_count == 0. Forward loss equals a non-QAT reference run bit-exactly.
 *
 *   TEST 3 — B_l init determinism:
 *     Two LocalFBState objects created with same seed produce bit-identical B_l.
 *     Two objects with different seeds produce different B_l[0].
 *
 *   TEST 4 — begin_step clears covered_count:
 *     After qat_context_begin_step, covered_count == 0 (freshness guard).
 *     Uses a micro-config to keep the test fast.
 *
 *   TEST 5 — LocalFBState create/destroy (small config, memory hygiene):
 *     Create and destroy a LocalFBState; no ASan/leaks issues.
 *
 * Build command (matches Makefile CFLAGS_DBG pattern):
 *   clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *         -DACCELERATE_NEW_LAPACK
 *         -Isrc/core -Isrc/model -Isrc/training -Isrc/tests
 *         tests/test_localfb.c <C_SRCS>
 *         -o build/tests/test_localfb
 *         -framework Accelerate -framework Metal -framework Foundation
 */

#include "../src/tests/unity.h"
#include "embedding.h"
#include "grad.h"
#include "hspa_block.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "ipc_train.h" /* TrainStepResult, AdamState */
#include "local_feedback_train.h"
#include "ops.h"
#include "rmsnorm.h"
#include "router.h"
#include "train_config.h"
#include "weight_init.h"
/* F7 (review qat_increment4b_fix §Flags): the direct `qat_context.h` include
 * was unused — its symbols are reached transitively via local_feedback_train.h
 * and hspa_model.h.  Removed to clear the clangd unused-includes lint. */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
 * Config for Tests 1 + 2 (8L / 8K: exactly the 224-weight shape).
 * Smaller D + V than 100M for speed; L and K must be 8 to get 224.
 * ==========================================================================*/

#define LFB_TEST_D 64
#define LFB_TEST_L 8 /* MUST be 8: 8 layers gives 32 attn + 192 FFN = 224 */
#define LFB_TEST_K 8 /* MUST be 8: 8 experts gives 192 FFN weight objects */
#define LFB_TEST_k 2
#define LFB_TEST_V 256
#define LFB_TEST_DFF 128
#define LFB_SEQ_LEN 4

#define LFB_EXPECTED_ATTN (LFB_TEST_L * 4)                        /* 32 */
#define LFB_EXPECTED_FFN (LFB_TEST_L * LFB_TEST_K * 3)            /* 192 */
#define LFB_EXPECTED_TOTAL (LFB_EXPECTED_ATTN + LFB_EXPECTED_FFN) /* 224 */

/* Smaller config for fast determinism / begin_step tests. */
#define LFB_MICRO_D 64
#define LFB_MICRO_L 4
#define LFB_MICRO_K 4
#define LFB_MICRO_k 1
#define LFB_MICRO_V 128
#define LFB_MICRO_DFF 64
#define LFB_MICRO_SEQ 4

/* ==========================================================================
 * Config helpers
 * ==========================================================================*/

static HSPAConfig lfb_test_8l8k_config(void) {
    HSPAConfig cfg;
    cfg.d_model = LFB_TEST_D;
    cfg.n_layers = LFB_TEST_L;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = LFB_TEST_D / 4; /* 16 */
    cfg.n_experts = LFB_TEST_K;
    cfg.n_active = LFB_TEST_k;
    cfg.d_ff = LFB_TEST_DFF;
    cfg.vocab_size = LFB_TEST_V;
    cfg.max_seq_len = LFB_SEQ_LEN;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps = 1e-5f;
    cfg.storage_dtype = DTYPE_FP32;
    cfg.compute_dtype = DTYPE_FP32;
    return cfg;
}

static HSPAConfig lfb_micro_config(void) {
    HSPAConfig cfg;
    cfg.d_model = LFB_MICRO_D;
    cfg.n_layers = LFB_MICRO_L;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = LFB_MICRO_D / 4; /* 16 */
    cfg.n_experts = LFB_MICRO_K;
    cfg.n_active = LFB_MICRO_k;
    cfg.d_ff = LFB_MICRO_DFF;
    cfg.vocab_size = LFB_MICRO_V;
    cfg.max_seq_len = LFB_MICRO_SEQ;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps = 1e-5f;
    cfg.storage_dtype = DTYPE_FP32;
    cfg.compute_dtype = DTYPE_FP32;
    return cfg;
}

static TrainConfig lfb_tcfg_enabled(int32_t d_model) {
    TrainConfig tcfg = train_config_micro();
    tcfg.use_adam = false;
    tcfg.base_lr = 0.001f;
    tcfg.lr_warmup_steps = 0;
    tcfg.grad_accum_steps = 1;
    tcfg.mup_base_width = d_model;
    tcfg.use_qat = true;
    return tcfg;
}

static TrainConfig lfb_tcfg_disabled(int32_t d_model) {
    TrainConfig tcfg = lfb_tcfg_enabled(d_model);
    tcfg.use_qat = false;
    return tcfg;
}

static void fill_tokens(int32_t *tokens, int32_t *targets, int32_t seq_len, int32_t vocab_size,
                        uint32_t seed) {
    srand(seed);
    for (int32_t i = 0; i < seq_len; i++) {
        tokens[i] = (int32_t) (rand() % vocab_size);
    }
    for (int32_t i = 0; i < seq_len - 1; i++) {
        targets[i] = tokens[i + 1];
    }
    targets[seq_len - 1] = tokens[0];
}

/* Deep-copy all in-scope weight tensors from src to dst (identical configs). */
static void copy_model_weights(HSPAModel *dst, const HSPAModel *src, const HSPAConfig *cfg) {
    int32_t D = cfg->d_model;
    int32_t V = cfg->vocab_size;
    int32_t dff = cfg->d_ff;
    int32_t q_dim = cfg->n_heads * cfg->head_dim;
    int32_t kv_dim = cfg->n_kv_heads * cfg->head_dim;

    memcpy(dst->embed->weight->data, src->embed->weight->data, (size_t) (V * D) * sizeof(float));

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        const HSPABlock *sb = src->layers[l];
        HSPABlock *db = dst->layers[l];

        memcpy(db->attn->W_q->data, sb->attn->W_q->data, (size_t) (D * q_dim) * sizeof(float));
        memcpy(db->attn->W_k->data, sb->attn->W_k->data, (size_t) (D * kv_dim) * sizeof(float));
        memcpy(db->attn->W_v->data, sb->attn->W_v->data, (size_t) (D * kv_dim) * sizeof(float));
        memcpy(db->attn->W_o->data, sb->attn->W_o->data, (size_t) (q_dim * D) * sizeof(float));

        memcpy(db->attn_norm->weight->data, sb->attn_norm->weight->data,
               (size_t) D * sizeof(float));
        memcpy(db->ffn_norm->weight->data, sb->ffn_norm->weight->data, (size_t) D * sizeof(float));

        for (int32_t j = 0; j < cfg->n_experts; j++) {
            memcpy(db->experts[j]->W_gate->data, sb->experts[j]->W_gate->data,
                   (size_t) (D * dff) * sizeof(float));
            memcpy(db->experts[j]->W_up->data, sb->experts[j]->W_up->data,
                   (size_t) (D * dff) * sizeof(float));
            memcpy(db->experts[j]->W_down->data, sb->experts[j]->W_down->data,
                   (size_t) (dff * D) * sizeof(float));
        }
    }

    if (src->final_norm && dst->final_norm) {
        memcpy(dst->final_norm->weight->data, src->final_norm->weight->data,
               (size_t) D * sizeof(float));
    }

    /* Copy router weights so routing decisions are identical across clones. */
    int32_t K = cfg->n_experts;
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        const FEPRouter *sr = src->layers[l]->router;
        FEPRouter *dr = dst->layers[l]->router;
        if (sr && dr) {
            memcpy(dr->W_mu->data, sr->W_mu->data, (size_t) (D * K) * sizeof(float));
            memcpy(dr->W_sigma->data, sr->W_sigma->data, (size_t) (D * K) * sizeof(float));
            if (sr->expert_bias && dr->expert_bias) {
                memcpy(dr->expert_bias, sr->expert_bias, (size_t) K * sizeof(float));
            }
        }
    }
}

/* ==========================================================================
 * TEST 1 — Clause 1: covered_count == 224 after one enabled step (8L/8K).
 * ==========================================================================*/
static void test_localfb_qat_covered_count_equals_224(void) {
    HSPAConfig cfg = lfb_test_8l8k_config();
    TrainConfig tcfg = lfb_tcfg_enabled(LFB_TEST_D);

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    LocalFBState *state = localfb_state_create(&cfg, /*seed=*/42);
    ASSERT_NOT_NULL(state);

    int32_t tokens[LFB_SEQ_LEN], targets[LFB_SEQ_LEN];
    fill_tokens(tokens, targets, LFB_SEQ_LEN, LFB_TEST_V, /*seed=*/1234);

    TrainStepResult r =
        localfb_train_step(model, grads, /*adam=*/NULL, state, tokens, targets, LFB_SEQ_LEN, &cfg,
                           &tcfg, /*step=*/0, /*micro_batch_idx=*/0);

    ASSERT_TRUE(isfinite(r.loss.lm));

    int covered = localfb_last_qat_covered_count();
    ASSERT_EQUAL_INT(LFB_EXPECTED_TOTAL, covered);

    localfb_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ==========================================================================
 * TEST 2 — Disabled pass-through: covered_count==0 and bit-identical loss.
 * ==========================================================================*/
static void test_localfb_disabled_passthrough(void) {
    HSPAConfig cfg = lfb_test_8l8k_config();
    TrainConfig tcfg_off = lfb_tcfg_disabled(LFB_TEST_D);

    /* Build a reference model and make a deep-copy clone. */
    HSPAModel *model_ref = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model_ref);
    weight_init_depth_mup(model_ref, &cfg);

    HSPAModel *model_b = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model_b);
    copy_model_weights(model_b, model_ref, &cfg);

    ModelGrad *grads_ref = grad_create(&cfg);
    ModelGrad *grads_b = grad_create(&cfg);
    ASSERT_NOT_NULL(grads_ref);
    ASSERT_NOT_NULL(grads_b);

    LocalFBState *state_ref = localfb_state_create(&cfg, /*seed=*/42);
    LocalFBState *state_b = localfb_state_create(&cfg, /*seed=*/42);
    ASSERT_NOT_NULL(state_ref);
    ASSERT_NOT_NULL(state_b);

    int32_t tokens[LFB_SEQ_LEN], targets[LFB_SEQ_LEN];
    fill_tokens(tokens, targets, LFB_SEQ_LEN, LFB_TEST_V, /*seed=*/5678);

    /* Run both with use_qat=false. */
    TrainStepResult r_ref = localfb_train_step(model_ref, grads_ref, NULL, state_ref, tokens,
                                               targets, LFB_SEQ_LEN, &cfg, &tcfg_off, 0, 0);

    TrainStepResult r_b = localfb_train_step(model_b, grads_b, NULL, state_b, tokens, targets,
                                             LFB_SEQ_LEN, &cfg, &tcfg_off, 0, 0);

    /* covered_count must be 0 when QAT is disabled. */
    ASSERT_EQUAL_INT(0, localfb_last_qat_covered_count());

    /* Both runs produce identical loss (disabled QAT is a pure pass-through). */
    ASSERT_EQUAL_FLOAT(r_ref.loss.lm, r_b.loss.lm, 0.0f);

    localfb_state_destroy(state_ref);
    localfb_state_destroy(state_b);
    grad_destroy(grads_ref);
    grad_destroy(grads_b);
    hspa_model_destroy(model_ref);
    hspa_model_destroy(model_b);
}

/* ==========================================================================
 * TEST 3 — B_l init determinism: same seed → bit-identical; different → differ.
 * Uses micro-config to keep the test fast (D is what matters for B_l shape).
 * ==========================================================================*/
static void test_localfb_bl_determinism(void) {
    HSPAConfig cfg = lfb_micro_config();

    /* Same seed: bit-identical B_l[0]. */
    LocalFBState *s1 = localfb_state_create(&cfg, /*seed=*/99);
    LocalFBState *s2 = localfb_state_create(&cfg, /*seed=*/99);
    ASSERT_NOT_NULL(s1);
    ASSERT_NOT_NULL(s2);

    /* Inspect B_l[0] via the accessor. */
    const Tensor *b0_s1 = localfb_state_get_B(s1, 0);
    const Tensor *b0_s2 = localfb_state_get_B(s2, 0);
    ASSERT_NOT_NULL(b0_s1);
    ASSERT_NOT_NULL(b0_s2);

    int32_t numel = b0_s1->shape[0] * b0_s1->shape[1];
    const float *d1 = (const float *) b0_s1->data;
    const float *d2 = (const float *) b0_s2->data;
    int match = 1;
    for (int32_t i = 0; i < numel; i++) {
        if (d1[i] != d2[i]) {
            match = 0;
            break;
        }
    }
    ASSERT_TRUE(match);

    /* Different seed: at least one element must differ. */
    LocalFBState *s3 = localfb_state_create(&cfg, /*seed=*/100);
    ASSERT_NOT_NULL(s3);
    const Tensor *b0_s3 = localfb_state_get_B(s3, 0);
    ASSERT_NOT_NULL(b0_s3);
    const float *d3 = (const float *) b0_s3->data;
    int differ = 0;
    for (int32_t i = 0; i < numel; i++) {
        if (d1[i] != d3[i]) {
            differ = 1;
            break;
        }
    }
    ASSERT_TRUE(differ);

    localfb_state_destroy(s1);
    localfb_state_destroy(s2);
    localfb_state_destroy(s3);
}

/* ==========================================================================
 * TEST 4 — begin_step clears covered_count.
 * Verifies indirectly: two consecutive steps each give micro_expected count
 * (not accumulated across steps), proving begin_step cleared previous count.
 * ==========================================================================*/
static void test_localfb_begin_step_clears(void) {
    HSPAConfig cfg = lfb_micro_config();
    TrainConfig tcfg = lfb_tcfg_enabled(LFB_MICRO_D);

    /* With micro config (L=4, K=4): expected covered = 4*4 + 4*4*3 = 16+48 = 64 */
    int32_t micro_expected = cfg.n_layers * 4                    /* attention: 4 weights/layer */
                             + cfg.n_layers * cfg.n_experts * 3; /* FFN: 3 weights/expert */

    HSPAModel *model = hspa_model_create(&cfg);
    ModelGrad *grads = grad_create(&cfg);
    LocalFBState *state = localfb_state_create(&cfg, /*seed=*/77);
    ASSERT_NOT_NULL(model);
    ASSERT_NOT_NULL(grads);
    ASSERT_NOT_NULL(state);
    weight_init_depth_mup(model, &cfg);

    int32_t tokens[LFB_MICRO_SEQ], targets[LFB_MICRO_SEQ];
    fill_tokens(tokens, targets, LFB_MICRO_SEQ, LFB_MICRO_V, /*seed=*/42);

    /* Step 0. */
    localfb_train_step(model, grads, NULL, state, tokens, targets, LFB_MICRO_SEQ, &cfg, &tcfg, 0,
                       0);
    int count0 = localfb_last_qat_covered_count();
    ASSERT_EQUAL_INT(micro_expected, count0);

    /* Step 1: begin_step must have cleared the cache; count must still be
     * micro_expected (not 2×micro_expected). */
    localfb_train_step(model, grads, NULL, state, tokens, targets, LFB_MICRO_SEQ, &cfg, &tcfg, 1,
                       0);
    int count1 = localfb_last_qat_covered_count();
    ASSERT_EQUAL_INT(micro_expected, count1);

    localfb_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ==========================================================================
 * TEST 5 — State create / destroy memory hygiene.
 * Just creates and destroys; ASan/leaks will catch any issues.
 * ==========================================================================*/
static void test_localfb_state_create_destroy(void) {
    HSPAConfig cfg = lfb_micro_config();
    LocalFBState *state = localfb_state_create(&cfg, /*seed=*/1);
    ASSERT_NOT_NULL(state);

    /* B_l accessor must return non-NULL for valid layer indices. */
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        const Tensor *Bl = localfb_state_get_B(state, l);
        ASSERT_NOT_NULL(Bl);
        /* B_l shape: [D, D] per Nokland 2016 §3.2 minimum-viable form. */
        ASSERT_EQUAL_INT(2, Bl->ndim);
        ASSERT_EQUAL_INT(cfg.d_model, Bl->shape[0]);
        ASSERT_EQUAL_INT(cfg.d_model, Bl->shape[1]);
    }

    localfb_state_destroy(state);
    /* If we reach here with ASan enabled, no use-after-free occurred. */
}

/* ==========================================================================
 * Increment-4b tests (TDD RED, written before 4b implementation)
 * ==========================================================================*/

/* --------------------------------------------------------------------------
 * TEST 6 — B_l orthogonal init: Q^T Q ≈ I within tolerance.
 *
 * After increment-4b replaces localfb_init_bl() with the MGS-QR Haar-uniform
 * initializer (Launay, Poli & Krzakala 2019 arXiv:1906.04554 §3.2), every
 * B_l must satisfy ||B_l[:,i] · B_l[:,j] - delta_ij|| < tol for all i,j.
 *
 * Tolerance: 1e-4f is appropriate for D=64 single-precision MGS.
 *
 * Uses micro config (D=64) so the [D,D] QTQ check is fast.
 * --------------------------------------------------------------------------*/
static void test_localfb_bl_orthogonal_init(void) {
    HSPAConfig cfg = lfb_micro_config();
    int32_t D = cfg.d_model;

    LocalFBState *state = localfb_state_create(&cfg, /*seed=*/42);
    ASSERT_NOT_NULL(state);

    /* Check all layers — pick layer 0 as representative fast check; verify
     * that the property holds for at least the first layer. */
    const Tensor *B0 = localfb_state_get_B(state, 0);
    ASSERT_NOT_NULL(B0);

    const float *data = (const float *) B0->data;
    float max_off_diag = 0.0f;
    float max_diag_err = 0.0f;

    /* Compute Q^T Q naively (D small, so O(D^3) is fine in test). */
    for (int32_t i = 0; i < D; i++) {
        for (int32_t j = 0; j < D; j++) {
            /* dot(column i, column j) */
            float dot = 0.0f;
            for (int32_t k = 0; k < D; k++) {
                /* B is row-major [D, D], column i is data[k*D + i]. */
                dot += data[(size_t) k * D + i] * data[(size_t) k * D + j];
            }
            float expected = (i == j) ? 1.0f : 0.0f;
            float err = dot - expected;
            if (err < 0.0f)
                err = -err;
            if (i == j) {
                if (err > max_diag_err)
                    max_diag_err = err;
            } else {
                if (err > max_off_diag)
                    max_off_diag = err;
            }
        }
    }

    /* Diagonal entries must be ~1 (column norms). */
    ASSERT_TRUE(max_diag_err < 1e-4f);

    /* Off-diagonal entries must be ~0 (column orthogonality). */
    ASSERT_TRUE(max_off_diag < 1e-4f);

    localfb_state_destroy(state);
}

/* --------------------------------------------------------------------------
 * TEST 7 — B_l is NOT corrupted by srand() calls in the training loop.
 *
 * The 4a localfb_init_bl() uses global srand/rand, so a srand() call
 * issued by the training step preamble (line 373: srand(step+1+...))
 * would corrupt the B_l state on re-initialization.  In increment-4b
 * this is migrated to a local xorshift32 PRNG; srand() must not affect
 * bit-identical B_l.
 *
 * Protocol:
 *   1. Create state with seed=55.
 *   2. Snapshot B_l[0] before any step.
 *   3. Run one training step (which calls srand internally).
 *   4. Create a SECOND state with the same seed=55.
 *   5. B_l[0] of the second state must be bit-identical to the snapshot.
 *
 * This proves B_l init uses local PRNG, not global rand state.
 * --------------------------------------------------------------------------*/
static void test_localfb_bl_srand_isolation(void) {
    HSPAConfig cfg = lfb_micro_config();
    TrainConfig tcfg = lfb_tcfg_enabled(LFB_MICRO_D);
    int32_t D = cfg.d_model;
    int32_t numel = D * D;

    /* Create first state and snapshot B_l[0]. */
    LocalFBState *state_a = localfb_state_create(&cfg, /*seed=*/55);
    ASSERT_NOT_NULL(state_a);

    const Tensor *b0_a = localfb_state_get_B(state_a, 0);
    ASSERT_NOT_NULL(b0_a);

    float *snapshot = (float *) malloc((size_t) numel * sizeof(float));
    ASSERT_NOT_NULL(snapshot);
    memcpy(snapshot, b0_a->data, (size_t) numel * sizeof(float));

    /* Run one training step (the preamble calls srand internally). */
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[LFB_MICRO_SEQ], targets[LFB_MICRO_SEQ];
    fill_tokens(tokens, targets, LFB_MICRO_SEQ, LFB_MICRO_V, /*seed=*/999);

    localfb_train_step(model, grads, NULL, state_a, tokens, targets, LFB_MICRO_SEQ, &cfg, &tcfg,
                       /*step=*/0, /*micro_batch_idx=*/0);

    /* Create a second state with the same seed — must reproduce the same B_l. */
    LocalFBState *state_b = localfb_state_create(&cfg, /*seed=*/55);
    ASSERT_NOT_NULL(state_b);

    const Tensor *b0_b = localfb_state_get_B(state_b, 0);
    ASSERT_NOT_NULL(b0_b);

    int match = 1;
    const float *d_b = (const float *) b0_b->data;
    for (int32_t i = 0; i < numel; i++) {
        if (snapshot[i] != d_b[i]) {
            match = 0;
            break;
        }
    }
    ASSERT_TRUE(match);

    free(snapshot);
    localfb_state_destroy(state_a);
    localfb_state_destroy(state_b);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* --------------------------------------------------------------------------
 * TEST 8 — DFA backward smoke: weights change after a training step.
 *
 * With the backward stub in place (4a), weights never change because no
 * gradient is accumulated or applied.  After increment-4b completes the
 * DFA backward + optimizer, at least one weight must differ.
 *
 * Protocol:
 *   1. Create model, snapshot W_q[0] (first attention weight, layer 0).
 *   2. Run one training step with use_qat=true, use_adam=false (SGD).
 *   3. Compare W_q[0] before and after — at least one element must differ.
 *
 * Also: covered_count must still be 224 after a full step with backward.
 * --------------------------------------------------------------------------*/
static void test_localfb_backward_weights_change(void) {
    HSPAConfig cfg = lfb_test_8l8k_config();
    TrainConfig tcfg = lfb_tcfg_enabled(LFB_TEST_D);
    tcfg.use_adam = false;
    tcfg.base_lr = 1e-2f; /* Large LR so weight change is detectable. */
    int32_t D = cfg.d_model;
    int32_t q_dim = cfg.n_heads * cfg.head_dim;

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    /* Snapshot W_q[0]. */
    int32_t wq_numel = D * q_dim;
    float *wq_before = (float *) malloc((size_t) wq_numel * sizeof(float));
    ASSERT_NOT_NULL(wq_before);
    memcpy(wq_before, model->layers[0]->attn->W_q->data, (size_t) wq_numel * sizeof(float));

    ModelGrad *grads = grad_create(&cfg);
    LocalFBState *state = localfb_state_create(&cfg, /*seed=*/42);
    ASSERT_NOT_NULL(grads);
    ASSERT_NOT_NULL(state);

    int32_t tokens[LFB_SEQ_LEN], targets[LFB_SEQ_LEN];
    fill_tokens(tokens, targets, LFB_SEQ_LEN, LFB_TEST_V, /*seed=*/2024);

    TrainStepResult r =
        localfb_train_step(model, grads, /*adam=*/NULL, state, tokens, targets, LFB_SEQ_LEN, &cfg,
                           &tcfg, /*step=*/1, /*micro_batch_idx=*/0);

    /* Loss must be finite. */
    ASSERT_TRUE(isfinite(r.loss.lm));

    /* grad_norm must be nonzero (backward produced gradients). */
    ASSERT_TRUE(r.grad_norm > 0.0f);

    /* At least one weight must have changed. */
    const float *wq_after = (const float *) model->layers[0]->attn->W_q->data;
    int changed = 0;
    for (int32_t i = 0; i < wq_numel; i++) {
        if (wq_before[i] != wq_after[i]) {
            changed = 1;
            break;
        }
    }
    ASSERT_TRUE(changed);

    /* covered_count must remain 224 (backward must NOT register B_l through QAT). */
    ASSERT_EQUAL_INT(LFB_EXPECTED_TOTAL, localfb_last_qat_covered_count());

    free(wq_before);
    localfb_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* --------------------------------------------------------------------------
 * TEST 9 — 10-step smoke run for all 5 seeds (§5.9 FF-BUILDABILITY-2).
 *
 * Runs 10 consecutive training steps for each seed in {42,43,44,45,46}.
 * Asserts per dispatch spec (STEP 3):
 *   (a) Every step's loss is finite (no NaN/Inf).
 *   (b) At least one weight changes per run (dW nonzero → optimizer fired).
 *   (c) covered_count == 224 on the final step of each seed run.
 *
 * Note: monotonic loss decrease is NOT asserted here.  DFA is an
 * alternative-gradient method whose feedback signal is deliberately
 * misaligned with the true gradient; 10 steps on 4 tokens is not
 * sufficient for reliable descent with fixed random B_l matrices.
 * The dispatch spec says "loss moves" (loss.lm is finite and grad_norm > 0),
 * not "loss decreases monotonically".  Test 8 (weights_change) already
 * covers the weight-update gate; here we exercise 5 seeds × 10 steps.
 *
 * §5.9 FF-BUILDABILITY-2: all 5 seeds must produce finite loss every step.
 * --------------------------------------------------------------------------*/
static void test_localfb_10step_smoke_all_seeds(void) {
    static const uint32_t seeds[5] = {42, 43, 44, 45, 46};

    for (int32_t s_idx = 0; s_idx < 5; s_idx++) {
        uint32_t seed = seeds[s_idx];

        HSPAConfig cfg = lfb_test_8l8k_config();
        TrainConfig tcfg = lfb_tcfg_enabled(LFB_TEST_D);
        tcfg.use_adam = false;
        tcfg.base_lr = 1e-3f; /* Moderate LR; descent not required, only finiteness. */

        HSPAModel *model = hspa_model_create(&cfg);
        ModelGrad *grads = grad_create(&cfg);
        LocalFBState *state = localfb_state_create(&cfg, seed);

        ASSERT_NOT_NULL(model);
        ASSERT_NOT_NULL(grads);
        ASSERT_NOT_NULL(state);
        weight_init_depth_mup(model, &cfg);

        int32_t tokens[LFB_SEQ_LEN], targets[LFB_SEQ_LEN];
        fill_tokens(tokens, targets, LFB_SEQ_LEN, LFB_TEST_V, /*seed=*/seed + 1000u);

        /* Snapshot W_q[0] before the run. */
        int32_t D = cfg.d_model;
        int32_t q_dim = cfg.n_heads * cfg.head_dim;
        int32_t wq_n = D * q_dim;
        float *wq_before = (float *) malloc((size_t) wq_n * sizeof(float));
        ASSERT_NOT_NULL(wq_before);
        memcpy(wq_before, model->layers[0]->attn->W_q->data, (size_t) wq_n * sizeof(float));

        for (int32_t step = 0; step < 10; step++) {
            TrainStepResult r =
                localfb_train_step(model, grads, /*adam=*/NULL, state, tokens, targets, LFB_SEQ_LEN,
                                   &cfg, &tcfg, step, /*micro_batch_idx=*/0);

            /* (a) Every step must be finite (§5.9 FF-BUILDABILITY-2). */
            ASSERT_TRUE(isfinite(r.loss.lm));
            ASSERT_TRUE(isfinite(r.grad_norm));
        }

        /* (b) At least one weight must have changed over 10 steps. */
        const float *wq_after = (const float *) model->layers[0]->attn->W_q->data;
        int changed = 0;
        for (int32_t i = 0; i < wq_n; i++) {
            if (wq_before[i] != wq_after[i]) {
                changed = 1;
                break;
            }
        }
        ASSERT_TRUE(changed);

        /* (c) covered_count must still be 224 on last step. */
        ASSERT_EQUAL_INT(LFB_EXPECTED_TOTAL, localfb_last_qat_covered_count());

        free(wq_before);
        localfb_state_destroy(state);
        grad_destroy(grads);
        hspa_model_destroy(model);
    }
}

/* ==========================================================================
 * Increment-4b-fix tests (TDD RED — written before the B-1/B-2 fix)
 * ==========================================================================*/

/* --------------------------------------------------------------------------
 * TEST 10 — B_ffn_l exists and FFN feedback uses it properly.
 *
 * Uses the 8L/8K config where d_ff=128 > D=64.  The OLD code creates a
 * dl_ff view with shape[1]=d_ff but stride[0]=D, reading past the end of
 * the [seq_len, D] delta_l buffer when d_ff > D (64 floats OOB at test
 * shape).  The NEW code must allocate a real B_ffn_l [D, d_ff] matrix and
 * project delta_top through it to obtain a proper [seq_len, d_ff] buffer.
 *
 * Assertions that pin the fix:
 *   (a) localfb_state_get_B_ffn(state, l) is non-NULL for every layer.
 *       With the old code (no B_ffn_l field), this call does not even link.
 *   (b) B_ffn_l has shape [D, d_ff] = [64, 128].
 *   (c) B_ffn_l is NOT the same pointer as B_l (they are separate allocations).
 *   (d) Run completes with finite loss (no crash / OOB abort).
 *   (e) dW_gate norm > 0 (FFN gradient path executed).
 *   (f) covered_count == 224: B_ffn_l must NOT enter the QAT census.
 *
 * (a) is the primary RED gate: the test will NOT LINK until
 * localfb_state_get_B_ffn is implemented in local_feedback_train.c.
 * --------------------------------------------------------------------------*/
static void test_localfb_ffn_feedback_dimensioned_correctly(void) {
    /* 8L/8K config: d_ff=128 > D=64.  This is the OOB-triggering shape. */
    HSPAConfig cfg = lfb_test_8l8k_config();
    TrainConfig tcfg = lfb_tcfg_enabled(LFB_TEST_D);
    tcfg.use_adam = false;
    tcfg.base_lr = 1e-2f;

    int32_t D    = cfg.d_model;   /* 64 */
    int32_t d_ff = cfg.d_ff;      /* 128 -- d_ff > D here */

    LocalFBState *state = localfb_state_create(&cfg, /*seed=*/777);
    ASSERT_NOT_NULL(state);

    /* (a) + (b) + (c): B_ffn_l must exist for every layer. */
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        const Tensor *B_ffn = localfb_state_get_B_ffn(state, l);
        ASSERT_NOT_NULL(B_ffn);                      /* (a) exists */
        ASSERT_EQUAL_INT(D,    B_ffn->shape[0]);     /* (b) rows = D */
        ASSERT_EQUAL_INT(d_ff, B_ffn->shape[1]);     /* (b) cols = d_ff */

        const Tensor *B = localfb_state_get_B(state, l);
        ASSERT_NOT_NULL(B);
        ASSERT_TRUE(B_ffn->data != B->data);         /* (c) separate allocation */
    }

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    int32_t tokens[LFB_SEQ_LEN], targets[LFB_SEQ_LEN];
    fill_tokens(tokens, targets, LFB_SEQ_LEN, LFB_TEST_V, /*seed=*/3141);

    /* (d) Run one step — this triggers the backward including the FFN path. */
    TrainStepResult r =
        localfb_train_step(model, grads, NULL, state, tokens, targets, LFB_SEQ_LEN, &cfg, &tcfg,
                           /*step=*/0, /*micro_batch_idx=*/0);
    ASSERT_TRUE(isfinite(r.loss.lm));

    /* (e) dW_gate for expert 0, layer 0: at least one element nonzero. */
    {
        const float *dW_gate_data = (const float *) grads->block_grads[0].expert_grads[0].dW_gate->data;
        int32_t dW_gate_numel = D * d_ff;
        float gate_norm = 0.0f;
        for (int32_t i = 0; i < dW_gate_numel; i++) {
            gate_norm += dW_gate_data[i] * dW_gate_data[i];
        }
        ASSERT_TRUE(gate_norm > 0.0f);
        ASSERT_TRUE(isfinite(gate_norm));
    }

    /* (f) covered_count == 224: B_ffn_l must NOT enter the QAT census. */
    ASSERT_EQUAL_INT(LFB_EXPECTED_TOTAL, localfb_last_qat_covered_count());

    localfb_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* --------------------------------------------------------------------------
 * TEST 11 — dW ACCUMULATES across grad_accum_steps > 1.
 *
 * B-2: op_matmul_tn writes beta=0 (overwrite), but LocalFB comments say
 * "+=". When grad_accum_steps=2, micro-batch 0 computes dW and stores it,
 * then micro-batch 1 SHOULD ADD its dW to what is already there.  With the
 * broken code, micro-batch 1 CLOBBERS micro-batch 0 (beta=0), and the final
 * grads hold only micro-batch 1's gradient.  After the 1/accum_steps scale
 * in the optimizer the effective update is only 0.5 * G instead of 1.0 * G.
 *
 * Protocol (both models start from IDENTICAL initial weights):
 *   model_ref: grad_accum_steps=1, one micro-batch → weight update = G * lr.
 *   model_accum: grad_accum_steps=2, SAME tokens run TWICE.
 *     Fixed: grads = G + G = 2G, scaled 1/2 → update = G * lr (== ref).
 *     Broken: grads = G (clobbered), scaled 1/2 → update = 0.5 * G * lr (< ref).
 *
 * Because both models start from identical weights and use the same tokens,
 * the per-element dW is the same G.  The ratio accum_shift_sq / ref_shift_sq
 * should be ~1.0 when fixed, ~0.25 when broken (since we measure squared norm).
 * Assert: accum_shift_sq >= ref_shift_sq * 0.75 (catches the 0.25x broken case).
 *
 * Key implementation note: model_ref and model_accum must start from
 * IDENTICAL weights.  We create both models, copy all weights from ref to
 * accum, THEN run the steps.  This ensures both models produce the same G.
 *
 * IMPORTANT: use micro config to keep test fast; D=d_ff in micro config.
 * --------------------------------------------------------------------------*/
static void test_localfb_dw_accumulates_across_micro_batches(void) {
    /* Micro config: D=64, d_ff=64, so q_dim=D and d_ff=D — clean square case. */
    HSPAConfig cfg = lfb_micro_config();
    int32_t D     = cfg.d_model;
    int32_t q_dim = cfg.n_heads * cfg.head_dim;
    int32_t wq_n  = D * q_dim;

    /* Large LR so the weight shift is numerically distinguishable. */
    TrainConfig tcfg_1 = lfb_tcfg_enabled(LFB_MICRO_D);
    tcfg_1.use_adam         = false;
    tcfg_1.base_lr          = 1e-1f;
    tcfg_1.grad_accum_steps = 1;

    TrainConfig tcfg_2 = tcfg_1;
    tcfg_2.grad_accum_steps = 2;

    /* Same tokens used for all micro-batches: G_batch0 == G_batch1 == G. */
    int32_t tokens[LFB_MICRO_SEQ], targets[LFB_MICRO_SEQ];
    fill_tokens(tokens, targets, LFB_MICRO_SEQ, LFB_MICRO_V, /*seed=*/2025);

    /* -----------------------------------------------------------------------
     * Create BOTH models and copy weights from ref to accum BEFORE any step.
     * This guarantees both models produce the same G when given the same tokens.
     * ----------------------------------------------------------------------- */
    HSPAModel *model_ref = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model_ref);
    weight_init_depth_mup(model_ref, &cfg);

    HSPAModel *model_accum = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model_accum);
    copy_model_weights(model_accum, model_ref, &cfg);  /* identical starting weights */

    ModelGrad *grads_ref   = grad_create(&cfg);
    ModelGrad *grads_accum = grad_create(&cfg);
    ASSERT_NOT_NULL(grads_ref);
    ASSERT_NOT_NULL(grads_accum);

    /* Same LocalFB seed for both so B_l, B_ffn_l, B_kv_l are identical. */
    LocalFBState *state_ref   = localfb_state_create(&cfg, /*seed=*/11);
    LocalFBState *state_accum = localfb_state_create(&cfg, /*seed=*/11);
    ASSERT_NOT_NULL(state_ref);
    ASSERT_NOT_NULL(state_accum);

    /* Snapshot W_q[0] initial state (same for both models). */
    float *wq_init = (float *) malloc((size_t) wq_n * sizeof(float));
    ASSERT_NOT_NULL(wq_init);
    memcpy(wq_init, model_ref->layers[0]->attn->W_q->data, (size_t) wq_n * sizeof(float));

    /* ----- REF: accum_steps=1, one micro-batch ----- */
    localfb_train_step(model_ref, grads_ref, NULL, state_ref,
                       tokens, targets, LFB_MICRO_SEQ, &cfg, &tcfg_1,
                       /*step=*/0, /*micro_batch_idx=*/0);

    /* ref_shift_sq = ||W_q_after - W_q_init||^2 */
    float ref_shift_sq = 0.0f;
    {
        const float *wq_after = (const float *) model_ref->layers[0]->attn->W_q->data;
        for (int32_t i = 0; i < wq_n; i++) {
            float d = wq_after[i] - wq_init[i];
            ref_shift_sq += d * d;
        }
    }
    ASSERT_TRUE(ref_shift_sq > 0.0f);  /* sanity: ref weights actually changed */
    ASSERT_TRUE(isfinite(ref_shift_sq));

    /* ----- ACCUM: accum_steps=2, micro-batch 0 (no optimizer yet) ----- */
    localfb_train_step(model_accum, grads_accum, NULL, state_accum,
                       tokens, targets, LFB_MICRO_SEQ, &cfg, &tcfg_2,
                       /*step=*/0, /*micro_batch_idx=*/0);

    /* Raw dW_q after batch 0 must be nonzero. */
    float dw_after_batch0_sq = 0.0f;
    {
        const float *dw = (const float *) grads_accum->block_grads[0].attn_grad.dW_q->data;
        for (int32_t i = 0; i < wq_n; i++) {
            dw_after_batch0_sq += dw[i] * dw[i];
        }
    }
    ASSERT_TRUE(dw_after_batch0_sq > 0.0f);

    /* ----- ACCUM: micro-batch 1 (triggers optimizer) ----- */
    localfb_train_step(model_accum, grads_accum, NULL, state_accum,
                       tokens, targets, LFB_MICRO_SEQ, &cfg, &tcfg_2,
                       /*step=*/0, /*micro_batch_idx=*/1);

    /* accum_shift_sq = ||W_q_after_accum - W_q_init||^2
     * Fixed:  effective update = (G + G) / 2 = G  → shift_sq ≈ ref_shift_sq.
     * Broken: effective update = G / 2            → shift_sq ≈ ref_shift_sq / 4.
     * Assert: accum_shift_sq >= ref_shift_sq * 0.75 (catches the broken 0.25x case). */
    float accum_shift_sq = 0.0f;
    {
        const float *wq_after = (const float *) model_accum->layers[0]->attn->W_q->data;
        for (int32_t i = 0; i < wq_n; i++) {
            float d = wq_after[i] - wq_init[i];
            accum_shift_sq += d * d;
        }
    }
    ASSERT_TRUE(accum_shift_sq > 0.0f);
    ASSERT_TRUE(isfinite(accum_shift_sq));
    /* Threshold: 0.75 × ref catches the broken 0.25x case with margin. */
    ASSERT_TRUE(accum_shift_sq >= ref_shift_sq * 0.75f);

    free(wq_init);
    localfb_state_destroy(state_ref);
    localfb_state_destroy(state_accum);
    grad_destroy(grads_ref);
    grad_destroy(grads_accum);
    hspa_model_destroy(model_ref);
    hspa_model_destroy(model_accum);
}

/* ==========================================================================
 * TEST 12 — F5: finite-difference gradient-correctness for dW (B-1-RESIDUAL).
 *
 * This is the gold-standard backstop the review (qat_increment4b_fix §F5)
 * pulled FORWARD from 4c because the dimensional-mismatch defect class slipped
 * the producer twice.  It validates the analytic dW the PRODUCTION code
 * (localfb_train_step) writes into `grads`, for one weight of EACH shape
 * family, on the 8L/8K config where d_ff=128 != D=64 — so the d_ff/D
 * distinction (the exact thing that recurred at the dW_down site) is actually
 * exercised, not papered over by a square config.
 *
 * --- Why a DFA *surrogate* loss, not the end-to-end LM loss ---
 * DFA (Nokland 2016) does NOT compute the true gradient dL/dW.  Each layer's
 * weight gradient is formed from a FIXED-RANDOM projection of the global output
 * error (delta_l = delta_top @ B_l, etc.), deliberately bypassing the backprop
 * chain.  A finite difference of the end-to-end LM loss would therefore NOT
 * equal the DFA analytic dW even for perfectly correct code (they differ by
 * design — that is what feedback alignment IS).  The mathematically exact
 * object whose gradient the DFA code computes is the per-layer SURROGATE
 *
 *     S_W = sum_{s,c} ( pre_activation_face(W) )[s,c] * feedback_signal[s,c]
 *
 * i.e. the inner product of the sublayer's *output face* with its fixed
 * feedback signal, holding the feedback signal frozen (DFA does not backprop
 * through it).  dS_W/dW IS exactly the analytic DFA dW.  S_W is linear in W,
 * so the central difference matches the analytic gradient to ~1e-3 relative —
 * a sharp test that goes RED on any dimensional mispairing and GREEN on the
 * correct shapes.
 *
 * --- use_qat MUST be false here (decision (b) in the dispatch) ---
 * Under QAT/STE the quantized forward is piecewise-constant in the FP32 master
 * weight, so a finite difference of it is meaningless.  We run use_qat=false so
 * the forward is differentiable; this validates the dimensional/math
 * correctness of every dW site (which is what recurred).  covered==224 is
 * asserted separately, in the use_qat=true tests (1/8/9/10).
 *
 * --- What this test would have caught ---
 * The broken dW_down fed the D-wide `delta_l` into lfb_accum_tn with M=d_ff:
 * on d_ff>D it over-reads `delta_l` (pool-masked from ASan) AND computes a
 * dimensionally-wrong gradient (a D-vs-d_ff relabel, not h_ffn^T @ delta_l).
 * The W_down leg below reproduces the CORRECT reference (h_ffn_j^T @ delta_l,
 * with the d_ff-wide SwiGLU activation recomputed from the cached h_l proxy)
 * and asserts the production dW_down matches it AND matches the surrogate
 * central difference.  Under the broken code both comparisons fail → RED.
 * ==========================================================================*/

/* Replicate localfb_train_step's FORWARD exactly (frozen RNG) to recover the
 * per-layer input proxy h_l[l] (= post-block hidden of layer l, which the arm
 * uses as the universal activation proxy) and the global error delta_top.
 *
 *   - srand(step+1+micro) BEFORE the layer loop, matching the production
 *     preamble (local_feedback_train.c), so router exploration noise (the only
 *     rand() consumer in the forward) is drawn identically.  This makes the
 *     forward a deterministic function of the weights, so the analytic dW (from
 *     the production step) and the surrogate central difference live on the
 *     SAME loss surface.
 *   - qat=NULL (use_qat=false path): forward is differentiable.
 *
 * h_l_out[l] must be a caller-provided [seq_len, D] buffer for each layer.
 * delta_top_out must be a caller-provided [seq_len, D] buffer. */
static void localfb_ref_forward_signals(HSPAModel *model, const HSPAConfig *cfg,
                                        const int32_t *tokens, const int32_t *targets,
                                        int32_t seq_len, int32_t step, int32_t micro_batch_idx,
                                        float **h_l_out, float *delta_top_out) {
    int32_t L = cfg->n_layers;
    int32_t D = cfg->d_model;
    int32_t V = cfg->vocab_size;

    MemoryPool *pool = pool_create((size_t) seq_len * (size_t) D * sizeof(float) * 32 +
                                       (size_t) seq_len * (size_t) V * sizeof(float) * 4 +
                                       (size_t) D * (size_t) cfg->d_ff * sizeof(float) * 8 +
                                       8 * 1024 * 1024,
                                   POOL_SCRATCH);
    ASSERT_NOT_NULL(pool);
    MemoryPool *blk = pool_create((size_t) seq_len * (size_t) D * sizeof(float) * 16 +
                                      (size_t) D * (size_t) cfg->d_ff * sizeof(float) * 4 +
                                      4 * 1024 * 1024,
                                  POOL_SCRATCH);
    ASSERT_NOT_NULL(blk);

    int32_t sd_shape[] = {seq_len, D};
    int32_t sv_shape[] = {seq_len, V};
    Tensor *x = tensor_create(pool, sd_shape, 2, DTYPE_FP32);
    Tensor *blk_out = tensor_create(pool, sd_shape, 2, DTYPE_FP32);
    Tensor *normed = tensor_create(pool, sd_shape, 2, DTYPE_FP32);
    Tensor *logits = tensor_create(pool, sv_shape, 2, DTYPE_FP32);
    Tensor *grad_logits = tensor_create(pool, sv_shape, 2, DTYPE_FP32);
    Tensor *delta_top = tensor_create(pool, sd_shape, 2, DTYPE_FP32);
    ASSERT_NOT_NULL(x);
    ASSERT_NOT_NULL(blk_out);
    ASSERT_NOT_NULL(normed);
    ASSERT_NOT_NULL(logits);
    ASSERT_NOT_NULL(grad_logits);
    ASSERT_NOT_NULL(delta_top);

    /* Match the production preamble exactly. */
    srand((unsigned int) (step + 1 + micro_batch_idx));

    tensor_fill(x, 0.0f);
    embedding_forward(x, model->embed, tokens, seq_len);

    for (int32_t l = 0; l < L; l++) {
        model->kv_caches[l]->pos = 0;
        pool_reset(blk);
        tensor_fill(blk_out, 0.0f);
        hspa_block_forward(blk_out, model->layers[l], x, model->kv_caches[l], 0, blk,
                           /*training=*/true, cfg->n_active, /*qat=*/NULL);
        /* Capture post-block hidden = the activation proxy h_l[l]. */
        memcpy(h_l_out[l], blk_out->data, (size_t) seq_len * (size_t) D * sizeof(float));
        memcpy(x->data, blk_out->data, (size_t) seq_len * (size_t) D * sizeof(float));
    }

    /* Final RMSNorm (per-row), matching lfb_rms_norm_2d (which calls
     * rmsnorm_forward = op_rms_norm per row). */
    tensor_fill(normed, 0.0f);
    {
        int32_t row_shape[] = {D};
        Tensor *row_in = tensor_create(blk, row_shape, 1, DTYPE_FP32);
        Tensor *row_out = tensor_create(blk, row_shape, 1, DTYPE_FP32);
        ASSERT_NOT_NULL(row_in);
        ASSERT_NOT_NULL(row_out);
        for (int32_t s = 0; s < seq_len; s++) {
            for (int32_t d = 0; d < D; d++) {
                int32_t i2[] = {s, d};
                int32_t i1[] = {d};
                tensor_set(row_in, i1, tensor_get(x, i2));
            }
            rmsnorm_forward(row_out, model->final_norm, row_in);
            for (int32_t d = 0; d < D; d++) {
                int32_t i2[] = {s, d};
                int32_t i1[] = {d};
                tensor_set(normed, i2, tensor_get(row_out, i1));
            }
        }
    }

    /* logits = normed @ W_embed^T  ([S,D] @ [V,D]^T = [S,V]). */
    tensor_fill(logits, 0.0f);
    op_matmul_nt(logits, normed, model->embed->weight);

    /* grad_logits = softmax(logits) - onehot(targets). */
    tensor_fill(grad_logits, 0.0f);
    (void) op_cross_entropy(logits, targets, seq_len, grad_logits);

    /* delta_top = grad_logits @ W_embed  ([S,V] @ [V,D] = [S,D]). */
    tensor_fill(delta_top, 0.0f);
    op_matmul(delta_top, grad_logits, model->embed->weight);

    memcpy(delta_top_out, delta_top->data, (size_t) seq_len * (size_t) D * sizeof(float));

    pool_destroy(blk);
    pool_destroy(pool);
}

/* Surrogate scalar S_W = sum( (pre_act @ W) (.) feedback ), used for the
 * central-difference legs.  pre_act is [S, M_in], W is [M_in, N_out],
 * feedback is [S, N_out].  Returns the inner product of (pre_act @ W) with
 * feedback.  Pure FP32, allocation-free except for one scratch row.
 *
 * NOTE: this is the DFA surrogate output face for a *linear* sublayer
 * (no nonlinearity applied to W's own output) — which is exactly the DFA
 * gradient model for W_q (q-proj), W_gate (gate pre-activation), and W_down
 * (down-proj output): the fixed feedback lands at the linear output face. */
static double localfb_surrogate(const float *pre_act, const float *W, const float *feedback,
                                int32_t S, int32_t M_in, int32_t N_out) {
    double acc = 0.0;
    for (int32_t s = 0; s < S; s++) {
        const float *prow = pre_act + (size_t) s * M_in;
        const float *frow = feedback + (size_t) s * N_out;
        for (int32_t n = 0; n < N_out; n++) {
            double y = 0.0;
            for (int32_t m = 0; m < M_in; m++) {
                y += (double) prow[m] * (double) W[(size_t) m * N_out + n];
            }
            acc += y * (double) frow[n];
        }
    }
    return acc;
}

static void test_localfb_finite_difference_dW(void) {
    /* 8L/8K config: d_ff=128 > D=64.  This is the d_ff != D shape that exercises
     * the dimensional distinction at the dW_down site (the recurred defect). */
    HSPAConfig cfg = lfb_test_8l8k_config();
    TrainConfig tcfg = lfb_tcfg_disabled(LFB_TEST_D); /* use_qat=false (decision b) */
    tcfg.use_adam = false;
    tcfg.base_lr = 1e-2f;
    tcfg.grad_accum_steps = 1;
    /* Disable grad clipping so `grads` holds the raw analytic gradient
     * (a huge max_norm makes grad_clip a no-op). */
    tcfg.grad_clip_norm = 1e30f;

    int32_t L = cfg.n_layers;
    int32_t D = cfg.d_model;       /* 64 */
    int32_t d_ff = cfg.d_ff;       /* 128 != D */
    int32_t q_dim = cfg.n_heads * cfg.head_dim;       /* 64 == D here */
    int32_t S = LFB_SEQ_LEN;
    const int32_t l0 = 0;  /* layer under test */
    const int32_t j0 = 0;  /* expert under test (for FFN families) */

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    /* Snapshot the EXACT pre-step weights into a clone.  We must NOT re-init a
     * fresh model: weight_init_depth_mup() draws from the GLOBAL rand() stream
     * (op_rand_uniform), so a second init after the training step (which itself
     * consumes rand() for router exploration) would produce DIFFERENT weights.
     * copy_model_weights gives a bit-identical clone of the pre-step state. */
    HSPAModel *model0 = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model0);
    copy_model_weights(model0, model, &cfg);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    LocalFBState *state = localfb_state_create(&cfg, /*seed=*/4242);
    ASSERT_NOT_NULL(state);

    int32_t tokens[LFB_SEQ_LEN], targets[LFB_SEQ_LEN];
    fill_tokens(tokens, targets, LFB_SEQ_LEN, LFB_TEST_V, /*seed=*/8675);

    /* ---- 1. Run ONE production training step → grads hold the analytic dW.
     * (step=0, micro=0 → production preamble srand(1).)  use_adam=false,
     * accum=1, no clip → grads are the raw, unscaled analytic DFA gradient.
     * The optimizer mutates `model`'s weights, but `model0` retains the
     * pre-step weights the step's gradient was computed at. */
    TrainStepResult r =
        localfb_train_step(model, grads, /*adam=*/NULL, state, tokens, targets, S, &cfg, &tcfg,
                           /*step=*/0, /*micro_batch_idx=*/0);
    ASSERT_TRUE(isfinite(r.loss.lm));

    /* ---- 2. Reproduce the FROZEN forward signals on model0 (pre-step weights).
     * srand(1) inside the replica matches the production preamble exactly, so
     * router exploration (the only rand() consumer in the forward) is drawn
     * identically → routing matches → analytic dW and surrogate live on the
     * SAME loss surface. */
    float **h_l = (float **) calloc((size_t) L, sizeof(float *));
    ASSERT_NOT_NULL(h_l);
    for (int32_t l = 0; l < L; l++) {
        h_l[l] = (float *) malloc((size_t) S * (size_t) D * sizeof(float));
        ASSERT_NOT_NULL(h_l[l]);
    }
    float *delta_top = (float *) malloc((size_t) S * (size_t) D * sizeof(float));
    ASSERT_NOT_NULL(delta_top);

    localfb_ref_forward_signals(model0, &cfg, tokens, targets, S, /*step=*/0,
                                /*micro_batch_idx=*/0, h_l, delta_top);

    /* Build the three feedback signals exactly as production:
     *   delta_l    = delta_top @ B_l[l0]      [S, D]
     *   delta_ff_l = delta_top @ B_ffn[l0]    [S, d_ff]
     *   delta_kv_l = delta_top @ B_kv[l0]     [S, kv_dim]   (computed, not asserted) */
    const Tensor *B_l = localfb_state_get_B(state, l0);
    const Tensor *B_ffn = localfb_state_get_B_ffn(state, l0);
    ASSERT_NOT_NULL(B_l);
    ASSERT_NOT_NULL(B_ffn);

    float *delta_l = (float *) malloc((size_t) S * (size_t) D * sizeof(float));
    float *delta_ff = (float *) malloc((size_t) S * (size_t) d_ff * sizeof(float));
    ASSERT_NOT_NULL(delta_l);
    ASSERT_NOT_NULL(delta_ff);

    /* delta_l = delta_top @ B_l  (B_l is [D,D] row-major). */
    for (int32_t s = 0; s < S; s++) {
        for (int32_t n = 0; n < D; n++) {
            double acc = 0.0;
            for (int32_t k = 0; k < D; k++) {
                acc += (double) delta_top[(size_t) s * D + k] *
                       (double) ((const float *) B_l->data)[(size_t) k * D + n];
            }
            delta_l[(size_t) s * D + n] = (float) acc;
        }
    }
    /* delta_ff = delta_top @ B_ffn  (B_ffn is [D,d_ff] row-major). */
    for (int32_t s = 0; s < S; s++) {
        for (int32_t n = 0; n < d_ff; n++) {
            double acc = 0.0;
            for (int32_t k = 0; k < D; k++) {
                acc += (double) delta_top[(size_t) s * D + k] *
                       (double) ((const float *) B_ffn->data)[(size_t) k * d_ff + n];
            }
            delta_ff[(size_t) s * d_ff + n] = (float) acc;
        }
    }

    /* ====================================================================
     * FAMILY 1 — dW_q  [D, q_dim].  Production: dW_q = h_l^T @ delta_l.
     * Surrogate: S = <h_l @ W_q, delta_l_q>.  pre_act = h_l [S,D], W = W_q
     * [D,q_dim], feedback = delta_l[:, :q_dim] [S,q_dim].
     * (q_dim == D here, so feedback width == D.)
     * ==================================================================== */
    {
        const float *Wq = (const float *) model0->layers[l0]->attn->W_q->data;
        const float *dWq_prod =
            (const float *) grads->block_grads[l0].attn_grad.dW_q->data;

        /* Reference analytic dW_q[i,j] = sum_s h_l[s,i] * delta_l[s,j]. */
        int32_t i0 = 3, jj0 = 5; /* arbitrary interior element */
        double ref = 0.0;
        for (int32_t s = 0; s < S; s++) {
            ref += (double) h_l[l0][(size_t) s * D + i0] *
                   (double) delta_l[(size_t) s * D + jj0];
        }
        /* Production must match the reference at this element. */
        ASSERT_EQUAL_FLOAT(ref, dWq_prod[(size_t) i0 * q_dim + jj0],
                           1e-3 + 1e-3 * fabs(ref));

        /* Central difference of the surrogate w.r.t. W_q[i0,jj0]. */
        float *Wq_copy = (float *) malloc((size_t) D * (size_t) q_dim * sizeof(float));
        ASSERT_NOT_NULL(Wq_copy);
        memcpy(Wq_copy, Wq, (size_t) D * (size_t) q_dim * sizeof(float));
        float eps = 1e-2f;
        Wq_copy[(size_t) i0 * q_dim + jj0] += eps;
        double Splus = localfb_surrogate(h_l[l0], Wq_copy, delta_l, S, D, q_dim);
        Wq_copy[(size_t) i0 * q_dim + jj0] -= 2.0f * eps;
        double Sminus = localfb_surrogate(h_l[l0], Wq_copy, delta_l, S, D, q_dim);
        double fd = (Splus - Sminus) / (2.0 * (double) eps);
        ASSERT_EQUAL_FLOAT(ref, fd, 1e-2 + 1e-2 * fabs(ref));
        free(Wq_copy);
    }

    /* ====================================================================
     * FAMILY 2 — dW_gate  [D, d_ff].  Production: dW_gate = h_l^T @ delta_ff.
     * Surrogate: S = <h_l @ W_gate, delta_ff>.  pre_act = h_l [S,D],
     * W = W_gate [D,d_ff], feedback = delta_ff [S,d_ff].  Exercises the
     * d_ff-wide feedback (the OTHER half of the d_ff/D distinction).
     * ==================================================================== */
    {
        const float *Wg = (const float *) model0->layers[l0]->experts[j0]->W_gate->data;
        const float *dWg_prod =
            (const float *) grads->block_grads[l0].expert_grads[j0].dW_gate->data;

        int32_t i0 = 7, jj0 = 100; /* jj0 in [D, d_ff): only valid if d_ff-wide. */
        double ref = 0.0;
        for (int32_t s = 0; s < S; s++) {
            ref += (double) h_l[l0][(size_t) s * D + i0] *
                   (double) delta_ff[(size_t) s * d_ff + jj0];
        }
        ASSERT_EQUAL_FLOAT(ref, dWg_prod[(size_t) i0 * d_ff + jj0],
                           1e-3 + 1e-3 * fabs(ref));

        float *Wg_copy = (float *) malloc((size_t) D * (size_t) d_ff * sizeof(float));
        ASSERT_NOT_NULL(Wg_copy);
        memcpy(Wg_copy, Wg, (size_t) D * (size_t) d_ff * sizeof(float));
        float eps = 1e-2f;
        Wg_copy[(size_t) i0 * d_ff + jj0] += eps;
        double Splus = localfb_surrogate(h_l[l0], Wg_copy, delta_ff, S, D, d_ff);
        Wg_copy[(size_t) i0 * d_ff + jj0] -= 2.0f * eps;
        double Sminus = localfb_surrogate(h_l[l0], Wg_copy, delta_ff, S, D, d_ff);
        double fd = (Splus - Sminus) / (2.0 * (double) eps);
        ASSERT_EQUAL_FLOAT(ref, fd, 1e-2 + 1e-2 * fabs(ref));
        free(Wg_copy);
    }

    /* ====================================================================
     * FAMILY 3 — dW_down  [d_ff, D].  THE RECURRED DEFECT SITE.
     * Production (correct): dW_down_j = h_ffn_j^T @ delta_l, where h_ffn_j is
     * the d_ff-wide SwiGLU activation silu(h_l @ W_gate_j) (.) (h_l @ W_up_j).
     * The M-side is genuinely d_ff-wide (h_ffn_j), the feedback is D-wide
     * (delta_l).  The broken code fed the D-wide delta_l with M=d_ff → wrong.
     *
     * Surrogate: S = <h_ffn_j @ W_down_j, delta_l>.  pre_act = h_ffn_j
     * [S,d_ff], W = W_down_j [d_ff,D], feedback = delta_l [S,D].
     * Perturbed element is in row i0 in [0,d_ff) — a row index that ONLY
     * exists because the M-side is d_ff-wide.  Under the broken (D-wide M)
     * code the production dW_down for rows >= D is structurally wrong.
     * ==================================================================== */
    {
        /* Recompute h_ffn_j0 = silu(h_l @ W_gate_j0) (.) (h_l @ W_up_j0). */
        const Tensor *Wgate_t = model0->layers[l0]->experts[j0]->W_gate; /* [D,d_ff] */
        const Tensor *Wup_t = model0->layers[l0]->experts[j0]->W_up;     /* [D,d_ff] */
        const float *Wdown = (const float *) model0->layers[l0]->experts[j0]->W_down->data;
        const float *dWdown_prod =
            (const float *) grads->block_grads[l0].expert_grads[j0].dW_down->data;

        /* Build h_l as a Tensor view to reuse op_matmul / op_swiglu. */
        int32_t sd[] = {S, D};
        int32_t sff[] = {S, d_ff};
        MemoryPool *p = pool_create((size_t) S * (size_t) d_ff * sizeof(float) * 8 +
                                        1 * 1024 * 1024,
                                    POOL_SCRATCH);
        ASSERT_NOT_NULL(p);
        Tensor *h_l_t = tensor_create(p, sd, 2, DTYPE_FP32);
        Tensor *gate = tensor_create(p, sff, 2, DTYPE_FP32);
        Tensor *up = tensor_create(p, sff, 2, DTYPE_FP32);
        Tensor *hffn = tensor_create(p, sff, 2, DTYPE_FP32);
        ASSERT_NOT_NULL(h_l_t);
        ASSERT_NOT_NULL(gate);
        ASSERT_NOT_NULL(up);
        ASSERT_NOT_NULL(hffn);
        memcpy(h_l_t->data, h_l[l0], (size_t) S * (size_t) D * sizeof(float));
        op_matmul(gate, h_l_t, Wgate_t); /* [S,D]@[D,d_ff] = [S,d_ff] */
        op_matmul(up, h_l_t, Wup_t);
        op_swiglu(hffn, gate, up); /* silu(gate) (.) up */
        const float *h_ffn = (const float *) hffn->data;

        /* Reference: dW_down[i,j] = sum_s h_ffn[s,i] * delta_l[s,j].
         * Pick a row i0 in [D, d_ff) — ONLY valid because M is d_ff-wide. */
        int32_t i0 = D + 17; /* 81, in [64,128): the broken code can't reach this correctly */
        int32_t jj0 = 9;     /* column in [0, D) */
        ASSERT_TRUE(i0 < d_ff);
        ASSERT_TRUE(jj0 < D);
        double ref = 0.0;
        for (int32_t s = 0; s < S; s++) {
            ref += (double) h_ffn[(size_t) s * d_ff + i0] *
                   (double) delta_l[(size_t) s * D + jj0];
        }
        /* The production dW_down MUST match the d_ff-wide reference. */
        ASSERT_EQUAL_FLOAT(ref, dWdown_prod[(size_t) i0 * D + jj0],
                           1e-3 + 1e-3 * fabs(ref));

        /* Also validate a row in [0, D) so we cover both regions. */
        int32_t i1 = 5, jj1 = 2;
        double ref1 = 0.0;
        for (int32_t s = 0; s < S; s++) {
            ref1 += (double) h_ffn[(size_t) s * d_ff + i1] *
                    (double) delta_l[(size_t) s * D + jj1];
        }
        ASSERT_EQUAL_FLOAT(ref1, dWdown_prod[(size_t) i1 * D + jj1],
                           1e-3 + 1e-3 * fabs(ref1));

        /* Central difference of the surrogate w.r.t. W_down[i0,jj0]. */
        float *Wd_copy = (float *) malloc((size_t) d_ff * (size_t) D * sizeof(float));
        ASSERT_NOT_NULL(Wd_copy);
        memcpy(Wd_copy, Wdown, (size_t) d_ff * (size_t) D * sizeof(float));
        float eps = 1e-2f;
        Wd_copy[(size_t) i0 * D + jj0] += eps;
        double Splus = localfb_surrogate(h_ffn, Wd_copy, delta_l, S, d_ff, D);
        Wd_copy[(size_t) i0 * D + jj0] -= 2.0f * eps;
        double Sminus = localfb_surrogate(h_ffn, Wd_copy, delta_l, S, d_ff, D);
        double fd = (Splus - Sminus) / (2.0 * (double) eps);
        ASSERT_EQUAL_FLOAT(ref, fd, 1e-2 + 1e-2 * fabs(ref));
        free(Wd_copy);

        pool_destroy(p);
    }

    for (int32_t l = 0; l < L; l++) {
        free(h_l[l]);
    }
    free(h_l);
    free(delta_top);
    free(delta_l);
    free(delta_ff);
    localfb_state_destroy(state);
    grad_destroy(grads);
    hspa_model_destroy(model0);
    hspa_model_destroy(model);
}

/* ==========================================================================
 * Main
 * ==========================================================================*/

int main(void) {
    printf("========================================\n");
    printf("  LocalFB Forward + Coverage Tests\n");
    printf("  (Program 3 P7 increment-4a/4b, D-613)\n");
    printf("========================================\n");

    printf("\n--- Clause 1: 224-coverage (8L/8K config) ---\n");
    RUN_TEST(test_localfb_qat_covered_count_equals_224);

    printf("\n--- Clause 2: disabled pass-through ---\n");
    RUN_TEST(test_localfb_disabled_passthrough);

    printf("\n--- B_l determinism ---\n");
    RUN_TEST(test_localfb_bl_determinism);

    printf("\n--- begin_step freshness ---\n");
    RUN_TEST(test_localfb_begin_step_clears);

    printf("\n--- Memory hygiene ---\n");
    RUN_TEST(test_localfb_state_create_destroy);

    printf("\n--- 4b: B_l orthogonal init (Launay 2019) ---\n");
    RUN_TEST(test_localfb_bl_orthogonal_init);

    printf("\n--- 4b: B_l srand isolation (local PRNG) ---\n");
    RUN_TEST(test_localfb_bl_srand_isolation);

    printf("\n--- 4b: DFA backward weights change ---\n");
    RUN_TEST(test_localfb_backward_weights_change);

    printf("\n--- 4b: 10-step smoke + 5 seeds (FF-BUILDABILITY-2) ---\n");
    RUN_TEST(test_localfb_10step_smoke_all_seeds);

    printf("\n--- 4b-fix: FFN feedback [seq_len, d_ff] dimensioned (B-1) ---\n");
    RUN_TEST(test_localfb_ffn_feedback_dimensioned_correctly);

    printf("\n--- 4b-fix: dW accumulates across grad_accum_steps>1 (B-2) ---\n");
    RUN_TEST(test_localfb_dw_accumulates_across_micro_batches);

    printf("\n--- 4b-fix-2: F5 finite-difference dW correctness (B-1-RESIDUAL) ---\n");
    RUN_TEST(test_localfb_finite_difference_dW);

    TEST_REPORT();
}
