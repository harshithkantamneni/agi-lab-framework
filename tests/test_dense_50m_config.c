/* test_dense_50m_config.c -- Tests for Dense-A (Program 2 compute-matched control) config.
 *
 * Validates the `hspa_config_dense_50m_a()` factory required as apparatus for
 * Dense-50M config PC-1 pre-flight.
 *
 * Pre-reg spec (dense_50m_control_design.md §1.3, question.md operational form):
 *   L=8, D=512, H=8, n_kv_heads=4, head_dim=64, d_ff=768, V=32768,
 *   n_experts=1, n_active=1, max_seq_len=512, rms_norm_eps=1e-5f,
 *   FP32 storage/compute, ipc_iterations=5
 *   Total params ~34.7M (± 0.5M tolerance per Director dispatch)
 *   FLOPs-matched to Rev-2 MoE active path (d_ff_dense = 2 * d_ff_moe
 *   = 2 * 384 = 768, compensating for k=2 active experts in the MoE).
 *
 * Tests:
 *   1. test_dense_a_fields         -- All field values match locked spec
 *   2. test_dense_a_head_dim       -- H * head_dim == D (invariant)
 *   3. test_dense_a_gqa_ratio      -- n_heads / n_kv_heads == 2 (GQA 2:1 preserved)
 *   4. test_dense_a_n_experts_one  -- n_experts==1 AND n_active==1 (defining Dense-A)
 *   5. test_dense_a_param_count    -- Canonical formula yields ~34.7M (± 0.5M)
 *   6. test_dense_a_vocab          -- V == 32768 (Program 2 benchmark vocab)
 *   7. test_dense_a_dff_flops_match-- d_ff == 768 == 2 * 384 (MoE-active FLOPs match)
 */

#include "../src/tests/unity.h"
#include "hspa_config.h"
#include "ipc_train.h"

#include <stdint.h>

/* Canonical parameter count per scale_experiment.c and phase_c_design §1.2.
 * total = V*D + L * (4*D*H*head_dim + K*3*D*d_ff + 2*D*K + 2*D) + D
 *
 * At K=1 (Dense-A): the K*3*D*d_ff term collapses to 3*D*d_ff (single dense FFN),
 * and the 2*D*K router term collapses to 2*D (one mu, one sigma). Router at K=1
 * is structurally present but informationally trivial (softmax of 1 = 1.0). */
static int64_t canonical_param_count(const HSPAConfig *cfg) {
    int64_t V = cfg->vocab_size;
    int64_t D = cfg->d_model;
    int64_t L = cfg->n_layers;
    int64_t H = cfg->n_heads;
    int64_t Hd = cfg->head_dim;
    int64_t K = cfg->n_experts;
    int64_t Dff = cfg->d_ff;
    int64_t per_layer = 4 * D * H * Hd      /* attention QKVO */
                      + K * 3 * D * Dff     /* expert gate+up+down (K=1 for Dense-A) */
                      + 2 * D * K           /* router mu+sigma */
                      + 2 * D;              /* attn_norm + ffn_norm */
    return V * D + L * per_layer + D;       /* embedding + layers + final norm */
}

static void test_dense_a_fields(void) {
    HSPAConfig cfg = hspa_config_dense_50m_a();
    ASSERT_EQUAL_INT(8,     cfg.n_layers);
    ASSERT_EQUAL_INT(512,   cfg.d_model);
    ASSERT_EQUAL_INT(8,     cfg.n_heads);
    ASSERT_EQUAL_INT(4,     cfg.n_kv_heads);
    ASSERT_EQUAL_INT(64,    cfg.head_dim);
    ASSERT_EQUAL_INT(768,   cfg.d_ff);
    ASSERT_EQUAL_INT(1,     cfg.n_experts);
    ASSERT_EQUAL_INT(1,     cfg.n_active);
    ASSERT_EQUAL_INT(32768, cfg.vocab_size);
    ASSERT_EQUAL_INT(512,   cfg.max_seq_len);
    ASSERT_EQUAL_INT(5,     cfg.ipc_iterations);
    ASSERT_EQUAL_FLOAT(1e-5f, cfg.rms_norm_eps, 1e-9);
    ASSERT_EQUAL_INT((int)DTYPE_FP32, (int)cfg.storage_dtype);
    ASSERT_EQUAL_INT((int)DTYPE_FP32, (int)cfg.compute_dtype);
}

static void test_dense_a_head_dim(void) {
    HSPAConfig cfg = hspa_config_dense_50m_a();
    /* H * head_dim must equal D (H=8, head_dim=64, D=512). */
    ASSERT_EQUAL_INT(cfg.d_model, cfg.n_heads * cfg.head_dim);
}

static void test_dense_a_gqa_ratio(void) {
    HSPAConfig cfg = hspa_config_dense_50m_a();
    /* GQA 2:1 ratio preserved (H=8, n_kv_heads=4). */
    ASSERT_EQUAL_INT(2, cfg.n_heads / cfg.n_kv_heads);
    /* No remainder — n_kv_heads divides n_heads evenly. */
    ASSERT_EQUAL_INT(0, cfg.n_heads % cfg.n_kv_heads);
}

static void test_dense_a_n_experts_one(void) {
    HSPAConfig cfg = hspa_config_dense_50m_a();
    /* Defining property: Dense-A has exactly one expert, always active.
     * This distinguishes it from all MoE factories (micro/small/medium/100m
     * all have n_experts>=4 with n_active>=2). */
    ASSERT_EQUAL_INT(1, cfg.n_experts);
    ASSERT_EQUAL_INT(1, cfg.n_active);
    ASSERT_TRUE(cfg.n_experts == cfg.n_active);  /* 100% activation */
}

static void test_dense_a_param_count(void) {
    HSPAConfig cfg = hspa_config_dense_50m_a();
    int64_t total = canonical_param_count(&cfg);
    /* Target 34.7M per question.md operational form; tolerance ±0.5M per
     * Director dispatch. Canonical formula yields 34,619,904 from the locked
     * spec (V=32768, D=512, L=8, H=8, Hd=64, K=1, Dff=768):
     *   V*D        = 16,777,216
     *   per_layer  = 4*512*8*64 + 1*3*512*768 + 2*512*1 + 2*512
     *              = 1,048,576 + 1,179,648 + 1,024 + 1,024 = 2,230,272
     *   L*per_layer= 17,842,176
     *   + D        = 512
     *   total      = 34,619,904 */
    int64_t expected_target = 34700000;  /* 34.7M per question.md */
    int64_t tolerance       = 500000;    /* ±0.5M per dispatch */
    int64_t delta = total > expected_target ? (total - expected_target)
                                              : (expected_target - total);
    ASSERT_TRUE(delta <= tolerance);
    /* Also assert the deterministic canonical value (formula is pure integer
     * arithmetic so the computed value must be exact). This pins the factory
     * against silent drift. */
    ASSERT_EQUAL_INT(34619904, total);
}

static void test_dense_a_vocab(void) {
    HSPAConfig cfg = hspa_config_dense_50m_a();
    /* V=32768 matches Program 2 benchmark tokenizer + MoE comparator. */
    ASSERT_EQUAL_INT(32768, cfg.vocab_size);
}

static void test_dense_a_dff_flops_match(void) {
    HSPAConfig cfg = hspa_config_dense_50m_a();
    /* FLOPs match with Rev-2 MoE active path: MoE has d_ff=384 with k=2
     * active experts per token → dense-equivalent d_ff = 2*384 = 768.
     * This is the compute-match that makes the dense-vs-MoE comparison
     * meaningful (per question.md operational form). */
    ASSERT_EQUAL_INT(768, cfg.d_ff);
    ASSERT_EQUAL_INT(2 * 384, cfg.d_ff);
}

int main(void) {
    printf("=== Dense-A (Program 2 control) config tests ===\n");
    RUN_TEST(test_dense_a_fields);
    RUN_TEST(test_dense_a_head_dim);
    RUN_TEST(test_dense_a_gqa_ratio);
    RUN_TEST(test_dense_a_n_experts_one);
    RUN_TEST(test_dense_a_param_count);
    RUN_TEST(test_dense_a_vocab);
    RUN_TEST(test_dense_a_dff_flops_match);
    TEST_REPORT();
}
