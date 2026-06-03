/* test_phase_c_config.c -- Tests for Phase C 100M config factory.
 *
 * Validates the `hspa_config_100m()` factory added for Phase C scale-up
 * per data/engineering/phase_c_design.md §1.7.
 *
 * Tests:
 *   1. test_phase_c_100m_fields      -- Field values match spec (§1.7 table)
 *   2. test_phase_c_100m_head_dim    -- H * head_dim == D (invariant)
 *   3. test_phase_c_100m_gqa_ratio   -- n_heads / n_kv_heads == 2 (GQA 2:1)
 *   4. test_phase_c_100m_param_count -- Canonical formula yields 119,649,024
 *   5. test_phase_c_100m_vocab       -- V defaults to 32768 (Phase B default)
 *   6. test_phase_c_100m_vs_medium   -- All unchanged fields match medium
 *
 * Chief_scientist spec (phase_c_design.md §1.7):
 *   L=8, D=768, H=12, n_kv_heads=6, head_dim=64, d_ff=512, K=8, k=2, V=32768
 *   max_seq_len=512, rms_norm_eps=1e-5, FP32 storage/compute
 */

#include "../src/tests/unity.h"
#include "hspa_config.h"
#include "ipc_train.h"

#include <stdint.h>

/* Canonical parameter count per scale_experiment.c:387-395 and phase_c_design §1.2.
 * total = V*D + L * (4*D*H*head_dim + K*3*D*d_ff + 2*D*K + 2*D) + D */
static int64_t canonical_param_count(const HSPAConfig *cfg) {
    int64_t V = cfg->vocab_size;
    int64_t D = cfg->d_model;
    int64_t L = cfg->n_layers;
    int64_t H = cfg->n_heads;
    int64_t Hd = cfg->head_dim;
    int64_t K = cfg->n_experts;
    int64_t Dff = cfg->d_ff;
    int64_t per_layer = 4 * D * H * Hd      /* attention QKVO */
                      + K * 3 * D * Dff     /* expert gate+up+down */
                      + 2 * D * K           /* router mu+sigma */
                      + 2 * D;              /* attn_norm + ffn_norm */
    return V * D + L * per_layer + D;       /* embedding + layers + final norm */
}

static void test_phase_c_100m_fields(void) {
    HSPAConfig cfg = hspa_config_100m();
    ASSERT_EQUAL_INT(8,     cfg.n_layers);
    ASSERT_EQUAL_INT(768,   cfg.d_model);
    ASSERT_EQUAL_INT(12,    cfg.n_heads);
    ASSERT_EQUAL_INT(6,     cfg.n_kv_heads);
    ASSERT_EQUAL_INT(64,    cfg.head_dim);
    ASSERT_EQUAL_INT(512,   cfg.d_ff);
    ASSERT_EQUAL_INT(8,     cfg.n_experts);
    ASSERT_EQUAL_INT(2,     cfg.n_active);
    ASSERT_EQUAL_INT(32768, cfg.vocab_size);
    ASSERT_EQUAL_INT(512,   cfg.max_seq_len);
    ASSERT_EQUAL_FLOAT(1e-5f, cfg.rms_norm_eps, 1e-9);
    ASSERT_EQUAL_INT((int)DTYPE_FP32, (int)cfg.storage_dtype);
    ASSERT_EQUAL_INT((int)DTYPE_FP32, (int)cfg.compute_dtype);
}

static void test_phase_c_100m_head_dim(void) {
    HSPAConfig cfg = hspa_config_100m();
    /* H * head_dim must equal D (per §1.7: H=12, head_dim=64, D=768). */
    ASSERT_EQUAL_INT(cfg.d_model, cfg.n_heads * cfg.head_dim);
}

static void test_phase_c_100m_gqa_ratio(void) {
    HSPAConfig cfg = hspa_config_100m();
    /* GQA 2:1 ratio preserved from medium (H=12, n_kv_heads=6). */
    ASSERT_EQUAL_INT(2, cfg.n_heads / cfg.n_kv_heads);
    /* No remainder — n_kv_heads divides n_heads evenly. */
    ASSERT_EQUAL_INT(0, cfg.n_heads % cfg.n_kv_heads);
}

static void test_phase_c_100m_param_count(void) {
    HSPAConfig cfg = hspa_config_100m();
    int64_t total = canonical_param_count(&cfg);
    /* Expected exactly 119,649,024 per phase_c_design.md §1.2.
     * Allow ±10 tolerance per task spec (rounding guard — formula is exact
     * integer arithmetic so actual delta should be 0). */
    int64_t expected = 119649024;
    int64_t delta = total > expected ? (total - expected) : (expected - total);
    ASSERT_TRUE(delta <= 10);
    /* Also assert the exact value holds (delta == 0) — the formula is pure
     * integer arithmetic so there's no rounding source here. */
    ASSERT_EQUAL_INT(expected, total);
}

static void test_phase_c_100m_vocab(void) {
    HSPAConfig cfg = hspa_config_100m();
    /* Phase C defaults vocab to 32768 (was 4096 for medium), per §1.7
     * "vocab_size (V): 32768 default NOW". */
    ASSERT_EQUAL_INT(32768, cfg.vocab_size);
}

static void test_phase_c_100m_vs_medium(void) {
    HSPAConfig m = hspa_config_medium();
    HSPAConfig c = hspa_config_100m();
    /* Unchanged per §1.7: n_layers, head_dim, n_experts, n_active,
     * max_seq_len, rms_norm_eps, storage/compute dtype. */
    ASSERT_EQUAL_INT(m.n_layers,     c.n_layers);
    ASSERT_EQUAL_INT(m.head_dim,     c.head_dim);
    ASSERT_EQUAL_INT(m.n_experts,    c.n_experts);
    ASSERT_EQUAL_INT(m.n_active,     c.n_active);
    ASSERT_EQUAL_INT(m.max_seq_len,  c.max_seq_len);
    ASSERT_EQUAL_FLOAT(m.rms_norm_eps, c.rms_norm_eps, 1e-9);
    ASSERT_EQUAL_INT((int)m.storage_dtype, (int)c.storage_dtype);
    ASSERT_EQUAL_INT((int)m.compute_dtype, (int)c.compute_dtype);
    /* Changed per §1.7: d_model, n_heads, n_kv_heads, d_ff, vocab_size. */
    ASSERT_TRUE(c.d_model    > m.d_model);
    ASSERT_TRUE(c.n_heads    > m.n_heads);
    ASSERT_TRUE(c.n_kv_heads > m.n_kv_heads);
    ASSERT_TRUE(c.d_ff       > m.d_ff);
    ASSERT_TRUE(c.vocab_size >= m.vocab_size);
}

int main(void) {
    printf("=== Phase C 100M config tests ===\n");
    RUN_TEST(test_phase_c_100m_fields);
    RUN_TEST(test_phase_c_100m_head_dim);
    RUN_TEST(test_phase_c_100m_gqa_ratio);
    RUN_TEST(test_phase_c_100m_param_count);
    RUN_TEST(test_phase_c_100m_vocab);
    RUN_TEST(test_phase_c_100m_vs_medium);
    TEST_REPORT();
}
