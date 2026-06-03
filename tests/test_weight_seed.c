/* test_weight_seed.c -- Regression test for the `--weight-seed <int>` CLI flag.
 *
 * Context
 * =======
 * Prereq for Grant Rec #2 (N=3 multi-seed 17M × 300-step protocol per
 * data/engineering/multiseed_10m_rev2_design.md §10). Before this patch,
 * scale_experiment.c seeded the RNG via srand((unsigned int)time(NULL)),
 * so back-to-back launches within the same wall-clock second shared the
 * SAME seed and seed-variance numbers were untrustworthy.
 *
 * What this test verifies
 * =======================
 * The test does NOT invoke the full scale_experiment binary. It exercises
 * the exact same init path the binary uses:
 *     srand(seed_value);
 *     weight_init_depth_mup(model, &cfg);
 * This is the chain from `scale_experiment.c:437` (srand) to the per-layer
 * `init_normal()` calls in `src/training/weight_init.c` (via op_rand_uniform
 * -> rand()). If the CLI flag wiring is correct, the same `seed_value` fed
 * to srand() here will match what the binary produces for
 * `--weight-seed <seed_value>` at the same init-path call site.
 *
 * Tests
 * -----
 *   1. `srand(42)` twice produces BYTE-IDENTICAL weights across two
 *       independent model instances (the reproducibility claim).
 *   2. `srand(42)` vs `srand(43)` produces DIFFERENT weights (distinct
 *       seeds reach distinct RNG state).
 *   3. `srand(time(NULL))` smoke — two invocations ≥1 second apart produce
 *       different weights (non-determinism of the default / legacy path).
 *       This is the "default behavior unchanged" regression guard.
 *
 * Micro-model
 * -----------
 * Uses a tiny micro config (D=64, L=2, K=4, V=256, d_ff=128) that matches
 * tests/test_backprop_balance_regression.c. The full scale_experiment
 * binary with 17M params would be too heavy for a unit test; the same
 * srand + weight_init_depth_mup chain is exercised either way.
 */

#include "../src/tests/unity.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "tensor.h"
#include "weight_init.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* Micro config — matches test_backprop_balance_regression.c. Tiny but exercises
 * every init path in weight_init_depth_mup (embedding, attention, FFN, router). */
#define MODEL_D 64
#define MODEL_L 2
#define MODEL_K 4
#define MODEL_k 2
#define MODEL_V 256
#define MODEL_DFF 128

static HSPAConfig micro_config(void) {
    HSPAConfig cfg;
    cfg.d_model        = MODEL_D;
    cfg.n_layers       = MODEL_L;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = MODEL_D / 4;
    cfg.n_experts      = MODEL_K;
    cfg.n_active       = MODEL_k;
    cfg.d_ff           = MODEL_DFF;
    cfg.vocab_size     = MODEL_V;
    cfg.max_seq_len    = 32;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* Seed RNG, build a fresh model, init weights. The caller owns the model
 * and must destroy it. */
static HSPAModel *seed_and_init(unsigned int seed, const HSPAConfig *cfg) {
    srand(seed);
    HSPAModel *m = hspa_model_create(cfg);
    if (!m) return NULL;
    weight_init_depth_mup(m, cfg);
    return m;
}

/* Compare embedding weights for byte-equality. Embedding is the FIRST
 * thing weight_init_depth_mup initializes (see weight_init.c:65), so any
 * seed-induced divergence shows up here. We also compare layer-0 W_q to
 * make sure the chain continues identically past embedding. Returns 0 if
 * identical, positive bytes-differing count otherwise. */
static size_t count_weight_bytes_diff(const HSPAModel *a, const HSPAModel *b) {
    size_t diffs = 0;

    /* Embedding */
    {
        size_t n = (size_t)a->embed->weight->shape[0]
                 * (size_t)a->embed->weight->shape[1];
        const float *fa = (const float *)a->embed->weight->data;
        const float *fb = (const float *)b->embed->weight->data;
        for (size_t i = 0; i < n; i++) {
            if (fa[i] != fb[i]) diffs++;
        }
    }
    /* Layer 0 W_q (sample from downstream init path) */
    {
        const Tensor *ta = a->layers[0]->attn->W_q;
        const Tensor *tb = b->layers[0]->attn->W_q;
        size_t n = (size_t)ta->shape[0] * (size_t)ta->shape[1];
        const float *fa = (const float *)ta->data;
        const float *fb = (const float *)tb->data;
        for (size_t i = 0; i < n; i++) {
            if (fa[i] != fb[i]) diffs++;
        }
    }
    /* Layer 0 router W_mu (sample from third init path — router_mu_scale) */
    {
        const Tensor *ta = a->layers[0]->router->W_mu;
        const Tensor *tb = b->layers[0]->router->W_mu;
        size_t n = (size_t)ta->shape[0] * (size_t)ta->shape[1];
        const float *fa = (const float *)ta->data;
        const float *fb = (const float *)tb->data;
        for (size_t i = 0; i < n; i++) {
            if (fa[i] != fb[i]) diffs++;
        }
    }
    return diffs;
}

/* ========================================================================
 * TEST 1: Same seed -> byte-identical weights
 * ======================================================================== */
static void test_same_seed_reproducible(void) {
    TEST_BEGIN(same_seed_reproducible);
    HSPAConfig cfg = micro_config();

    HSPAModel *a = seed_and_init(42u, &cfg);
    ASSERT_NOT_NULL(a);
    HSPAModel *b = seed_and_init(42u, &cfg);
    ASSERT_NOT_NULL(b);

    size_t diffs = count_weight_bytes_diff(a, b);
    hspa_model_destroy(a);
    hspa_model_destroy(b);

    if (diffs != 0) {
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "REGRESSION: --weight-seed 42 ran twice produced %zu "
                 "differing weight elements. Seed-based reproducibility is "
                 "broken. Check srand() call site in scale_experiment.c.",
                 diffs);
        TEST_FAIL(buf);
    }
    TEST_END();
}

/* ========================================================================
 * TEST 2: Different seeds -> different weights
 * ======================================================================== */
static void test_different_seed_diverges(void) {
    TEST_BEGIN(different_seed_diverges);
    HSPAConfig cfg = micro_config();

    HSPAModel *a = seed_and_init(42u, &cfg);
    ASSERT_NOT_NULL(a);
    HSPAModel *b = seed_and_init(43u, &cfg);
    ASSERT_NOT_NULL(b);

    size_t diffs = count_weight_bytes_diff(a, b);
    hspa_model_destroy(a);
    hspa_model_destroy(b);

    if (diffs == 0) {
        TEST_FAIL("REGRESSION: --weight-seed 42 and --weight-seed 43 produced "
                  "byte-identical weights. srand() is not actually seeding the "
                  "Box-Muller RNG used by weight_init_depth_mup.");
    }
    TEST_END();
}

/* ========================================================================
 * TEST 3: Default (time-based) smoke -- two calls ≥1s apart should diverge
 * ======================================================================== */
static void test_time_seed_diverges_smoke(void) {
    TEST_BEGIN(time_seed_diverges_smoke);
    HSPAConfig cfg = micro_config();

    HSPAModel *a = seed_and_init((unsigned int)time(NULL), &cfg);
    ASSERT_NOT_NULL(a);
    sleep(1);  /* ensure distinct wall-clock second */
    HSPAModel *b = seed_and_init((unsigned int)time(NULL), &cfg);
    ASSERT_NOT_NULL(b);

    size_t diffs = count_weight_bytes_diff(a, b);
    hspa_model_destroy(a);
    hspa_model_destroy(b);

    /* This is a smoke test: without the explicit flag, the default path
     * (time(NULL)) should continue to be non-deterministic across seconds.
     * If this ever asserts zero diffs, either the system clock is broken
     * or srand() is being overridden. */
    if (diffs == 0) {
        TEST_FAIL("SMOKE: time-based seeds 1 second apart produced identical "
                  "weights. Either the legacy default path is accidentally "
                  "deterministic or the clock is frozen.");
    }
    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */
int main(void) {
    printf("\n========================================\n");
    printf("  Weight-Seed Reproducibility Tests\n");
    printf("========================================\n");
    printf("  Prereq for Grant Rec #2 (multi-seed N=3 protocol).\n");
    printf("  Spec: data/engineering/multiseed_10m_rev2_design.md §10.\n");
    printf("\n");

    test_same_seed_reproducible();
    test_different_seed_diverges();
    test_time_seed_diverges_smoke();

    TEST_REPORT();
}
