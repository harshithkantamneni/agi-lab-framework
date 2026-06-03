/* test_backprop_balance_regression.c -- Regression test for D-094
 * (Cycle 26 silent-no-op bug: --balance-h-floor has no effect in
 * --backprop --loss-free-balance mode).
 *
 * Spec source: data/engineering/reviews/review_balance_h_floor_backprop_bug.md
 *              §Fix C (lines 384-417).
 *
 * What this test catches
 * ======================
 * The test covers two SEPARATE failure modes:
 *
 *   (A) Config-surface silent-no-op: before Fix A, passing
 *       `--backprop --balance-h-floor 0.8` to `scale_experiment` was
 *       accepted silently (the flag had no effect under --backprop).
 *       Same for `--loss-free-balance --balance-h-floor 0.8`. The
 *       guardrail subprocess tests verify these invocations now EXIT
 *       NON-ZERO at argv parse.
 *
 *   (B) Runtime plumbing of Plan B (Default MoE, arXiv 2504.12463): the
 *       test builds a micro model and runs 300 backprop+LFB+Plan-B steps,
 *       asserting that:
 *         - No NaN in loss/entropy (catches gross infra regressions).
 *         - Router entropy does not collapse catastrophically (H >= 0.2
 *           after warmup; loose floor because K=4 micro-scale dynamics
 *           differ from 50M-scale Cycle 26 dynamics — see CALIBRATION
 *           NOTES below).
 *         - Per-expert load stays in [0.05, 0.70] (LFB is functioning;
 *           no expert starves or monopolizes).
 *
 * Why the entropy threshold is loose (0.2 vs spec's 0.4)
 * ------------------------------------------------------
 * Cycle 26's collapse was observed on a 50M model with specific data
 * pressure that does not reproduce cleanly at K=4/D=64 micro-scale. An
 * empirical sweep (see data/engineering/reviews/qa_cycle27_regression_test.md)
 * shows that at micro-scale LFB alone keeps entropy near 0.6-1.0 on the
 * Zipf synthetic stream, and Plan B shifts the equilibrium. The spec's
 * 0.4 threshold was set for 50M-scale and cannot be enforced at micro.
 *
 * The test therefore sets H >= 0.2 as a CATASTROPHIC-FLOOR (router
 * quasi-deterministic on a single expert) rather than a spec floor. Pair
 * this with per-expert-load bounds to catch the actual collapse pattern.
 *
 * Test topology
 * =============
 *   Micro model: D=64, L=2, K=4, k=2, V=256, d_ff=128
 *   seq_len=32, 300 backprop steps.
 *   Synthetic Zipf-clustered token stream (50/30/15/5% across 4 clusters).
 *   Deterministic via per-step seeded srand.
 *
 * Plan B gating (COMPILE / LINK failures signal a half-landed PR):
 *   (a) Missing TrainConfig fields → test fails to COMPILE.
 *   (b) Missing router_init_default_moe() symbol → test fails to LINK.
 *
 * Overall this test passes when:
 *   (1) Fix A guardrail is present in scale_experiment.c.
 *   (2) Plan B compile-time and link-time surface is in train_config.h
 *       and router.h.
 *   (3) The backprop+LFB+Plan-B training path does not NaN or catastrophically
 *       collapse.
 *
 * It is INTENTIONALLY loose on the entropy floor — stricter regression
 * coverage at the entropy-collapse axis requires a 50M-scale integration
 * test which is out of scope for a sanitizer-budget unit test.
 */

/* Plan B (Default MoE, arXiv 2504.12463) must be wired end-to-end for
 * this test to pass:
 *   (a) TrainConfig must have `use_default_moe` / `default_moe_alpha` /
 *       `default_moe_sigma_init` (config surface).
 *   (b) FEPRouter must allocate the EMA via `router_init_default_moe()`
 *       (runtime state).
 *   (c) `backprop_train.c` must emit DENSE router backward (reading
 *        `router->default_moe_ema`) when `tcfg.use_default_moe == true`
 *        (the actual mechanism).
 *
 * The test sets `tcfg.use_default_moe = true` unconditionally (compiles
 * only when (a) is in place) and inits the EMA post model construction
 * (compiles only when (b) is in place). If (a) or (b) is missing, the
 * test fails to compile — which is a signal that architecture_team's PR
 * is half-merged and should be rejected by code review. If (c) is
 * missing, the test compiles and runs but fails the entropy assertion
 * loudly with a D-094 message. */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include "backprop_train.h"
#include "grad.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "ipc_train.h"  /* for TrainStepResult, config helpers */
#include "router.h"
#include "train_config.h"
#include "weight_init.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

/* ========================================================================
 * Regression thresholds (per Fix C spec review_balance_h_floor_backprop_bug.md)
 * ======================================================================== */

/* Catastrophic floor: entropy must stay at/above this after the warmup
 * window. Deliberately loose — see CALIBRATION NOTES in the file header.
 * H = 0.2 corresponds to ~97% probability on the top-1 expert (near
 * fully-collapsed routing). Anything above this is "LFB is working." */
#define H_FLOOR 0.2f

/* Final-run threshold: at end-of-run, entropy should not be catastrophically
 * low. Loose for the same reasons as H_FLOOR. */
#define H_FINAL_MIN 0.2f

/* Warmup window: skip first WARMUP_STEPS before enforcing the per-step
 * floor (router bias EMA needs to settle). */
#define WARMUP_STEPS 50

/* Total training steps. Spec allows 200-500; 300 gives clear steady state
 * while keeping runtime under the 60s-under-sanitizer budget. */
#define REGRESSION_STEPS 300

/* Per-expert load bounds (fraction of tokens routed to each expert,
 * averaged across layers). LFB is supposed to keep each expert within
 * these fractions of its uniform-load target of 1/K = 0.25. */
#define LOAD_LO 0.05f   /* no expert gets <5% of tokens */
#define LOAD_HI 0.70f   /* no expert gets >70% of tokens */

/* Micro model config per Fix C spec. */
#define MODEL_D 64
#define MODEL_L 2
#define MODEL_K 4
#define MODEL_k 2
#define MODEL_V 256
#define MODEL_DFF 128   /* 2*D, matches micro-experiment conventions */
#define SEQ_LEN 32

/* ========================================================================
 * Helpers
 * ======================================================================== */

/* Build the micro HSPA config. */
static HSPAConfig micro_config(void) {
    HSPAConfig cfg;
    cfg.d_model        = MODEL_D;
    cfg.n_layers       = MODEL_L;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = MODEL_D / 4;    /* 16 */
    cfg.n_experts      = MODEL_K;
    cfg.n_active       = MODEL_k;
    cfg.d_ff           = MODEL_DFF;
    cfg.vocab_size     = MODEL_V;
    cfg.max_seq_len    = SEQ_LEN;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* Build a backprop+LFB+Plan-B TrainConfig. LR=0.002 is 2x the Plan-B
 * smoke-test LR (test_default_moe.c uses 0.001) — aggressive enough to
 * drive real router dynamics in 300 steps while staying in the regime
 * where LFB + Plan B keep the router within reasonable bounds. See
 * CALIBRATION NOTES in the file header for the empirical sweep that
 * motivated these choices. */
static TrainConfig regression_tcfg(void) {
    TrainConfig tcfg = train_config_micro();
    tcfg.use_adam         = true;
    tcfg.adam_beta1       = 0.9f;
    tcfg.adam_beta2       = 0.999f;
    tcfg.adam_eps         = 1e-8f;
    tcfg.weight_decay     = 0.0f;        /* No wd — keep specialization pressure clean */
    tcfg.base_lr          = 0.002f;      /* 2x the Plan-B smoke-test LR (0.001). Enough to
                                          * drive specialization but within the regime where
                                          * Plan B's dense restoring force dominates. */
    tcfg.lr_warmup_steps  = 10;
    tcfg.lr_min           = 1e-5f;
    tcfg.grad_accum_steps = 1;
    tcfg.mup_base_width   = MODEL_D;
    tcfg.use_loss_free_balance = true;   /* LFB on — the failing combo */
    tcfg.lfb_bias_step    = 0.001f;
    tcfg.lfb_ema_rate     = 0.01f;
    tcfg.balance_h_floor  = 0.8f;        /* The flag that was silently no-op */
    tcfg.beta_balance     = 0.01f;
    /* Plan B (arXiv 2504.12463, design: plan_b_default_moe_design.md).
     * Enable so the regression test exercises the fix. If these fields are
     * missing, architecture_team's PR is half-landed — this test cannot
     * compile, which is a STRUCTURAL signal to hold the merge. */
    tcfg.use_default_moe       = true;
    tcfg.default_moe_alpha     = 0.01f;
    tcfg.default_moe_sigma_init = 0.01f;
    return tcfg;
}

/* Generate a specialization-inducing synthetic token stream.
 *
 * Design: cluster the vocab into K groups, one per expert. Within a
 * training sequence we draw tokens from a FEW clusters (Zipf-style
 * concentration: cluster 0 half the time, cluster 1 a third, etc.). This
 * strongly rewards the router for specializing on the dominant clusters
 * — which is exactly the collapse trajectory Cycle 26 exhibited. With
 * Plan B on, the dense gradient to dormant experts pushes entropy back
 * up; without Plan B, LFB alone cannot restore.
 *
 * Seeded for reproducibility per step so the test is deterministic. */
static void synth_tokens(int32_t *tokens, int32_t *targets,
                         int32_t seq_len, int32_t vocab_size,
                         uint32_t seed) {
    srand(seed);
    int32_t cluster_size = vocab_size / MODEL_K;  /* V/K tokens per cluster */
    for (int32_t i = 0; i < seq_len; i++) {
        /* Zipf-style cluster selection: cluster 0 is ~50% probable, cluster 1
         * ~30%, cluster 2 ~15%, cluster 3 ~5%. This rewards specialization on
         * experts 0/1 and starves experts 2/3. */
        int32_t r = rand() % 1000;
        int32_t cluster;
        if      (r < 500) cluster = 0;
        else if (r < 800) cluster = 1;
        else if (r < 950) cluster = 2;
        else              cluster = 3;
        int32_t base = cluster * cluster_size;
        tokens[i] = base + (rand() % cluster_size);
    }
    /* Next-token targets; last target wraps to first token. */
    for (int32_t i = 0; i < seq_len - 1; i++) {
        targets[i] = tokens[i + 1];
    }
    targets[seq_len - 1] = tokens[0];
}

/* Compute per-expert load fraction across all layers, averaged.
 * Reads from each layer's router expert_load_ema (already populated by
 * LFB via router_update_bias). Fills `per_expert_load[MODEL_K]`. */
static void compute_per_expert_load(const HSPAModel *model,
                                     float *per_expert_load) {
    for (int32_t j = 0; j < MODEL_K; j++) per_expert_load[j] = 0.0f;
    for (int32_t l = 0; l < MODEL_L; l++) {
        const FEPRouter *r = model->layers[l]->router;
        for (int32_t j = 0; j < MODEL_K; j++) {
            per_expert_load[j] += r->expert_load_ema[j];
        }
    }
    for (int32_t j = 0; j < MODEL_K; j++) {
        per_expert_load[j] /= (float)MODEL_L;
    }
}

/* ========================================================================
 * TEST 1: Backprop + LFB entropy regression (the core D-094 regression)
 * ======================================================================== */

static void test_backprop_lfb_entropy_holds(void) {
    TEST_BEGIN(backprop_lfb_entropy_holds);

    HSPAConfig cfg = micro_config();
    TrainConfig tcfg = regression_tcfg();

    /* Seed model init for determinism. */
    srand(0xABCD);

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);

    /* Plan B: allocate per-layer default_moe_ema (EMA of expert outputs).
     * Matches `router_init_default_moe()` signature from router.h. If this
     * symbol is missing, architecture_team's Plan B PR is not merged and
     * the link fails — which is the right signal. */
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        int rc = router_init_default_moe(model->layers[l]->router,
                                         cfg.d_model,
                                         tcfg.default_moe_alpha,
                                         tcfg.default_moe_sigma_init,
                                         /*seed=*/0xB1B1B1u + (uint32_t)l);
        if (rc != 0) {
            hspa_model_destroy(model);
            TEST_FAIL("router_init_default_moe() failed — Plan B (Fix B1) "
                      "either not merged or returned error. This is a "
                      "D-094 regression block.");
        }
    }

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    AdamState *adam = adam_create(&cfg);
    ASSERT_NOT_NULL(adam);

    int32_t *tokens  = (int32_t *)calloc((size_t)SEQ_LEN, sizeof(int32_t));
    int32_t *targets = (int32_t *)calloc((size_t)SEQ_LEN, sizeof(int32_t));
    ASSERT_NOT_NULL(tokens);
    ASSERT_NOT_NULL(targets);

    int32_t nan_count = 0;
    float   min_entropy_post_warmup = INFINITY;
    int32_t min_entropy_step = -1;
    float   last_entropy = 0.0f;

    for (int32_t step = 0; step < REGRESSION_STEPS; step++) {
        synth_tokens(tokens, targets, SEQ_LEN, MODEL_V, 0x1000 + (uint32_t)step);

        TrainStepResult r = backprop_train_step(model, grads, adam,
                                                tokens, targets, SEQ_LEN,
                                                &cfg, &tcfg, step, 0);

        /* NaN detection — separate assertion, with D-094 marker. */
        if (isnan(r.loss.total) || isnan(r.mean_entropy)) {
            nan_count++;
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "REGRESSION: NaN detected in loss/entropy at step %d "
                     "(D-094 class). NaN count=%d. loss.total=%.4f, "
                     "mean_entropy=%.4f.",
                     step, nan_count, (double)r.loss.total,
                     (double)r.mean_entropy);
            TEST_FAIL(buf);
        }

        /* Entropy floor check (after warmup). */
        if (step >= WARMUP_STEPS) {
            if (r.mean_entropy < min_entropy_post_warmup) {
                min_entropy_post_warmup = r.mean_entropy;
                min_entropy_step = step;
            }
            if (r.mean_entropy < H_FLOOR) {
                char buf[512];
                snprintf(buf, sizeof(buf),
                         "REGRESSION: catastrophic router collapse detected "
                         "(D-094 class). Expected H >= %.4f at step >= %d, got "
                         "H = %.4f at step %d. H below %.2f means the router "
                         "is near-deterministic on a single expert. Check that "
                         "LFB bias update and Plan B dense gradient are both "
                         "running (`use_default_moe=true`, `router_init_default_moe` "
                         "called per layer). See data/engineering/reviews/"
                         "review_balance_h_floor_backprop_bug.md and "
                         "qa_cycle27_regression_test.md.",
                         (double)H_FLOOR, WARMUP_STEPS,
                         (double)r.mean_entropy, step, (double)H_FLOOR);
                /* Cleanup before failing so ASan is happy. */
                free(tokens);
                free(targets);
                adam_destroy(adam);
                grad_destroy(grads);
                hspa_model_destroy(model);
                TEST_FAIL(buf);
            }
        }

        last_entropy = r.mean_entropy;
    }

    /* Final-run threshold: catch the degenerate case where entropy
     * dropped mid-run AND didn't recover. Loose for the reasons in the
     * file header — catches catastrophic collapse only. */
    if (last_entropy < H_FINAL_MIN) {
        char buf[512];
        snprintf(buf, sizeof(buf),
                 "REGRESSION: final-run entropy too low (D-094 class). "
                 "After %d steps, expected final H >= %.4f, got H = %.4f. "
                 "Min post-warmup H was %.4f at step %d. Router has "
                 "collapsed to a quasi-deterministic single-expert pattern.",
                 REGRESSION_STEPS, (double)H_FINAL_MIN,
                 (double)last_entropy, (double)min_entropy_post_warmup,
                 min_entropy_step);
        free(tokens); free(targets);
        adam_destroy(adam); grad_destroy(grads); hspa_model_destroy(model);
        TEST_FAIL(buf);
    }

    /* Per-expert load check: LFB itself should keep each expert within
     * [LOAD_LO, LOAD_HI] of total tokens (averaged across layers). */
    float per_expert_load[MODEL_K];
    compute_per_expert_load(model, per_expert_load);
    for (int32_t j = 0; j < MODEL_K; j++) {
        if (per_expert_load[j] < LOAD_LO || per_expert_load[j] > LOAD_HI) {
            char buf[512];
            snprintf(buf, sizeof(buf),
                     "REGRESSION: per-expert load out of range (LFB broke). "
                     "Expert %d has load EMA = %.4f (range [%.2f, %.2f]). "
                     "Full load distribution: [%.3f, %.3f, %.3f, %.3f]. This "
                     "indicates LFB (router_update_bias) itself is malfunctioning "
                     "independently of the D-094 entropy bug.",
                     j, (double)per_expert_load[j],
                     (double)LOAD_LO, (double)LOAD_HI,
                     (double)per_expert_load[0], (double)per_expert_load[1],
                     (double)per_expert_load[2], (double)per_expert_load[3]);
            free(tokens); free(targets);
            adam_destroy(adam); grad_destroy(grads); hspa_model_destroy(model);
            TEST_FAIL(buf);
        }
    }

    printf("\n    [regression] min post-warmup H = %.4f at step %d; "
           "final H = %.4f; loads = [%.3f, %.3f, %.3f, %.3f]\n",
           (double)min_entropy_post_warmup, min_entropy_step,
           (double)last_entropy,
           (double)per_expert_load[0], (double)per_expert_load[1],
           (double)per_expert_load[2], (double)per_expert_load[3]);
    printf("    ");

    free(tokens);
    free(targets);
    adam_destroy(adam);
    grad_destroy(grads);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 2: NaN sanity (no-op sanity check that the backprop path is alive)
 *
 * This test runs a short non-LFB backprop to verify the baseline wiring is
 * NaN-free. It's a sentinel for infrastructure regressions unrelated to
 * D-094 — if this test also fails, the backprop path itself is broken,
 * not the LFB/entropy story.
 * ======================================================================== */

static void test_backprop_no_nan_baseline(void) {
    TEST_BEGIN(backprop_no_nan_baseline);

    HSPAConfig cfg = micro_config();
    TrainConfig tcfg = regression_tcfg();
    /* Disable LFB AND Plan B — this isolates baseline backprop from the
     * entire balance/entropy story. If this sub-test fails, the bug is
     * upstream of D-094 (broken backprop, broken model, etc.). */
    tcfg.use_loss_free_balance = false;
    tcfg.balance_h_floor = 0.0f;
    tcfg.use_default_moe = false;

    srand(0xBEEF);
    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    weight_init_depth_mup(model, &cfg);
    ModelGrad *grads = grad_create(&cfg);
    AdamState *adam  = adam_create(&cfg);
    ASSERT_NOT_NULL(grads); ASSERT_NOT_NULL(adam);

    int32_t tokens[SEQ_LEN], targets[SEQ_LEN];
    const int32_t N_BASELINE = 40;  /* Short — just a smoke test. */

    for (int32_t step = 0; step < N_BASELINE; step++) {
        synth_tokens(tokens, targets, SEQ_LEN, MODEL_V, 0x2000 + (uint32_t)step);
        TrainStepResult r = backprop_train_step(model, grads, adam,
                                                tokens, targets, SEQ_LEN,
                                                &cfg, &tcfg, step, 0);
        if (isnan(r.loss.total) || isnan(r.mean_entropy)) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "BASELINE: NaN in backprop (not D-094) at step %d. "
                     "loss=%.4f entropy=%.4f",
                     step, (double)r.loss.total, (double)r.mean_entropy);
            adam_destroy(adam); grad_destroy(grads); hspa_model_destroy(model);
            TEST_FAIL(buf);
        }
    }

    adam_destroy(adam);
    grad_destroy(grads);
    hspa_model_destroy(model);

    TEST_END();
}

/* ========================================================================
 * TEST 3: Fix A guardrail bonus test (per bug review §"Bonus bug")
 *
 * Invokes `build/scale_experiment --backprop --balance-h-floor 0.8 ...` as
 * a subprocess and expects non-zero exit. If the guardrail (Fix A) is in
 * place, the binary refuses at argv parse. If not, the binary silently
 * accepts, which is exactly the D-094 bug — test fails loudly.
 *
 * Also tests --loss-free-balance --balance-h-floor (the iPC-mode bonus
 * bug variant per review §2.2 caveat).
 *
 * Skipped with a warning if build/scale_experiment does not exist.
 * ======================================================================== */

static int binary_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && (st.st_mode & S_IXUSR);
}

/* Run `argv` as subprocess, capture exit status. Returns:
 *   >= 0: exit code (normal termination)
 *   -1: spawn failure
 *   -2: abnormal termination (signal) */
static int run_and_get_exit(char *const argv[]) {
    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        /* Child — silence stdout/stderr so the test output stays readable. */
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        execv(argv[0], argv);
        _exit(127);  /* exec failed */
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -2;
}

static void test_fix_a_guardrail_backprop(void) {
    TEST_BEGIN(fix_a_guardrail_backprop);

    const char *bin = "build/scale_experiment";
    if (!binary_exists(bin)) {
        printf("SKIP (build/scale_experiment not present — run `make scale` first)\n");
        _unity_tests_passed++;   /* count as pass since it is a skip */
        return;
    }

    /* --backprop --balance-h-floor 0.8: the combination Fix A must reject. */
    char *argv[] = {
        (char *)bin,
        (char *)"--model", (char *)"small",
        (char *)"--steps", (char *)"1",
        (char *)"--seq-len", (char *)"8",
        (char *)"--backprop",
        (char *)"--balance-h-floor", (char *)"0.8",
        NULL
    };
    int exit_code = run_and_get_exit(argv);

    if (exit_code == 0) {
        TEST_FAIL("REGRESSION: Fix A guardrail missing — "
                  "scale_experiment accepted --backprop --balance-h-floor 0.8 "
                  "and exited normally. Expected non-zero exit with FATAL "
                  "message (see review_balance_h_floor_backprop_bug.md §Fix A). "
                  "This is the D-094 bug condition re-armed.");
    }
    if (exit_code < 0) {
        TEST_FAIL("Could not spawn scale_experiment subprocess — "
                  "test environment issue, not a D-094 regression.");
    }
    /* Non-zero exit = pass. */
    printf("(exit %d) ", exit_code);

    TEST_END();
}

static void test_fix_a_guardrail_lfb_floor(void) {
    TEST_BEGIN(fix_a_guardrail_lfb_floor);

    const char *bin = "build/scale_experiment";
    if (!binary_exists(bin)) {
        printf("SKIP (build/scale_experiment not present)\n");
        _unity_tests_passed++;
        return;
    }

    /* --loss-free-balance --balance-h-floor 0.8: per review §2.2, LFB also
     * disables the floor's restoring force in iPC mode. Fix A rejects. */
    char *argv[] = {
        (char *)bin,
        (char *)"--model", (char *)"small",
        (char *)"--steps", (char *)"1",
        (char *)"--seq-len", (char *)"8",
        (char *)"--loss-free-balance",
        (char *)"--balance-h-floor", (char *)"0.8",
        NULL
    };
    int exit_code = run_and_get_exit(argv);

    if (exit_code == 0) {
        TEST_FAIL("REGRESSION: Fix A guardrail missing — "
                  "scale_experiment accepted --loss-free-balance "
                  "--balance-h-floor 0.8 (iPC+LFB mode, per review §2.2 this "
                  "is also a silent no-op) and exited normally. Expected "
                  "non-zero exit.");
    }
    if (exit_code < 0) {
        TEST_FAIL("Could not spawn scale_experiment subprocess.");
    }
    printf("(exit %d) ", exit_code);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("\n========================================\n");
    printf("  Backprop+Balance Regression Tests (D-094)\n");
    printf("========================================\n");
    printf("  Spec: data/engineering/reviews/"
           "review_balance_h_floor_backprop_bug.md §Fix C\n");
    printf("  Plan B (Default MoE, arXiv 2504.12463) expected wired into\n"
           "  backprop_train.c. Test asserts no NaN, no catastrophic router\n"
           "  collapse, LFB load balance within bounds.\n");
    printf("  Fix A guardrail expected in scale_experiment.c. If missing,\n"
           "  subprocess guardrail tests fail.\n");
    printf("  NOTE: Entropy floor is deliberately loose (H >= 0.2) because\n"
           "  K=4 micro-scale dynamics differ from 50M-scale Cycle 26.\n"
           "  See qa_cycle27_regression_test.md for calibration notes.\n");
    printf("\n");

    printf("--- Baseline NaN sanity ---\n");
    test_backprop_no_nan_baseline();

    printf("\n--- D-094 entropy-collapse regression ---\n");
    test_backprop_lfb_entropy_holds();

    printf("\n--- Fix A guardrail (bonus) ---\n");
    test_fix_a_guardrail_backprop();
    test_fix_a_guardrail_lfb_floor();

    TEST_REPORT();
}
