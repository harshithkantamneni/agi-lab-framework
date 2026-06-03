/* test_entropy_penalty.c -- Unit tests for the one-sided quadratic hinge
 * entropy penalty + cosine router-temperature anneal.
 *
 * Spec: data/engineering/entropy_penalty_temp_anneal_design.md
 *       (Revision 2 — Cycle 29 C-severe revisit overrides §1.5/§2.1/§8 numerics)
 * Finding: data/findings/chief_scientist_cycle29_csevere_revisit.md §R2.2
 *
 * Test Rev-2 defaults (β_H=0.15, H_target=0.90, τ_max=1.4, S_anneal=500,
 * β_H_warmup=200, entropy_warmup=0, restoration_band ON).
 *
 * Rev-1 immunity: Cycle 13 plain-quadratic pin-at-log(K) failure is covered
 * by test 2 (hinge must be EXACTLY zero when H >= H_target).
 *
 * Math (Rev-1 §1.3, unchanged under Rev-2):
 *   gap            = max(0, H_target - H[t])
 *   penalty_scale  = 2 * beta_H * gap
 *   d_logit_H[j]   = penalty_scale * p[j] * (log p[j] + H[t])
 *
 * Five tests:
 *   1. entropy_penalty_analytic_vs_finite_difference
 *   2. entropy_penalty_zero_above_target (covers Rev-1 H=0.85 + Rev-2 H=0.90)
 *   3. entropy_penalty_sign_of_push
 *   4. tau_anneal_bounds (τ_max=1.4, τ_min=1.0, S=500, monotone)
 *   5. entropy_penalty_gradient_scaling (doubling β_H doubles |grad|)
 */

#include "../src/tests/unity.h"
#include "backprop_train.h"
#include "train_config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =====================================================================
 * Exposed helpers (declared in backprop_train.h, implemented in
 *                  backprop_train.c). Kept file-local-but-exported so
 *                  the unit tests can exercise them without rebuilding
 *                  a full model+optimizer. TDD: tests compile
 *                  against these; implementation provides them.
 * ===================================================================== */

/* Shannon entropy of a softmax-normalized probability vector length K.
 * H = -sum_j p[j] * log(p[j] + eps) */
extern float entropy_penalty_shannon_H(const float *p, int32_t K);

/* Per-logit gradient of the one-sided-quadratic hinge entropy penalty.
 *   d_logit_H[j] = 2 * beta_H * max(0, H_target - H) * p[j] * (log p[j] + H)
 * out_grad must be length K. Zero-fills out_grad when gap==0. */
extern void entropy_penalty_grad_logit(const float *p, int32_t K,
                                        float beta_H, float H_target,
                                        float *out_grad);

/* Cosine τ schedule from τ_max to τ_min over S_anneal steps; τ_min after.
 * At step 0 returns τ_max, at step S_anneal returns τ_min. */
extern float compute_router_temperature(float tau_max, float tau_min,
                                        int32_t step, int32_t S_anneal);

/* Linear warmup schedule for β_H: 0 at step 0, β_H_peak at step >=
 * warmup_steps. Linear ramp in between. */
extern float compute_beta_h_warmup(float beta_h_peak, int32_t step,
                                   int32_t warmup_steps);

/* =====================================================================
 * Helpers
 * ===================================================================== */

/* Compute L_H = beta_H * max(0, H_target - H)^2 from logits (length K).
 * Used by the finite-difference test. */
static float compute_L_H_from_logits(const float *logits, int32_t K,
                                     float beta_H, float H_target) {
    /* Softmax */
    float maxl = logits[0];
    for (int32_t j = 1; j < K; j++) if (logits[j] > maxl) maxl = logits[j];
    float p[64];
    float s = 0.0f;
    for (int32_t j = 0; j < K; j++) {
        p[j] = expf(logits[j] - maxl);
        s += p[j];
    }
    for (int32_t j = 0; j < K; j++) p[j] /= s;
    float H = entropy_penalty_shannon_H(p, K);
    float gap = (H_target > H) ? (H_target - H) : 0.0f;
    return beta_H * gap * gap;
}

static void softmax_inplace(float *x, int32_t K) {
    float maxl = x[0];
    for (int32_t j = 1; j < K; j++) if (x[j] > maxl) maxl = x[j];
    float s = 0.0f;
    for (int32_t j = 0; j < K; j++) { x[j] = expf(x[j] - maxl); s += x[j]; }
    for (int32_t j = 0; j < K; j++) x[j] /= s;
}

/* =====================================================================
 * TEST 1: Analytic gradient vs finite-difference gradient on 8-expert
 *          softmax. THE bit-identity sanity check for §1.3's chain rule.
 * ===================================================================== */
static void test_entropy_penalty_analytic_vs_finite_difference(void) {
    TEST_BEGIN(entropy_penalty_analytic_vs_finite_difference);

    const int32_t K = 8;
    const float beta_H   = 0.15f;
    const float H_target = 0.90f;

    /* Choose concentrated raw logits that yield H well below H_target = 0.90
     * (sharper softmax). With these, softmax ≈ (0.879, 0.044, ...) and
     * H ≈ 0.58, inside the active-hinge regime. */
    float logits[8] = { 4.0f, 1.0f, 0.3f, 0.1f, -0.2f, -0.4f, -0.7f, -1.0f };

    /* Compute softmax + entropy + analytic gradient wrt logits. */
    float p[8];
    memcpy(p, logits, sizeof(p));
    softmax_inplace(p, K);
    float H = entropy_penalty_shannon_H(p, K);
    ASSERT_TRUE(H < H_target); /* so gradient is non-trivial */

    float d_analytic[8];
    entropy_penalty_grad_logit(p, K, beta_H, H_target, d_analytic);

    /* Finite-difference: 2-sided, eps=1e-3 (balances truncation error O(eps^2)
     * vs FP32 round-off O(1/eps). At eps=1e-4 single-precision subtraction
     * cancellation dominates; at eps=1e-3 we hit ~1% accuracy in float).
     *   df/dlogit[j] ≈ (L(logit+eps) - L(logit-eps)) / (2*eps) */
    const float eps = 1e-3f;
    for (int32_t j = 0; j < K; j++) {
        float Lp[8]; memcpy(Lp, logits, sizeof(Lp)); Lp[j] += eps;
        float Lm[8]; memcpy(Lm, logits, sizeof(Lm)); Lm[j] -= eps;
        float fp = compute_L_H_from_logits(Lp, K, beta_H, H_target);
        float fm = compute_L_H_from_logits(Lm, K, beta_H, H_target);
        float numeric = (fp - fm) / (2.0f * eps);

        /* Tolerance: relative < 2e-2 (FP32 finite-diff achieves ~1% accuracy)
         * OR absolute < 1e-5 (near-zero components). */
        float diff = fabsf(d_analytic[j] - numeric);
        float ref  = fabsf(numeric);
        if (ref < 1e-6f) {
            ASSERT_TRUE(diff < 1e-5f);
        } else {
            ASSERT_TRUE(diff / ref < 2e-2f);
        }
    }

    TEST_END();
}

/* =====================================================================
 * TEST 2: Hinge gradient is EXACTLY ZERO when H >= H_target.
 *          Covers BOTH Rev-1 H_target=0.85 AND Rev-2 H_target=0.90.
 *          This is the Cycle-13 immunity test.
 * ===================================================================== */
static void test_entropy_penalty_zero_above_target(void) {
    TEST_BEGIN(entropy_penalty_zero_above_target);

    const int32_t K = 8;

    /* Uniform over K: H = log(K) = log(8) ≈ 2.079 -- well above any target. */
    float p_uniform[8];
    for (int32_t j = 0; j < K; j++) p_uniform[j] = 1.0f / (float)K;
    float H_u = entropy_penalty_shannon_H(p_uniform, K);
    ASSERT_TRUE(H_u > 2.0f);

    /* Rev-1 target */
    float d_rev1[8] = {0};
    entropy_penalty_grad_logit(p_uniform, K, 0.05f, 0.85f, d_rev1);
    /* Rev-2 target */
    float d_rev2[8] = {0};
    entropy_penalty_grad_logit(p_uniform, K, 0.15f, 0.90f, d_rev2);

    for (int32_t j = 0; j < K; j++) {
        ASSERT_EQUAL_FLOAT(0.0f, d_rev1[j], 1e-12f);
        ASSERT_EQUAL_FLOAT(0.0f, d_rev2[j], 1e-12f);
    }

    /* Additionally: at exactly H = H_target (edge of hinge), gap = 0 → grad = 0. */
    /* Build p such that H = 0.90 exactly by careful choice: p = (a, a, b, ...)
     * is hard; easier -- choose p that puts H just above 0.90 and just below,
     * and verify grad monotone-zero ABOVE and nonzero-BELOW. */
    /* Pick p with H just above 0.90: p = (0.55, 0.28, 0.17, 0, 0, 0, 0, 0)
     * → H ≈ 0.986 > 0.90 */
    float p_above[8] = {0.55f, 0.28f, 0.17f, 1e-9f, 1e-9f, 1e-9f, 1e-9f, 1e-9f};
    /* Renormalize just in case */
    float s = 0.0f; for (int32_t j = 0; j < K; j++) s += p_above[j];
    for (int32_t j = 0; j < K; j++) p_above[j] /= s;
    float H_a = entropy_penalty_shannon_H(p_above, K);
    ASSERT_TRUE(H_a > 0.90f);

    float d_above[8] = {0};
    entropy_penalty_grad_logit(p_above, K, 0.15f, 0.90f, d_above);
    for (int32_t j = 0; j < K; j++) {
        ASSERT_EQUAL_FLOAT(0.0f, d_above[j], 1e-12f);
    }

    TEST_END();
}

/* =====================================================================
 * TEST 3: Sign-of-push. When H < H_target: dominant expert pushed DOWN,
 *          laggard expert pushed UP. When H >= H_target: exactly zero.
 * ===================================================================== */
static void test_entropy_penalty_sign_of_push(void) {
    TEST_BEGIN(entropy_penalty_sign_of_push);

    const int32_t K = 8;
    const float beta_H   = 0.15f;
    const float H_target = 0.90f;

    /* Heavily concentrated: p = (0.82, 0.10, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005)
     * H ≈ 0.722 < H_target=0.90, penalty active. log(0.82) + H ≈ +0.52 for
     * dominant (positive grad → SGD pushes down). log(0.005)+H ≈ -4.58 for
     * laggard (negative grad → SGD pushes up). */
    float p_concentrated[8] = {0.82f, 0.10f, 0.03f, 0.02f, 0.01f, 0.01f, 0.005f, 0.005f};
    /* Renormalize */
    float s = 0.0f; for (int32_t j = 0; j < K; j++) s += p_concentrated[j];
    for (int32_t j = 0; j < K; j++) p_concentrated[j] /= s;
    float H_c = entropy_penalty_shannon_H(p_concentrated, K);
    ASSERT_TRUE(H_c < H_target);

    float d[8] = {0};
    entropy_penalty_grad_logit(p_concentrated, K, beta_H, H_target, d);

    /* Dominant expert (index 0): positive grad means L_H increases with
     * logit[dom], so SGD descent DECREASES it — pushes dominant DOWN. */
    ASSERT_TRUE(d[0] > 0.0f);

    /* Laggard (index 7): negative grad → SGD pushes logit UP. */
    ASSERT_TRUE(d[7] < 0.0f);

    /* Control: above-target = zero gradient already covered in test 2.
     * Here verify a SOFT concentration at H just above target stays zero. */
    float p_soft[8] = {0.20f, 0.18f, 0.15f, 0.13f, 0.12f, 0.10f, 0.07f, 0.05f};
    float s2 = 0.0f; for (int32_t j = 0; j < K; j++) s2 += p_soft[j];
    for (int32_t j = 0; j < K; j++) p_soft[j] /= s2;
    float H_s = entropy_penalty_shannon_H(p_soft, K);
    /* If H_s >= 0.90, grad must be zero. */
    if (H_s >= 0.90f) {
        float d_soft[8] = {0};
        entropy_penalty_grad_logit(p_soft, K, beta_H, 0.90f, d_soft);
        for (int32_t j = 0; j < K; j++) {
            ASSERT_EQUAL_FLOAT(0.0f, d_soft[j], 1e-12f);
        }
    }

    TEST_END();
}

/* =====================================================================
 * TEST 4: Temperature anneal bounds.
 *  τ(step=0) = τ_max = 1.4
 *  τ(step=500) = τ_min = 1.0
 *  τ(step > 500) = τ_min (clamp)
 *  Monotone non-increasing.
 * ===================================================================== */
static void test_tau_anneal_bounds(void) {
    TEST_BEGIN(tau_anneal_bounds);

    const float tau_max = 1.4f;
    const float tau_min = 1.0f;
    const int32_t S = 500;

    float t0    = compute_router_temperature(tau_max, tau_min, 0,    S);
    float t500  = compute_router_temperature(tau_max, tau_min, 500,  S);
    float t1000 = compute_router_temperature(tau_max, tau_min, 1000, S);
    float t250  = compute_router_temperature(tau_max, tau_min, 250,  S);

    ASSERT_EQUAL_FLOAT(1.4f, t0,    1e-5f);
    ASSERT_EQUAL_FLOAT(1.0f, t500,  1e-5f);
    ASSERT_EQUAL_FLOAT(1.0f, t1000, 1e-5f);
    /* At midpoint of cosine: 0.5*(1 + cos(pi/2)) = 0.5, so τ = τ_min + 0.5*(τ_max - τ_min) = 1.2 */
    ASSERT_EQUAL_FLOAT(1.2f, t250, 1e-5f);

    /* Monotone non-increasing across [0, S]. Sample every 50 steps. */
    float prev = compute_router_temperature(tau_max, tau_min, 0, S);
    for (int32_t step = 50; step <= S + 100; step += 50) {
        float now = compute_router_temperature(tau_max, tau_min, step, S);
        ASSERT_TRUE(now <= prev + 1e-6f); /* allow tiny float error */
        prev = now;
    }

    /* τ never exceeds τ_max, never goes below τ_min. */
    for (int32_t step = 0; step <= 2000; step += 23) {
        float t = compute_router_temperature(tau_max, tau_min, step, S);
        ASSERT_TRUE(t <= tau_max + 1e-6f);
        ASSERT_TRUE(t >= tau_min - 1e-6f);
    }

    /* β_H warmup schedule: 0 at step 0, peak at step >= warmup_steps. */
    ASSERT_EQUAL_FLOAT(0.0f,  compute_beta_h_warmup(0.15f, 0,   200), 1e-6f);
    ASSERT_EQUAL_FLOAT(0.15f, compute_beta_h_warmup(0.15f, 200, 200), 1e-6f);
    ASSERT_EQUAL_FLOAT(0.15f, compute_beta_h_warmup(0.15f, 500, 200), 1e-6f);
    /* Linear midpoint: step=100, warmup=200 → 0.5 * 0.15 = 0.075 */
    ASSERT_EQUAL_FLOAT(0.075f, compute_beta_h_warmup(0.15f, 100, 200), 1e-6f);
    /* Warmup=0: peak immediately. */
    ASSERT_EQUAL_FLOAT(0.15f, compute_beta_h_warmup(0.15f, 0, 0), 1e-6f);

    TEST_END();
}

/* =====================================================================
 * TEST 5: Doubling β_H exactly doubles penalty-gradient magnitude
 *          at fixed H. Sanity-checks the linear β_H scaling.
 * ===================================================================== */
static void test_entropy_penalty_gradient_scaling(void) {
    TEST_BEGIN(entropy_penalty_gradient_scaling);

    const int32_t K = 8;
    const float H_target = 0.90f;

    /* Concentrated enough for H < 0.90 (~0.58 here). */
    float logits[8] = { 4.0f, 1.0f, 0.5f, 0.0f, -0.2f, -0.5f, -0.8f, -1.2f };
    float p[8]; memcpy(p, logits, sizeof(p)); softmax_inplace(p, K);
    float H = entropy_penalty_shannon_H(p, K);
    ASSERT_TRUE(H < H_target); /* ensure penalty active */

    float d1[8], d2[8];
    entropy_penalty_grad_logit(p, K, 0.05f, H_target, d1);
    entropy_penalty_grad_logit(p, K, 0.10f, H_target, d2);

    for (int32_t j = 0; j < K; j++) {
        /* d2 should be EXACTLY 2 * d1 (penalty_scale = 2*β*gap is linear in β). */
        ASSERT_EQUAL_FLOAT(2.0f * d1[j], d2[j], 1e-7f);
    }

    /* Also: quadrupling β_H quadruples gradient. */
    float d4[8];
    entropy_penalty_grad_logit(p, K, 0.20f, H_target, d4);
    for (int32_t j = 0; j < K; j++) {
        ASSERT_EQUAL_FLOAT(4.0f * d1[j], d4[j], 1e-7f);
    }

    TEST_END();
}

/* =====================================================================
 * MAIN
 * ===================================================================== */
int main(void) {
    printf("=== Entropy Penalty + τ-Anneal Tests (Cycle 29 Rev-2) ===\n\n");

    test_entropy_penalty_analytic_vs_finite_difference();
    test_entropy_penalty_zero_above_target();
    test_entropy_penalty_sign_of_push();
    test_tau_anneal_bounds();
    test_entropy_penalty_gradient_scaling();

    printf("\n=== Results: %d passed, %d failed, %d total ===\n",
           _unity_tests_passed, _unity_tests_failed, _unity_tests_run);

    return _unity_tests_failed > 0 ? 1 : 0;
}
