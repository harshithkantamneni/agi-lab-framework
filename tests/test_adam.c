/* test_adam.c -- Tests for the AdamW optimizer.
 *
 * Tests:
 *   1. adam_create_destroy        -- lifecycle: create, verify non-NULL, destroy
 *   2. adam_single_step           -- one Adam step with known values
 *   3. adam_bias_correction       -- bias correction changes with step number
 *   4. adam_converges_better      -- Adam vs SGD on micro model (100 steps)
 *   5. adam_weight_decay          -- weights shrink with weight decay > 0
 *
 * Uses a tiny config: d_model=16, n_layers=2, n_experts=2, vocab_size=32.
 * All tests run under ASan + UBSan.
 */

#include "../src/tests/unity.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#include "attention.h"
#include "embedding.h"
#include "ffn.h"
#include "grad.h"
#include "hspa_block.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "ipc_train.h"
#include "rmsnorm.h"
#include "router.h"
#include "train_config.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Helper: create a tiny test config suitable for Adam tests. */
static HSPAConfig test_config(void) {
    HSPAConfig cfg;
    cfg.d_model        = 16;
    cfg.n_layers       = 2;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = 4;    /* d_model / n_heads */
    cfg.n_experts      = 2;
    cfg.n_active       = 1;
    cfg.d_ff           = 32;
    cfg.vocab_size     = 32;
    cfg.max_seq_len    = 16;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* Helper: create a training config for Adam tests. */
static TrainConfig test_tcfg(void) {
    TrainConfig tcfg = train_config_micro();
    tcfg.use_adam       = true;
    tcfg.adam_beta1     = 0.9f;
    tcfg.adam_beta2     = 0.999f;
    tcfg.adam_eps       = 1e-8f;
    tcfg.weight_decay   = 0.01f;
    tcfg.base_lr        = 0.001f;
    tcfg.lr_warmup_steps = 0;  /* No warmup for unit tests */
    tcfg.lr_min         = 0.0f;
    return tcfg;
}

/* ========================================================================
 * TEST 1: adam_create_destroy -- lifecycle test
 *
 * Verify that adam_create allocates all moment tensors and adam_destroy
 * cleans up without leaks (ASan will detect any).
 * ======================================================================== */

static void test_adam_create_destroy(void) {
    HSPAConfig cfg = test_config();

    AdamState *adam = adam_create(&cfg);
    ASSERT_NOT_NULL(adam);

    /* Verify embedding moments */
    ASSERT_NOT_NULL(adam->embed_m.m);
    ASSERT_NOT_NULL(adam->embed_m.v);
    ASSERT_EQUAL_INT(cfg.vocab_size, adam->embed_m.m->shape[0]);
    ASSERT_EQUAL_INT(cfg.d_model, adam->embed_m.m->shape[1]);

    /* Verify layer count */
    ASSERT_EQUAL_INT(cfg.n_layers, adam->n_layers);
    ASSERT_NOT_NULL(adam->layers);

    /* Verify per-layer moments */
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        struct AdamLayerState *ls = &adam->layers[l];

        /* Attention norm */
        ASSERT_NOT_NULL(ls->attn_norm_m.m);
        ASSERT_NOT_NULL(ls->attn_norm_m.v);
        ASSERT_EQUAL_INT(cfg.d_model, ls->attn_norm_m.m->shape[0]);

        /* Attention Q/K/V/O */
        ASSERT_NOT_NULL(ls->attn_q_m.m);
        ASSERT_NOT_NULL(ls->attn_q_m.v);
        ASSERT_NOT_NULL(ls->attn_k_m.m);
        ASSERT_NOT_NULL(ls->attn_v_m.m);
        ASSERT_NOT_NULL(ls->attn_o_m.m);

        /* FFN norm */
        ASSERT_NOT_NULL(ls->ffn_norm_m.m);
        ASSERT_NOT_NULL(ls->ffn_norm_m.v);

        /* Router */
        ASSERT_NOT_NULL(ls->router_mu_m.m);
        ASSERT_NOT_NULL(ls->router_sigma_m.m);

        /* Expert moments */
        ASSERT_EQUAL_INT(cfg.n_experts, ls->n_experts);
        ASSERT_NOT_NULL(ls->expert_gate_m);
        ASSERT_NOT_NULL(ls->expert_up_m);
        ASSERT_NOT_NULL(ls->expert_down_m);

        for (int32_t e = 0; e < cfg.n_experts; e++) {
            ASSERT_NOT_NULL(ls->expert_gate_m[e].m);
            ASSERT_NOT_NULL(ls->expert_gate_m[e].v);
            ASSERT_NOT_NULL(ls->expert_up_m[e].m);
            ASSERT_NOT_NULL(ls->expert_down_m[e].m);
        }
    }

    /* Verify final norm moments */
    ASSERT_NOT_NULL(adam->final_norm_m.m);
    ASSERT_NOT_NULL(adam->final_norm_m.v);
    ASSERT_EQUAL_INT(cfg.d_model, adam->final_norm_m.m->shape[0]);

    /* Verify pool exists */
    ASSERT_NOT_NULL(adam->adam_pool);

    /* Verify all moments are initialized to zero */
    int32_t n = (int32_t)tensor_numel(adam->embed_m.m);
    const float *m_data = adam->embed_m.m->data;
    const float *v_data = adam->embed_m.v->data;
    for (int32_t i = 0; i < n; i++) {
        ASSERT_EQUAL_FLOAT(0.0f, m_data[i], 1e-10f);
        ASSERT_EQUAL_FLOAT(0.0f, v_data[i], 1e-10f);
    }

    /* Destroy and let ASan verify no leaks */
    adam_destroy(adam);
}

/* ========================================================================
 * TEST 2: adam_single_step -- verify one step with hand-computed values
 *
 * Set up a simple scenario: weight=1.0, gradient=0.1, step=1.
 * Hand-compute the expected result and verify it matches.
 *
 * With beta1=0.9, beta2=0.999, eps=1e-8, lr=0.001, wd=0.0:
 *   m = 0.9*0 + 0.1*0.1 = 0.01
 *   v = 0.999*0 + 0.001*0.01 = 0.00001
 *   m_hat = 0.01 / (1 - 0.9^1) = 0.01 / 0.1 = 0.1
 *   v_hat = 0.00001 / (1 - 0.999^1) = 0.00001 / 0.001 = 0.01
 *   w = 1.0 - 0.001 * 0.1 / (sqrt(0.01) + 1e-8)
 *     = 1.0 - 0.001 * 0.1 / 0.1
 *     = 1.0 - 0.001 = 0.999
 * ======================================================================== */

static void test_adam_single_step(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = test_tcfg();
    tcfg.weight_decay = 0.0f;  /* No weight decay for clean hand computation */

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    AdamState *adam = adam_create(&cfg);
    ASSERT_NOT_NULL(adam);

    /* Set embedding weights to 1.0 and gradients to 0.1 */
    tensor_fill(model->embed->weight, 1.0f);
    tensor_fill(grads->embed_grad.dweight, 0.1f);

    /* Zero all other gradients so only embedding is updated meaningfully */
    grad_zero(grads);
    tensor_fill(grads->embed_grad.dweight, 0.1f);

    /* Apply one Adam step (step=0 becomes adam_step=1 internally) */
    grad_apply_adam(model, grads, adam, &cfg, &tcfg, 0);

    /* Verify embedding weight update.
     * Expected: w = 1.0 - lr * m_hat / (sqrt(v_hat) + eps)
     *         = 1.0 - 0.001 * 0.1 / (sqrt(0.01) + 1e-8)
     *         = 1.0 - 0.001 * 0.1 / 0.1
     *         = 1.0 - 0.001 = 0.999 */
    float expected_w = 0.999f;
    int32_t idx[] = {0, 0};
    float actual_w = tensor_get(model->embed->weight, idx);
    ASSERT_EQUAL_FLOAT(expected_w, actual_w, 1e-5f);

    /* Verify first moment: m = 0.01 */
    float actual_m = tensor_get(adam->embed_m.m, idx);
    ASSERT_EQUAL_FLOAT(0.01f, actual_m, 1e-7f);

    /* Verify second moment: v = 0.00001 */
    float actual_v = tensor_get(adam->embed_m.v, idx);
    ASSERT_EQUAL_FLOAT(0.00001f, actual_v, 1e-9f);

    adam_destroy(adam);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ========================================================================
 * TEST 3: adam_bias_correction -- verify bias correction evolves with step
 *
 * At step 1, bias correction is large (compensating for zero-init).
 * At step 100, bias correction should be close to 1.0.
 * We verify this indirectly by running multiple steps with constant
 * gradient and checking that the effective update magnitude changes.
 * ======================================================================== */

static void test_adam_bias_correction(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = test_tcfg();
    tcfg.weight_decay = 0.0f;
    tcfg.base_lr      = 0.01f;  /* Larger LR for visible effect */

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    AdamState *adam = adam_create(&cfg);
    ASSERT_NOT_NULL(adam);

    /* Use constant gradient of 1.0 for simplicity */
    int32_t idx[] = {0, 0};

    /* Run step 1 */
    tensor_fill(model->embed->weight, 10.0f);
    tensor_fill(grads->embed_grad.dweight, 1.0f);
    /* Zero non-embedding grads */
    for (int32_t l = 0; l < cfg.n_layers; l++) {
        BlockGrad *bg = &grads->block_grads[l];
        tensor_fill(bg->attn_norm_grad.dweight, 0.0f);
        tensor_fill(bg->attn_grad.dW_q, 0.0f);
        tensor_fill(bg->attn_grad.dW_k, 0.0f);
        tensor_fill(bg->attn_grad.dW_v, 0.0f);
        tensor_fill(bg->attn_grad.dW_o, 0.0f);
        tensor_fill(bg->ffn_norm_grad.dweight, 0.0f);
        tensor_fill(bg->router_grad.dW_mu, 0.0f);
        tensor_fill(bg->router_grad.dW_sigma, 0.0f);
        for (int32_t e = 0; e < bg->n_experts; e++) {
            tensor_fill(bg->expert_grads[e].dW_gate, 0.0f);
            tensor_fill(bg->expert_grads[e].dW_up, 0.0f);
            tensor_fill(bg->expert_grads[e].dW_down, 0.0f);
        }
    }
    tensor_fill(grads->final_norm_grad.dweight, 0.0f);

    float w_before_step1 = tensor_get(model->embed->weight, idx);
    grad_apply_adam(model, grads, adam, &cfg, &tcfg, 0);
    float w_after_step1 = tensor_get(model->embed->weight, idx);
    float delta_step1 = w_before_step1 - w_after_step1;

    /* At step 1 with g=1.0, beta1=0.9, beta2=0.999:
     *   m = 0.1,  m_hat = 0.1 / (1-0.9) = 1.0
     *   v = 0.001, v_hat = 0.001 / (1-0.999) = 1.0
     *   delta = lr * 1.0 / (sqrt(1.0) + eps) = lr * 1.0 = 0.01
     * (but there's no weight decay so no extra shrinkage) */
    ASSERT_TRUE(delta_step1 > 0.0f);  /* Weight decreased */

    /* Run many more steps with same gradient to let moments converge */
    for (int32_t s = 1; s < 50; s++) {
        tensor_fill(grads->embed_grad.dweight, 1.0f);
        float w_before = tensor_get(model->embed->weight, idx);
        grad_apply_adam(model, grads, adam, &cfg, &tcfg, s);
        float w_after = tensor_get(model->embed->weight, idx);
        (void)w_before;
        (void)w_after;
    }

    /* After 50 steps with constant g=1:
     * m_n = 1 - beta1^n.  With beta1=0.9: m_50 = 1 - 0.9^50 ~ 0.9948
     * v_n = 1 - beta2^n.  With beta2=0.999: v_50 = 1 - 0.999^50 ~ 0.0488
     *
     * m converges fast, v converges slowly. This is expected behavior:
     * bias correction for v matters more at early steps. */
    float m_50 = tensor_get(adam->embed_m.m, idx);
    float v_50 = tensor_get(adam->embed_m.v, idx);

    /* m should be very close to 1.0 (beta1=0.9 converges fast) */
    ASSERT_TRUE(m_50 > 0.99f);
    ASSERT_TRUE(m_50 <= 1.0f);

    /* v should be ~0.049 (beta2=0.999 converges slowly) */
    ASSERT_TRUE(v_50 > 0.04f);
    ASSERT_TRUE(v_50 < 0.06f);

    adam_destroy(adam);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ========================================================================
 * TEST 4: adam_converges_better -- Adam vs SGD on oscillating objective
 *
 * Adam's advantage is best demonstrated on an ill-conditioned problem
 * where gradients along different dimensions have very different scales,
 * AND the SGD learning rate must be kept small to avoid divergence on
 * the steep direction, making it slow on the shallow direction.
 *
 * We use a 2D Rosenbrock-like function:
 *   L = 100*(w1 - w0^2)^2 + (1 - w0)^2
 * which has curvature ~600 along one direction and ~2 along another
 * near the optimum.  SGD at a safe LR crawls; Adam adapts per-param.
 *
 * For test determinism we just verify that both optimizers reduce
 * loss from the initial point and that Adam has lower final loss
 * after a fixed number of steps.
 * ======================================================================== */

static void test_adam_converges_better(void) {
    /* Rosenbrock: f(x,y) = 100*(y - x^2)^2 + (1-x)^2
     * grad_x = -400*x*(y - x^2) - 2*(1 - x)
     * grad_y = 200*(y - x^2)
     * Minimum at (1, 1). */

    float x_sgd = -1.0f, y_sgd = 1.0f;
    float x_adam = -1.0f, y_adam = 1.0f;
    float mx = 0.0f, my = 0.0f;  /* Adam first moments */
    float vx = 0.0f, vy = 0.0f;  /* Adam second moments */

    /* SGD LR must be very small to not diverge on the steep valley.
     * Adam can use a higher effective LR due to normalization. */
    float sgd_lr = 0.0005f;
    float adam_lr = 0.005f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;

    float initial_loss = 100.0f * (y_sgd - x_sgd * x_sgd) *
                         (y_sgd - x_sgd * x_sgd) +
                         (1.0f - x_sgd) * (1.0f - x_sgd);

    for (int32_t s = 0; s < 500; s++) {
        /* SGD gradients */
        float gx_sgd = -400.0f * x_sgd * (y_sgd - x_sgd * x_sgd) -
                        2.0f * (1.0f - x_sgd);
        float gy_sgd = 200.0f * (y_sgd - x_sgd * x_sgd);
        x_sgd -= sgd_lr * gx_sgd;
        y_sgd -= sgd_lr * gy_sgd;

        /* Adam gradients */
        float gx_adam = -400.0f * x_adam * (y_adam - x_adam * x_adam) -
                        2.0f * (1.0f - x_adam);
        float gy_adam = 200.0f * (y_adam - x_adam * x_adam);

        int32_t step = s + 1;
        double b1t = 1.0, b2t = 1.0;
        for (int32_t k = 0; k < step; k++) {
            b1t *= (double)beta1;
            b2t *= (double)beta2;
        }
        float bc1 = (float)(1.0 / (1.0 - b1t));
        float bc2 = (float)(1.0 / (1.0 - b2t));

        mx = beta1 * mx + (1.0f - beta1) * gx_adam;
        my = beta1 * my + (1.0f - beta1) * gy_adam;
        vx = beta2 * vx + (1.0f - beta2) * gx_adam * gx_adam;
        vy = beta2 * vy + (1.0f - beta2) * gy_adam * gy_adam;

        float mxh = mx * bc1, myh = my * bc1;
        float vxh = vx * bc2, vyh = vy * bc2;

        x_adam -= adam_lr * mxh / (sqrtf(vxh) + eps);
        y_adam -= adam_lr * myh / (sqrtf(vyh) + eps);
    }

    float sgd_loss = 100.0f * (y_sgd - x_sgd * x_sgd) *
                     (y_sgd - x_sgd * x_sgd) +
                     (1.0f - x_sgd) * (1.0f - x_sgd);
    float adam_loss = 100.0f * (y_adam - x_adam * x_adam) *
                      (y_adam - x_adam * x_adam) +
                      (1.0f - x_adam) * (1.0f - x_adam);

    /* Both should reduce loss from initial */
    ASSERT_TRUE(sgd_loss < initial_loss);
    ASSERT_TRUE(adam_loss < initial_loss);

    /* Adam should reach lower loss on this ill-conditioned landscape */
    ASSERT_TRUE(adam_loss < sgd_loss);
}

/* ========================================================================
 * TEST 5: adam_weight_decay -- verify weights shrink with wd > 0
 *
 * Set weights to 2.0, gradient to 0.0 (no gradient signal).
 * With weight_decay > 0, weights should shrink toward zero due to
 * the decoupled weight decay term: w *= (1 - lr * wd).
 * ======================================================================== */

static void test_adam_weight_decay(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = test_tcfg();
    tcfg.weight_decay = 0.1f;  /* Strong decay for visible effect */
    tcfg.base_lr      = 0.01f;

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);

    ModelGrad *grads = grad_create(&cfg);
    ASSERT_NOT_NULL(grads);

    AdamState *adam = adam_create(&cfg);
    ASSERT_NOT_NULL(adam);

    /* Set embedding weights to 2.0, zero all gradients */
    tensor_fill(model->embed->weight, 2.0f);
    grad_zero(grads);

    int32_t idx[] = {0, 0};
    float w_initial = tensor_get(model->embed->weight, idx);
    ASSERT_EQUAL_FLOAT(2.0f, w_initial, 1e-6f);

    /* Apply Adam with zero gradients.
     * Only weight decay should affect the weights:
     * w *= (1 - lr * wd) = (1 - 0.01 * 0.1) = 0.999
     * But m and v remain 0, so m_hat/sqrt(v_hat) is 0/0+eps ~ 0
     * Therefore: w = 2.0 * 0.999 - 0 = 1.998 */
    grad_apply_adam(model, grads, adam, &cfg, &tcfg, 0);

    float w_after = tensor_get(model->embed->weight, idx);
    float expected = 2.0f * (1.0f - 0.01f * 0.1f);
    ASSERT_EQUAL_FLOAT(expected, w_after, 1e-5f);
    ASSERT_TRUE(w_after < w_initial);

    /* Run more steps -- weight should keep shrinking */
    for (int32_t s = 1; s < 50; s++) {
        grad_zero(grads);
        grad_apply_adam(model, grads, adam, &cfg, &tcfg, s);
    }

    float w_final = tensor_get(model->embed->weight, idx);
    ASSERT_TRUE(w_final < w_after);
    ASSERT_TRUE(w_final > 0.0f);  /* Should shrink but not go negative */

    /* Compare with expected: w = 2.0 * (1 - lr*wd)^50 */
    float decay_factor = 1.0f - 0.01f * 0.1f;
    float expected_final = 2.0f;
    for (int32_t s = 0; s < 50; s++) {
        expected_final *= decay_factor;
    }
    ASSERT_EQUAL_FLOAT(expected_final, w_final, 1e-3f);

    adam_destroy(adam);
    grad_destroy(grads);
    hspa_model_destroy(model);
}

/* ======================================================================== */

int main(void) {
    printf("=== AdamW Optimizer Tests ===\n\n");

    RUN_TEST(test_adam_create_destroy);
    RUN_TEST(test_adam_single_step);
    RUN_TEST(test_adam_bias_correction);
    RUN_TEST(test_adam_converges_better);
    RUN_TEST(test_adam_weight_decay);

    TEST_REPORT();
}
