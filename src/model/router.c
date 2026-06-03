/* router.c -- FEP (Free Energy Principle) based expert router.
 *
 * Implements active inference routing from H-N1 (Neuroscience team):
 *   1. Compute routing logits via W_mu^T @ x
 *   2. Compute routing uncertainties via softplus(W_sigma^T @ x)
 *   3. Compute expected free energy G per expert
 *   4. During training: add Gaussian noise scaled by sigma (exploration)
 *   5. Select top-k experts with LOWEST G
 *   6. Compute softmax weights over -G[selected]
 *   7. Compute routing entropy H(g) for VFE loss
 *
 * Source: H-N1 (Neuroscience team FEP routing design)
 */

#include "router.h"
#include "ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- Helpers ---- */

static float softplus(float x) {
    /* softplus(x) = log(1 + exp(x)), with overflow protection */
    if (x > 20.0f) {
        return x; /* exp(x) dominates, log(1+exp(x)) ~ x */
    }
    return logf(1.0f + expf(x));
}

/* Simple Box-Muller transform for N(0,1) samples.
 * Uses a static state -- adequate for single-threaded training. */
static float randn(void) {
    float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

/* ---- API ---- */

FEPRouter *router_create(MemoryPool *pool, const HSPAConfig *cfg) {
    if (!pool || !cfg) {
        return NULL;
    }

    FEPRouter *router = (FEPRouter *)calloc(1, sizeof(FEPRouter));
    if (!router) {
        return NULL;
    }

    int32_t shape[] = {cfg->d_model, cfg->n_experts};

    /* W_mu: mean routing logits [d_model, n_experts] */
    router->W_mu = tensor_create(pool, shape, 2, DTYPE_FP32);
    if (!router->W_mu) {
        free(router);
        return NULL;
    }
    /* Zero-initialize. Real init (e.g. Xavier) comes from training setup. */
    tensor_fill(router->W_mu, 0.0f);

    /* W_sigma: uncertainty (log-sigma) [d_model, n_experts] */
    router->W_sigma = tensor_create(pool, shape, 2, DTYPE_FP32);
    if (!router->W_sigma) {
        tensor_destroy(router->W_mu);
        free(router);
        return NULL;
    }
    /* Initialize to 0.0 so softplus(0) = log(2) ~ 0.693 initial uncertainty. */
    tensor_fill(router->W_sigma, 0.0f);

    router->sigma_prior = 1.0f;

    /* Auxiliary-loss-free load balancing (DeepSeek-style) */
    router->n_experts = cfg->n_experts;

    router->expert_bias = (float *)calloc((size_t)cfg->n_experts, sizeof(float));
    if (!router->expert_bias) {
        tensor_destroy(router->W_sigma);
        tensor_destroy(router->W_mu);
        free(router);
        return NULL;
    }
    /* expert_bias starts at zero -- no initial bias */

    router->expert_load_ema = (float *)malloc((size_t)cfg->n_experts * sizeof(float));
    if (!router->expert_load_ema) {
        free(router->expert_bias);
        tensor_destroy(router->W_sigma);
        tensor_destroy(router->W_mu);
        free(router);
        return NULL;
    }
    /* Initialize EMA to uniform load assumption: 1/n_experts */
    float uniform_load = 1.0f / (float)cfg->n_experts;
    for (int32_t j = 0; j < cfg->n_experts; j++) {
        router->expert_load_ema[j] = uniform_load;
    }

    router->bias_update_rate = 0.01f;
    router->bias_step_size   = 0.001f;

    /* Plan B default: Default MoE state OFF. Owner (training setup) calls
     * router_init_default_moe() to allocate + initialize the EMA tensor. */
    router->default_moe_ema   = NULL;
    router->default_moe_dim   = 0;
    router->default_moe_alpha = 0.0f;

    /* Cycle 29 Rev-2: default τ=1.0 (no-op). Training loop overrides
     * per-step when --temp-anneal is set (see scale_experiment.c). */
    router->router_temperature = 1.0f;

    return router;
}

RoutingDecision router_forward(const FEPRouter *router, const Tensor *x,
                               const HSPAConfig *cfg, bool training) {
    RoutingDecision rd;
    rd.expert_ids    = NULL;
    rd.expert_weights = NULL;
    rd.entropy       = 0.0f;

    if (!router || !x || !cfg) {
        return rd;
    }

    int32_t K = cfg->n_experts;   /* total experts */
    int32_t k = cfg->n_active;    /* active experts (top-k) */
    int32_t d = cfg->d_model;

    /* ------------------------------------------------------------------
     * Step 1: Compute routing logits = x^T @ W_mu -> [n_experts]
     *
     * x is [d_model], W_mu is [d_model, n_experts].
     * Reshape x to [1, d_model], matmul to get [1, n_experts].
     * ------------------------------------------------------------------ */
    float *logits = (float *)calloc((size_t)K, sizeof(float));
    if (!logits) {
        return rd;
    }

    for (int32_t j = 0; j < K; j++) {
        float dot = 0.0f;
        for (int32_t i = 0; i < d; i++) {
            int32_t x_idx[] = {i};
            int32_t w_idx[] = {i, j};
            dot += tensor_get(x, x_idx) * tensor_get(router->W_mu, w_idx);
        }
        logits[j] = dot;
    }

    /* ------------------------------------------------------------------
     * Step 2: Compute routing uncertainties
     * log_var = W_sigma^T @ x -> [n_experts]
     * sigma = softplus(log_var)
     * ------------------------------------------------------------------ */
    float *sigma = (float *)calloc((size_t)K, sizeof(float));
    if (!sigma) {
        free(logits);
        return rd;
    }

    for (int32_t j = 0; j < K; j++) {
        float dot = 0.0f;
        for (int32_t i = 0; i < d; i++) {
            int32_t x_idx[] = {i};
            int32_t w_idx[] = {i, j};
            dot += tensor_get(x, x_idx) * tensor_get(router->W_sigma, w_idx);
        }
        sigma[j] = softplus(dot);
    }

    /* ------------------------------------------------------------------
     * Step 3: Expected free energy per expert
     *
     * mu_probs = softmax(logits)
     * ambiguity[i]  = 0.5 * sigma[i]^2
     * info_gain[i]  = log(1 + sigma[i]^2 / sigma_prior^2)
     * complexity[i] = mu_probs[i] * (log(mu_probs[i]) + log(K))
     * G[i] = ambiguity[i] - info_gain[i] + complexity[i]
     * ------------------------------------------------------------------ */

    /* Compute softmax(logits) -> mu_probs, using numerically stable form */
    float *mu_probs = (float *)calloc((size_t)K, sizeof(float));
    if (!mu_probs) {
        free(sigma);
        free(logits);
        return rd;
    }

    float max_logit = logits[0];
    for (int32_t j = 1; j < K; j++) {
        if (logits[j] > max_logit) {
            max_logit = logits[j];
        }
    }

    float sum_exp = 0.0f;
    for (int32_t j = 0; j < K; j++) {
        mu_probs[j] = expf(logits[j] - max_logit);
        sum_exp += mu_probs[j];
    }
    for (int32_t j = 0; j < K; j++) {
        mu_probs[j] /= sum_exp;
    }

    /* Compute G[i] for each expert */
    float *G = (float *)calloc((size_t)K, sizeof(float));
    if (!G) {
        free(mu_probs);
        free(sigma);
        free(logits);
        return rd;
    }

    float sigma_prior_sq = router->sigma_prior * router->sigma_prior;
    float log_K = logf((float)K);
    float eps = 1e-8f;

    for (int32_t j = 0; j < K; j++) {
        float sigma_sq = sigma[j] * sigma[j];

        /* Ambiguity: prefer low uncertainty */
        float ambiguity = 0.5f * sigma_sq;

        /* Information gain: prefer high info gain (this is subtracted) */
        float info_gain = logf(1.0f + sigma_sq / sigma_prior_sq);

        /* Complexity: prefer uniform distribution (minimize KL from uniform) */
        float complexity = mu_probs[j] * (logf(mu_probs[j] + eps) + log_K);

        G[j] = ambiguity - info_gain + complexity;
    }

    /* ------------------------------------------------------------------
     * Step 4: If training, add noise for exploration
     * G[i] += epsilon * sigma[i], epsilon ~ N(0, 1)
     * ------------------------------------------------------------------ */
    if (training) {
        for (int32_t j = 0; j < K; j++) {
            G[j] += randn() * sigma[j];
        }
    }

    /* Step 4b: Apply auxiliary-loss-free bias (non-differentiable shift).
     * Positive bias makes an expert MORE likely (subtracts from G,
     * since lowest G wins). This does NOT inject any gradient signal. */
    if (router->expert_bias) {
        for (int32_t j = 0; j < K; j++) {
            G[j] -= router->expert_bias[j];
        }
    }

    /* ------------------------------------------------------------------
     * Step 5: Select top-k experts with LOWEST G
     * Simple O(K * k) selection: find min, record, set to INFINITY, repeat.
     * ------------------------------------------------------------------ */
    rd.expert_ids = (int32_t *)malloc((size_t)k * sizeof(int32_t));
    if (!rd.expert_ids) {
        free(G);
        free(mu_probs);
        free(sigma);
        free(logits);
        return rd;
    }

    /* Work on a copy of G so we can set selected entries to INFINITY */
    float *G_work = (float *)malloc((size_t)K * sizeof(float));
    if (!G_work) {
        free(rd.expert_ids);
        rd.expert_ids = NULL;
        free(G);
        free(mu_probs);
        free(sigma);
        free(logits);
        return rd;
    }
    memcpy(G_work, G, (size_t)K * sizeof(float));

    for (int32_t sel = 0; sel < k; sel++) {
        int32_t min_idx = 0;
        float min_val = G_work[0];
        for (int32_t j = 1; j < K; j++) {
            if (G_work[j] < min_val) {
                min_val = G_work[j];
                min_idx = j;
            }
        }
        rd.expert_ids[sel] = min_idx;
        G_work[min_idx] = INFINITY;
    }

    /* ------------------------------------------------------------------
     * Step 6: Compute weights from negative free energy
     * weights = softmax(-G[selected]) over the selected experts only
     * ------------------------------------------------------------------ */
    rd.expert_weights = (float *)malloc((size_t)k * sizeof(float));
    if (!rd.expert_weights) {
        free(G_work);
        free(rd.expert_ids);
        rd.expert_ids = NULL;
        free(G);
        free(mu_probs);
        free(sigma);
        free(logits);
        return rd;
    }

    /* Find max of -G[selected] for numerical stability */
    float max_neg_G = -G[rd.expert_ids[0]];
    for (int32_t i = 1; i < k; i++) {
        float neg_G = -G[rd.expert_ids[i]];
        if (neg_G > max_neg_G) {
            max_neg_G = neg_G;
        }
    }

    float weight_sum = 0.0f;
    for (int32_t i = 0; i < k; i++) {
        float neg_G = -G[rd.expert_ids[i]];
        rd.expert_weights[i] = expf(neg_G - max_neg_G);
        weight_sum += rd.expert_weights[i];
    }

    for (int32_t i = 0; i < k; i++) {
        rd.expert_weights[i] /= weight_sum;
    }

    /* ------------------------------------------------------------------
     * Step 7: Compute routing entropy H(g) for VFE loss
     * H(g) = -sum(mu_probs[i] * log(mu_probs[i] + eps)) for all K experts
     * ------------------------------------------------------------------ */
    float entropy = 0.0f;
    for (int32_t j = 0; j < K; j++) {
        if (mu_probs[j] > eps) {
            entropy -= mu_probs[j] * logf(mu_probs[j] + eps);
        }
    }
    rd.entropy = entropy;

    /* Cleanup scratch */
    free(G_work);
    free(G);
    free(mu_probs);
    free(sigma);
    free(logits);

    return rd;
}

void router_update_bias(FEPRouter *router, const int32_t *expert_counts,
                        int32_t total_tokens, int32_t n_experts) {
    if (!router || !expert_counts || total_tokens <= 0 || n_experts <= 0) {
        return;
    }
    if (!router->expert_bias || !router->expert_load_ema) {
        return;
    }

    float alpha  = router->bias_update_rate;
    float step   = router->bias_step_size;
    float target = 1.0f / (float)n_experts;

    for (int32_t j = 0; j < n_experts; j++) {
        float actual_load = (float)expert_counts[j] / (float)total_tokens;

        /* Update EMA of expert load */
        router->expert_load_ema[j] =
            (1.0f - alpha) * router->expert_load_ema[j] + alpha * actual_load;

        /* Adjust bias: increase for underloaded, decrease for overloaded */
        if (router->expert_load_ema[j] > target) {
            router->expert_bias[j] -= step;
        } else {
            router->expert_bias[j] += step;
        }
    }
}

void router_destroy(FEPRouter *router) {
    if (!router) {
        return;
    }

    if (router->W_mu) {
        tensor_destroy(router->W_mu);
    }
    if (router->W_sigma) {
        tensor_destroy(router->W_sigma);
    }

    free(router->expert_bias);
    free(router->expert_load_ema);
    free(router->default_moe_ema);

    free(router);
}

/* ---- Plan B: Default MoE EMA init / destroy ---- */

/* Deterministic Gaussian sampler seeded by caller. Keeps Box-Muller local
 * so we don't perturb the global srand() used elsewhere in router_forward(). */
static float plan_b_randn(uint32_t *state) {
    /* xorshift32 → uniform in (0,1), then Box-Muller. */
    uint32_t s = *state;
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    *state = s;
    float u1 = ((float)(s & 0x00FFFFFFu) + 1.0f) / (float)(0x01000000u + 1u);
    s ^= s << 13; s ^= s >> 17; s ^= s << 5;
    *state = s;
    float u2 = ((float)(s & 0x00FFFFFFu) + 1.0f) / (float)(0x01000000u + 1u);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

int router_init_default_moe(FEPRouter *router, int32_t d_model,
                            float alpha, float sigma_init, uint32_t seed) {
    if (!router || d_model <= 0 || sigma_init <= 0.0f || alpha < 0.0f) {
        return -1;
    }
    if (router->default_moe_ema) {
        /* Already initialized; refuse to leak. Caller must
         * router_destroy_default_moe() first. */
        return -1;
    }

    size_t n = (size_t)router->n_experts * (size_t)d_model;
    float *ema = (float *)malloc(n * sizeof(float));
    if (!ema) {
        return -1;
    }

    /* Deterministic Gaussian init (σ_init) — paper §5 / design §9-R2. */
    uint32_t rs = seed ? seed : 0xDEADBEEFu;
    for (size_t i = 0; i < n; i++) {
        ema[i] = sigma_init * plan_b_randn(&rs);
    }

    router->default_moe_ema   = ema;
    router->default_moe_dim   = d_model;
    router->default_moe_alpha = alpha;
    return 0;
}

void router_destroy_default_moe(FEPRouter *router) {
    if (!router) return;
    free(router->default_moe_ema);
    router->default_moe_ema   = NULL;
    router->default_moe_dim   = 0;
    router->default_moe_alpha = 0.0f;
}
