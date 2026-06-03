/* attention.c -- Grouped Query Attention (GQA) with KV Cache.
 *
 * Implements multi-head attention where n_heads query heads share
 * n_kv_heads key/value heads. Each group of (n_heads / n_kv_heads)
 * query heads shares one KV head.
 *
 * Forward pass:
 *   1. Project Q, K, V from input x via linear projections
 *   2. Store K, V into pre-allocated KV cache
 *   3. Per head: compute scaled dot-product attention with causal mask
 *   4. Concatenate head outputs, project through W_o
 *
 * Performance note: this CPU reference implementation uses temporary 2D
 * tensors and tensor_get/tensor_set for per-head computation. Metal
 * kernels will replace this path later.
 */

#include "attention.h"
#include "ops.h"
#include "qat_context.h"

#include <math.h>
#include <stdlib.h>

/* ---- Internal: config parameters stored for the forward pass ---- */

typedef struct {
    int32_t d_model;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t head_dim;
    int32_t max_seq_len;
} AttnParams;

/* Stored alongside Attention via struct embedding (attention.h only
 * exposes the 4 weight tensors). We store params right after the
 * public struct in a single allocation. Since the header defines
 * Attention as a typedef'd struct with exactly 4 fields, we store
 * the config in a separate allocation pointed to via a convention:
 * we allocate Attention + AttnParams together. */

/* We cannot extend Attention (defined in header), so we store params
 * in a side allocation freed in attention_destroy. We use a file-scope
 * mapping from Attention* -> AttnParams*. For simplicity with the
 * single-threaded design, we use a small static table. */

#define MAX_ATTN_INSTANCES 64

static struct {
    const Attention *attn;
    AttnParams params;
} _attn_registry[MAX_ATTN_INSTANCES];
static int _attn_registry_count = 0;

static void attn_register(const Attention *attn, const AttnParams *p) {
    if (_attn_registry_count < MAX_ATTN_INSTANCES) {
        _attn_registry[_attn_registry_count].attn = attn;
        _attn_registry[_attn_registry_count].params = *p;
        _attn_registry_count++;
    }
}

static const AttnParams *attn_lookup(const Attention *attn) {
    for (int i = 0; i < _attn_registry_count; i++) {
        if (_attn_registry[i].attn == attn) {
            return &_attn_registry[i].params;
        }
    }
    return NULL;
}

static void attn_unregister(const Attention *attn) {
    for (int i = 0; i < _attn_registry_count; i++) {
        if (_attn_registry[i].attn == attn) {
            /* Swap with last entry */
            _attn_registry[i] = _attn_registry[_attn_registry_count - 1];
            _attn_registry_count--;
            return;
        }
    }
}

/* ---- API ---- */

Attention *attention_create(MemoryPool *pool, const HSPAConfig *cfg) {
    if (!pool || !cfg || cfg->d_model <= 0 || cfg->n_heads <= 0 ||
        cfg->n_kv_heads <= 0 || cfg->head_dim <= 0) {
        return NULL;
    }

    Attention *attn = (Attention *)calloc(1, sizeof(Attention));
    if (!attn) {
        return NULL;
    }

    /* W_q: [d_model, n_heads * head_dim] */
    int32_t wq_shape[] = {cfg->d_model, cfg->n_heads * cfg->head_dim};
    attn->W_q = tensor_create(pool, wq_shape, 2, DTYPE_FP32);
    if (!attn->W_q) {
        free(attn);
        return NULL;
    }

    /* W_k: [d_model, n_kv_heads * head_dim] */
    int32_t wk_shape[] = {cfg->d_model, cfg->n_kv_heads * cfg->head_dim};
    attn->W_k = tensor_create(pool, wk_shape, 2, DTYPE_FP32);
    if (!attn->W_k) {
        tensor_destroy(attn->W_q);
        free(attn);
        return NULL;
    }

    /* W_v: [d_model, n_kv_heads * head_dim] */
    int32_t wv_shape[] = {cfg->d_model, cfg->n_kv_heads * cfg->head_dim};
    attn->W_v = tensor_create(pool, wv_shape, 2, DTYPE_FP32);
    if (!attn->W_v) {
        tensor_destroy(attn->W_k);
        tensor_destroy(attn->W_q);
        free(attn);
        return NULL;
    }

    /* W_o: [n_heads * head_dim, d_model] */
    int32_t wo_shape[] = {cfg->n_heads * cfg->head_dim, cfg->d_model};
    attn->W_o = tensor_create(pool, wo_shape, 2, DTYPE_FP32);
    if (!attn->W_o) {
        tensor_destroy(attn->W_v);
        tensor_destroy(attn->W_k);
        tensor_destroy(attn->W_q);
        free(attn);
        return NULL;
    }

    /* Zero-initialize all weights. Real init at training time. */
    tensor_fill(attn->W_q, 0.0f);
    tensor_fill(attn->W_k, 0.0f);
    tensor_fill(attn->W_v, 0.0f);
    tensor_fill(attn->W_o, 0.0f);

    /* Register config params for the forward pass */
    AttnParams p;
    p.d_model     = cfg->d_model;
    p.n_heads     = cfg->n_heads;
    p.n_kv_heads  = cfg->n_kv_heads;
    p.head_dim    = cfg->head_dim;
    p.max_seq_len = cfg->max_seq_len;
    attn_register(attn, &p);

    return attn;
}

void attention_destroy(Attention *attn) {
    if (!attn) {
        return;
    }

    attn_unregister(attn);

    if (attn->W_o) tensor_destroy(attn->W_o);
    if (attn->W_v) tensor_destroy(attn->W_v);
    if (attn->W_k) tensor_destroy(attn->W_k);
    if (attn->W_q) tensor_destroy(attn->W_q);

    free(attn);
}

KVCache *kv_cache_create(MemoryPool *pool, const HSPAConfig *cfg) {
    if (!pool || !cfg || cfg->max_seq_len <= 0 ||
        cfg->n_kv_heads <= 0 || cfg->head_dim <= 0) {
        return NULL;
    }

    KVCache *cache = (KVCache *)calloc(1, sizeof(KVCache));
    if (!cache) {
        return NULL;
    }

    /* k_cache: [max_seq_len, n_kv_heads, head_dim] */
    int32_t k_shape[] = {cfg->max_seq_len, cfg->n_kv_heads, cfg->head_dim};
    cache->k_cache = tensor_create(pool, k_shape, 3, DTYPE_FP32);
    if (!cache->k_cache) {
        free(cache);
        return NULL;
    }

    /* v_cache: [max_seq_len, n_kv_heads, head_dim] */
    int32_t v_shape[] = {cfg->max_seq_len, cfg->n_kv_heads, cfg->head_dim};
    cache->v_cache = tensor_create(pool, v_shape, 3, DTYPE_FP32);
    if (!cache->v_cache) {
        tensor_destroy(cache->k_cache);
        free(cache);
        return NULL;
    }

    tensor_fill(cache->k_cache, 0.0f);
    tensor_fill(cache->v_cache, 0.0f);
    cache->pos = 0;

    return cache;
}

void kv_cache_destroy(KVCache *cache) {
    if (!cache) {
        return;
    }

    if (cache->v_cache) tensor_destroy(cache->v_cache);
    if (cache->k_cache) tensor_destroy(cache->k_cache);

    free(cache);
}

/* ---- Forward pass ---- */

void attention_forward(Tensor *out, const Attention *attn, const Tensor *x,
                       KVCache *cache, int32_t pos, QATContext *qat) {
    if (!out || !attn || !x || !cache) {
        return;
    }

    const AttnParams *p = attn_lookup(attn);
    if (!p) {
        return;
    }

    int32_t seq_len    = x->shape[0];
    int32_t d_model    = p->d_model;
    int32_t n_heads    = p->n_heads;
    int32_t n_kv_heads = p->n_kv_heads;
    int32_t head_dim   = p->head_dim;
    int32_t heads_per_group = n_heads / n_kv_heads;
    int32_t cache_len  = pos + seq_len; /* total tokens available after writing */

    /* We need scratch tensors. Use a scratch pool that is reset at end. */
    /* Estimate: generous scratch for all intermediates.
     * Q: seq_len * n_heads * head_dim
     * K: seq_len * n_kv_heads * head_dim
     * V: seq_len * n_kv_heads * head_dim
     * context: seq_len * n_heads * head_dim
     * per-head temps: several small 2D tensors
     * Total: several MB is enough for test sizes. */
    size_t scratch_size = (size_t)(seq_len + cache_len + 16) *
                          (size_t)(d_model + n_heads * head_dim + 256) *
                          sizeof(float) * 4;
    if (scratch_size < 1024 * 1024) {
        scratch_size = 1024 * 1024; /* minimum 1 MB */
    }
    MemoryPool *scratch = pool_create(scratch_size, POOL_SCRATCH);
    if (!scratch) {
        return;
    }

    /* Step 1: Q = x @ W_q  ->  [seq_len, n_heads * head_dim] */
    int32_t q_shape[] = {seq_len, n_heads * head_dim};
    Tensor *Q = tensor_create(scratch, q_shape, 2, DTYPE_FP32);

    /* Step 2: K = x @ W_k  ->  [seq_len, n_kv_heads * head_dim] */
    int32_t k_shape[] = {seq_len, n_kv_heads * head_dim};
    Tensor *K_proj = tensor_create(scratch, k_shape, 2, DTYPE_FP32);

    /* Step 3: V = x @ W_v  ->  [seq_len, n_kv_heads * head_dim] */
    int32_t v_shape[] = {seq_len, n_kv_heads * head_dim};
    Tensor *V_proj = tensor_create(scratch, v_shape, 2, DTYPE_FP32);

    if (!Q || !K_proj || !V_proj) {
        if (Q) tensor_destroy(Q);
        if (K_proj) tensor_destroy(K_proj);
        if (V_proj) tensor_destroy(V_proj);
        pool_destroy(scratch);
        return;
    }

    op_matmul_qat(Q, x, attn->W_q, qat);
    op_matmul_qat(K_proj, x, attn->W_k, qat);
    op_matmul_qat(V_proj, x, attn->W_v, qat);

    /* Step 4: Store K, V into cache at positions [pos : pos+seq_len] */
    for (int32_t s = 0; s < seq_len; s++) {
        for (int32_t kv = 0; kv < n_kv_heads; kv++) {
            for (int32_t d = 0; d < head_dim; d++) {
                int32_t k_flat_col = kv * head_dim + d;
                int32_t kproj_idx[] = {s, k_flat_col};
                float k_val = tensor_get(K_proj, kproj_idx);

                int32_t cache_idx[] = {pos + s, kv, d};
                tensor_set(cache->k_cache, cache_idx, k_val);

                int32_t vproj_idx[] = {s, k_flat_col};
                float v_val = tensor_get(V_proj, vproj_idx);
                tensor_set(cache->v_cache, cache_idx, v_val);
            }
        }
    }

    /* Step 5: Per-head attention with GQA grouping */

    /* Allocate combined context: [seq_len, n_heads * head_dim] */
    int32_t ctx_shape[] = {seq_len, n_heads * head_dim};
    Tensor *context = tensor_create(scratch, ctx_shape, 2, DTYPE_FP32);
    if (!context) {
        tensor_destroy(V_proj);
        tensor_destroy(K_proj);
        tensor_destroy(Q);
        pool_destroy(scratch);
        return;
    }
    tensor_fill(context, 0.0f);

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int32_t h = 0; h < n_heads; h++) {
        int32_t kv_head = h / heads_per_group; /* GQA grouping */

        /* 5a: Extract q_h [seq_len, head_dim] from Q */
        int32_t qh_shape[] = {seq_len, head_dim};
        Tensor *q_h = tensor_create(scratch, qh_shape, 2, DTYPE_FP32);

        /* 5b: Extract k_h [cache_len, head_dim] from k_cache */
        int32_t kh_shape[] = {cache_len, head_dim};
        Tensor *k_h = tensor_create(scratch, kh_shape, 2, DTYPE_FP32);

        /* 5c: Extract v_h [cache_len, head_dim] from v_cache */
        int32_t vh_shape[] = {cache_len, head_dim};
        Tensor *v_h = tensor_create(scratch, vh_shape, 2, DTYPE_FP32);

        /* k_h transposed for matmul: [head_dim, cache_len] */
        int32_t kht_shape[] = {head_dim, cache_len};
        Tensor *k_h_t = tensor_create(scratch, kht_shape, 2, DTYPE_FP32);

        /* scores: [seq_len, cache_len] */
        int32_t sc_shape[] = {seq_len, cache_len};
        Tensor *scores = tensor_create(scratch, sc_shape, 2, DTYPE_FP32);

        /* attn_weights: [seq_len, cache_len] */
        Tensor *attn_w = tensor_create(scratch, sc_shape, 2, DTYPE_FP32);

        /* context_h: [seq_len, head_dim] */
        Tensor *ctx_h = tensor_create(scratch, qh_shape, 2, DTYPE_FP32);

        if (!q_h || !k_h || !v_h || !k_h_t || !scores || !attn_w || !ctx_h) {
            /* Cleanup on failure -- scratch pool destroy will free all */
            tensor_destroy(context);
            tensor_destroy(V_proj);
            tensor_destroy(K_proj);
            tensor_destroy(Q);
            pool_destroy(scratch);
            return;
        }

        /* Copy q_h from Q: Q[s, h*head_dim + d] */
        for (int32_t s = 0; s < seq_len; s++) {
            for (int32_t d = 0; d < head_dim; d++) {
                int32_t qi[] = {s, h * head_dim + d};
                int32_t qhi[] = {s, d};
                tensor_set(q_h, qhi, tensor_get(Q, qi));
            }
        }

        /* Copy k_h from k_cache: k_cache[t, kv_head, d] */
        for (int32_t t = 0; t < cache_len; t++) {
            for (int32_t d = 0; d < head_dim; d++) {
                int32_t ci[] = {t, kv_head, d};
                int32_t khi[] = {t, d};
                tensor_set(k_h, khi, tensor_get(cache->k_cache, ci));
            }
        }

        /* Copy v_h from v_cache: v_cache[t, kv_head, d] */
        for (int32_t t = 0; t < cache_len; t++) {
            for (int32_t d = 0; d < head_dim; d++) {
                int32_t ci[] = {t, kv_head, d};
                int32_t vhi[] = {t, d};
                tensor_set(v_h, vhi, tensor_get(cache->v_cache, ci));
            }
        }

        /* Transpose k_h -> k_h_t: [head_dim, cache_len] */
        for (int32_t t = 0; t < cache_len; t++) {
            for (int32_t d = 0; d < head_dim; d++) {
                int32_t src[] = {t, d};
                int32_t dst[] = {d, t};
                tensor_set(k_h_t, dst, tensor_get(k_h, src));
            }
        }

        /* 5d: scores = q_h @ k_h^T -> [seq_len, cache_len] */
        op_matmul(scores, q_h, k_h_t);

        /* 5e: Scale by 1/sqrt(head_dim) */
        op_scale(scores, scores, scale);

        /* 5f: Apply causal mask.
         * For query at position (pos + i), it can attend to key at position j
         * only if j <= pos + i. So for i,j: if pos + i < j, mask. */
        for (int32_t i = 0; i < seq_len; i++) {
            for (int32_t j = 0; j < cache_len; j++) {
                if (pos + i < j) {
                    int32_t si[] = {i, j};
                    tensor_set(scores, si, -INFINITY);
                }
            }
        }

        /* 5g: attn_weights = softmax(scores, dim=1) */
        op_softmax(attn_w, scores, 1);

        /* 5h: context_h = attn_weights @ v_h -> [seq_len, head_dim] */
        op_matmul(ctx_h, attn_w, v_h);

        /* 5i: Write context_h into combined context at columns h*head_dim */
        for (int32_t s = 0; s < seq_len; s++) {
            for (int32_t d = 0; d < head_dim; d++) {
                int32_t ch_idx[] = {s, d};
                int32_t c_idx[] = {s, h * head_dim + d};
                tensor_set(context, c_idx, tensor_get(ctx_h, ch_idx));
            }
        }

        /* Per-head temporaries will be freed when scratch is destroyed */
        tensor_destroy(ctx_h);
        tensor_destroy(attn_w);
        tensor_destroy(scores);
        tensor_destroy(k_h_t);
        tensor_destroy(v_h);
        tensor_destroy(k_h);
        tensor_destroy(q_h);
    }

    /* Step 6: out = context @ W_o -> [seq_len, d_model] */
    op_matmul_qat(out, context, attn->W_o, qat);

    /* Step 7: Update cache position */
    cache->pos = pos + seq_len;

    /* Cleanup scratch */
    tensor_destroy(context);
    tensor_destroy(V_proj);
    tensor_destroy(K_proj);
    tensor_destroy(Q);
    pool_destroy(scratch);
}
