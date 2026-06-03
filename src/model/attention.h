/* attention.h -- Grouped Query Attention (GQA).
 *
 * Multi-head attention with grouped KV heads for memory efficiency.
 * n_heads=32 query heads share n_kv_heads=8 key/value heads.
 * Each group of (n_heads / n_kv_heads) = 4 query heads shares one KV head.
 *
 * KV cache is pre-allocated at init for the full context length.
 */

#ifndef ATTENTION_H
#define ATTENTION_H

#include "hspa_config.h"
#include "memory_pool.h"
#include "qat_context.h"
#include "tensor.h"

#include <stdint.h>

typedef struct {
    Tensor *W_q; /* Query projection: [d_model, n_heads * head_dim] */
    Tensor *W_k; /* Key projection:   [d_model, n_kv_heads * head_dim] */
    Tensor *W_v; /* Value projection:  [d_model, n_kv_heads * head_dim] */
    Tensor *W_o; /* Output projection: [n_heads * head_dim, d_model] */
} Attention;

typedef struct {
    Tensor *k_cache; /* Key cache:   [max_seq_len, n_kv_heads, head_dim] */
    Tensor *v_cache; /* Value cache: [max_seq_len, n_kv_heads, head_dim] */
    int32_t pos;     /* Current position in cache (next write slot) */
} KVCache;

/* Create attention layer, allocating weights from `pool`.
 * Returns NULL on failure. */
Attention *attention_create(MemoryPool *pool, const HSPAConfig *cfg);

/* Forward pass: computes multi-head GQA attention.
 * out: [seq_len, d_model] (pre-allocated)
 * x: [seq_len, d_model] input
 * cache: KV cache (updated in-place with new K,V)
 * pos: starting position in the sequence
 * qat: QATContext for fake-quantizing weight matrices; NULL = plain matmul */
void attention_forward(Tensor *out, const Attention *attn, const Tensor *x,
                       KVCache *cache, int32_t pos, QATContext *qat);

/* Destroy the attention layer struct. */
void attention_destroy(Attention *attn);

/* Create KV cache pre-allocated for full context length.
 * Returns NULL on failure. */
KVCache *kv_cache_create(MemoryPool *pool, const HSPAConfig *cfg);

/* Destroy the KV cache struct. */
void kv_cache_destroy(KVCache *cache);

#endif /* ATTENTION_H */
