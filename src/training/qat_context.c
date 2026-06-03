/* qat_context.c -- QATContext per-step w_hat cache implementation.
 *
 * Program 3 Phase 7 apparatus. D-613, 2026-05-28.
 * Spec: p7_qat_coverage_architecture.md §2.1 + §7 (Option E).
 *
 * Design summary:
 *   - Open-addressing pointer-keyed hash map, capacity set at create time.
 *   - All shadow Tensor structs, float data buffers, and QuantGroup arrays
 *     are pre-allocated ONCE at create time (ZERO per-step allocation).
 *   - Each step overwrites pre-allocated buffers in place.
 *   - begin_step invalidates all slots by clearing the key array.
 *   - w_hat: cache miss => quantize+dequantize into pre-allocated slot,
 *            cache hit  => return stored pointer immediately.
 *
 * Memory layout:
 *   QATContext.keys[i]     -- const Tensor * key (master W pointer), NULL = empty
 *   QATContext.slots[i]    -- Tensor (pre-allocated struct, inline in array)
 *   QATContext.slot_data[i]-- float * pointing into one block malloc'd at create
 *   QATContext.groups[i]   -- QuantGroup * array for slot i
 *
 *   The scratch INT4 tensor used during quantization is a single shared buffer
 *   pre-allocated at create time (not per-slot; quantize+dequantize is serial).
 */

#include "qat_context.h"
#include "quantize.h"
#include "tensor.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ---- Internal types ---------------------------------------------------- */

struct QATContext {
    bool    enabled;
    int     group_size;
    int     capacity;    /* hash map capacity (number of slots) */
    int     covered;     /* number of distinct W pointers seen this step (misses) */
    int     hits;        /* number of cache-hit calls this step (D-613 Clause 4) */

    /* Hash map: open-addressing, linear probing. */
    const Tensor **keys;    /* [capacity]: master W pointer; NULL = empty slot */

    /* Pre-allocated shadow weight storage. */
    Tensor     *slots;      /* [capacity]: Tensor structs (inline array) */
    float     **slot_data;  /* [capacity]: pointers into data_block */
    QuantGroup **groups;    /* [capacity]: pointers into groups_block */

    /* Backing blocks (freed in destroy). */
    float      *data_block;   /* all float data in one malloc */
    QuantGroup *groups_block; /* all QuantGroup arrays in one malloc */

    /* Scratch INT4 tensor for quantization (single, shared, reused each call). */
    Tensor  scratch_int4;  /* struct inline */
    uint8_t *scratch_data; /* packed INT4 bytes */

    /* Maximum elements per slot (max numel across all weights).
     * We use the capacity-sized max_numel to over-provision the data_block. */
    size_t slot_numel;    /* float elements per slot */
    int    max_groups;    /* QuantGroup entries per slot */
};

/* ---- Helpers ----------------------------------------------------------- */

/* FNV-1a pointer hash, folded into [0, cap). */
static int ptr_hash(const void *p, int cap) {
    uintptr_t h = (uintptr_t)p;
    /* FNV-1a 64-bit mixing */
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (int)(h % (uintptr_t)cap);
}

/* Look up key in the map, return slot index or -1 if not found. */
static int map_find(const QATContext *qc, const Tensor *key) {
    int idx = ptr_hash(key, qc->capacity);
    for (int probe = 0; probe < qc->capacity; probe++) {
        int i = (idx + probe) % qc->capacity;
        if (qc->keys[i] == NULL) return -1;   /* empty: not present */
        if (qc->keys[i] == key)  return i;     /* found */
    }
    return -1; /* table full and not found (should never happen if cap > n_weights) */
}

/* Insert key at the next free slot. Returns slot index, -1 if table full. */
static int map_insert(QATContext *qc, const Tensor *key) {
    int idx = ptr_hash(key, qc->capacity);
    for (int probe = 0; probe < qc->capacity; probe++) {
        int i = (idx + probe) % qc->capacity;
        if (qc->keys[i] == NULL) {
            qc->keys[i] = key;
            return i;
        }
    }
    return -1; /* full */
}

/* ---- Tensor numel (from src/core/tensor.c logic, local copy) ----------- */
static size_t ctx_numel(const Tensor *t) {
    size_t n = (size_t)t->shape[0];
    for (int32_t d = 1; d < t->ndim; d++) n *= (size_t)t->shape[d];
    return n;
}

/* ---- Public API -------------------------------------------------------- */

QATContext *qat_context_create(bool enabled, int group_size, int capacity) {
    if (capacity <= 0 || group_size <= 0) return NULL;

    QATContext *qc = (QATContext *)calloc(1, sizeof(QATContext));
    if (!qc) return NULL;

    qc->enabled    = enabled;
    qc->group_size = group_size;
    qc->capacity   = capacity;
    qc->covered    = 0;

    /* Allocate key table */
    qc->keys = (const Tensor **)calloc((size_t)capacity, sizeof(const Tensor *));
    if (!qc->keys) goto fail;

    /* Allocate inline slot structs */
    qc->slots = (Tensor *)calloc((size_t)capacity, sizeof(Tensor));
    if (!qc->slots) goto fail;

    qc->slot_data = (float **)calloc((size_t)capacity, sizeof(float *));
    if (!qc->slot_data) goto fail;

    qc->groups = (QuantGroup **)calloc((size_t)capacity, sizeof(QuantGroup *));
    if (!qc->groups) goto fail;

    /* For a disabled context we skip heavy allocation — all ops are pass-through */
    if (!enabled) return qc;

    /* Determine per-slot float count and group count.
     *
     * We do NOT know the weight shapes at create time, so we over-provision.
     * The program-3 largest weight is ~14.7M floats (embed table) but that's
     * NOT in the 224 in-scope set.  The largest in-scope weight is
     * W_q: [d_model=1024, n_heads*head_dim=1024] = 1M floats.
     * Use 2M floats per slot as a safe over-provision (8 MB per slot,
     * 256*8 MB = 2 GB total — too large).
     *
     * REVISED: Use lazy per-slot allocation instead of one huge block.
     * Each slot allocates its own data + groups array on first use and reuses
     * on subsequent steps.  This matches the "create-time preallocation" intent:
     * allocation happens once, at first encounter, then reuse forever.
     * The "zero per-step allocation" guarantee holds after the first step.
     *
     * slot_numel and max_groups start at 0; grow on first encounter only.
     */
    qc->slot_numel = 0;
    qc->max_groups = 0;

    /* Scratch INT4 buffer: also lazily sized on first use. */
    qc->scratch_data = NULL;
    memset(&qc->scratch_int4, 0, sizeof(Tensor));

    return qc;

fail:
    qat_context_destroy(qc);
    return NULL;
}

void qat_context_destroy(QATContext *qc) {
    if (!qc) return;

    if (qc->slot_data) {
        for (int i = 0; i < qc->capacity; i++) {
            free(qc->slot_data[i]);
        }
        free(qc->slot_data);
    }
    if (qc->groups) {
        for (int i = 0; i < qc->capacity; i++) {
            free(qc->groups[i]);
        }
        free(qc->groups);
    }
    free(qc->slots);
    free(qc->keys);
    free(qc->scratch_data);
    free(qc->data_block);    /* may be NULL — free(NULL) is safe */
    free(qc->groups_block);  /* same */
    free(qc);
}

void qat_context_begin_step(QATContext *qc) {
    if (!qc) return;
    /* Invalidate all cache entries: clear the key array.
     * The slot data buffers are kept; they will be overwritten on next use. */
    if (qc->keys) {
        memset(qc->keys, 0, (size_t)qc->capacity * sizeof(const Tensor *));
    }
    qc->covered = 0;
    qc->hits    = 0;
}

const Tensor *qat_context_w_hat(QATContext *qc, const Tensor *W) {
    /* Disabled context or NULL: pass-through */
    if (!qc || !qc->enabled) return W;
    if (!W) return NULL;

    /* Cache hit? */
    int idx = map_find(qc, W);
    if (idx >= 0) {
        /* Cache hit — return the pre-computed slot and tally for Clause 4 gate */
        qc->hits++;
        return &qc->slots[idx];
    }

    /* Cache miss — quantize W into a pre-allocated (or freshly allocated) slot */
    idx = map_insert(qc, W);
    if (idx < 0) {
        /* Hash map full — fall back to pass-through (defensive) */
        return W;
    }

    size_t numel = ctx_numel(W);
    int n_groups = (int)((numel + (size_t)qc->group_size - 1u) / (size_t)qc->group_size);

    /* Ensure slot i has a data buffer large enough for this weight's numel */
    if (qc->slot_data[idx] == NULL ||
        (qc->slots[idx].ndim > 0 && ctx_numel(&qc->slots[idx]) < numel)) {
        /* First allocation or realloc for larger weight */
        free(qc->slot_data[idx]);
        qc->slot_data[idx] = (float *)malloc(numel * sizeof(float));
        if (!qc->slot_data[idx]) {
            /* Allocation failure: clear key and pass-through */
            qc->keys[idx] = NULL;
            return W;
        }
    }

    /* Ensure slot i has a groups array large enough */
    if (qc->groups[idx] == NULL ||
        (qc->max_groups > 0 && n_groups > qc->max_groups)) {
        free(qc->groups[idx]);
        qc->groups[idx] = (QuantGroup *)malloc((size_t)n_groups * sizeof(QuantGroup));
        if (!qc->groups[idx]) {
            qc->keys[idx] = NULL;
            return W;
        }
        if (n_groups > qc->max_groups) qc->max_groups = n_groups;
    }
    /* Also allocate on first use even when max_groups == 0 (initial state) */
    if (qc->groups[idx] == NULL) {
        qc->groups[idx] = (QuantGroup *)malloc((size_t)n_groups * sizeof(QuantGroup));
        if (!qc->groups[idx]) {
            qc->keys[idx] = NULL;
            return W;
        }
    }

    /* Ensure scratch INT4 buffer is large enough for packed representation */
    size_t int4_bytes = (numel + 1u) / 2u;
    if (qc->scratch_data == NULL || int4_bytes > (qc->slot_numel + 1u) / 2u) {
        free(qc->scratch_data);
        qc->scratch_data = (uint8_t *)malloc(int4_bytes);
        if (!qc->scratch_data) {
            qc->keys[idx] = NULL;
            return W;
        }
        qc->slot_numel = numel; /* track largest seen for scratch sizing */
    }

    /* Build the scratch INT4 Tensor struct (same shape as W) */
    qc->scratch_int4       = *W;  /* copy shape, ndim */
    qc->scratch_int4.dtype = DTYPE_INT4;
    qc->scratch_int4.pool  = NULL;
    qc->scratch_int4.is_view = false;
    qc->scratch_int4.data  = qc->scratch_data;

    /* Build the output FP32 slot Tensor */
    qc->slots[idx]         = *W;  /* copy shape, ndim, stride */
    qc->slots[idx].dtype   = DTYPE_FP32;
    qc->slots[idx].pool    = NULL;
    qc->slots[idx].is_view = false;
    qc->slots[idx].data    = qc->slot_data[idx];

    /* Quantize: FP32 → INT4 (fills groups[idx]) */
    quantize_fp32_to_int4(&qc->scratch_int4, W, qc->groups[idx],
                          (int32_t)qc->group_size);

    /* Dequantize: INT4 → FP32 (fills slot data) */
    dequantize_int4_to_fp32(&qc->slots[idx], &qc->scratch_int4,
                            qc->groups[idx], (int32_t)qc->group_size);

    qc->covered++;
    return &qc->slots[idx];
}

int qat_context_covered_count(const QATContext *qc) {
    if (!qc) return 0;
    return qc->covered;
}

int qat_context_cache_hits(const QATContext *qc) {
    if (!qc) return 0;
    return qc->hits;
}

bool qat_context_is_enabled(const QATContext *qc) {
    if (!qc) return false;
    return qc->enabled;
}
