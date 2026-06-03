/* memory_pool.h -- Arena-based memory pool allocator for unified CPU/GPU memory.
 *
 * Three pool types:
 *   POOL_WEIGHTS      -- Static pre-allocated region for model weights.
 *   POOL_ACTIVATIONS  -- Ring-buffer for forward/backward activations.
 *   POOL_SCRATCH      -- Temporary scratch space, reset between ops.
 *
 * Alignment guarantees:
 *   - Minimum 16-byte alignment (NEON SIMD).
 *   - Optional 4096-byte alignment for Metal page-aligned buffers.
 *
 * Ownership: each pool owns its backing memory. pool_destroy frees everything.
 *
 * Thread safety: none (single-threaded by design on M3).
 */

#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- Pool type ---- */
typedef enum {
    POOL_WEIGHTS,     /* Static: alloc only, no individual free */
    POOL_ACTIVATIONS, /* Ring buffer: alloc forward, reset to rewind */
    POOL_SCRATCH      /* Temporary: alloc, then reset entire pool */
} PoolType;

/* ---- Allocation header (internal, for free-list management) ---- */
typedef struct AllocHeader {
    size_t size;             /* usable size (excluding header + padding) */
    size_t total_size;       /* total block size including header + alignment */
    bool in_use;             /* whether this block is currently allocated */
    struct AllocHeader *next; /* next block in free list (POOL_WEIGHTS only) */
} AllocHeader;

/* ---- Memory pool ---- */
typedef struct MemoryPool {
    uint8_t *base;       /* backing memory (mmap'd or malloc'd) */
    size_t capacity;     /* total size of backing memory */
    size_t used;         /* bytes currently committed */
    size_t peak;         /* high-water mark */
    size_t alloc_count;  /* number of live allocations */
    PoolType type;       /* pool behaviour */

    /* Ring-buffer state (POOL_ACTIVATIONS) */
    size_t ring_head;    /* next allocation offset */

    /* Free-list head (POOL_WEIGHTS) */
    AllocHeader *free_list;
} MemoryPool;

/* ---- API ---- */

/* Create a pool with the given capacity and type.
 * Returns NULL on failure. */
MemoryPool *pool_create(size_t size, PoolType type);

/* Allocate `size` bytes from `pool`, aligned to `alignment`.
 * alignment must be a power of two and >= 16.
 * Returns NULL if the pool is exhausted. */
void *pool_alloc(MemoryPool *pool, size_t size, size_t alignment);

/* Free a specific allocation back to the pool.
 * Only meaningful for POOL_WEIGHTS (returns block to free list).
 * For POOL_ACTIVATIONS and POOL_SCRATCH, this is a no-op (use pool_reset). */
void pool_free(MemoryPool *pool, void *ptr);

/* Reset the pool.
 * POOL_ACTIVATIONS: rewinds ring head to 0.
 * POOL_SCRATCH: rewinds to 0.
 * POOL_WEIGHTS: no-op (weights are static). */
void pool_reset(MemoryPool *pool);

/* Query: how many bytes are currently in use? */
size_t pool_used(MemoryPool *pool);

/* Query: peak bytes ever in use (high-water mark). */
size_t pool_peak(MemoryPool *pool);

/* Destroy the pool and free all backing memory. */
void pool_destroy(MemoryPool *pool);

/* ---- Constants ---- */
#define POOL_MIN_ALIGNMENT  16    /* NEON SIMD minimum */
#define POOL_PAGE_ALIGNMENT 4096  /* Metal page alignment */

#endif /* MEMORY_POOL_H */
