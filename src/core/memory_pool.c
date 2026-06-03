/* memory_pool.c -- Arena-based memory pool allocator.
 *
 * Implementation details:
 *   - Backing memory is allocated via posix_memalign for page alignment.
 *   - POOL_WEIGHTS uses a simple free-list for individual block management.
 *   - POOL_ACTIVATIONS and POOL_SCRATCH use bump allocation (fast path).
 *   - Alignment is handled by rounding up the allocation offset.
 *
 * Memory layout for POOL_WEIGHTS blocks:
 *   [AllocHeader | padding | user_data ... ]
 *   ^                       ^
 *   block start              returned pointer (aligned)
 */

#include "memory_pool.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ---- Internal helpers ---- */

/* Round `x` up to the next multiple of `align` (align must be power of 2). */
static inline size_t align_up(size_t x, size_t align) {
    return (x + align - 1) & ~(align - 1);
}

/* Check if `n` is a power of two. */
static inline int is_power_of_two(size_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

/* ---- API implementation ---- */

MemoryPool *pool_create(size_t size, PoolType type) {
    if (size == 0) {
        return NULL;
    }

    MemoryPool *pool = (MemoryPool *)calloc(1, sizeof(MemoryPool));
    if (!pool) {
        return NULL;
    }

    /* Allocate backing memory with page alignment for Metal compatibility. */
    void *base = NULL;
    int rc = posix_memalign(&base, POOL_PAGE_ALIGNMENT, size);
    if (rc != 0 || !base) {
        free(pool);
        return NULL;
    }

    /* Zero-fill for deterministic behavior. */
    memset(base, 0, size);

    pool->base = (uint8_t *)base;
    pool->capacity = size;
    pool->used = 0;
    pool->peak = 0;
    pool->alloc_count = 0;
    pool->type = type;
    pool->ring_head = 0;
    pool->free_list = NULL;

    return pool;
}

void *pool_alloc(MemoryPool *pool, size_t size, size_t alignment) {
    if (!pool || size == 0) {
        return NULL;
    }
    if (!is_power_of_two(alignment) || alignment < POOL_MIN_ALIGNMENT) {
        alignment = POOL_MIN_ALIGNMENT;
    }

    if (pool->type == POOL_WEIGHTS) {
        /* --- POOL_WEIGHTS: Free-list allocator --- */

        /* Check free list for a suitable block first. */
        AllocHeader **prev_ptr = &pool->free_list;
        AllocHeader *block = pool->free_list;
        while (block) {
            if (!block->in_use && block->size >= size) {
                /* Reuse this block. */
                block->in_use = true;
                pool->alloc_count++;
                pool->used += block->size;
                if (pool->used > pool->peak) {
                    pool->peak = pool->used;
                }
                /* Return the aligned data pointer (immediately after header, aligned). */
                uintptr_t header_end = (uintptr_t)block + sizeof(AllocHeader);
                uintptr_t data_ptr = align_up(header_end, alignment);
                return (void *)data_ptr;
            }
            prev_ptr = &block->next;
            block = block->next;
        }
        (void)prev_ptr;

        /* No free block found -- bump allocate from the arena. */
        size_t offset = pool->ring_head; /* ring_head doubles as bump pointer for weights */

        /* Space needed: header + alignment padding + data */
        size_t header_space = align_up(sizeof(AllocHeader), alignment);
        size_t total_needed = header_space + size;

        if (offset + total_needed > pool->capacity) {
            return NULL; /* out of memory */
        }

        /* Place the header at the current offset. */
        AllocHeader *hdr = (AllocHeader *)(pool->base + offset);
        hdr->size = size;
        hdr->total_size = total_needed;
        hdr->in_use = true;
        hdr->next = NULL;

        /* Link into the free list for later pool_free. */
        if (pool->free_list == NULL) {
            pool->free_list = hdr;
        } else {
            /* Append to end of free list. */
            AllocHeader *tail = pool->free_list;
            while (tail->next) {
                tail = tail->next;
            }
            tail->next = hdr;
        }

        pool->ring_head = offset + total_needed;
        pool->used += size;
        pool->alloc_count++;
        if (pool->used > pool->peak) {
            pool->peak = pool->used;
        }

        uintptr_t data_ptr = (uintptr_t)(pool->base + offset) + header_space;
        return (void *)data_ptr;

    } else {
        /* --- POOL_ACTIVATIONS / POOL_SCRATCH: Bump allocator --- */
        size_t aligned_offset = align_up(pool->ring_head, alignment);
        size_t end = aligned_offset + size;

        if (end > pool->capacity) {
            return NULL; /* out of memory */
        }

        pool->ring_head = end;
        pool->used = end;
        pool->alloc_count++;
        if (pool->used > pool->peak) {
            pool->peak = pool->used;
        }

        return (void *)(pool->base + aligned_offset);
    }
}

void pool_free(MemoryPool *pool, void *ptr) {
    if (!pool || !ptr) {
        return;
    }

    if (pool->type == POOL_WEIGHTS) {
        /* Walk the free list to find the block owning this pointer. */
        AllocHeader *block = pool->free_list;
        while (block) {
            uintptr_t block_start = (uintptr_t)block;
            uintptr_t block_end = block_start + block->total_size;
            uintptr_t p = (uintptr_t)ptr;

            if (p >= block_start && p < block_end && block->in_use) {
                block->in_use = false;
                pool->alloc_count--;
                if (pool->used >= block->size) {
                    pool->used -= block->size;
                }
                return;
            }
            block = block->next;
        }
        /* Pointer not found -- silent no-op (defensive). */
    }
    /* For ACTIVATIONS and SCRATCH, free is a no-op. Use pool_reset(). */
}

void pool_reset(MemoryPool *pool) {
    if (!pool) {
        return;
    }

    switch (pool->type) {
    case POOL_ACTIVATIONS:
    case POOL_SCRATCH:
        pool->ring_head = 0;
        pool->used = 0;
        pool->alloc_count = 0;
        break;
    case POOL_WEIGHTS:
        /* Weights are static -- reset is a no-op. */
        break;
    }
}

size_t pool_used(MemoryPool *pool) {
    if (!pool) {
        return 0;
    }
    return pool->used;
}

size_t pool_peak(MemoryPool *pool) {
    if (!pool) {
        return 0;
    }
    return pool->peak;
}

void pool_destroy(MemoryPool *pool) {
    if (!pool) {
        return;
    }
    free(pool->base);
    pool->base = NULL;
    pool->capacity = 0;
    pool->used = 0;
    free(pool);
}
