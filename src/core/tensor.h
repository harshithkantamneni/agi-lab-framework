/* tensor.h -- N-dimensional tensor with dtype-aware storage.
 *
 * Supports up to 4 dimensions (batch, seq, heads, features).
 * Data types: FP32, FP16, INT8, INT4 (packed: 2 values per byte).
 *
 * Views: no-copy slicing via stride manipulation. A view shares
 * the parent's data pointer and pool -- destroying a view does NOT
 * free the underlying data.
 *
 * Ownership: tensors allocated via tensor_create own their data
 * (allocated from the given pool). Views do not own data.
 */

#ifndef TENSOR_H
#define TENSOR_H

#include "memory_pool.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* ---- Data types ---- */
typedef enum {
    DTYPE_FP32 = 0, /* 32-bit float, 4 bytes/element */
    DTYPE_FP16 = 1, /* 16-bit float (_Float16), 2 bytes/element */
    DTYPE_INT8 = 2, /* 8-bit signed integer, 1 byte/element */
    DTYPE_INT4 = 3  /* 4-bit signed integer, packed 2 per byte */
} DType;

/* ---- Tensor ---- */
typedef struct {
    void *data;          /* pointer into pool (or parent tensor for views) */
    int32_t shape[4];    /* dimensions (unused dims are 1) */
    int32_t stride[4];   /* element strides per dimension */
    int32_t ndim;        /* number of active dimensions (1-4) */
    DType dtype;         /* element type */
    MemoryPool *pool;    /* owning pool (NULL for views) */
    bool is_view;        /* true if this is a view (data not owned) */
} Tensor;

/* ---- API ---- */

/* Create a new tensor, allocating storage from `pool`.
 * shape: array of ndim dimension sizes.
 * ndim: 1-4.
 * Returns NULL on failure. */
Tensor *tensor_create(MemoryPool *pool, const int32_t *shape, int32_t ndim, DType dtype);

/* Create a view (no-copy slice) along `dim`, from `start` to `end` (exclusive).
 * The returned tensor shares data with `src`.
 * Returns NULL on invalid arguments. */
Tensor *tensor_view(Tensor *src, int32_t dim, int32_t start, int32_t end);

/* Fill all elements with the given value (cast to tensor's dtype). */
void tensor_fill(Tensor *t, float value);

/* Get a single element, returning it as float.
 * indices: array of ndim indices. */
float tensor_get(const Tensor *t, const int32_t *indices);

/* Set a single element from a float value.
 * indices: array of ndim indices. */
void tensor_set(Tensor *t, const int32_t *indices, float value);

/* Total storage in bytes (accounts for INT4 packing). */
size_t tensor_nbytes(const Tensor *t);

/* Number of elements in the tensor. */
size_t tensor_numel(const Tensor *t);

/* Check if memory layout is contiguous (no gaps between elements). */
bool tensor_is_contiguous(const Tensor *t);

/* Destroy the tensor struct. If is_view=false and pool!=NULL,
 * the data is freed back to the pool. */
void tensor_destroy(Tensor *t);

/* ---- Dtype helpers ---- */

/* Bytes per element for the given dtype (INT4 returns 0 -- use tensor_nbytes). */
size_t dtype_size(DType dtype);

/* Human-readable name. */
const char *dtype_name(DType dtype);

#endif /* TENSOR_H */
