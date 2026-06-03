/* tensor.c -- N-dimensional tensor with dtype-aware storage.
 *
 * Implementation details:
 *   - The Tensor struct is malloc'd (small metadata, ~64 bytes).
 *   - Tensor data is allocated from the given MemoryPool via pool_alloc.
 *   - Views share the parent's data pointer; destroying a view does not free data.
 *   - Strides are in ELEMENTS (not bytes).
 *   - INT4 packing: 2 values per byte. Element access uses linear offset
 *     computed from strides, then maps to byte_index = offset/2, nibble = offset%2.
 *   - Default layout is row-major: stride[ndim-1] = 1, stride[i] = stride[i+1] * shape[i+1].
 */

#include "tensor.h"

#include <stdlib.h>
#include <string.h>

/* ---- INT4 packing helpers (duplicated from ops.h to avoid circular dep) ---- */

static inline int8_t int4_unpack_lo_(uint8_t packed) {
    int8_t val = (int8_t)(packed & 0x0F);
    if (val & 0x08) val |= (int8_t)0xF0;
    return val;
}

static inline int8_t int4_unpack_hi_(uint8_t packed) {
    int8_t val = (int8_t)((packed >> 4) & 0x0F);
    if (val & 0x08) val |= (int8_t)0xF0;
    return val;
}

/* ---- dtype helpers ---- */

size_t dtype_size(DType dtype) {
    switch (dtype) {
    case DTYPE_FP32: return 4;
    case DTYPE_FP16: return 2;
    case DTYPE_INT8: return 1;
    case DTYPE_INT4: return 0; /* packed -- use tensor_nbytes() instead */
    }
    return 0;
}

const char *dtype_name(DType dtype) {
    switch (dtype) {
    case DTYPE_FP32: return "fp32";
    case DTYPE_FP16: return "fp16";
    case DTYPE_INT8: return "int8";
    case DTYPE_INT4: return "int4";
    }
    return "unknown";
}

/* ---- internal: compute linear flat offset from indices and strides ---- */

static inline size_t flat_offset(const Tensor *t, const int32_t *indices) {
    size_t off = 0;
    for (int32_t d = 0; d < t->ndim; d++) {
        off += (size_t)indices[d] * (size_t)t->stride[d];
    }
    return off;
}

/* ---- API implementation ---- */

Tensor *tensor_create(MemoryPool *pool, const int32_t *shape, int32_t ndim, DType dtype) {
    if (!pool || !shape || ndim < 1 || ndim > 4) {
        return NULL;
    }

    /* Validate shape dimensions are positive */
    for (int32_t d = 0; d < ndim; d++) {
        if (shape[d] <= 0) {
            return NULL;
        }
    }

    /* Allocate the struct on the heap (small metadata) */
    Tensor *t = (Tensor *)calloc(1, sizeof(Tensor));
    if (!t) {
        return NULL;
    }

    t->ndim = ndim;
    t->dtype = dtype;
    t->pool = pool;
    t->is_view = false;

    /* Copy shape, set unused dims to 1 */
    for (int32_t d = 0; d < 4; d++) {
        t->shape[d] = (d < ndim) ? shape[d] : 1;
    }

    /* Compute row-major strides (in elements).
     * stride[ndim-1] = 1, stride[i] = stride[i+1] * shape[i+1] */
    for (int32_t d = 0; d < 4; d++) {
        t->stride[d] = 0;
    }
    t->stride[ndim - 1] = 1;
    for (int32_t d = ndim - 2; d >= 0; d--) {
        t->stride[d] = t->stride[d + 1] * t->shape[d + 1];
    }

    /* Compute storage size in bytes */
    size_t numel = tensor_numel(t);
    size_t nbytes;
    if (dtype == DTYPE_INT4) {
        /* 2 values per byte, round up */
        nbytes = (numel + 1) / 2;
    } else {
        nbytes = numel * dtype_size(dtype);
    }

    /* Allocate data from pool with 16-byte alignment (NEON SIMD) */
    t->data = pool_alloc(pool, nbytes, POOL_MIN_ALIGNMENT);
    if (!t->data) {
        free(t);
        return NULL;
    }

    /* Zero-initialize the data for deterministic behavior */
    memset(t->data, 0, nbytes);

    return t;
}

Tensor *tensor_view(Tensor *src, int32_t dim, int32_t start, int32_t end) {
    if (!src || dim < 0 || dim >= src->ndim) {
        return NULL;
    }
    if (start < 0 || end <= start || end > src->shape[dim]) {
        return NULL;
    }

    Tensor *v = (Tensor *)calloc(1, sizeof(Tensor));
    if (!v) {
        return NULL;
    }

    v->ndim = src->ndim;
    v->dtype = src->dtype;
    v->pool = NULL; /* views do not own data */
    v->is_view = true;

    /* Copy shape, adjusting the sliced dimension */
    for (int32_t d = 0; d < 4; d++) {
        v->shape[d] = src->shape[d];
        v->stride[d] = src->stride[d];
    }
    v->shape[dim] = end - start;

    /* Compute the data pointer offset.
     * The view starts at src->data + start * src->stride[dim] elements.
     * For non-INT4 types, offset in bytes = start * stride[dim] * dtype_size.
     * For INT4, the offset is in nibbles, so we keep it in element units
     * and handle it in get/set via the stride mechanism. */
    if (src->dtype == DTYPE_INT4) {
        /* For INT4, we cannot do sub-byte pointer arithmetic directly.
         * We offset the data pointer by (start * stride[dim]) / 2 bytes,
         * but only if the offset is byte-aligned (even number of elements).
         * For odd offsets, we would need a nibble offset field, but the
         * current struct doesn't have one. For now, we require INT4 views
         * to be on even element boundaries for the sliced dimension. */
        size_t elem_offset = (size_t)start * (size_t)src->stride[dim];
        /* Offset the data pointer by whole bytes */
        v->data = (uint8_t *)src->data + (elem_offset / 2);
    } else {
        size_t byte_offset = (size_t)start * (size_t)src->stride[dim] * dtype_size(src->dtype);
        v->data = (uint8_t *)src->data + byte_offset;
    }

    return v;
}

void tensor_fill(Tensor *t, float value) {
    if (!t || !t->data) {
        return;
    }

    size_t numel = tensor_numel(t);

    switch (t->dtype) {
    case DTYPE_FP32: {
        float *data = (float *)t->data;
        /* For contiguous tensors, fill directly */
        if (tensor_is_contiguous(t)) {
            for (size_t i = 0; i < numel; i++) {
                data[i] = value;
            }
        } else {
            /* Non-contiguous: use indices */
            for (size_t i = 0; i < numel; i++) {
                /* Decompose linear index into multi-dim indices */
                int32_t indices[4];
                size_t rem = i;
                for (int32_t d = 0; d < t->ndim; d++) {
                    size_t dim_size = 1;
                    for (int32_t dd = d + 1; dd < t->ndim; dd++) {
                        dim_size *= (size_t)t->shape[dd];
                    }
                    indices[d] = (int32_t)(rem / dim_size);
                    rem %= dim_size;
                }
                tensor_set(t, indices, value);
            }
        }
        break;
    }
    case DTYPE_FP16: {
        _Float16 *data = (_Float16 *)t->data;
        _Float16 val16 = (_Float16)value;
        if (tensor_is_contiguous(t)) {
            for (size_t i = 0; i < numel; i++) {
                data[i] = val16;
            }
        } else {
            for (size_t i = 0; i < numel; i++) {
                int32_t indices[4];
                size_t rem = i;
                for (int32_t d = 0; d < t->ndim; d++) {
                    size_t dim_size = 1;
                    for (int32_t dd = d + 1; dd < t->ndim; dd++) {
                        dim_size *= (size_t)t->shape[dd];
                    }
                    indices[d] = (int32_t)(rem / dim_size);
                    rem %= dim_size;
                }
                tensor_set(t, indices, value);
            }
        }
        break;
    }
    case DTYPE_INT8: {
        int8_t *data = (int8_t *)t->data;
        int8_t val8 = (int8_t)value;
        if (tensor_is_contiguous(t)) {
            for (size_t i = 0; i < numel; i++) {
                data[i] = val8;
            }
        } else {
            for (size_t i = 0; i < numel; i++) {
                int32_t indices[4];
                size_t rem = i;
                for (int32_t d = 0; d < t->ndim; d++) {
                    size_t dim_size = 1;
                    for (int32_t dd = d + 1; dd < t->ndim; dd++) {
                        dim_size *= (size_t)t->shape[dd];
                    }
                    indices[d] = (int32_t)(rem / dim_size);
                    rem %= dim_size;
                }
                tensor_set(t, indices, value);
            }
        }
        break;
    }
    case DTYPE_INT4: {
        /* INT4 packing: use set for each element */
        for (size_t i = 0; i < numel; i++) {
            int32_t indices[4];
            size_t rem = i;
            for (int32_t d = 0; d < t->ndim; d++) {
                size_t dim_size = 1;
                for (int32_t dd = d + 1; dd < t->ndim; dd++) {
                    dim_size *= (size_t)t->shape[dd];
                }
                indices[d] = (int32_t)(rem / dim_size);
                rem %= dim_size;
            }
            tensor_set(t, indices, value);
        }
        break;
    }
    }
}

float tensor_get(const Tensor *t, const int32_t *indices) {
    if (!t || !t->data || !indices) {
        return 0.0f;
    }

    size_t off = flat_offset(t, indices);

    switch (t->dtype) {
    case DTYPE_FP32: {
        const float *data = (const float *)t->data;
        return data[off];
    }
    case DTYPE_FP16: {
        const _Float16 *data = (const _Float16 *)t->data;
        return (float)data[off];
    }
    case DTYPE_INT8: {
        const int8_t *data = (const int8_t *)t->data;
        return (float)data[off];
    }
    case DTYPE_INT4: {
        /* INT4: 2 values packed per byte.
         * Byte index = off / 2, nibble position = off % 2.
         * Low nibble (offset even) = bits[3:0], high nibble (offset odd) = bits[7:4]. */
        const uint8_t *data = (const uint8_t *)t->data;
        size_t byte_idx = off / 2;
        uint8_t packed = data[byte_idx];
        if (off % 2 == 0) {
            return (float)int4_unpack_lo_(packed);
        } else {
            return (float)int4_unpack_hi_(packed);
        }
    }
    }
    return 0.0f;
}

void tensor_set(Tensor *t, const int32_t *indices, float value) {
    if (!t || !t->data || !indices) {
        return;
    }

    size_t off = flat_offset(t, indices);

    switch (t->dtype) {
    case DTYPE_FP32: {
        float *data = (float *)t->data;
        data[off] = value;
        break;
    }
    case DTYPE_FP16: {
        _Float16 *data = (_Float16 *)t->data;
        data[off] = (_Float16)value;
        break;
    }
    case DTYPE_INT8: {
        int8_t *data = (int8_t *)t->data;
        data[off] = (int8_t)value;
        break;
    }
    case DTYPE_INT4: {
        /* Clamp to [-8, 7] and pack into nibble */
        int8_t val = (int8_t)value;
        if (val < -8) val = -8;
        if (val > 7) val = 7;

        uint8_t *data = (uint8_t *)t->data;
        size_t byte_idx = off / 2;
        uint8_t packed = data[byte_idx];

        if (off % 2 == 0) {
            /* Replace low nibble, keep high */
            packed = (packed & 0xF0) | ((uint8_t)(val & 0x0F));
        } else {
            /* Replace high nibble, keep low */
            packed = (packed & 0x0F) | ((uint8_t)((val & 0x0F) << 4));
        }
        data[byte_idx] = packed;
        break;
    }
    }
}

size_t tensor_nbytes(const Tensor *t) {
    if (!t) {
        return 0;
    }
    size_t numel = tensor_numel(t);
    if (t->dtype == DTYPE_INT4) {
        /* 2 values per byte, round up for odd counts */
        return (numel + 1) / 2;
    }
    return numel * dtype_size(t->dtype);
}

size_t tensor_numel(const Tensor *t) {
    if (!t) {
        return 0;
    }
    size_t n = 1;
    for (int32_t d = 0; d < t->ndim; d++) {
        n *= (size_t)t->shape[d];
    }
    return n;
}

bool tensor_is_contiguous(const Tensor *t) {
    if (!t) {
        return false;
    }

    /* Row-major contiguous check:
     * stride[ndim-1] must be 1, and stride[d] must equal stride[d+1] * shape[d+1]. */
    if (t->stride[t->ndim - 1] != 1) {
        return false;
    }
    for (int32_t d = t->ndim - 2; d >= 0; d--) {
        if (t->stride[d] != t->stride[d + 1] * t->shape[d + 1]) {
            return false;
        }
    }
    return true;
}

void tensor_destroy(Tensor *t) {
    if (!t) {
        return;
    }

    /* If this tensor owns its data (not a view), free data back to pool.
     * pool_free is only meaningful for POOL_WEIGHTS; for other pool types,
     * data is reclaimed via pool_reset, so pool_free is a no-op. */
    if (!t->is_view && t->pool && t->data) {
        pool_free(t->pool, t->data);
    }

    /* Free the struct itself (always malloc'd) */
    free(t);
}
