/* quantize.c -- Group quantization for INT4/INT8 weight storage.
 *
 * Implements asymmetric group quantization:
 *   - INT4 unsigned [0, 15]: scale = (max - min) / 15, zero_point = min
 *   - INT8 unsigned [0, 255]: scale = (max - min) / 255, zero_point = min
 *
 * The quantized value for element x is:
 *   q = clamp(round((x - zero_point) / scale), 0, max_level)
 *
 * Dequantization:
 *   x_hat = q * scale + zero_point
 *
 * INT4 packing: two unsigned 4-bit values per byte.
 *   byte = (q[odd] << 4) | q[even]
 * This matches the packing layout in tensor.h but values are unsigned [0,15]
 * rather than signed [-8,7].
 */

#include "quantize.h"

#include <math.h>
#include <string.h>

/* ---- Internal helpers ---- */

/* Decompose linear index into multi-dim indices (same as ops.c). */
static void q_linear_to_indices(const Tensor *t, size_t linear, int32_t *indices) {
    size_t rem = linear;
    for (int32_t d = 0; d < t->ndim; d++) {
        size_t below = 1;
        for (int32_t dd = d + 1; dd < t->ndim; dd++) {
            below *= (size_t)t->shape[dd];
        }
        indices[d] = (int32_t)(rem / below);
        rem %= below;
    }
}

/* Read element i (linear index) from a FP32 tensor. */
static float read_fp32(const Tensor *t, size_t i) {
    int32_t indices[4] = {0, 0, 0, 0};
    q_linear_to_indices(t, i, indices);
    return tensor_get(t, indices);
}

/* Write element i (linear index) to a FP32 tensor. */
static void write_fp32(Tensor *t, size_t i, float val) {
    int32_t indices[4] = {0, 0, 0, 0};
    q_linear_to_indices(t, i, indices);
    tensor_set(t, indices, val);
}

/* ---- INT4 group quantization ---- */

/* Pack an unsigned 4-bit value (0-15) into the appropriate nibble of a byte
 * in the INT4 tensor's data buffer. elem_idx is the linear element index. */
static void int4_pack_unsigned(Tensor *dst, size_t elem_idx, uint8_t val) {
    uint8_t *data = (uint8_t *)dst->data;
    size_t byte_idx = elem_idx / 2;
    uint8_t packed = data[byte_idx];

    if (elem_idx % 2 == 0) {
        /* Low nibble */
        packed = (packed & 0xF0) | (val & 0x0F);
    } else {
        /* High nibble */
        packed = (packed & 0x0F) | ((val & 0x0F) << 4);
    }
    data[byte_idx] = packed;
}

/* Unpack an unsigned 4-bit value (0-15) from the INT4 tensor at linear index. */
static uint8_t int4_unpack_unsigned(const Tensor *src, size_t elem_idx) {
    const uint8_t *data = (const uint8_t *)src->data;
    size_t byte_idx = elem_idx / 2;
    uint8_t packed = data[byte_idx];

    if (elem_idx % 2 == 0) {
        return packed & 0x0F;
    } else {
        return (packed >> 4) & 0x0F;
    }
}

void quantize_fp32_to_int4(Tensor *dst, const Tensor *src,
                           QuantGroup *groups, int32_t group_size) {
    if (!dst || !src || !groups || group_size <= 0) {
        return;
    }

    size_t numel = tensor_numel(src);
    int32_t num_groups = quantize_num_groups(src, group_size);

    for (int32_t g = 0; g < num_groups; g++) {
        size_t start = (size_t)g * (size_t)group_size;
        size_t end = start + (size_t)group_size;
        if (end > numel) {
            end = numel;
        }

        /* Find min and max in this group */
        float min_val = read_fp32(src, start);
        float max_val = min_val;
        for (size_t i = start + 1; i < end; i++) {
            float val = read_fp32(src, i);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        /* Compute scale and zero_point */
        float range = max_val - min_val;
        float scale;
        if (range < 1e-10f) {
            /* Constant group: scale is 0, all values map to 0.
             * Dequantize will return zero_point for all elements. */
            scale = 0.0f;
        } else {
            scale = range / 15.0f;
        }
        float zero_point = min_val;

        groups[g].scale = scale;
        groups[g].zero_point = zero_point;

        /* Quantize each element in the group */
        for (size_t i = start; i < end; i++) {
            float val = read_fp32(src, i);
            uint8_t q;
            if (scale < 1e-10f) {
                /* Constant group */
                q = 0;
            } else {
                float scaled = (val - zero_point) / scale;
                int32_t rounded = (int32_t)roundf(scaled);
                if (rounded < 0) rounded = 0;
                if (rounded > 15) rounded = 15;
                q = (uint8_t)rounded;
            }
            int4_pack_unsigned(dst, i, q);
        }
    }
}

void dequantize_int4_to_fp32(Tensor *dst, const Tensor *src,
                              const QuantGroup *groups, int32_t group_size) {
    if (!dst || !src || !groups || group_size <= 0) {
        return;
    }

    size_t numel = tensor_numel(src);
    int32_t num_groups = quantize_num_groups(src, group_size);

    for (int32_t g = 0; g < num_groups; g++) {
        size_t start = (size_t)g * (size_t)group_size;
        size_t end = start + (size_t)group_size;
        if (end > numel) {
            end = numel;
        }

        float scale = groups[g].scale;
        float zero_point = groups[g].zero_point;

        for (size_t i = start; i < end; i++) {
            uint8_t q = int4_unpack_unsigned(src, i);
            float val = (float)q * scale + zero_point;
            write_fp32(dst, i, val);
        }
    }
}

/* ---- INT8 group quantization ---- */

void quantize_fp32_to_int8(Tensor *dst, const Tensor *src,
                           QuantGroup *groups, int32_t group_size) {
    if (!dst || !src || !groups || group_size <= 0) {
        return;
    }

    size_t numel = tensor_numel(src);
    int32_t num_groups = quantize_num_groups(src, group_size);

    for (int32_t g = 0; g < num_groups; g++) {
        size_t start = (size_t)g * (size_t)group_size;
        size_t end = start + (size_t)group_size;
        if (end > numel) {
            end = numel;
        }

        /* Find min and max in this group */
        float min_val = read_fp32(src, start);
        float max_val = min_val;
        for (size_t i = start + 1; i < end; i++) {
            float val = read_fp32(src, i);
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }

        /* Compute scale and zero_point */
        float range = max_val - min_val;
        float scale;
        if (range < 1e-10f) {
            scale = 0.0f;
        } else {
            scale = range / 255.0f;
        }
        float zero_point = min_val;

        groups[g].scale = scale;
        groups[g].zero_point = zero_point;

        /* Quantize each element.
         * INT8 uses the tensor's native int8 storage via tensor_set,
         * but we store unsigned [0,255] mapped through the signed int8 type.
         * We write raw bytes directly since tensor_set treats int8 as signed. */
        int8_t *data = (int8_t *)dst->data;
        /* Compute the linear offset for contiguous layout */
        for (size_t i = start; i < end; i++) {
            float val = read_fp32(src, i);
            uint8_t q;
            if (scale < 1e-10f) {
                q = 0;
            } else {
                float scaled = (val - zero_point) / scale;
                int32_t rounded = (int32_t)roundf(scaled);
                if (rounded < 0) rounded = 0;
                if (rounded > 255) rounded = 255;
                q = (uint8_t)rounded;
            }
            /* Store as raw byte. The int8_t storage is reinterpreted as uint8_t. */
            data[i] = (int8_t)q;
        }
    }
}

void dequantize_int8_to_fp32(Tensor *dst, const Tensor *src,
                              const QuantGroup *groups, int32_t group_size) {
    if (!dst || !src || !groups || group_size <= 0) {
        return;
    }

    size_t numel = tensor_numel(src);
    int32_t num_groups = quantize_num_groups(src, group_size);

    for (int32_t g = 0; g < num_groups; g++) {
        size_t start = (size_t)g * (size_t)group_size;
        size_t end = start + (size_t)group_size;
        if (end > numel) {
            end = numel;
        }

        float scale = groups[g].scale;
        float zero_point = groups[g].zero_point;

        const int8_t *data = (const int8_t *)src->data;
        for (size_t i = start; i < end; i++) {
            /* Read as unsigned byte */
            uint8_t q = (uint8_t)data[i];
            float val = (float)q * scale + zero_point;
            write_fp32(dst, i, val);
        }
    }
}

/* ---- Utility functions ---- */

int32_t quantize_num_groups(const Tensor *t, int32_t group_size) {
    if (!t || group_size <= 0) {
        return 0;
    }
    size_t numel = tensor_numel(t);
    /* ceil(numel / group_size) */
    return (int32_t)((numel + (size_t)group_size - 1) / (size_t)group_size);
}

float quantize_error_mse(const Tensor *original_fp32,
                         const Tensor *reconstructed_fp32) {
    if (!original_fp32 || !reconstructed_fp32) {
        return 0.0f;
    }

    size_t n = tensor_numel(original_fp32);
    if (n == 0) {
        return 0.0f;
    }

    float sum_sq_err = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float orig = read_fp32(original_fp32, i);
        float recon = read_fp32(reconstructed_fp32, i);
        float diff = orig - recon;
        sum_sq_err += diff * diff;
    }

    return sum_sq_err / (float)n;
}
