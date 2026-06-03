/* quantize.h -- Group quantization for INT4/INT8 weight storage.
 *
 * Group quantization: each group of `group_size` consecutive elements
 * shares one (scale, zero_point) pair stored as FP32.
 *
 * This is ASYMMETRIC (unsigned) quantization for group-based schemes:
 *   INT4: maps to [0, 15], scale = (max - min) / 15, zero_point = min
 *   INT8: maps to [0, 255], scale = (max - min) / 255, zero_point = min
 *
 * For INT4 with group_size=128:
 *   - Each group = 128 elements = 64 bytes (packed) + 8 bytes (scale+zp) = 72 bytes
 *   - Overhead: 8/64 = 12.5% vs raw INT4
 *   - For 16B params: 7.45 GB raw + 0.93 GB groups = 8.38 GB total
 *
 * Note: The existing ops.h op_quantize_int4/op_dequantize use per-tensor
 * symmetric signed INT4 [-8, 7]. This module uses per-group asymmetric
 * unsigned INT4 [0, 15] for better range utilization in weight quantization.
 *
 * The INT4 packing format is the same as tensor.h: two 4-bit values per byte,
 * low nibble = even index, high nibble = odd index. But values are unsigned [0,15].
 */

#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "tensor.h"

/* Per-group quantization metadata. */
typedef struct {
    float scale;      /* dequant: real = quantized * scale + zero_point */
    float zero_point; /* offset (equal to group minimum for asymmetric) */
} QuantGroup;

/* Quantize FP32 tensor to INT4 with group quantization.
 * dst must be pre-allocated with correct INT4 shape (same shape as src).
 * groups: output array of QuantGroup, caller allocates (num_groups entries).
 * group_size: typically 128. Last group may be smaller if numel % group_size != 0. */
void quantize_fp32_to_int4(Tensor *dst, const Tensor *src,
                           QuantGroup *groups, int32_t group_size);

/* Dequantize INT4 tensor to FP32 using group quantization params.
 * dst must be pre-allocated with correct FP32 shape (same shape as src). */
void dequantize_int4_to_fp32(Tensor *dst, const Tensor *src,
                              const QuantGroup *groups, int32_t group_size);

/* Quantize FP32 to INT8 with group quantization. */
void quantize_fp32_to_int8(Tensor *dst, const Tensor *src,
                           QuantGroup *groups, int32_t group_size);

/* Dequantize INT8 to FP32. */
void dequantize_int8_to_fp32(Tensor *dst, const Tensor *src,
                              const QuantGroup *groups, int32_t group_size);

/* Compute number of groups needed for a tensor with given group_size.
 * Returns ceil(numel / group_size). */
int32_t quantize_num_groups(const Tensor *t, int32_t group_size);

/* Compute quantization error: MSE between original and reconstructed FP32 tensors. */
float quantize_error_mse(const Tensor *original_fp32,
                         const Tensor *reconstructed_fp32);

#endif /* QUANTIZE_H */
