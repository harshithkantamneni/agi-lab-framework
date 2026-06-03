/* ops.h -- Tensor operations (CPU implementations).
 *
 * All operations follow the pattern: op_*(Tensor *out, const Tensor *inputs...).
 * The caller pre-allocates `out` with the correct shape and dtype.
 *
 * FP32 matmul uses Accelerate vDSP/cblas for baseline performance.
 * INT4 quantized matmul dequantizes on the fly.
 *
 * Metal GPU kernels will replace these in the future (see tensor_metal_spec.md).
 */

#ifndef OPS_H
#define OPS_H

#include "tensor.h"

/* ---- Element-wise operations ---- */

/* out[i] = a[i] + b[i].  a, b, out must have the same shape and dtype. */
void op_add(Tensor *out, const Tensor *a, const Tensor *b);

/* out[i] = a[i] * b[i].  (Hadamard product.) */
void op_mul(Tensor *out, const Tensor *a, const Tensor *b);

/* out[i] = a[i] * s. */
void op_scale(Tensor *out, const Tensor *a, float s);

/* out[i] = max(0, a[i]). */
void op_relu(Tensor *out, const Tensor *a);

/* SwiGLU: out[i] = gate[i] * silu(gate[i]) * up[i]
 * where silu(x) = x * sigmoid(x).
 * Actually the standard SwiGLU is: out = silu(gate) * up. */
void op_swiglu(Tensor *out, const Tensor *gate, const Tensor *up);

/* ---- Reduction operations ---- */

/* Softmax along dimension `dim`: out[i] = exp(a[i]) / sum(exp(a[j])). */
void op_softmax(Tensor *out, const Tensor *a, int dim);

/* Sum all elements, returns scalar. */
float op_sum(const Tensor *a);

/* Max of all elements, returns scalar. */
float op_max(const Tensor *a);

/* ---- BLAS operations ---- */

/* Matrix multiply: out = a @ b.
 * a: (M, K), b: (K, N), out: (M, N).
 * All FP32.  Uses Accelerate cblas_sgemm. */
void op_matmul(Tensor *out, const Tensor *a, const Tensor *b);

/* Quantized matmul: out = a_fp16 @ dequant(b_int4).
 * a: (M, K) FP16, b: (K, N) INT4, out: (M, N) FP16.
 * Dequantizes b on the fly. */
void op_matmul_q4(Tensor *out, const Tensor *a_fp16, const Tensor *b_int4);

/* ---- Normalization ---- */

/* RMS Norm: out = a / rms(a) * weight, where rms = sqrt(mean(a^2) + eps). */
void op_rms_norm(Tensor *out, const Tensor *a, const Tensor *weight, float eps);

/* ---- Training operations ---- */

/* out[i] = a[i] - b[i]. */
void op_sub(Tensor *out, const Tensor *a, const Tensor *b);

/* out[i] = a[i] + b[i] * s.  (Fused add-scale, useful for gradient updates.) */
void op_add_scaled(Tensor *out, const Tensor *a, const Tensor *b, float s);

/* Matrix multiply with transposed first argument: out = a^T @ b.
 * a: (K, M), b: (K, N), out: (M, N). Uses Accelerate cblas_sgemm. */
void op_matmul_tn(Tensor *out, const Tensor *a, const Tensor *b);

/* Matrix multiply with transposed second argument: out = a @ b^T.
 * a: (M, K), b: (N, K), out: (M, N). Uses Accelerate cblas_sgemm. */
void op_matmul_nt(Tensor *out, const Tensor *a, const Tensor *b);

/* SiLU (Swish) activation: out[i] = a[i] * sigmoid(a[i]).
 * Used in SwiGLU forward/backward. */
void op_silu(Tensor *out, const Tensor *a);

/* SiLU derivative: out[i] = silu'(a[i]) = sigmoid(a[i]) * (1 + a[i] * (1 - sigmoid(a[i]))).
 * Needed for SwiGLU backward pass. */
void op_silu_backward(Tensor *out, const Tensor *a);

/* Cross-entropy loss for language modeling.
 * logits: [seq_len, vocab_size], targets: array of seq_len token IDs.
 * Returns average cross-entropy loss over the sequence.
 * If grad_logits is non-NULL, writes dL/d(logits) into it (same shape as logits). */
float op_cross_entropy(const Tensor *logits, const int32_t *targets, int32_t seq_len,
                       Tensor *grad_logits);

/* Fill tensor with random uniform values in [lo, hi]. */
void op_rand_uniform(Tensor *out, float lo, float hi);

/* Fill tensor with Xavier (Glorot) uniform initialization.
 * Assumes 2D weight [fan_out, fan_in] or [fan_in, fan_out].
 * Scale = sqrt(6 / (fan_in + fan_out)). */
void op_init_xavier(Tensor *out, int32_t fan_in, int32_t fan_out);

/* ---- Quantization ---- */

/* Dequantize INT4 packed tensor to FP32 (or FP16).
 * INT4 values are in range [-8, 7], uniformly mapped. */
void op_dequantize(Tensor *out, const Tensor *in_int4);

/* Quantize FP32 tensor to INT4 packed format.
 * Applies per-tensor min/max scaling. */
void op_quantize_int4(Tensor *out_int4, const Tensor *in_fp32);

/* ---- QAT-aware matmul wrappers (Program 3 Phase 7, D-613) ---- */

/* Forward declaration; qat_context.h is in src/training/ which is already
 * in the include path via -Isrc/training. */
struct QATContext;

/* Matrix multiply with optional fake-quantization of W.
 * out = in @ W  (or in @ w_hat if qat is non-NULL and enabled).
 * If qat == NULL, identical to op_matmul(out, in, W).
 * in: (M, K), W: (K, N), out: (M, N). All FP32. */
void op_matmul_qat(Tensor *out, const Tensor *in, const Tensor *W,
                   struct QATContext *qat);

/* Matrix multiply-NT with optional fake-quantization of W.
 * out = in @ W^T  (or in @ w_hat^T if qat is non-NULL and enabled).
 * If qat == NULL, identical to op_matmul_nt(out, in, W).
 * in: (M, K), W: (N, K), out: (M, N). All FP32. */
void op_matmul_nt_qat(Tensor *out, const Tensor *in, const Tensor *W,
                      struct QATContext *qat);

/* ---- GPU acceleration ---- */

/* Register a GPU matmul implementation for large matrices.
 * fn: function with signature (const float *A, const float *B, float *C, int M, int N, int K)
 * Passing NULL disables GPU acceleration.
 * When set, op_matmul and op_matmul_nt automatically dispatch to GPU
 * for matrices exceeding the size threshold (M*N*K > 16,777,216). */
void op_set_gpu_matmul(void (*fn)(const float *, const float *, float *, int, int, int));

/* ---- INT4 packing helpers ---- */

/* Pack two 4-bit signed values into one byte. */
static inline uint8_t int4_pack(int8_t lo, int8_t hi) {
    return (uint8_t)(((uint8_t)(hi & 0x0F) << 4) | ((uint8_t)(lo & 0x0F)));
}

/* Unpack low nibble (sign-extended). */
static inline int8_t int4_unpack_lo(uint8_t packed) {
    int8_t val = (int8_t)(packed & 0x0F);
    if (val & 0x08) val |= (int8_t)0xF0; /* sign extend */
    return val;
}

/* Unpack high nibble (sign-extended). */
static inline int8_t int4_unpack_hi(uint8_t packed) {
    int8_t val = (int8_t)((packed >> 4) & 0x0F);
    if (val & 0x08) val |= (int8_t)0xF0; /* sign extend */
    return val;
}

#endif /* OPS_H */
