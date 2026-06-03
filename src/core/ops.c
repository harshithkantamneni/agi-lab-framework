/* ops.c -- Tensor operations (CPU implementations).
 *
 * All operations follow: op_*(Tensor *out, const Tensor *inputs...).
 * Caller pre-allocates `out` with correct shape and dtype.
 *
 * FP32 matmul uses Accelerate cblas_sgemm for baseline performance.
 * INT4 quantized matmul dequantizes on the fly.
 */

#include "ops.h"
#include "qat_context.h"

#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ---- GPU matmul dispatch ---- */
/* Function pointer for GPU-accelerated matmul. Set via op_set_gpu_matmul().
 * When NULL, all matmuls use CPU (cblas_sgemm). */
static void (*gpu_matmul_fn)(const float *, const float *, float *,
                             int, int, int) = NULL;

/* Threshold: use GPU when M*N*K exceeds this.
 * Set to INT64_MAX to disable GPU dispatch entirely -- the Metal kernel's
 * per-call buffer allocation overhead makes it 3-100x slower than CPU AMX
 * (Accelerate cblas_sgemm) at all practical sizes.  GPU code path is
 * preserved for future use when the Metal bridge is fixed.
 * Previous value: (int64_t)16777216 (256^3). */
#define GPU_MATMUL_THRESHOLD INT64_MAX

void op_set_gpu_matmul(void (*fn)(const float *, const float *, float *,
                                  int, int, int)) {
    gpu_matmul_fn = fn;
}

/* ---- Internal: decompose linear index to multi-dim indices ---- */

static void linear_to_indices(const Tensor *t, size_t linear, int32_t *indices) {
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

/* ---- Element-wise operations ---- */

void op_add(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) {
        return;
    }
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float va = tensor_get(a, indices);
        float vb = tensor_get(b, indices);
        tensor_set(out, indices, va + vb);
    }
}

void op_mul(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) {
        return;
    }
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float va = tensor_get(a, indices);
        float vb = tensor_get(b, indices);
        tensor_set(out, indices, va * vb);
    }
}

void op_scale(Tensor *out, const Tensor *a, float s) {
    if (!out || !a) {
        return;
    }
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        tensor_set(out, indices, tensor_get(a, indices) * s);
    }
}

void op_relu(Tensor *out, const Tensor *a) {
    if (!out || !a) {
        return;
    }
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float val = tensor_get(a, indices);
        tensor_set(out, indices, val > 0.0f ? val : 0.0f);
    }
}

void op_swiglu(Tensor *out, const Tensor *gate, const Tensor *up) {
    if (!out || !gate || !up) {
        return;
    }
    size_t n = tensor_numel(gate);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(gate, i, indices);
        float g = tensor_get(gate, indices);
        float u = tensor_get(up, indices);
        /* SwiGLU: silu(gate) * up, where silu(x) = x * sigmoid(x) */
        float silu_g = g / (1.0f + expf(-g));
        tensor_set(out, indices, silu_g * u);
    }
}

/* ---- Reduction operations ---- */

void op_softmax(Tensor *out, const Tensor *a, int dim) {
    if (!out || !a) {
        return;
    }

    if (a->ndim == 1) {
        /* Simple 1D case */
        size_t n = (size_t)a->shape[0];

        /* Find max for numerical stability */
        float max_val = -INFINITY;
        for (size_t i = 0; i < n; i++) {
            int32_t idx[] = {(int32_t)i};
            float val = tensor_get(a, idx);
            if (val > max_val) max_val = val;
        }

        /* Compute exp(x - max) and sum */
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            int32_t idx[] = {(int32_t)i};
            float val = expf(tensor_get(a, idx) - max_val);
            tensor_set(out, idx, val);
            sum += val;
        }

        /* Normalize */
        for (size_t i = 0; i < n; i++) {
            int32_t idx[] = {(int32_t)i};
            tensor_set(out, idx, tensor_get(out, idx) / sum);
        }
    } else if (a->ndim == 2 && dim == 1) {
        /* 2D softmax along dim 1 (each row independently) */
        int32_t rows = a->shape[0];
        int32_t cols = a->shape[1];

        for (int32_t r = 0; r < rows; r++) {
            /* Find max in this row */
            float max_val = -INFINITY;
            for (int32_t c = 0; c < cols; c++) {
                int32_t idx[] = {r, c};
                float val = tensor_get(a, idx);
                if (val > max_val) max_val = val;
            }

            /* Compute exp(x - max) and sum */
            float sum = 0.0f;
            for (int32_t c = 0; c < cols; c++) {
                int32_t idx[] = {r, c};
                float val = expf(tensor_get(a, idx) - max_val);
                tensor_set(out, idx, val);
                sum += val;
            }

            /* Normalize */
            for (int32_t c = 0; c < cols; c++) {
                int32_t idx[] = {r, c};
                tensor_set(out, idx, tensor_get(out, idx) / sum);
            }
        }
    }
    /* TODO: generalize to arbitrary ndim and dim */
}

float op_sum(const Tensor *a) {
    if (!a) {
        return 0.0f;
    }
    float sum = 0.0f;
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        sum += tensor_get(a, indices);
    }
    return sum;
}

float op_max(const Tensor *a) {
    if (!a) {
        return -INFINITY;
    }
    float max_val = -INFINITY;
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float val = tensor_get(a, indices);
        if (val > max_val) max_val = val;
    }
    return max_val;
}

/* ---- BLAS operations ---- */

void op_matmul(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) {
        return;
    }

    /* a: (M, K), b: (K, N), out: (M, N) -- all FP32 */
    int32_t M = a->shape[0];
    int32_t K = a->shape[1];
    int32_t N = b->shape[1];

    if (a->dtype == DTYPE_FP32 && b->dtype == DTYPE_FP32 &&
        tensor_is_contiguous(a) && tensor_is_contiguous(b) &&
        tensor_is_contiguous(out)) {
        /* GPU path for large matrices */
        if (gpu_matmul_fn && (int64_t)M * N * K > GPU_MATMUL_THRESHOLD) {
            gpu_matmul_fn((const float *)a->data, (const float *)b->data,
                          (float *)out->data, M, N, K);
        } else {
            /* CPU AMX path via Accelerate cblas_sgemm */
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        M, N, K,
                        1.0f,
                        (const float *)a->data, K,
                        (const float *)b->data, N,
                        0.0f,
                        (float *)out->data, N);
        }
    } else {
        /* Fallback: naive triple-loop for non-contiguous or mixed dtype */
        for (int32_t m = 0; m < M; m++) {
            for (int32_t n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int32_t k = 0; k < K; k++) {
                    int32_t ai[] = {m, k};
                    int32_t bi[] = {k, n};
                    sum += tensor_get(a, ai) * tensor_get(b, bi);
                }
                int32_t oi[] = {m, n};
                tensor_set(out, oi, sum);
            }
        }
    }
}

void op_matmul_q4(Tensor *out, const Tensor *a_fp16, const Tensor *b_int4) {
    if (!out || !a_fp16 || !b_int4) {
        return;
    }

    /* a: (M, K) FP16, b: (K, N) INT4, out: (M, N) FP16 */
    int32_t M = a_fp16->shape[0];
    int32_t K = a_fp16->shape[1];
    int32_t N = b_int4->shape[1];

    /* Dequantize-on-the-fly matmul.
     * INT4 values from tensor_get are in [-8,7]; divide by 7.0 to get
     * approximate float, consistent with op_dequantize. */
    for (int32_t m = 0; m < M; m++) {
        for (int32_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int32_t k = 0; k < K; k++) {
                int32_t ai[] = {m, k};
                int32_t bi[] = {k, n};
                float a_val = tensor_get(a_fp16, ai);
                float b_val = tensor_get(b_int4, bi) / 7.0f;
                sum += a_val * b_val;
            }
            int32_t oi[] = {m, n};
            tensor_set(out, oi, sum);
        }
    }
}

/* ---- Normalization ---- */

void op_rms_norm(Tensor *out, const Tensor *a, const Tensor *weight, float eps) {
    if (!out || !a || !weight) {
        return;
    }

    size_t n = tensor_numel(a);

    /* Compute mean of squares */
    float sum_sq = 0.0f;
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float val = tensor_get(a, indices);
        sum_sq += val * val;
    }
    float rms = sqrtf(sum_sq / (float)n + eps);

    /* Normalize and scale by weight */
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float val = tensor_get(a, indices);
        float w = tensor_get(weight, indices);
        tensor_set(out, indices, (val / rms) * w);
    }
}

/* ---- Training operations ---- */

void op_sub(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) {
        return;
    }
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float va = tensor_get(a, indices);
        float vb = tensor_get(b, indices);
        tensor_set(out, indices, va - vb);
    }
}

void op_add_scaled(Tensor *out, const Tensor *a, const Tensor *b, float s) {
    if (!out || !a || !b) {
        return;
    }
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float va = tensor_get(a, indices);
        float vb = tensor_get(b, indices);
        tensor_set(out, indices, va + vb * s);
    }
}

void op_matmul_tn(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) {
        return;
    }
    /* a: (K, M), b: (K, N), out: (M, N)
     * Computes a^T @ b */
    int32_t K = a->shape[0];
    int32_t M = a->shape[1];
    int32_t N = b->shape[1];

    if (a->dtype == DTYPE_FP32 && b->dtype == DTYPE_FP32 &&
        tensor_is_contiguous(a) && tensor_is_contiguous(b) &&
        tensor_is_contiguous(out)) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    (const float *)a->data, M, /* lda = number of columns of a */
                    (const float *)b->data, N,
                    0.0f,
                    (float *)out->data, N);
    } else {
        for (int32_t m = 0; m < M; m++) {
            for (int32_t n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int32_t k = 0; k < K; k++) {
                    int32_t ai[] = {k, m}; /* a^T[m, k] = a[k, m] */
                    int32_t bi[] = {k, n};
                    sum += tensor_get(a, ai) * tensor_get(b, bi);
                }
                int32_t oi[] = {m, n};
                tensor_set(out, oi, sum);
            }
        }
    }
}

void op_matmul_nt(Tensor *out, const Tensor *a, const Tensor *b) {
    if (!out || !a || !b) {
        return;
    }
    /* a: (M, K), b: (N, K), out: (M, N)
     * Computes a @ b^T */
    int32_t M = a->shape[0];
    int32_t K = a->shape[1];
    int32_t N = b->shape[0];

    if (a->dtype == DTYPE_FP32 && b->dtype == DTYPE_FP32 &&
        tensor_is_contiguous(a) && tensor_is_contiguous(b) &&
        tensor_is_contiguous(out)) {
        if (gpu_matmul_fn && (int64_t)M * N * K > GPU_MATMUL_THRESHOLD) {
            /* GPU path: transpose B ([N,K] -> [K,N]) then C = A @ B^T */
            float *bt = (float *)malloc((size_t)K * N * sizeof(float));
            if (bt) {
                const float *bdata = (const float *)b->data;
                for (int32_t i = 0; i < N; i++) {
                    for (int32_t j = 0; j < K; j++) {
                        bt[j * N + i] = bdata[i * K + j];
                    }
                }
                gpu_matmul_fn((const float *)a->data, bt,
                              (float *)out->data, M, N, K);
                free(bt);
            } else {
                /* Fallback to CPU if transpose alloc fails */
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            M, N, K, 1.0f,
                            (const float *)a->data, K,
                            (const float *)b->data, K,
                            0.0f, (float *)out->data, N);
            }
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K,
                        1.0f,
                        (const float *)a->data, K,
                        (const float *)b->data, K,
                        0.0f,
                        (float *)out->data, N);
        }
    } else {
        for (int32_t m = 0; m < M; m++) {
            for (int32_t n = 0; n < N; n++) {
                float sum = 0.0f;
                for (int32_t k = 0; k < K; k++) {
                    int32_t ai[] = {m, k};
                    int32_t bi[] = {n, k}; /* b^T[k, n] = b[n, k] */
                    sum += tensor_get(a, ai) * tensor_get(b, bi);
                }
                int32_t oi[] = {m, n};
                tensor_set(out, oi, sum);
            }
        }
    }
}

void op_silu(Tensor *out, const Tensor *a) {
    if (!out || !a) {
        return;
    }
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float x = tensor_get(a, indices);
        float sig = 1.0f / (1.0f + expf(-x));
        tensor_set(out, indices, x * sig);
    }
}

void op_silu_backward(Tensor *out, const Tensor *a) {
    if (!out || !a) {
        return;
    }
    size_t n = tensor_numel(a);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(a, i, indices);
        float x = tensor_get(a, indices);
        float sig = 1.0f / (1.0f + expf(-x));
        /* silu'(x) = sig(x) + x * sig(x) * (1 - sig(x)) = sig(x) * (1 + x * (1 - sig(x))) */
        tensor_set(out, indices, sig * (1.0f + x * (1.0f - sig)));
    }
}

float op_cross_entropy(const Tensor *logits, const int32_t *targets, int32_t seq_len,
                       Tensor *grad_logits) {
    if (!logits || !targets || seq_len <= 0) {
        return 0.0f;
    }

    int32_t vocab_size = logits->shape[1];
    float total_loss = 0.0f;

    for (int32_t s = 0; s < seq_len; s++) {
        /* Find max logit for numerical stability */
        float max_logit = -INFINITY;
        for (int32_t v = 0; v < vocab_size; v++) {
            int32_t idx[] = {s, v};
            float val = tensor_get(logits, idx);
            if (val > max_logit) max_logit = val;
        }

        /* Compute log-sum-exp */
        float sum_exp = 0.0f;
        for (int32_t v = 0; v < vocab_size; v++) {
            int32_t idx[] = {s, v};
            sum_exp += expf(tensor_get(logits, idx) - max_logit);
        }
        float log_sum_exp = max_logit + logf(sum_exp);

        /* Loss = -logits[s, target] + log_sum_exp */
        int32_t target = targets[s];
        int32_t target_idx[] = {s, target};
        total_loss += -tensor_get(logits, target_idx) + log_sum_exp;

        /* Gradient: dL/d(logits[s,v]) = softmax(logits[s,v]) - 1{v == target} */
        if (grad_logits) {
            for (int32_t v = 0; v < vocab_size; v++) {
                int32_t idx[] = {s, v};
                float prob = expf(tensor_get(logits, idx) - log_sum_exp);
                float grad = prob - (v == target ? 1.0f : 0.0f);
                tensor_set(grad_logits, idx, grad / (float)seq_len);
            }
        }
    }

    return total_loss / (float)seq_len;
}

void op_rand_uniform(Tensor *out, float lo, float hi) {
    if (!out) {
        return;
    }
    float range = hi - lo;
    size_t n = tensor_numel(out);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(out, i, indices);
        float val = lo + range * ((float)rand() / (float)RAND_MAX);
        tensor_set(out, indices, val);
    }
}

void op_init_xavier(Tensor *out, int32_t fan_in, int32_t fan_out) {
    if (!out) {
        return;
    }
    /* Xavier uniform: U(-limit, limit) where limit = sqrt(6/(fan_in + fan_out)) */
    float limit = sqrtf(6.0f / (float)(fan_in + fan_out));
    op_rand_uniform(out, -limit, limit);
}

/* ---- Quantization ---- */

void op_dequantize(Tensor *out, const Tensor *in_int4) {
    if (!out || !in_int4) {
        return;
    }

    /* INT4 values are in [-8, 7]. Dequantize by mapping to approximately [-1, 1].
     * Scale factor: divide by 7.0 so that +7 maps to +1.0.
     * This must match the scaling used in op_quantize_int4. */
    size_t n = tensor_numel(in_int4);
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(in_int4, i, indices);
        float raw = tensor_get(in_int4, indices); /* returns [-8, 7] */
        tensor_set(out, indices, raw / 7.0f);
    }
}

void op_quantize_int4(Tensor *out_int4, const Tensor *in_fp32) {
    if (!out_int4 || !in_fp32) {
        return;
    }

    /* Symmetric absmax quantization to INT4 [-8, 7].
     * scale = 7.0 / max(|x|), then q = round(x * scale), clamped to [-8, 7].
     * Dequantize reverses with val / 7.0. */
    size_t n = tensor_numel(in_fp32);

    /* Find absolute max */
    float abs_max = 0.0f;
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(in_fp32, i, indices);
        float val = tensor_get(in_fp32, indices);
        float av = fabsf(val);
        if (av > abs_max) abs_max = av;
    }

    float scale = 7.0f / fmaxf(abs_max, 1e-8f);

    /* Quantize each element */
    for (size_t i = 0; i < n; i++) {
        int32_t indices[4] = {0, 0, 0, 0};
        linear_to_indices(in_fp32, i, indices);
        float val = tensor_get(in_fp32, indices);
        float scaled = roundf(val * scale);
        if (scaled < -8.0f) scaled = -8.0f;
        if (scaled > 7.0f) scaled = 7.0f;
        tensor_set(out_int4, indices, scaled);
    }
}

/* ---- QAT-aware matmul wrappers (Program 3 Phase 7, D-613) -------------- */

/*
 * op_matmul_qat -- out = in @ W (or in @ w_hat if qat enabled).
 *
 * When qat is NULL or disabled, delegates directly to op_matmul (zero overhead,
 * bit-identical output — the regression guard).
 * When qat is enabled, fetches the cached fake-quantized w_hat for W from the
 * context and multiplies against that instead.
 */
void op_matmul_qat(Tensor *out, const Tensor *in, const Tensor *W,
                   struct QATContext *qat) {
    const Tensor *Wq = qat_context_w_hat(qat, W);
    op_matmul(out, in, Wq);
}

/*
 * op_matmul_nt_qat -- out = in @ W^T (or in @ w_hat^T if qat enabled).
 *
 * Same NULL/disabled passthrough guarantee as op_matmul_qat.
 */
void op_matmul_nt_qat(Tensor *out, const Tensor *in, const Tensor *W,
                      struct QATContext *qat) {
    const Tensor *Wq = qat_context_w_hat(qat, W);
    op_matmul_nt(out, in, Wq);
}
