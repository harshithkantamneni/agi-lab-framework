/* rmsnorm.h -- RMSNorm (Root Mean Square Layer Normalization).
 *
 * RMSNorm(x) = x / rms(x) * weight
 * where rms(x) = sqrt(mean(x^2) + eps)
 *
 * Simpler than LayerNorm: no mean subtraction, no bias.
 * Used before attention and FFN in each HSPA block.
 */

#ifndef RMSNORM_H
#define RMSNORM_H

#include "memory_pool.h"
#include "tensor.h"

#include <stdint.h>

typedef struct {
    Tensor *weight; /* Learnable scale: [dim], initialized to 1.0 */
    float eps;      /* Epsilon for numerical stability */
} RMSNorm;

/* Create an RMSNorm layer with the given dimension and epsilon.
 * Weight is allocated from `pool` and initialized to all 1.0.
 * Returns NULL on failure. */
RMSNorm *rmsnorm_create(MemoryPool *pool, int32_t dim, float eps);

/* Forward pass: out = RMSNorm(x) = x / rms(x) * weight.
 * x and out must have the same shape (1D: [dim]).
 * Delegates to op_rms_norm from ops.h. */
void rmsnorm_forward(Tensor *out, const RMSNorm *norm, const Tensor *x);

/* Destroy the RMSNorm layer (frees the struct, not the pool data). */
void rmsnorm_destroy(RMSNorm *norm);

#endif /* RMSNORM_H */
