/* rmsnorm.c -- RMSNorm implementation.
 *
 * Uses op_rms_norm from ops.h for the actual computation.
 * Weight is initialized to all 1.0 (identity scaling).
 */

#include "rmsnorm.h"
#include "ops.h"

#include <stdlib.h>

RMSNorm *rmsnorm_create(MemoryPool *pool, int32_t dim, float eps) {
    if (!pool || dim <= 0) {
        return NULL;
    }

    RMSNorm *norm = (RMSNorm *)calloc(1, sizeof(RMSNorm));
    if (!norm) {
        return NULL;
    }

    int32_t shape[] = {dim};
    norm->weight = tensor_create(pool, shape, 1, DTYPE_FP32);
    if (!norm->weight) {
        free(norm);
        return NULL;
    }

    /* Initialize weight to 1.0 (identity scaling). */
    tensor_fill(norm->weight, 1.0f);
    norm->eps = eps;

    return norm;
}

void rmsnorm_forward(Tensor *out, const RMSNorm *norm, const Tensor *x) {
    if (!out || !norm || !x) {
        return;
    }

    /* Delegate to op_rms_norm: out = x / rms(x) * weight */
    op_rms_norm(out, x, norm->weight, norm->eps);
}

void rmsnorm_destroy(RMSNorm *norm) {
    if (!norm) {
        return;
    }

    /* Weight data is owned by the pool -- we only destroy the tensor metadata.
     * tensor_destroy handles this: if pool is set, it frees from pool. */
    if (norm->weight) {
        tensor_destroy(norm->weight);
    }

    free(norm);
}
