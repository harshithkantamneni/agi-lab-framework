/* hspa_config.h -- HSPA v1 model configuration.
 *
 * All hyperparameters for the Hierarchical Sparse Predictive Architecture.
 * Expert count (n_experts) is configurable per Director Decision D-004.
 *
 * Default config: 16B total params, 32 layers, d_model=4096, 16 experts/layer
 * (top-2), 4-bit quantized storage, FP32 compute (initial; FP16 later).
 */

#ifndef HSPA_CONFIG_H
#define HSPA_CONFIG_H

#include "tensor.h"
#include <stdint.h>

typedef struct {
    int32_t d_model;        /* Hidden dimension: 4096 */
    int32_t n_layers;       /* Number of transformer layers: 32 */
    int32_t n_heads;        /* Number of attention heads: 32 */
    int32_t n_kv_heads;     /* Number of KV heads (GQA): 8 */
    int32_t head_dim;       /* Head dimension: 128 (d_model / n_heads) */
    int32_t n_experts;      /* Experts per layer: 16 (CONFIGURABLE -- D-004) */
    int32_t n_active;       /* Active experts per token (top-k): 2 */
    int32_t d_ff;           /* Expert FFN intermediate dim: ~1850 (derived) */
    int32_t vocab_size;     /* Vocabulary size: 50000 */
    int32_t max_seq_len;    /* Maximum sequence length: 2048 */
    int32_t ipc_iterations; /* iPC inference iterations (T): 5 */
    float   rms_norm_eps;   /* RMSNorm epsilon: 1e-6 */
    DType   storage_dtype;  /* Weight storage type: DTYPE_INT4 */
    DType   compute_dtype;  /* Compute type: DTYPE_FP32 (start), DTYPE_FP16 (later) */
} HSPAConfig;

/* Returns the HSPA v1 default configuration with all standard parameters.
 * Expert count defaults to 16 but can be changed after creation (D-004). */
HSPAConfig hspa_config_default(void);

#endif /* HSPA_CONFIG_H */
