/* checkpoint.h -- Save and load model checkpoints.
 *
 * Binary checkpoint format (versioned):
 *   Header:  magic (4B) + version (4B)
 *   Config:  HSPAConfig + TrainConfig (fixed-size structs)
 *   Meta:    step, epoch, tokens_trained, best_ppl, best_loss
 *   Weights: all model parameters in deterministic order
 *   Adam:    m/v moment tensors (same order as weights)
 *   Router:  expert_bias + expert_load_ema per layer
 *
 * Weight order (must match load):
 *   1. embed->weight
 *   2. For each layer:
 *      a. attn_norm->weight
 *      b. attn Q, K, V, O
 *      c. ffn_norm->weight
 *      d. router W_mu, W_sigma
 *      e. For each expert: gate, up, down
 *   3. final_norm->weight
 */

#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "data_loader.h"   /* for StreamLoaderState */
#include "grad.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "train_config.h"

#include <stdbool.h>
#include <stdint.h>

#define CKPT_MAGIC   0x48535041  /* "HSPA" */
#define CKPT_VERSION 6

/* V1 TrainConfig size (before balance_h_target was added at end of struct).
 * Used for backward-compatible checkpoint loading. */
#define CKPT_V1_TCFG_SIZE 100

/* V2 TrainConfig size (before D-031 loss-free balance fields).
 * V2 added balance_h_target (float) at offset 100, total 104 bytes.
 * V3 adds: use_loss_free_balance (bool+pad=4) + lfb_bias_step (4) + lfb_ema_rate (4) = 116 bytes.
 *
 * V4 (Cycle 25 Phase B, D-091 handoff): adds an optional trailing
 * StreamLoaderState block after router bias. Flag byte (0/1) indicates presence.
 * V3 checkpoints load identically — no new tcfg fields.
 *
 * V5 (Cycle 27, D-Plan-B): adds an optional Default MoE EMA block BETWEEN
 * router bias and the V4 stream block. A 1-byte presence flag precedes the
 * per-layer [K*D] EMA data. V1-V4 checkpoints read unchanged: V5 readers
 * emit a 0 presence byte for V4 files (they stop before the EMA block).
 */
#define CKPT_V2_TCFG_SIZE 104

/* V3/V4 TrainConfig size (before D-Plan-B default MoE fields). V5 adds:
 * use_default_moe (bool+pad=4) + default_moe_alpha (4) + default_moe_sigma_init (4) = 128 bytes total. */
#define CKPT_V4_TCFG_SIZE 116

/* V5 TrainConfig size (before Program-3 Phase-7 qat_group_size field).
 * V6 adds: qat_group_size (int=4) = 172 bytes total.
 * Placed at end of struct for backward compat (older readers zero-fill). */
#define CKPT_V5_TCFG_SIZE 168

/* Training metadata saved alongside the model. */
typedef struct {
    int32_t step;
    int32_t epoch;
    int64_t tokens_trained;
    float   best_ppl;
    float   best_loss;
} CheckpointMeta;

/* Save a complete checkpoint (model + optimizer + router bias + meta).
 * path: file path to write (overwritten if exists).
 * adam: may be NULL if optimizer state is not needed.
 * Returns true on success, false on I/O error. */
bool checkpoint_save(const char *path,
                     const HSPAModel *model,
                     const AdamState *adam,
                     const HSPAConfig *cfg,
                     const TrainConfig *tcfg,
                     const CheckpointMeta *meta);

/* Save with optional StreamDataLoader state (Cycle 25 Phase B, D-091).
 * stream_state may be NULL -- writes a 0 presence byte then (same as legacy).
 * All other arguments behave identically to checkpoint_save. */
bool checkpoint_save_ex(const char *path,
                        const HSPAModel *model,
                        const AdamState *adam,
                        const HSPAConfig *cfg,
                        const TrainConfig *tcfg,
                        const CheckpointMeta *meta,
                        const StreamLoaderState *stream_state);

/* Load a complete checkpoint, restoring model weights + optimizer + meta.
 * The model and adam must already be created (matching cfg).
 * adam: may be NULL to skip loading optimizer state (uses fseek, no alloc).
 * tcfg_out: if non-NULL, receives the stored TrainConfig.
 * meta_out: if non-NULL, receives the stored training metadata.
 * Returns true on success, false on error (version mismatch, I/O, shape). */
bool checkpoint_load(const char *path,
                     HSPAModel *model,
                     AdamState *adam,
                     const HSPAConfig *cfg,
                     TrainConfig *tcfg_out,
                     CheckpointMeta *meta_out);

/* Load with optional StreamDataLoader state output (Cycle 25 Phase B, D-091).
 * stream_state_out:    if non-NULL, receives the stored StreamLoaderState.
 *                      n_sources==0 on output iff no stream state was present.
 * Returns true on success. */
bool checkpoint_load_ex(const char *path,
                        HSPAModel *model,
                        AdamState *adam,
                        const HSPAConfig *cfg,
                        TrainConfig *tcfg_out,
                        CheckpointMeta *meta_out,
                        StreamLoaderState *stream_state_out);

/* Query checkpoint metadata without loading weights.
 * Reads only the header + config + meta sections.
 * cfg_out: if non-NULL, receives the stored HSPAConfig.
 * tcfg_out: if non-NULL, receives the stored TrainConfig.
 * meta_out: if non-NULL, receives the stored training metadata.
 * Returns true on success. */
bool checkpoint_peek(const char *path,
                     HSPAConfig *cfg_out,
                     TrainConfig *tcfg_out,
                     CheckpointMeta *meta_out);

#endif /* CHECKPOINT_H */
