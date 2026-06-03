/* checkpoint.c -- Save and load model checkpoints.
 *
 * Binary format: see checkpoint.h for layout documentation.
 * All tensors are written as raw FP32 arrays in deterministic order.
 * Atomic writes: data goes to path.tmp, then rename to path.
 *
 * === TrainConfig / CKPT_VERSION guard (devops Cycle 31, D-098 P2 followup) ===
 *
 * Cycle 29 Rev-2 added 40 bytes to TrainConfig (entropy-penalty + tau-anneal
 * fields) WITHOUT bumping CKPT_VERSION. That left two incompatible layouts
 * (Cycle 27 Plan B = 128B, Cycle 29 Rev-2 = 168B) under the same version
 * byte (V5). `tools/checkpoint_load_dump.py` had to paper over the mess by
 * auto-detecting the sub-variant from total file size (see
 * data/findings/devops_cycle30_v5_parse_fix.md). That tooling burden is a
 * direct consequence of the missing version bump.
 *
 * To prevent a repeat, the static_assert block below hard-codes the expected
 * sizeof(TrainConfig) for the CURRENT CKPT_VERSION. If anyone edits
 * train_config.h without updating this block AND bumping CKPT_VERSION +
 * adding a CKPT_V{N}_TCFG_SIZE entry + a load-branch, the build fails
 * loudly at compile time with "expected N bytes, got M".
 *
 * === Rules for the next TrainConfig change (Phase C or later) ===
 *
 *   1. Bump CKPT_VERSION in checkpoint.h (5 -> 6).
 *   2. Add CKPT_V5_TCFG_SIZE = 168 in checkpoint.h (freeze the old size).
 *   3. Add a `version == 5` branch in checkpoint_load_ex() and
 *      checkpoint_peek() that reads CKPT_V5_TCFG_SIZE bytes into the zeroed
 *      TrainConfig so old checkpoints still load under the new layout.
 *   4. Update TCFG_SIZE_FOR_CURRENT_VERSION below to the NEW size.
 *   5. Update tools/checkpoint_load_dump.py's V{N}_TCFG_CANDIDATES.
 *   6. Add a regression test for the new layout in
 *      tests/test_checkpoint_load_dump.py.
 *
 * The CKPT_VERSION-to-size map below is the single source of truth consumed
 * by both _Static_assert (this file) AND the Python reader. Keep them in
 * sync.
 */

#include "checkpoint.h"
#include "attention.h"
#include "embedding.h"
#include "ffn.h"
#include "hspa_block.h"
#include "rmsnorm.h"
#include "router.h"
#include "tensor.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Max path length for tmp file */
#define CKPT_PATH_MAX 4096

/* === Compile-time CKPT_VERSION <-> sizeof(TrainConfig) guard ============
 *
 * If CKPT_VERSION changes, the line below MUST be updated to the new
 * sizeof(TrainConfig) and the old size preserved as CKPT_V{old}_TCFG_SIZE
 * in checkpoint.h. If TrainConfig changes size without a CKPT_VERSION
 * bump, this static_assert fires at compile time.
 *
 * Current value (CKPT_VERSION=6, Program-3 P7 layout): 172 bytes.
 * V6 added qat_group_size (int, 4 bytes) at the end of TrainConfig.
 * See src/training/train_config.h for the active field list.
 * ======================================================================== */
#define CKPT_CURRENT_TCFG_SIZE 172

_Static_assert(CKPT_VERSION == 6,
               "CKPT_VERSION bumped: update CKPT_CURRENT_TCFG_SIZE in "
               "checkpoint.c, add CKPT_V{old}_TCFG_SIZE in checkpoint.h, "
               "add a load branch for the old version, and update the "
               "Python tool's V{N}_TCFG_CANDIDATES. See file header.");

_Static_assert(sizeof(TrainConfig) == CKPT_CURRENT_TCFG_SIZE,
               "sizeof(TrainConfig) != CKPT_CURRENT_TCFG_SIZE. Either the "
               "struct changed (bump CKPT_VERSION + add CKPT_V{old}_TCFG_SIZE "
               "+ load branch) or CKPT_CURRENT_TCFG_SIZE drifted from the "
               "struct. See the header comment in checkpoint.c for the full "
               "migration checklist.");

/* Also catch silent drift in the archived sizes (defensive: if someone ever
 * edits CKPT_V{N}_TCFG_SIZE without understanding what it pins). */
_Static_assert(CKPT_V1_TCFG_SIZE == 100, "CKPT_V1_TCFG_SIZE must stay 100");
_Static_assert(CKPT_V2_TCFG_SIZE == 104, "CKPT_V2_TCFG_SIZE must stay 104");
_Static_assert(CKPT_V4_TCFG_SIZE == 116, "CKPT_V4_TCFG_SIZE must stay 116");
_Static_assert(CKPT_V5_TCFG_SIZE == 168, "CKPT_V5_TCFG_SIZE must stay 168");
_Static_assert(CKPT_V1_TCFG_SIZE < CKPT_V2_TCFG_SIZE,
               "V1 size must be < V2 size");
_Static_assert(CKPT_V2_TCFG_SIZE < CKPT_V4_TCFG_SIZE,
               "V2 size must be < V4 size");
_Static_assert(CKPT_V4_TCFG_SIZE < CKPT_V5_TCFG_SIZE,
               "V4 size must be < V5 size");
_Static_assert(CKPT_V5_TCFG_SIZE < CKPT_CURRENT_TCFG_SIZE,
               "V5 size must be < current size");

/* ---- Internal helpers ---- */

static bool write_tensor(FILE *f, const Tensor *t) {
    size_t n = tensor_nbytes(t);
    return fwrite(t->data, 1, n, f) == n;
}

static bool read_tensor(FILE *f, Tensor *t) {
    size_t n = tensor_nbytes(t);
    return fread(t->data, 1, n, f) == n;
}

/* Skip exactly n bytes in a readable file. */
static bool skip_bytes(FILE *f, size_t n) {
    return fseek(f, (long)n, SEEK_CUR) == 0;
}

static bool write_floats(FILE *f, const float *data, int32_t count) {
    size_t n = (size_t)count * sizeof(float);
    return fwrite(data, 1, n, f) == n;
}

static bool read_floats(FILE *f, float *data, int32_t count) {
    size_t n = (size_t)count * sizeof(float);
    return fread(data, 1, n, f) == n;
}

static bool write_adam_moment(FILE *f, const AdamMoment *am) {
    return write_tensor(f, am->m) && write_tensor(f, am->v);
}

static bool read_adam_moment(FILE *f, AdamMoment *am) {
    return read_tensor(f, am->m) && read_tensor(f, am->v);
}

/* Compute total byte size of Adam state for a given config (for fseek skip). */
static size_t adam_byte_size(const HSPAConfig *cfg) {
    size_t D = (size_t)cfg->d_model;
    size_t V = (size_t)cfg->vocab_size;
    size_t L = (size_t)cfg->n_layers;
    size_t K = (size_t)cfg->n_experts;
    size_t nh = (size_t)cfg->n_heads;
    size_t nkv = (size_t)cfg->n_kv_heads;
    size_t hd = (size_t)cfg->head_dim;
    size_t dff = (size_t)cfg->d_ff;

    /* Each parameter has m + v = 2 tensors, each tensor is numel * sizeof(float) */
    size_t embed = V * D;
    size_t per_layer =
        D +                             /* attn_norm */
        D * (nh * hd) +                 /* W_q */
        D * (nkv * hd) +                /* W_k */
        D * (nkv * hd) +                /* W_v */
        (nh * hd) * D +                 /* W_o */
        D +                             /* ffn_norm */
        D * K +                         /* router W_mu */
        D * K +                         /* router W_sigma */
        K * (D * dff + D * dff + dff * D); /* experts: gate + up + down */
    size_t final_norm = D;

    size_t total_params = embed + L * per_layer + final_norm;
    return total_params * 2 * sizeof(float); /* *2 for m and v */
}

/* Write all model weights in deterministic order. */
static bool write_weights(FILE *f, const HSPAModel *model, const HSPAConfig *cfg) {
    if (!write_tensor(f, model->embed->weight)) return false;

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        HSPABlock *blk = model->layers[l];
        if (!write_tensor(f, blk->attn_norm->weight)) return false;
        if (!write_tensor(f, blk->attn->W_q)) return false;
        if (!write_tensor(f, blk->attn->W_k)) return false;
        if (!write_tensor(f, blk->attn->W_v)) return false;
        if (!write_tensor(f, blk->attn->W_o)) return false;
        if (!write_tensor(f, blk->ffn_norm->weight)) return false;
        if (!write_tensor(f, blk->router->W_mu)) return false;
        if (!write_tensor(f, blk->router->W_sigma)) return false;

        for (int32_t e = 0; e < cfg->n_experts; e++) {
            if (!write_tensor(f, blk->experts[e]->W_gate)) return false;
            if (!write_tensor(f, blk->experts[e]->W_up)) return false;
            if (!write_tensor(f, blk->experts[e]->W_down)) return false;
        }
    }

    if (!write_tensor(f, model->final_norm->weight)) return false;
    return true;
}

/* Read all model weights in the same deterministic order. */
static bool read_weights(FILE *f, HSPAModel *model, const HSPAConfig *cfg) {
    if (!read_tensor(f, model->embed->weight)) return false;

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        HSPABlock *blk = model->layers[l];
        if (!read_tensor(f, blk->attn_norm->weight)) return false;
        if (!read_tensor(f, blk->attn->W_q)) return false;
        if (!read_tensor(f, blk->attn->W_k)) return false;
        if (!read_tensor(f, blk->attn->W_v)) return false;
        if (!read_tensor(f, blk->attn->W_o)) return false;
        if (!read_tensor(f, blk->ffn_norm->weight)) return false;
        if (!read_tensor(f, blk->router->W_mu)) return false;
        if (!read_tensor(f, blk->router->W_sigma)) return false;

        for (int32_t e = 0; e < cfg->n_experts; e++) {
            if (!read_tensor(f, blk->experts[e]->W_gate)) return false;
            if (!read_tensor(f, blk->experts[e]->W_up)) return false;
            if (!read_tensor(f, blk->experts[e]->W_down)) return false;
        }
    }

    if (!read_tensor(f, model->final_norm->weight)) return false;
    return true;
}

/* Write Adam optimizer state in the same weight order. */
static bool write_adam_state(FILE *f, const AdamState *adam, const HSPAConfig *cfg) {
    if (!write_adam_moment(f, &adam->embed_m)) return false;

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        struct AdamLayerState *ls = &adam->layers[l];
        if (!write_adam_moment(f, &ls->attn_norm_m)) return false;
        if (!write_adam_moment(f, &ls->attn_q_m)) return false;
        if (!write_adam_moment(f, &ls->attn_k_m)) return false;
        if (!write_adam_moment(f, &ls->attn_v_m)) return false;
        if (!write_adam_moment(f, &ls->attn_o_m)) return false;
        if (!write_adam_moment(f, &ls->ffn_norm_m)) return false;
        if (!write_adam_moment(f, &ls->router_mu_m)) return false;
        if (!write_adam_moment(f, &ls->router_sigma_m)) return false;

        for (int32_t e = 0; e < cfg->n_experts; e++) {
            if (!write_adam_moment(f, &ls->expert_gate_m[e])) return false;
            if (!write_adam_moment(f, &ls->expert_up_m[e])) return false;
            if (!write_adam_moment(f, &ls->expert_down_m[e])) return false;
        }
    }

    if (!write_adam_moment(f, &adam->final_norm_m)) return false;
    return true;
}

/* Read Adam optimizer state. */
static bool read_adam_state(FILE *f, AdamState *adam, const HSPAConfig *cfg) {
    if (!read_adam_moment(f, &adam->embed_m)) return false;

    for (int32_t l = 0; l < cfg->n_layers; l++) {
        struct AdamLayerState *ls = &adam->layers[l];
        if (!read_adam_moment(f, &ls->attn_norm_m)) return false;
        if (!read_adam_moment(f, &ls->attn_q_m)) return false;
        if (!read_adam_moment(f, &ls->attn_k_m)) return false;
        if (!read_adam_moment(f, &ls->attn_v_m)) return false;
        if (!read_adam_moment(f, &ls->attn_o_m)) return false;
        if (!read_adam_moment(f, &ls->ffn_norm_m)) return false;
        if (!read_adam_moment(f, &ls->router_mu_m)) return false;
        if (!read_adam_moment(f, &ls->router_sigma_m)) return false;

        for (int32_t e = 0; e < cfg->n_experts; e++) {
            if (!read_adam_moment(f, &ls->expert_gate_m[e])) return false;
            if (!read_adam_moment(f, &ls->expert_up_m[e])) return false;
            if (!read_adam_moment(f, &ls->expert_down_m[e])) return false;
        }
    }

    if (!read_adam_moment(f, &adam->final_norm_m)) return false;
    return true;
}

/* Write router bias/ema for all layers. */
static bool write_router_bias(FILE *f, const HSPAModel *model, const HSPAConfig *cfg) {
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        FEPRouter *r = model->layers[l]->router;
        if (!write_floats(f, r->expert_bias, cfg->n_experts)) return false;
        if (!write_floats(f, r->expert_load_ema, cfg->n_experts)) return false;
    }
    return true;
}

/* Read router bias/ema for all layers. */
static bool read_router_bias(FILE *f, HSPAModel *model, const HSPAConfig *cfg) {
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        FEPRouter *r = model->layers[l]->router;
        if (!read_floats(f, r->expert_bias, cfg->n_experts)) return false;
        if (!read_floats(f, r->expert_load_ema, cfg->n_experts)) return false;
    }
    return true;
}

/* V5: Write Default MoE EMA block for all layers (D-Plan-B).
 * Returns true if the model has any layer with default_moe_ema active. */
static bool any_layer_has_default_moe(const HSPAModel *model, const HSPAConfig *cfg) {
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        FEPRouter *r = model->layers[l]->router;
        if (r->default_moe_ema != NULL) return true;
    }
    return false;
}

/* Write K*D EMA for each layer. Caller must have verified presence via
 * any_layer_has_default_moe(). If a layer has no EMA tensor, writes zeros. */
static bool write_default_moe_ema(FILE *f, const HSPAModel *model, const HSPAConfig *cfg) {
    int32_t K = cfg->n_experts;
    int32_t D = cfg->d_model;
    int32_t KD = K * D;
    /* Scratch zero buffer for layers without EMA (shouldn't normally happen
     * when presence flag is 1, but defensive). */
    float *zeros = NULL;
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        FEPRouter *r = model->layers[l]->router;
        if (r->default_moe_ema) {
            if (!write_floats(f, r->default_moe_ema, KD)) {
                free(zeros);
                return false;
            }
        } else {
            if (!zeros) {
                zeros = (float *)calloc((size_t)KD, sizeof(float));
                if (!zeros) return false;
            }
            if (!write_floats(f, zeros, KD)) {
                free(zeros);
                return false;
            }
        }
    }
    free(zeros);
    return true;
}

/* Read K*D EMA for each layer, allocating the EMA tensor if needed.
 * Uses tcfg->default_moe_alpha if nonzero; else preserves router->default_moe_alpha;
 * else defaults to 0.01 so the loaded tensor can keep evolving. */
static bool read_default_moe_ema(FILE *f, HSPAModel *model, const HSPAConfig *cfg,
                                 const TrainConfig *tcfg) {
    int32_t K = cfg->n_experts;
    int32_t D = cfg->d_model;
    int32_t KD = K * D;
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        FEPRouter *r = model->layers[l]->router;
        if (!r->default_moe_ema) {
            r->default_moe_ema = (float *)calloc((size_t)KD, sizeof(float));
            if (!r->default_moe_ema) return false;
            r->default_moe_dim = D;
            /* Pick alpha from TrainConfig if provided, else keep a sane default. */
            float alpha = (tcfg && tcfg->default_moe_alpha > 0.0f)
                              ? tcfg->default_moe_alpha
                              : (r->default_moe_alpha > 0.0f ? r->default_moe_alpha : 0.01f);
            r->default_moe_alpha = alpha;
        }
        if (!read_floats(f, r->default_moe_ema, KD)) return false;
    }
    return true;
}

/* ---- Public API ---- */

bool checkpoint_save(const char *path,
                     const HSPAModel *model,
                     const AdamState *adam,
                     const HSPAConfig *cfg,
                     const TrainConfig *tcfg,
                     const CheckpointMeta *meta) {
    return checkpoint_save_ex(path, model, adam, cfg, tcfg, meta, NULL);
}

bool checkpoint_save_ex(const char *path,
                        const HSPAModel *model,
                        const AdamState *adam,
                        const HSPAConfig *cfg,
                        const TrainConfig *tcfg,
                        const CheckpointMeta *meta,
                        const StreamLoaderState *stream_state) {
    /* Atomic write: write to .tmp, fsync, rename */
    char tmp_path[CKPT_PATH_MAX];
    snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);

    FILE *f = fopen(tmp_path, "wb");
    if (!f) return false;

    /* Header */
    uint32_t magic = CKPT_MAGIC;
    uint32_t version = CKPT_VERSION;
    if (fwrite(&magic, 4, 1, f) != 1) goto fail;
    if (fwrite(&version, 4, 1, f) != 1) goto fail;

    /* Configs */
    if (fwrite(cfg, sizeof(HSPAConfig), 1, f) != 1) goto fail;
    if (fwrite(tcfg, sizeof(TrainConfig), 1, f) != 1) goto fail;

    /* Meta */
    if (fwrite(meta, sizeof(CheckpointMeta), 1, f) != 1) goto fail;

    /* Flag: has_adam */
    uint8_t has_adam = (adam != NULL) ? 1 : 0;
    if (fwrite(&has_adam, 1, 1, f) != 1) goto fail;

    /* Weights */
    if (!write_weights(f, model, cfg)) goto fail;

    /* Adam state (optional) */
    if (adam) {
        if (!write_adam_state(f, adam, cfg)) goto fail;
    }

    /* Router bias */
    if (!write_router_bias(f, model, cfg)) goto fail;

    /* V5: optional Default MoE EMA block. Presence byte first; when 1,
     * each layer writes K*D floats in layer order. V4 readers stop before
     * this byte; V5 readers of V4 files return has_default_moe=0 naturally
     * because we gate reads on (version >= 5). */
    uint8_t has_default_moe = any_layer_has_default_moe(model, cfg) ? 1 : 0;
    if (fwrite(&has_default_moe, 1, 1, f) != 1) goto fail;
    if (has_default_moe) {
        if (!write_default_moe_ema(f, model, cfg)) goto fail;
    }

    /* V4: optional StreamLoaderState block. Presence byte first so V3 readers
     * would stop here cleanly and V4 readers know whether to read further. */
    uint8_t has_stream = (stream_state != NULL) ? 1 : 0;
    if (fwrite(&has_stream, 1, 1, f) != 1) goto fail;
    if (stream_state) {
        if (fwrite(stream_state, sizeof(StreamLoaderState), 1, f) != 1) goto fail;
    }

    /* Flush to disk before rename */
    if (fflush(f) != 0) goto fail;
    if (fsync(fileno(f)) != 0) goto fail;
    fclose(f);

    /* Atomic rename (APFS guarantees atomicity) */
    if (rename(tmp_path, path) != 0) {
        unlink(tmp_path);
        return false;
    }
    return true;

fail:
    fclose(f);
    unlink(tmp_path);
    return false;
}

bool checkpoint_load(const char *path,
                     HSPAModel *model,
                     AdamState *adam,
                     const HSPAConfig *cfg,
                     TrainConfig *tcfg_out,
                     CheckpointMeta *meta_out) {
    return checkpoint_load_ex(path, model, adam, cfg, tcfg_out, meta_out, NULL);
}

bool checkpoint_load_ex(const char *path,
                        HSPAModel *model,
                        AdamState *adam,
                        const HSPAConfig *cfg,
                        TrainConfig *tcfg_out,
                        CheckpointMeta *meta_out,
                        StreamLoaderState *stream_state_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    /* Header */
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1) goto fail;
    if (fread(&version, 4, 1, f) != 1) goto fail;
    if (magic != CKPT_MAGIC || (version < 1 || version > CKPT_VERSION)) goto fail;

    /* Configs */
    HSPAConfig stored_cfg;
    TrainConfig stored_tcfg;
    memset(&stored_tcfg, 0, sizeof(stored_tcfg));
    if (fread(&stored_cfg, sizeof(HSPAConfig), 1, f) != 1) goto fail;
    if (version == 1) {
        /* V1: TrainConfig without balance_h_target (smaller struct) */
        if (fread(&stored_tcfg, CKPT_V1_TCFG_SIZE, 1, f) != 1) goto fail;
    } else if (version == 2) {
        /* V2: TrainConfig without D-031 loss-free balance fields */
        if (fread(&stored_tcfg, CKPT_V2_TCFG_SIZE, 1, f) != 1) goto fail;
        /* New fields zero-filled by memset above (loss-free off, bias params 0) */
    } else if (version == 3 || version == 4) {
        /* V3/V4: TrainConfig before D-Plan-B default-MoE fields.
         * Remaining default_moe_* fields zero-filled by memset above. */
        if (fread(&stored_tcfg, CKPT_V4_TCFG_SIZE, 1, f) != 1) goto fail;
    } else if (version == 5) {
        /* V5: TrainConfig before Program-3 P7 qat_group_size field (168 bytes).
         * qat_group_size zero-filled by memset; caller should treat 0 as 128. */
        if (fread(&stored_tcfg, CKPT_V5_TCFG_SIZE, 1, f) != 1) goto fail;
    } else {
        /* V6+: current TrainConfig (172 bytes) */
        if (fread(&stored_tcfg, sizeof(TrainConfig), 1, f) != 1) goto fail;
    }
    if (tcfg_out) *tcfg_out = stored_tcfg;

    /* Validate ALL shape-affecting config fields (HIGH-2 fix) */
    if (stored_cfg.d_model    != cfg->d_model    ||
        stored_cfg.n_layers   != cfg->n_layers   ||
        stored_cfg.n_heads    != cfg->n_heads    ||
        stored_cfg.n_kv_heads != cfg->n_kv_heads ||
        stored_cfg.head_dim   != cfg->head_dim   ||
        stored_cfg.n_experts  != cfg->n_experts  ||
        stored_cfg.d_ff       != cfg->d_ff       ||
        stored_cfg.vocab_size != cfg->vocab_size) {
        goto fail;
    }

    /* Meta */
    CheckpointMeta meta;
    if (fread(&meta, sizeof(CheckpointMeta), 1, f) != 1) goto fail;
    if (meta_out) *meta_out = meta;

    /* has_adam flag */
    uint8_t has_adam;
    if (fread(&has_adam, 1, 1, f) != 1) goto fail;

    /* Weights */
    if (!read_weights(f, model, cfg)) goto fail;

    /* Adam state: read if present, skip via fseek if caller doesn't need it */
    if (has_adam) {
        if (adam) {
            if (!read_adam_state(f, adam, cfg)) goto fail;
        } else {
            /* Skip Adam data without allocating (HIGH-4 fix) */
            if (!skip_bytes(f, adam_byte_size(cfg))) goto fail;
        }
    }

    /* Router bias */
    if (!read_router_bias(f, model, cfg)) goto fail;

    /* V5: optional Default MoE EMA. V1-V4 checkpoints lack this block. */
    if (version >= 5) {
        uint8_t has_default_moe = 0;
        if (fread(&has_default_moe, 1, 1, f) != 1) goto fail;
        if (has_default_moe) {
            if (!read_default_moe_ema(f, model, cfg, &stored_tcfg)) goto fail;
        }
    }

    /* V4: optional StreamLoaderState. V1-V3 checkpoints end here. */
    if (stream_state_out) {
        memset(stream_state_out, 0, sizeof(*stream_state_out));
    }
    if (version >= 4) {
        uint8_t has_stream = 0;
        if (fread(&has_stream, 1, 1, f) != 1) goto fail;
        if (has_stream) {
            StreamLoaderState tmp;
            if (fread(&tmp, sizeof(tmp), 1, f) != 1) goto fail;
            if (stream_state_out) *stream_state_out = tmp;
        }
    }

    fclose(f);
    return true;

fail:
    fclose(f);
    return false;
}

bool checkpoint_peek(const char *path,
                     HSPAConfig *cfg_out,
                     TrainConfig *tcfg_out,
                     CheckpointMeta *meta_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    /* Header */
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1) goto fail;
    if (fread(&version, 4, 1, f) != 1) goto fail;
    if (magic != CKPT_MAGIC || (version < 1 || version > CKPT_VERSION)) goto fail;

    /* Configs */
    HSPAConfig cfg;
    TrainConfig tcfg;
    memset(&tcfg, 0, sizeof(tcfg));
    if (fread(&cfg, sizeof(HSPAConfig), 1, f) != 1) goto fail;
    if (version == 1) {
        if (fread(&tcfg, CKPT_V1_TCFG_SIZE, 1, f) != 1) goto fail;
    } else if (version == 2) {
        if (fread(&tcfg, CKPT_V2_TCFG_SIZE, 1, f) != 1) goto fail;
    } else if (version == 3 || version == 4) {
        if (fread(&tcfg, CKPT_V4_TCFG_SIZE, 1, f) != 1) goto fail;
    } else if (version == 5) {
        /* V5: 168 bytes; qat_group_size zero-filled (V6 added it). */
        if (fread(&tcfg, CKPT_V5_TCFG_SIZE, 1, f) != 1) goto fail;
    } else {
        /* V6+: current full TrainConfig */
        if (fread(&tcfg, sizeof(TrainConfig), 1, f) != 1) goto fail;
    }
    if (cfg_out) *cfg_out = cfg;
    if (tcfg_out) *tcfg_out = tcfg;

    /* Meta */
    CheckpointMeta meta;
    if (fread(&meta, sizeof(CheckpointMeta), 1, f) != 1) goto fail;
    if (meta_out) *meta_out = meta;

    fclose(f);
    return true;

fail:
    fclose(f);
    return false;
}
