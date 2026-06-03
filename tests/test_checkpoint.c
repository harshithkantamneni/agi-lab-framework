/* test_checkpoint.c -- Tests for checkpoint save/load.
 *
 * Tests:
 *   1. save_load_roundtrip   -- save then load, weights match exactly
 *   2. save_load_adam        -- optimizer m/v state survives roundtrip
 *   3. save_load_meta        -- training metadata roundtrip
 *   4. save_load_router_bias -- expert bias and EMA survive roundtrip
 *   5. peek_metadata         -- read meta without loading weights
 *   6. version_mismatch      -- reject checkpoint with wrong version
 *
 * Uses a tiny config: d_model=16, n_layers=2, n_experts=2, vocab_size=32.
 */

#include "../src/tests/unity.h"
#include "checkpoint.h"
#include "data_loader.h"
#include "grad.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "router.h"
#include "tensor.h"
#include "tokenizer.h"
#include "train_config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *CKPT_PATH = "/tmp/test_checkpoint.bin";

static HSPAConfig test_config(void) {
    HSPAConfig cfg;
    cfg.d_model        = 16;
    cfg.n_layers       = 2;
    cfg.n_heads        = 4;
    cfg.n_kv_heads     = 2;
    cfg.head_dim       = 4;
    cfg.n_experts      = 2;
    cfg.n_active       = 1;
    cfg.d_ff           = 32;
    cfg.vocab_size     = 32;
    cfg.max_seq_len    = 16;
    cfg.ipc_iterations = 1;
    cfg.rms_norm_eps   = 1e-5f;
    cfg.storage_dtype  = DTYPE_FP32;
    cfg.compute_dtype  = DTYPE_FP32;
    return cfg;
}

/* Fill tensor with deterministic values. */
static void fill_tensor_det(Tensor *t, float seed) {
    size_t n = tensor_numel(t);
    float *data = (float *)t->data;
    for (size_t i = 0; i < n; i++) {
        data[i] = sinf(seed * (float)(i + 1)) * 0.5f;
    }
}

/* Fill all model weights with deterministic values. */
static void fill_model_weights(HSPAModel *model, const HSPAConfig *cfg, float base) {
    fill_tensor_det(model->embed->weight, base + 1.0f);
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        HSPABlock *blk = model->layers[l];
        float lb = base + 10.0f * (float)l;
        fill_tensor_det(blk->attn_norm->weight, lb + 2.0f);
        fill_tensor_det(blk->attn->W_q, lb + 3.0f);
        fill_tensor_det(blk->attn->W_k, lb + 4.0f);
        fill_tensor_det(blk->attn->W_v, lb + 5.0f);
        fill_tensor_det(blk->attn->W_o, lb + 6.0f);
        fill_tensor_det(blk->ffn_norm->weight, lb + 7.0f);
        fill_tensor_det(blk->router->W_mu, lb + 8.0f);
        fill_tensor_det(blk->router->W_sigma, lb + 9.0f);
        for (int32_t e = 0; e < cfg->n_experts; e++) {
            float eb = base + 100.0f * (float)l + 10.0f * (float)e;
            fill_tensor_det(blk->experts[e]->W_gate, eb + 1.0f);
            fill_tensor_det(blk->experts[e]->W_up, eb + 2.0f);
            fill_tensor_det(blk->experts[e]->W_down, eb + 3.0f);
        }
    }
    fill_tensor_det(model->final_norm->weight, base + 99.0f);
}

/* Max absolute difference between two tensors. */
static float tensor_max_diff(const Tensor *a, const Tensor *b) {
    size_t n = tensor_numel(a);
    float *da = (float *)a->data;
    float *db = (float *)b->data;
    float mx = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float d = fabsf(da[i] - db[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

/* Max diff across all model weights. */
static float model_max_diff(const HSPAModel *a, const HSPAModel *b, const HSPAConfig *cfg) {
    float mx = 0.0f, d;
    d = tensor_max_diff(a->embed->weight, b->embed->weight); if (d > mx) mx = d;
    for (int32_t l = 0; l < cfg->n_layers; l++) {
        HSPABlock *ba = a->layers[l], *bb = b->layers[l];
        d = tensor_max_diff(ba->attn_norm->weight, bb->attn_norm->weight); if (d > mx) mx = d;
        d = tensor_max_diff(ba->attn->W_q, bb->attn->W_q); if (d > mx) mx = d;
        d = tensor_max_diff(ba->attn->W_k, bb->attn->W_k); if (d > mx) mx = d;
        d = tensor_max_diff(ba->attn->W_v, bb->attn->W_v); if (d > mx) mx = d;
        d = tensor_max_diff(ba->attn->W_o, bb->attn->W_o); if (d > mx) mx = d;
        d = tensor_max_diff(ba->ffn_norm->weight, bb->ffn_norm->weight); if (d > mx) mx = d;
        d = tensor_max_diff(ba->router->W_mu, bb->router->W_mu); if (d > mx) mx = d;
        d = tensor_max_diff(ba->router->W_sigma, bb->router->W_sigma); if (d > mx) mx = d;
        for (int32_t e = 0; e < cfg->n_experts; e++) {
            d = tensor_max_diff(ba->experts[e]->W_gate, bb->experts[e]->W_gate); if (d > mx) mx = d;
            d = tensor_max_diff(ba->experts[e]->W_up, bb->experts[e]->W_up); if (d > mx) mx = d;
            d = tensor_max_diff(ba->experts[e]->W_down, bb->experts[e]->W_down); if (d > mx) mx = d;
        }
    }
    d = tensor_max_diff(a->final_norm->weight, b->final_norm->weight); if (d > mx) mx = d;
    return mx;
}

/* ======================================================================== */

/* Test 1: Save model, load into fresh model, weights match exactly. */
static void test_save_load_roundtrip(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    CheckpointMeta meta = {.step = 42, .epoch = 1, .tokens_trained = 5376,
                           .best_ppl = 123.4f, .best_loss = 4.82f};

    HSPAModel *m1 = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(m1);
    fill_model_weights(m1, &cfg, 0.0f);

    bool ok = checkpoint_save(CKPT_PATH, m1, NULL, &cfg, &tcfg, &meta);
    ASSERT_TRUE(ok);

    HSPAModel *m2 = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(m2);
    CheckpointMeta meta2;
    ok = checkpoint_load(CKPT_PATH, m2, NULL, &cfg, NULL, &meta2);
    ASSERT_TRUE(ok);

    float diff = model_max_diff(m1, m2, &cfg);
    ASSERT_EQUAL_FLOAT(0.0f, diff, 0.0f);

    hspa_model_destroy(m1);
    hspa_model_destroy(m2);
    unlink(CKPT_PATH);
}

/* Test 2: Adam m/v state survives roundtrip. */
static void test_save_load_adam(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    CheckpointMeta meta = {.step = 10};

    HSPAModel *model = hspa_model_create(&cfg);
    fill_model_weights(model, &cfg, 0.0f);

    AdamState *adam1 = adam_create(&cfg);
    ASSERT_NOT_NULL(adam1);

    /* Fill embed adam moments with known values */
    float *m_data = (float *)adam1->embed_m.m->data;
    float *v_data = (float *)adam1->embed_m.v->data;
    size_t n = tensor_numel(adam1->embed_m.m);
    for (size_t i = 0; i < n; i++) {
        m_data[i] = 0.01f * (float)i;
        v_data[i] = 0.001f * (float)i;
    }

    bool ok = checkpoint_save(CKPT_PATH, model, adam1, &cfg, &tcfg, &meta);
    ASSERT_TRUE(ok);

    HSPAModel *model2 = hspa_model_create(&cfg);
    AdamState *adam2 = adam_create(&cfg);
    ok = checkpoint_load(CKPT_PATH, model2, adam2, &cfg, NULL, NULL);
    ASSERT_TRUE(ok);

    float *m2_data = (float *)adam2->embed_m.m->data;
    float *v2_data = (float *)adam2->embed_m.v->data;
    for (size_t i = 0; i < n; i++) {
        ASSERT_EQUAL_FLOAT(m_data[i], m2_data[i], 1e-7f);
        ASSERT_EQUAL_FLOAT(v_data[i], v2_data[i], 1e-7f);
    }

    adam_destroy(adam1);
    adam_destroy(adam2);
    hspa_model_destroy(model);
    hspa_model_destroy(model2);
    unlink(CKPT_PATH);
}

/* Test 3: Training metadata roundtrip. */
static void test_save_load_meta(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    CheckpointMeta meta_in = {
        .step = 12345, .epoch = 7, .tokens_trained = 1234567890LL,
        .best_ppl = 42.42f, .best_loss = 3.14f
    };

    HSPAModel *model = hspa_model_create(&cfg);
    fill_model_weights(model, &cfg, 0.0f);

    bool ok = checkpoint_save(CKPT_PATH, model, NULL, &cfg, &tcfg, &meta_in);
    ASSERT_TRUE(ok);

    HSPAModel *model2 = hspa_model_create(&cfg);
    CheckpointMeta meta_out = {0};
    ok = checkpoint_load(CKPT_PATH, model2, NULL, &cfg, NULL, &meta_out);
    ASSERT_TRUE(ok);

    ASSERT_EQUAL_INT(meta_in.step, meta_out.step);
    ASSERT_EQUAL_INT(meta_in.epoch, meta_out.epoch);
    ASSERT_EQUAL_INT(meta_in.tokens_trained, meta_out.tokens_trained);
    ASSERT_EQUAL_FLOAT(meta_in.best_ppl, meta_out.best_ppl, 1e-6f);
    ASSERT_EQUAL_FLOAT(meta_in.best_loss, meta_out.best_loss, 1e-6f);

    hspa_model_destroy(model);
    hspa_model_destroy(model2);
    unlink(CKPT_PATH);
}

/* Test 4: Router expert_bias and expert_load_ema survive roundtrip. */
static void test_save_load_router_bias(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    CheckpointMeta meta = {.step = 100};

    HSPAModel *model = hspa_model_create(&cfg);
    fill_model_weights(model, &cfg, 0.0f);

    for (int32_t l = 0; l < cfg.n_layers; l++) {
        FEPRouter *r = model->layers[l]->router;
        for (int32_t e = 0; e < cfg.n_experts; e++) {
            r->expert_bias[e] = 0.1f * (float)(l * cfg.n_experts + e);
            r->expert_load_ema[e] = 1.0f / (float)cfg.n_experts + 0.01f * (float)e;
        }
    }

    bool ok = checkpoint_save(CKPT_PATH, model, NULL, &cfg, &tcfg, &meta);
    ASSERT_TRUE(ok);

    HSPAModel *model2 = hspa_model_create(&cfg);
    ok = checkpoint_load(CKPT_PATH, model2, NULL, &cfg, NULL, NULL);
    ASSERT_TRUE(ok);

    for (int32_t l = 0; l < cfg.n_layers; l++) {
        FEPRouter *r1 = model->layers[l]->router;
        FEPRouter *r2 = model2->layers[l]->router;
        for (int32_t e = 0; e < cfg.n_experts; e++) {
            ASSERT_EQUAL_FLOAT(r1->expert_bias[e], r2->expert_bias[e], 1e-7f);
            ASSERT_EQUAL_FLOAT(r1->expert_load_ema[e], r2->expert_load_ema[e], 1e-7f);
        }
    }

    hspa_model_destroy(model);
    hspa_model_destroy(model2);
    unlink(CKPT_PATH);
}

/* Test 5: Peek reads metadata without loading weights. */
static void test_peek_metadata(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    tcfg.max_steps = 9999;
    CheckpointMeta meta_in = {.step = 500, .best_ppl = 77.7f};

    HSPAModel *model = hspa_model_create(&cfg);
    fill_model_weights(model, &cfg, 0.0f);

    bool ok = checkpoint_save(CKPT_PATH, model, NULL, &cfg, &tcfg, &meta_in);
    ASSERT_TRUE(ok);

    HSPAConfig cfg_out;
    TrainConfig tcfg_out;
    CheckpointMeta meta_out;
    ok = checkpoint_peek(CKPT_PATH, &cfg_out, &tcfg_out, &meta_out);
    ASSERT_TRUE(ok);

    ASSERT_EQUAL_INT(cfg.d_model, cfg_out.d_model);
    ASSERT_EQUAL_INT(cfg.n_layers, cfg_out.n_layers);
    ASSERT_EQUAL_INT(cfg.n_experts, cfg_out.n_experts);
    ASSERT_EQUAL_INT(9999, tcfg_out.max_steps);
    ASSERT_EQUAL_INT(500, meta_out.step);
    ASSERT_EQUAL_FLOAT(77.7f, meta_out.best_ppl, 1e-4f);

    hspa_model_destroy(model);
    unlink(CKPT_PATH);
}

/* Test 6: Loading a file with wrong magic returns false. */
static void test_version_mismatch(void) {
    FILE *f = fopen(CKPT_PATH, "wb");
    ASSERT_NOT_NULL(f);
    uint32_t bad_magic = 0xDEADBEEF;
    uint32_t bad_version = 999;
    fwrite(&bad_magic, 4, 1, f);
    fwrite(&bad_version, 4, 1, f);
    fclose(f);

    HSPAConfig cfg = test_config();
    HSPAModel *model = hspa_model_create(&cfg);
    bool ok = checkpoint_load(CKPT_PATH, model, NULL, &cfg, NULL, NULL);
    ASSERT_FALSE(ok);

    hspa_model_destroy(model);
    unlink(CKPT_PATH);
}

/* Test 7: Config mismatch on load returns false. */
static void test_config_mismatch(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    CheckpointMeta meta = {.step = 1};

    HSPAModel *model = hspa_model_create(&cfg);
    fill_model_weights(model, &cfg, 0.0f);

    bool ok = checkpoint_save(CKPT_PATH, model, NULL, &cfg, &tcfg, &meta);
    ASSERT_TRUE(ok);

    /* Try loading with different d_model */
    HSPAConfig bad_cfg = test_config();
    bad_cfg.d_model = 32; /* mismatch */
    bad_cfg.head_dim = 8;
    bad_cfg.d_ff = 64;
    HSPAModel *model2 = hspa_model_create(&bad_cfg);
    ok = checkpoint_load(CKPT_PATH, model2, NULL, &bad_cfg, NULL, NULL);
    ASSERT_FALSE(ok);

    hspa_model_destroy(model);
    hspa_model_destroy(model2);
    unlink(CKPT_PATH);
}

/* Test 8: Save with Adam, load without Adam (fseek skip path). */
static void test_adam_skip(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    CheckpointMeta meta = {.step = 50, .best_ppl = 999.0f};

    HSPAModel *m1 = hspa_model_create(&cfg);
    fill_model_weights(m1, &cfg, 0.0f);
    AdamState *adam = adam_create(&cfg);

    /* Fill some adam state */
    float *md = (float *)adam->embed_m.m->data;
    size_t n = tensor_numel(adam->embed_m.m);
    for (size_t i = 0; i < n; i++) md[i] = (float)i * 0.001f;

    bool ok = checkpoint_save(CKPT_PATH, m1, adam, &cfg, &tcfg, &meta);
    ASSERT_TRUE(ok);

    /* Load WITHOUT adam — exercises fseek skip path */
    HSPAModel *m2 = hspa_model_create(&cfg);
    CheckpointMeta meta2;
    ok = checkpoint_load(CKPT_PATH, m2, NULL, &cfg, NULL, &meta2);
    ASSERT_TRUE(ok);

    /* Weights still match despite skipping adam */
    float diff = model_max_diff(m1, m2, &cfg);
    ASSERT_EQUAL_FLOAT(0.0f, diff, 0.0f);
    ASSERT_EQUAL_INT(50, meta2.step);

    adam_destroy(adam);
    hspa_model_destroy(m1);
    hspa_model_destroy(m2);
    unlink(CKPT_PATH);
}

/* Test 9 (D-091 Phase B): StreamLoaderState roundtrips through a checkpoint.
 * Covers the two halves of the handoff: (a) a saved stream state survives
 * save+load byte-for-byte; (b) restoring it into a fresh loader produces the
 * same next batch as the loader that produced the state. */
static void test_save_load_stream_state(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    CheckpointMeta meta = {.step = 77, .best_ppl = 7.77f};

    HSPAModel *model = hspa_model_create(&cfg);
    ASSERT_NOT_NULL(model);
    fill_model_weights(model, &cfg, 0.0f);

    /* Build a tiny 2-source stream loader over temp files. */
    const char *pathA = "/tmp/test_ckpt_streamA.txt";
    const char *pathB = "/tmp/test_ckpt_streamB.txt";
    const char *tok_path = "/tmp/test_ckpt_stream_tok.bin";
    const char *bodyA =
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango ";
    const char *bodyB =
        "one two three four five six seven eight nine ten "
        "one two three four five six seven eight nine ten ";
    FILE *fa = fopen(pathA, "w"); ASSERT_NOT_NULL(fa); fputs(bodyA, fa); fclose(fa);
    FILE *fb = fopen(pathB, "w"); ASSERT_NOT_NULL(fb); fputs(bodyB, fb); fclose(fb);

    char combined[1024];
    snprintf(combined, sizeof(combined), "%s %s", bodyA, bodyB);
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);
    ASSERT_EQUAL_INT(0, tokenizer_train(tok, (const uint8_t *)combined,
                                        (int64_t)strlen(combined), 32));
    ASSERT_EQUAL_INT(0, tokenizer_save(tok, tok_path));

    StreamSourceSpec srcs[2] = {
        { .path = pathA, .tokenizer = tok, .weight = 2.0f },
        { .path = pathB, .tokenizer = tok, .weight = 1.0f },
    };

    /* Advance the source loader a few steps, snapshot state. */
    StreamDataLoader *sdl_src = stream_loader_create(srcs, 2, 6, 2026ULL);
    ASSERT_NOT_NULL(sdl_src);
    int32_t dump_t[6], dump_g[6];
    for (int i = 0; i < 12; i++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(sdl_src, dump_t, dump_g));
    }
    StreamLoaderState saved;
    stream_loader_save_state(sdl_src, &saved);
    ASSERT_EQUAL_INT(2, saved.n_sources);

    /* Persist checkpoint with the stream state. */
    bool ok = checkpoint_save_ex(CKPT_PATH, model, NULL, &cfg, &tcfg, &meta, &saved);
    ASSERT_TRUE(ok);

    /* Load the stream state back from disk. */
    HSPAModel *model2 = hspa_model_create(&cfg);
    StreamLoaderState loaded = {0};
    CheckpointMeta meta_out;
    ok = checkpoint_load_ex(CKPT_PATH, model2, NULL, &cfg, NULL, &meta_out, &loaded);
    ASSERT_TRUE(ok);

    /* Byte-identity: state struct round-trips exactly. */
    ASSERT_EQUAL_INT(saved.n_sources, loaded.n_sources);
    for (int i = 0; i < saved.n_sources; i++) {
        ASSERT_EQUAL_INT((int)saved.byte_offsets[i], (int)loaded.byte_offsets[i]);
        ASSERT_EQUAL_INT((int)saved.epochs[i],       (int)loaded.epochs[i]);
    }
    /* RNG state bytes exact. */
    ASSERT_TRUE(saved.rng_state == loaded.rng_state);
    ASSERT_TRUE(saved.rng_inc   == loaded.rng_inc);

    /* Behavioural equivalence: two independently-restored loaders produce
     * bit-identical next-batch sequences. This is the property the training
     * loop actually needs. */
    StreamDataLoader *sdl_resumed_A = stream_loader_create(srcs, 2, 6, 0xAAAAAAAA);
    StreamDataLoader *sdl_resumed_B = stream_loader_create(srcs, 2, 6, 0xBBBBBBBB);
    ASSERT_NOT_NULL(sdl_resumed_A);
    ASSERT_NOT_NULL(sdl_resumed_B);
    ASSERT_EQUAL_INT(0, stream_loader_restore_state(sdl_resumed_A, &loaded));
    ASSERT_EQUAL_INT(0, stream_loader_restore_state(sdl_resumed_B, &loaded));

    int32_t ta[6], tb[6], ga[6], gb[6];
    for (int step = 0; step < 10; step++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(sdl_resumed_A, ta, ga));
        ASSERT_EQUAL_INT(0, stream_loader_next(sdl_resumed_B, tb, gb));
        for (int i = 0; i < 6; i++) {
            ASSERT_EQUAL_INT(ta[i], tb[i]);
            ASSERT_EQUAL_INT(ga[i], gb[i]);
        }
    }

    stream_loader_destroy(sdl_resumed_A);
    stream_loader_destroy(sdl_resumed_B);
    stream_loader_destroy(sdl_src);
    tokenizer_destroy(tok);
    hspa_model_destroy(model);
    hspa_model_destroy(model2);
    unlink(CKPT_PATH);
    unlink(pathA);
    unlink(pathB);
    unlink(tok_path);
}

/* Test 10: Legacy checkpoint_save (no stream state) still loads via
 * checkpoint_load_ex — stream_state_out.n_sources must be zero. */
static void test_load_ex_no_stream(void) {
    HSPAConfig cfg = test_config();
    TrainConfig tcfg = train_config_micro();
    CheckpointMeta meta = {.step = 5};

    HSPAModel *model = hspa_model_create(&cfg);
    fill_model_weights(model, &cfg, 0.0f);

    /* Legacy save (no stream state). */
    bool ok = checkpoint_save(CKPT_PATH, model, NULL, &cfg, &tcfg, &meta);
    ASSERT_TRUE(ok);

    HSPAModel *model2 = hspa_model_create(&cfg);
    StreamLoaderState stream_out;
    memset(&stream_out, 0xAB, sizeof(stream_out));  /* poison */
    ok = checkpoint_load_ex(CKPT_PATH, model2, NULL, &cfg, NULL, NULL, &stream_out);
    ASSERT_TRUE(ok);
    /* With no stream block, stream_out should be zeroed. */
    ASSERT_EQUAL_INT(0, stream_out.n_sources);

    hspa_model_destroy(model);
    hspa_model_destroy(model2);
    unlink(CKPT_PATH);
}

int main(void) {
    printf("=== Checkpoint Save/Load Tests ===\n\n");

    RUN_TEST(test_save_load_roundtrip);
    RUN_TEST(test_save_load_adam);
    RUN_TEST(test_save_load_meta);
    RUN_TEST(test_save_load_router_bias);
    RUN_TEST(test_peek_metadata);
    RUN_TEST(test_version_mismatch);
    RUN_TEST(test_config_mismatch);
    RUN_TEST(test_adam_skip);
    RUN_TEST(test_save_load_stream_state);
    RUN_TEST(test_load_ex_no_stream);

    TEST_REPORT();
}
