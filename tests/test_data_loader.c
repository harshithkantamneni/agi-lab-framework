/* test_data_loader.c -- Tests for the data loader module.
 *
 * Tests cover:
 *   1. Create / destroy lifecycle (no leak under ASan)
 *   2. data_loader_next returns correct next-token targets
 *   3. Sequential window advancement
 *   4. Epoch boundary: returns -1 when exhausted
 *   5. data_loader_reset restarts from beginning
 *   6. Multiple epochs produce identical sequences
 *   7. Position tracking
 *   8. Edge cases: too few tokens, seq_len=1
 *   9. load_and_tokenize on a small text file with a trained tokenizer
 *  10. Config variant param counts (small, medium)
 *
 * Built with ASan/UBSan via the Makefile test rules.
 */

#include "../src/tests/unity.h"
#include "data_loader.h"
#include "ipc_train.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ========================================================================
 * CREATE / DESTROY
 * ======================================================================== */

static void test_data_loader_create_destroy(void) {
    int32_t tokens[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int64_t n = 10;

    DataLoader *dl = data_loader_create(tokens, n, 4);
    ASSERT_NOT_NULL(dl);
    ASSERT_EQUAL_INT(10, (int)data_loader_total(dl));
    ASSERT_EQUAL_INT(0, (int)data_loader_position(dl));

    data_loader_destroy(dl);
}

/* ========================================================================
 * NULL / INVALID ARGUMENTS
 * ======================================================================== */

static void test_data_loader_invalid_args(void) {
    int32_t tokens[] = {1, 2, 3};

    /* NULL tokens */
    ASSERT_NULL(data_loader_create(NULL, 3, 2));

    /* seq_len < 1 */
    ASSERT_NULL(data_loader_create(tokens, 3, 0));
    ASSERT_NULL(data_loader_create(tokens, 3, -1));

    /* Not enough tokens: need seq_len + 1 = 4, but only have 3 */
    ASSERT_NULL(data_loader_create(tokens, 3, 3));

    /* Exactly enough: seq_len=2 needs 3 tokens */
    DataLoader *dl = data_loader_create(tokens, 3, 2);
    ASSERT_NOT_NULL(dl);
    data_loader_destroy(dl);
}

/* ========================================================================
 * NEXT-TOKEN TARGETS CORRECTNESS
 * ======================================================================== */

static void test_data_loader_next_token_targets(void) {
    /* tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] */
    int32_t data[10];
    for (int i = 0; i < 10; i++) data[i] = i;

    DataLoader *dl = data_loader_create(data, 10, 4);
    ASSERT_NOT_NULL(dl);

    int32_t toks[4], tgts[4];

    /* First batch: tokens=[0,1,2,3], targets=[1,2,3,4] */
    ASSERT_EQUAL_INT(0, data_loader_next(dl, toks, tgts));
    ASSERT_EQUAL_INT(0, toks[0]);
    ASSERT_EQUAL_INT(1, toks[1]);
    ASSERT_EQUAL_INT(2, toks[2]);
    ASSERT_EQUAL_INT(3, toks[3]);
    ASSERT_EQUAL_INT(1, tgts[0]);
    ASSERT_EQUAL_INT(2, tgts[1]);
    ASSERT_EQUAL_INT(3, tgts[2]);
    ASSERT_EQUAL_INT(4, tgts[3]);

    /* Second batch: tokens=[4,5,6,7], targets=[5,6,7,8] */
    ASSERT_EQUAL_INT(0, data_loader_next(dl, toks, tgts));
    ASSERT_EQUAL_INT(4, toks[0]);
    ASSERT_EQUAL_INT(5, toks[1]);
    ASSERT_EQUAL_INT(6, toks[2]);
    ASSERT_EQUAL_INT(7, toks[3]);
    ASSERT_EQUAL_INT(5, tgts[0]);
    ASSERT_EQUAL_INT(6, tgts[1]);
    ASSERT_EQUAL_INT(7, tgts[2]);
    ASSERT_EQUAL_INT(8, tgts[3]);

    /* Third batch: pos=8, need 8+4+1=13 > 10 -- exhausted */
    ASSERT_EQUAL_INT(-1, data_loader_next(dl, toks, tgts));

    data_loader_destroy(dl);
}

/* ========================================================================
 * POSITION TRACKING
 * ======================================================================== */

static void test_data_loader_position_tracking(void) {
    int32_t data[20];
    for (int i = 0; i < 20; i++) data[i] = i;

    DataLoader *dl = data_loader_create(data, 20, 5);
    ASSERT_NOT_NULL(dl);

    int32_t toks[5], tgts[5];

    ASSERT_EQUAL_INT(0, (int)data_loader_position(dl));

    ASSERT_EQUAL_INT(0, data_loader_next(dl, toks, tgts));
    ASSERT_EQUAL_INT(5, (int)data_loader_position(dl));

    ASSERT_EQUAL_INT(0, data_loader_next(dl, toks, tgts));
    ASSERT_EQUAL_INT(10, (int)data_loader_position(dl));

    ASSERT_EQUAL_INT(0, data_loader_next(dl, toks, tgts));
    ASSERT_EQUAL_INT(15, (int)data_loader_position(dl));

    /* pos=15, need 15+5+1=21 > 20 -- exhausted */
    ASSERT_EQUAL_INT(-1, data_loader_next(dl, toks, tgts));
    ASSERT_EQUAL_INT(15, (int)data_loader_position(dl));

    data_loader_destroy(dl);
}

/* ========================================================================
 * RESET AND MULTI-EPOCH
 * ======================================================================== */

static void test_data_loader_reset(void) {
    int32_t data[] = {10, 20, 30, 40, 50};
    DataLoader *dl = data_loader_create(data, 5, 2);
    ASSERT_NOT_NULL(dl);

    int32_t toks1[2], tgts1[2];
    int32_t toks2[2], tgts2[2];

    /* Epoch 1, batch 1 */
    ASSERT_EQUAL_INT(0, data_loader_next(dl, toks1, tgts1));
    /* tokens=[10,20], targets=[20,30] */
    ASSERT_EQUAL_INT(10, toks1[0]);
    ASSERT_EQUAL_INT(20, toks1[1]);
    ASSERT_EQUAL_INT(20, tgts1[0]);
    ASSERT_EQUAL_INT(30, tgts1[1]);

    /* Epoch 1, batch 2 */
    ASSERT_EQUAL_INT(0, data_loader_next(dl, toks1, tgts1));
    /* tokens=[30,40], targets=[40,50] */
    ASSERT_EQUAL_INT(30, toks1[0]);
    ASSERT_EQUAL_INT(40, toks1[1]);
    ASSERT_EQUAL_INT(40, tgts1[0]);
    ASSERT_EQUAL_INT(50, tgts1[1]);

    /* Exhausted */
    ASSERT_EQUAL_INT(-1, data_loader_next(dl, toks1, tgts1));

    /* Reset for epoch 2 */
    data_loader_reset(dl);
    ASSERT_EQUAL_INT(0, (int)data_loader_position(dl));

    /* Epoch 2 should produce identical results */
    ASSERT_EQUAL_INT(0, data_loader_next(dl, toks2, tgts2));
    ASSERT_EQUAL_INT(10, toks2[0]);
    ASSERT_EQUAL_INT(20, toks2[1]);
    ASSERT_EQUAL_INT(20, tgts2[0]);
    ASSERT_EQUAL_INT(30, tgts2[1]);

    data_loader_destroy(dl);
}

/* ========================================================================
 * EDGE CASE: seq_len = 1
 * ======================================================================== */

static void test_data_loader_seq_len_one(void) {
    int32_t data[] = {100, 200, 300};
    DataLoader *dl = data_loader_create(data, 3, 1);
    ASSERT_NOT_NULL(dl);

    int32_t tok, tgt;

    /* Batch 1: token=100, target=200 */
    ASSERT_EQUAL_INT(0, data_loader_next(dl, &tok, &tgt));
    ASSERT_EQUAL_INT(100, tok);
    ASSERT_EQUAL_INT(200, tgt);

    /* Batch 2: token=200, target=300 */
    ASSERT_EQUAL_INT(0, data_loader_next(dl, &tok, &tgt));
    ASSERT_EQUAL_INT(200, tok);
    ASSERT_EQUAL_INT(300, tgt);

    /* pos=2, need 2+1+1=4 > 3 -- exhausted */
    ASSERT_EQUAL_INT(-1, data_loader_next(dl, &tok, &tgt));

    data_loader_destroy(dl);
}

/* ========================================================================
 * LOAD_AND_TOKENIZE (roundtrip through file and tokenizer)
 * ======================================================================== */

static void test_load_and_tokenize(void) {
    /* Step 1: Write a small text file */
    const char *text_path = "/tmp/test_dl_text.txt";
    const char *tok_path  = "/tmp/test_dl_tokenizer.bin";
    const char *text =
        "the cat sat on the mat. the cat sat on the mat. "
        "the cat sat on the mat. the cat sat on the mat. "
        "the cat sat on the mat. the cat sat on the mat. "
        "the cat sat on the mat. the cat sat on the mat. ";

    FILE *f = fopen(text_path, "w");
    ASSERT_NOT_NULL(f);
    fprintf(f, "%s", text);
    fclose(f);

    /* Step 2: Train a small tokenizer and save it */
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    int result = tokenizer_train(tok, (const uint8_t *)text,
                                 (int64_t)strlen(text), 20);
    ASSERT_EQUAL_INT(0, result);

    result = tokenizer_save(tok, tok_path);
    ASSERT_EQUAL_INT(0, result);
    tokenizer_destroy(tok);

    /* Step 3: Use load_and_tokenize */
    int32_t *tokens = NULL;
    int64_t n = load_and_tokenize(text_path, tok_path, &tokens);
    ASSERT_TRUE(n > 0);
    ASSERT_NOT_NULL(tokens);

    /* The text should compress due to merges -- fewer tokens than bytes */
    ASSERT_TRUE(n < (int64_t)strlen(text));

    /* All token IDs should be in valid range [0, 256 + 20) */
    for (int64_t i = 0; i < n; i++) {
        ASSERT_TRUE(tokens[i] >= 0);
        ASSERT_TRUE(tokens[i] < 276);
    }

    /* Step 4: Verify we can create a DataLoader from the result */
    if (n >= 5) {
        DataLoader *dl = data_loader_create(tokens, n, 4);
        ASSERT_NOT_NULL(dl);

        int32_t toks[4], tgts[4];
        ASSERT_EQUAL_INT(0, data_loader_next(dl, toks, tgts));
        /* targets should be shifted by 1 */
        ASSERT_EQUAL_INT(tokens[1], tgts[0]);
        ASSERT_EQUAL_INT(tokens[2], tgts[1]);
        ASSERT_EQUAL_INT(tokens[3], tgts[2]);
        ASSERT_EQUAL_INT(tokens[4], tgts[3]);

        data_loader_destroy(dl);
    }

    free(tokens);
    unlink(text_path);
    unlink(tok_path);
}

/* ========================================================================
 * CONFIG VARIANTS: verify param count estimates
 * ======================================================================== */

static void test_config_small(void) {
    HSPAConfig cfg = hspa_config_small();

    ASSERT_EQUAL_INT(6, cfg.n_layers);
    ASSERT_EQUAL_INT(256, cfg.d_model);
    ASSERT_EQUAL_INT(4, cfg.n_heads);
    ASSERT_EQUAL_INT(2, cfg.n_kv_heads);
    ASSERT_EQUAL_INT(64, cfg.head_dim);
    ASSERT_EQUAL_INT(192, cfg.d_ff);
    ASSERT_EQUAL_INT(8, cfg.n_experts);
    ASSERT_EQUAL_INT(2, cfg.n_active);
    ASSERT_EQUAL_INT(4096, cfg.vocab_size);
    ASSERT_EQUAL_INT(512, cfg.max_seq_len);
    ASSERT_EQUAL_INT(5, cfg.ipc_iterations);
}

static void test_config_medium(void) {
    HSPAConfig cfg = hspa_config_medium();

    ASSERT_EQUAL_INT(8, cfg.n_layers);
    ASSERT_EQUAL_INT(512, cfg.d_model);
    ASSERT_EQUAL_INT(8, cfg.n_heads);
    ASSERT_EQUAL_INT(4, cfg.n_kv_heads);
    ASSERT_EQUAL_INT(64, cfg.head_dim);
    ASSERT_EQUAL_INT(384, cfg.d_ff);
    ASSERT_EQUAL_INT(8, cfg.n_experts);
    ASSERT_EQUAL_INT(2, cfg.n_active);
    /* D-182 P-MEDIUM-V4096-LATENT remediation: ipc_train.c:84 now hardcodes
     * V=32768 (Phase-B 32K tokenizer, all Program 2+ workflows). Previous
     * V=4096 was a dead default masked by --stream runtime override at
     * scale_experiment.c:589. */
    ASSERT_EQUAL_INT(32768, cfg.vocab_size);
    ASSERT_EQUAL_INT(512, cfg.max_seq_len);
    ASSERT_EQUAL_INT(5, cfg.ipc_iterations);
}

/* ========================================================================
 * STREAM DATA LOADER  (Cycle 24 Phase A, D-090)
 * ========================================================================
 * Multi-source streaming loader with weighted mixing.
 * All tests write tiny temp files under /tmp and clean up after themselves.
 */

/* Helpers ---------------------------------------------------------------- */

/* Write a text file. Returns 0 on success, -1 on failure. */
static int write_text_file(const char *path, const char *body) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fputs(body, f);
    fclose(f);
    return 0;
}

/* Create and save a tiny BPE tokenizer trained on `text` with `n_merges` merges.
 * Returns a loaded tokenizer (caller destroys). Saves to `tok_path` on disk too.
 * Returns NULL on failure. */
static Tokenizer *make_tokenizer(const char *text, int32_t n_merges,
                                 const char *tok_path) {
    Tokenizer *tok = tokenizer_create();
    if (!tok) return NULL;
    if (tokenizer_train(tok, (const uint8_t *)text, (int64_t)strlen(text),
                        n_merges) != 0) {
        tokenizer_destroy(tok);
        return NULL;
    }
    if (tokenizer_save(tok, tok_path) != 0) {
        tokenizer_destroy(tok);
        return NULL;
    }
    return tok;
}

/* Lifecycle -------------------------------------------------------------- */

static void test_stream_loader_create_destroy(void) {
    const char *path   = "/tmp/sdl_life.txt";
    const char *tpath  = "/tmp/sdl_life_tok.bin";
    const char *body   = "hello world hello world hello world hello world hello world ";
    ASSERT_EQUAL_INT(0, write_text_file(path, body));

    Tokenizer *tok = make_tokenizer(body, 16, tpath);
    ASSERT_NOT_NULL(tok);

    StreamSourceSpec src = { .path = path, .tokenizer = tok, .weight = 1.0f };
    StreamDataLoader *dl = stream_loader_create(&src, 1, 4, 42ULL);
    ASSERT_NOT_NULL(dl);
    ASSERT_EQUAL_INT(1, stream_loader_n_sources(dl));

    stream_loader_destroy(dl);
    tokenizer_destroy(tok);
    unlink(path);
    unlink(tpath);
}

/* Invalid args --------------------------------------------------------- */

static void test_stream_loader_invalid_args(void) {
    const char *path  = "/tmp/sdl_inv.txt";
    const char *tpath = "/tmp/sdl_inv_tok.bin";
    const char *body  = "abc abc abc abc abc abc abc abc abc abc abc abc abc abc ";
    ASSERT_EQUAL_INT(0, write_text_file(path, body));

    Tokenizer *tok = make_tokenizer(body, 8, tpath);
    ASSERT_NOT_NULL(tok);

    StreamSourceSpec src = { .path = path, .tokenizer = tok, .weight = 1.0f };

    /* NULL sources */
    ASSERT_NULL(stream_loader_create(NULL, 1, 4, 0));

    /* n_sources <= 0 */
    ASSERT_NULL(stream_loader_create(&src, 0, 4, 0));
    ASSERT_NULL(stream_loader_create(&src, -1, 4, 0));

    /* n_sources > MAX */
    ASSERT_NULL(stream_loader_create(&src, STREAM_LOADER_MAX_SOURCES + 1, 4, 0));

    /* seq_len < 1 */
    ASSERT_NULL(stream_loader_create(&src, 1, 0, 0));
    ASSERT_NULL(stream_loader_create(&src, 1, -1, 0));

    /* Missing file */
    StreamSourceSpec bad = { .path = "/tmp/nonexistent_file_xyz_12345.txt",
                             .tokenizer = tok, .weight = 1.0f };
    ASSERT_NULL(stream_loader_create(&bad, 1, 4, 0));

    /* All-zero weights */
    StreamSourceSpec zero = { .path = path, .tokenizer = tok, .weight = 0.0f };
    ASSERT_NULL(stream_loader_create(&zero, 1, 4, 0));

    /* NULL tokenizer */
    StreamSourceSpec nt = { .path = path, .tokenizer = NULL, .weight = 1.0f };
    ASSERT_NULL(stream_loader_create(&nt, 1, 4, 0));

    /* NULL path */
    StreamSourceSpec np = { .path = NULL, .tokenizer = tok, .weight = 1.0f };
    ASSERT_NULL(stream_loader_create(&np, 1, 4, 0));

    tokenizer_destroy(tok);
    unlink(path);
    unlink(tpath);
}

/* Single-source basic batch --------------------------------------------- */

static void test_stream_loader_single_source_next(void) {
    const char *path  = "/tmp/sdl_single.txt";
    const char *tpath = "/tmp/sdl_single_tok.bin";
    /* Big enough to fill several batches */
    const char *body =
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. ";
    ASSERT_EQUAL_INT(0, write_text_file(path, body));

    Tokenizer *tok = make_tokenizer(body, 32, tpath);
    ASSERT_NOT_NULL(tok);

    StreamSourceSpec src = { .path = path, .tokenizer = tok, .weight = 1.0f };
    StreamDataLoader *dl = stream_loader_create(&src, 1, 8, 123ULL);
    ASSERT_NOT_NULL(dl);

    int32_t toks[8], tgts[8];
    /* Three successive batches should succeed */
    ASSERT_EQUAL_INT(0, stream_loader_next(dl, toks, tgts));
    ASSERT_EQUAL_INT(0, stream_loader_next(dl, toks, tgts));
    ASSERT_EQUAL_INT(0, stream_loader_next(dl, toks, tgts));

    /* Within a single batch the next-token invariant must hold:
     * targets[i] == tokens[i+1] for i in [0, seq_len-2]. */
    for (int i = 0; i < 7; i++) {
        ASSERT_EQUAL_INT(toks[i + 1], tgts[i]);
    }

    stream_loader_destroy(dl);
    tokenizer_destroy(tok);
    unlink(path);
    unlink(tpath);
}

/* Determinism ----------------------------------------------------------- */

static void test_stream_loader_determinism(void) {
    const char *pathA = "/tmp/sdl_detA.txt";
    const char *pathB = "/tmp/sdl_detB.txt";
    const char *tpath = "/tmp/sdl_det_tok.bin";
    const char *bodyA =
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango uniform ";
    const char *bodyB =
        "zulu yankee xray whiskey victor uniform tango sierra romeo quebec "
        "papa oscar november mike lima kilo juliet india hotel golf foxtrot ";

    ASSERT_EQUAL_INT(0, write_text_file(pathA, bodyA));
    ASSERT_EQUAL_INT(0, write_text_file(pathB, bodyB));

    /* Train tokenizer on combined body so it covers both vocabs */
    char combined[1024];
    snprintf(combined, sizeof(combined), "%s %s", bodyA, bodyB);
    Tokenizer *tok = make_tokenizer(combined, 48, tpath);
    ASSERT_NOT_NULL(tok);

    StreamSourceSpec srcs[2] = {
        { .path = pathA, .tokenizer = tok, .weight = 1.0f },
        { .path = pathB, .tokenizer = tok, .weight = 1.0f },
    };

    const int N = 20;
    const int SEQ = 6;
    int32_t run1[20 * 6], run1_t[20 * 6];
    int32_t run2[20 * 6], run2_t[20 * 6];

    /* Run 1 with seed=7 */
    StreamDataLoader *dl1 = stream_loader_create(srcs, 2, SEQ, 7ULL);
    ASSERT_NOT_NULL(dl1);
    for (int b = 0; b < N; b++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(dl1, run1 + b * SEQ,
                                               run1_t + b * SEQ));
    }
    stream_loader_destroy(dl1);

    /* Run 2 with same seed=7 */
    StreamDataLoader *dl2 = stream_loader_create(srcs, 2, SEQ, 7ULL);
    ASSERT_NOT_NULL(dl2);
    for (int b = 0; b < N; b++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(dl2, run2 + b * SEQ,
                                               run2_t + b * SEQ));
    }
    stream_loader_destroy(dl2);

    /* Must be bit-identical */
    for (int i = 0; i < N * SEQ; i++) {
        ASSERT_EQUAL_INT(run1[i], run2[i]);
        ASSERT_EQUAL_INT(run1_t[i], run2_t[i]);
    }

    tokenizer_destroy(tok);
    unlink(pathA);
    unlink(pathB);
    unlink(tpath);
}

/* Different seed → different sequence ------------------------------------ */

static void test_stream_loader_different_seed(void) {
    const char *pathA = "/tmp/sdl_seedA.txt";
    const char *pathB = "/tmp/sdl_seedB.txt";
    const char *tpath = "/tmp/sdl_seed_tok.bin";
    const char *bodyA =
        "aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa aa ";
    const char *bodyB =
        "bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb bb ";

    ASSERT_EQUAL_INT(0, write_text_file(pathA, bodyA));
    ASSERT_EQUAL_INT(0, write_text_file(pathB, bodyB));

    char combined[512];
    snprintf(combined, sizeof(combined), "%s %s", bodyA, bodyB);
    Tokenizer *tok = make_tokenizer(combined, 16, tpath);
    ASSERT_NOT_NULL(tok);

    StreamSourceSpec srcs[2] = {
        { .path = pathA, .tokenizer = tok, .weight = 1.0f },
        { .path = pathB, .tokenizer = tok, .weight = 1.0f },
    };

    const int N = 30, SEQ = 4;
    int32_t runA[30 * 4], runA_t[30 * 4];
    int32_t runB[30 * 4], runB_t[30 * 4];

    StreamDataLoader *dl1 = stream_loader_create(srcs, 2, SEQ, 1ULL);
    StreamDataLoader *dl2 = stream_loader_create(srcs, 2, SEQ, 2ULL);
    ASSERT_NOT_NULL(dl1);
    ASSERT_NOT_NULL(dl2);

    for (int b = 0; b < N; b++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(dl1, runA + b * SEQ, runA_t + b * SEQ));
        ASSERT_EQUAL_INT(0, stream_loader_next(dl2, runB + b * SEQ, runB_t + b * SEQ));
    }
    stream_loader_destroy(dl1);
    stream_loader_destroy(dl2);

    /* At least one byte must differ (tokens OR targets). If they are bit-identical
     * with different seeds, something is wrong. */
    int differ = 0;
    for (int i = 0; i < N * SEQ; i++) {
        if (runA[i] != runB[i] || runA_t[i] != runB_t[i]) { differ = 1; break; }
    }
    ASSERT_TRUE(differ);

    tokenizer_destroy(tok);
    unlink(pathA);
    unlink(pathB);
    unlink(tpath);
}

/* Weighted mixing: empirical frequencies approximate target ------------- */

static void test_stream_loader_weighted_mixing(void) {
    /* Two fully-distinguishable sources: body 'A' has only 'a' bytes, 'B' has
     * only 'b' bytes (plus required newlines). Regardless of how the BPE
     * tokenizer merges, a batch that came from source A decodes to all-'a',
     * and a batch from source B decodes to all-'b'. We decode and classify. */
    const char *pathA = "/tmp/sdl_wtA.txt";
    const char *pathB = "/tmp/sdl_wtB.txt";
    const char *tpath = "/tmp/sdl_wt_tok.bin";
    char bodyA[2048], bodyB[2048];
    /* Use newlines between 'a' / 'b' blocks so chunk-boundary logic has
     * whitespace to split on. */
    for (int i = 0; i < 2000; i++) {
        bodyA[i] = (i % 8 == 7) ? '\n' : 'a';
        bodyB[i] = (i % 8 == 7) ? '\n' : 'b';
    }
    bodyA[2000] = 0; bodyB[2000] = 0;

    ASSERT_EQUAL_INT(0, write_text_file(pathA, bodyA));
    ASSERT_EQUAL_INT(0, write_text_file(pathB, bodyB));

    char combined[4096];
    snprintf(combined, sizeof(combined), "%s%s", bodyA, bodyB);
    Tokenizer *tok = make_tokenizer(combined, 8, tpath);
    ASSERT_NOT_NULL(tok);

    /* 75/25 mixing: source A should be picked ~3x as often as source B */
    StreamSourceSpec srcs[2] = {
        { .path = pathA, .tokenizer = tok, .weight = 3.0f },
        { .path = pathB, .tokenizer = tok, .weight = 1.0f },
    };

    StreamDataLoader *dl = stream_loader_create(srcs, 2, 4, 999ULL);
    ASSERT_NOT_NULL(dl);

    const int BATCHES = 500;
    int count_a = 0, count_b = 0, unknown = 0;
    int32_t toks[4], tgts[4];
    uint8_t decoded[64];

    for (int b = 0; b < BATCHES; b++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(dl, toks, tgts));
        /* Decode batch tokens and check which letter dominates */
        int64_t nbytes = tokenizer_decode(tok, toks, 4, decoded, sizeof(decoded));
        ASSERT_TRUE(nbytes > 0);
        int n_a = 0, n_b = 0;
        for (int64_t i = 0; i < nbytes; i++) {
            if (decoded[i] == 'a') n_a++;
            else if (decoded[i] == 'b') n_b++;
        }
        if      (n_a > 0 && n_b == 0) count_a++;
        else if (n_b > 0 && n_a == 0) count_b++;
        else                          unknown++;
    }
    /* No batch should mix letters: each source is pure 'a' or pure 'b'. */
    ASSERT_EQUAL_INT(0, unknown);

    /* Expected: 75% A / 25% B. With 500 draws, sd ≈ sqrt(0.75*0.25/500)=1.9 %.
     * Accept 60..90 % for A to leave a wide safety margin against RNG variance. */
    double frac_a = (double)count_a / (double)BATCHES;
    ASSERT_TRUE(frac_a > 0.60);
    ASSERT_TRUE(frac_a < 0.90);
    ASSERT_TRUE(count_a + count_b == BATCHES);

    stream_loader_destroy(dl);
    tokenizer_destroy(tok);
    unlink(pathA);
    unlink(pathB);
    unlink(tpath);
}

/* Wraparound: tiny file, consume past its end --------------------------- */

static void test_stream_loader_wraparound(void) {
    const char *path  = "/tmp/sdl_wrap.txt";
    const char *tpath = "/tmp/sdl_wrap_tok.bin";
    /* Small file so we're forced to wrap many times */
    const char *body = "ab cd ef gh ij kl mn op ";  /* ~24 bytes */
    ASSERT_EQUAL_INT(0, write_text_file(path, body));

    Tokenizer *tok = make_tokenizer(body, 8, tpath);
    ASSERT_NOT_NULL(tok);

    StreamSourceSpec src = { .path = path, .tokenizer = tok, .weight = 1.0f };
    StreamDataLoader *dl = stream_loader_create(&src, 1, 4, 17ULL);
    ASSERT_NOT_NULL(dl);

    int32_t toks[4], tgts[4];
    /* Pull WAY more tokens than the file contains. Must succeed every time. */
    for (int b = 0; b < 200; b++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(dl, toks, tgts));
    }

    /* Epoch counter must have incremented at least once. */
    ASSERT_TRUE(stream_loader_source_epochs(dl, 0) >= 1);

    stream_loader_destroy(dl);
    tokenizer_destroy(tok);
    unlink(path);
    unlink(tpath);
}

/* Save / restore state --------------------------------------------------
 *
 * We test the invariant the loader actually promises: after restoring a saved
 * state, the subsequent batch sequence is DETERMINISTIC (reproducible across
 * independent restored loaders). We do NOT claim bit-identity with the
 * sequence of the original loader that produced the state, because the design
 * discards the in-flight ring buffer on save/restore (documented tradeoff).
 *
 * This is the property Phase B checkpoint/resume needs: a saved state can be
 * restored into a freshly-created loader and produces a well-defined stream. */

static void test_stream_loader_save_restore(void) {
    const char *pathA = "/tmp/sdl_srA.txt";
    const char *pathB = "/tmp/sdl_srB.txt";
    const char *tpath = "/tmp/sdl_sr_tok.bin";
    const char *bodyA =
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet "
        "kilo lima mike november oscar papa quebec romeo sierra tango "
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet ";
    const char *bodyB =
        "one two three four five six seven eight nine ten "
        "one two three four five six seven eight nine ten "
        "one two three four five six seven eight nine ten ";
    ASSERT_EQUAL_INT(0, write_text_file(pathA, bodyA));
    ASSERT_EQUAL_INT(0, write_text_file(pathB, bodyB));

    char combined[2048];
    snprintf(combined, sizeof(combined), "%s %s", bodyA, bodyB);
    Tokenizer *tok = make_tokenizer(combined, 32, tpath);
    ASSERT_NOT_NULL(tok);

    StreamSourceSpec srcs[2] = {
        { .path = pathA, .tokenizer = tok, .weight = 2.0f },
        { .path = pathB, .tokenizer = tok, .weight = 1.0f },
    };

    const int SEQ = 8;
    const int WARMUP = 10;
    const int PROBE  = 15;

    /* Step 1: produce a checkpoint by running a loader for WARMUP batches. */
    StreamDataLoader *dl_source = stream_loader_create(srcs, 2, SEQ, 2026ULL);
    ASSERT_NOT_NULL(dl_source);
    int32_t dump_t[8], dump_g[8];
    for (int b = 0; b < WARMUP; b++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(dl_source, dump_t, dump_g));
    }
    StreamLoaderState st;
    stream_loader_save_state(dl_source, &st);
    ASSERT_EQUAL_INT(2, st.n_sources);
    stream_loader_destroy(dl_source);

    /* Step 2: two independent loaders restore the same state and must produce
     * bit-identical subsequent sequences. */
    int32_t runA[15 * 8], runA_t[15 * 8];
    int32_t runB[15 * 8], runB_t[15 * 8];

    StreamDataLoader *dlA = stream_loader_create(srcs, 2, SEQ, 0xDEADBEEFULL);
    ASSERT_NOT_NULL(dlA);
    ASSERT_EQUAL_INT(0, stream_loader_restore_state(dlA, &st));
    for (int b = 0; b < PROBE; b++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(dlA, runA + b * SEQ, runA_t + b * SEQ));
    }
    stream_loader_destroy(dlA);

    /* Note: we deliberately use a DIFFERENT construction seed to prove that
     * restore fully overrides the construction seed. Both loaders must end up
     * in the exact same state post-restore. */
    StreamDataLoader *dlB = stream_loader_create(srcs, 2, SEQ, 0xCAFEBABEULL);
    ASSERT_NOT_NULL(dlB);
    ASSERT_EQUAL_INT(0, stream_loader_restore_state(dlB, &st));
    for (int b = 0; b < PROBE; b++) {
        ASSERT_EQUAL_INT(0, stream_loader_next(dlB, runB + b * SEQ, runB_t + b * SEQ));
    }
    stream_loader_destroy(dlB);

    /* Deterministic: the two restored loaders must match exactly. */
    for (int i = 0; i < PROBE * SEQ; i++) {
        ASSERT_EQUAL_INT(runA[i],   runB[i]);
        ASSERT_EQUAL_INT(runA_t[i], runB_t[i]);
    }

    tokenizer_destroy(tok);
    unlink(pathA);
    unlink(pathB);
    unlink(tpath);
}

/* Restore mismatch detection ------------------------------------------- */

static void test_stream_loader_restore_mismatch(void) {
    const char *pathA = "/tmp/sdl_mmA.txt";
    const char *tpath = "/tmp/sdl_mm_tok.bin";
    const char *body  = "one two three four five six seven eight nine ten ";
    ASSERT_EQUAL_INT(0, write_text_file(pathA, body));

    Tokenizer *tok = make_tokenizer(body, 8, tpath);
    ASSERT_NOT_NULL(tok);

    StreamSourceSpec src = { .path = pathA, .tokenizer = tok, .weight = 1.0f };
    StreamDataLoader *dl = stream_loader_create(&src, 1, 4, 0ULL);
    ASSERT_NOT_NULL(dl);

    StreamLoaderState st = { 0 };
    st.n_sources = 7;  /* wrong */
    ASSERT_EQUAL_INT(-1, stream_loader_restore_state(dl, &st));

    stream_loader_destroy(dl);
    tokenizer_destroy(tok);
    unlink(pathA);
    unlink(tpath);
}

/* ========================================================================
 * HELD-OUT MISALIGNED-CONTENT LEAK REGRESSION  (P7 increment-5d, B-2)
 * ========================================================================
 *
 * This is the load-bearing leak test the code_reviewer (CHANGES_REQUIRED,
 * 2026-05-29) required: it drives the PRODUCTION source_refill() path via
 * stream_loader_create_heldout()/stream_loader_next() over a source whose
 * content is NOT whitespace-aligned at the 64 KiB chunk boundary — the exact
 * shape on which the B-1 coordinate-mismatch bug leaks held-out content into
 * the training stream.  The pre-existing harness fixtures all FORCE a
 * whitespace byte at each chunk boundary (make_temp_text_file), which is the
 * single input shape where the buggy implementation is accidentally correct;
 * this test deliberately avoids that.
 *
 * Binding property under test (repro C3 / experimental_design §5.6.4):
 *   No byte of a held-out chunk's FILE range [k*CB,(k+1)*CB) is ever encoded
 *   into the training ring.
 *
 * Mechanism:
 *   - 3-chunk source.  Each chunk filled with a DISTINCT signature byte
 *     ('A','B','C') interleaved with spaces every 100 bytes.  100 does NOT
 *     divide 65536, so the last byte of every chunk is a signature byte (not
 *     whitespace) → the whitespace-split holds back a pending tail →
 *     encode_len < BYTE_SLAB on every refill → content/file coordinate drift.
 *   - The held-out RNG is biased to source-index 4 (seed+4 / stream+4) where
 *     the label sequence is [train, HELDOUT, train, ...] (verified offline via
 *     tools/heldout_partition.py).  So chunk 1 ('B') is held out.
 *   - We pop many batches, decode each, and count 'B' bytes in the decoded
 *     training output.  Held-out chunk 1's signature must NEVER appear.
 *
 * Discriminating control (test_heldout_misaligned_control):
 *   The SAME source/tokenizer with thresh=0.0 (every chunk labeled train,
 *   nothing skipped) MUST yield a NON-ZERO 'B' count — proving the probe is
 *   real (counts content that is present) and not vacuously zero.
 *
 * RED on the unfixed (byte_offset-keyed) code; GREEN after the file-position
 * coordinate fix; proven discriminating by fault-injection (revert fix → RED).
 */

#define HM_CHUNK_BYTES   65536
#define HM_HELDOUT_SEED  UINT64_C(0x50524F475F335F48)
#define HM_HELDOUT_STREAM UINT64_C(0x48454C444F55544F)

/* Write a 3-chunk source where chunk k (0,1,2) is filled with signature byte
 * sig[k], interleaved with a space every 100 bytes.  The boundary byte is a
 * signature byte (100 does not divide 65536), so the content is misaligned at
 * the 64 KiB chunk boundary. */
static int hm_write_misaligned_source(const char *path, const char sig[3]) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    for (int k = 0; k < 3; k++) {
        for (int64_t pos = 0; pos < HM_CHUNK_BYTES; pos++) {
            int c = (pos % 100 == 99) ? ' ' : sig[k];
            fputc(c, f);
        }
    }
    fclose(f);
    return 0;
}

/* Drain `batches` batches from a single-source held-out loader and count the
 * occurrences of `target_byte` in the decoded token stream.  Returns the
 * count, or -1 on any loader/decoder failure. */
static int64_t hm_count_decoded_byte(const char *path, Tokenizer *tok,
                                     double thresh, int target_byte,
                                     int batches) {
    HeldoutConfig hcfg = {
        /* Bias to source-index 4: that source's label sequence is
         * [train, HELDOUT, train, ...] so chunk 1 is held out. */
        .seed        = HM_HELDOUT_SEED + 4,
        .stream      = HM_HELDOUT_STREAM + 4,
        .thresh      = thresh,
        .chunk_bytes = HM_CHUNK_BYTES,
    };
    StreamSourceSpec src = { .path = path, .tokenizer = tok, .weight = 1.0f };
    const int SEQ = 16;
    StreamDataLoader *dl =
        stream_loader_create_heldout(&src, 1, SEQ, 4242ULL, &hcfg);
    if (!dl) return -1;

    int32_t toks[16], tgts[16];
    uint8_t decoded[256];
    int64_t count = 0;
    for (int b = 0; b < batches; b++) {
        if (stream_loader_next(dl, toks, tgts) != 0) {
            stream_loader_destroy(dl);
            return -1;
        }
        int64_t nb = tokenizer_decode(tok, toks, SEQ, decoded, sizeof(decoded));
        if (nb < 0) {
            stream_loader_destroy(dl);
            return -1;
        }
        for (int64_t i = 0; i < nb; i++) {
            if (decoded[i] == target_byte) count++;
        }
    }
    stream_loader_destroy(dl);
    return count;
}

/* B-2: the held-out chunk's signature byte must NEVER appear in the training
 * stream when held-out is active on misaligned content. */
static void test_heldout_misaligned_no_leak(void) {
    const char *text_path = "/tmp/test_heldout_misaligned_src.txt";
    const char *tok_path  = "/tmp/test_heldout_misaligned_tok.bin";
    const char sig[3] = { 'A', 'B', 'C' };  /* chunk 1 ('B') is held out */

    ASSERT_EQUAL_INT(0, hm_write_misaligned_source(text_path, sig));

    /* Tokenizer trained over all three signatures + space so every byte is
     * encodable and decode round-trips exactly. */
    const char *train_text =
        "AAAA AAAA AAAA BBBB BBBB BBBB CCCC CCCC CCCC "
        "A B C AB BC CA ABC AAA BBB CCC AA BB CC ";
    Tokenizer *tok = make_tokenizer(train_text, 24, tok_path);
    ASSERT_NOT_NULL(tok);

    /* Enough batches to cycle through the source several times (3 chunks,
     * ~196 KB, SEQ=16). 200000 batches is well past multiple wraps. */
    int64_t leak = hm_count_decoded_byte(text_path, tok, 0.15, 'B', 200000);
    ASSERT_TRUE(leak >= 0);  /* loader/decoder did not error */
    /* THE load-bearing assertion: zero held-out-chunk bytes in the train
     * stream.  RED (leak > 0) on the byte_offset-keyed bug; GREEN (0) after
     * the file-position coordinate fix. */
    ASSERT_EQUAL_INT(0, (int)leak);

    tokenizer_destroy(tok);
    unlink(text_path);
    unlink(tok_path);
}

/* B-2 discriminating control: with thresh=0 (nothing held out) the SAME 'B'
 * signature MUST appear, proving the probe detects present content and the
 * no-leak assertion above is not vacuously satisfied. */
static void test_heldout_misaligned_control(void) {
    const char *text_path = "/tmp/test_heldout_misaligned_ctl_src.txt";
    const char *tok_path  = "/tmp/test_heldout_misaligned_ctl_tok.bin";
    const char sig[3] = { 'A', 'B', 'C' };

    ASSERT_EQUAL_INT(0, hm_write_misaligned_source(text_path, sig));

    const char *train_text =
        "AAAA AAAA AAAA BBBB BBBB BBBB CCCC CCCC CCCC "
        "A B C AB BC CA ABC AAA BBB CCC AA BB CC ";
    Tokenizer *tok = make_tokenizer(train_text, 24, tok_path);
    ASSERT_NOT_NULL(tok);

    /* thresh=0.0 ⇒ u < 0 is always false ⇒ every chunk is TRAIN ⇒ nothing
     * skipped ⇒ chunk 1's 'B' content is read into the stream. */
    int64_t present = hm_count_decoded_byte(text_path, tok, 0.0, 'B', 5000);
    ASSERT_TRUE(present > 0);  /* probe is discriminating */

    tokenizer_destroy(tok);
    unlink(text_path);
    unlink(tok_path);
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("\n========================================\n");
    printf("  Data Loader Tests\n");
    printf("========================================\n\n");

    printf("--- Lifecycle ---\n");
    RUN_TEST(test_data_loader_create_destroy);
    RUN_TEST(test_data_loader_invalid_args);

    printf("\n--- Next-Token Targets ---\n");
    RUN_TEST(test_data_loader_next_token_targets);
    RUN_TEST(test_data_loader_seq_len_one);

    printf("\n--- Position Tracking ---\n");
    RUN_TEST(test_data_loader_position_tracking);

    printf("\n--- Epoch Boundary & Reset ---\n");
    RUN_TEST(test_data_loader_reset);

    printf("\n--- Load & Tokenize ---\n");
    RUN_TEST(test_load_and_tokenize);

    printf("\n--- Config Variants ---\n");
    RUN_TEST(test_config_small);
    RUN_TEST(test_config_medium);

    printf("\n--- Stream Loader: lifecycle ---\n");
    RUN_TEST(test_stream_loader_create_destroy);
    RUN_TEST(test_stream_loader_invalid_args);

    printf("\n--- Stream Loader: batches ---\n");
    RUN_TEST(test_stream_loader_single_source_next);

    printf("\n--- Stream Loader: RNG properties ---\n");
    RUN_TEST(test_stream_loader_determinism);
    RUN_TEST(test_stream_loader_different_seed);
    RUN_TEST(test_stream_loader_weighted_mixing);

    printf("\n--- Stream Loader: wraparound ---\n");
    RUN_TEST(test_stream_loader_wraparound);

    printf("\n--- Stream Loader: resume ---\n");
    RUN_TEST(test_stream_loader_save_restore);
    RUN_TEST(test_stream_loader_restore_mismatch);

    printf("\n--- Held-out: misaligned-content leak (B-2) ---\n");
    RUN_TEST(test_heldout_misaligned_no_leak);
    RUN_TEST(test_heldout_misaligned_control);

    TEST_REPORT();
}
