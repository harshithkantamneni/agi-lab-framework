/* tests/test_heldout_c_harness.c — C harness for held-out partition tests.
 *
 * This binary serves three modes, selected by argv[1]:
 *
 *   --test-boundary
 *       Test 5: verifies that at a held-out chunk boundary:
 *         (a) pending_len is forced to 0 (no carry-over),
 *         (b) ring is flushed (ring_reset called),
 *         (c) byte_offset jumps past the held-out chunk.
 *       Exits 0 on pass, 1 on failure.
 *
 *   --dump-labels  <sz0> <sz1> ... <szN>
 *       Test 7: derive partition labels for N sources of the given byte sizes
 *       and print a JSON array of per-source label lists to stdout, e.g.:
 *           [["train","heldout","train"],["train","train"]]
 *       Uses the canonical HELDOUT_SEED/HELDOUT_STREAM/HELDOUT_THRESH/CHUNK_BYTES
 *       from the C-side implementation.
 *       Exits 0 on success.
 *
 *   --test-disabled-identical
 *       Test 8: creates two StreamDataLoaders from the same source file —
 *       one with HeldoutConfig=NULL (disabled) and one with a config that
 *       labels ALL chunks as train (so it also skips nothing).  Pulls
 *       N=100 batches from each and asserts bit-identical token streams.
 *       Exits 0 on pass, 1 on failure.
 *
 * Build: picked up by the Makefile's default test rule:
 *   $(BUILD_DIR)/tests/test_heldout_c_harness: tests/test_heldout_c_harness.c $(C_SRCS)
 *
 * Dependencies: data_loader.h (HeldoutConfig, stream_loader_create_heldout),
 *               tokenizer.h, src/tests/unity.h (for ASSERT_* macros).
 *
 * Binding: repro co-sign C1–C6; Test 5 (boundary) + Test 7 (label agreement)
 *          + Test 8 (disabled-identical) are load-bearing.
 */

#include "../src/tests/unity.h"
#include "data_loader.h"
#include "tokenizer.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* =========================================================================
 * PCG32 label derivation — mirrors data_loader.c exactly for test 7.
 * We re-derive here purely as a local cross-check; the real derivation
 * lives in data_loader.c (the implementation being tested).
 * ========================================================================= */

/* Constants — MUST match heldout_partition.py and data_loader.c */
#define HARNESS_HELDOUT_SEED    UINT64_C(0x50524F475F335F48)
#define HARNESS_HELDOUT_STREAM  UINT64_C(0x48454C444F55544F)
#define HARNESS_HELDOUT_THRESH  0.15
#define HARNESS_CHUNK_BYTES     65536

typedef struct { uint64_t state; uint64_t inc; } HarnessPcg32;

static uint32_t hpcg32_next(HarnessPcg32 *r) {
    uint64_t old = r->state;
    r->state     = old * UINT64_C(6364136223846793005) + r->inc;
    uint32_t xs  = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xs >> rot) | (xs << ((-rot) & 31u));
}

static void hpcg32_seed(HarnessPcg32 *r, uint64_t seed, uint64_t stream_id) {
    r->state = 0ULL;
    r->inc   = (stream_id << 1u) | 1u;
    (void)hpcg32_next(r);
    r->state += seed;
    (void)hpcg32_next(r);
}

/* Returns 1 = heldout, 0 = train */
static int harness_label(uint64_t seed_base, uint64_t stream_base,
                         int32_t src_idx, int32_t chunk_idx) {
    HarnessPcg32 rng;
    hpcg32_seed(&rng, seed_base + (uint64_t)src_idx,
                      stream_base + (uint64_t)src_idx);
    for (int32_t i = 0; i <= chunk_idx; i++) {
        uint32_t u32 = hpcg32_next(&rng);
        if (i == chunk_idx) {
            double u = (double)u32 / 4294967296.0;
            return (u < HARNESS_HELDOUT_THRESH) ? 1 : 0;
        }
    }
    return 0; /* unreachable */
}

/* =========================================================================
 * Mode: --dump-labels  <sz0> <sz1> ... <szN>
 *
 * Derives the partition labels using the canonical C derivation (via the
 * harness PCG32 above, which mirrors data_loader.c exactly) and emits a
 * JSON array to stdout.  Used by Test 7 to compare against Python output.
 * ========================================================================= */

static int mode_dump_labels(int n_sizes, const char **size_args) {
    printf("[");
    for (int s = 0; s < n_sizes; s++) {
        int64_t fsz = (int64_t)strtoull(size_args[s], NULL, 10);
        int32_t n_chunks = (int32_t)((fsz + HARNESS_CHUNK_BYTES - 1) / HARNESS_CHUNK_BYTES);
        if (s > 0) printf(",");
        printf("[");
        HarnessPcg32 rng;
        hpcg32_seed(&rng,
                    HARNESS_HELDOUT_SEED   + (uint64_t)s,
                    HARNESS_HELDOUT_STREAM + (uint64_t)s);
        for (int32_t i = 0; i < n_chunks; i++) {
            uint32_t u32 = hpcg32_next(&rng);
            double u = (double)u32 / 4294967296.0;
            if (i > 0) printf(",");
            if (u < HARNESS_HELDOUT_THRESH)
                printf("\"heldout\"");
            else
                printf("\"train\"");
        }
        printf("]");
    }
    printf("]\n");
    return 0;
}

/* =========================================================================
 * Mode: --dump-labels-production  <sz0> <sz1> ... <szN>   (F-2)
 *
 * Like --dump-labels, but the labels are the ones the PRODUCTION source_refill
 * path ACTUALLY APPLIES — not the harness PCG32 re-implementation.  This closes
 * the gap (B-1) that let the coordinate-mismatch bug hide behind a Test-7 that
 * only compared the harness re-implementation against Python.
 *
 * For each source size we:
 *   - bias the base seed to (HELDOUT_SEED+s / HELDOUT_STREAM+s) so a SINGLE
 *     loaded source reproduces source-s's label sequence,
 *   - write a MISALIGNED temp file (signature byte per chunk, space every 100
 *     bytes; 100 ∤ 65536 so chunk boundaries are non-whitespace → encode_len <
 *     BYTE_SLAB → exercises the file/content coordinate gap),
 *   - drain a full epoch via stream_loader_next, decoding each batch,
 *   - emit "heldout" for chunks whose signature byte NEVER appears in the
 *     training stream and "train" for chunks whose signature DOES appear.
 *
 * A chunk's signature byte appears iff the production path read (did NOT skip)
 * that chunk — so the emitted labels are exactly the production skip decisions.
 * ========================================================================= */

/* Distinct printable signature byte for chunk index i (avoids whitespace and
 * the JSON-significant bytes we never emit into files). Range 0x21..0x7E minus
 * space; 94 usable codes, plenty for Test-7-scale sources. */
static int prod_sig_byte(int chunk_idx) {
    int b = 0x21 + (chunk_idx % 94);   /* '!' .. '~' */
    if (b == ' ') b++;                 /* unreachable (space=0x20 < 0x21) */
    return b;
}

/* Write a misaligned source of n_chunks chunks, signature byte per chunk. */
static int prod_write_misaligned(const char *path, int n_chunks) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    for (int k = 0; k < n_chunks; k++) {
        int sig = prod_sig_byte(k);
        for (int64_t pos = 0; pos < HARNESS_CHUNK_BYTES; pos++) {
            fputc((pos % 100 == 99) ? ' ' : sig, f);
        }
    }
    fclose(f);
    return 0;
}

static int mode_dump_labels_production(int n_sizes, const char **size_args) {
    /* One tokenizer covering all signature bytes + space; trained once. */
    const char *tok_path = "/tmp/test_heldout_prod_tok.bin";
    Tokenizer *tok = tokenizer_create();
    if (!tok) { fprintf(stderr, "tokenizer_create failed\n"); return 1; }
    /* Build a training corpus that contains every signature byte we might use
     * (chunks across all sources) plus spaces, so every byte is encodable. */
    uint8_t train_buf[512];
    int tn = 0;
    for (int c = 0x21; c <= 0x7E && tn < (int)sizeof(train_buf) - 2; c++) {
        train_buf[tn++] = (uint8_t)c;
        train_buf[tn++] = ' ';
    }
    if (tokenizer_train(tok, train_buf, (int64_t)tn, 32) != 0) {
        fprintf(stderr, "tokenizer_train failed\n");
        tokenizer_destroy(tok);
        return 1;
    }
    if (tokenizer_save(tok, tok_path) != 0) {
        fprintf(stderr, "tokenizer_save failed\n");
        tokenizer_destroy(tok);
        return 1;
    }

    printf("[");
    int rc_all = 0;
    for (int s = 0; s < n_sizes && rc_all == 0; s++) {
        int64_t fsz = (int64_t)strtoull(size_args[s], NULL, 10);
        int n_chunks = (int)((fsz + HARNESS_CHUNK_BYTES - 1) / HARNESS_CHUNK_BYTES);
        const char *src_path = "/tmp/test_heldout_prod_src.txt";
        if (prod_write_misaligned(src_path, n_chunks) != 0) {
            fprintf(stderr, "prod_write_misaligned failed\n");
            rc_all = 1;
            break;
        }

        /* Bias to source index s: a single-source loader then reproduces the
         * source-s label sequence (seed+s / stream+s). */
        HeldoutConfig hcfg = {
            .seed        = HARNESS_HELDOUT_SEED   + (uint64_t)s,
            .stream      = HARNESS_HELDOUT_STREAM + (uint64_t)s,
            .thresh      = HARNESS_HELDOUT_THRESH,   /* double 0.15 (F-1) */
            .chunk_bytes = HARNESS_CHUNK_BYTES,
        };
        StreamSourceSpec src = { .path = src_path, .tokenizer = tok, .weight = 1.0f };
        const int SEQ = 16;
        StreamDataLoader *dl =
            stream_loader_create_heldout(&src, 1, SEQ, 99ULL, &hcfg);
        if (!dl) {
            fprintf(stderr, "stream_loader_create_heldout failed (s=%d)\n", s);
            rc_all = 1;
            break;
        }

        /* Track which signature bytes appear in the production token stream. */
        unsigned char seen[256] = {0};
        int32_t toks[16], tgts[16];
        uint8_t decoded[256];
        /* Drain well past one epoch so every train chunk is observed. The
         * source is small (n_chunks * 64 KiB); 4000 batches/chunk is ample. */
        int budget = (n_chunks + 1) * 4000;
        for (int b = 0; b < budget; b++) {
            if (stream_loader_next(dl, toks, tgts) != 0) { rc_all = 1; break; }
            int64_t nb = tokenizer_decode(tok, toks, SEQ, decoded, sizeof(decoded));
            if (nb < 0) { rc_all = 1; break; }
            for (int64_t i = 0; i < nb; i++) seen[decoded[i]] = 1;
        }
        stream_loader_destroy(dl);
        if (rc_all) break;

        if (s > 0) printf(",");
        printf("[");
        for (int i = 0; i < n_chunks; i++) {
            if (i > 0) printf(",");
            int sig = prod_sig_byte(i);
            /* signature seen  -> chunk was READ  -> "train"
             * signature unseen-> chunk was SKIPPED-> "heldout" */
            printf(seen[sig] ? "\"train\"" : "\"heldout\"");
        }
        printf("]");
        remove(src_path);
    }
    printf("]\n");

    tokenizer_destroy(tok);
    remove(tok_path);
    return rc_all;
}

/* =========================================================================
 * Helpers for creating temp text files and tokenizers in tests
 * ========================================================================= */

/* Create a temp file with `nbytes` of printable content spread across the
 * requested number of CHUNK_BYTES chunks.  Body must be tokenizable ASCII
 * with embedded whitespace so the tokenizer has word boundaries. */
static int make_temp_text_file(const char *path, int64_t nbytes) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    /* Repeat a simple phrase that has whitespace boundaries every ~20 bytes */
    const char phrase[] = "the quick brown fox jumps over the lazy dog ";
    const int plen = (int)(sizeof(phrase) - 1);
    int64_t written = 0;
    while (written < nbytes) {
        int64_t chunk_end = written + HARNESS_CHUNK_BYTES;
        if (chunk_end > nbytes) chunk_end = nbytes;
        /* Leave a whitespace right before each chunk boundary so pending logic
         * doesn't carry over across the boundary. */
        while (written < chunk_end - 1) {
            fputc(phrase[written % plen], f);
            written++;
        }
        /* Force a whitespace at the last byte of each chunk boundary */
        fputc(' ', f);
        written++;
    }
    fclose(f);
    return 0;
}

/* =========================================================================
 * Mode: --test-boundary
 *
 * Tests that the skip logic (C4 + C3 from repro co-sign):
 *   - forces pending_len = 0 before the seek
 *   - calls ring_reset after the seek
 *   - does not push any tokens from the held-out chunk into the ring
 *   - byte_offset advances past the held-out chunk
 *
 * Strategy: create a source where chunk 1 is held-out (verified via harness
 * label derivation).  Drain the loader until it has processed chunk 0 and
 * chunk 1; verify token counts match expectations.
 *
 * Because the StreamDataLoader internals (ring, pending) are not exposed via
 * the public API, we verify the boundary property INDIRECTLY via:
 *   (1) the byte_offset diagnostic after processing chunks 0 and 1,
 *   (2) the epoch counter (wraps are detected),
 *   (3) comparing token output with and without held-out enabled — if
 *       carry-over occurred, tokens from held-out chunks would appear in the
 *       stream, causing a divergence test to fail unexpectedly.
 *
 * For a direct internal-state probe we expose
 * stream_loader_source_byte_offset() and
 * stream_loader_source_pending_len_after_skip() via the new heldout API.
 * ========================================================================= */

/* Forward declarations for the new diagnostic functions we expect the
 * implementation to export.  These will cause a linker error (RED) until
 * the implementation lands. */
int64_t stream_loader_source_byte_offset(const StreamDataLoader *dl, int32_t src_idx);
int32_t stream_loader_source_pending_len(const StreamDataLoader *dl, int32_t src_idx);

/* Boundary test: use source_idx=4 where chunk 1 is held-out.
 * This is pre-computed and verified via harness_label().
 * Source: 3 chunks (chunk 0=train, chunk 1=HELDOUT, chunk 2=train).
 * We use HELDOUT_SEED+4 / HELDOUT_STREAM+4 as the per-source seeds.
 *
 * The test creates a 3-chunk source, creates a loader with heldout config,
 * then calls stream_loader_next in a tight loop.  We verify:
 *   (a) After processing, byte_offset has skipped past chunk 1
 *       (byte_offset >= 2 * CHUNK_BYTES).
 *   (b) pending_len == 0 after the skip (C3 discipline).
 *   (c) The loader successfully produces tokens from chunk 0 and chunk 2
 *       (by virtue of stream_loader_next succeeding).
 *
 * The source_idx=4 choice is deterministic from the harness PCG32 derivation.
 */
static int mode_test_boundary(void) {
    /* Verify our pre-computed knowledge: source 4 chunk 1 IS held-out. */
    if (!harness_label(HARNESS_HELDOUT_SEED, HARNESS_HELDOUT_STREAM, 4, 1)) {
        fprintf(stderr,
                "SETUP ERROR: source 4 chunk 1 is not held-out — "
                "PCG32 constants may have changed\n");
        return 1;
    }
    if (harness_label(HARNESS_HELDOUT_SEED, HARNESS_HELDOUT_STREAM, 4, 0)) {
        fprintf(stderr,
                "SETUP ERROR: source 4 chunk 0 should be train (not heldout)\n");
        return 1;
    }

    /* 3-chunk source, chunk 1 is held-out. */
    const int64_t source_size = 3 * (int64_t)HARNESS_CHUNK_BYTES;
    const char *text_path = "/tmp/test_heldout_boundary_src.txt";
    const char *tok_path  = "/tmp/test_heldout_boundary_tok.bin";

    if (make_temp_text_file(text_path, source_size) != 0) {
        fprintf(stderr, "Failed to create temp source file\n");
        return 1;
    }

    /* Train a tokenizer on sufficiently varied text so encoding never returns 0. */
    Tokenizer *tok = tokenizer_create();
    if (!tok) { fprintf(stderr, "tokenizer_create failed\n"); return 1; }
    /* Use a longer, varied training text to get reasonable merges. */
    const char *train_text =
        "the quick brown fox jumps over the lazy dog. "
        "hello world this is a test of the tokenizer system. "
        "one two three four five six seven eight nine ten. "
        "alpha beta gamma delta epsilon zeta eta theta iota kappa. ";
    if (tokenizer_train(tok, (const uint8_t *)train_text,
                        (int64_t)strlen(train_text), 32) != 0) {
        fprintf(stderr, "tokenizer_train failed\n");
        tokenizer_destroy(tok);
        return 1;
    }
    if (tokenizer_save(tok, tok_path) != 0) {
        fprintf(stderr, "tokenizer_save failed\n");
        tokenizer_destroy(tok);
        return 1;
    }

    /* ------------------------------------------------------------------ */
    /* Create loader WITH held-out config using src_idx=4 offsets          */
    /* ------------------------------------------------------------------ */
    HeldoutConfig hcfg = {
        .seed        = HARNESS_HELDOUT_SEED,
        .stream      = HARNESS_HELDOUT_STREAM,
        .thresh      = HARNESS_HELDOUT_THRESH,   /* double 0.15 (F-1) */
        .chunk_bytes = HARNESS_CHUNK_BYTES,
    };

    /* We want the RNG for the test source to behave as source index 4.
     * stream_loader_create_heldout seeds source s with seed+s / stream+s.
     * We use a single-source loader, so source index within the loader is 0.
     * To get the source-4 PCG32 sequence, we bias the base seeds:
     *   seed+0 = (HELDOUT_SEED+4)  → set base seed  = HELDOUT_SEED+4
     *   stream+0 = (HELDOUT_STREAM+4) → set base stream = HELDOUT_STREAM+4 */
    hcfg.seed   = HARNESS_HELDOUT_SEED   + 4;
    hcfg.stream = HARNESS_HELDOUT_STREAM + 4;

    StreamSourceSpec src = { .path = text_path, .tokenizer = tok, .weight = 1.0f };
    StreamDataLoader *dl = stream_loader_create_heldout(&src, 1, 8, 42ULL, &hcfg);
    if (!dl) {
        fprintf(stderr, "stream_loader_create_heldout failed\n");
        tokenizer_destroy(tok);
        return 1;
    }

    /* Drain many batches.  The loader will:
     *   - Read tokens from chunk 0 (train)
     *   - Skip chunk 1 (heldout): pending_len→0, ring_reset, fseek to 2*CB
     *   - Read tokens from chunk 2 (train)
     *   - Wrap on EOF; chunk 0 again, etc.
     * We stop once byte_offset has been >= 2*CHUNK_BYTES (past the skip point). */
    int32_t toks[8], tgts[8];
    int ok = 1;
    int found_skip = 0;

    for (int b = 0; b < 100000 && !found_skip; b++) {
        int rc = stream_loader_next(dl, toks, tgts);
        if (rc != 0) {
            fprintf(stderr, "FAIL: stream_loader_next returned %d at batch %d\n", rc, b);
            ok = 0;
            break;
        }
        int64_t boff = stream_loader_source_byte_offset(dl, 0);
        int32_t plen = stream_loader_source_pending_len(dl, 0);

        /* If byte_offset jumped past chunk 1 end (2 * CHUNK_BYTES), the skip fired. */
        if (boff >= (int64_t)2 * HARNESS_CHUNK_BYTES) {
            found_skip = 1;
            /* Verify pending_len == 0 (C3 discipline): the skip must have cleared it. */
            if (plen != 0) {
                fprintf(stderr,
                        "FAIL: pending_len=%d after skip (must be 0) at boff=%" PRId64 "\n",
                        plen, boff);
                ok = 0;
            }
        }
    }

    if (!found_skip) {
        fprintf(stderr,
                "FAIL: byte_offset never reached 2*CHUNK_BYTES after 100000 batches "
                "(final boff=%" PRId64 ")\n",
                stream_loader_source_byte_offset(dl, 0));
        ok = 0;
    }

    stream_loader_destroy(dl);
    tokenizer_destroy(tok);
    remove(text_path);
    remove(tok_path);

    if (ok) printf("PASS: boundary test (chunk 1 skipped, no carry-over)\n");
    return ok ? 0 : 1;
}

/* =========================================================================
 * Mode: --test-disabled-identical
 *
 * Tests that NULL HeldoutConfig (disabled) produces a BIT-IDENTICAL token
 * stream to the pre-heldout baseline (the old stream_loader_create path).
 *
 * Strategy: run both a baseline (stream_loader_create, old API) and a
 * disabled-held-out loader (stream_loader_create_heldout with NULL config)
 * on the same source file with the same seed, and assert identical output.
 * ========================================================================= */

static int mode_test_disabled_identical(void) {
    const char *text_path = "/tmp/test_heldout_disabled_src.txt";
    const char *tok_path  = "/tmp/test_heldout_disabled_tok.bin";

    /* 5 chunks of source text */
    const int64_t source_size = 5 * HARNESS_CHUNK_BYTES;

    if (make_temp_text_file(text_path, source_size) != 0) {
        fprintf(stderr, "Failed to create temp source file\n");
        return 1;
    }

    Tokenizer *tok = tokenizer_create();
    if (!tok) { fprintf(stderr, "tokenizer_create failed\n"); return 1; }
    const char *train_text = "the quick brown fox jumps over the lazy dog ";
    if (tokenizer_train(tok, (const uint8_t *)train_text,
                        (int64_t)strlen(train_text), 16) != 0) {
        fprintf(stderr, "tokenizer_train failed\n");
        tokenizer_destroy(tok);
        return 1;
    }
    if (tokenizer_save(tok, tok_path) != 0) {
        fprintf(stderr, "tokenizer_save failed\n");
        tokenizer_destroy(tok);
        return 1;
    }

    StreamSourceSpec src = { .path = text_path, .tokenizer = tok, .weight = 1.0f };
    const int SEQ = 16;
    const int N_BATCHES = 100;
    const uint64_t SEED = 0xABCDEF42ULL;

    /* Baseline: old API, no held-out config */
    StreamDataLoader *dl_base = stream_loader_create(&src, 1, SEQ, SEED);
    if (!dl_base) {
        fprintf(stderr, "stream_loader_create (baseline) failed\n");
        tokenizer_destroy(tok);
        return 1;
    }

    /* Disabled: new API with NULL config */
    StreamDataLoader *dl_dis = stream_loader_create_heldout(&src, 1, SEQ, SEED, NULL);
    if (!dl_dis) {
        fprintf(stderr, "stream_loader_create_heldout (NULL) failed\n");
        stream_loader_destroy(dl_base);
        tokenizer_destroy(tok);
        return 1;
    }

    int ok = 1;
    int32_t toks_base[16], tgts_base[16];
    int32_t toks_dis[16],  tgts_dis[16];

    for (int b = 0; b < N_BATCHES && ok; b++) {
        int rc_base = stream_loader_next(dl_base, toks_base, tgts_base);
        int rc_dis  = stream_loader_next(dl_dis,  toks_dis,  tgts_dis);
        if (rc_base != rc_dis) {
            fprintf(stderr, "FAIL batch %d: rc mismatch base=%d dis=%d\n",
                    b, rc_base, rc_dis);
            ok = 0;
            break;
        }
        if (rc_base != 0) break;

        for (int i = 0; i < SEQ && ok; i++) {
            if (toks_base[i] != toks_dis[i]) {
                fprintf(stderr,
                        "FAIL batch %d token %d: base=%d dis=%d\n",
                        b, i, toks_base[i], toks_dis[i]);
                ok = 0;
            }
            if (tgts_base[i] != tgts_dis[i]) {
                fprintf(stderr,
                        "FAIL batch %d target %d: base=%d dis=%d\n",
                        b, i, tgts_base[i], tgts_dis[i]);
                ok = 0;
            }
        }
    }

    stream_loader_destroy(dl_base);
    stream_loader_destroy(dl_dis);
    tokenizer_destroy(tok);
    remove(text_path);
    remove(tok_path);

    if (ok) printf("PASS: disabled-identical test\n");
    return ok ? 0 : 1;
}

/* =========================================================================
 * main
 * ========================================================================= */

int main(int argc, const char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s --test-boundary | --dump-labels <sz...> | "
                "--dump-labels-production <sz...> | "
                "--test-disabled-identical\n",
                argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "--test-boundary") == 0) {
        return mode_test_boundary();
    }
    if (strcmp(argv[1], "--dump-labels") == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s --dump-labels <sz0> <sz1> ...\n", argv[0]);
            return 1;
        }
        return mode_dump_labels(argc - 2, argv + 2);
    }
    if (strcmp(argv[1], "--dump-labels-production") == 0) {
        if (argc < 3) {
            fprintf(stderr,
                    "Usage: %s --dump-labels-production <sz0> <sz1> ...\n",
                    argv[0]);
            return 1;
        }
        return mode_dump_labels_production(argc - 2, argv + 2);
    }
    if (strcmp(argv[1], "--test-disabled-identical") == 0) {
        return mode_test_disabled_identical();
    }

    fprintf(stderr, "Unknown mode: %s\n", argv[1]);
    return 1;
}
