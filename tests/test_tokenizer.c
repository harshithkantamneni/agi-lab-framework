/* test_tokenizer.c -- Tests for BPE tokenizer.
 *
 * Tests cover:
 *   1. Create/destroy lifecycle
 *   2. Byte-level encode (no merges)
 *   3. Byte-level decode (no merges)
 *   4. Training on simple repeated patterns
 *   5. Encoding with trained merges
 *   6. Roundtrip encode -> decode
 *   7. Save to file, load, verify equivalence
 *   8. Empty input handling
 *   9. Multi-byte UTF-8 text
 *
 * Built with: clang -std=c17 -g -O0 -mcpu=apple-m3 -fsanitize=address,undefined
 *             -I src/core tests/test_tokenizer.c src/core/tokenizer.c
 *             -o build/tests/test_tokenizer
 */

#include "../src/tests/unity.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ========================================================================
 * CREATE / DESTROY
 * ======================================================================== */

static void test_tokenizer_create_destroy(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    /* Must have exactly 256 byte-level tokens */
    ASSERT_EQUAL_INT(256, tok->vocab_size);

    /* Each byte token should be a single byte matching its ID */
    for (int i = 0; i < 256; i++) {
        ASSERT_NOT_NULL(tok->vocab[i]);
        ASSERT_EQUAL_INT(1, tok->vocab_lengths[i]);
        ASSERT_EQUAL_INT(i, (int)tok->vocab[i][0]);
    }

    /* No merges yet */
    ASSERT_EQUAL_INT(0, tok->n_merges);

    /* Special tokens are unset (-1) */
    ASSERT_EQUAL_INT(-1, tok->pad_id);
    ASSERT_EQUAL_INT(-1, tok->unk_id);
    ASSERT_EQUAL_INT(-1, tok->bos_id);
    ASSERT_EQUAL_INT(-1, tok->eos_id);

    tokenizer_destroy(tok);
}

/* ========================================================================
 * BYTE-LEVEL ENCODE (no merges)
 * ======================================================================== */

static void test_tokenizer_encode_bytes(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    const char *text = "Hello";
    int64_t text_len = (int64_t)strlen(text);
    int32_t ids[64];

    int32_t n = tokenizer_encode(tok, (const uint8_t *)text, text_len, ids, 64);
    ASSERT_EQUAL_INT(5, n);

    /* Each character maps to its ASCII value */
    ASSERT_EQUAL_INT('H', ids[0]);
    ASSERT_EQUAL_INT('e', ids[1]);
    ASSERT_EQUAL_INT('l', ids[2]);
    ASSERT_EQUAL_INT('l', ids[3]);
    ASSERT_EQUAL_INT('o', ids[4]);

    tokenizer_destroy(tok);
}

/* ========================================================================
 * BYTE-LEVEL DECODE (no merges)
 * ======================================================================== */

static void test_tokenizer_decode_bytes(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    /* Encode "Test" and then decode back */
    int32_t ids[] = {'T', 'e', 's', 't'};
    uint8_t out[64];
    int64_t n = tokenizer_decode(tok, ids, 4, out, 64);
    ASSERT_EQUAL_INT(4, (int)n);
    ASSERT_TRUE(memcmp(out, "Test", 4) == 0);

    tokenizer_destroy(tok);
}

/* ========================================================================
 * TRAINING ON SIMPLE PATTERN
 * ======================================================================== */

static void test_tokenizer_train_simple(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    /* "aaabbaab" repeated many times: "aa" should be the most frequent pair */
    const char *pattern = "aaabbaab";
    size_t pat_len = strlen(pattern);
    size_t repeats = 100;
    size_t total_len = pat_len * repeats;

    uint8_t *text = (uint8_t *)malloc(total_len);
    ASSERT_NOT_NULL(text);
    for (size_t i = 0; i < repeats; i++) {
        memcpy(text + i * pat_len, pattern, pat_len);
    }

    /* Train just 1 merge to check what gets merged first */
    int result = tokenizer_train(tok, text, (int64_t)total_len, 1);
    ASSERT_EQUAL_INT(0, result);

    /* Should have learned exactly 1 merge */
    ASSERT_EQUAL_INT(1, tok->n_merges);
    ASSERT_EQUAL_INT(257, tok->vocab_size);

    /* The first merge should be ('a', 'a') since it appears most often.
     * In "aaabbaab": positions (0,1) give "aa", then (3,4) give "bb",
     * then (5,6) give "aa", then (6,7) give "ab".
     * Count: aa=2*100=200, ab=1*100=100, bb=1*100=100
     * But with the greedy left-to-right replacement, the count of "aa" in
     * "aaabbaab" is: positions 0-1 "aa" and 5-6 "aa" = 2 per repeat.
     * Actually for "aaa": positions 0-1 and 1-2, so "aa" appears at 0-1, 1-2,
     * and 5-6 = 3 per repeat.  "bb" appears at 3-4 = 1 per repeat.
     * "ab" appears at 2-3 and 6-7 = 2 per repeat.
     * So aa=300, ab=200, bb=100.  "aa" wins. */
    ASSERT_EQUAL_INT('a', tok->merges[0].left);
    ASSERT_EQUAL_INT('a', tok->merges[0].right);
    ASSERT_EQUAL_INT(256, tok->merges[0].merged_id);

    /* The merged token should be "aa" (2 bytes) */
    ASSERT_EQUAL_INT(2, tok->vocab_lengths[256]);
    ASSERT_TRUE(tok->vocab[256][0] == 'a');
    ASSERT_TRUE(tok->vocab[256][1] == 'a');

    free(text);
    tokenizer_destroy(tok);
}

/* ========================================================================
 * ENCODING WITH MERGES
 * ======================================================================== */

static void test_tokenizer_encode_with_merges(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    /* Train on repeated "abab" to learn the merge (a,b) -> 256 */
    const char *pattern = "abab";
    size_t pat_len = strlen(pattern);
    size_t repeats = 100;
    size_t total_len = pat_len * repeats;

    uint8_t *text = (uint8_t *)malloc(total_len);
    ASSERT_NOT_NULL(text);
    for (size_t i = 0; i < repeats; i++) {
        memcpy(text + i * pat_len, pattern, pat_len);
    }

    /* Train 1 merge: should learn (a,b) -> 256 */
    int result = tokenizer_train(tok, text, (int64_t)total_len, 1);
    ASSERT_EQUAL_INT(0, result);
    ASSERT_EQUAL_INT(1, tok->n_merges);

    /* Now encode "abab" -- should produce 2 tokens instead of 4 */
    int32_t ids[64];
    int32_t n = tokenizer_encode(tok, (const uint8_t *)"abab", 4, ids, 64);
    ASSERT_EQUAL_INT(2, n);
    ASSERT_EQUAL_INT(256, ids[0]);
    ASSERT_EQUAL_INT(256, ids[1]);

    /* Encode "abc" -- only the "ab" part merges, "c" stays */
    n = tokenizer_encode(tok, (const uint8_t *)"abc", 3, ids, 64);
    ASSERT_EQUAL_INT(2, n);
    ASSERT_EQUAL_INT(256, ids[0]);
    ASSERT_EQUAL_INT('c', ids[1]);

    /* Encode "cab" -- "c" stays, then "ab" merges */
    n = tokenizer_encode(tok, (const uint8_t *)"cab", 3, ids, 64);
    ASSERT_EQUAL_INT(2, n);
    ASSERT_EQUAL_INT('c', ids[0]);
    ASSERT_EQUAL_INT(256, ids[1]);

    free(text);
    tokenizer_destroy(tok);
}

/* ========================================================================
 * ROUNDTRIP: ENCODE -> DECODE
 * ======================================================================== */

static void test_tokenizer_roundtrip(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    /* Train some merges on a corpus */
    const char *corpus =
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. "
        "the quick brown fox jumps over the lazy dog. ";
    int64_t corpus_len = (int64_t)strlen(corpus);

    int result = tokenizer_train(tok, (const uint8_t *)corpus, corpus_len, 10);
    ASSERT_EQUAL_INT(0, result);

    /* Now roundtrip various texts */
    const char *tests[] = {
        "the quick brown fox",
        "hello world",
        "abcdefghij",
        "the the the",
    };
    int n_tests = 4;

    for (int t = 0; t < n_tests; t++) {
        const char *text = tests[t];
        int64_t text_len = (int64_t)strlen(text);

        int32_t ids[256];
        int32_t n_tokens =
            tokenizer_encode(tok, (const uint8_t *)text, text_len, ids, 256);
        ASSERT_TRUE(n_tokens > 0);

        uint8_t decoded[256];
        int64_t dec_len = tokenizer_decode(tok, ids, n_tokens, decoded, 256);
        ASSERT_EQUAL_INT((int)text_len, (int)dec_len);
        ASSERT_TRUE(memcmp(decoded, text, (size_t)text_len) == 0);
    }

    tokenizer_destroy(tok);
}

/* ========================================================================
 * SAVE / LOAD
 * ======================================================================== */

static void test_tokenizer_save_load(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    /* Train on a corpus */
    const char *corpus =
        "abababababababababababababab"
        "cdcdcdcdcdcdcdcdcdcdcdcdcd"
        "abababababababababababababab";
    int64_t corpus_len = (int64_t)strlen(corpus);

    int result =
        tokenizer_train(tok, (const uint8_t *)corpus, corpus_len, 5);
    ASSERT_EQUAL_INT(0, result);

    /* Set special tokens */
    tok->pad_id = 0;
    tok->bos_id = 1;
    tok->eos_id = 2;

    /* Encode a test string with the trained tokenizer */
    const char *test_text = "ababcdcd";
    int64_t test_len = (int64_t)strlen(test_text);
    int32_t ids_orig[64];
    int32_t n_orig = tokenizer_encode(tok, (const uint8_t *)test_text,
                                      test_len, ids_orig, 64);
    ASSERT_TRUE(n_orig > 0);

    /* Save to temp file */
    const char *tmp_path = "/tmp/test_tokenizer_bpe.bin";
    result = tokenizer_save(tok, tmp_path);
    ASSERT_EQUAL_INT(0, result);

    /* Load from file */
    Tokenizer *loaded = tokenizer_load(tmp_path);
    ASSERT_NOT_NULL(loaded);

    /* Verify metadata */
    ASSERT_EQUAL_INT(tok->vocab_size, loaded->vocab_size);
    ASSERT_EQUAL_INT(tok->n_merges, loaded->n_merges);
    ASSERT_EQUAL_INT(tok->pad_id, loaded->pad_id);
    ASSERT_EQUAL_INT(tok->bos_id, loaded->bos_id);
    ASSERT_EQUAL_INT(tok->eos_id, loaded->eos_id);

    /* Encode same text with loaded tokenizer -- must match */
    int32_t ids_loaded[64];
    int32_t n_loaded = tokenizer_encode(loaded, (const uint8_t *)test_text,
                                        test_len, ids_loaded, 64);
    ASSERT_EQUAL_INT(n_orig, n_loaded);
    for (int32_t i = 0; i < n_orig; i++) {
        ASSERT_EQUAL_INT(ids_orig[i], ids_loaded[i]);
    }

    /* Clean up */
    unlink(tmp_path);
    tokenizer_destroy(loaded);
    tokenizer_destroy(tok);
}

/* ========================================================================
 * EMPTY INPUT
 * ======================================================================== */

static void test_tokenizer_empty_input(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    int32_t ids[64];
    uint8_t decoded[64];

    /* Encode empty text */
    int32_t n = tokenizer_encode(tok, (const uint8_t *)"", 0, ids, 64);
    ASSERT_EQUAL_INT(0, n);

    /* Encode NULL text */
    n = tokenizer_encode(tok, NULL, 0, ids, 64);
    ASSERT_EQUAL_INT(0, n);

    /* Decode zero tokens */
    int64_t d = tokenizer_decode(tok, ids, 0, decoded, 64);
    ASSERT_EQUAL_INT(0, (int)d);

    /* Decode NULL ids */
    d = tokenizer_decode(tok, NULL, 0, decoded, 64);
    ASSERT_EQUAL_INT(0, (int)d);

    /* Train on empty text should fail gracefully */
    int result = tokenizer_train(tok, (const uint8_t *)"", 0, 10);
    ASSERT_EQUAL_INT(-1, result);

    tokenizer_destroy(tok);
}

/* ========================================================================
 * UTF-8 TEXT (byte-level transparency)
 * ======================================================================== */

static void test_tokenizer_utf8(void) {
    Tokenizer *tok = tokenizer_create();
    ASSERT_NOT_NULL(tok);

    /* Multi-byte UTF-8 characters:
     *   U+00E9 (e with acute) = 0xC3 0xA9  (2 bytes)
     *   U+4E16 (Chinese "world") = 0xE4 0xB8 0x96  (3 bytes)
     *   U+1F600 (grinning face) = 0xF0 0x9F 0x98 0x80  (4 bytes)
     */
    const uint8_t utf8_text[] = {
        0xC3, 0xA9,             /* e-acute */
        0xE4, 0xB8, 0x96,      /* Chinese character */
        0xF0, 0x9F, 0x98, 0x80 /* emoji */
    };
    int64_t utf8_len = sizeof(utf8_text);

    /* Encode: should produce one token per byte */
    int32_t ids[64];
    int32_t n = tokenizer_encode(tok, utf8_text, utf8_len, ids, 64);
    ASSERT_EQUAL_INT(9, n);

    /* Each token ID should match the byte value */
    for (int i = 0; i < 9; i++) {
        ASSERT_EQUAL_INT((int)utf8_text[i], ids[i]);
    }

    /* Roundtrip: decode should produce the exact same bytes */
    uint8_t decoded[64];
    int64_t dec_len = tokenizer_decode(tok, ids, n, decoded, 64);
    ASSERT_EQUAL_INT(9, (int)dec_len);
    ASSERT_TRUE(memcmp(decoded, utf8_text, 9) == 0);

    tokenizer_destroy(tok);
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("\n========================================\n");
    printf("  BPE Tokenizer Tests\n");
    printf("========================================\n\n");

    printf("--- Lifecycle ---\n");
    RUN_TEST(test_tokenizer_create_destroy);

    printf("\n--- Byte-level Encode/Decode ---\n");
    RUN_TEST(test_tokenizer_encode_bytes);
    RUN_TEST(test_tokenizer_decode_bytes);

    printf("\n--- Training ---\n");
    RUN_TEST(test_tokenizer_train_simple);
    RUN_TEST(test_tokenizer_encode_with_merges);

    printf("\n--- Roundtrip ---\n");
    RUN_TEST(test_tokenizer_roundtrip);

    printf("\n--- Save/Load ---\n");
    RUN_TEST(test_tokenizer_save_load);

    printf("\n--- Edge Cases ---\n");
    RUN_TEST(test_tokenizer_empty_input);
    RUN_TEST(test_tokenizer_utf8);

    TEST_REPORT();
}
