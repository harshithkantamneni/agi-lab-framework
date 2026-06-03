/* validate_tokenizer_32k.c -- Binary-format + round-trip validation for
 * a tokenizer file (typically tokenizer_32k.bin) using the production
 * tokenizer_load() / tokenizer_encode() / tokenizer_decode() API.
 *
 * This is the "drop-in compat" check: if this binary runs clean with the
 * production tokenizer.c it means the Python trainer produced a file whose
 * format is byte-compatible with the C loader.
 *
 * Build:
 *   clang -std=c17 -Wall -Wextra -Werror -O2 -mcpu=apple-m3 -Isrc/core \
 *       tests/validate_tokenizer_32k.c src/core/tokenizer.c \
 *       -o build/tests/validate_tokenizer_32k
 *
 * Run:
 *   build/tests/validate_tokenizer_32k data/training/tokenizer_32k.bin
 */

#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *SAMPLES[] = {
    "The quick brown fox jumps over the lazy dog.",
    ("def fibonacci(n):\n    if n <= 1:\n        return n\n"
     "    return fibonacci(n-1) + fibonacci(n-2)"),
    "Solve for x: 3x^2 + 5x - 2 = 0",
    "Hello, world!",
    "Machine learning and artificial intelligence are transforming the world.",
    "The integral of x^2 dx = x^3/3 + C",
    "import numpy as np\nresult = np.array([1, 2, 3])",
    "class Foo:\n    def __init__(self, x):\n        self.x = x\n",
    "Question: What is 12 * 13?\nAnswer: 156",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Unicode: naive=>naïve, cafe=>café, emoji alpha=Α omega=Ω",
    "if x == 42: print('answer')  # hitchhiker's",
};

static const int N_SAMPLES = (int)(sizeof(SAMPLES) / sizeof(SAMPLES[0]));

static int roundtrip_one(const Tokenizer *tok, const char *text,
                         int *out_n_tokens) {
    int64_t tlen = (int64_t)strlen(text);
    int32_t cap = (int32_t)(tlen + 16);
    int32_t *ids = (int32_t *)malloc((size_t)cap * sizeof(int32_t));
    if (!ids) {
        return -1;
    }

    int32_t n = tokenizer_encode(tok, (const uint8_t *)text, tlen, ids, cap);
    if (n < 0) {
        free(ids);
        return -1;
    }

    uint8_t *dec = (uint8_t *)malloc((size_t)(tlen + 32));
    if (!dec) {
        free(ids);
        return -1;
    }
    int64_t dlen = tokenizer_decode(tok, ids, n, dec, tlen + 32);
    if (dlen < 0 || dlen != tlen || memcmp(dec, text, (size_t)tlen) != 0) {
        fprintf(stderr, "  MISMATCH: in=\"%s\" decoded_len=%lld\n", text,
                (long long)dlen);
        free(ids);
        free(dec);
        return -1;
    }
    *out_n_tokens = (int)n;
    free(ids);
    free(dec);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tokenizer.bin>\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];

    Tokenizer *tok = tokenizer_load(path);
    if (!tok) {
        fprintf(stderr, "FATAL: tokenizer_load failed for %s\n", path);
        return 1;
    }

    printf("Loaded: %s\n", path);
    printf("  vocab_size = %d\n", tok->vocab_size);
    printf("  n_merges   = %d\n", tok->n_merges);
    printf("  pad=%d unk=%d bos=%d eos=%d\n", tok->pad_id, tok->unk_id,
           tok->bos_id, tok->eos_id);

    int failures = 0;
    printf("\nRound-trip validation on %d samples:\n", N_SAMPLES);
    for (int i = 0; i < N_SAMPLES; i++) {
        int n_tokens = 0;
        int r = roundtrip_one(tok, SAMPLES[i], &n_tokens);
        int nbytes = (int)strlen(SAMPLES[i]);
        double ratio = (n_tokens > 0) ? (double)nbytes / (double)n_tokens : 0.0;
        printf("  [%2d] %s %d bytes -> %d tokens (%.2f B/tok)\n", i + 1,
               (r == 0) ? "OK  " : "FAIL", nbytes, n_tokens, ratio);
        if (r != 0) failures++;
    }

    tokenizer_destroy(tok);

    if (failures != 0) {
        fprintf(stderr, "\n%d failures\n", failures);
        return 1;
    }
    printf("\nALL PASS\n");
    return 0;
}
