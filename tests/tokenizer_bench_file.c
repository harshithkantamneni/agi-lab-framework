/* tokenizer_bench_file.c -- Measure compression ratio of a tokenizer on a
 * single text file (read up to --max-bytes from the middle, aligned to newline).
 *
 * Build:
 *   clang -std=c17 -Wall -Wextra -Werror -O2 -mcpu=apple-m3 -Isrc/core \
 *       tests/tokenizer_bench_file.c src/core/tokenizer.c \
 *       -o build/tests/tokenizer_bench_file
 *
 * Usage:
 *   build/tests/tokenizer_bench_file <tokenizer.bin> <corpus.txt> [max_bytes]
 *     max_bytes: default 2000000 (2 MB)
 */

#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s <tokenizer.bin> <corpus.txt> [max_bytes]\n",
                argv[0]);
        return 1;
    }
    const char *tok_path = argv[1];
    const char *corpus_path = argv[2];
    int64_t max_bytes = (argc >= 4) ? (int64_t)atoll(argv[3]) : 2000000;

    Tokenizer *tok = tokenizer_load(tok_path);
    if (!tok) {
        fprintf(stderr, "FATAL: load %s failed\n", tok_path);
        return 1;
    }

    struct stat st;
    if (stat(corpus_path, &st) != 0) {
        fprintf(stderr, "FATAL: stat %s failed\n", corpus_path);
        tokenizer_destroy(tok);
        return 1;
    }
    int64_t file_size = (int64_t)st.st_size;
    int64_t sample = (file_size < max_bytes) ? file_size : max_bytes;
    int64_t offset = 0;
    if (file_size > max_bytes) {
        offset = file_size / 2 - max_bytes / 2;
        if (offset < 0) offset = 0;
    }

    FILE *f = fopen(corpus_path, "rb");
    if (!f) {
        fprintf(stderr, "FATAL: fopen %s failed\n", corpus_path);
        tokenizer_destroy(tok);
        return 1;
    }
    fseek(f, offset, SEEK_SET);

    /* Align to next newline */
    if (offset > 0) {
        int c;
        while ((c = fgetc(f)) != EOF && c != '\n') {
            if (--sample <= 0) {
                break;
            }
        }
    }

    uint8_t *buf = (uint8_t *)malloc((size_t)sample);
    if (!buf) {
        fclose(f);
        tokenizer_destroy(tok);
        return 1;
    }
    int64_t n = (int64_t)fread(buf, 1, (size_t)sample, f);
    fclose(f);
    if (n <= 0) {
        free(buf);
        tokenizer_destroy(tok);
        return 1;
    }

    int32_t *ids = (int32_t *)malloc((size_t)(n + 16) * sizeof(int32_t));
    if (!ids) {
        free(buf);
        tokenizer_destroy(tok);
        return 1;
    }
    int32_t n_ids = tokenizer_encode(tok, buf, n, ids, (int32_t)(n + 16));
    if (n_ids < 0) {
        fprintf(stderr, "FATAL: encode failed\n");
        free(ids);
        free(buf);
        tokenizer_destroy(tok);
        return 1;
    }

    /* Round-trip */
    uint8_t *dec = (uint8_t *)malloc((size_t)(n + 32));
    int64_t dlen = -1;
    int rt_ok = 0;
    if (dec) {
        dlen = tokenizer_decode(tok, ids, n_ids, dec, n + 32);
        rt_ok = (dlen == n && memcmp(dec, buf, (size_t)n) == 0);
    }

    double bpt = (n_ids > 0) ? (double)n / (double)n_ids : 0.0;
    printf(
        "%-30s bytes=%lld tokens=%d B/tok=%.3f round_trip=%s (vocab=%d)\n",
        corpus_path, (long long)n, n_ids, bpt, rt_ok ? "OK" : "FAIL",
        tok->vocab_size);

    free(dec);
    free(ids);
    free(buf);
    tokenizer_destroy(tok);
    return rt_ok ? 0 : 1;
}
