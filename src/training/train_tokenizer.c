/* train_tokenizer.c -- Standalone utility to train a BPE tokenizer.
 *
 * Reads a raw text file, trains BPE with the specified number of merges,
 * saves the tokenizer to a binary file, and prints verification info.
 *
 * Usage: train_tokenizer <text_file> <output_path> [max_merges]
 *   text_file:   raw UTF-8 text for training
 *   output_path: where to save the trained tokenizer binary
 *   max_merges:  number of BPE merges to learn (default: 3840 -> vocab 4096)
 *
 * Build (standalone):
 *   clang -std=c17 -O2 -mcpu=apple-m3 -Isrc/core
 *     src/training/train_tokenizer.c src/core/tokenizer.c
 *     -o build/train_tokenizer
 */

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s <text_file> <output_path> [max_merges]\n", prog);
    fprintf(stderr, "  text_file:   raw UTF-8 training text\n");
    fprintf(stderr, "  output_path: where to save tokenizer binary\n");
    fprintf(stderr, "  max_merges:  BPE merges to learn (default 3840)\n");
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char *text_path   = argv[1];
    const char *output_path = argv[2];
    int32_t max_merges      = 3840; /* 256 byte tokens + 3840 merges = 4096 vocab */

    if (argc >= 4) {
        max_merges = atoi(argv[3]);
        if (max_merges <= 0) {
            fprintf(stderr, "Error: max_merges must be positive\n");
            return 1;
        }
    }

    printf("=== BPE Tokenizer Training ===\n");
    printf("  Input text:   %s\n", text_path);
    printf("  Output:       %s\n", output_path);
    printf("  Max merges:   %d (target vocab: %d)\n\n",
           max_merges, 256 + max_merges);

    /* Read training text */
    FILE *f = fopen(text_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open '%s'\n", text_path);
        return 1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(f);
        fprintf(stderr, "Error: file is empty or unreadable\n");
        return 1;
    }

    printf("  Text size:    %ld bytes (%.2f MB)\n", file_size,
           (double)file_size / (1024.0 * 1024.0));

    uint8_t *text = (uint8_t *)malloc((size_t)file_size);
    if (!text) {
        fclose(f);
        fprintf(stderr, "Error: out of memory\n");
        return 1;
    }

    size_t nread = fread(text, 1, (size_t)file_size, f);
    fclose(f);

    if ((long)nread != file_size) {
        free(text);
        fprintf(stderr, "Error: short read\n");
        return 1;
    }

    /* Create and train tokenizer */
    Tokenizer *tok = tokenizer_create();
    if (!tok) {
        free(text);
        fprintf(stderr, "Error: tokenizer_create failed\n");
        return 1;
    }

    printf("\nTraining BPE (%d merges)...\n", max_merges);

    int result = tokenizer_train(tok, text, (int64_t)file_size, max_merges);
    if (result != 0) {
        free(text);
        tokenizer_destroy(tok);
        fprintf(stderr, "Error: training failed\n");
        return 1;
    }

    printf("  Learned %d merges\n", tok->n_merges);
    printf("  Final vocab size: %d\n\n", tok->vocab_size);

    /* Save tokenizer */
    result = tokenizer_save(tok, output_path);
    if (result != 0) {
        free(text);
        tokenizer_destroy(tok);
        fprintf(stderr, "Error: save failed\n");
        return 1;
    }
    printf("Tokenizer saved to %s\n\n", output_path);

    /* Verification: encode a few sample strings */
    printf("=== Verification ===\n");
    const char *samples[] = {
        "The quick brown fox",
        "machine learning is",
        "2 + 2 = 4",
        "Hello, world!",
    };
    int n_samples = 4;

    for (int i = 0; i < n_samples; i++) {
        const char *s = samples[i];
        int64_t slen = (int64_t)strlen(s);
        int32_t ids[256];
        int32_t n = tokenizer_encode(tok, (const uint8_t *)s, slen, ids, 256);
        printf("  \"%s\" -> %d tokens [", s, n);
        for (int32_t j = 0; j < n && j < 10; j++) {
            if (j > 0) printf(", ");
            printf("%d", ids[j]);
        }
        if (n > 10) printf(", ...");
        printf("]\n");
    }

    /* Also show compression ratio on the training text */
    int32_t *all_ids = (int32_t *)malloc((size_t)file_size * sizeof(int32_t));
    if (all_ids) {
        int32_t total = tokenizer_encode(tok, text, (int64_t)file_size,
                                         all_ids, (int32_t)file_size);
        if (total > 0) {
            printf("\n  Training text: %ld bytes -> %d tokens "
                   "(%.2f bytes/token)\n",
                   file_size, total, (double)file_size / total);
        }
        free(all_ids);
    }

    free(text);
    tokenizer_destroy(tok);

    printf("\nDone.\n");
    return 0;
}
