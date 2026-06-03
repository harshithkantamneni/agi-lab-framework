/* tokenizer.h -- Byte Pair Encoding tokenizer.
 *
 * Trains on raw UTF-8 text, learning merge rules to build a vocabulary
 * of up to TOK_MAX_VOCAB sub-word tokens.  Starts with 256 byte-level
 * tokens (0x00-0xFF) and grows via BPE merges.
 *
 * Ownership: tokenizer_create / tokenizer_load allocate; tokenizer_destroy frees.
 * All encode/decode buffers are caller-allocated.
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdbool.h>
#include <stdint.h>

/* Maximum vocabulary size. BPE starts with 256 byte tokens
 * and learns merges up to this limit. */
#define TOK_MAX_VOCAB 32768

/* A single merge rule: pair (a, b) -> new_id */
typedef struct {
    int32_t left;
    int32_t right;
    int32_t merged_id;
} MergeRule;

typedef struct {
    /* Vocabulary: id -> byte string */
    uint8_t **vocab;        /* vocab[id] = byte sequence */
    int32_t *vocab_lengths; /* Length of each vocab entry */
    int32_t vocab_size;     /* Current vocabulary size */

    /* Merge rules in order of priority (first = highest priority) */
    MergeRule *merges;
    int32_t n_merges;

    /* Special token IDs */
    int32_t pad_id;
    int32_t unk_id;
    int32_t bos_id;
    int32_t eos_id;
} Tokenizer;

/* Create a new tokenizer with just the 256 byte-level tokens.
 * Returns NULL on failure. */
Tokenizer *tokenizer_create(void);

/* Train BPE merges from a text buffer.
 * Learns up to max_merges merge rules (vocab grows to 256 + max_merges).
 * text: raw UTF-8 text
 * text_len: length in bytes
 * max_merges: number of merges to learn (e.g. 32000 - 256 = 31744)
 * Returns 0 on success, -1 on failure. */
int tokenizer_train(Tokenizer *tok, const uint8_t *text, int64_t text_len,
                    int32_t max_merges);

/* Encode text to token IDs.
 * text: raw UTF-8 input
 * text_len: length in bytes
 * out_ids: output buffer for token IDs (caller-allocated)
 * max_ids: size of output buffer
 * Returns number of tokens written, or -1 on error. */
int32_t tokenizer_encode(const Tokenizer *tok, const uint8_t *text,
                         int64_t text_len, int32_t *out_ids, int32_t max_ids);

/* Decode token IDs back to text.
 * ids: input token ID array
 * n_ids: number of tokens
 * out_text: output buffer for UTF-8 text (caller-allocated)
 * max_len: size of output buffer
 * Returns number of bytes written, or -1 on error. */
int64_t tokenizer_decode(const Tokenizer *tok, const int32_t *ids,
                         int32_t n_ids, uint8_t *out_text, int64_t max_len);

/* Save tokenizer to file (vocabulary + merge rules).
 * Format: "BPE\0" header + vocab_size + n_merges + special tokens + vocab + merges.
 * Returns 0 on success, -1 on failure. */
int tokenizer_save(const Tokenizer *tok, const char *path);

/* Load tokenizer from file.
 * Returns NULL on failure. */
Tokenizer *tokenizer_load(const char *path);

/* Destroy tokenizer and free all memory. */
void tokenizer_destroy(Tokenizer *tok);

#endif /* TOKENIZER_H */
