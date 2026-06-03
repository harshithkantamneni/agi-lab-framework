/* tokenizer.c -- Byte Pair Encoding tokenizer.
 *
 * Implementation:
 *   - Training: count all adjacent pairs, merge the most frequent, repeat.
 *     O(n * m) where n = text length, m = number of merges.  Fine for
 *     initial implementation; optimize with priority queue later if needed.
 *   - Encoding: convert text to byte tokens, then apply merges in priority
 *     order (first merge = highest priority).
 *   - Decoding: concatenate byte sequences for each token ID.
 *   - Save/Load: simple binary format with header, vocab, and merge rules.
 *
 * Memory ownership:
 *   - Tokenizer struct and all internal arrays are malloc/calloc'd.
 *   - tokenizer_destroy frees everything.
 *   - Encode/decode output buffers are caller-owned.
 */

#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Internal: pair counting via hash table ---- */

/* Simple open-addressing hash table for (int32_t, int32_t) -> int64_t count. */
typedef struct {
    int32_t left;
    int32_t right;
    int64_t count;
    bool occupied;
} PairEntry;

typedef struct {
    PairEntry *entries;
    int32_t capacity;
    int32_t size;
} PairTable;

static PairTable *pair_table_create(int32_t capacity) {
    PairTable *pt = (PairTable *)calloc(1, sizeof(PairTable));
    if (!pt) {
        return NULL;
    }
    /* Round capacity up to next power of two for efficient masking */
    int32_t cap = 64;
    while (cap < capacity) {
        cap *= 2;
    }
    pt->entries = (PairEntry *)calloc((size_t)cap, sizeof(PairEntry));
    if (!pt->entries) {
        free(pt);
        return NULL;
    }
    pt->capacity = cap;
    pt->size = 0;
    return pt;
}

static void pair_table_destroy(PairTable *pt) {
    if (!pt) {
        return;
    }
    free(pt->entries);
    free(pt);
}

static inline uint32_t pair_hash(int32_t a, int32_t b) {
    /* Simple hash combining two ints */
    uint32_t h = (uint32_t)a * 2654435761u;
    h ^= (uint32_t)b * 2246822519u;
    h ^= h >> 16;
    return h;
}

/* Increment count for pair (a, b). Returns the new count, or -1 on failure. */
static int64_t pair_table_inc(PairTable *pt, int32_t a, int32_t b) {
    if (!pt) {
        return -1;
    }

    /* Rehash if load factor > 0.7 */
    if (pt->size * 10 > pt->capacity * 7) {
        int32_t new_cap = pt->capacity * 2;
        PairEntry *new_entries =
            (PairEntry *)calloc((size_t)new_cap, sizeof(PairEntry));
        if (!new_entries) {
            return -1;
        }
        uint32_t mask = (uint32_t)(new_cap - 1);
        for (int32_t i = 0; i < pt->capacity; i++) {
            if (pt->entries[i].occupied) {
                uint32_t idx =
                    pair_hash(pt->entries[i].left, pt->entries[i].right) & mask;
                while (new_entries[idx].occupied) {
                    idx = (idx + 1) & mask;
                }
                new_entries[idx] = pt->entries[i];
            }
        }
        free(pt->entries);
        pt->entries = new_entries;
        pt->capacity = new_cap;
    }

    uint32_t mask = (uint32_t)(pt->capacity - 1);
    uint32_t idx = pair_hash(a, b) & mask;

    while (pt->entries[idx].occupied) {
        if (pt->entries[idx].left == a && pt->entries[idx].right == b) {
            pt->entries[idx].count++;
            return pt->entries[idx].count;
        }
        idx = (idx + 1) & mask;
    }

    /* New entry */
    pt->entries[idx].left = a;
    pt->entries[idx].right = b;
    pt->entries[idx].count = 1;
    pt->entries[idx].occupied = true;
    pt->size++;
    return 1;
}

/* Find the pair with the highest count. Writes best pair to *out_a, *out_b.
 * Returns the count, or 0 if table is empty. */
static int64_t pair_table_best(const PairTable *pt, int32_t *out_a,
                               int32_t *out_b) {
    int64_t best_count = 0;
    *out_a = -1;
    *out_b = -1;
    for (int32_t i = 0; i < pt->capacity; i++) {
        if (pt->entries[i].occupied && pt->entries[i].count > best_count) {
            best_count = pt->entries[i].count;
            *out_a = pt->entries[i].left;
            *out_b = pt->entries[i].right;
        }
    }
    return best_count;
}

/* ---- Vocabulary helpers ---- */

/* Allocate and set a vocab entry to a single byte. */
static int vocab_set_byte(Tokenizer *tok, int32_t id, uint8_t byte_val) {
    tok->vocab[id] = (uint8_t *)malloc(1);
    if (!tok->vocab[id]) {
        return -1;
    }
    tok->vocab[id][0] = byte_val;
    tok->vocab_lengths[id] = 1;
    return 0;
}

/* Allocate and set a vocab entry by concatenating two existing entries. */
static int vocab_set_concat(Tokenizer *tok, int32_t id, int32_t left_id,
                            int32_t right_id) {
    int32_t left_len = tok->vocab_lengths[left_id];
    int32_t right_len = tok->vocab_lengths[right_id];
    int32_t total = left_len + right_len;

    tok->vocab[id] = (uint8_t *)malloc((size_t)total);
    if (!tok->vocab[id]) {
        return -1;
    }
    memcpy(tok->vocab[id], tok->vocab[left_id], (size_t)left_len);
    memcpy(tok->vocab[id] + left_len, tok->vocab[right_id], (size_t)right_len);
    tok->vocab_lengths[id] = total;
    return 0;
}

/* ---- Public API ---- */

Tokenizer *tokenizer_create(void) {
    Tokenizer *tok = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    if (!tok) {
        return NULL;
    }

    tok->vocab =
        (uint8_t **)calloc(TOK_MAX_VOCAB, sizeof(uint8_t *));
    tok->vocab_lengths =
        (int32_t *)calloc(TOK_MAX_VOCAB, sizeof(int32_t));
    if (!tok->vocab || !tok->vocab_lengths) {
        tokenizer_destroy(tok);
        return NULL;
    }

    /* Initialize the 256 byte-level tokens */
    for (int i = 0; i < 256; i++) {
        if (vocab_set_byte(tok, i, (uint8_t)i) != 0) {
            tokenizer_destroy(tok);
            return NULL;
        }
    }
    tok->vocab_size = 256;

    /* Merge rules: none yet */
    tok->merges = NULL;
    tok->n_merges = 0;

    /* Special tokens: -1 means unset */
    tok->pad_id = -1;
    tok->unk_id = -1;
    tok->bos_id = -1;
    tok->eos_id = -1;

    return tok;
}

int tokenizer_train(Tokenizer *tok, const uint8_t *text, int64_t text_len,
                    int32_t max_merges) {
    if (!tok || !text || text_len <= 0 || max_merges <= 0) {
        return -1;
    }

    /* Cap merges to not exceed max vocab */
    if (tok->vocab_size + max_merges > TOK_MAX_VOCAB) {
        max_merges = TOK_MAX_VOCAB - tok->vocab_size;
    }
    if (max_merges <= 0) {
        return 0; /* Already at capacity */
    }

    /* Step 1: Convert text to byte-level token sequence.
     * We use a dynamic array of int32_t token IDs.
     * Also maintain a parallel "active" array (1 = present, 0 = merged away)
     * to efficiently skip removed tokens during pair counting. */
    int32_t *ids = (int32_t *)malloc((size_t)text_len * sizeof(int32_t));
    if (!ids) {
        return -1;
    }
    for (int64_t i = 0; i < text_len; i++) {
        ids[i] = (int32_t)text[i];
    }
    int64_t ids_len = text_len;

    /* Allocate merge rules buffer */
    MergeRule *new_merges =
        (MergeRule *)realloc(tok->merges,
                             (size_t)(tok->n_merges + max_merges) *
                                 sizeof(MergeRule));
    if (!new_merges) {
        free(ids);
        return -1;
    }
    tok->merges = new_merges;

    /* Step 2-8: BPE training loop */
    for (int32_t m = 0; m < max_merges; m++) {
        if (ids_len < 2) {
            break; /* Nothing to merge */
        }

        /* Count all adjacent pairs */
        int32_t table_cap = (int32_t)(ids_len < 64 ? 64 : ids_len);
        PairTable *pt = pair_table_create(table_cap);
        if (!pt) {
            free(ids);
            return -1;
        }

        for (int64_t i = 0; i < ids_len - 1; i++) {
            pair_table_inc(pt, ids[i], ids[i + 1]);
        }

        /* Find most frequent pair */
        int32_t best_a, best_b;
        int64_t best_count = pair_table_best(pt, &best_a, &best_b);
        pair_table_destroy(pt);

        if (best_count < 2) {
            break; /* No pair occurs more than once; stop */
        }

        /* Create new token */
        int32_t new_id = tok->vocab_size;
        if (vocab_set_concat(tok, new_id, best_a, best_b) != 0) {
            free(ids);
            return -1;
        }
        tok->vocab_size++;

        /* Record merge rule */
        tok->merges[tok->n_merges].left = best_a;
        tok->merges[tok->n_merges].right = best_b;
        tok->merges[tok->n_merges].merged_id = new_id;
        tok->n_merges++;

        /* Replace all occurrences of (best_a, best_b) with new_id in-place */
        int64_t write = 0;
        for (int64_t i = 0; i < ids_len; i++) {
            if (i < ids_len - 1 && ids[i] == best_a &&
                ids[i + 1] == best_b) {
                ids[write++] = new_id;
                i++; /* Skip the second element of the pair */
            } else {
                ids[write++] = ids[i];
            }
        }
        ids_len = write;
    }

    free(ids);
    return 0;
}

int32_t tokenizer_encode(const Tokenizer *tok, const uint8_t *text,
                         int64_t text_len, int32_t *out_ids,
                         int32_t max_ids) {
    if (!tok || !out_ids || max_ids <= 0) {
        return -1;
    }
    if (!text || text_len <= 0) {
        return 0; /* Empty input produces zero tokens */
    }

    /* Step 1: Convert text to byte tokens */
    int32_t *work = (int32_t *)malloc((size_t)text_len * sizeof(int32_t));
    if (!work) {
        return -1;
    }
    for (int64_t i = 0; i < text_len; i++) {
        work[i] = (int32_t)text[i];
    }
    int64_t work_len = text_len;

    /* Step 2: Apply merge rules in priority order */
    for (int32_t m = 0; m < tok->n_merges; m++) {
        int32_t left = tok->merges[m].left;
        int32_t right = tok->merges[m].right;
        int32_t merged = tok->merges[m].merged_id;

        int64_t write = 0;
        for (int64_t i = 0; i < work_len; i++) {
            if (i < work_len - 1 && work[i] == left &&
                work[i + 1] == right) {
                work[write++] = merged;
                i++; /* Skip the right element */
            } else {
                work[write++] = work[i];
            }
        }
        work_len = write;
    }

    /* Step 3: Copy to output, respecting max_ids */
    int32_t n_out = (int32_t)(work_len < max_ids ? work_len : max_ids);
    memcpy(out_ids, work, (size_t)n_out * sizeof(int32_t));

    free(work);
    return n_out;
}

int64_t tokenizer_decode(const Tokenizer *tok, const int32_t *ids,
                         int32_t n_ids, uint8_t *out_text, int64_t max_len) {
    if (!tok || !out_text || max_len <= 0) {
        return -1;
    }
    if (!ids || n_ids <= 0) {
        return 0; /* Empty input produces empty output */
    }

    int64_t written = 0;
    for (int32_t i = 0; i < n_ids; i++) {
        int32_t id = ids[i];
        if (id < 0 || id >= tok->vocab_size) {
            return -1; /* Invalid token ID */
        }

        int32_t len = tok->vocab_lengths[id];
        if (written + len > max_len) {
            return -1; /* Output buffer too small */
        }

        memcpy(out_text + written, tok->vocab[id], (size_t)len);
        written += len;
    }

    return written;
}

/* ---- Save/Load binary format ----
 *
 * Header (24 bytes):
 *   magic:      "BPE\0"   (4 bytes)
 *   vocab_size: int32     (4 bytes)
 *   n_merges:   int32     (4 bytes)
 *   pad_id:     int32     (4 bytes)
 *   unk_id:     int32     (4 bytes)
 *   bos_id:     int32     (4 bytes)
 *   eos_id:     int32     (4 bytes)
 *
 * Vocab (variable):
 *   For each entry: length (int32) + bytes (length bytes)
 *
 * Merges (n_merges * 12 bytes):
 *   For each: left (int32) + right (int32) + merged_id (int32)
 */

int tokenizer_save(const Tokenizer *tok, const char *path) {
    if (!tok || !path) {
        return -1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        return -1;
    }

    /* Header */
    const char magic[4] = {'B', 'P', 'E', '\0'};
    if (fwrite(magic, 1, 4, fp) != 4) {
        goto fail;
    }
    if (fwrite(&tok->vocab_size, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fwrite(&tok->n_merges, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fwrite(&tok->pad_id, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fwrite(&tok->unk_id, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fwrite(&tok->bos_id, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fwrite(&tok->eos_id, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }

    /* Vocab entries */
    for (int32_t i = 0; i < tok->vocab_size; i++) {
        int32_t len = tok->vocab_lengths[i];
        if (fwrite(&len, sizeof(int32_t), 1, fp) != 1) {
            goto fail;
        }
        if (len > 0) {
            if (fwrite(tok->vocab[i], 1, (size_t)len, fp) != (size_t)len) {
                goto fail;
            }
        }
    }

    /* Merge rules */
    for (int32_t i = 0; i < tok->n_merges; i++) {
        if (fwrite(&tok->merges[i].left, sizeof(int32_t), 1, fp) != 1) {
            goto fail;
        }
        if (fwrite(&tok->merges[i].right, sizeof(int32_t), 1, fp) != 1) {
            goto fail;
        }
        if (fwrite(&tok->merges[i].merged_id, sizeof(int32_t), 1, fp) != 1) {
            goto fail;
        }
    }

    fclose(fp);
    return 0;

fail:
    fclose(fp);
    return -1;
}

Tokenizer *tokenizer_load(const char *path) {
    if (!path) {
        return NULL;
    }

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return NULL;
    }

    /* Read and verify magic */
    char magic[4];
    if (fread(magic, 1, 4, fp) != 4) {
        goto fail;
    }
    if (magic[0] != 'B' || magic[1] != 'P' || magic[2] != 'E' ||
        magic[3] != '\0') {
        goto fail;
    }

    /* Read header */
    int32_t vocab_size, n_merges, pad_id, unk_id, bos_id, eos_id;
    if (fread(&vocab_size, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fread(&n_merges, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fread(&pad_id, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fread(&unk_id, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fread(&bos_id, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }
    if (fread(&eos_id, sizeof(int32_t), 1, fp) != 1) {
        goto fail;
    }

    /* Validate */
    if (vocab_size < 256 || vocab_size > TOK_MAX_VOCAB || n_merges < 0) {
        goto fail;
    }

    /* Allocate tokenizer */
    Tokenizer *tok = (Tokenizer *)calloc(1, sizeof(Tokenizer));
    if (!tok) {
        goto fail;
    }

    tok->vocab =
        (uint8_t **)calloc(TOK_MAX_VOCAB, sizeof(uint8_t *));
    tok->vocab_lengths =
        (int32_t *)calloc(TOK_MAX_VOCAB, sizeof(int32_t));
    if (!tok->vocab || !tok->vocab_lengths) {
        tokenizer_destroy(tok);
        fclose(fp);
        return NULL;
    }

    tok->vocab_size = vocab_size;
    tok->pad_id = pad_id;
    tok->unk_id = unk_id;
    tok->bos_id = bos_id;
    tok->eos_id = eos_id;

    /* Read vocab entries */
    for (int32_t i = 0; i < vocab_size; i++) {
        int32_t len;
        if (fread(&len, sizeof(int32_t), 1, fp) != 1) {
            tokenizer_destroy(tok);
            fclose(fp);
            return NULL;
        }
        if (len < 0 || len > 1024) {
            /* Sanity check: no single token should be > 1KB */
            tokenizer_destroy(tok);
            fclose(fp);
            return NULL;
        }
        tok->vocab_lengths[i] = len;
        if (len > 0) {
            tok->vocab[i] = (uint8_t *)malloc((size_t)len);
            if (!tok->vocab[i]) {
                tokenizer_destroy(tok);
                fclose(fp);
                return NULL;
            }
            if (fread(tok->vocab[i], 1, (size_t)len, fp) != (size_t)len) {
                tokenizer_destroy(tok);
                fclose(fp);
                return NULL;
            }
        }
    }

    /* Read merge rules */
    if (n_merges > 0) {
        tok->merges =
            (MergeRule *)malloc((size_t)n_merges * sizeof(MergeRule));
        if (!tok->merges) {
            tokenizer_destroy(tok);
            fclose(fp);
            return NULL;
        }
        tok->n_merges = n_merges;
        for (int32_t i = 0; i < n_merges; i++) {
            if (fread(&tok->merges[i].left, sizeof(int32_t), 1, fp) != 1 ||
                fread(&tok->merges[i].right, sizeof(int32_t), 1, fp) != 1 ||
                fread(&tok->merges[i].merged_id, sizeof(int32_t), 1, fp) !=
                    1) {
                tokenizer_destroy(tok);
                fclose(fp);
                return NULL;
            }
        }
    } else {
        tok->merges = NULL;
        tok->n_merges = 0;
    }

    fclose(fp);
    return tok;

fail:
    fclose(fp);
    return NULL;
}

void tokenizer_destroy(Tokenizer *tok) {
    if (!tok) {
        return;
    }

    if (tok->vocab) {
        for (int32_t i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    free(tok->vocab_lengths);
    free(tok->merges);
    free(tok);
}
