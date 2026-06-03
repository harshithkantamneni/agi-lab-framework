/* data_loader.h -- Data loaders for next-token prediction training.
 *
 * This header defines TWO loaders:
 *
 *  1. DataLoader        -- Sequential single-array, in-RAM loader.
 *                          Used by scale_experiment.c for pre-tokenised WikiText.
 *
 *  2. StreamDataLoader  -- Multi-source streaming loader with configurable
 *                          mixing weights. Added in Cycle 24 Phase A (D-090)
 *                          for mixture training (text / code / math).
 *
 * Both share the (tokens, targets) output contract where
 *   targets[i] = tokens[i + 1]   (next-token prediction within a batch).
 *
 * ------------------------------------------------------------
 * DataLoader (legacy single-file)
 * ------------------------------------------------------------
 * Slides a window of seq_len through a flat array of token IDs.
 * Each call to data_loader_next fills:
 *   tokens[0..seq_len-1]  = data[pos..pos+seq_len-1]
 *   targets[0..seq_len-1] = data[pos+1..pos+seq_len]
 *
 * When the data is exhausted (not enough tokens for a full window + 1 target),
 * returns -1.  Call data_loader_reset to restart from the beginning.
 *
 * Ownership: data_loader_create keeps a pointer to the token array (does NOT
 *            copy or own it). Caller must keep the token array alive for the
 *            lifetime of the loader.
 */

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <stdint.h>

#include "tokenizer.h"   /* for Tokenizer — used by StreamDataLoader API */

typedef struct DataLoader DataLoader;

/* Create a data loader from a tokenized array.
 * tokens: array of int32_t token IDs (NOT owned -- caller must keep alive)
 * n_tokens: total number of tokens in the array
 * seq_len: sequence length for each batch sample
 * Returns NULL on invalid arguments (seq_len < 1, n_tokens < seq_len + 1). */
DataLoader *data_loader_create(const int32_t *tokens, int64_t n_tokens,
                               int32_t seq_len);

/* Get next batch: fills out_tokens[seq_len] and out_targets[seq_len].
 * out_targets[i] = data[pos + i + 1] for next-token prediction.
 * Advances internal position by seq_len.
 * Returns 0 on success, -1 if exhausted (call data_loader_reset). */
int data_loader_next(DataLoader *dl, int32_t *out_tokens, int32_t *out_targets);

/* Reset to beginning of data (for next epoch). */
void data_loader_reset(DataLoader *dl);

/* Seek to a specific token position (must be aligned to seq_len).
 * Use for checkpoint resume instead of calling data_loader_next in a loop. */
void data_loader_seek(DataLoader *dl, int64_t pos);

/* Get current position (number of tokens consumed so far). */
int64_t data_loader_position(const DataLoader *dl);

/* Get total number of tokens in the dataset. */
int64_t data_loader_total(const DataLoader *dl);

/* Destroy the data loader and free its internal state.
 * Does NOT free the underlying token array. */
void data_loader_destroy(DataLoader *dl);

/* ---- Text loading utility ---- */

/* Load a text file, tokenize with a saved BPE tokenizer, return token array.
 * text_path: path to raw UTF-8 text file
 * tokenizer_path: path to saved tokenizer binary (from tokenizer_save)
 * out_tokens: on success, *out_tokens is set to a malloc'd int32_t array
 *             (caller must free)
 * Returns number of tokens on success, or -1 on error. */
int64_t load_and_tokenize(const char *text_path, const char *tokenizer_path,
                          int32_t **out_tokens);

/* ============================================================
 *  StreamDataLoader -- multi-source streaming mixture loader
 * ============================================================
 *
 * Streams raw UTF-8 text from multiple files, tokenises on the fly with a
 * per-source tokenizer, and interleaves output batches by weighted random
 * sampling. Designed for mixture training (text / code / math) on a box
 * where no single source fits in RAM.
 *
 * Memory: each source keeps ~160 KB of buffers (token ring + byte slab +
 *         encode staging). With 5 sources, total working set is <1 MB.
 *
 * Sampling: each call to stream_loader_next picks ONE source by weighted
 *         RNG, then fills the entire (tokens, targets) batch from that
 *         source's ring. This preserves intra-batch coherence and hits
 *         the target mixing ratio statistically over many batches.
 *
 * EOF handling: each source auto-wraps (fseek to 0, epoch counter++).
 *
 * Resumability: RNG state and per-source byte offsets can be saved and
 *         restored via StreamLoaderState. In-flight ring buffer contents
 *         are NOT preserved (a few hundred tokens of loss on restore;
 *         this is training, not cryptography).
 *
 * Ownership: the loader does NOT own the tokenizer pointers -- the caller
 *         must keep each Tokenizer alive for the lifetime of the loader.
 *         The loader owns its internal buffers and file handles.
 */

#define STREAM_LOADER_MAX_SOURCES 32

typedef struct StreamDataLoader StreamDataLoader;

typedef struct {
    const char      *path;       /* UTF-8 text file path. Not owned; must outlive loader. */
    const Tokenizer *tokenizer;  /* Tokenizer to apply to this source. Not owned. */
    float            weight;     /* Sampling weight (>= 0). Normalised internally. */
} StreamSourceSpec;

/* Serialisable state for checkpoint resume. */
typedef struct {
    uint64_t rng_state;                                  /* PCG32 state */
    uint64_t rng_inc;                                    /* PCG32 stream id */
    int32_t  n_sources;                                  /* for sanity checking on restore */
    int64_t  byte_offsets[STREAM_LOADER_MAX_SOURCES];    /* ftell position per source */
    int64_t  epochs[STREAM_LOADER_MAX_SOURCES];          /* wraparound counter per source */
} StreamLoaderState;

/* ---- Held-out partition (P7 increment-5d) --------------------------------
 *
 * HeldoutConfig pins the parameters for the 15% i.i.d. held-out partition.
 * When supplied (non-NULL) to stream_loader_create_heldout(), the loader
 * skips held-out chunks in source_refill().  A NULL pointer disables the
 * feature entirely; the loader produces a BYTE-IDENTICAL token stream to
 * the baseline stream_loader_create() (the gating guarantee / C4-seam).
 *
 * Constants (per repro co-sign §2.2/§2.3):
 *   seed         = 0x50524F475F335F48  (HELDOUT_SEED,   "PROG_3_H")
 *   stream       = 0x48454C444F55544F  (HELDOUT_STREAM, "HELDOUTO")
 *   thresh       = 0.15  (IEEE double)
 *   chunk_bytes  = 65536               (MUST equal BYTE_SLAB)
 *
 * The partition labeling RNG is a SEPARATE Pcg32 per source, seeded
 *   pcg32_seed(state=seed+s, stream=stream+s)
 * and NEVER shares state with dl->rng (repro C4).
 *
 * F-1 fix: `thresh` is `double` (NOT float).  The production skip compares
 * `double u < thresh`; the Python mirror, the manifest writer, and the C test
 * harness all compare against the double literal 0.15.  Storing thresh as
 * float made the production site compare against (double)0.15f =
 * 0.15000000596046448, which differs from 0.15 by ~6e-9 — a latent leak vector
 * (a chunk in the band u ∈ [0.15, 0.15000000596046448) would be trained-on by
 * the float site yet scored as held-out by the double sites).  As a double, all
 * four label sites use the bit-identical IEEE 0.15.
 */
typedef struct {
    uint64_t seed;        /* base seed; per-source = seed + source_index     */
    uint64_t stream;      /* base stream id; per-source = stream + src_index */
    double   thresh;      /* held-out threshold in [0, 1); default 0.15      */
    int32_t  chunk_bytes; /* must equal BYTE_SLAB (65536)                    */
} HeldoutConfig;

/* Create a streaming multi-source loader.
 *
 * sources:    array of n_sources StreamSourceSpec. Each path is opened now.
 * n_sources:  1..STREAM_LOADER_MAX_SOURCES
 * seq_len:    tokens per output sample (must be >= 1, <= ~4096 due to ring cap)
 * seed:       PCG32 seed (0 is fine, internally mixed)
 *
 * Returns NULL on invalid args, any file open failure, or allocation failure.
 * On failure all partially-opened files are closed. */
StreamDataLoader *stream_loader_create(const StreamSourceSpec *sources,
                                       int32_t                 n_sources,
                                       int32_t                 seq_len,
                                       uint64_t                seed);

/* Create a streaming loader with an optional held-out partition.
 *
 * Identical to stream_loader_create() when heldout == NULL (legacy path,
 * produces a BIT-IDENTICAL token stream — the disabled-is-bit-identical
 * gating guarantee).
 *
 * When heldout is non-NULL the loader skips held-out chunks: at each chunk
 * boundary the per-source PCG32 rng (seeded from heldout->seed/stream) is
 * consulted; if the chunk is held-out, pending_len is forced to 0 and the
 * ring is flushed before fseek'ing past the chunk (repro C3). */
StreamDataLoader *stream_loader_create_heldout(const StreamSourceSpec *sources,
                                               int32_t                 n_sources,
                                               int32_t                 seq_len,
                                               uint64_t                seed,
                                               const HeldoutConfig    *heldout);

/* Fill out_tokens[seq_len] and out_targets[seq_len] from a weighted-random
 * source. Both buffers are caller-allocated.
 *
 * Within a single call both arrays come from the SAME source, so the
 * next-token-prediction invariant targets[i] == tokens[i+1] (for i<seq_len-1)
 * holds inside a batch. Across batches, the source may switch.
 *
 * Returns 0 on success, -1 only on unrecoverable I/O failure (every source
 * failed to produce enough tokens even after wraparound). Normal EOF on a
 * source is handled silently via wraparound. */
int stream_loader_next(StreamDataLoader *dl,
                       int32_t *out_tokens, int32_t *out_targets);

/* Copy current loader state into *out. out->n_sources is set to the actual
 * number of sources so the caller can sanity-check on restore. */
void stream_loader_save_state(const StreamDataLoader *dl, StreamLoaderState *out);

/* Restore state from *in. in->n_sources must match the loader's n_sources.
 * Reseeks each file to the saved byte offset and flushes pending/ring buffers.
 * Returns 0 on success, -1 on mismatch or seek failure. */
int stream_loader_restore_state(StreamDataLoader *dl, const StreamLoaderState *in);

/* Diagnostics (NULL-safe). */
int64_t stream_loader_source_epochs(const StreamDataLoader *dl, int32_t src_idx);
int64_t stream_loader_source_bytes_read(const StreamDataLoader *dl, int32_t src_idx);
int32_t stream_loader_n_sources(const StreamDataLoader *dl);

/* Held-out partition diagnostics — used by the test harness (Test 5 + 7 + 8).
 * Both are NULL-safe and return 0 on invalid arguments. */
int64_t stream_loader_source_byte_offset(const StreamDataLoader *dl, int32_t src_idx);
int32_t stream_loader_source_pending_len(const StreamDataLoader *dl, int32_t src_idx);

/* Destroy loader. Closes all files and frees all buffers. Does NOT free the
 * tokenizers — those are owned by the caller. NULL-safe. */
void stream_loader_destroy(StreamDataLoader *dl);

#endif /* DATA_LOADER_H */
