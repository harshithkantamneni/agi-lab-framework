/* data_loader.c -- Sequential data loader for next-token prediction.
 *
 * Simple sliding-window loader: walks through a flat token array,
 * producing (tokens, targets) pairs where targets[i] = tokens[i+1].
 * Advances by seq_len each call.  Signals epoch boundary when exhausted.
 *
 * The load_and_tokenize utility reads a text file, loads a saved BPE
 * tokenizer, encodes the text, and returns the token array.
 */

#include "data_loader.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- DataLoader internal state ---- */

struct DataLoader {
    const int32_t *data;     /* Token array (not owned) */
    int64_t        n_tokens; /* Total token count */
    int32_t        seq_len;  /* Window size per sample */
    int64_t        pos;      /* Current read position */
};

/* ---- Create / Destroy ---- */

DataLoader *data_loader_create(const int32_t *tokens, int64_t n_tokens,
                               int32_t seq_len) {
    if (!tokens || seq_len < 1 || n_tokens < (int64_t)seq_len + 1) {
        return NULL;
    }

    DataLoader *dl = (DataLoader *)malloc(sizeof(DataLoader));
    if (!dl) return NULL;

    dl->data     = tokens;
    dl->n_tokens = n_tokens;
    dl->seq_len  = seq_len;
    dl->pos      = 0;
    return dl;
}

int data_loader_next(DataLoader *dl, int32_t *out_tokens, int32_t *out_targets) {
    if (!dl || !out_tokens || !out_targets) return -1;

    /* Need seq_len tokens + 1 for the last target */
    int64_t needed = (int64_t)dl->seq_len + 1;
    if (dl->pos + needed > dl->n_tokens) {
        return -1; /* Exhausted -- caller should reset for next epoch */
    }

    /* Fill tokens: data[pos .. pos + seq_len - 1] */
    memcpy(out_tokens, dl->data + dl->pos,
           (size_t)dl->seq_len * sizeof(int32_t));

    /* Fill targets: data[pos + 1 .. pos + seq_len]  (next-token prediction) */
    memcpy(out_targets, dl->data + dl->pos + 1,
           (size_t)dl->seq_len * sizeof(int32_t));

    dl->pos += dl->seq_len;
    return 0;
}

void data_loader_reset(DataLoader *dl) {
    if (dl) dl->pos = 0;
}

void data_loader_seek(DataLoader *dl, int64_t pos) {
    if (!dl) return;
    /* Clamp to valid range (must leave room for seq_len + 1 tokens). */
    int64_t max_pos = dl->n_tokens - dl->seq_len - 1;
    if (max_pos < 0) max_pos = 0;
    if (pos < 0) pos = 0;
    if (pos > max_pos) pos = max_pos;
    /* Align to seq_len boundary. */
    pos = (pos / dl->seq_len) * dl->seq_len;
    dl->pos = pos;
}

int64_t data_loader_position(const DataLoader *dl) {
    return dl ? dl->pos : 0;
}

int64_t data_loader_total(const DataLoader *dl) {
    return dl ? dl->n_tokens : 0;
}

void data_loader_destroy(DataLoader *dl) {
    free(dl);
}

/* ---- Text loading utility ---- */

int64_t load_and_tokenize(const char *text_path, const char *tokenizer_path,
                          int32_t **out_tokens) {
    if (!text_path || !tokenizer_path || !out_tokens) return -1;
    *out_tokens = NULL;

    /* Read the text file */
    FILE *f = fopen(text_path, "rb");
    if (!f) {
        fprintf(stderr, "load_and_tokenize: cannot open text file '%s'\n",
                text_path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(f);
        fprintf(stderr, "load_and_tokenize: empty or unreadable file '%s'\n",
                text_path);
        return -1;
    }
    if (file_size > (long)INT32_MAX) {
        fclose(f);
        fprintf(stderr, "load_and_tokenize: file too large (%ld bytes, max %d)\n",
                file_size, INT32_MAX);
        return -1;
    }

    uint8_t *text = (uint8_t *)malloc((size_t)file_size);
    if (!text) {
        fclose(f);
        return -1;
    }

    size_t read = fread(text, 1, (size_t)file_size, f);
    fclose(f);
    if ((long)read != file_size) {
        free(text);
        fprintf(stderr, "load_and_tokenize: short read on '%s'\n", text_path);
        return -1;
    }

    /* Load the tokenizer */
    Tokenizer *tok = tokenizer_load(tokenizer_path);
    if (!tok) {
        free(text);
        fprintf(stderr, "load_and_tokenize: cannot load tokenizer '%s'\n",
                tokenizer_path);
        return -1;
    }

    /* Allocate output buffer -- worst case is 1 token per byte (no merges) */
    int32_t *ids = (int32_t *)malloc((size_t)file_size * sizeof(int32_t));
    if (!ids) {
        tokenizer_destroy(tok);
        free(text);
        return -1;
    }

    /* Encode */
    int32_t n = tokenizer_encode(tok, text, (int64_t)file_size, ids,
                                 (int32_t)file_size);
    tokenizer_destroy(tok);
    free(text);

    if (n <= 0) {
        free(ids);
        fprintf(stderr, "load_and_tokenize: encoding failed\n");
        return -1;
    }

    /* Shrink allocation to actual size */
    int32_t *shrunk = (int32_t *)realloc(ids, (size_t)n * sizeof(int32_t));
    if (shrunk) ids = shrunk; /* realloc failure is non-fatal here */

    *out_tokens = ids;
    return (int64_t)n;
}

/* ============================================================================
 *  StreamDataLoader -- streaming multi-source mixture loader
 * ============================================================================
 *
 * Design: see data/engineering/data_loader_design.md (Cycle 24 Phase A, D-090).
 *
 * Memory layout per source (~160 KB, all caller-unaware):
 *   - FILE *f         : open file handle for the source (owned by the source)
 *   - uint8_t byte_buf[BYTE_SLAB] : raw-bytes read buffer (owned)
 *   - uint8_t pending[PENDING_CAP]: carry-over bytes between fread calls (owned)
 *   - int32_t enc_buf[ENC_CAP]    : encode staging (owned)
 *   - int32_t ring[RING_CAP]      : token ring buffer (owned)
 *
 * Single PCG32 RNG for source selection; state is seeded from the caller's
 * seed and preserved on save/restore.
 */

/* --- Tunables (all per-source; total <= 1 MB for 5 sources) -------------
 * BYTE_SLAB : bytes per fread.
 * PENDING_CAP: carry-over bytes between reads (to avoid mid-word splits).
 * ENC_CAP   : max int32 token IDs that tokenizer_encode will write per slab.
 *             Sized to >= BYTE_SLAB so worst-case (1 token per byte) fits.
 * RING_CAP  : per-source token ring. MUST be power of two; must be >= ENC_CAP
 *             plus a seq_len margin so we can always push a freshly-encoded
 *             slab without overwriting un-consumed tokens.
 *
 * Per-source footprint:
 *   byte_buf   : 64 KB
 *   pending    : 256 B
 *   enc_buf    : 256 KB (64K int32)
 *   ring       : 512 KB (128K int32)
 *   struct +   : ~1 KB
 *   --------
 *   ~832 KB per source. 5 sources = ~4 MB. Well under the 100 MB budget.
 */
#define BYTE_SLAB    (64 * 1024)      /* 64 KB */
#define PENDING_CAP  (256)            /* 256 B */
#define ENC_CAP      (64 * 1024)      /* 64K int32 = 256 KB */
#define RING_CAP     (128 * 1024)     /* 128K int32 = 512 KB */
#define RING_MASK    (RING_CAP - 1)   /* power-of-two ring mask */
_Static_assert((RING_CAP & RING_MASK) == 0, "RING_CAP must be power of two");
_Static_assert(ENC_CAP <= RING_CAP, "ENC_CAP must fit in ring");
_Static_assert(BYTE_SLAB <= ENC_CAP,
               "worst-case 1 token/byte must fit in encode buffer");

/* --- PCG32 (tiny, fast, deterministic) ---------------------------------- */
/* Reference: https://www.pcg-random.org/
 * state (64) + inc (64, must be odd). Period 2^64. Fine for RNG seeding. */
typedef struct {
    uint64_t state;
    uint64_t inc;
} Pcg32;

static uint32_t pcg32_next(Pcg32 *r) {
    uint64_t old = r->state;
    r->state     = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot        = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static void pcg32_seed(Pcg32 *r, uint64_t seed, uint64_t stream_id) {
    r->state = 0ULL;
    r->inc   = (stream_id << 1u) | 1u;    /* must be odd */
    (void)pcg32_next(r);
    r->state += seed;
    (void)pcg32_next(r);
}

/* Uniform float in [0, 1) */
static float pcg32_uniform(Pcg32 *r) {
    uint32_t u = pcg32_next(r);
    /* 24 bits of mantissa */
    return (float)(u >> 8) * (1.0f / 16777216.0f);
}

/* --- Per-source state --------------------------------------------------- */
typedef struct {
    FILE            *f;                    /* owned file handle */
    char            *path_copy;            /* owned copy of path for reopen on restore */
    const Tokenizer *tok;                  /* not owned */

    /* Raw-byte read buffer (refilled by fread) */
    uint8_t byte_buf[BYTE_SLAB];

    /* Pending bytes carried over from a chunk that ended mid-word */
    uint8_t pending[PENDING_CAP];
    int32_t pending_len;                   /* 0..PENDING_CAP */

    /* Encode staging */
    int32_t enc_buf[ENC_CAP];

    /* Token ring (power-of-two for mask indexing) */
    int32_t ring[RING_CAP];
    int32_t head;                          /* next free slot index */
    int32_t tail;                          /* next read slot index */
    int32_t available;                     /* head - tail modulo RING_CAP, cached */

    /* Diagnostics / resumability */
    int64_t byte_offset;                   /* encoded position in file, monotonic per epoch */
    int64_t epochs;                        /* wraparound counter */
    int64_t bytes_read_total;              /* lifetime bytes actually consumed */

    /* TRUE file read position (P7 increment-5d B-1 fix).
     * Absolute byte offset in the file at which the next fread() will read
     * fresh bytes — i.e. the OS file pointer for un-buffered content.  This is
     * the coordinate the held-out chunk-labeling decision MUST key off, NOT
     * byte_offset (which is the cumulative ENCODED-content position and lags
     * file_pos by the whitespace-split's held-back pending tail).  The B-1
     * defect was keying the decision off byte_offset while seeking absolute
     * FILE offsets; the two diverge on every misaligned refill, firing the
     * held-out skip ~one chunk late and leaking held-out content.
     * Maintained always; only CONSULTED when heldout != NULL. */
    int64_t file_pos;

    /* Held-out partition RNG (P7 increment-5d).
     * A SEPARATE Pcg32 instance per source — NEVER dl->rng (repro C4).
     * Only used when the loader has a non-NULL HeldoutConfig.
     * Seeded at create-time with pcg32_seed(HELDOUT_SEED+s, HELDOUT_STREAM+s).
     * Advances exactly once per chunk boundary encountered during source_refill(). */
    Pcg32   heldout_rng;                   /* per-source held-out labeling RNG */
    int32_t heldout_chunk_idx;             /* current chunk index (0-based)    */
    int32_t src_idx;                       /* index of this source in the loader */
} StreamSource;

struct StreamDataLoader {
    StreamSource sources[STREAM_LOADER_MAX_SOURCES];
    int32_t      n_sources;
    int32_t      seq_len;

    /* Normalised CDF for weighted source selection. cdf[n_sources-1] == 1.0 */
    float cdf[STREAM_LOADER_MAX_SOURCES];

    Pcg32 rng;

    /* Held-out partition config (P7 increment-5d).
     * NULL (disabled) => legacy path, BYTE-IDENTICAL to pre-heldout behavior.
     * Non-NULL => each source_refill() checks the held-out label before reading.
     * Owned by this struct (malloc'd copy of the caller's HeldoutConfig). */
    HeldoutConfig *heldout;  /* NULL or owned malloc'd copy */
};

/* --- Ring helpers ------------------------------------------------------- */

static void ring_reset(StreamSource *s) {
    s->head = 0;
    s->tail = 0;
    s->available = 0;
}

/* Push `n` tokens from `src` into the ring. If the ring would overflow, older
 * tokens are overwritten (we move the tail forward). Overwrite is acceptable
 * because the ring is always drained in `next()` before we refill. */
static void ring_push(StreamSource *s, const int32_t *src, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        s->ring[s->head] = src[i];
        s->head = (s->head + 1) & RING_MASK;
        if (s->available < RING_CAP) {
            s->available++;
        } else {
            /* overwrote oldest — advance tail */
            s->tail = (s->tail + 1) & RING_MASK;
        }
    }
}

/* Pop `n` tokens from the ring into `dst`. Caller must ensure available >= n. */
static void ring_pop(StreamSource *s, int32_t *dst, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        dst[i] = s->ring[s->tail];
        s->tail = (s->tail + 1) & RING_MASK;
    }
    s->available -= n;
}

/* --- Source byte-slab reader + encoder ---------------------------------- */

/* Whitespace bytes we split chunks on. */
static int is_chunk_break(uint8_t b) {
    return b == ' ' || b == '\n' || b == '\t' || b == '\r';
}

/* Read one slab from source, encode it into the ring. Handles EOF wraparound.
 * Returns number of tokens pushed into the ring on this call (may be 0 if
 * nothing could be encoded, e.g. empty file or held-out chunk skipped).
 *
 * heldout: if non-NULL, the skip discipline (repro §3.1/C3) is active.
 *   At each CHUNK_BYTES (== BYTE_SLAB) boundary the per-source RNG is
 *   consulted.  Held-out chunks are skipped by:
 *     (a) forcing pending_len = 0  (no carry-over),
 *     (b) calling ring_reset(s),
 *     (c) fseek'ing past the chunk.
 *   When heldout == NULL the function is BYTE-IDENTICAL to the old behavior
 *   (the gating guarantee / disabled-is-bit-identical, Test 8). */
static int32_t source_refill(StreamSource *s, const HeldoutConfig *heldout) {
    /* --- Held-out skip check (P7 increment-5d B-1 fix; repro §3.1/C3) ---- *
     *
     * The held-out partition divides each source into CHUNK_BYTES-aligned
     * slabs (chunk k spans the FILE range [k*CHUNK_BYTES, (k+1)*CHUNK_BYTES)).
     * CHUNK_BYTES MUST equal BYTE_SLAB (repro C2).
     *
     * COORDINATE (B-1 fix): the decision keys off `file_pos`, the TRUE file
     * read position (absolute offset of the next byte fread will read), NOT
     * byte_offset (encoded-content position, which lags file_pos by the
     * whitespace-split pending tail).  The binding property is that no byte of
     * a held-out chunk's FILE range is ever encoded into the ring, so the
     * decision MUST be made — and the skip MUST fire — BEFORE any byte of the
     * chunk is read.  To guarantee this we (1) decide the label the instant
     * file_pos reaches a chunk boundary and skip there if held out, and
     * (2) bound each fread so it never reads past the current chunk boundary
     * (see `room` below).  Bounding the read keeps file_pos landing exactly on
     * boundaries and keeps any pending tail strictly inside one chunk.
     *
     * Tracking: heldout_chunk_idx = number of chunks already decided = index
     * of the next chunk to decide.  The decision for chunk heldout_chunk_idx
     * fires when file_pos >= heldout_chunk_idx * CHUNK_BYTES.  Because reads
     * are boundary-bounded, file_pos reaches the boundary exactly (never
     * overshoots), so the check fires once per chunk, in order.
     *
     * When heldout == NULL the entire block is skipped → BYTE-IDENTICAL path.
     */
    int64_t cb = 0;
    if (heldout != NULL) {
        cb = (int64_t)heldout->chunk_bytes;
        if (cb > 0) {
            int64_t next_boundary = (int64_t)s->heldout_chunk_idx * cb;
            if (s->file_pos >= next_boundary) {
                /* At the boundary for chunk heldout_chunk_idx: draw its label
                 * (repro §2.3) BEFORE reading any of its bytes.
                 * Full-32-bit uniform: u = pcg32_next(rng) / 2^32.
                 * Threshold compared as IEEE double (F-1): all four label sites
                 * (this skip, the Python mirror, the manifest, the harness) use
                 * the bit-identical double 0.15. */
                uint32_t u32 = pcg32_next(&s->heldout_rng);
                double   u   = (double)u32 / 4294967296.0;
                s->heldout_chunk_idx++;  /* advance past this chunk's decision */

                if (u < heldout->thresh) {
                    /* HELD-OUT: discard pending, flush ring, fseek past the
                     * whole chunk (C3) — fires before any chunk byte is read. */
                    s->pending_len = 0;                       /* C3(a) */
                    ring_reset(s);                            /* C3(b) */
                    int64_t skip_to = next_boundary + cb;
                    if (fseek(s->f, (long)skip_to, SEEK_SET) == 0) {
                        s->file_pos    = skip_to;             /* C3(c) */
                        s->byte_offset = skip_to;             /* diag: past skip */
                    }
                    return 0;  /* 0 tokens pushed; caller loops to next refill */
                }
                /* TRAIN: fall through to the (boundary-bounded) read below. */
            }
        }
    }

    /* Copy pending tail into the front of byte_buf */
    int32_t buf_off = 0;
    if (s->pending_len > 0) {
        memcpy(s->byte_buf, s->pending, (size_t)s->pending_len);
        buf_off = s->pending_len;
        s->pending_len = 0;
    }

    /* Read into the rest of byte_buf.  With held-out active, bound the read so
     * it never crosses the current chunk boundary (B-1 fix): this keeps
     * file_pos landing exactly on boundaries so the decision above fires once
     * per chunk and never reads a held-out chunk's bytes.  Without held-out the
     * room is the full remaining slab → BYTE-IDENTICAL legacy behavior. */
    size_t room = (size_t)(BYTE_SLAB - buf_off);
    if (cb > 0) {
        int64_t cur_chunk     = s->file_pos / cb;
        int64_t chunk_end     = (cur_chunk + 1) * cb;
        int64_t room_in_chunk = chunk_end - s->file_pos;
        if (room_in_chunk < 0) room_in_chunk = 0;
        if ((int64_t)room > room_in_chunk) room = (size_t)room_in_chunk;
    }
    size_t got  = fread(s->byte_buf + buf_off, 1, room, s->f);
    s->file_pos += (int64_t)got;   /* TRUE file position advance */
    int32_t total_bytes = (int32_t)(buf_off + (int32_t)got);

    int at_eof = (got == 0);

    /* If we're not at EOF, we need to stop at a whitespace boundary so we don't
     * cut a token mid-byte. Scan back for the last whitespace. */
    int32_t encode_len = total_bytes;
    if (!at_eof && total_bytes > 0) {
        int32_t split = -1;
        for (int32_t i = total_bytes - 1; i >= 0; i--) {
            if (is_chunk_break(s->byte_buf[i])) { split = i + 1; break; }
        }
        if (split > 0 && split < total_bytes) {
            int32_t tail_len = total_bytes - split;
            if (tail_len > PENDING_CAP) {
                /* Tail is bigger than our pending cap — very unusual, only happens
                 * when the slab has a single enormous whitespace-free run.
                 * Fall through: encode the full buffer. Rare enough that we accept
                 * a possible mid-token cut here. */
            } else {
                memcpy(s->pending, s->byte_buf + split, (size_t)tail_len);
                s->pending_len = tail_len;
                encode_len = split;
            }
        }
        /* If split == -1: no whitespace in the whole slab. Encode as-is. */
    }

    int32_t n_tokens_pushed = 0;

    if (encode_len > 0) {
        int32_t n = tokenizer_encode(s->tok, s->byte_buf, (int64_t)encode_len,
                                     s->enc_buf, ENC_CAP);
        if (n > 0) {
            ring_push(s, s->enc_buf, n);
            n_tokens_pushed = n;
        }
        s->byte_offset      += encode_len;
        s->bytes_read_total += encode_len;
    }

    /* Handle EOF: flush anything still pending (no split possible), then wrap. */
    if (at_eof) {
        if (s->pending_len > 0) {
            int32_t n = tokenizer_encode(s->tok, s->pending, (int64_t)s->pending_len,
                                         s->enc_buf, ENC_CAP);
            if (n > 0) {
                ring_push(s, s->enc_buf, n);
                n_tokens_pushed += n;
            }
            s->byte_offset      += s->pending_len;
            s->bytes_read_total += s->pending_len;
            s->pending_len = 0;
        }
        /* Wrap around */
        if (fseek(s->f, 0, SEEK_SET) == 0) {
            s->byte_offset = 0;
            s->file_pos    = 0;   /* B-1 fix: file coordinate restarts at 0 */
            s->epochs     += 1;
            /* Epoch stability (repro §4): the partition is over chunk INDICES
             * which restart at 0 on each epoch.  Reset heldout_chunk_idx and
             * re-seed the per-source RNG so labels are bit-identical every epoch.
             * heldout is passed as a parameter; we use the same seed derivation.
             * Note: heldout may be NULL (disabled path). */
            if (heldout != NULL) {
                s->heldout_chunk_idx = 0;
                pcg32_seed(&s->heldout_rng,
                           heldout->seed   + (uint64_t)s->src_idx,
                           heldout->stream + (uint64_t)s->src_idx);
            }
        }
        /* If fseek fails (extremely rare — only on a closed/broken handle) the
         * next call will also get EOF and we'll try again. Not worth aborting. */
    }

    return n_tokens_pushed;
}

/* Ensure the ring for source `s` has at least `need` tokens available.
 * Returns 0 on success, -1 if we can't make progress after many refills
 * (source is truly empty / broken). */
static int source_ensure(StreamSource *s, int32_t need,
                         const HeldoutConfig *heldout) {
    /* Guard: if the encoded source is completely empty (weight-zero file or
     * a file of only whitespace), a naive loop would spin forever. Cap the
     * number of refill attempts.
     * Note: when held-out is active many refills may return 0 (skipped chunks);
     * we need a larger cap so sources with many held-out chunks can progress.
     * The cap is per-refill (not per-token), so 512 is still bounded. */
    const int MAX_REFILLS = 512;
    int32_t refills = 0;
    while (s->available < need) {
        int32_t pushed = source_refill(s, heldout);
        refills++;
        if (pushed == 0 && refills >= MAX_REFILLS) {
            /* Give up — source is effectively empty */
            return -1;
        }
    }
    return 0;
}

/* --- Weighted source selection ----------------------------------------- */

/* Pick a source index by weighted random draw against the CDF. */
static int32_t pick_source(StreamDataLoader *dl) {
    float u = pcg32_uniform(&dl->rng);
    /* Linear search — fine for n_sources <= 32 */
    for (int32_t i = 0; i < dl->n_sources; i++) {
        if (u < dl->cdf[i]) return i;
    }
    return dl->n_sources - 1;   /* FP epsilon catch-all */
}

/* --- Public API -------------------------------------------------------- */

StreamDataLoader *stream_loader_create(const StreamSourceSpec *sources,
                                       int32_t                 n_sources,
                                       int32_t                 seq_len,
                                       uint64_t                seed) {
    if (!sources || n_sources <= 0 || n_sources > STREAM_LOADER_MAX_SOURCES) {
        return NULL;
    }
    if (seq_len < 1 || seq_len >= RING_CAP) return NULL;

    /* Validate and compute total weight */
    float total_w = 0.0f;
    for (int32_t i = 0; i < n_sources; i++) {
        if (!sources[i].path || !sources[i].tokenizer) return NULL;
        if (sources[i].weight < 0.0f) return NULL;
        total_w += sources[i].weight;
    }
    if (total_w <= 0.0f) return NULL;

    /* Owner: the returned StreamDataLoader owns everything allocated here.
     * Lifetime: until stream_loader_destroy. */
    StreamDataLoader *dl = (StreamDataLoader *)calloc(1, sizeof(StreamDataLoader));
    if (!dl) return NULL;

    dl->n_sources = n_sources;
    dl->seq_len   = seq_len;
    pcg32_seed(&dl->rng, seed, 0xDA7A10AD7E57ULL);    /* arbitrary stream id */

    /* Build CDF */
    float running = 0.0f;
    for (int32_t i = 0; i < n_sources; i++) {
        running += sources[i].weight / total_w;
        dl->cdf[i] = running;
    }
    dl->cdf[n_sources - 1] = 1.0f;  /* eliminate FP drift */

    /* Open files, copy paths */
    for (int32_t i = 0; i < n_sources; i++) {
        StreamSource *s = &dl->sources[i];
        s->tok = sources[i].tokenizer;
        s->f   = fopen(sources[i].path, "rb");
        if (!s->f) goto fail;

        size_t plen = strlen(sources[i].path);
        s->path_copy = (char *)malloc(plen + 1);
        if (!s->path_copy) goto fail;
        memcpy(s->path_copy, sources[i].path, plen + 1);

        s->pending_len      = 0;
        s->byte_offset      = 0;
        s->file_pos         = 0;
        s->epochs           = 0;
        s->bytes_read_total = 0;
        s->src_idx          = i;
        s->heldout_chunk_idx = -1;  /* not yet used — no heldout config */
        ring_reset(s);
    }

    return dl;

fail:
    /* Close any files / free paths we already created */
    for (int32_t i = 0; i < n_sources; i++) {
        if (dl->sources[i].f) { fclose(dl->sources[i].f); dl->sources[i].f = NULL; }
        if (dl->sources[i].path_copy) {
            free(dl->sources[i].path_copy);
            dl->sources[i].path_copy = NULL;
        }
    }
    free(dl);
    return NULL;
}

StreamDataLoader *stream_loader_create_heldout(const StreamSourceSpec *sources,
                                               int32_t                 n_sources,
                                               int32_t                 seq_len,
                                               uint64_t                seed,
                                               const HeldoutConfig    *heldout) {
    /* NULL heldout: exact legacy behavior — delegate to stream_loader_create.
     * This is the gating guarantee: disabled => BYTE-IDENTICAL token stream. */
    if (heldout == NULL) {
        return stream_loader_create(sources, n_sources, seq_len, seed);
    }

    /* Non-NULL: create the loader via the standard path, then set up the
     * held-out partition per-source RNGs and store a copy of the config. */
    StreamDataLoader *dl = stream_loader_create(sources, n_sources, seq_len, seed);
    if (!dl) return NULL;

    /* Copy the HeldoutConfig (owner: the StreamDataLoader; freed in destroy). */
    HeldoutConfig *hcopy = (HeldoutConfig *)malloc(sizeof(HeldoutConfig));
    if (!hcopy) {
        stream_loader_destroy(dl);
        return NULL;
    }
    *hcopy = *heldout;
    dl->heldout = hcopy;

    /* Seed one dedicated Pcg32 per source (repro §2.3, C4).
     * Per-source seed  = heldout->seed   + s
     * Per-source stream = heldout->stream + s  */
    for (int32_t s = 0; s < n_sources; s++) {
        pcg32_seed(&dl->sources[s].heldout_rng,
                   heldout->seed   + (uint64_t)s,
                   heldout->stream + (uint64_t)s);
        dl->sources[s].heldout_chunk_idx = 0;
    }

    return dl;
}

int stream_loader_next(StreamDataLoader *dl,
                       int32_t *out_tokens, int32_t *out_targets) {
    if (!dl || !out_tokens || !out_targets) return -1;

    /* Pick a source by weighted RNG. If that source is broken, try the others.
     * Cap the attempts so a pathological all-empty config fails fast instead
     * of spinning forever. */
    const int MAX_ATTEMPTS = STREAM_LOADER_MAX_SOURCES * 2;
    int32_t need = dl->seq_len + 1;  /* need one extra for target shift */

    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        int32_t idx = pick_source(dl);
        StreamSource *s = &dl->sources[idx];
        if (source_ensure(s, need, dl->heldout) != 0) {
            /* This source is empty/broken — try a different one next attempt */
            continue;
        }
        /* Fill tokens (seq_len) and targets (seq_len, shifted by 1) from the
         * same source's ring. Targets are a peek — they stay in the ring and
         * become the next batch's tokens (continuous stream). Tokens are
         * popped to advance. */
        for (int32_t i = 0; i < dl->seq_len; i++) {
            out_targets[i] = s->ring[(s->tail + i + 1) & RING_MASK];
        }
        ring_pop(s, out_tokens, dl->seq_len);
        /* Post-condition: for i in [0, seq_len-2],
         *   out_targets[i] == s->ring[(old_tail + i + 1) & MASK]
         *                  == out_tokens[i+1]
         * which is the next-token-prediction invariant. */
        return 0;
    }
    return -1;
}

void stream_loader_save_state(const StreamDataLoader *dl, StreamLoaderState *out) {
    if (!dl || !out) return;
    memset(out, 0, sizeof(*out));
    out->rng_state = dl->rng.state;
    out->rng_inc   = dl->rng.inc;
    out->n_sources = dl->n_sources;
    for (int32_t i = 0; i < dl->n_sources; i++) {
        /* Resume coordinate.
         *  - Legacy (heldout == NULL): byte_offset is the encoded-content
         *    position; restore fseeks there and re-reads (a few duplicate
         *    pending-tail bytes — documented, unchanged behavior).
         *  - Held-out (heldout != NULL): the skip decision keys off the TRUE
         *    file position (B-1 fix), so byte_offsets[] carries file_pos and
         *    restore reconstructs heldout_chunk_idx + the per-source RNG state
         *    from it.  byte_offset == file_pos on a chunk-aligned position
         *    after a skip, so the two coincide at the points that matter; this
         *    overload keeps StreamLoaderState (and the on-disk checkpoint V4/V5
         *    format) byte-identical — no struct growth, no version bump.  A
         *    given loader is held-out OR legacy for its whole lifetime, so the
         *    field's meaning is unambiguous per checkpoint. */
        out->byte_offsets[i] = (dl->heldout != NULL)
                                   ? dl->sources[i].file_pos
                                   : dl->sources[i].byte_offset;
        out->epochs[i]       = dl->sources[i].epochs;
    }
}

int stream_loader_restore_state(StreamDataLoader *dl, const StreamLoaderState *in) {
    if (!dl || !in) return -1;
    if (in->n_sources != dl->n_sources) return -1;

    /* Restore RNG */
    dl->rng.state = in->rng_state;
    dl->rng.inc   = in->rng_inc | 1u;  /* inc must be odd */

    /* Restore each source */
    for (int32_t i = 0; i < dl->n_sources; i++) {
        StreamSource *s = &dl->sources[i];
        if (!s->f) return -1;
        if (fseek(s->f, (long)in->byte_offsets[i], SEEK_SET) != 0) return -1;
        s->byte_offset = in->byte_offsets[i];
        s->epochs      = in->epochs[i];
        /* Discard any buffered / in-ring tokens — simple and safe. The cost is
         * at most a few hundred tokens of duplicate reads. */
        s->pending_len = 0;
        ring_reset(s);

        if (dl->heldout != NULL) {
            /* Held-out resume (B-1 fix): the saved offset IS the true file
             * position.  Reconstruct file_pos, the count of chunks already
             * decided this epoch, and replay the per-source RNG to that count
             * so a resumed loader skips the SAME held-out chunks (cross-cell +
             * resume determinism).  Chunk m's label is decided when file_pos
             * reaches m*cb; so at file_pos f within an epoch the number of
             * already-decided chunks is:
             *   f == 0          -> 0
             *   f on a boundary -> f/cb        (chunk f/cb not yet decided)
             *   f inside chunk m-> m+1 = (f-1)/cb + 1
             * which collapses to the formula below. */
            s->file_pos = in->byte_offsets[i];
            int64_t cb = (int64_t)dl->heldout->chunk_bytes;
            int64_t decided = 0;
            if (cb > 0 && s->file_pos > 0) {
                decided = (s->file_pos - 1) / cb + 1;
            }
            pcg32_seed(&s->heldout_rng,
                       dl->heldout->seed   + (uint64_t)i,
                       dl->heldout->stream + (uint64_t)i);
            for (int64_t k = 0; k < decided; k++) {
                (void)pcg32_next(&s->heldout_rng);
            }
            s->heldout_chunk_idx = (int32_t)decided;
        } else {
            s->file_pos = in->byte_offsets[i];
        }
    }
    return 0;
}

int64_t stream_loader_source_epochs(const StreamDataLoader *dl, int32_t src_idx) {
    if (!dl || src_idx < 0 || src_idx >= dl->n_sources) return 0;
    return dl->sources[src_idx].epochs;
}

int64_t stream_loader_source_bytes_read(const StreamDataLoader *dl, int32_t src_idx) {
    if (!dl || src_idx < 0 || src_idx >= dl->n_sources) return 0;
    return dl->sources[src_idx].bytes_read_total;
}

int32_t stream_loader_n_sources(const StreamDataLoader *dl) {
    return dl ? dl->n_sources : 0;
}

void stream_loader_destroy(StreamDataLoader *dl) {
    if (!dl) return;
    for (int32_t i = 0; i < dl->n_sources; i++) {
        if (dl->sources[i].f) {
            fclose(dl->sources[i].f);
            dl->sources[i].f = NULL;
        }
        if (dl->sources[i].path_copy) {
            free(dl->sources[i].path_copy);
            dl->sources[i].path_copy = NULL;
        }
    }
    /* Free the held-out config copy if present (owned by this loader). */
    if (dl->heldout) {
        free(dl->heldout);
        dl->heldout = NULL;
    }
    free(dl);
}

/* --- Held-out partition diagnostics (Test 5 + 7 + 8 harness) ----------- */

int64_t stream_loader_source_byte_offset(const StreamDataLoader *dl, int32_t src_idx) {
    if (!dl || src_idx < 0 || src_idx >= dl->n_sources) return 0;
    return dl->sources[src_idx].byte_offset;
}

int32_t stream_loader_source_pending_len(const StreamDataLoader *dl, int32_t src_idx) {
    if (!dl || src_idx < 0 || src_idx >= dl->n_sources) return 0;
    return dl->sources[src_idx].pending_len;
}
