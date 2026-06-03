/* test_eval_model_score_per_choice.c -- Integration tests for eval_model
 * --mode score-per-choice and left-truncation fix.
 *
 * Strategy: subprocess integration tests. Each test spawns build/eval_model
 * with a tiny fixture checkpoint + tokenizer and a synthetic JSONL input,
 * then parses stdout to verify correctness.
 *
 * Because eval_model.c has its own main(), these tests cannot link against it
 * directly. The approach mirrors how eval is used in production: fork+exec the
 * binary, pipe input, capture output. ASan/UBSan run INSIDE the child process.
 *
 * Build requirements (verified via Makefile special rule):
 *   -Wall -Wextra -Werror -mcpu=apple-m3 -fsanitize=address,undefined
 *
 * Fixtures:
 *   The tests rely on a pre-built eval_model binary at build/eval_model and
 *   fixture checkpoint + tokenizer. If these are absent, tests report SKIP
 *   (not FAIL) so the suite stays green in environments where the full build
 *   is not present. A SKIP is printed to stderr with a [FIXTURE-MISSING] tag.
 *
 * Per-test fixture JSONL:
 *   Each test constructs its own JSONL in a tmp file, passes it via --input.
 */

#include "../src/tests/unity.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

/* ---- Configuration ---- */

/* Paths determined at build time via Makefile defines; fallback to relative. */
#ifndef EVAL_MODEL_BIN
#define EVAL_MODEL_BIN "build/eval_model"
#endif

/* Sanitizer-instrumented SUT binary (built by `make eval-debug`).
 * When defined by the Makefile and the file is present, run_eval_model()
 * invokes this binary instead of EVAL_MODEL_BIN so that ASan/UBSan run
 * inside the child process.  When absent (build/eval_model_dbg not yet
 * built), run_eval_model() falls back to EVAL_MODEL_BIN silently so that
 * `make test` remains green even if `make eval-debug` was not run first.
 *
 * The iter-1 sizeof(prompt) bug slipped past 4 PASSING Unity tests because
 * the forked child ran the non-sanitized release binary.  This fallback
 * mechanism ensures that `make eval-debug && make test` catches such bugs
 * via ASan abort in the child, without breaking the baseline `make test`.
 */
#ifndef EVAL_MODEL_DBG_BIN
#define EVAL_MODEL_DBG_BIN "build/eval_model_dbg"
#endif

#ifndef FIXTURE_CHECKPOINT
#define FIXTURE_CHECKPOINT "data/checkpoints/phase3_factorial/A42/final.ckpt"
#endif

#ifndef FIXTURE_TOKENIZER
#define FIXTURE_TOKENIZER "data/training/tokenizer_32k.bin"
#endif

/* Legacy seq_len boundary constant (kept for regression value only).
 *
 * TRUNC_SEQ_LEN=2: with the A42 fixture tokenizer, a 1200-char repeated
 * "hello hello..." prompt encodes to n_prompt=1 (BPE collapses the repeated
 * word sequence), n_combined=2 for a 2-choice item with a 1-token choice.
 * With max_seq_len=2 the truncation guard (n_combined > max_seq_len) is
 * FALSE (2 > 2 is false), so the memmove branch is NOT exercised.  Test (b)
 * still passes because n_combined == TRUNC_SEQ_LEN == 2, but it does so
 * vacuously — no tokens are actually dropped.
 *
 * D-325 iter-2 code_reviewer FLAG-1 identified this coverage gap.  The
 * memmove truncation branch is exercised by the UNICODE fixture below.
 * TRUNC_SEQ_LEN is preserved here for the legacy boundary assertion in
 * test_left_truncation_preserves_choice_tokens (vacuous path). */
#define TRUNC_SEQ_LEN 2

/* Unicode-fixture constants for the non-vacuous truncation test.
 *
 * The fixture prompt consists of 30 rare Unicode symbols (U+2603-U+261E).
 * The A42 tokenizer cannot merge these symbols (they are rare/absent in the
 * training corpus), so each symbol tokenizes to multiple bytes via the BPE
 * byte-fallback path.  Empirically verified (D-325 iter-3):
 *   n_prompt = 90  (pre-truncation)
 *   n_combined = 91 (n_prompt=90 + n_choice=1 for " yes")
 *
 * With UNICODE_TRUNC_SEQ_LEN=5:
 *   drop = 91 - 5 = 86 tokens
 *   memmove executes (shifts token_ids left by 86 positions)
 *   n_combined_post = 5 = UNICODE_TRUNC_SEQ_LEN
 *   n_prompt_post = 90 - 86 = 4  (4 prompt tokens survive)
 *
 * The test asserts n_prompt_post < UNICODE_PROMPT_PRETRUNC_NTOKENS (=90),
 * which is ONLY true if actual left-truncation fired (memmove branch
 * executed).  If the memmove is disabled (right-truncation or no truncation),
 * n_combined_post would NOT equal UNICODE_TRUNC_SEQ_LEN, causing the
 * ASSERT_EQUAL_INT(UNICODE_TRUNC_SEQ_LEN, dbg_n_combined) assertion to fail.
 *
 * Re-probe with `--debug-token-window --max-seq-len 1` before changing these
 * constants: they are empirically determined from the live A42 tokenizer. */
#define UNICODE_TRUNC_SEQ_LEN          5
#define UNICODE_PROMPT_PRETRUNC_NTOKENS 90

/* ---- Subprocess helpers ---- */

/* Resolve the best eval_model binary to run as the SUT.
 *
 * Preference order:
 *   1. build/eval_model_dbg (EVAL_MODEL_DBG_BIN) — sanitizer-instrumented;
 *      built by `make eval-debug`.  When present, ASan/UBSan run inside the
 *      forked child so memory bugs abort with a report rather than passing
 *      silently through 4 green Unity tests (D-325 FLAG-3 root cause).
 *   2. build/eval_model (EVAL_MODEL_BIN) — release binary fallback so that
 *      `make test` stays green in environments where eval-debug was not built.
 *
 * Returns EVAL_MODEL_DBG_BIN if that file is accessible, else EVAL_MODEL_BIN.
 */
static const char *resolve_eval_model_bin(void) {
    FILE *f = fopen(EVAL_MODEL_DBG_BIN, "rb");
    if (f) {
        fclose(f);
        return EVAL_MODEL_DBG_BIN;
    }
    return EVAL_MODEL_BIN;
}

/* Run eval_model binary with given args + write input_data to its stdin (or
 * pass via --input <tmpfile> if use_file is true). Capture stdout into
 * out_buf (up to out_max-1 bytes). Capture stderr into err_buf (up to
 * err_max-1 bytes, may be NULL to discard). Returns the process exit code,
 * or -1 on fork/exec failure.
 *
 * The binary invoked is resolved by resolve_eval_model_bin(): the sanitized
 * build/eval_model_dbg is preferred when available so that ASan/UBSan run
 * inside the child process (D-325 FLAG-3). */
static int run_eval_model(const char *const argv[], const char *input_data,
                          int use_file, char *out_buf, size_t out_max,
                          char *err_buf, size_t err_max) {
    /* Write input_data to a temp file */
    char tmpfile[64];
    snprintf(tmpfile, sizeof(tmpfile), "/tmp/test_spc_input_%d.jsonl",
             (int)getpid());

    if (input_data) {
        FILE *f = fopen(tmpfile, "w");
        if (!f) {
            fprintf(stderr, "run_eval_model: cannot write tmpfile: %s\n",
                    strerror(errno));
            return -1;
        }
        fputs(input_data, f);
        fclose(f);
    }

    /* Resolve the best available SUT binary (dbg preferred over release). */
    const char *sut_bin = resolve_eval_model_bin();

    /* Build argv for child: override argv[0] with resolved binary,
     * copy remaining args, then append --input tmpfile. */
    const char *child_argv[64];
    int n = 0;
    /* argv[0] is the caller's binary path; replace with resolved path */
    child_argv[n++] = sut_bin;
    /* Copy argv[1..] */
    int src = 1;
    for (; argv[src] != NULL && n < 62; src++, n++) {
        child_argv[n] = argv[src];
    }
    if (use_file && input_data) {
        child_argv[n++] = "--input";
        child_argv[n++] = tmpfile;
    }
    child_argv[n] = NULL;

    /* Pipes for stdout and stderr */
    int pout[2], perr[2];
    if (pipe(pout) != 0 || pipe(perr) != 0) {
        return -1;
    }

    pid_t pid = fork();
    if (pid < 0) {
        return -1;
    }

    if (pid == 0) {
        /* Child */
        close(pout[0]);
        close(perr[0]);
        dup2(pout[1], STDOUT_FILENO);
        dup2(perr[1], STDERR_FILENO);
        close(pout[1]);
        close(perr[1]);

        if (!use_file && input_data) {
            /* We already wrote to the file; use --input path anyway.
             * (stdin piping is more complex; file is simpler.) */
        }

        execv(child_argv[0], (char *const *)child_argv);
        /* exec failed */
        _exit(127);
    }

    /* Parent: close write ends, drain read ends */
    close(pout[1]);
    close(perr[1]);

    /* Drain stdout */
    size_t out_len = 0;
    if (out_buf && out_max > 1) {
        ssize_t r;
        while ((r = read(pout[0], out_buf + out_len,
                         out_max - 1 - out_len)) > 0) {
            out_len += (size_t)r;
        }
        out_buf[out_len] = '\0';
    }

    /* Drain stderr */
    size_t err_len = 0;
    if (err_buf && err_max > 1) {
        ssize_t r;
        while ((r = read(perr[0], err_buf + err_len,
                         err_max - 1 - err_len)) > 0) {
            err_len += (size_t)r;
        }
        err_buf[err_len] = '\0';
    } else {
        /* Drain and discard */
        char discard[256];
        while (read(perr[0], discard, sizeof(discard)) > 0) {}
    }

    close(pout[0]);
    close(perr[0]);

    int status = 0;
    waitpid(pid, &status, 0);

    /* Remove tmpfile */
    if (input_data) {
        remove(tmpfile);
    }

    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

/* Check whether the required fixture files exist. Returns 1 if present,
 * 0 if any are missing (caller should SKIP the test).
 *
 * Checks the resolved binary (dbg if available, else release) so that
 * `fixtures_available()` and `run_eval_model()` agree on which binary is used. */
static int fixtures_available(void) {
    FILE *f;
    f = fopen(resolve_eval_model_bin(), "rb");
    if (!f) {
        return 0;
    }
    fclose(f);
    f = fopen(FIXTURE_CHECKPOINT, "rb");
    if (!f) {
        return 0;
    }
    fclose(f);
    f = fopen(FIXTURE_TOKENIZER, "rb");
    if (!f) {
        return 0;
    }
    fclose(f);
    return 1;
}

/* ---- Minimal JSONL parser for test assertions ---- */

/* Parse an integer value for a key from a flat JSON object string.
 * Returns 1 on success, 0 on failure. */
static int json_parse_int(const char *json, const char *key, int *out) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(json, pattern);
    if (!p) {
        return 0;
    }
    p += strlen(pattern);
    while (*p == ' ') p++;
    char *end;
    long v = strtol(p, &end, 10);
    if (end == p) {
        return 0;
    }
    *out = (int)v;
    return 1;
}

/* Count the number of times substring needle appears in haystack. */
static int count_occurrences(const char *haystack, const char *needle) {
    int count = 0;
    const char *p = haystack;
    size_t nlen = strlen(needle);
    while ((p = strstr(p, needle)) != NULL) {
        count++;
        p += nlen;
    }
    return count;
}

/* Extract the last line of a string (the aggregate JSON line). */
static const char *last_line(const char *buf) {
    const char *last = buf;
    const char *p = buf;
    while (*p) {
        if (*p == '\n' && *(p + 1) != '\0') {
            last = p + 1;
        }
        p++;
    }
    return last;
}

/* ---- Test fixture JSONL ---- */

/* A single 4-choice MCQ item. Prompt is short; choices are short single words.
 * Label = 2 (third choice). */
static const char *FIXTURE_4CHOICE =
    "{\"prompt\": \"The capital of France is\", "
    "\"choices\": [\" Berlin\", \" London\", \" Paris\", \" Rome\"], "
    "\"label\": 2}\n";

/* BPE-resistant truncation fixture (FLAG-1, D-325 iter-3).
 *
 * 30 rare Unicode symbols (U+2603 SNOWMAN through U+2620 SKULL AND CROSSBONES)
 * that the A42 tokenizer encodes via byte-fallback (no BPE merges because these
 * symbols are absent / rare in the training corpus).  Result: n_prompt=90 tokens
 * (each 3-byte UTF-8 codepoint → ~3 byte-fallback tokens), n_combined=91.
 *
 * With --max-seq-len UNICODE_TRUNC_SEQ_LEN (=5):
 *   drop = 91 - 5 = 86  (memmove fires, shifting token_ids left by 86)
 *   n_combined_post = 5 = UNICODE_TRUNC_SEQ_LEN
 *   n_prompt_post   = 4  (< UNICODE_PROMPT_PRETRUNC_NTOKENS=90 — non-vacuous)
 *
 * Why this is BPE-resistant: repeated common English tokens (e.g. "hello ")
 * collapse aggressively under BPE — a 1200-char prompt may encode to 1 token.
 * Rare Unicode symbols are never seen as multi-symbol bigrams during BPE training
 * and thus always fall through to the byte representation, yielding predictable
 * high token counts regardless of context. */
static const char *FIXTURE_UNICODE_TRUNC =
    "{\"prompt\": \"\xe2\x98\x83\xe2\x98\x84\xe2\x98\x85\xe2\x98\x86"
    "\xe2\x98\x87\xe2\x98\x88\xe2\x98\x89\xe2\x98\x8a\xe2\x98\x8b"
    "\xe2\x98\x8c\xe2\x98\x8d\xe2\x98\x8e\xe2\x98\x8f\xe2\x98\x90"
    "\xe2\x98\x91\xe2\x98\x92\xe2\x98\x93\xe2\x98\x94\xe2\x98\x95"
    "\xe2\x98\x96\xe2\x98\x97\xe2\x98\x98\xe2\x98\x99\xe2\x98\x9a"
    "\xe2\x98\x9b\xe2\x98\x9c\xe2\x98\x9d\xe2\x98\x9e\xe2\x98\x9f"
    "\xe2\x98\xa0\", "
    "\"choices\": [\" yes\", \" no\"], "
    "\"label\": 0}\n";

/* Legacy repeated-word fixture (kept for backward compatibility with the
 * vacuous boundary test path in test_left_truncation_preserves_choice_tokens).
 * BPE compresses this to n_combined=2 with the A42 tokenizer, so truncation
 * guard does NOT fire (n_combined=2 == TRUNC_SEQ_LEN=2). */
static const char *FIXTURE_LONG_PROMPT_TEMPLATE =
    "{\"prompt\": \"%s\", "
    "\"choices\": [\" yes\", \" no\"], "
    "\"label\": 0}\n";

/* Build a long-prompt fixture by repeating a word to exceed seq_len tokens.
 * Writes into out_buf (caller provides). */
static void build_long_prompt_jsonl(char *out_buf, size_t out_max,
                                    int repeat_count) {
    /* Build the repeated prompt string */
    char prompt[8192] = {0};
    size_t plen = 0;
    const char *word = "hello ";
    size_t wlen = strlen(word);
    for (int i = 0; i < repeat_count && plen + wlen < sizeof(prompt) - 1;
         i++) {
        memcpy(prompt + plen, word, wlen);
        plen += wlen;
    }
    prompt[plen] = '\0';

    snprintf(out_buf, out_max, FIXTURE_LONG_PROMPT_TEMPLATE, prompt);
}

/* ---- Tests ---- */

/* ========================================================================
 * TEST (a): 4-choice MCQ emits 4 logprobs in one JSONL row.
 *
 * Runs eval_model --mode score-per-choice on FIXTURE_4CHOICE.
 * Asserts:
 *   - Exactly two non-blank lines on stdout: the per-choice JSONL row
 *     followed by the aggregate accuracy line.
 *   - First row contains "choices" array with exactly 4 entries
 *     (count "choice_idx" occurrences = 4).
 *   - Each entry has finite "logprob".
 *   - "label" field = 2.
 *   - "predicted" is an integer in [0,3].
 *   - "predicted" matches argmax of logprobs parsed from the row.
 * ======================================================================== */
static void test_4choice_emits_4_logprobs(void) {
    TEST_BEGIN(test_4choice_emits_4_logprobs);

    if (!fixtures_available()) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_4choice_emits_4_logprobs: SKIP — "
                "build/eval_model or fixtures not found\n");
        TEST_END();
        return;
    }

    char out_buf[65536] = {0};
    char err_buf[16384] = {0};

    const char *argv[] = {
        EVAL_MODEL_BIN,
        "--checkpoint", FIXTURE_CHECKPOINT,
        "--tokenizer",  FIXTURE_TOKENIZER,
        "--mode",       "score-per-choice",
        NULL
    };

    int rc = run_eval_model(argv, FIXTURE_4CHOICE, 1,
                            out_buf, sizeof(out_buf),
                            err_buf, sizeof(err_buf));

    if (rc == 127) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_4choice_emits_4_logprobs: SKIP — "
                "exec failed (binary not found)\n");
        TEST_END();
        return;
    }

    ASSERT_EQUAL_INT(0, rc);

    /* Count JSONL rows (non-blank lines) */
    int row_count = 0;
    const char *p = out_buf;
    while (*p) {
        const char *eol = strchr(p, '\n');
        size_t len = eol ? (size_t)(eol - p) : strlen(p);
        if (len > 0) {
            row_count++;
        }
        p = eol ? eol + 1 : p + len;
        if (!eol) break;
    }
    /* score-per-choice emits 2 non-blank lines: the per-choice row + aggregate */
    ASSERT_EQUAL_INT(2, row_count);

    /* "choice_idx" should appear exactly 4 times (in the first/per-choice row) */
    int n_choice_idx = count_occurrences(out_buf, "\"choice_idx\"");
    ASSERT_EQUAL_INT(4, n_choice_idx);

    /* "logprob" appears once per choice + possibly once in key names */
    int n_logprob = count_occurrences(out_buf, "\"logprob\"");
    ASSERT_EQUAL_INT(4, n_logprob);

    /* label == 2 */
    int label = -1;
    ASSERT_TRUE(json_parse_int(out_buf, "label", &label));
    ASSERT_EQUAL_INT(2, label);

    /* predicted is in [0,3] */
    int predicted = -1;
    ASSERT_TRUE(json_parse_int(out_buf, "predicted", &predicted));
    ASSERT_TRUE(predicted >= 0 && predicted <= 3);

    /* Each logprob is finite (not NaN/Inf) */
    /* Walk the choices array: look for "logprob": <value> entries */
    const char *scan = out_buf;
    int finite_count = 0;
    while ((scan = strstr(scan, "\"logprob\":")) != NULL) {
        scan += strlen("\"logprob\":");
        while (*scan == ' ') scan++;
        char *end;
        double lp = strtod(scan, &end);
        if (end != scan) {
            ASSERT_TRUE(isfinite(lp));
            finite_count++;
        }
    }
    ASSERT_EQUAL_INT(4, finite_count);

    /* predicted must equal the argmax of the 4 logprobs */
    /* Extract all 4 logprob values by scanning "logprob": entries */
    double logprobs[4] = {0.0, 0.0, 0.0, 0.0};
    scan = out_buf;
    int lp_idx = 0;
    while (lp_idx < 4 && (scan = strstr(scan, "\"logprob\":")) != NULL) {
        scan += strlen("\"logprob\":");
        while (*scan == ' ') scan++;
        char *end;
        double lp = strtod(scan, &end);
        if (end != scan) {
            logprobs[lp_idx++] = lp;
        }
    }
    ASSERT_EQUAL_INT(4, lp_idx);

    int argmax = 0;
    for (int i = 1; i < 4; i++) {
        if (logprobs[i] > logprobs[argmax]) {
            argmax = i;
        }
    }
    ASSERT_EQUAL_INT(argmax, predicted);

    TEST_END();
}

/* ========================================================================
 * TEST (b): Left-truncation memmove branch fires and preserves choice tokens.
 *
 * Two sub-tests in one:
 *
 * (b1) NON-VACUOUS path — BPE-resistant Unicode fixture (FLAG-1, D-325 iter-3)
 *
 *   Uses FIXTURE_UNICODE_TRUNC (30 rare Unicode symbols → n_prompt=90 pre-trunc)
 *   with --max-seq-len UNICODE_TRUNC_SEQ_LEN (=5).
 *
 *   n_combined_pre = 91 > 5 = UNICODE_TRUNC_SEQ_LEN
 *   => truncation guard fires
 *   => memmove shifts token_ids left by drop=86
 *   => n_combined_post = 5 = UNICODE_TRUNC_SEQ_LEN
 *   => n_prompt_post = 4  < UNICODE_PROMPT_PRETRUNC_NTOKENS (=90)
 *
 *   Asserts (all fail if the memmove branch is disabled):
 *   1. n_combined_post == UNICODE_TRUNC_SEQ_LEN  (truncation ceiling held)
 *   2. n_prompt_post >= 1                         (prompt survived, not skipped)
 *   3. n_prompt_post < UNICODE_PROMPT_PRETRUNC_NTOKENS  (tokens were actually dropped)
 *   4. n_tokens in JSONL row == n_combined_post - n_prompt_post  (choice preserved)
 *   5. JSONL row was emitted (not skipped due to pathological truncation)
 *
 *   Assertion 3 is the non-vacuous discriminant: it is only satisfiable when
 *   the memmove branch executed and reduced n_prompt from 90 to 4.  If
 *   no truncation fires (drop=0), n_prompt_post == n_prompt_pre == 90 >=
 *   UNICODE_PROMPT_PRETRUNC_NTOKENS, failing assertion 3.
 *
 * (b2) LEGACY VACUOUS path — repeated-word fixture (retained for boundary
 *      coverage documentation; see TRUNC_SEQ_LEN constant comment above)
 *
 *   Kept as a regression guard: confirms that a prompt which BPE compresses
 *   to n_combined=TRUNC_SEQ_LEN still passes (no pathological skip, choice
 *   token intact).  The memmove branch does NOT fire here (drop=0), but the
 *   test documents and pins this known behavior explicitly.
 * ======================================================================== */
static void test_left_truncation_preserves_choice_tokens(void) {
    TEST_BEGIN(test_left_truncation_preserves_choice_tokens);

    if (!fixtures_available()) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_left_truncation: SKIP — "
                "fixtures not found\n");
        TEST_END();
        return;
    }

    /* --- (b1) Non-vacuous: Unicode fixture, memmove branch MUST fire --- */

    char out_b1[65536] = {0};
    char err_b1[32768] = {0};

    char seq_len_unicode_str[16];
    snprintf(seq_len_unicode_str, sizeof(seq_len_unicode_str), "%d",
             UNICODE_TRUNC_SEQ_LEN);

    const char *argv_b1[] = {
        EVAL_MODEL_BIN,
        "--checkpoint",       FIXTURE_CHECKPOINT,
        "--tokenizer",        FIXTURE_TOKENIZER,
        "--mode",             "score-per-choice",
        "--max-seq-len",      seq_len_unicode_str,
        "--debug-token-window",
        NULL
    };

    int rc_b1 = run_eval_model(argv_b1, FIXTURE_UNICODE_TRUNC, 1,
                               out_b1, sizeof(out_b1),
                               err_b1, sizeof(err_b1));

    if (rc_b1 == 127) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_left_truncation (b1): SKIP\n");
        TEST_END();
        return;
    }

    ASSERT_EQUAL_INT(0, rc_b1);

    /* Parse DEBUG-TOKEN-WINDOW from stderr for choice 0 */
    const char *dbg_b1 = strstr(err_b1, "[DEBUG-TOKEN-WINDOW] c=0");
    ASSERT_NOT_NULL(dbg_b1);

    int b1_n_prompt = -1, b1_n_combined = -1;
    ASSERT_TRUE(json_parse_int(dbg_b1, "n_prompt", &b1_n_prompt));
    ASSERT_TRUE(json_parse_int(dbg_b1, "n_combined", &b1_n_combined));

    /* Assert 1: truncation ceiling held — n_combined_post == UNICODE_TRUNC_SEQ_LEN */
    ASSERT_EQUAL_INT(UNICODE_TRUNC_SEQ_LEN, b1_n_combined);

    /* Assert 2: at least one prompt token survived (not skipped) */
    ASSERT_TRUE(b1_n_prompt >= 1);

    /* Assert 3 (non-vacuous discriminant): tokens were actually dropped.
     * If the memmove branch were disabled, n_prompt_post would equal
     * UNICODE_PROMPT_PRETRUNC_NTOKENS (=90), failing this assertion. */
    ASSERT_TRUE(b1_n_prompt < UNICODE_PROMPT_PRETRUNC_NTOKENS);

    /* Assert 4: JSONL row emitted (not skipped) */
    ASSERT_TRUE(strlen(out_b1) > 2);

    /* Assert 5: n_tokens in the JSONL row == n_combined_post - n_prompt_post */
    int b1_expected_n_tokens = b1_n_combined - b1_n_prompt;
    ASSERT_TRUE(b1_expected_n_tokens > 0);

    const char *scan_b1 = strstr(out_b1, "\"n_tokens\":");
    ASSERT_NOT_NULL(scan_b1);
    scan_b1 += strlen("\"n_tokens\":");
    while (*scan_b1 == ' ') scan_b1++;
    int b1_actual_n_tokens = atoi(scan_b1);
    ASSERT_EQUAL_INT(b1_expected_n_tokens, b1_actual_n_tokens);

    /* --- (b2) Legacy vacuous: repeated-word fixture, boundary pin --- */

    char jsonl_buf[16384];
    build_long_prompt_jsonl(jsonl_buf, sizeof(jsonl_buf), 200);

    char out_b2[65536] = {0};
    char err_b2[32768] = {0};

    char seq_len_str[16];
    snprintf(seq_len_str, sizeof(seq_len_str), "%d", TRUNC_SEQ_LEN);

    const char *argv_b2[] = {
        EVAL_MODEL_BIN,
        "--checkpoint",       FIXTURE_CHECKPOINT,
        "--tokenizer",        FIXTURE_TOKENIZER,
        "--mode",             "score-per-choice",
        "--max-seq-len",      seq_len_str,
        "--debug-token-window",
        NULL
    };

    int rc_b2 = run_eval_model(argv_b2, jsonl_buf, 1,
                               out_b2, sizeof(out_b2),
                               err_b2, sizeof(err_b2));

    if (rc_b2 == 127) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_left_truncation (b2): SKIP\n");
        TEST_END();
        return;
    }

    ASSERT_EQUAL_INT(0, rc_b2);

    /* Parse DEBUG-TOKEN-WINDOW for choice 0 */
    const char *dbg_b2 = strstr(err_b2, "[DEBUG-TOKEN-WINDOW] c=0");
    ASSERT_NOT_NULL(dbg_b2);

    int b2_n_prompt = -1, b2_n_combined = -1;
    ASSERT_TRUE(json_parse_int(dbg_b2, "n_prompt", &b2_n_prompt));
    ASSERT_TRUE(json_parse_int(dbg_b2, "n_combined", &b2_n_combined));

    /* Vacuous boundary: n_combined == TRUNC_SEQ_LEN (memmove does NOT fire;
     * drop=0 because n_combined == TRUNC_SEQ_LEN, not n_combined > TRUNC_SEQ_LEN).
     * This pins the BPE-aggressive-compression behavior for regression. */
    ASSERT_EQUAL_INT(TRUNC_SEQ_LEN, b2_n_combined);

    /* At least one prompt token must survive */
    ASSERT_TRUE(b2_n_prompt >= 1);

    /* JSONL row must have been emitted */
    ASSERT_TRUE(strlen(out_b2) > 2);

    /* n_tokens in JSONL row for choice 0 == n_combined - n_prompt */
    int b2_expected_n_tokens = b2_n_combined - b2_n_prompt;
    ASSERT_TRUE(b2_expected_n_tokens > 0);

    const char *scan_b2 = strstr(out_b2, "\"n_tokens\":");
    ASSERT_NOT_NULL(scan_b2);
    scan_b2 += strlen("\"n_tokens\":");
    while (*scan_b2 == ' ') scan_b2++;
    int b2_actual_n_tokens = atoi(scan_b2);
    ASSERT_EQUAL_INT(b2_expected_n_tokens, b2_actual_n_tokens);

    TEST_END();
}

/* ========================================================================
 * TEST (c): n_tokens per choice equals n_combined - n_prompt.
 *
 * Same as (a) but verifies n_tokens for EACH choice independently.
 * Runs with --debug-token-window, parses n_prompt/n_combined per choice
 * from stderr, and cross-checks against "n_tokens" in the JSONL choices array.
 * ======================================================================== */
static void test_n_tokens_matches_choice_window(void) {
    TEST_BEGIN(test_n_tokens_matches_choice_window);

    if (!fixtures_available()) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_n_tokens_matches_choice_window: SKIP\n");
        TEST_END();
        return;
    }

    char out_buf[65536] = {0};
    char err_buf[32768] = {0};

    const char *argv[] = {
        EVAL_MODEL_BIN,
        "--checkpoint",       FIXTURE_CHECKPOINT,
        "--tokenizer",        FIXTURE_TOKENIZER,
        "--mode",             "score-per-choice",
        "--debug-token-window",
        NULL
    };

    int rc = run_eval_model(argv, FIXTURE_4CHOICE, 1,
                            out_buf, sizeof(out_buf),
                            err_buf, sizeof(err_buf));

    if (rc == 127) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_n_tokens_matches_choice_window: SKIP\n");
        TEST_END();
        return;
    }

    ASSERT_EQUAL_INT(0, rc);

    /* For each choice index 0..3, parse [DEBUG-TOKEN-WINDOW] c=X from stderr
     * and the corresponding n_tokens from the JSONL output. */

    /* Collect n_tokens from JSONL choices array (in order) */
    int jsonl_n_tokens[4] = {-1, -1, -1, -1};
    const char *scan = out_buf;
    int idx = 0;
    while (idx < 4 && (scan = strstr(scan, "\"n_tokens\":")) != NULL) {
        scan += strlen("\"n_tokens\":");
        while (*scan == ' ') scan++;
        jsonl_n_tokens[idx++] = atoi(scan);
    }
    ASSERT_EQUAL_INT(4, idx);

    /* For each choice, parse debug line and cross-check */
    for (int c = 0; c < 4; c++) {
        char marker[64];
        snprintf(marker, sizeof(marker), "[DEBUG-TOKEN-WINDOW] c=%d", c);
        const char *dbg = strstr(err_buf, marker);
        ASSERT_NOT_NULL(dbg);

        int n_prompt = -1, n_combined = -1;
        ASSERT_TRUE(json_parse_int(dbg, "n_prompt", &n_prompt));
        ASSERT_TRUE(json_parse_int(dbg, "n_combined", &n_combined));

        int expected = n_combined - n_prompt;
        ASSERT_TRUE(expected > 0);
        ASSERT_EQUAL_INT(expected, jsonl_n_tokens[c]);
    }

    TEST_END();
}

/* ========================================================================
 * TEST (d): Aggregate accuracy is unchanged.
 *
 * Runs the same fixture twice:
 *   1. --mode score           (baseline, no per-choice output)
 *   2. --mode score-per-choice (new mode)
 *
 * Asserts: the last line of stdout (the aggregate JSON) is byte-identical
 * in both runs.
 * ======================================================================== */
static void test_aggregate_accuracy_unchanged(void) {
    TEST_BEGIN(test_aggregate_accuracy_unchanged);

    if (!fixtures_available()) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_aggregate_accuracy_unchanged: SKIP\n");
        TEST_END();
        return;
    }

    char out_score[65536] = {0};
    char out_spc[65536]   = {0};

    const char *argv_score[] = {
        EVAL_MODEL_BIN,
        "--checkpoint", FIXTURE_CHECKPOINT,
        "--tokenizer",  FIXTURE_TOKENIZER,
        "--mode",       "score",
        NULL
    };

    const char *argv_spc[] = {
        EVAL_MODEL_BIN,
        "--checkpoint", FIXTURE_CHECKPOINT,
        "--tokenizer",  FIXTURE_TOKENIZER,
        "--mode",       "score-per-choice",
        NULL
    };

    int rc1 = run_eval_model(argv_score, FIXTURE_4CHOICE, 1,
                             out_score, sizeof(out_score), NULL, 0);
    int rc2 = run_eval_model(argv_spc, FIXTURE_4CHOICE, 1,
                             out_spc, sizeof(out_spc), NULL, 0);

    if (rc1 == 127 || rc2 == 127) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_aggregate_accuracy_unchanged: SKIP\n");
        TEST_END();
        return;
    }

    ASSERT_EQUAL_INT(0, rc1);
    ASSERT_EQUAL_INT(0, rc2);

    /* Baseline: last line is the aggregate JSON */
    const char *agg_score = last_line(out_score);
    ASSERT_NOT_NULL(agg_score);
    ASSERT_TRUE(strlen(agg_score) > 2);

    /* score-per-choice: last line should ALSO be the aggregate JSON */
    const char *agg_spc = last_line(out_spc);
    ASSERT_NOT_NULL(agg_spc);
    ASSERT_TRUE(strlen(agg_spc) > 2);

    /* Must be byte-identical */
    ASSERT_EQUAL_STR(agg_score, agg_spc);

    /* Sanity: the aggregate line must contain "accuracy" */
    ASSERT_TRUE(strstr(agg_score, "accuracy") != NULL);

    TEST_END();
}

/* ========================================================================
 * TEST (e): Long-prompt logprobs differ from short-prompt logprobs.
 *
 * Regression test for the sizeof(pointer) truncation bug fixed in D-325 iter 2:
 * when `prompt` was a heap char*, sizeof(prompt)==8, causing json_get_string()
 * to cap every MCQ prompt at 7 characters.  The model then scored all choices
 * against the same 7-char context regardless of actual prompt length.
 *
 * Key insight: if the bug is present, running score-per-choice on a 30-char
 * prompt produces IDENTICAL logprobs to running it on the 7-char prefix of
 * that same prompt (because both truncate to the same 7 chars at read time).
 * The fix restores correct behaviour: the full prompt changes token context,
 * so logprobs diverge between the two fixtures.
 *
 * This test runs score-per-choice on two fixtures that share the same 4
 * choices but differ only in prompt length:
 *   FULL fixture : prompt = "The quick brown fox jumps over" (30 chars, > 7)
 *   TRUNC fixture: prompt = "The qui" (exactly 7 chars, the buggy-mode prefix)
 *
 * Under the bug, both runs see the same 7-char context → logprobs are
 * bit-identical.  Under the fix, they diverge.
 *
 * Asserts:
 *   - All 8 logprobs (4 per fixture) are finite.
 *   - At least one of the 4 pairs (full[i], trunc[i]) differs (divergence).
 *   - Each fixture's "predicted" == argmax of its own 4 logprobs.
 *   - Cross-mode consistency: aggregate "correct" from --mode score on the
 *     FULL fixture matches (predicted_full == label_full ? 1 : 0).
 * ======================================================================== */

/* Full-prompt fixture: 30-char prompt, 4 choices of varying length, label=0 */
static const char *FIXTURE_FULL_PROMPT =
    "{\"prompt\": \"The quick brown fox jumps over\", "
    "\"choices\": [\" the lazy dog\", \" a fence\", \" it\", \" many tall hedges near the river\"], "
    "\"label\": 0}\n";

/* 7-char-prefix fixture: identical choices, prompt is the first 7 chars of
 * FIXTURE_FULL_PROMPT's prompt.  Under the bug, eval_model reads only these
 * 7 chars from either prompt, so logprobs should be bit-identical to the full
 * fixture if the bug is present. */
static const char *FIXTURE_7CHAR_PREFIX =
    "{\"prompt\": \"The qui\", "
    "\"choices\": [\" the lazy dog\", \" a fence\", \" it\", \" many tall hedges near the river\"], "
    "\"label\": 0}\n";

/* Helper: extract 4 logprob values from score-per-choice stdout.
 * Returns 1 on success, 0 if fewer than 4 were found. */
static int parse_4_logprobs(const char *out_buf, double lp[4]) {
    const char *scan = out_buf;
    int idx = 0;
    while (idx < 4 && (scan = strstr(scan, "\"logprob\":")) != NULL) {
        scan += strlen("\"logprob\":");
        while (*scan == ' ') scan++;
        char *end;
        double v = strtod(scan, &end);
        if (end == scan) break;
        lp[idx++] = v;
    }
    return idx == 4;
}

static void test_long_prompt_logprobs_differ_from_7char_prefix(void) {
    TEST_BEGIN(test_long_prompt_logprobs_differ_from_7char_prefix);

    if (!fixtures_available()) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_long_prompt_logprobs_differ_from_7char_prefix: SKIP — "
                "build/eval_model or fixtures not found\n");
        TEST_END();
        return;
    }

    char out_full[65536]  = {0};
    char out_trunc[65536] = {0};
    char out_score[65536] = {0};

    const char *argv_spc[] = {
        EVAL_MODEL_BIN,
        "--checkpoint", FIXTURE_CHECKPOINT,
        "--tokenizer",  FIXTURE_TOKENIZER,
        "--mode",       "score-per-choice",
        NULL
    };
    const char *argv_score[] = {
        EVAL_MODEL_BIN,
        "--checkpoint", FIXTURE_CHECKPOINT,
        "--tokenizer",  FIXTURE_TOKENIZER,
        "--mode",       "score",
        NULL
    };

    int rc_full  = run_eval_model(argv_spc, FIXTURE_FULL_PROMPT, 1,
                                  out_full,  sizeof(out_full),  NULL, 0);
    int rc_trunc = run_eval_model(argv_spc, FIXTURE_7CHAR_PREFIX, 1,
                                  out_trunc, sizeof(out_trunc), NULL, 0);
    int rc_score = run_eval_model(argv_score, FIXTURE_FULL_PROMPT, 1,
                                  out_score, sizeof(out_score), NULL, 0);

    if (rc_full == 127 || rc_trunc == 127 || rc_score == 127) {
        fprintf(stderr,
                "[FIXTURE-MISSING] test_long_prompt_logprobs_differ_from_7char_prefix: SKIP\n");
        TEST_END();
        return;
    }

    ASSERT_EQUAL_INT(0, rc_full);
    ASSERT_EQUAL_INT(0, rc_trunc);
    ASSERT_EQUAL_INT(0, rc_score);

    /* Parse logprobs for both fixtures */
    double lp_full[4], lp_trunc[4];
    ASSERT_TRUE(parse_4_logprobs(out_full,  lp_full));
    ASSERT_TRUE(parse_4_logprobs(out_trunc, lp_trunc));

    /* All 8 logprobs must be finite */
    for (int i = 0; i < 4; i++) {
        ASSERT_TRUE(isfinite(lp_full[i]));
        ASSERT_TRUE(isfinite(lp_trunc[i]));
    }

    /* Under the bug, all 4 pairs are bit-identical (both prompts truncated
     * to the same 7 chars).  Under the fix, at least one pair diverges. */
    int diverged = 0;
    for (int i = 0; i < 4; i++) {
        if (lp_full[i] != lp_trunc[i]) {
            diverged = 1;
            break;
        }
    }
    ASSERT_TRUE(diverged);

    /* predicted for the full prompt must equal argmax(lp_full) */
    int predicted_full = -1;
    ASSERT_TRUE(json_parse_int(out_full, "predicted", &predicted_full));
    ASSERT_TRUE(predicted_full >= 0 && predicted_full <= 3);

    int argmax = 0;
    for (int i = 1; i < 4; i++) {
        if (lp_full[i] > lp_full[argmax]) argmax = i;
    }
    ASSERT_EQUAL_INT(argmax, predicted_full);

    /* Cross-mode consistency: --mode score on FULL fixture must agree with
     * predicted_full on whether the item is correct (label == 0) */
    int score_correct = -1;
    ASSERT_TRUE(json_parse_int(out_score, "correct", &score_correct));
    int expected_correct = (predicted_full == 0) ? 1 : 0;
    ASSERT_EQUAL_INT(expected_correct, score_correct);

    TEST_END();
}

/* ========================================================================
 * MAIN
 * ======================================================================== */

int main(void) {
    printf("=== eval_model score-per-choice tests ===\n\n");

    printf("--- (a) 4-choice MCQ emits 4 logprobs ---\n");
    test_4choice_emits_4_logprobs();

    printf("\n--- (b) Left-truncation preserves choice tokens ---\n");
    test_left_truncation_preserves_choice_tokens();

    printf("\n--- (c) n_tokens matches choice window ---\n");
    test_n_tokens_matches_choice_window();

    printf("\n--- (d) Aggregate accuracy unchanged ---\n");
    test_aggregate_accuracy_unchanged();

    printf("\n--- (e) Long-prompt logprobs differ from 7-char prefix ---\n");
    test_long_prompt_logprobs_differ_from_7char_prefix();

    printf("\n");
    TEST_REPORT();
}
