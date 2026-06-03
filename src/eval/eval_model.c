/* eval_model.c -- Evaluation binary for HSPA models.
 *
 * Loads a trained checkpoint and evaluates it in one of three modes:
 *
 *   perplexity: Reads text, tokenizes, forward pass, computes PPL.
 *     Output: {"perplexity": X, "avg_nll": Y, "tokens": N}
 *
 *   score: Reads JSONL with prompt+choices, scores each choice by
 *     sum of log-probabilities, picks the highest.
 *     Output: {"accuracy": X, "correct": N, "total": N}
 *
 *   score-per-choice: Same scoring as 'score', but ALSO emits one JSONL
 *     row per item on stdout before the final aggregate line. Schema:
 *     {"item_id":"<N>","choices":[{"choice_idx":0,"logprob":-X,"n_tokens":K},...],
 *      "label":<L>,"predicted":<P>}
 *     The final aggregate line is byte-identical to 'score' mode output.
 *
 * Usage:
 *   ./build/eval_model --checkpoint PATH --tokenizer PATH \
 *       --mode perplexity|score|score-per-choice \
 *       [--input PATH] [--max-seq-len N] [--debug-token-window]
 */

#include "checkpoint.h"
#include "eval_utils.h"
#include "hspa_config.h"
#include "hspa_model.h"
#include "tensor.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- Limits ---- */
#define MAX_INPUT_SIZE  (64 * 1024 * 1024) /* 64 MB max input file (D-344 bump from 16 MB; adapter-side batching at hspa_lm_eval_adapter.py BATCH_MAX_BYTES targets 12 MB; this provides ~5x headroom) */
#define MAX_TOKENS      4096               /* Max tokens per sequence */
#define MAX_LINE        (256 * 1024)       /* Max JSONL line length */
#define MAX_CHOICES     8                  /* Max choices per item */
#define MAX_CHOICE_LEN  4096               /* Max chars per choice string */

/* ---- CLI argument parsing ---- */

typedef struct {
    const char *checkpoint_path;
    const char *tokenizer_path;
    const char *input_path;       /* NULL = stdin */
    const char *mode;             /* "perplexity", "score", or "score-per-choice" */
    int32_t max_seq_len;          /* 0 = use model default */
    bool debug_token_window;      /* emit [DEBUG-TOKEN-WINDOW] lines to stderr */
    /* Program 3, Phase 7, D-613 T5: eval-time QAT honesty.
     * When true, applies fake_quantize_int4 over the same 224-weight census
     * as the training arms so eval scores reflect quantized-weight quality. */
    bool    use_qat_eval;         /* --qat-eval: apply 4-bit fake-quant during eval */
    int32_t qat_group_size;       /* --qat-group-size N: INT4 group size; default 128 */
} EvalArgs;

static void print_usage(const char *prog) {
    fprintf(stderr,
            "Usage: %s --checkpoint PATH --tokenizer PATH "
            "--mode perplexity|score|score-per-choice "
            "[--input PATH] [--max-seq-len N] [--debug-token-window]\n",
            prog);
}

static bool parse_args(int argc, char **argv, EvalArgs *args) {
    memset(args, 0, sizeof(*args));
    args->qat_group_size = 128; /* default per P3 binding spec */

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--checkpoint") == 0 && i + 1 < argc) {
            args->checkpoint_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            args->tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            args->mode = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            args->input_path = argv[++i];
        } else if (strcmp(argv[i], "--max-seq-len") == 0 && i + 1 < argc) {
            args->max_seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--debug-token-window") == 0) {
            args->debug_token_window = true;
        } else if (strcmp(argv[i], "--qat-eval") == 0) {
            args->use_qat_eval = true;
        } else if (strcmp(argv[i], "--qat-group-size") == 0 && i + 1 < argc) {
            args->qat_group_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return false;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            return false;
        }
    }

    if (!args->checkpoint_path || !args->tokenizer_path || !args->mode) {
        fprintf(stderr, "Error: --checkpoint, --tokenizer, and --mode are required.\n");
        return false;
    }

    if (strcmp(args->mode, "perplexity") != 0 &&
        strcmp(args->mode, "score") != 0 &&
        strcmp(args->mode, "score-per-choice") != 0) {
        fprintf(stderr,
                "Error: --mode must be 'perplexity', 'score', or "
                "'score-per-choice'.\n");
        return false;
    }

    return true;
}

/* ---- File I/O helpers ---- */

/* Read entire file into malloc'd buffer. Returns NULL on error.
 * Sets *out_len to the number of bytes read. */
static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open '%s': ", path);
        perror("");
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (fsize <= 0 || (size_t)fsize > MAX_INPUT_SIZE) {
        fprintf(stderr, "Error: file '%s' is too large or empty (%ld bytes).\n",
                path, fsize);
        fclose(f);
        return NULL;
    }

    char *buf = (char *)malloc((size_t)fsize + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }

    size_t nread = fread(buf, 1, (size_t)fsize, f);
    fclose(f);

    buf[nread] = '\0';
    if (out_len) {
        *out_len = nread;
    }
    return buf;
}

/* Read all of stdin into malloc'd buffer. */
static char *read_stdin(size_t *out_len) {
    size_t capacity = 4096;
    size_t length = 0;
    char *buf = (char *)malloc(capacity);
    if (!buf) {
        return NULL;
    }

    while (!feof(stdin)) {
        if (length + 4096 > capacity) {
            capacity *= 2;
            if (capacity > MAX_INPUT_SIZE) {
                free(buf);
                return NULL;
            }
            char *tmp = (char *)realloc(buf, capacity);
            if (!tmp) {
                free(buf);
                return NULL;
            }
            buf = tmp;
        }
        size_t n = fread(buf + length, 1, 4096, stdin);
        length += n;
        if (n < 4096) {
            break;
        }
    }

    buf[length] = '\0';
    if (out_len) {
        *out_len = length;
    }
    return buf;
}

/* ---- Minimal JSONL parsing for score mode ---- */

/* Extract a JSON string value for a given key from a JSON line.
 * Writes into out_buf (max out_max bytes). Returns true on success. */
static bool json_get_string(const char *line, const char *key,
                            char *out_buf, size_t out_max) {
    /* Search for "key": " */
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(line, pattern);
    if (!p) {
        return false;
    }
    p += strlen(pattern);

    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') {
        p++;
    }

    if (*p != '"') {
        return false;
    }
    p++; /* skip opening quote */

    size_t i = 0;
    while (*p && *p != '"' && i < out_max - 1) {
        if (*p == '\\' && *(p + 1)) {
            p++; /* skip escape backslash */
            switch (*p) {
            case 'n':
                out_buf[i++] = '\n';
                break;
            case 't':
                out_buf[i++] = '\t';
                break;
            case '"':
                out_buf[i++] = '"';
                break;
            case '\\':
                out_buf[i++] = '\\';
                break;
            default:
                out_buf[i++] = *p;
                break;
            }
        } else {
            out_buf[i++] = *p;
        }
        p++;
    }
    out_buf[i] = '\0';
    return true;
}

/* Extract an integer value for a given key. Returns -1 on error. */
static int json_get_int(const char *line, const char *key) {
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    const char *p = strstr(line, pattern);
    if (!p) {
        return -1;
    }
    p += strlen(pattern);

    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') {
        p++;
    }

    return atoi(p);
}

/* Extract a JSON array of strings for "choices" key.
 * Returns the number of choices parsed. */
static int json_get_choices(const char *line, char choices[][MAX_CHOICE_LEN],
                            int max_choices) {
    const char *p = strstr(line, "\"choices\":");
    if (!p) {
        return 0;
    }
    p += strlen("\"choices\":");

    /* Skip to opening bracket */
    while (*p && *p != '[') {
        p++;
    }
    if (*p != '[') {
        return 0;
    }
    p++;

    int count = 0;
    while (*p && *p != ']' && count < max_choices) {
        /* Skip whitespace and commas */
        while (*p == ' ' || *p == '\t' || *p == '\n' || *p == ',') {
            p++;
        }
        if (*p == ']') {
            break;
        }
        if (*p != '"') {
            break;
        }
        p++; /* skip opening quote */

        size_t i = 0;
        while (*p && *p != '"' && i < MAX_CHOICE_LEN - 1) {
            if (*p == '\\' && *(p + 1)) {
                p++;
                switch (*p) {
                case 'n':
                    choices[count][i++] = '\n';
                    break;
                case 't':
                    choices[count][i++] = '\t';
                    break;
                case '"':
                    choices[count][i++] = '"';
                    break;
                case '\\':
                    choices[count][i++] = '\\';
                    break;
                default:
                    choices[count][i++] = *p;
                    break;
                }
            } else {
                choices[count][i++] = *p;
            }
            p++;
        }
        choices[count][i] = '\0';

        if (*p == '"') {
            p++; /* skip closing quote */
        }
        count++;
    }
    return count;
}

/* ---- Perplexity mode ---- */

static int run_perplexity(HSPAModel *model, const Tokenizer *tok,
                          const char *text, size_t text_len,
                          int32_t max_seq_len,
                          bool use_qat_eval, int32_t qat_group_size) {
    /* Tokenize input text */
    int32_t *token_ids = (int32_t *)malloc(MAX_TOKENS * sizeof(int32_t));
    if (!token_ids) {
        fprintf(stderr, "Error: malloc failed for token buffer.\n");
        return 1;
    }

    int32_t n_tokens = tokenizer_encode(tok, (const uint8_t *)text,
                                        (int64_t)text_len, token_ids,
                                        MAX_TOKENS);
    if (n_tokens <= 1) {
        fprintf(stderr, "Error: tokenization produced %d tokens (need >= 2).\n",
                n_tokens);
        free(token_ids);
        return 1;
    }

    /* Truncate to max_seq_len if needed */
    if (max_seq_len > 0 && n_tokens > max_seq_len) {
        n_tokens = max_seq_len;
    }

    fprintf(stderr, "Tokenized: %d tokens\n", n_tokens);

    /* Allocate logits tensor: [n_tokens, vocab_size] */
    int32_t logits_shape[] = {n_tokens, model->cfg.vocab_size};
    Tensor *logits = tensor_create(model->activation_pool, logits_shape, 2,
                                   DTYPE_FP32);
    if (!logits) {
        fprintf(stderr, "Error: failed to allocate logits tensor.\n");
        free(token_ids);
        return 1;
    }

    /* Forward pass — optionally with QAT fake-quantization (D-613 T5) */
    fprintf(stderr, "Running forward pass...\n");
    if (use_qat_eval) {
        QATContext *qat = qat_context_create(true, qat_group_size, 256);
        if (!qat) {
            fprintf(stderr, "Error: qat_context_create failed.\n");
            tensor_destroy(logits);
            free(token_ids);
            return 1;
        }
        qat_context_begin_step(qat);
        hspa_model_forward_qat(logits, model, token_ids, n_tokens, false, qat);
        qat_context_destroy(qat);
    } else {
        hspa_model_forward_qat(logits, model, token_ids, n_tokens, false, NULL);
    }

    /* Compute log-probabilities (start=0: evaluate all next-token predictions) */
    float sum_logprob = compute_logprob(logits, token_ids, n_tokens, 0);

    /* n_tokens - 1 predictions (logits[0] predicts token[1], etc.) */
    int32_t n_predicted = n_tokens - 1;
    float avg_nll = -sum_logprob / (float)n_predicted;
    float perplexity = expf(avg_nll);

    /* Output JSON */
    printf("{\"perplexity\": %.4f, \"avg_nll\": %.6f, \"tokens\": %d}\n",
           perplexity, avg_nll, n_tokens);

    tensor_destroy(logits);
    free(token_ids);
    return 0;
}

/* ---- Score mode ---- */

/* run_score: score JSONL MCQ items.
 *
 * emit_per_choice: if true, emit one JSONL row per item on stdout BEFORE the
 *   aggregate line. Schema per row:
 *   {"item_id":"<N>","choices":[{"choice_idx":0,"logprob":-X,"n_tokens":K},...],
 *    "label":<L>,"predicted":<P>}
 * debug_token_window: if true, emit [DEBUG-TOKEN-WINDOW] diagnostics to stderr
 *   for each (item, choice) so tests can verify truncation behaviour.
 *
 * In both cases the final stdout line is the aggregate accuracy JSON,
 * byte-identical to what the plain 'score' mode emits. */
static int run_score(HSPAModel *model, const Tokenizer *tok,
                     const char *input_data, int32_t max_seq_len,
                     bool emit_per_choice, bool debug_token_window,
                     bool use_qat_eval, int32_t qat_group_size) {
    int32_t correct = 0;
    int32_t total = 0;
    int32_t truncation_skipped = 0;

    /* Allocate reusable buffers (heap — avoid 300KB+ stack per iteration) */
    int32_t *token_ids = (int32_t *)malloc(MAX_TOKENS * sizeof(int32_t));
    if (!token_ids) {
        fprintf(stderr, "Error: malloc failed.\n");
        return 1;
    }

    char *line_buf = (char *)malloc(MAX_LINE);
    char *prompt = (char *)malloc(MAX_LINE);
    char (*choices)[MAX_CHOICE_LEN] = (char (*)[MAX_CHOICE_LEN])malloc(
        MAX_CHOICES * MAX_CHOICE_LEN);
    int32_t *prompt_tokens = (int32_t *)malloc(MAX_TOKENS * sizeof(int32_t));

    if (!line_buf || !prompt || !choices || !prompt_tokens) {
        free(token_ids);
        free(line_buf);
        free(prompt);
        free(choices);
        free(prompt_tokens);
        return 1;
    }

    /* Per-item per-choice result buffer (heap, not stack) */
    float *choice_logprobs = (float *)malloc(MAX_CHOICES * sizeof(float));
    int32_t *choice_ntokens = (int32_t *)malloc(MAX_CHOICES * sizeof(int32_t));
    if (!choice_logprobs || !choice_ntokens) {
        free(token_ids);
        free(line_buf);
        free(prompt);
        free(choices);
        free(prompt_tokens);
        free(choice_logprobs);
        free(choice_ntokens);
        return 1;
    }

    /* Process each JSONL line */
    const char *cursor = input_data;
    int32_t item_id = 0;
    while (*cursor) {
        /* Extract one line */
        const char *eol = strchr(cursor, '\n');
        size_t line_len;
        if (eol) {
            line_len = (size_t)(eol - cursor);
        } else {
            line_len = strlen(cursor);
        }

        if (line_len == 0 || line_len >= MAX_LINE) {
            cursor = eol ? eol + 1 : cursor + line_len;
            continue;
        }

        memcpy(line_buf, cursor, line_len);
        line_buf[line_len] = '\0';
        cursor = eol ? eol + 1 : cursor + line_len;

        /* Skip blank lines */
        if (line_buf[0] == '\0' || line_buf[0] == '\n') {
            continue;
        }

        /* Parse JSONL fields (buffers heap-allocated above) */

        if (!json_get_string(line_buf, "prompt", prompt, MAX_LINE)) {
            fprintf(stderr, "Warning: skipping line without 'prompt' field.\n");
            continue;
        }

        int n_choices = json_get_choices(line_buf, choices, MAX_CHOICES);
        if (n_choices <= 0) {
            fprintf(stderr, "Warning: skipping line without valid 'choices'.\n");
            continue;
        }

        int label = json_get_int(line_buf, "label");
        if (label < 0 || label >= n_choices) {
            fprintf(stderr, "Warning: invalid label %d (n_choices=%d).\n",
                    label, n_choices);
            continue;
        }

        /* Score each choice: tokenize(prompt + choice), forward, log-prob of choice tokens */
        float best_logprob = -INFINITY;
        int best_choice = -1;
        bool item_skipped = false;

        for (int c = 0; c < n_choices; c++) {
            choice_logprobs[c] = -INFINITY;
            choice_ntokens[c] = 0;
        }

        for (int c = 0; c < n_choices; c++) {
            /* Concatenate prompt + choice */
            size_t prompt_len = strlen(prompt);
            size_t choice_len = strlen(choices[c]);
            size_t combined_len = prompt_len + choice_len;
            char *combined = (char *)malloc(combined_len + 1);
            if (!combined) {
                continue;
            }
            memcpy(combined, prompt, prompt_len);
            memcpy(combined + prompt_len, choices[c], choice_len);
            combined[combined_len] = '\0';

            /* Tokenize combined text */
            int32_t n_combined = tokenizer_encode(
                tok, (const uint8_t *)combined, (int64_t)combined_len,
                token_ids, MAX_TOKENS);

            /* Also tokenize just the prompt to know where choice tokens start */
            int32_t n_prompt = tokenizer_encode(
                tok, (const uint8_t *)prompt, (int64_t)prompt_len,
                prompt_tokens, MAX_TOKENS);

            free(combined);

            if (n_combined <= n_prompt || n_combined < 2) {
                continue;
            }

            /* Left-truncation: drop the first (n_combined - max_seq_len) tokens
             * from token_ids (shift left), preserving choice tokens at the end.
             * Pre-reg §4.1:175 — left-truncation is the registered policy. */
            if (max_seq_len > 0 && n_combined > max_seq_len) {
                int32_t drop = n_combined - max_seq_len;
                memmove(token_ids, token_ids + drop,
                        (size_t)(n_combined - drop) * sizeof(int32_t));
                n_combined = max_seq_len;
                n_prompt -= drop;
                if (n_prompt <= 0) {
                    truncation_skipped++;
                    item_skipped = true;
                    break;
                }
            }

            if (debug_token_window) {
                fprintf(stderr,
                        "[DEBUG-TOKEN-WINDOW] c=%d \"n_prompt\":%d \"n_combined\":%d\n",
                        c, (int)n_prompt, (int)n_combined);
            }

            if (n_prompt >= n_combined) {
                continue;
            }

            /* Allocate logits and run forward pass */
            int32_t logits_shape[] = {n_combined, model->cfg.vocab_size};
            Tensor *logits = tensor_create(model->activation_pool,
                                           logits_shape, 2, DTYPE_FP32);
            if (!logits) {
                continue;
            }

            /* Forward pass — optionally with QAT fake-quantization (D-613 T5) */
            if (use_qat_eval) {
                QATContext *qat = qat_context_create(true, qat_group_size, 256);
                if (!qat) {
                    tensor_destroy(logits);
                    continue;
                }
                qat_context_begin_step(qat);
                hspa_model_forward_qat(logits, model, token_ids, n_combined,
                                       false, qat);
                qat_context_destroy(qat);
            } else {
                hspa_model_forward_qat(logits, model, token_ids, n_combined,
                                       false, NULL);
            }

            /* Compute log-prob of choice tokens only.
             * start = n_prompt - 1: log-prob of token[n_prompt] from logits[n_prompt-1] */
            float logprob = compute_logprob(logits, token_ids, n_combined,
                                            n_prompt - 1);

            tensor_destroy(logits);

            /* Reset KV cache positions for next choice */
            for (int32_t l = 0; l < model->cfg.n_layers; l++) {
                model->kv_caches[l]->pos = 0;
            }

            /* Reset activation pool for next forward pass */
            pool_reset(model->activation_pool);

            choice_logprobs[c] = logprob;
            choice_ntokens[c] = n_combined - n_prompt;

            if (logprob > best_logprob) {
                best_logprob = logprob;
                best_choice = c;
            }
        }

        if (item_skipped) {
            item_id++;
            continue;
        }

        if (emit_per_choice && best_choice >= 0) {
            /* Emit per-choice JSONL row before updating aggregate counters */
            printf("{\"item_id\":\"%d\",\"choices\":[", (int)item_id);
            for (int c = 0; c < n_choices; c++) {
                if (c > 0) {
                    printf(",");
                }
                printf("{\"choice_idx\":%d,\"logprob\":%.6f,\"n_tokens\":%d}",
                       c, (double)choice_logprobs[c], (int)choice_ntokens[c]);
            }
            printf("],\"label\":%d,\"predicted\":%d}\n", label,
                   best_choice);
        }

        if (best_choice == label) {
            correct++;
        }
        total++;
        item_id++;

        if (total % 10 == 0) {
            fprintf(stderr, "Scored %d items (%.1f%% so far)\n", total,
                    100.0f * (float)correct / (float)total);
        }
    }

    if (truncation_skipped > 0) {
        fprintf(stderr,
                "Warning: %d item(s) skipped — entire prompt dropped by "
                "left-truncation (max_seq_len too small for choice tokens)\n",
                (int)truncation_skipped);
    }

    /* Output JSON — byte-identical to 'score' mode regardless of emit_per_choice */
    float accuracy = total > 0 ? (float)correct / (float)total : 0.0f;
    printf("{\"accuracy\": %.4f, \"correct\": %d, \"total\": %d}\n",
           accuracy, correct, total);

    free(choice_ntokens);
    free(choice_logprobs);
    free(prompt_tokens);
    free(choices);
    free(prompt);
    free(line_buf);
    free(token_ids);
    return 0;
}

/* ---- Main ---- */

int main(int argc, char **argv) {
    EvalArgs args;
    if (!parse_args(argc, argv, &args)) {
        print_usage(argv[0]);
        return 1;
    }

    /* Step 1: Peek at checkpoint to get model config */
    HSPAConfig cfg;
    CheckpointMeta meta;
    if (!checkpoint_peek(args.checkpoint_path, &cfg, NULL, &meta)) {
        fprintf(stderr, "Error: cannot read checkpoint '%s'.\n",
                args.checkpoint_path);
        return 1;
    }

    fprintf(stderr, "Checkpoint: step=%d, tokens=%lld, best_loss=%.4f\n",
            meta.step, (long long)meta.tokens_trained, meta.best_loss);
    fprintf(stderr, "Model: d=%d, L=%d, E=%d, k=%d, V=%d\n",
            cfg.d_model, cfg.n_layers, cfg.n_experts, cfg.n_active,
            cfg.vocab_size);

    /* Override max_seq_len if requested */
    if (args.max_seq_len > 0 && args.max_seq_len < cfg.max_seq_len) {
        cfg.max_seq_len = args.max_seq_len;
    }

    /* Step 2: Create model and load checkpoint */
    HSPAModel *model = hspa_model_create(&cfg);
    if (!model) {
        fprintf(stderr, "Error: failed to create model.\n");
        return 1;
    }

    if (!checkpoint_load(args.checkpoint_path, model, NULL, &cfg, NULL,
                         &meta)) {
        fprintf(stderr, "Error: failed to load checkpoint weights.\n");
        hspa_model_destroy(model);
        return 1;
    }

    fprintf(stderr, "Model loaded successfully.\n");

    /* Step 3: Load tokenizer */
    Tokenizer *tok = tokenizer_load(args.tokenizer_path);
    if (!tok) {
        fprintf(stderr, "Error: failed to load tokenizer '%s'.\n",
                args.tokenizer_path);
        hspa_model_destroy(model);
        return 1;
    }

    fprintf(stderr, "Tokenizer loaded: vocab_size=%d\n", tok->vocab_size);

    /* Step 4: Read input data */
    size_t input_len = 0;
    char *input_data = NULL;
    if (args.input_path) {
        input_data = read_file(args.input_path, &input_len);
    } else {
        input_data = read_stdin(&input_len);
    }
    if (!input_data || input_len == 0) {
        fprintf(stderr, "Error: no input data.\n");
        tokenizer_destroy(tok);
        hspa_model_destroy(model);
        return 1;
    }

    /* Step 5: Run evaluation */
    int result;
    if (strcmp(args.mode, "perplexity") == 0) {
        result = run_perplexity(model, tok, input_data, input_len,
                                args.max_seq_len,
                                args.use_qat_eval, args.qat_group_size);
    } else {
        bool emit_per_choice = (strcmp(args.mode, "score-per-choice") == 0);
        result = run_score(model, tok, input_data, args.max_seq_len,
                           emit_per_choice, args.debug_token_window,
                           args.use_qat_eval, args.qat_group_size);
    }

    /* Cleanup */
    free(input_data);
    tokenizer_destroy(tok);
    hspa_model_destroy(model);

    return result;
}
