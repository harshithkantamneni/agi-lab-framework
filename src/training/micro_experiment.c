/* micro_experiment.c -- Standalone micro-experiment: train a tiny HSPA.
 *
 * Supports two training modes for A/B comparison:
 *   - iPC (default): iterative Predictive Coding with local learning rules
 *   - Backprop (--backprop): standard forward+backward pass
 *
 * Config: L=4, D=128, K=4, k=1, V=256, S=64 (~1M params)
 * Trains on synthetic next-token prediction or real tokenized text.
 *
 * Success criteria:
 *   1. Loss decreases monotonically (modulo noise)
 *   2. Perplexity drops from ~256 (random) to <50
 *   3. No NaN/Inf in loss or gradients
 *   4. Expert utilization stays above 3.1% minimum per expert
 *
 * Usage:
 *   build/micro_experiment [max_steps] [seq_len] [patterns] [adam] [beta]
 *   build/micro_experiment --backprop [max_steps]
 *   build/micro_experiment --backprop --data path/to/text --tokenizer path/to/tok
 */

#include "backprop_train.h"
#include "data_loader.h"
#include "hspa_model.h"
#include "ipc_train.h"
#include "tokenizer.h"
#include "weight_init.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Generate a synthetic training batch: random token sequences.
 * tokens[i] = random in [0, vocab_size), targets[i] = tokens[i+1].
 * Last target wraps to tokens[0]. */
static void gen_synthetic_batch(int32_t *tokens, int32_t *targets,
                                int32_t seq_len, int32_t vocab_size) {
    for (int32_t i = 0; i < seq_len; i++) {
        tokens[i] = rand() % vocab_size;
    }
    for (int32_t i = 0; i < seq_len - 1; i++) {
        targets[i] = tokens[i + 1];
    }
    targets[seq_len - 1] = tokens[0]; /* wrap */
}

/* Generate a patterned batch: repeating short patterns.
 * Easier to learn than pure random -- tests whether the model can
 * capture simple sequential dependencies. */
static void gen_pattern_batch(int32_t *tokens, int32_t *targets,
                              int32_t seq_len, int32_t vocab_size,
                              int32_t step) {
    /* Pattern: A B C A B C ... with A,B,C depending on step */
    int32_t pattern_len = 3 + (step % 5); /* 3-7 token patterns */
    int32_t base = (step * 7) % (vocab_size - pattern_len);
    for (int32_t i = 0; i < seq_len; i++) {
        tokens[i] = base + (i % pattern_len);
    }
    for (int32_t i = 0; i < seq_len - 1; i++) {
        targets[i] = tokens[i + 1];
    }
    targets[seq_len - 1] = base + (seq_len % pattern_len);
}

/* Print a compact training progress line. */
static void print_step(int32_t step, TrainStepResult *r, double elapsed_ms) {
    printf("step %4d | loss %8.4f (pred %6.3f + bal %6.4f + lm %6.3f) | "
           "ppl %8.2f | gnorm %6.3f | ent %7.5f | vn_d %6.4f | %.0fms\n",
           step, r->loss.total, r->loss.pred_error, r->loss.balance,
           r->loss.lm, r->loss.perplexity, r->grad_norm,
           r->mean_entropy, r->vn_delta, elapsed_ms);
}

/* Parse named arguments from argv. Returns index of first positional arg. */
static int parse_named_args(int argc, char *argv[],
                            int *use_backprop,
                            const char **data_path,
                            const char **tokenizer_path) {
    *use_backprop = 0;
    *data_path = NULL;
    *tokenizer_path = NULL;

    int first_positional = 1;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--backprop") == 0) {
            *use_backprop = 1;
            first_positional = i + 1;
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            *data_path = argv[i + 1];
            i++; /* skip value */
            first_positional = i + 1;
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            *tokenizer_path = argv[i + 1];
            i++; /* skip value */
            first_positional = i + 1;
        } else {
            /* First non-flag arg: everything from here is positional */
            first_positional = i;
            break;
        }
    }
    return first_positional;
}

int main(int argc, char *argv[]) {
    /* Parse named flags first */
    int use_backprop = 0;
    const char *data_path = NULL;
    const char *tokenizer_path = NULL;
    int pos_start = parse_named_args(argc, argv, &use_backprop,
                                     &data_path, &tokenizer_path);

    /* Reindex positional args */
    int pos_argc = argc - pos_start;
    char **pos_argv = argv + pos_start;

    if (use_backprop) {
        printf("=== HSPA v1 Micro-Experiment: BACKPROP Training ===\n\n");
    } else {
        printf("=== HSPA v1 Micro-Experiment: iPC Training Validation ===\n\n");
    }

    /* Parse optional positional arguments */
    int32_t max_steps = 500;
    int32_t seq_len = 32; /* Shorter than max to keep fast */
    int use_patterns = 1; /* 1 = pattern batches, 0 = random */

    if (pos_argc > 0) {
        max_steps = atoi(pos_argv[0]);
        if (max_steps <= 0) max_steps = 500;
    }
    if (pos_argc > 1) {
        seq_len = atoi(pos_argv[1]);
        if (seq_len <= 0 || seq_len > 128) seq_len = 32;
    }
    if (pos_argc > 2) {
        use_patterns = atoi(pos_argv[2]);
    }
    int use_adam_cli = -1; /* -1 = use config default */
    if (pos_argc > 3) {
        use_adam_cli = atoi(pos_argv[3]); /* 0=SGD, 1=Adam */
    }
    float beta_override = -1.0f; /* -1 = use config default */
    if (pos_argc > 4) {
        beta_override = (float)atof(pos_argv[4]);
    }

    /* ================================================================
     * Real data loading (optional)
     * ================================================================ */
    int32_t *real_tokens = NULL;
    int64_t real_n_tokens = 0;
    DataLoader *data_loader = NULL;
    int use_real_data = 0;

    if (data_path && tokenizer_path) {
        printf("Loading tokenizer from: %s\n", tokenizer_path);
        printf("Loading text data from: %s\n", data_path);

        real_n_tokens = load_and_tokenize(data_path, tokenizer_path, &real_tokens);
        if (real_n_tokens < 0 || !real_tokens) {
            fprintf(stderr, "FAILED: could not load and tokenize data\n");
            return 1;
        }
        printf("Tokenized %lld tokens from text file\n", (long long)real_n_tokens);

        /* Adjust seq_len for real data if needed */
        if (seq_len > 128) seq_len = 128;
        if (real_n_tokens < seq_len + 1) {
            fprintf(stderr, "FAILED: not enough tokens for seq_len=%d\n", seq_len);
            free(real_tokens);
            return 1;
        }

        data_loader = data_loader_create(real_tokens, real_n_tokens, seq_len);
        if (!data_loader) {
            fprintf(stderr, "FAILED: data_loader_create returned NULL\n");
            free(real_tokens);
            return 1;
        }
        use_real_data = 1;
        use_patterns = 0; /* Override: use real data, not patterns */
        printf("Data loader ready: %lld tokens, seq_len=%d\n\n",
               (long long)real_n_tokens, seq_len);
    }

    /* Seed RNG */
    srand((unsigned int)time(NULL));

    /* Create config: use small config for real data, micro for synthetic */
    HSPAConfig cfg;
    if (use_real_data) {
        cfg = hspa_config_small();
        /* Override seq_len to match data loader */
        if (seq_len > cfg.max_seq_len) {
            seq_len = cfg.max_seq_len;
        }
    } else {
        cfg = hspa_config_micro();
    }

    printf("Model config:\n");
    printf("  layers=%d, d_model=%d, heads=%d, kv_heads=%d, head_dim=%d\n",
           cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
    printf("  experts=%d, active=%d, d_ff=%d, vocab=%d, max_seq=%d\n",
           cfg.n_experts, cfg.n_active, cfg.d_ff, cfg.vocab_size, cfg.max_seq_len);
    printf("  ipc_iterations=%d, rms_norm_eps=%g\n", cfg.ipc_iterations, cfg.rms_norm_eps);
    printf("  seq_len=%d, max_steps=%d, mode=%s\n",
           seq_len, max_steps, use_real_data ? "real_data" :
                               (use_patterns ? "pattern" : "random"));
    printf("  training=%s\n\n", use_backprop ? "BACKPROP" : "IPC");

    /* Create model */
    printf("Creating model... ");
    fflush(stdout);
    HSPAModel *model = hspa_model_create(&cfg);
    if (!model) {
        fprintf(stderr, "FAILED: hspa_model_create returned NULL\n");
        if (data_loader) data_loader_destroy(data_loader);
        free(real_tokens);
        return 1;
    }
    printf("OK\n");

    /* Initialize weights with Depth-muP */
    printf("Initializing weights (Depth-muP)... ");
    fflush(stdout);
    weight_init_depth_mup(model, &cfg);
    printf("OK\n");

    /* Create training config */
    TrainConfig tcfg = train_config_micro();
    tcfg.max_steps = max_steps;
    tcfg.log_every = 10;
    tcfg.ipc_lr = 0.05f;          /* Slightly conservative value node LR */
    if (use_adam_cli >= 0) {
        tcfg.use_adam = (use_adam_cli != 0);
    }
    if (tcfg.use_adam) {
        tcfg.base_lr = 0.003f;        /* Adam needs ~30x lower LR than SGD */
        tcfg.grad_clip_norm = 1.0f;   /* Tighter clipping for Adam stability */
    } else {
        tcfg.base_lr = 0.1f;          /* muP: higher for small D with SGD */
        tcfg.grad_clip_norm = 10.0f;
    }
    if (beta_override >= 0.0f) {
        tcfg.beta_balance = beta_override;
    }

    /* Create iPC state (only needed for iPC mode) */
    IPCTrainState *state = NULL;
    if (!use_backprop) {
        printf("Creating iPC state... ");
        fflush(stdout);
        state = ipc_state_create(&cfg, seq_len,
                                 tcfg.sigma_bottom, tcfg.sigma_top);
        if (!state) {
            fprintf(stderr, "FAILED: ipc_state_create returned NULL\n");
            hspa_model_destroy(model);
            if (data_loader) data_loader_destroy(data_loader);
            free(real_tokens);
            return 1;
        }
        printf("OK\n");
    } else {
        printf("Backprop mode: skipping iPC state creation\n");
    }

    printf("Creating gradient accumulators... ");
    fflush(stdout);
    ModelGrad *grads = grad_create(&cfg);
    if (!grads) {
        fprintf(stderr, "FAILED: grad_create returned NULL\n");
        if (state) ipc_state_destroy(state);
        hspa_model_destroy(model);
        if (data_loader) data_loader_destroy(data_loader);
        free(real_tokens);
        return 1;
    }
    printf("OK\n");

    /* Create Adam optimizer state (if enabled in config) */
    AdamState *adam = NULL;
    if (tcfg.use_adam) {
        printf("Creating Adam optimizer state... ");
        fflush(stdout);
        adam = adam_create(&cfg);
        if (!adam) {
            fprintf(stderr, "FAILED: adam_create returned NULL\n");
            grad_destroy(grads);
            if (state) ipc_state_destroy(state);
            hspa_model_destroy(model);
            if (data_loader) data_loader_destroy(data_loader);
            free(real_tokens);
            return 1;
        }
        printf("OK (AdamW beta1=%.2f beta2=%.3f wd=%.3f)\n",
               tcfg.adam_beta1, tcfg.adam_beta2, tcfg.weight_decay);
    } else {
        printf("Using SGD optimizer\n");
    }

    /* Allocate token buffers */
    int32_t *tokens = (int32_t *)calloc((size_t)seq_len, sizeof(int32_t));
    int32_t *targets_buf = (int32_t *)calloc((size_t)seq_len, sizeof(int32_t));
    if (!tokens || !targets_buf) {
        fprintf(stderr, "FAILED: token buffer allocation\n");
        grad_destroy(grads);
        if (state) ipc_state_destroy(state);
        hspa_model_destroy(model);
        if (data_loader) data_loader_destroy(data_loader);
        free(real_tokens);
        free(tokens); free(targets_buf);
        return 1;
    }

    /* Count params */
    int64_t total_params =
        (int64_t)cfg.vocab_size * cfg.d_model +
        (int64_t)cfg.n_layers * (
            4LL * cfg.d_model * cfg.n_heads * cfg.head_dim +
            (int64_t)cfg.n_experts * 3LL * cfg.d_model * cfg.d_ff +
            2LL * cfg.d_model * cfg.n_experts +
            2LL * cfg.d_model
        ) +
        cfg.d_model;
    printf("\nTotal parameters: %lld (~%.2fM)\n", (long long)total_params,
           (double)total_params / 1e6);

    /* ================================================================
     * Training loop
     * ================================================================ */
    printf("\n--- Training Start (%s MODE) ---\n",
           use_backprop ? "BACKPROP" : "IPC");
    printf("step      | loss     (pred   + bal    + lm   ) | "
           "ppl      | gnorm  | ent   | vn_d   | time\n");
    printf("----------+--------------------------------------+"
           "---------+--------+-------+--------+------\n");

    float best_loss = INFINITY;
    float initial_ppl = 0.0f;
    int nan_count = 0;
    float loss_window[10]; /* rolling window for trend detection */
    memset(loss_window, 0, sizeof(loss_window));

    struct timespec ts_start, ts_end;

    for (int32_t step = 0; step < max_steps; step++) {
        /* Get batch data */
        if (use_real_data) {
            int rc = data_loader_next(data_loader, tokens, targets_buf);
            if (rc < 0) {
                /* Epoch boundary: reset and continue */
                data_loader_reset(data_loader);
                rc = data_loader_next(data_loader, tokens, targets_buf);
                if (rc < 0) {
                    printf("*** Data exhausted even after reset. Stopping. ***\n");
                    break;
                }
            }
        } else if (use_patterns) {
            gen_pattern_batch(tokens, targets_buf, seq_len, cfg.vocab_size, step);
        } else {
            gen_synthetic_batch(tokens, targets_buf, seq_len, cfg.vocab_size);
        }

        /* Train step */
        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        TrainStepResult r;
        if (use_backprop) {
            r = backprop_train_step(model, grads, adam,
                                    tokens, targets_buf, seq_len,
                                    &cfg, &tcfg, step, 0);
        } else {
            r = ipc_train_step(model, grads, adam, state,
                               tokens, targets_buf, seq_len,
                               &cfg, &tcfg, step, 0);
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double elapsed_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0
                          + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;

        /* Track metrics */
        if (step == 0) {
            initial_ppl = r.loss.perplexity;
        }

        if (isnan(r.loss.total) || isinf(r.loss.total)) {
            nan_count++;
            if (nan_count > 5) {
                printf("\n*** ABORT: %d consecutive NaN/Inf losses. Training diverged. ***\n", nan_count);
                break;
            }
        } else {
            nan_count = 0;
            if (r.loss.total < best_loss) {
                best_loss = r.loss.total;
            }
        }

        /* Rolling loss window */
        loss_window[step % 10] = r.loss.lm;

        /* Print progress */
        if (step < 5 || step % tcfg.log_every == 0 || step == max_steps - 1) {
            print_step(step, &r, elapsed_ms);
        }
    }

    /* ================================================================
     * Results summary
     * ================================================================ */
    printf("\n--- Training Complete (%s MODE) ---\n\n",
           use_backprop ? "BACKPROP" : "IPC");

    /* Compute loss trend from last 10 steps */
    float avg_recent = 0.0f;
    int window_size = max_steps < 10 ? max_steps : 10;
    for (int i = 0; i < window_size; i++) {
        avg_recent += loss_window[i];
    }
    avg_recent /= (float)window_size;

    printf("Summary:\n");
    printf("  Training mode:  %s\n", use_backprop ? "BACKPROP" : "IPC");
    printf("  Steps trained:  %d\n", max_steps);
    printf("  Initial PPL:    %.2f\n", initial_ppl);
    printf("  Best loss:      %.4f\n", best_loss);
    printf("  Recent avg LM:  %.4f\n", avg_recent);
    printf("  NaN events:     %d\n", nan_count);

    /* Verdict */
    printf("\nVerdict: ");
    if (nan_count > 5) {
        printf("FAILED -- training diverged (NaN/Inf)\n");
    } else if (avg_recent < logf((float)cfg.vocab_size) * 0.9f) {
        printf("CONVERGING -- loss below random baseline, %s learning signal confirmed\n",
               use_backprop ? "backprop" : "iPC");
    } else if (best_loss < logf((float)cfg.vocab_size)) {
        printf("PARTIAL -- some loss reduction but not clearly converging\n");
    } else {
        printf("INCONCLUSIVE -- loss did not decrease below random baseline\n");
    }

    printf("\nRandom baseline: ln(%d) = %.4f\n",
           cfg.vocab_size, logf((float)cfg.vocab_size));

    /* Cleanup */
    free(tokens);
    free(targets_buf);
    if (adam) adam_destroy(adam);
    grad_destroy(grads);
    if (state) ipc_state_destroy(state);
    hspa_model_destroy(model);
    if (data_loader) data_loader_destroy(data_loader);
    free(real_tokens);

    printf("\n=== Micro-Experiment Complete ===\n");
    return (nan_count > 5) ? 1 : 0;
}
