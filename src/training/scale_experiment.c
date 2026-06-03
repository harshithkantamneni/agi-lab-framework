/* scale_experiment.c -- Scale-up experiment: train HSPA on real text data.
 *
 * Validates that iPC training scales from micro (0.69M) to 10M-50M params
 * using real English text (WikiText-2) with BPE tokenization.
 *
 * Key metrics tracked:
 *   - Loss (total, pred_error, balance, LM) and perplexity
 *   - Router entropy and specialization (per-layer)
 *   - Gradient norms
 *   - Value node convergence delta
 *   - Training speed (ms/step)
 *   - Epoch boundaries
 *
 * Usage: build/scale_experiment [options]
 *   --model small|medium|100m Model size (default: small ~10M;
 *                              100m = Phase C ~119.6M per phase_c_design.md §1.7)
 *   --data PATH               Training text file
 *   --tokenizer PATH          BPE tokenizer binary
 *   --steps N                 Max training steps (default: 2000)
 *   --seq-len N               Sequence length (default: 128)
 *   --lr FLOAT                Base learning rate override
 *   --checkpoint-dir PATH     Where to save checkpoints
 */

#include "backprop_train.h"
#include "checkpoint.h"
#include "config_drift.h"
#include "data_loader.h"
#include "hspa_model.h"
#include "ipc_train.h"
#include "local_feedback_train.h"
#include "metal_bridge.h"
#include "ops.h"
#include "tokenizer.h"
#include "weight_init.h"

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef __APPLE__
#include <sys/qos.h>
#include <pthread.h>
#endif

/* ---- Signal handler + atexit for crash diagnosis ---- */
static volatile sig_atomic_t g_signal_received = 0;

static void signal_handler(int sig) {
    g_signal_received = sig;
    const char *name = "UNKNOWN";
    switch (sig) {
        case SIGTERM: name = "SIGTERM"; break;
        case SIGHUP:  name = "SIGHUP";  break;
        case SIGUSR1: name = "SIGUSR1"; break;
        case SIGUSR2: name = "SIGUSR2"; break;
        case SIGABRT: name = "SIGABRT"; break;
        case SIGSEGV: name = "SIGSEGV"; break;
        case SIGBUS:  name = "SIGBUS";  break;
    }
    /* async-signal-safe write */
    write(STDERR_FILENO, "\n*** SIGNAL: ", 13);
    write(STDERR_FILENO, name, strlen(name));
    write(STDERR_FILENO, " ***\n", 5);
    _exit(128 + sig);
}

static void atexit_handler(void) {
    fprintf(stderr, "\n*** ATEXIT: clean process exit ***\n");
    fflush(stderr);
}

/* ---- Progressive sparsification helper ---- */

/* Compute effective n_active at a given step via linear annealing.
 * Returns start at step 0, end at step >= total_steps. */
static int32_t compute_effective_k(int32_t step, int32_t start, int32_t end,
                                   int32_t total_steps) {
    if (start <= end || total_steps <= 0 || step >= total_steps)
        return end;
    float t = (float)step / (float)total_steps;
    int32_t ek = start - (int32_t)(t * (float)(start - end) + 0.5f);
    if (ek < end) ek = end;
    if (ek > start) ek = start;
    return ek;
}

/* ---- Argument parsing ---- */

typedef struct {
    const char *model_size;     /* "small", "medium", or "100m"/"large" (Phase C) */
    const char *data_path;
    const char *tokenizer_path;
    const char *checkpoint_dir;
    const char *resume_path;    /* checkpoint to resume from, or NULL */
    int32_t max_steps;
    int32_t seq_len;
    int32_t grad_accum;         /* gradient accumulation steps */
    int32_t checkpoint_every;   /* save checkpoint every N steps */
    float   lr_override;        /* <= 0 means use default */
    float   beta_balance;       /* <= 0 means use default (0.01) */
    float   balance_h_floor;    /* <= 0 means always-on balance (default) */
    float   balance_h_target;   /* entropy target for three-regime control. <= 0 means disabled */
    float   router_pred_scale;  /* router pred gradient scale. <= 0 means default (0.05) */
    int32_t n_active_start;     /* starting top-K for progressive sparsification. 0=disabled */
    int32_t sparsify_steps;     /* steps over which to anneal n_active_start → cfg.n_active */
    bool    loss_free_balance;  /* D-031: disable entropy gradient, use router bias only */
    float   bias_step_size;     /* router bias step size. <= 0 means default (0.001) */
    float   bias_update_rate;   /* router bias EMA rate. <= 0 means default (0.01) */
    bool    use_backprop;       /* D-082: use backprop instead of iPC */
    bool    use_qat;            /* Program 3 P7: enable 4-bit QAT for in-scope weights */
    int     qat_group_size;     /* INT4 quantization group size; default 128 */
    bool    use_localfb;        /* Program 3 P7: use LocalFB (DFA) arm instead of iPC */
    bool    use_stream;         /* D-091 Phase B: 5-source streaming mixture loader */
    uint64_t stream_seed;       /* RNG seed for stream source selection */
    uint32_t weight_seed;       /* Cycle 32: 0 = use time(NULL) (legacy); nonzero = explicit seed */
    /* D-Plan-B: Default MoE / Dense Backprop (arXiv 2504.12463). */
    int32_t default_moe_state;  /* 0=unset (auto), 1=force-on, -1=force-off */
    float   default_moe_alpha;  /* <=0 means default (0.01) */
    float   default_moe_sigma_init; /* <=0 means default (0.01) */
    /* Cycle 29 Rev-2: Entropy-penalty + τ-anneal (escalation stack). */
    int32_t entropy_penalty_state;      /* 0=unset, 1=on, -1=off */
    float   entropy_penalty_beta;       /* <=0 means default (0.15) */
    float   entropy_penalty_target;     /* <=0 means default (0.90) */
    int32_t entropy_warmup_steps;       /* <0 means default (0) */
    int32_t beta_h_warmup_steps;        /* <0 means default (200) */
    int32_t temp_anneal_state;          /* 0=unset, 1=on, -1=off */
    float   tau_max;                    /* <=0 means default (1.4) */
    float   tau_min;                    /* <=0 means default (1.0) */
    int32_t tau_anneal_steps;           /* <=0 means default (500) */
    int32_t temp_anneal_restoration_state; /* 0=unset, 1=on, -1=off */
} ScaleArgs;

static ScaleArgs parse_args(int argc, char *argv[]) {
    ScaleArgs args = {
        .model_size      = "small",
        .data_path       = "data/training/active/wikitext2_train.txt",
        .tokenizer_path  = "data/training/tokenizer_4k.bin",
        .checkpoint_dir  = "data/checkpoints",
        .resume_path     = NULL,
        .max_steps       = 2000,
        .seq_len         = 128,
        .grad_accum      = 4,
        .checkpoint_every = 500,
        .lr_override     = -1.0f,
        .beta_balance    = -1.0f,
        .balance_h_floor = -1.0f,
        .balance_h_target = -1.0f,
        .router_pred_scale = -1.0f,
        .n_active_start  = 0,
        .sparsify_steps  = 0,
        .loss_free_balance = false,
        .bias_step_size  = -1.0f,
        .bias_update_rate = -1.0f,
        .use_backprop    = false,
        .use_qat         = false,
        .qat_group_size  = 128,
        .use_localfb     = false,
        .use_stream      = false,
        .stream_seed     = 1337ULL,
        .weight_seed     = 0u,           /* 0 = legacy time(NULL) path (preserves existing behavior) */
        .default_moe_state = 0,          /* auto: ON when --backprop, else OFF */
        .default_moe_alpha = -1.0f,
        .default_moe_sigma_init = -1.0f,
        .entropy_penalty_state = 0,
        .entropy_penalty_beta  = -1.0f,
        .entropy_penalty_target = -1.0f,
        .entropy_warmup_steps  = -1,
        .beta_h_warmup_steps   = -1,
        .temp_anneal_state     = 0,
        .tau_max               = -1.0f,
        .tau_min               = -1.0f,
        .tau_anneal_steps      = -1,
        .temp_anneal_restoration_state = 0,
    };
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            args.model_size = argv[++i];
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            args.data_path = argv[++i];
        } else if (strcmp(argv[i], "--tokenizer") == 0 && i + 1 < argc) {
            args.tokenizer_path = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            args.max_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) {
            args.seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            args.lr_override = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--grad-accum") == 0 && i + 1 < argc) {
            args.grad_accum = atoi(argv[++i]);
            if (args.grad_accum < 1) args.grad_accum = 1;
        } else if (strcmp(argv[i], "--checkpoint-dir") == 0 && i + 1 < argc) {
            args.checkpoint_dir = argv[++i];
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            args.resume_path = argv[++i];
        } else if (strcmp(argv[i], "--checkpoint-every") == 0 && i + 1 < argc) {
            args.checkpoint_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--beta-balance") == 0 && i + 1 < argc) {
            args.beta_balance = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--balance-h-floor") == 0 && i + 1 < argc) {
            args.balance_h_floor = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--balance-h-target") == 0 && i + 1 < argc) {
            args.balance_h_target = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--router-pred-scale") == 0 && i + 1 < argc) {
            args.router_pred_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--n-active-start") == 0 && i + 1 < argc) {
            args.n_active_start = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--sparsify-steps") == 0 && i + 1 < argc) {
            args.sparsify_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--loss-free-balance") == 0) {
            args.loss_free_balance = true;
        } else if (strcmp(argv[i], "--backprop") == 0) {
            args.use_backprop = true;
        } else if (strcmp(argv[i], "--qat") == 0) {
            args.use_qat = true;
        } else if (strcmp(argv[i], "--qat-group-size") == 0 && i + 1 < argc) {
            args.qat_group_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--localfb") == 0) {
            args.use_localfb = true;
        } else if (strcmp(argv[i], "--stream") == 0) {
            args.use_stream = true;
        } else if (strcmp(argv[i], "--stream-seed") == 0 && i + 1 < argc) {
            args.stream_seed = strtoull(argv[++i], NULL, 0);
        } else if (strcmp(argv[i], "--weight-seed") == 0 && i + 1 < argc) {
            args.weight_seed = (uint32_t)strtoul(argv[++i], NULL, 0);
        } else if (strcmp(argv[i], "--bias-step-size") == 0 && i + 1 < argc) {
            args.bias_step_size = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--bias-update-rate") == 0 && i + 1 < argc) {
            args.bias_update_rate = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--default-moe") == 0) {
            args.default_moe_state = 1;
        } else if (strcmp(argv[i], "--no-default-moe") == 0) {
            args.default_moe_state = -1;
        } else if (strcmp(argv[i], "--default-moe-alpha") == 0 && i + 1 < argc) {
            args.default_moe_alpha = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--default-moe-sigma-init") == 0 && i + 1 < argc) {
            args.default_moe_sigma_init = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--entropy-penalty") == 0) {
            args.entropy_penalty_state = 1;
        } else if (strcmp(argv[i], "--no-entropy-penalty") == 0) {
            args.entropy_penalty_state = -1;
        } else if (strcmp(argv[i], "--entropy-penalty-beta") == 0 && i + 1 < argc) {
            args.entropy_penalty_state = 1;
            args.entropy_penalty_beta = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--entropy-penalty-target") == 0 && i + 1 < argc) {
            args.entropy_penalty_state = 1;
            args.entropy_penalty_target = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--entropy-warmup-steps") == 0 && i + 1 < argc) {
            args.entropy_warmup_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--beta-h-warmup-steps") == 0 && i + 1 < argc) {
            args.beta_h_warmup_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp-anneal") == 0) {
            args.temp_anneal_state = 1;
        } else if (strcmp(argv[i], "--no-temp-anneal") == 0) {
            args.temp_anneal_state = -1;
        } else if (strcmp(argv[i], "--tau-max") == 0 && i + 1 < argc) {
            args.temp_anneal_state = 1;
            args.tau_max = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--tau-min") == 0 && i + 1 < argc) {
            args.tau_min = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--tau-anneal-steps") == 0 && i + 1 < argc) {
            args.temp_anneal_state = 1;
            args.tau_anneal_steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp-anneal-restoration") == 0) {
            args.temp_anneal_restoration_state = 1;
        } else if (strcmp(argv[i], "--no-temp-anneal-restoration") == 0) {
            args.temp_anneal_restoration_state = -1;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: scale_experiment [--model small|medium|100m|large|dense50m] [--data PATH] "
                   "[--tokenizer PATH] [--steps N] [--seq-len N] [--lr FLOAT] "
                   "[--grad-accum N] [--checkpoint-dir PATH] [--resume PATH] "
                   "[--checkpoint-every N] [--beta-balance FLOAT] "
                   "[--balance-h-floor FLOAT] [--balance-h-target FLOAT] "
                   "[--router-pred-scale FLOAT] "
                   "[--n-active-start N] [--sparsify-steps N] "
                   "[--loss-free-balance] [--bias-step-size FLOAT] "
                   "[--bias-update-rate FLOAT] [--backprop] "
                   "[--default-moe | --no-default-moe] "
                   "[--default-moe-alpha FLOAT] [--default-moe-sigma-init FLOAT] "
                   "[--entropy-penalty | --no-entropy-penalty] "
                   "[--entropy-penalty-beta FLOAT] [--entropy-penalty-target FLOAT] "
                   "[--entropy-warmup-steps N] [--beta-h-warmup-steps N] "
                   "[--temp-anneal | --no-temp-anneal] "
                   "[--tau-max FLOAT] [--tau-min FLOAT] [--tau-anneal-steps N] "
                   "[--temp-anneal-restoration | --no-temp-anneal-restoration] "
                   "[--stream] [--stream-seed U64] [--weight-seed INT]\n"
                   "  Note: --balance-h-floor / --balance-h-target / "
                   "non-default --beta-balance are iPC-mode-only (without "
                   "--loss-free-balance). In --backprop or --loss-free-balance "
                   "mode these flags are rejected (see review_balance_h_floor_backprop_bug.md).\n");
            exit(0);
        }
    }
    return args;
}

/* ---- D-191 / P-CONFIG-DRIFT-LR-WIRE: shared LR-pin constants ----
 *
 * Three values that scale_experiment.c assigns to TrainConfig in the init path
 * are *also* read by the config-drift assertion. To prevent silent literal
 * drift between the two assignment sites (which would hide a real
 * scale_experiment.c → manifest disagreement behind a "they happen to match"
 * coincidence), we factor them into named constants here. Both the assertion
 * site (where ConfigDriftRuntime is populated) and the TrainConfig setup
 * (where tcfg.{grad_clip_norm,lr_warmup_steps} get assigned) reference these
 * same names. */
#define SCALE_GRAD_CLIP_NORM_DEFAULT  1.0f
#define SCALE_LR_WARMUP_STEPS_DEFAULT 200

/* ---- D-191 / P-CONFIG-DRIFT-LR-WIRE: runtime LR-base derivation ----
 *
 * Single source of truth for the per-arm base LR keyed off `args.model_size`.
 * Used both by the config-drift assertion (BEFORE model_create / training)
 * and by the TrainConfig setup later in main(). Centralizing the derivation
 * here ensures the assertion sees exactly the value training will use — no
 * duplicated literal that could silently drift apart.
 *
 * Logic (from D-181 stat-review trail; was inline at scale_experiment.c:714-718
 * pre-D-191):
 *   - --lr <override> wins (operator-typed; rare)
 *   - else: 0.002 default; 0.001 if --model medium (the historical 2× MoE LR)
 *
 * Note this still embeds the per-arm 2×-LR confound that D-181 surfaced; the
 * Phase-3 factorial design (D-190 1i) AMENDS spec_invariants.yaml per cell to
 * pin the arm-specific value, and the config_drift comparator (parallel
 * tooling_engineer dispatch) hard-FAILs if the manifest disagrees with this
 * helper's output. */
static float scale_compute_lr_base(const ScaleArgs *args) {
    if (args->lr_override > 0.0f) return args->lr_override;
    if (strcmp(args->model_size, "medium") == 0) return 0.001f;
    return 0.002f;
}

/* ---- Metric tracking ---- */

typedef struct {
    float best_loss;
    float best_ppl;
    float initial_ppl;
    float loss_window[50];      /* rolling window for trend */
    int   window_idx;
    int   nan_count;
    int   total_nan;
    int   epochs_completed;
    int64_t tokens_trained;
} Metrics;

static void metrics_init(Metrics *m) {
    m->best_loss = INFINITY;
    m->best_ppl = INFINITY;
    m->initial_ppl = 0.0f;
    memset(m->loss_window, 0, sizeof(m->loss_window));
    m->window_idx = 0;
    m->nan_count = 0;
    m->total_nan = 0;
    m->epochs_completed = 0;
    m->tokens_trained = 0;
}

static float metrics_avg_loss(const Metrics *m, int steps) {
    int n = steps < 50 ? steps : 50;
    if (n == 0) return 0.0f;
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += m->loss_window[i];
    return sum / (float)n;
}

/* ---- Logging ---- */

static void print_header(void) {
    printf("step   | loss      (pred   + bal    + lm    ) | "
           "ppl       | gnorm  | ent     | vn_d   | ep | ms\n");
    printf("-------+---------------------------------------+"
           "----------+--------+---------+--------+----+------\n");
}

static void print_step(int32_t step, TrainStepResult *r, double ms, int epoch) {
    printf("%5d  | %8.4f (%6.3f + %6.4f + %6.3f) | %9.2f | %6.3f | %7.5f | %6.4f | %2d | %.0f\n",
           step, r->loss.total, r->loss.pred_error, r->loss.balance,
           r->loss.lm, r->loss.perplexity, r->grad_norm,
           r->mean_entropy, r->vn_delta, epoch, ms);
}

/* ---- Main ---- */

int main(int argc, char *argv[]) {
    /* Line-buffer stdout so output appears immediately when piped to tee. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    /* Install signal handlers for crash diagnosis. */
    signal(SIGTERM, signal_handler);
    signal(SIGHUP,  signal_handler);
    signal(SIGUSR1, signal_handler);
    signal(SIGUSR2, signal_handler);
    signal(SIGABRT, signal_handler);
    signal(SIGSEGV, signal_handler);
    signal(SIGBUS,  signal_handler);
    atexit(atexit_handler);

    /* Force P-core scheduling so macOS doesn't demote to E-cores. */
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif

    printf("=== HSPA Scale-Up Experiment: Real Text Training ===\n\n");

    ScaleArgs args = parse_args(argc, argv);

    /* --- Guardrail: --balance-h-floor / --balance-h-target / --beta-balance
     * require an auxiliary-loss entropy mechanism that currently exists
     * ONLY in ipc_train.c, and only when use_loss_free_balance is false.
     * In --backprop or --loss-free-balance modes these flags are silently
     * ignored -- see review_balance_h_floor_backprop_bug.md. Refuse rather
     * than silently no-op. Explicit beta_balance == 0.01 (the default) is
     * tolerated because it matches the effective default either way. */
    {
        bool beta_override = (args.beta_balance > 0.0f
                              && args.beta_balance != 0.01f);
        bool h_knob_set    = (args.balance_h_floor > 0.0f
                              || args.balance_h_target > 0.0f);
        if (h_knob_set || beta_override) {
            if (args.use_backprop) {
                fprintf(stderr,
                    "FATAL: --balance-h-floor / --balance-h-target / "
                    "--beta-balance are not implemented in --backprop mode "
                    "(see data/engineering/reviews/review_balance_h_floor_backprop_bug.md). "
                    "Remove these flags, or drop --backprop.\n");
                return 1;
            }
            if (args.loss_free_balance) {
                fprintf(stderr,
                    "FATAL: --balance-h-floor / --balance-h-target / "
                    "--beta-balance are incompatible with --loss-free-balance: "
                    "LFB disables the entropy gradient that implements these. "
                    "Remove these flags, or drop --loss-free-balance.\n");
                return 1;
            }
        }
    }

    /* --- Guardrail: Cycle 29 Rev-2 entropy-penalty + τ-anneal escalation.
     * Hard caps enforced per chief_scientist spec
     * (data/engineering/entropy_penalty_temp_anneal_design.md §Rev-2):
     *   β_H ≤ 0.30       (over-smoothing risk above)
     *   H_target ≤ 1.0   (log(K)/log(K) upper bound for K=8)
     *   τ_max ≤ 1.5      (router instability risk above)
     *   --entropy-penalty / --temp-anneal require --backprop (iPC path
     *   does NOT implement the penalty gradient or τ-divisor). */
    if ((args.entropy_penalty_state > 0 || args.temp_anneal_state > 0)
        && !args.use_backprop) {
        fprintf(stderr,
            "FATAL: --entropy-penalty and --temp-anneal require --backprop. "
            "iPC mode does not implement the entropy-penalty gradient or "
            "τ-divisor. Add --backprop or drop these flags.\n");
        return 1;
    }
    if (args.entropy_penalty_beta > 0.30f) {
        fprintf(stderr,
            "FATAL: --entropy-penalty-beta=%.3f exceeds Rev-2 hard cap 0.30 "
            "(over-smoothing risk). See "
            "data/engineering/entropy_penalty_temp_anneal_design.md.\n",
            args.entropy_penalty_beta);
        return 1;
    }
    if (args.entropy_penalty_target > 1.0f) {
        fprintf(stderr,
            "FATAL: --entropy-penalty-target=%.3f exceeds 1.0 "
            "(log(K)/log(K) upper bound for K=8).\n",
            args.entropy_penalty_target);
        return 1;
    }
    if (args.tau_max > 1.5f) {
        fprintf(stderr,
            "FATAL: --tau-max=%.3f exceeds Rev-2 hard cap 1.5 "
            "(router instability risk).\n",
            args.tau_max);
        return 1;
    }
    if (args.tau_min > 0.0f && args.tau_max > 0.0f
        && args.tau_min > args.tau_max) {
        fprintf(stderr,
            "FATAL: --tau-min=%.3f > --tau-max=%.3f (anneal direction "
            "must be decreasing).\n", args.tau_min, args.tau_max);
        return 1;
    }

    /* Program 3 Phase 7: arm mutual-exclusion.
     * --backprop and --localfb are mutually exclusive: each selects a distinct
     * training arm (Arm A vs Arm C). Specifying both is a configuration error
     * that would produce an ambiguous dispatch. */
    if (args.use_backprop && args.use_localfb) {
        fprintf(stderr,
            "FATAL: --backprop and --localfb are mutually exclusive. "
            "Select exactly one training arm: "
            "Arm A (--backprop), Arm B (default iPC), or Arm C (--localfb).\n");
        return 1;
    }

    /* Seed RNG. Cycle 32: --weight-seed <int> (nonzero) pins weight init for
     * multi-seed reproducibility (Grant Rec #2 prereq; see
     * data/engineering/multiseed_10m_rev2_design.md §10). Default 0 preserves
     * legacy time(NULL) behavior. Banner line records which path was taken so
     * training.log is self-describing. */
    unsigned int seed_for_rand = args.weight_seed
                                 ? (unsigned int)args.weight_seed
                                 : (unsigned int)time(NULL);
    srand(seed_for_rand);
    if (args.weight_seed) {
        printf("  weight_seed=%u (explicit --weight-seed)\n", seed_for_rand);
    } else {
        printf("  weight_seed=time (legacy; srand=%u)\n", seed_for_rand);
    }

    /* ---- Step 1: Load data ---- */
    printf("[1/5] Loading training data...\n");

    /* Stream mode (D-091 Phase B): 5-source streaming mixture loader.
     * Legacy single-file path stays intact for non-stream callers. */
    int32_t *all_tokens = NULL;         /* legacy path: pre-tokenised array */
    int64_t  n_tokens   = 0;            /* legacy path: total token count */
    int32_t  max_token  = 0;            /* for vocab sanity check */
    Tokenizer *stream_tok = NULL;       /* stream path: shared tokenizer */

    /* D-091 source table. File sizes and weights straight from manifest + decision. */
    #define STREAM_N_SOURCES 5
    static const char *stream_paths[STREAM_N_SOURCES] = {
        "data/training/active/wikitext103_train.txt",
        "data/training/raw/openwebtext.txt",
        "data/training/raw/python_code.txt",
        "data/training/raw/gsm8k_train.txt",
        "data/training/raw/math_train.txt",
    };
    static const char *stream_names[STREAM_N_SOURCES] = {
        "WT103", "OWT", "Python", "GSM8K", "MATH",
    };
    /* D-091 recommended mix weights */
    static const float stream_weights[STREAM_N_SOURCES] = {
        0.05f, 0.45f, 0.30f, 0.08f, 0.12f,
    };

    if (args.use_stream) {
        printf("  Mode: STREAM (5-source mixture, D-091)\n");
        printf("  Tokenizer: %s\n", args.tokenizer_path);
        stream_tok = tokenizer_load(args.tokenizer_path);
        if (!stream_tok) {
            fprintf(stderr, "FAILED: tokenizer_load('%s')\n", args.tokenizer_path);
            return 1;
        }
        printf("  Sources:\n");
        for (int i = 0; i < STREAM_N_SOURCES; i++) {
            struct stat st;
            if (stat(stream_paths[i], &st) != 0) {
                fprintf(stderr, "FAILED: source missing '%s'\n", stream_paths[i]);
                tokenizer_destroy(stream_tok);
                return 1;
            }
            printf("    [%d] %-6s %-44s  %.1f MB  w=%.2f\n", i, stream_names[i],
                   stream_paths[i], (double)st.st_size / (1024.0 * 1024.0),
                   (double)stream_weights[i]);
        }
        /* For diagnostics: vocab must match the tokenizer. We take max_token
         * from the tokenizer's vocab_size itself. */
        max_token = stream_tok->vocab_size - 1;
        printf("  Tokenizer vocab: %d\n", stream_tok->vocab_size);
    } else {
        printf("  Text: %s\n", args.data_path);
        printf("  Tokenizer: %s\n", args.tokenizer_path);
        n_tokens = load_and_tokenize(args.data_path, args.tokenizer_path,
                                     &all_tokens);
        if (n_tokens < 0 || !all_tokens) {
            fprintf(stderr, "FAILED: could not load and tokenize data\n");
            return 1;
        }
        printf("  Tokens: %lld (%.2f MB text)\n", (long long)n_tokens,
               (double)n_tokens * 4.0 / (1024.0 * 1024.0));
        for (int64_t i = 0; i < n_tokens; i++) {
            if (all_tokens[i] > max_token) max_token = all_tokens[i];
        }
        printf("  Max token ID: %d\n", max_token);
    }

    /* ---- Step 2: Create model ---- */
    printf("\n[2/5] Creating model (size=%s)...\n", args.model_size);

    HSPAConfig cfg;
    bool is_dense_a = false;
    if (strcmp(args.model_size, "100m") == 0 ||
        strcmp(args.model_size, "large") == 0) {
        /* Phase C 100M config (phase_c_design.md §1.7). */
        cfg = hspa_config_100m();
    } else if (strcmp(args.model_size, "medium") == 0) {
        cfg = hspa_config_medium();
    } else if (strcmp(args.model_size, "small") == 0) {
        cfg = hspa_config_small();
    } else if (strcmp(args.model_size, "dense50m") == 0) {
        /* Dense-A compute-matched control for Program 2
         * (dense_50m_control_design.md §1.3, question.md operational form).
         * L=8 D=512 H=8 Hkv=4 head_dim=64 d_ff=768 V=32768 K=1 k=1, ~34.62M params.
         * FLOPs-matched to Rev-2 MoE (k=2, d_ff=384) active path. */
        cfg = hspa_config_dense_50m_a();
        is_dense_a = true;
    } else {
        /* Reject unknown model flags with a fatal error instead of silently
         * falling through to small (Cycle 32 Program-2 PC-1 lesson: silent
         * fallback produced a 10M MoE-small when Dense-A was requested, and
         * only the printed Config line caught it). */
        fprintf(stderr,
                "FATAL: --model '%s' is not recognized. "
                "Valid values: small | medium | 100m | large | dense50m.\n",
                args.model_size);
        exit(1);
    }

    /* Program 2 guardrail (dense_50m_control_design.md §1.3): MoE-specific
     * features must not be combined with the Dense-A (n_experts=1) factory.
     * Dense-A has a single expert; entropy penalty targets H~log(K) which at
     * K=1 saturates (H_max = log(1) = 0), loss-free-balance has no peer experts
     * to balance against, default-moe aux loss divides by inter-expert
     * variance (zero), and temp-anneal modulates softmax over K=1 (no-op at
     * best, misleading at worst). Reject at launch to prevent meaningless
     * runs and false-comparable metrics in the primary artifact. */
    if (is_dense_a) {
        bool moe_flag_set =
            args.loss_free_balance ||
            args.default_moe_state == 1 ||
            args.entropy_penalty_state == 1 ||
            args.temp_anneal_state == 1;
        if (moe_flag_set) {
            fprintf(stderr,
                    "FATAL: --model dense50m is incompatible with MoE-specific "
                    "flags. Detected enabled: "
                    "loss-free-balance=%d default-moe=%d entropy-penalty=%d "
                    "temp-anneal=%d. These features assume n_experts>1 "
                    "(routing/balancing across multiple experts); Dense-A has "
                    "n_experts=1. Remove these flags or drop --model dense50m.\n",
                    (int)args.loss_free_balance,
                    args.default_moe_state == 1,
                    args.entropy_penalty_state == 1,
                    args.temp_anneal_state == 1);
            exit(1);
        }
    }

    /* D-091 Phase B: the 32K tokenizer expands V from 4096 to 32768. Model
     * dims are unchanged; only the embedding/output dimensions grow. */
    if (args.use_stream) {
        cfg.vocab_size = stream_tok->vocab_size;   /* should be 32768 */
        printf("  [stream] vocab_size set to %d (tokenizer)\n", cfg.vocab_size);
    }

    /* Ensure vocab covers all tokens */
    if (max_token >= cfg.vocab_size) {
        printf("  WARNING: max token %d >= vocab %d, adjusting vocab_size\n",
               max_token, cfg.vocab_size);
        cfg.vocab_size = max_token + 1;
    }

    /* Clamp seq_len to model max */
    int32_t seq_len = args.seq_len;
    if (seq_len > cfg.max_seq_len) {
        printf("  Clamping seq_len %d to max %d\n", seq_len, cfg.max_seq_len);
        seq_len = cfg.max_seq_len;
    }

    printf("  Config: L=%d D=%d H=%d K=%d k=%d dff=%d V=%d S=%d\n",
           cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.n_experts,
           cfg.n_active, cfg.d_ff, cfg.vocab_size, seq_len);

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
    printf("  Parameters: %lld (~%.2fM)\n", (long long)total_params,
           (double)total_params / 1e6);

    /* ---- Config-drift assertion (P-CONFIG-DRIFT-DETECTION, D-121;
     *      P-CONFIG-DRIFT-LR-WIRE, D-191) ----
     * Opt-in via env LAB_PROGRAM. When set, compares the *actual* runtime
     * values just printed in the init banner above against the program's
     * frozen manifest at programs/$LAB_PROGRAM/spec_invariants.yaml. On any
     * mismatch the assertion prints "CONFIG-DRIFT FATAL: ..." and exit(1)s
     * BEFORE any model_create / weight_init / training step / checkpoint
     * write — preventing the D-119–D-121 Swiss-cheese class of failure where
     * a launch-template tokenizer omission silently let training proceed on
     * the wrong vocab. When LAB_PROGRAM is unset (typical for one-off
     * binary invocations and most existing tests), the call is a no-op.
     * See src/training/config_drift.h for activation/format details.
     *
     * D-191 / P-CONFIG-DRIFT-LR-WIRE extension: in addition to the original
     * 5 structural fields (model / tokenizer / vocab / param counts), we now
     * populate 14 LR/schedule/dynamics fields that PI D-181 condition #1
     * pinned in spec_invariants.yaml. Phase-3 factorial design (D-190 1i)
     * makes silent LR drift verdict-breaking, so each field flows through
     * the LIVE runtime context here:
     *   - lr_base       — scale_compute_lr_base(&args) (the same helper the
     *                     TrainConfig setup uses below, so the manifest is
     *                     compared against the value training WILL use)
     *   - lr_schedule   — "linear_warmup_cosine_decay" (single hardcoded
     *                     path in backprop_train.c / ipc_train.c)
     *   - lr_warmup_steps — 200 (hardcoded at scale_experiment.c later)
     *   - lr_min        — train_config_micro() default 1e-5f (read directly
     *                     from the helper, not duplicated literal)
     *   - optimizer     — "adamw" (single path: tcfg.use_adam=true below)
     *   - adam_beta1/2  — train_config_micro() defaults (0.9 / 0.999)
     *   - weight_decay  — train_config_micro() default (0.01)
     *   - grad_clip_norm — 1.0f (hardcoded at scale_experiment.c later)
     *   - grad_accum_steps — args.grad_accum (CLI arg)
     *   - batch_size    — 1 (scale_experiment uses one micro-batch per step;
     *                     train_config_micro() default)
     *   - seq_len       — local seq_len variable (already clamped to model max)
     *   - max_steps     — args.max_steps (CLI arg)
     *   - weight_seed   — args.weight_seed (CLI arg)
     */
    {
        /* Active param count: for K=1 dense this equals total; for MoE
         * (n_active < n_experts), recompute the per-token active count by
         * substituting n_active for n_experts in the FFN term. The router
         * cost remains K (not k) because the gate is computed before
         * top-k selection. */
        int64_t active_params;
        bool    has_active = false;
        if (cfg.n_active < cfg.n_experts) {
            active_params =
                (int64_t)cfg.vocab_size * cfg.d_model +
                (int64_t)cfg.n_layers * (
                    4LL * cfg.d_model * cfg.n_heads * cfg.head_dim +
                    (int64_t)cfg.n_active * 3LL * cfg.d_model * cfg.d_ff +
                    2LL * cfg.d_model * cfg.n_experts +
                    2LL * cfg.d_model
                ) +
                cfg.d_model;
            has_active = true;
        } else {
            active_params = total_params;  /* unused unless has_active=true */
        }

        /* D-191: train_config_micro() is the source of truth for adam_beta1
         * (0.9), adam_beta2 (0.999), weight_decay (0.01), lr_min (1e-5), and
         * batch_size (1). Those constants are stored in TrainConfig as float;
         * promoting `defaults.adam_beta1` (float) to double would inject 2.4e-8
         * representation noise (0.9f → 0.89999997615814209) that exceeds the
         * comparator's 1e-9 tolerance and would FATAL on a perfectly-correct
         * launch. To avoid that artifact, we use the double-precision INTENT
         * literals here, matching the values pinned in the manifest exactly.
         * If a future edit changes train_config_micro()'s defaults, BOTH this
         * block AND the manifest must update together — the comparator will
         * scream if they disagree, which is the design intent. */
        TrainConfig defaults = train_config_micro();
        (void)defaults;  /* batch_size still pulled from defaults below */

        ConfigDriftRuntime rt = {
            .model              = args.model_size,
            .tokenizer_path     = args.tokenizer_path,
            .vocab_size         = cfg.vocab_size,
            .param_count_total  = total_params,
            .param_count_active = active_params,
            .has_param_count_active = has_active,
            /* D-191 / P-CONFIG-DRIFT-LR-WIRE: 14 LR/schedule/dynamics fields */
            .lr_base            = (double)scale_compute_lr_base(&args),
            .lr_schedule        = "linear_warmup_cosine_decay",
            .lr_warmup_steps    = SCALE_LR_WARMUP_STEPS_DEFAULT,
            /* Double-precision intent literals — see comment above on
             * float-truncation artifact. These mirror train_config_micro()
             * defaults exactly. If train_config_micro() changes, update here
             * + spec_invariants.yaml together. */
            .lr_min             = 1e-5,
            .optimizer          = "adamw",
            .adam_beta1         = 0.9,
            .adam_beta2         = 0.999,
            .weight_decay       = 0.01,
            .grad_clip_norm     = 1.0,
            .grad_accum_steps   = args.grad_accum,
            .batch_size         = defaults.batch_size,
            .seq_len            = seq_len,
            .max_steps          = args.max_steps,
            .weight_seed        = (int32_t)args.weight_seed,
        };
        config_drift_assert_or_die(&rt);
    }

    /* Estimate memory */
    double model_mb = (double)total_params * 4.0 / (1024.0 * 1024.0);
    double adam_mb = model_mb * 2.0;  /* m and v moments */
    double grad_mb = model_mb;
    double state_mb = (double)(cfg.n_layers + 1) * seq_len * cfg.d_model * 4.0 / (1024.0 * 1024.0) * 3.0;
    double data_mb = (double)n_tokens * 4.0 / (1024.0 * 1024.0);
    printf("  Memory est: model=%.1fMB adam=%.1fMB grad=%.1fMB state=%.1fMB data=%.1fMB total=%.1fMB\n",
           model_mb, adam_mb, grad_mb, state_mb, data_mb,
           model_mb + adam_mb + grad_mb + state_mb + data_mb);

    HSPAModel *model = hspa_model_create(&cfg);
    if (!model) {
        fprintf(stderr, "FAILED: hspa_model_create returned NULL\n");
        free(all_tokens);
        if (stream_tok) tokenizer_destroy(stream_tok);
        return 1;
    }
    printf("  Model created.\n");

    /* ---- Step 3: Initialize weights ---- */
    printf("\n[3/5] Initializing weights (Depth-muP)...\n");
    weight_init_depth_mup(model, &cfg);
    printf("  Weights initialized.\n");

    /* ---- Metal GPU acceleration ---- */
    if (metal_init() == 0) {
        op_set_gpu_matmul(metal_matmul);
        printf("  Metal GPU acceleration: ENABLED\n");
    } else {
        printf("  Metal GPU acceleration: DISABLED (init failed)\n");
    }

    /* ---- Step 4: Create training state ---- */
    printf("\n[4/5] Creating training infrastructure...\n");

    TrainConfig tcfg = train_config_micro();
    tcfg.max_steps = args.max_steps;
    tcfg.log_every = 20;
    tcfg.use_adam = true;
    tcfg.ipc_lr = 0.05f;

    /* Scale LR for model size: smaller D needs higher base LR with muP.
     * Centralized in scale_compute_lr_base() (D-191) so the config-drift
     * assertion at init can compare against the SAME live value training will
     * use. The helper takes --lr override first, then keys off args.model_size:
     * default 0.002f; 0.001f for "medium" (the historical 2× MoE LR pinned in
     * Phase-2 manifest D-181). */
    tcfg.base_lr = scale_compute_lr_base(&args);
    tcfg.grad_clip_norm = SCALE_GRAD_CLIP_NORM_DEFAULT;
    tcfg.lr_warmup_steps = SCALE_LR_WARMUP_STEPS_DEFAULT;
    /* Balance loss with small beta: provides upward entropy force that
     * activates as entropy drops. At high entropy, pred gradient dominates
     * (drives specialization). At low entropy, beta_balance dominates
     * (prevents collapse). Combined with quadratic entropy-adaptive
     * scaling on pred gradient, creates stable equilibrium.
     * History: 0.0 (Cycle 10-12) → collapse at step 1040-1200.
     *          0.01 (Cycle 12 fix) → equilibrium target. */
    tcfg.beta_balance = (args.beta_balance > 0.0f) ? args.beta_balance : 0.01f;
    tcfg.balance_h_floor = (args.balance_h_floor > 0.0f) ? args.balance_h_floor : 0.0f;
    tcfg.balance_h_target = (args.balance_h_target > 0.0f) ? args.balance_h_target : 0.0f;
    tcfg.router_pred_scale = (args.router_pred_scale > 0.0f) ? args.router_pred_scale : 0.05f;
    tcfg.grad_accum_steps = args.grad_accum;
    tcfg.mup_base_width = 128;  /* micro-experiment reference width */

    /* D-031: Loss-Free Balancing — disable entropy gradient, use router bias only. */
    tcfg.use_loss_free_balance = args.loss_free_balance;
    tcfg.lfb_bias_step = (args.bias_step_size > 0.0f) ? args.bias_step_size : 0.0f;
    tcfg.lfb_ema_rate = (args.bias_update_rate > 0.0f) ? args.bias_update_rate : 0.0f;

    /* D-Plan-B: Default MoE / Dense Backprop. Defaults ON under --backprop to
     * close the gradient-starvation failure mode documented in
     * killed_ideas.md D-094; defaults OFF in iPC mode (which has its own
     * entropy gradient). User can force either way via --default-moe /
     * --no-default-moe. Paper alpha=0.01, sigma_init=0.01. */
    {
        bool auto_on = args.use_backprop;
        bool default_moe_on;
        if (args.default_moe_state > 0)      default_moe_on = true;
        else if (args.default_moe_state < 0) default_moe_on = false;
        else                                  default_moe_on = auto_on;
        tcfg.use_default_moe = default_moe_on;
        tcfg.default_moe_alpha = (args.default_moe_alpha > 0.0f)
                                     ? args.default_moe_alpha : 0.01f;
        tcfg.default_moe_sigma_init = (args.default_moe_sigma_init > 0.0f)
                                          ? args.default_moe_sigma_init : 0.01f;
    }

    /* --- Cycle 29 Rev-2: Entropy penalty + cosine τ-anneal wiring ---
     * Plumbs ScaleArgs into TrainConfig. Defaults match Rev-2 spec:
     *   β_H=0.15, H_target=0.90, warmup=0, β_H-ramp=200, τ_max=1.4,
     *   τ_min=1.0, S_anneal=500, restoration=ON. */
    {
        /* Entropy penalty master switch. */
        if (args.entropy_penalty_state > 0)      tcfg.use_entropy_penalty = true;
        else if (args.entropy_penalty_state < 0) tcfg.use_entropy_penalty = false;
        else                                      tcfg.use_entropy_penalty = false;

        if (args.entropy_penalty_beta > 0.0f)
            tcfg.entropy_beta_h = args.entropy_penalty_beta;
        if (args.entropy_penalty_target > 0.0f)
            tcfg.entropy_h_target = args.entropy_penalty_target;
        if (args.entropy_warmup_steps >= 0)
            tcfg.entropy_warmup_steps = args.entropy_warmup_steps;
        if (args.beta_h_warmup_steps >= 0)
            tcfg.entropy_beta_h_warmup_steps = args.beta_h_warmup_steps;

        /* τ-anneal master switch. */
        if (args.temp_anneal_state > 0)      tcfg.use_temp_anneal = true;
        else if (args.temp_anneal_state < 0) tcfg.use_temp_anneal = false;
        else                                   tcfg.use_temp_anneal = false;

        if (args.tau_max > 0.0f) tcfg.temp_anneal_max = args.tau_max;
        if (args.tau_min > 0.0f) tcfg.temp_anneal_min = args.tau_min;
        if (args.tau_anneal_steps > 0)
            tcfg.temp_anneal_steps = args.tau_anneal_steps;

        /* Restoration band: Rev-2 default ON; user can force either way. */
        if (args.temp_anneal_restoration_state > 0)
            tcfg.use_temp_anneal_restoration = true;
        else if (args.temp_anneal_restoration_state < 0)
            tcfg.use_temp_anneal_restoration = false;
        /* else leave tcfg default (true from train_config_micro). */
    }

    /* Program 3 Phase 7: QAT wiring.
     * --qat enables fake-INT4 quantization for the 224 in-scope weights.
     * --qat-group-size N sets the INT4 group size (default 128).
     * These fields are threaded to qat_context_create in all three arm step
     * functions; behavior at group_size=128 is bit-identical to prior hardcoded 128. */
    tcfg.use_qat        = args.use_qat;
    tcfg.qat_group_size = (args.qat_group_size > 0) ? args.qat_group_size : 128;

    /* Progressive sparsification: start with more active experts, anneal down.
     * Gives experts time to develop distinct representations before competition
     * narrows. Validated by Sigma-MoE-Tiny (arXiv 2512.16248). */
    tcfg.n_active_start = args.n_active_start;
    tcfg.sparsify_steps = args.sparsify_steps;
    if (tcfg.n_active_start > 0 && tcfg.n_active_start <= cfg.n_active) {
        printf("  WARNING: n_active_start (%d) <= n_active (%d), disabling sparsification\n",
               tcfg.n_active_start, cfg.n_active);
        tcfg.n_active_start = 0;
    }
    if (tcfg.n_active_start > cfg.n_experts) {
        printf("  WARNING: n_active_start (%d) > n_experts (%d), clamping\n",
               tcfg.n_active_start, cfg.n_experts);
        tcfg.n_active_start = cfg.n_experts;
    }

    printf("  Optimizer: AdamW (beta1=%.2f, beta2=%.3f, wd=%.3f)\n",
           tcfg.adam_beta1, tcfg.adam_beta2, tcfg.weight_decay);
    printf("  LR: %.4f (warmup=%d, cosine decay to %.1e over %d steps)\n",
           tcfg.base_lr, tcfg.lr_warmup_steps, tcfg.lr_min, tcfg.max_steps);
    printf("  Grad accum: %d micro-batches (effective batch=%d)\n",
           tcfg.grad_accum_steps, tcfg.grad_accum_steps);
    printf("  muP wd scaling: D=%d / D_base=%d = %.2fx\n",
           cfg.d_model, tcfg.mup_base_width,
           (float)cfg.d_model / (float)tcfg.mup_base_width);
    if (args.use_backprop) {
        printf("  Training: BACKPROP (forward + backward, no iPC iterations)\n");
    } else {
        printf("  iPC: T=%d, ipc_lr=%.3f, beta_balance=%.3f, h_floor=%.3f, h_target=%.3f, router_pred_scale=%.3f\n",
               tcfg.T, tcfg.ipc_lr, tcfg.beta_balance, tcfg.balance_h_floor,
               tcfg.balance_h_target, tcfg.router_pred_scale);
    }
    if (tcfg.balance_h_target > 0.0f && tcfg.balance_h_floor > 0.0f &&
        tcfg.balance_h_target < tcfg.balance_h_floor) {
        printf("  WARNING: h_target (%.3f) < h_floor (%.3f) -- no free zone, "
               "controller may oscillate\n",
               tcfg.balance_h_target, tcfg.balance_h_floor);
    }
    if (tcfg.n_active_start > 0) {
        printf("  Progressive sparsification: top-%d → top-%d over %d steps\n",
               tcfg.n_active_start, cfg.n_active, tcfg.sparsify_steps);
    }

    /* D-031: Loss-Free Balancing — apply router bias parameters */
    if (tcfg.use_loss_free_balance) {
        printf("  Loss-Free Balancing: ENABLED (entropy gradient disabled)\n");
        for (int32_t l = 0; l < cfg.n_layers; l++) {
            FEPRouter *r = model->layers[l]->router;
            if (tcfg.lfb_bias_step > 0.0f) {
                r->bias_step_size = tcfg.lfb_bias_step;
            }
            if (tcfg.lfb_ema_rate > 0.0f) {
                r->bias_update_rate = tcfg.lfb_ema_rate;
            }
        }
        printf("  Router bias: step_size=%.4f, ema_rate=%.3f\n",
               model->layers[0]->router->bias_step_size,
               model->layers[0]->router->bias_update_rate);
    }

    /* D-Plan-B: Default MoE / Dense Backprop -- allocate per-layer EMA
     * tensors. Initialize from N(0, sigma_init) with a layer-specific
     * seed so retention vs reproducibility are balanced. Checkpoint
     * resume will overwrite these tensors (see checkpoint.c V5 read).
     * Skipped in iPC mode unless user forced --default-moe. */
    if (tcfg.use_default_moe) {
        printf("  Default MoE (Dense Backprop): ENABLED "
               "(alpha=%.4f, sigma_init=%.4f)\n",
               tcfg.default_moe_alpha, tcfg.default_moe_sigma_init);
        for (int32_t l = 0; l < cfg.n_layers; l++) {
            FEPRouter *r = model->layers[l]->router;
            uint32_t seed = (uint32_t)(0xBEE57EAu + (uint32_t)l * 0x9E3779B9u);
            int rc = router_init_default_moe(r, cfg.d_model,
                                             tcfg.default_moe_alpha,
                                             tcfg.default_moe_sigma_init,
                                             seed);
            if (rc != 0) {
                fprintf(stderr,
                    "FATAL: router_init_default_moe failed at layer %d\n", l);
                hspa_model_destroy(model);
                free(all_tokens);
                return 1;
            }
        }
        size_t ema_bytes = (size_t)cfg.n_layers * (size_t)cfg.n_experts
                         * (size_t)cfg.d_model * sizeof(float);
        printf("  Default MoE EMA: %zu bytes (%.3f MB)\n",
               ema_bytes, (double)ema_bytes / (1024.0 * 1024.0));
    }

    /* --- Cycle 29 Rev-2 banner: entropy penalty + cosine τ-anneal ---
     * Prints a single grep-able line summarising the stacked config so
     * evaluators can confirm the experiment's hyper-parameters without
     * re-reading the training log. */
    if (tcfg.use_entropy_penalty || tcfg.use_temp_anneal) {
        printf("Entropy penalty + τ-anneal: ENABLED "
               "β_H=%.2f H_target=%.2f τ: %.2f→%.2f over %d steps, "
               "β_H warmup %d steps%s%s\n",
               tcfg.use_entropy_penalty ? tcfg.entropy_beta_h : 0.0f,
               tcfg.use_entropy_penalty ? tcfg.entropy_h_target : 0.0f,
               tcfg.use_temp_anneal ? tcfg.temp_anneal_max : 1.0f,
               tcfg.use_temp_anneal ? tcfg.temp_anneal_min : 1.0f,
               tcfg.use_temp_anneal ? tcfg.temp_anneal_steps : 0,
               tcfg.use_entropy_penalty ? tcfg.entropy_beta_h_warmup_steps : 0,
               tcfg.use_temp_anneal_restoration ? " restoration=ON" : "",
               tcfg.entropy_warmup_steps > 0 ? " (penalty-warmup>0)" : "");
    }

    /* Allocate iPC state (only needed for true iPC mode, not backprop or localfb).
     * If n_active_start > n_active, arrays must accommodate the larger k. */
    IPCTrainState *state = NULL;
    if (!args.use_backprop && !args.use_localfb) {
        HSPAConfig state_cfg = cfg;
        if (tcfg.n_active_start > cfg.n_active) {
            state_cfg.n_active = tcfg.n_active_start;
        }
        state = ipc_state_create(&state_cfg, seq_len,
                                 tcfg.sigma_bottom, tcfg.sigma_top);
        if (!state) {
            fprintf(stderr, "FAILED: ipc_state_create\n");
            hspa_model_destroy(model);
            free(all_tokens);
            return 1;
        }
    }

    /* Allocate LocalFB state (Arm C: Direct Feedback Alignment, Nøkland 2016).
     * Only created when --localfb is passed; NULL otherwise. */
    LocalFBState *localfb_state = NULL;
    if (args.use_localfb) {
        localfb_state = localfb_state_create(&cfg,
                                              args.weight_seed ? args.weight_seed : 1u);
        if (!localfb_state) {
            fprintf(stderr, "FAILED: localfb_state_create\n");
            hspa_model_destroy(model);
            free(all_tokens);
            return 1;
        }
    }

    ModelGrad *grads = grad_create(&cfg);
    if (!grads) {
        fprintf(stderr, "FAILED: grad_create\n");
        ipc_state_destroy(state);
        hspa_model_destroy(model);
        free(all_tokens);
        return 1;
    }

    AdamState *adam = adam_create(&cfg);
    if (!adam) {
        fprintf(stderr, "FAILED: adam_create\n");
        grad_destroy(grads);
        ipc_state_destroy(state);
        hspa_model_destroy(model);
        free(all_tokens);
        return 1;
    }

    /* D-091 Phase B: in stream mode we use a StreamDataLoader; legacy mode
     * uses the in-memory DataLoader. Exactly one is non-NULL at a time. */
    DataLoader       *dl  = NULL;
    StreamDataLoader *sdl = NULL;

    if (args.use_stream) {
        StreamSourceSpec srcs[STREAM_N_SOURCES];
        for (int i = 0; i < STREAM_N_SOURCES; i++) {
            srcs[i].path      = stream_paths[i];
            srcs[i].tokenizer = stream_tok;
            srcs[i].weight    = stream_weights[i];
        }
        sdl = stream_loader_create(srcs, STREAM_N_SOURCES, seq_len,
                                   args.stream_seed);
        if (!sdl) {
            fprintf(stderr, "FAILED: stream_loader_create\n");
            adam_destroy(adam);
            grad_destroy(grads);
            ipc_state_destroy(state);
            hspa_model_destroy(model);
            tokenizer_destroy(stream_tok);
            return 1;
        }
        printf("  StreamDataLoader: %d sources, seed=0x%llx\n",
               STREAM_N_SOURCES, (unsigned long long)args.stream_seed);
    } else {
        dl = data_loader_create(all_tokens, n_tokens, seq_len);
        if (!dl) {
            fprintf(stderr, "FAILED: data_loader_create\n");
            adam_destroy(adam);
            grad_destroy(grads);
            ipc_state_destroy(state);
            hspa_model_destroy(model);
            free(all_tokens);
            return 1;
        }
    }

    /* Resume from checkpoint if requested */
    int32_t start_step = 0;
    Metrics m;
    metrics_init(&m);

    if (args.resume_path) {
        CheckpointMeta   ckpt_meta;
        StreamLoaderState ckpt_stream = {0};
        if (checkpoint_load_ex(args.resume_path, model, adam, &cfg, NULL,
                               &ckpt_meta, &ckpt_stream)) {
            start_step = ckpt_meta.step;
            m.best_ppl = ckpt_meta.best_ppl;
            m.best_loss = ckpt_meta.best_loss;
            m.epochs_completed = ckpt_meta.epoch;
            m.tokens_trained = ckpt_meta.tokens_trained;
            if (args.use_stream) {
                /* Restore deterministic loader position if the checkpoint
                 * carries a valid StreamLoaderState (n_sources matches). */
                if (ckpt_stream.n_sources == STREAM_N_SOURCES) {
                    if (stream_loader_restore_state(sdl, &ckpt_stream) == 0) {
                        printf("  Stream state restored: byte_offsets=");
                        for (int i = 0; i < STREAM_N_SOURCES; i++) {
                            printf("%lld%s",
                                   (long long)ckpt_stream.byte_offsets[i],
                                   i + 1 < STREAM_N_SOURCES ? "," : "\n");
                        }
                    } else {
                        fprintf(stderr, "WARNING: stream_loader_restore_state failed\n");
                    }
                } else {
                    printf("  [stream] checkpoint has no stream state -- starting loader fresh\n");
                }
            } else {
                /* Legacy loader: seek analytically. */
                int64_t dl_total = data_loader_total(dl);
                int64_t pos_in_data = ckpt_meta.tokens_trained % dl_total;
                data_loader_seek(dl, pos_in_data);
            }
            printf("  Resumed from checkpoint: step=%d, best_ppl=%.2f\n",
                   start_step, m.best_ppl);
        } else {
            fprintf(stderr, "WARNING: could not load checkpoint %s, starting fresh\n",
                    args.resume_path);
        }
    }

    /* Create checkpoint directory */
    mkdir(args.checkpoint_dir, 0755);

    printf("  All training state created.\n");

    /* ---- Step 5: Training loop ---- */
    int32_t *tokens = (int32_t *)calloc((size_t)seq_len, sizeof(int32_t));
    int32_t *targets = (int32_t *)calloc((size_t)seq_len, sizeof(int32_t));
    if (!tokens || !targets) {
        fprintf(stderr, "FAILED: token buffer allocation\n");
        free(tokens); free(targets);
        if (dl)  data_loader_destroy(dl);
        if (sdl) stream_loader_destroy(sdl);
        adam_destroy(adam);
        grad_destroy(grads);
        ipc_state_destroy(state);
        hspa_model_destroy(model);
        free(all_tokens);
        if (stream_tok) tokenizer_destroy(stream_tok);
        return 1;
    }

    if (!args.resume_path) {
        metrics_init(&m);
    }

    if (args.use_stream) {
        printf("\n  Data: streaming (5 sources, infinite wraparound)\n");
    } else {
        int64_t steps_per_epoch = (n_tokens - 1) / seq_len;  /* approximate */
        printf("\n  Data: %lld tokens, seq_len=%d, ~%lld steps/epoch\n",
               (long long)n_tokens, seq_len, (long long)steps_per_epoch);
    }
    printf("  Random baseline: ln(%d) = %.4f (PPL = %d)\n\n",
           cfg.vocab_size, logf((float)cfg.vocab_size), cfg.vocab_size);

    printf("\n[5/5] Training...\n\n");
    print_header();

    struct timespec ts_start, ts_end;
    int32_t actual_steps = start_step;

    int32_t ga_steps = tcfg.grad_accum_steps > 0 ? tcfg.grad_accum_steps : 1;

    /* --- Cycle 29 Rev-2: τ-anneal + restoration-band state ---
     * freeze_pause_until: if non-zero, hold τ at its current value until
     * the global step reaches this threshold. Triggered when mean_entropy
     * drops below 0.55 during the cosine anneal window (§R2.5).
     * last_tau: cached value from the previous step, used to pin τ during
     * the 100-step freeze window. */
    int32_t tau_freeze_until = 0;
    float   tau_frozen_value = tcfg.temp_anneal_max;
    const float kRestoreEntropyThreshold = 0.55f;
    const int32_t kRestoreFreezeSteps    = 100;

    for (int32_t step = start_step; step < args.max_steps; step++) {
        clock_gettime(CLOCK_MONOTONIC, &ts_start);

        TrainStepResult r;
        memset(&r, 0, sizeof(r));

        /* Progressive sparsification: compute effective n_active for this step. */
        HSPAConfig step_cfg = cfg;
        int32_t effective_k = compute_effective_k(step, tcfg.n_active_start,
                                                  cfg.n_active, tcfg.sparsify_steps);
        if (effective_k != cfg.n_active) {
            step_cfg.n_active = effective_k;
        }

        /* --- Cycle 29 Rev-2: set per-layer router temperature for this step ---
         * τ(step) = τ_min + (τ_max−τ_min)·0.5·(1 + cos(π·min(step,S)/S))
         * Restoration band: if the previous log step saw mean_entropy <
         * threshold, freeze τ at its current value for kRestoreFreezeSteps
         * to let entropy recover before resuming the cosine decay. */
        float tau_now = 1.0f;
        if (tcfg.use_temp_anneal) {
            if (tcfg.use_temp_anneal_restoration && step < tau_freeze_until) {
                tau_now = tau_frozen_value;
            } else {
                tau_now = compute_router_temperature(tcfg.temp_anneal_max,
                                                     tcfg.temp_anneal_min,
                                                     step,
                                                     tcfg.temp_anneal_steps);
                tau_frozen_value = tau_now;
            }
            for (int32_t l = 0; l < cfg.n_layers; l++) {
                model->layers[l]->router->router_temperature = tau_now;
            }
        } else {
            /* Always pin at 1.0 when anneal disabled to ensure backprop
             * path sees a deterministic divisor regardless of prior runs. */
            for (int32_t l = 0; l < cfg.n_layers; l++) {
                model->layers[l]->router->router_temperature = 1.0f;
            }
        }

        /* Gradient accumulation loop over micro-batches */
        for (int32_t mb = 0; mb < ga_steps; mb++) {
            /* Get next batch from data loader */
            if (args.use_stream) {
                if (stream_loader_next(sdl, tokens, targets) < 0) {
                    fprintf(stderr, "ERROR: stream_loader_next failed\n");
                    break;
                }
                /* Streams auto-wrap, no epoch boundary to report. */
            } else if (data_loader_next(dl, tokens, targets) < 0) {
                /* Epoch boundary */
                data_loader_reset(dl);
                m.epochs_completed++;
                if (data_loader_next(dl, tokens, targets) < 0) {
                    fprintf(stderr, "ERROR: data_loader_next failed after reset\n");
                    break;
                }
                if (step > 0 && step % tcfg.log_every != 0 && mb == 0) {
                    printf("  --- Epoch %d boundary at step %d ---\n",
                           m.epochs_completed, step);
                }
            }

            if (args.use_backprop) {
                r = backprop_train_step(model, grads, adam,
                                        tokens, targets, seq_len,
                                        &step_cfg, &tcfg, step, mb);
            } else if (args.use_localfb) {
                /* Arm C: LocalFB / Direct Feedback Alignment (Nøkland 2016). */
                r = localfb_train_step(model, grads, adam, localfb_state,
                                       tokens, targets, seq_len,
                                       &step_cfg, &tcfg, step, mb);
            } else {
                r = ipc_train_step(model, grads, adam, state,
                                   tokens, targets, seq_len,
                                   &step_cfg, &tcfg, step, mb);
            }
        }

        /* Update router biases (auxiliary-loss-free expert balancing).
         * For iPC: count expert selections from state->routing.
         * For backprop: bias update is handled inside backprop_train_step.
         * For localfb: state is NULL (no IPCTrainState); skip this block. */
        if (!args.use_backprop && !args.use_localfb) {
            int32_t K = cfg.n_experts;
            int32_t k = effective_k;
            int32_t *expert_counts = (int32_t *)calloc((size_t)K, sizeof(int32_t));
            if (expert_counts) {
                for (int32_t l = 0; l < cfg.n_layers; l++) {
                    memset(expert_counts, 0, (size_t)K * sizeof(int32_t));
                    RoutingState *rs = &state->routing[l];
                    for (int32_t s = 0; s < rs->seq_len; s++) {
                        for (int32_t a = 0; a < k; a++) {
                            int32_t eid = rs->expert_ids[s * k + a];
                            if (eid >= 0 && eid < K) {
                                expert_counts[eid]++;
                            }
                        }
                    }
                    int32_t total_decisions = rs->seq_len * k;
                    router_update_bias(model->layers[l]->router,
                                       expert_counts, total_decisions, K);
                }
                free(expert_counts);
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_end);
        double elapsed_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1000.0
                          + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;

        /* Track metrics */
        actual_steps = step + 1;
        m.tokens_trained += (int64_t)seq_len * ga_steps;

        if (step == 0) {
            m.initial_ppl = r.loss.perplexity;
        }

        if (isnan(r.loss.total) || isinf(r.loss.total)) {
            m.nan_count++;
            m.total_nan++;
            if (m.nan_count > 10) {
                printf("\n*** ABORT: %d consecutive NaN/Inf. Training diverged. ***\n",
                       m.nan_count);
                break;
            }
        } else {
            m.nan_count = 0;
            if (r.loss.total < m.best_loss) m.best_loss = r.loss.total;
            if (r.loss.perplexity < m.best_ppl) m.best_ppl = r.loss.perplexity;
        }

        m.loss_window[m.window_idx] = r.loss.lm;
        m.window_idx = (m.window_idx + 1) % 50;

        /* Print progress */
        if (step < 5 || step % tcfg.log_every == 0 || step == args.max_steps - 1) {
            print_step(step, &r, elapsed_ms, m.epochs_completed);
            /* D-095 monitoring: dump per-(layer, expert) load EMA so the
             * max/min ratio kill-signal can be evaluated from log alone.
             * Reads ONLY router->expert_load_ema (already maintained); no
             * new state, no checkpoint impact, O(L*K) work per logged step. */
            float worst_ratio = 0.0f;
            int32_t worst_layer = -1;
            for (int32_t l = 0; l < cfg.n_layers; l++) {
                const FEPRouter *r2 = model->layers[l]->router;
                if (!r2 || !r2->expert_load_ema) continue;
                float mn = r2->expert_load_ema[0], mx = mn;
                printf("[load] step=%d layer=%d K=%d ema=[", step, l, cfg.n_experts);
                for (int32_t e = 0; e < cfg.n_experts; e++) {
                    float v = r2->expert_load_ema[e];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                    printf("%s%.5f", e ? "," : "", v);
                }
                float ratio = (mn > 1e-9f) ? (mx / mn) : 1e9f;
                printf("] min=%.5f max=%.5f ratio=%.2f\n", mn, mx, ratio);
                if (ratio > worst_ratio) { worst_ratio = ratio; worst_layer = l; }
            }
            int gate_hit = (worst_ratio > 5.0f) && (step >= 500);
            int kill_hit = (worst_ratio > 10.0f) && (step >= 500);
            printf("[load] step=%d SUMMARY worst_ratio=%.2f worst_layer=%d "
                   "gate_hit=%d kill_hit=%d\n",
                   step, worst_ratio, worst_layer, gate_hit, kill_hit);

            /* --- Cycle 29 Rev-2 entropy-penalty / τ diagnostic line --- */
            if (tcfg.use_entropy_penalty || tcfg.use_temp_anneal) {
                float beta_eff = tcfg.use_entropy_penalty
                    ? compute_beta_h_warmup(tcfg.entropy_beta_h, step,
                                            tcfg.entropy_beta_h_warmup_steps)
                    : 0.0f;
                float gap = tcfg.entropy_h_target - r.mean_entropy;
                if (gap < 0.0f) gap = 0.0f;
                float penalty_val = beta_eff * gap * gap;
                printf("[entpen] step=%d tau=%.4f beta_H_eff=%.4f H=%.4f "
                       "H_target=%.4f entropy-penalty-loss=%.6f freeze_until=%d\n",
                       step, tau_now, beta_eff, r.mean_entropy,
                       tcfg.entropy_h_target, penalty_val, tau_freeze_until);
            }
        }

        /* --- Rev-2 §R2.5 restoration band trigger ---
         * If mean_entropy dips below threshold during the anneal window,
         * pause τ decay for kRestoreFreezeSteps. Evaluated every step so
         * the freeze engages promptly even without logging. */
        if (tcfg.use_temp_anneal && tcfg.use_temp_anneal_restoration
            && step < tcfg.temp_anneal_steps
            && !isnan(r.mean_entropy)
            && r.mean_entropy < kRestoreEntropyThreshold
            && step >= tau_freeze_until) {
            tau_freeze_until = step + kRestoreFreezeSteps;
            printf("  [restore] H=%.4f < %.3f at step %d -> freezing τ=%.4f "
                   "until step %d\n",
                   r.mean_entropy, kRestoreEntropyThreshold, step,
                   tau_frozen_value, tau_freeze_until);
        }

        /* Log sparsification transitions */
        if (tcfg.n_active_start > cfg.n_active && tcfg.sparsify_steps > 0) {
            int32_t next_ek = compute_effective_k(step + 1, tcfg.n_active_start,
                                                  cfg.n_active, tcfg.sparsify_steps);
            if (next_ek != effective_k) {
                printf("  --- Sparsification: top-%d → top-%d at step %d ---\n",
                       effective_k, next_ek, step + 1);
            }
        }

        /* Periodic summary */
        if (step > 0 && step % 500 == 0) {
            float avg = metrics_avg_loss(&m, step < 50 ? step : 50);
            printf("  >>> Step %d summary: avg_lm=%.4f, best_ppl=%.2f, "
                   "epochs=%d, tokens=%lldK, NaN=%d <<<\n",
                   step, avg, m.best_ppl, m.epochs_completed,
                   (long long)(m.tokens_trained / 1000), m.total_nan);
        }

        /* Periodic checkpoint save */
        if (args.checkpoint_every > 0 && step > 0
            && step % args.checkpoint_every == 0) {
            char ckpt_path[512];
            snprintf(ckpt_path, sizeof(ckpt_path), "%s/step_%06d.ckpt",
                     args.checkpoint_dir, step);
            CheckpointMeta ckpt_meta = {
                .step = step,
                .epoch = m.epochs_completed,
                .tokens_trained = m.tokens_trained,
                .best_ppl = m.best_ppl,
                .best_loss = m.best_loss,
            };
            StreamLoaderState stream_snapshot;
            const StreamLoaderState *stream_ptr = NULL;
            if (args.use_stream && sdl) {
                stream_loader_save_state(sdl, &stream_snapshot);
                stream_ptr = &stream_snapshot;
            }
            if (checkpoint_save_ex(ckpt_path, model, adam, &cfg, &tcfg,
                                   &ckpt_meta, stream_ptr)) {
                printf("  [CKPT] Saved %s\n", ckpt_path);
            } else {
                printf("  [CKPT] WARNING: failed to save %s\n", ckpt_path);
            }
        }
    }

    /* Save final checkpoint */
    {
        char ckpt_path[512];
        snprintf(ckpt_path, sizeof(ckpt_path), "%s/final.ckpt",
                 args.checkpoint_dir);
        CheckpointMeta ckpt_meta = {
            .step = actual_steps,
            .epoch = m.epochs_completed,
            .tokens_trained = m.tokens_trained,
            .best_ppl = m.best_ppl,
            .best_loss = m.best_loss,
        };
        StreamLoaderState stream_snapshot;
        const StreamLoaderState *stream_ptr = NULL;
        if (args.use_stream && sdl) {
            stream_loader_save_state(sdl, &stream_snapshot);
            stream_ptr = &stream_snapshot;
        }
        if (checkpoint_save_ex(ckpt_path, model, adam, &cfg, &tcfg, &ckpt_meta,
                               stream_ptr)) {
            printf("  [CKPT] Final saved: %s\n", ckpt_path);
        }
    }

    /* ---- Results ---- */
    printf("\n--- Training Complete ---\n\n");

    float avg_recent = metrics_avg_loss(&m, actual_steps < 50 ? actual_steps : 50);
    float random_baseline = logf((float)cfg.vocab_size);

    printf("Summary:\n");
    printf("  Model:          %s (~%.2fM params)\n", args.model_size,
           (double)total_params / 1e6);
    printf("  Steps trained:  %d / %d\n", actual_steps, args.max_steps);
    printf("  Epochs:         %d\n", m.epochs_completed);
    printf("  Tokens trained: %lldK\n", (long long)(m.tokens_trained / 1000));
    printf("  Initial PPL:    %.2f\n", m.initial_ppl);
    printf("  Best PPL:       %.2f\n", m.best_ppl);
    printf("  Best loss:      %.4f\n", m.best_loss);
    printf("  Recent avg LM:  %.4f\n", avg_recent);
    printf("  Random baseline:%.4f (PPL=%d)\n", random_baseline, cfg.vocab_size);
    printf("  NaN events:     %d (total)\n", m.total_nan);

    if (args.use_stream && sdl) {
        printf("  Stream diagnostics:\n");
        int64_t total_bytes = 0;
        for (int i = 0; i < STREAM_N_SOURCES; i++) {
            int64_t b = stream_loader_source_bytes_read(sdl, i);
            int64_t e = stream_loader_source_epochs(sdl, i);
            total_bytes += b;
            printf("    [%s] bytes=%lld  epochs=%lld\n", stream_names[i],
                   (long long)b, (long long)e);
        }
        printf("    [TOT] bytes=%lld (%.2f MB)\n", (long long)total_bytes,
               (double)total_bytes / (1024.0 * 1024.0));
    }

    /* Verdict */
    printf("\nVerdict: ");
    if (m.nan_count > 10) {
        printf("FAILED -- training diverged (NaN/Inf)\n");
    } else if (avg_recent < random_baseline * 0.8f) {
        printf("CONVERGING -- LM loss well below random baseline (%.1f%% of random)\n",
               100.0f * avg_recent / random_baseline);
    } else if (avg_recent < random_baseline) {
        printf("LEARNING -- LM loss below random baseline but not strongly converging\n");
    } else if (m.best_loss < random_baseline) {
        printf("PARTIAL -- some loss reduction seen but recent loss near baseline\n");
    } else {
        printf("INCONCLUSIVE -- loss did not decrease below random baseline\n");
    }

    /* Cleanup Metal */
    op_set_gpu_matmul(NULL);
    metal_cleanup();

    /* Cleanup */
    free(tokens);
    free(targets);
    if (dl)  data_loader_destroy(dl);
    if (sdl) stream_loader_destroy(sdl);
    adam_destroy(adam);
    grad_destroy(grads);
    ipc_state_destroy(state);
    localfb_state_destroy(localfb_state);
    hspa_model_destroy(model);
    free(all_tokens);
    if (stream_tok) tokenizer_destroy(stream_tok);

    printf("\n=== Scale-Up Experiment Complete ===\n");
    return (m.nan_count > 10) ? 1 : 0;
}
