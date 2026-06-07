/* config_drift.h -- Per-program runtime config-drift detection (P-CONFIG-DRIFT-DETECTION,
 * D-121, Item 6 of P-PHASE2-REDO).
 *
 * Compares actual runtime values (model name, tokenizer file, vocab_size, param counts)
 * against a frozen per-program manifest. Prevents the D-119–D-121 Swiss-cheese class of
 * failure: launch template silently relies on a binary-default tokenizer, training runs
 * on the wrong vocab, ~24h wall is lost.
 *
 * Activation (opt-in via env var):
 *   LAB_PROGRAM=<name>            → load programs/<name>/spec_invariants.yaml and assert
 *   LAB_PROGRAM unset             → assertion is a no-op (legacy / one-off invocations)
 *   LAB_PROGRAM_ROOT=<dir>        → override "programs" root (used by tests; production
 *                                    callers leave it unset to read from ./programs/)
 *
 * Failure mode: on any mismatch (or missing manifest with LAB_PROGRAM set), the assert
 * function prints "CONFIG-DRIFT FATAL: expected X, got Y" to stderr for each offending
 * field, then calls exit(1) BEFORE any checkpoint write or training step runs.
 *
 * Manifest format (key=value text, .yaml extension for human familiarity but no YAML
 * dependency — every line is `key = value`, with `#` comments and blank lines allowed).
 * Each `arm = <name>` line opens a new section; subsequent keys belong to that section
 * until the next `arm =` or EOF. Unknown keys produce a warning, not an error
 * (forward-compatibility).
 *
 * Required fields per arm: model, tokenizer_basename, vocab_size, param_count_total
 * Optional field: param_count_active (only meaningful for MoE arms)
 *
 * Example manifest (programs/program_2_example/spec_invariants.yaml):
 *   arm = dense50m
 *   model = dense50m
 *   tokenizer_basename = tokenizer_32k.bin
 *   vocab_size = 32768
 *   param_count_total = 34619904
 *
 *   arm = medium
 *   model = medium
 *   tokenizer_basename = tokenizer_32k.bin
 *   vocab_size = 32768
 *   param_count_total = 62994944
 *   param_count_active = 34619904
 */

#ifndef CONFIG_DRIFT_H
#define CONFIG_DRIFT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Maximum length of any string field in an arm record. Rationale: model + arm names
 * are short identifiers (<=32 in practice); tokenizer basenames are file basenames
 * (<=64 in practice). 128 gives generous headroom without wasting RAM. */
#define CONFIG_DRIFT_STR_MAX 128

/* Maximum number of arms a single manifest may declare. Programs typically have 1-3
 * arms (dense, moe, occasionally a sensitivity arm). 16 is far above any plausible
 * real-world need. */
#define CONFIG_DRIFT_MAX_ARMS 16

/* One arm's expected config (filled by the manifest parser).
 *
 * D-182 / P-CONFIG-DRIFT-LR-WIRE extension: added 14 LR/schedule/dynamics fields
 * appended at end with per-field has_* flags. Field offsets for the original 5
 * (arm_name, model, tokenizer_basename, vocab_size, param_count_total +
 * param_count_active/has_param_count_active) are preserved so existing tests
 * compiled against the D-121 layout remain ABI-stable. */
typedef struct {
    char    arm_name[CONFIG_DRIFT_STR_MAX];           /* section key, e.g. "dense50m" */
    char    model[CONFIG_DRIFT_STR_MAX];              /* expected --model value */
    char    tokenizer_basename[CONFIG_DRIFT_STR_MAX]; /* expected basename(--tokenizer) */
    int32_t vocab_size;
    int64_t param_count_total;
    int64_t param_count_active;     /* only valid if has_param_count_active */
    bool    has_param_count_active; /* false for dense arms */

    /* ---- D-182 LR/schedule/dynamics extension (P-CONFIG-DRIFT-LR-WIRE) ---- */
    /* All fields are optional in legacy manifests: the parser sets has_<field> on
     * successful parse and the comparator only checks fields where has_<field>
     * is true. This keeps Program 1 / older manifests parseable without forcing
     * a 14-line edit to every spec_invariants.yaml in the lab. */

    /* Learning rate */
    double  lr_base;            /* base LR baked into scale_experiment.c */
    bool    has_lr_base;
    char    lr_schedule[CONFIG_DRIFT_STR_MAX]; /* e.g. "linear_warmup_cosine_decay" */
    bool    has_lr_schedule;
    int32_t lr_warmup_steps;    /* linear warmup steps */
    bool    has_lr_warmup_steps;
    double  lr_min;             /* cosine-decay floor */
    bool    has_lr_min;

    /* Optimizer */
    char    optimizer[CONFIG_DRIFT_STR_MAX]; /* e.g. "adamw" */
    bool    has_optimizer;
    double  adam_beta1;
    bool    has_adam_beta1;
    double  adam_beta2;
    bool    has_adam_beta2;
    double  weight_decay;       /* AdamW decoupled WD (pre-muP-scale base) */
    bool    has_weight_decay;
    double  grad_clip_norm;     /* global L2 grad clip */
    bool    has_grad_clip_norm;

    /* Batch / sequence shape */
    int32_t grad_accum_steps;
    bool    has_grad_accum_steps;
    int32_t batch_size;         /* tokens per micro-batch */
    bool    has_batch_size;
    int32_t seq_len;            /* --seq-len value */
    bool    has_seq_len;

    /* Training budget + reproducibility */
    int32_t max_steps;          /* --steps value */
    bool    has_max_steps;
    int32_t weight_seed;        /* --weight-seed value */
    bool    has_weight_seed;

    /* ---- D-194 LAB_CELL discriminator (P-PHASE3-CELL-ID) ----
     * Append-only ABI: NEW key `cell_id` (single-character `A`/`B`/`C`/`D`
     * for Program 2 Phase-3 factorial; the field is bounded to 8 bytes for
     * generous headroom on future programs that may use longer cell tokens).
     * Legacy manifests that do NOT carry cell_id parse with has_cell_id=false;
     * the comparator + lookup discipline gates on this flag. See landing memo
     * data/engineering/d194_config_drift_cell_id_landing.md for the LAB_CELL
     * env-var semantics + the find_arm vs find_arm_by_cell split. */
    char    cell_id[8];
    bool    has_cell_id;
} ConfigDriftArm;

/* A loaded manifest = list of arms. */
typedef struct {
    ConfigDriftArm arms[CONFIG_DRIFT_MAX_ARMS];
    int            n_arms;
} ConfigDriftManifest;

/* Runtime values to check against the manifest. The caller fills these from the
 * actual loaded model + tokenizer, then passes the struct to config_drift_assert.
 *
 * D-182 / P-CONFIG-DRIFT-LR-WIRE extension: 14 LR/schedule/dynamics fields
 * appended at end. On the runtime side the has_* flags are OPTIONAL — the
 * comparator only fires on fields the MANIFEST declares (manifest.has_<f>=true).
 * If runtime supplies a field but manifest does not, that's silently OK
 * (defense-in-depth for legacy programs that don't pin training dynamics).
 *
 * The 14 fields are the COORDINATION CONTRACT with implementation_engineer_c
 * (who populates this struct in scale_experiment.c init from the C-binary's
 * runtime context). Renaming or removing any of these fields breaks that
 * contract — extend by appending only. */
typedef struct {
    const char *model;              /* runtime --model value (e.g. "dense50m") */
    const char *tokenizer_path;     /* full --tokenizer path; basename is extracted internally */
    int32_t     vocab_size;         /* actual loaded tokenizer vocab size */
    int64_t     param_count_total;  /* actual computed total params */
    int64_t     param_count_active; /* only checked if has_param_count_active */
    bool        has_param_count_active;

    /* ---- D-182 LR/schedule/dynamics extension (P-CONFIG-DRIFT-LR-WIRE) ---- */

    /* Learning rate */
    double      lr_base;
    const char *lr_schedule;        /* may be NULL if not populated */
    int32_t     lr_warmup_steps;
    double      lr_min;

    /* Optimizer */
    const char *optimizer;          /* may be NULL if not populated */
    double      adam_beta1;
    double      adam_beta2;
    double      weight_decay;
    double      grad_clip_norm;

    /* Batch / sequence shape */
    int32_t     grad_accum_steps;
    int32_t     batch_size;
    int32_t     seq_len;

    /* Training budget + reproducibility */
    int32_t     max_steps;
    int32_t     weight_seed;
} ConfigDriftRuntime;

/* ---- Manifest loading ----------------------------------------------------- */

/* Initialize an empty manifest (zero-arm). Always safe to call before load. */
void config_drift_manifest_init(ConfigDriftManifest *mf);

/* Free any heap state owned by the manifest. (Currently no heap state — but
 * future-proofs for parser changes.) */
void config_drift_manifest_free(ConfigDriftManifest *mf);

/* Parse the manifest at `path` into `mf`. Returns 0 on success, non-zero on
 * file error or parse error. On error, mf is in an undefined state and
 * should be re-initialized before reuse. */
int config_drift_load(const char *path, ConfigDriftManifest *mf);

/* Return the arm whose `model` field matches `model_name`, or NULL if none.
 * (We match on model rather than arm_name so the runtime --model flag is the
 * single source-of-truth for arm selection — it's what the operator typed.)
 *
 * D-194 cell-id discipline: this lookup explicitly SKIPS arms that have
 * `has_cell_id=true`, so it always resolves to the legacy (non-cell)
 * Phase-2 backward-compat arm pair regardless of file order in the manifest.
 * Phase-3 cell selection uses config_drift_find_arm_by_cell() below. */
const ConfigDriftArm *config_drift_find_arm(const ConfigDriftManifest *mf,
                                            const char *model_name);

/* D-194 — return the arm whose `cell_id` matches, or NULL if none. Only
 * arms with has_cell_id=true are considered. Used by config_drift_assert_or_die
 * when the LAB_CELL env var is set (Phase-3 factorial cell selection). */
const ConfigDriftArm *config_drift_find_arm_by_cell(const ConfigDriftManifest *mf,
                                                    const char *cell_id);

/* ---- The assertion ------------------------------------------------------- */

/* Compare runtime values against the program's manifest, exit(1) on mismatch.
 *
 * Behavior:
 *   1. If env LAB_PROGRAM is unset → return immediately (no-op).
 *   2. Else load programs/$LAB_PROGRAM/spec_invariants.yaml (or
 *      $LAB_PROGRAM_ROOT/$LAB_PROGRAM/spec_invariants.yaml if LAB_PROGRAM_ROOT
 *      is set, for test isolation).
 *   3. If the manifest is missing or unparseable → CONFIG-DRIFT FATAL + exit(1).
 *   4. Arm lookup (D-194 cell-id discriminator):
 *        - If env LAB_CELL is set + non-empty: locate via cell_id match
 *          (config_drift_find_arm_by_cell). If no arm has cell_id==$LAB_CELL,
 *          CONFIG-DRIFT FATAL + exit(1).
 *        - Else: legacy fall-back via model-key match
 *          (config_drift_find_arm). The legacy lookup skips cell-aware arms,
 *          so Phase-2 evaluator/replay tooling that runs without LAB_CELL
 *          continues to hit the legacy arm pair unchanged from D-191 behavior.
 *   5. If no arm matches → CONFIG-DRIFT FATAL + exit(1).
 *   6. For each field in the matched arm, compare runtime vs expected. On any
 *      mismatch: print "CONFIG-DRIFT FATAL: <field> expected X, got Y" and
 *      track the failure. If any field mismatched, exit(1) at the end.
 *   7. Otherwise return silently (success).
 *
 * The function never returns non-zero on success — it either returns void or
 * calls exit(1). This keeps call-sites simple. */
void config_drift_assert_or_die(const ConfigDriftRuntime *rt);

#endif /* CONFIG_DRIFT_H */
