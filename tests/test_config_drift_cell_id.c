/* test_config_drift_cell_id.c -- Tests for D-194 cell-id discriminator
 * extension to config_drift (P-PHASE3-CELL-ID; binding-adjacent extension
 * gated on PI+Director unanimous co-sign).
 *
 * D-194 BACKGROUND:
 *   Phase-3 of program_2_dense_vs_moe_sub100m runs a 2x2 factorial
 *   (architecture x LR) at N=3 per cell. The locked prereg
 *   (`programs/program_2_dense_vs_moe_sub100m/phase3_p6_prereg.md` D-192
 *   unanimous lock) defines:
 *
 *     | Cell | Architecture | LR    | --model    | --lr   |
 *     | ---- | ------------ | ----- | ---------- | ------ |
 *     | A    | Dense-A      | 1e-3  | dense50m   | 0.001  |
 *     | B    | Dense-A      | 2e-3  | dense50m   | 0.002  |
 *     | C    | MoE Rev-2    | 1e-3  | medium     | 0.001  |
 *     | D    | MoE Rev-2    | 2e-3  | medium     | 0.002  |
 *
 *   Two arm blocks (one per `model`) cannot pin both LR variants. The D-194
 *   extension adds 4 cell-arm blocks (`phase3_cell_A` ... `phase3_cell_D`),
 *   each with a NEW `cell_id` discriminator key + the cell-specific lr_base.
 *   The binary uses `LAB_CELL=<A|B|C|D>` env var to select the cell-arm at
 *   runtime via `config_drift_find_arm_by_cell()`. With LAB_CELL UNSET, the
 *   binary falls back to the legacy `find_arm(model)` path which skips
 *   cell-aware arms — preserving Phase-2 backward-compat unchanged.
 *
 * Test plan (exhaustive 4 cells x 14 fields PASS + per-cell mismatch FATAL +
 * LAB_CELL gate-edge cases):
 *
 *   1. test_manifest_parses_with_cell_id      — parser side: real manifest
 *      loads with 4 cell-aware + 2 legacy arms, all has_cell_id flags as
 *      expected.
 *   2. test_find_arm_skips_cell_aware         — find_arm(model) on the real
 *      manifest hits the LEGACY dense50m/medium arms (NOT cell A/cell C).
 *   3. test_find_arm_by_cell_returns_correct  — find_arm_by_cell(A/B/C/D)
 *      returns the matching cell-arm block.
 *   4. test_full_match_cell_<X>               — 4 tests, one per cell:
 *      LAB_CELL=<X> + runtime fully populated with the cell's pinned values
 *      → silent rc=0 (the 4-cells x 14-fields PASS surface).
 *   5. test_<cell>_<field>_mismatch_fatal     — per-cell field-mismatch FATAL
 *      coverage. Each cell gets a representative field mutation; the lr_base
 *      mutations are the verdict-protection anchors (the fields the prereg
 *      varies across cells). Coverage matrix:
 *        cell A: lr_base, lr_warmup_steps, weight_decay
 *        cell B: lr_base, optimizer
 *        cell C: lr_base, batch_size
 *        cell D: lr_base, max_steps, adam_beta1
 *   6. test_lab_cell_set_no_match_fatal       — LAB_CELL=Z (non-existent)
 *      against the real cell-aware manifest → FATAL with "no arm has
 *      cell_id" message.
 *   7. test_lab_cell_set_legacy_manifest      — LAB_CELL=A against a
 *      synthetic legacy 2-arm manifest (no cell_id keys) → FATAL with
 *      "no arm has cell_id" message + the (none — manifest has no cell-aware
 *      arms; ...) explainer.
 *   8. test_lab_cell_unset_fallback_legacy    — LAB_CELL UNSET against the
 *      real cell-aware manifest with a Phase-2-shape runtime (--model
 *      dense50m, lr_base=0.002 matching the legacy dense50m arm) → silent
 *      rc=0 (model-key fallback resolves to legacy arm, not cell A).
 *
 * Why fork+waitpid: config_drift_assert_or_die calls exit(1) on FATAL.
 * Pattern lifted from test_config_drift_lr_pin.c.
 */

#include "../src/tests/unity.h"
#include "config_drift.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

/* ---- mkdir_p / write_file / rm_file fixture helpers (mirror of D-191) ---- */

static int mkdir_p(const char *path) {
    char tmp[1024];
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(tmp)) return -1;
    memcpy(tmp, path, len + 1);
    if (tmp[len - 1] == '/') tmp[len - 1] = '\0';
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
            *p = '/';
        }
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
    return 0;
}

static int write_file(const char *path, const char *content) {
    char dir[1024];
    snprintf(dir, sizeof(dir), "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) {
        *slash = '\0';
        if (mkdir_p(dir) != 0) return -1;
    }
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    size_t n = strlen(content);
    size_t w = fwrite(content, 1, n, f);
    fclose(f);
    return (w == n) ? 0 : -1;
}

static void rm_file(const char *path) { unlink(path); }

/* ---- Fork-based subprocess runner (lifted from test_config_drift_lr_pin.c) */

typedef int (*child_fn_t)(void *ctx);

static int run_in_subprocess(child_fn_t body, void *ctx,
                             char *stderr_out, size_t stderr_cap) {
    int pipefd[2];
    if (pipe(pipefd) != 0) return -1;
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]); close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            close(devnull);
        }
        int rc = body(ctx);
        fflush(stderr);
        _exit(rc);
    }
    close(pipefd[1]);
    size_t off = 0;
    if (stderr_cap > 0) stderr_out[0] = '\0';
    char buf[256];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
        if (off + (size_t)n + 1 < stderr_cap) {
            memcpy(stderr_out + off, buf, (size_t)n);
            off += (size_t)n;
            stderr_out[off] = '\0';
        }
    }
    close(pipefd[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    if (WIFSIGNALED(status)) return 128 + WTERMSIG(status);
    return -1;
}

/* ---- Per-cell runtime constructors (match the D-194 manifest pins) ----
 *
 * Cell A: dense50m  + lr_base=0.001
 * Cell B: dense50m  + lr_base=0.002  (== legacy dense50m arm)
 * Cell C: medium    + lr_base=0.001  (== legacy medium arm)
 * Cell D: medium    + lr_base=0.002
 *
 * NOTE: weight_seed is intentionally NOT pinned per cell-arm in the manifest
 * (it is a per-cell variable; 42/43/44 across the 3 seeds). The runtime
 * supplies it but `arm->has_weight_seed=false` for cell-aware arms means
 * the comparator silently skips the field. We populate the runtime with 42
 * for symmetry with the existing test scaffolding; no FATAL even if it
 * didn't match. */

static void make_cell_A_runtime_full(ConfigDriftRuntime *rt) {
    memset(rt, 0, sizeof(*rt));
    rt->model              = "dense50m";
    rt->tokenizer_path     = "data/training/tokenizer_32k.bin";
    rt->vocab_size         = 32768;
    rt->param_count_total  = 34619904;
    rt->has_param_count_active = false;
    rt->lr_base            = 0.001;  /* cell A pin */
    rt->lr_schedule        = "linear_warmup_cosine_decay";
    rt->lr_warmup_steps    = 200;
    rt->lr_min             = 1e-05;
    rt->optimizer          = "adamw";
    rt->adam_beta1         = 0.9;
    rt->adam_beta2         = 0.999;
    rt->weight_decay       = 0.01;
    rt->grad_clip_norm     = 1.0;
    rt->grad_accum_steps   = 4;
    rt->batch_size         = 1;
    rt->seq_len            = 512;
    rt->max_steps          = 5000;
    rt->weight_seed        = 42;  /* not pinned in cell-arm; harmless */
}

static void make_cell_B_runtime_full(ConfigDriftRuntime *rt) {
    make_cell_A_runtime_full(rt);
    rt->lr_base = 0.002;  /* cell B pin */
}

static void make_cell_C_runtime_full(ConfigDriftRuntime *rt) {
    make_cell_A_runtime_full(rt);
    rt->model              = "medium";
    rt->param_count_total  = 62988800;
    rt->param_count_active = 34677248;
    rt->has_param_count_active = true;
    /* lr_base inherited from cell_A (0.001) — same as cell C pin */
}

static void make_cell_D_runtime_full(ConfigDriftRuntime *rt) {
    make_cell_C_runtime_full(rt);
    rt->lr_base = 0.002;  /* cell D pin */
}

/* Establish LAB_PROGRAM env so the assertion loads the real Program 2
 * manifest. LAB_CELL is set by each test that needs cell-aware lookup. */
static void enter_program_2_env(void) {
    setenv("LAB_PROGRAM", "program_2_dense_vs_moe_sub100m", 1);
    unsetenv("LAB_PROGRAM_ROOT");
    unsetenv("LAB_CELL");
}

/* ---- Test 1: parser recognizes cell_id + 4 cell-aware + 2 legacy arms ---- */

static void test_manifest_parses_with_cell_id(void) {
    const char *path =
        "programs/program_2_dense_vs_moe_sub100m/spec_invariants.yaml";

    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    int rc = config_drift_load(path, &mf);
    ASSERT_EQUAL_INT(0, rc);
    /* 4 cell-arms + 2 legacy arms = 6 total. */
    ASSERT_EQUAL_INT(6, mf.n_arms);

    /* Count cell-aware arms by has_cell_id flag. */
    int n_cell_arms = 0;
    int n_legacy_arms = 0;
    for (int i = 0; i < mf.n_arms; i++) {
        if (mf.arms[i].has_cell_id) n_cell_arms++;
        else n_legacy_arms++;
    }
    ASSERT_EQUAL_INT(4, n_cell_arms);
    ASSERT_EQUAL_INT(2, n_legacy_arms);

    /* Each cell-arm has the expected cell_id + matching lr_base from the
     * locked prereg's §2.1 factorial table. */
    const ConfigDriftArm *cA = config_drift_find_arm_by_cell(&mf, "A");
    ASSERT_NOT_NULL(cA);
    ASSERT_TRUE(cA->has_cell_id);
    ASSERT_EQUAL_STR("A", cA->cell_id);
    ASSERT_EQUAL_STR("dense50m", cA->model);
    ASSERT_EQUAL_FLOAT(0.001, cA->lr_base, 1e-9);
    ASSERT_FALSE(cA->has_weight_seed);  /* per-cell variable; not pinned */

    const ConfigDriftArm *cB = config_drift_find_arm_by_cell(&mf, "B");
    ASSERT_NOT_NULL(cB);
    ASSERT_TRUE(cB->has_cell_id);
    ASSERT_EQUAL_STR("B", cB->cell_id);
    ASSERT_EQUAL_STR("dense50m", cB->model);
    ASSERT_EQUAL_FLOAT(0.002, cB->lr_base, 1e-9);

    const ConfigDriftArm *cC = config_drift_find_arm_by_cell(&mf, "C");
    ASSERT_NOT_NULL(cC);
    ASSERT_EQUAL_STR("C", cC->cell_id);
    ASSERT_EQUAL_STR("medium", cC->model);
    ASSERT_EQUAL_FLOAT(0.001, cC->lr_base, 1e-9);

    const ConfigDriftArm *cD = config_drift_find_arm_by_cell(&mf, "D");
    ASSERT_NOT_NULL(cD);
    ASSERT_EQUAL_STR("D", cD->cell_id);
    ASSERT_EQUAL_STR("medium", cD->model);
    ASSERT_EQUAL_FLOAT(0.002, cD->lr_base, 1e-9);

    /* Non-existent cell returns NULL. */
    ASSERT_NULL(config_drift_find_arm_by_cell(&mf, "Z"));

    config_drift_manifest_free(&mf);
}

/* ---- Test 2: find_arm(model) on the real manifest hits legacy arms only ---- */

static void test_find_arm_skips_cell_aware(void) {
    const char *path =
        "programs/program_2_dense_vs_moe_sub100m/spec_invariants.yaml";
    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    ASSERT_EQUAL_INT(0, config_drift_load(path, &mf));

    /* Both `dense50m` and `medium` model names appear in BOTH cell-aware AND
     * legacy arms. find_arm() must always return the legacy (non-cell)
     * variant — i.e. lr_base=0.002 for dense50m (Phase-2 dense pin) and
     * lr_base=0.001 for medium (Phase-2 MoE pin). */
    const ConfigDriftArm *d = config_drift_find_arm(&mf, "dense50m");
    ASSERT_NOT_NULL(d);
    ASSERT_FALSE(d->has_cell_id);  /* legacy-only */
    ASSERT_EQUAL_STR("dense50m", d->arm_name);  /* legacy arm name */
    ASSERT_TRUE(d->has_weight_seed);  /* legacy arm pins weight_seed */
    ASSERT_EQUAL_INT(42, d->weight_seed);
    ASSERT_EQUAL_FLOAT(0.002, d->lr_base, 1e-9);  /* Phase-2 dense LR */

    const ConfigDriftArm *m = config_drift_find_arm(&mf, "medium");
    ASSERT_NOT_NULL(m);
    ASSERT_FALSE(m->has_cell_id);
    ASSERT_EQUAL_STR("medium", m->arm_name);
    ASSERT_TRUE(m->has_weight_seed);
    ASSERT_EQUAL_INT(42, m->weight_seed);
    ASSERT_EQUAL_FLOAT(0.001, m->lr_base, 1e-9);  /* Phase-2 MoE LR */

    config_drift_manifest_free(&mf);
}

/* ---- Test 3: find_arm_by_cell returns matching cell-arm ---- */

static void test_find_arm_by_cell_returns_correct(void) {
    const char *path =
        "programs/program_2_dense_vs_moe_sub100m/spec_invariants.yaml";
    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    ASSERT_EQUAL_INT(0, config_drift_load(path, &mf));

    /* Each find returns the right cell-arm; arm_name follows the
     * `phase3_cell_<X>` convention. */
    const ConfigDriftArm *cA = config_drift_find_arm_by_cell(&mf, "A");
    ASSERT_NOT_NULL(cA);
    ASSERT_EQUAL_STR("phase3_cell_A", cA->arm_name);
    const ConfigDriftArm *cB = config_drift_find_arm_by_cell(&mf, "B");
    ASSERT_NOT_NULL(cB);
    ASSERT_EQUAL_STR("phase3_cell_B", cB->arm_name);
    const ConfigDriftArm *cC = config_drift_find_arm_by_cell(&mf, "C");
    ASSERT_NOT_NULL(cC);
    ASSERT_EQUAL_STR("phase3_cell_C", cC->arm_name);
    const ConfigDriftArm *cD = config_drift_find_arm_by_cell(&mf, "D");
    ASSERT_NOT_NULL(cD);
    ASSERT_EQUAL_STR("phase3_cell_D", cD->arm_name);

    /* Empty cell_id returns NULL. */
    ASSERT_NULL(config_drift_find_arm_by_cell(&mf, ""));
    /* NULL cell_id is safe and returns NULL. */
    ASSERT_NULL(config_drift_find_arm_by_cell(&mf, NULL));

    config_drift_manifest_free(&mf);
}

/* ---- Per-cell match / mismatch test infrastructure ----
 *
 * Each MismatchCase carries the cell letter (sets LAB_CELL), the runtime
 * constructor, an optional field-mutation closure, and the expected FATAL
 * needle (`<arm_name> <field>` per the D-191 arm-prefixed printer format).
 */

typedef struct {
    const char *cell_id;        /* "A", "B", "C", or "D" */
    void (*make_rt)(ConfigDriftRuntime *);  /* canonical full-match constructor */
    void (*mutate)(ConfigDriftRuntime *);   /* NULL = no mutation (full match) */
    const char *expected_arm;   /* "phase3_cell_A" etc. */
    const char *expected_field; /* "lr_base", "weight_decay", ... */
} CellMismatchCase;

static int child_run_assert_cell(void *ctx) {
    const CellMismatchCase *c = (const CellMismatchCase *)ctx;
    enter_program_2_env();
    setenv("LAB_CELL", c->cell_id, 1);
    ConfigDriftRuntime rt;
    c->make_rt(&rt);
    if (c->mutate) c->mutate(&rt);
    config_drift_assert_or_die(&rt);
    return 0;  /* full-match path */
}

static void expect_cell_mismatch(const CellMismatchCase *c) {
    char err[4096];
    int rc = run_in_subprocess(child_run_assert_cell, (void *)c,
                               err, sizeof(err));
    if (rc != 1) {
        printf("\n    [DEBUG] cell=%s expected rc=1, got rc=%d; stderr:\n%s\n",
               c->cell_id, rc, err);
    }
    ASSERT_EQUAL_INT(1, rc);
    char needle[256];
    snprintf(needle, sizeof(needle),
             "CONFIG-DRIFT FATAL: %s %s",
             c->expected_arm, c->expected_field);
    if (!strstr(err, needle)) {
        printf("\n    [DEBUG] needle '%s' not in stderr:\n%s\n", needle, err);
    }
    ASSERT_TRUE(strstr(err, needle) != NULL);
}

static void expect_cell_match(const CellMismatchCase *c) {
    char err[4096];
    int rc = run_in_subprocess(child_run_assert_cell, (void *)c,
                               err, sizeof(err));
    if (rc != 0) {
        printf("\n    [DEBUG] cell=%s expected rc=0, got rc=%d; stderr:\n%s\n",
               c->cell_id, rc, err);
    }
    ASSERT_EQUAL_INT(0, rc);
}

/* ---- Test 4: full-match per cell (4 tests; the 4 cells x 14 fields PASS
 *      surface — every populated field across all 4 cells silently passes). */

static void test_full_match_cell_A(void) {
    CellMismatchCase c = {
        .cell_id = "A",
        .make_rt = make_cell_A_runtime_full,
        .mutate = NULL,
        .expected_arm = "phase3_cell_A",
        .expected_field = "(none)",
    };
    expect_cell_match(&c);
}
static void test_full_match_cell_B(void) {
    CellMismatchCase c = {
        .cell_id = "B",
        .make_rt = make_cell_B_runtime_full,
        .mutate = NULL,
        .expected_arm = "phase3_cell_B",
        .expected_field = "(none)",
    };
    expect_cell_match(&c);
}
static void test_full_match_cell_C(void) {
    CellMismatchCase c = {
        .cell_id = "C",
        .make_rt = make_cell_C_runtime_full,
        .mutate = NULL,
        .expected_arm = "phase3_cell_C",
        .expected_field = "(none)",
    };
    expect_cell_match(&c);
}
static void test_full_match_cell_D(void) {
    CellMismatchCase c = {
        .cell_id = "D",
        .make_rt = make_cell_D_runtime_full,
        .mutate = NULL,
        .expected_arm = "phase3_cell_D",
        .expected_field = "(none)",
    };
    expect_cell_match(&c);
}

/* ---- Test 5: per-cell field-mismatch FATAL coverage ----
 *
 * Per-cell coverage matrix (one mutation per row; verdict-protection anchors
 * are the lr_base mutations on each cell — that's the field the prereg varies
 * across cells). Other field mutations sample the 13 D-181-pinned non-LR
 * fields broadly so a regression in any one site is caught.
 *
 *   cell A: lr_base, lr_warmup_steps, weight_decay
 *   cell B: lr_base, optimizer
 *   cell C: lr_base, batch_size
 *   cell D: lr_base, max_steps, adam_beta1
 */

/* Field mutations (one per field; reused across cells). */
static void mut_lr_base_off(ConfigDriftRuntime *rt) {
    /* Set to a value that mismatches BOTH 0.001 and 0.002 cell pins so this
     * single mutation works whether the cell pins lr_base=0.001 or 0.002. */
    rt->lr_base = 0.0009;
}
static void mut_lr_warmup_off(ConfigDriftRuntime *rt) { rt->lr_warmup_steps = 100; }
static void mut_weight_decay_off(ConfigDriftRuntime *rt) { rt->weight_decay = 0.1; }
static void mut_optimizer_off(ConfigDriftRuntime *rt) { rt->optimizer = "sgd"; }
static void mut_batch_size_off(ConfigDriftRuntime *rt) { rt->batch_size = 2; }
static void mut_max_steps_off(ConfigDriftRuntime *rt) { rt->max_steps = 4999; }
static void mut_adam_beta1_off(ConfigDriftRuntime *rt) { rt->adam_beta1 = 0.95; }

/* Cell A: lr_base, lr_warmup_steps, weight_decay */
static void test_cell_A_lr_base_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "A", .make_rt = make_cell_A_runtime_full,
        .mutate = mut_lr_base_off,
        .expected_arm = "phase3_cell_A", .expected_field = "lr_base",
    };
    expect_cell_mismatch(&c);
}
static void test_cell_A_lr_warmup_steps_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "A", .make_rt = make_cell_A_runtime_full,
        .mutate = mut_lr_warmup_off,
        .expected_arm = "phase3_cell_A", .expected_field = "lr_warmup_steps",
    };
    expect_cell_mismatch(&c);
}
static void test_cell_A_weight_decay_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "A", .make_rt = make_cell_A_runtime_full,
        .mutate = mut_weight_decay_off,
        .expected_arm = "phase3_cell_A", .expected_field = "weight_decay",
    };
    expect_cell_mismatch(&c);
}

/* Cell B: lr_base, optimizer */
static void test_cell_B_lr_base_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "B", .make_rt = make_cell_B_runtime_full,
        .mutate = mut_lr_base_off,
        .expected_arm = "phase3_cell_B", .expected_field = "lr_base",
    };
    expect_cell_mismatch(&c);
}
static void test_cell_B_optimizer_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "B", .make_rt = make_cell_B_runtime_full,
        .mutate = mut_optimizer_off,
        .expected_arm = "phase3_cell_B", .expected_field = "optimizer",
    };
    expect_cell_mismatch(&c);
}

/* Cell C: lr_base, batch_size */
static void test_cell_C_lr_base_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "C", .make_rt = make_cell_C_runtime_full,
        .mutate = mut_lr_base_off,
        .expected_arm = "phase3_cell_C", .expected_field = "lr_base",
    };
    expect_cell_mismatch(&c);
}
static void test_cell_C_batch_size_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "C", .make_rt = make_cell_C_runtime_full,
        .mutate = mut_batch_size_off,
        .expected_arm = "phase3_cell_C", .expected_field = "batch_size",
    };
    expect_cell_mismatch(&c);
}

/* Cell D: lr_base, max_steps, adam_beta1 */
static void test_cell_D_lr_base_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "D", .make_rt = make_cell_D_runtime_full,
        .mutate = mut_lr_base_off,
        .expected_arm = "phase3_cell_D", .expected_field = "lr_base",
    };
    expect_cell_mismatch(&c);
}
static void test_cell_D_max_steps_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "D", .make_rt = make_cell_D_runtime_full,
        .mutate = mut_max_steps_off,
        .expected_arm = "phase3_cell_D", .expected_field = "max_steps",
    };
    expect_cell_mismatch(&c);
}
static void test_cell_D_adam_beta1_mismatch_fatal(void) {
    CellMismatchCase c = {
        .cell_id = "D", .make_rt = make_cell_D_runtime_full,
        .mutate = mut_adam_beta1_off,
        .expected_arm = "phase3_cell_D", .expected_field = "adam_beta1",
    };
    expect_cell_mismatch(&c);
}

/* ---- Test 6: LAB_CELL set to a non-existent cell value → FATAL ----
 *
 * The existing 4 cells in the manifest are A/B/C/D. Setting LAB_CELL=Z must
 * FATAL with an unambiguous message that includes the requested cell value
 * AND lists the available cells (so the operator can immediately see the
 * typo). */

static int child_lab_cell_z(void *ctx) {
    (void)ctx;
    enter_program_2_env();
    setenv("LAB_CELL", "Z", 1);  /* not a valid cell */
    ConfigDriftRuntime rt;
    make_cell_A_runtime_full(&rt);  /* runtime values don't matter — gate
                                     * fires on cell-id lookup before
                                     * field-comparison */
    config_drift_assert_or_die(&rt);
    return 0;
}

static void test_lab_cell_set_no_match_fatal(void) {
    char err[4096];
    int rc = run_in_subprocess(child_lab_cell_z, NULL, err, sizeof(err));
    ASSERT_EQUAL_INT(1, rc);
    /* Required substrings: requested cell value + the explainer header. */
    ASSERT_TRUE(strstr(err, "LAB_CELL='Z'") != NULL);
    ASSERT_TRUE(strstr(err, "no arm in") != NULL);
    ASSERT_TRUE(strstr(err, "has cell_id='Z'") != NULL);
    /* Available cells list should include all 4 cell-arm names. */
    ASSERT_TRUE(strstr(err, "phase3_cell_A") != NULL);
    ASSERT_TRUE(strstr(err, "phase3_cell_D") != NULL);
}

/* ---- Test 7: LAB_CELL set + legacy 2-arm manifest (no cell_id) → FATAL ----
 *
 * Defense for the case where a legacy program manifest (no D-194 cell-id
 * extension) is selected via LAB_PROGRAM AND LAB_CELL is mistakenly set
 * (operator typo or misconfiguration). Must FATAL with the (none — manifest
 * has no cell-aware arms; ...) explainer. */

static const char *MANIFEST_LEGACY_2ARM =
    "# Legacy 2-arm manifest with no cell_id keys (D-181 schema).\n"
    "arm = dense50m\n"
    "model = dense50m\n"
    "tokenizer_basename = tokenizer_32k.bin\n"
    "vocab_size = 32768\n"
    "param_count_total = 34619904\n"
    "lr_base = 0.002\n"
    "lr_schedule = linear_warmup_cosine_decay\n"
    "lr_warmup_steps = 200\n"
    "lr_min = 1e-05\n"
    "optimizer = adamw\n"
    "adam_beta1 = 0.9\n"
    "adam_beta2 = 0.999\n"
    "weight_decay = 0.01\n"
    "grad_clip_norm = 1.0\n"
    "grad_accum_steps = 4\n"
    "batch_size = 1\n"
    "seq_len = 512\n"
    "max_steps = 5000\n"
    "weight_seed = 42\n"
    "\n"
    "arm = medium\n"
    "model = medium\n"
    "tokenizer_basename = tokenizer_32k.bin\n"
    "vocab_size = 32768\n"
    "param_count_total = 62988800\n"
    "param_count_active = 34677248\n"
    "lr_base = 0.001\n"
    "lr_schedule = linear_warmup_cosine_decay\n"
    "lr_warmup_steps = 200\n"
    "lr_min = 1e-05\n"
    "optimizer = adamw\n"
    "adam_beta1 = 0.9\n"
    "adam_beta2 = 0.999\n"
    "weight_decay = 0.01\n"
    "grad_clip_norm = 1.0\n"
    "grad_accum_steps = 4\n"
    "batch_size = 1\n"
    "seq_len = 512\n"
    "max_steps = 5000\n"
    "weight_seed = 42\n";

static int child_lab_cell_against_legacy(void *ctx) {
    const char *root = (const char *)ctx;
    setenv("LAB_PROGRAM", "legacy_no_cell_id", 1);
    setenv("LAB_PROGRAM_ROOT", root, 1);
    setenv("LAB_CELL", "A", 1);  /* operator typo: cell-aware lookup against
                                  * a legacy manifest */
    ConfigDriftRuntime rt;
    make_cell_A_runtime_full(&rt);
    config_drift_assert_or_die(&rt);
    return 0;
}

static void test_lab_cell_set_legacy_manifest(void) {
    const char *root = "/tmp/cd_cellid_test_legacy";
    const char *manifest =
        "/tmp/cd_cellid_test_legacy/legacy_no_cell_id/spec_invariants.yaml";
    ASSERT_EQUAL_INT(0, write_file(manifest, MANIFEST_LEGACY_2ARM));

    char err[4096];
    int rc = run_in_subprocess(child_lab_cell_against_legacy, (void *)root,
                               err, sizeof(err));
    ASSERT_EQUAL_INT(1, rc);
    ASSERT_TRUE(strstr(err, "LAB_CELL='A'") != NULL);
    ASSERT_TRUE(strstr(err, "no arm") != NULL);
    /* Critical explainer: tells the operator that no cell-aware arms exist
     * in this manifest (so they know the manifest, not LAB_CELL, is wrong). */
    ASSERT_TRUE(strstr(err, "manifest has no cell-aware arms") != NULL);

    rm_file(manifest);
    rmdir("/tmp/cd_cellid_test_legacy/legacy_no_cell_id");
    rmdir(root);
}

/* ---- Test 8: LAB_CELL UNSET + cell-aware manifest → legacy fallback PASS ----
 *
 * THIS IS THE BACKWARD-COMPAT INVARIANT. Phase-2 evaluator/replay tooling
 * that runs against the cell-aware D-194 manifest WITHOUT LAB_CELL set must
 * resolve via the model-key fallback to the LEGACY arm pair (lr_base=0.002
 * for dense50m, lr_base=0.001 for medium — Phase-2 D-181 pins).
 *
 * Construct a Phase-2-shape runtime: --model dense50m + lr_base=0.002
 * (the Phase-2 dense LR). With LAB_CELL UNSET, the comparator must:
 *   - call find_arm("dense50m") which skips cell-aware arms
 *   - resolve to the legacy `dense50m` arm with lr_base=0.002
 *   - compare runtime.lr_base=0.002 against arm.lr_base=0.002 → MATCH
 *   - silent rc=0
 *
 * If the comparator instead resolved to cell A (lr_base=0.001), the runtime
 * lr_base=0.002 would FATAL — that's the regression this test guards against.
 */

static int child_lab_cell_unset_phase2_shape(void *ctx) {
    (void)ctx;
    setenv("LAB_PROGRAM", "program_2_dense_vs_moe_sub100m", 1);
    unsetenv("LAB_PROGRAM_ROOT");
    unsetenv("LAB_CELL");

    ConfigDriftRuntime rt;
    /* Phase-2 dense50m runtime — uses Phase-2 LR=0.002 (NOT cell A's 0.001). */
    rt.model              = "dense50m";
    rt.tokenizer_path     = "data/training/tokenizer_32k.bin";
    rt.vocab_size         = 32768;
    rt.param_count_total  = 34619904;
    rt.has_param_count_active = false;
    rt.lr_base            = 0.002;  /* Phase-2 dense pin */
    rt.lr_schedule        = "linear_warmup_cosine_decay";
    rt.lr_warmup_steps    = 200;
    rt.lr_min             = 1e-05;
    rt.optimizer          = "adamw";
    rt.adam_beta1         = 0.9;
    rt.adam_beta2         = 0.999;
    rt.weight_decay       = 0.01;
    rt.grad_clip_norm     = 1.0;
    rt.grad_accum_steps   = 4;
    rt.batch_size         = 1;
    rt.seq_len            = 512;
    rt.max_steps          = 5000;
    rt.weight_seed        = 42;  /* Phase-2 weight_seed pin (LEGACY arm pins it) */

    config_drift_assert_or_die(&rt);
    return 0;
}

static void test_lab_cell_unset_fallback_legacy(void) {
    char err[4096];
    int rc = run_in_subprocess(child_lab_cell_unset_phase2_shape, NULL,
                               err, sizeof(err));
    if (rc != 0) {
        printf("\n    [DEBUG] expected rc=0 (Phase-2 fallback PASS), got rc=%d; "
               "stderr:\n%s\n", rc, err);
    }
    ASSERT_EQUAL_INT(0, rc);
    /* No FATAL on stderr (WARNs from earlier may exist; FATAL must not). */
    ASSERT_TRUE(strstr(err, "CONFIG-DRIFT FATAL") == NULL);
}

/* ---- Test 9 (parser-edge): empty cell_id value → parse error ---- */

static const char *MANIFEST_EMPTY_CELL_ID =
    "arm = test_empty_cell\n"
    "cell_id = \n"  /* empty value should FATAL the parser */
    "model = dense50m\n"
    "tokenizer_basename = tokenizer_32k.bin\n"
    "vocab_size = 32768\n"
    "param_count_total = 34619904\n";

static void test_parser_rejects_empty_cell_id(void) {
    const char *path = "/tmp/cd_cellid_test_empty.yaml";
    ASSERT_EQUAL_INT(0, write_file(path, MANIFEST_EMPTY_CELL_ID));

    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    int rc = config_drift_load(path, &mf);
    /* Parser must reject empty cell_id (cell_id is a discriminator — empty
     * is meaningless and would silently match LAB_CELL="" which is the
     * "unset" case in the assert function — semantic conflict). */
    ASSERT_EQUAL_INT(-1, rc);

    config_drift_manifest_free(&mf);
    rm_file(path);
}

/* ---- Test 10 (parser-edge): cell_id too long → parse error ---- */

static const char *MANIFEST_LONG_CELL_ID =
    "arm = test_long_cell\n"
    "cell_id = TOOLONGCELLID\n"  /* 13 chars > 7-char max */
    "model = dense50m\n"
    "tokenizer_basename = tokenizer_32k.bin\n"
    "vocab_size = 32768\n"
    "param_count_total = 34619904\n";

static void test_parser_rejects_long_cell_id(void) {
    const char *path = "/tmp/cd_cellid_test_long.yaml";
    ASSERT_EQUAL_INT(0, write_file(path, MANIFEST_LONG_CELL_ID));

    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    int rc = config_drift_load(path, &mf);
    ASSERT_EQUAL_INT(-1, rc);

    config_drift_manifest_free(&mf);
    rm_file(path);
}

/* ---- main: register and report ---- */

int main(void) {
    printf("=== config_drift cell-id tests "
           "(D-194, P-PHASE3-CELL-ID) ===\n");

    /* Test 1-3: parser + lookup wiring. */
    RUN_TEST(test_manifest_parses_with_cell_id);
    RUN_TEST(test_find_arm_skips_cell_aware);
    RUN_TEST(test_find_arm_by_cell_returns_correct);

    /* Test 4: full-match per cell (4 cells x 14 fields PASS surface). */
    RUN_TEST(test_full_match_cell_A);
    RUN_TEST(test_full_match_cell_B);
    RUN_TEST(test_full_match_cell_C);
    RUN_TEST(test_full_match_cell_D);

    /* Test 5: per-cell field-mismatch FATAL coverage (10 mutations across
     * 4 cells; lr_base on each cell is the verdict-protection anchor). */
    RUN_TEST(test_cell_A_lr_base_mismatch_fatal);
    RUN_TEST(test_cell_A_lr_warmup_steps_mismatch_fatal);
    RUN_TEST(test_cell_A_weight_decay_mismatch_fatal);
    RUN_TEST(test_cell_B_lr_base_mismatch_fatal);
    RUN_TEST(test_cell_B_optimizer_mismatch_fatal);
    RUN_TEST(test_cell_C_lr_base_mismatch_fatal);
    RUN_TEST(test_cell_C_batch_size_mismatch_fatal);
    RUN_TEST(test_cell_D_lr_base_mismatch_fatal);
    RUN_TEST(test_cell_D_max_steps_mismatch_fatal);
    RUN_TEST(test_cell_D_adam_beta1_mismatch_fatal);

    /* Test 6-7: LAB_CELL gate-edge cases (set+no-match, set+legacy). */
    RUN_TEST(test_lab_cell_set_no_match_fatal);
    RUN_TEST(test_lab_cell_set_legacy_manifest);

    /* Test 8: backward-compat invariant (LAB_CELL unset + cell-aware
     * manifest → Phase-2 model-key fallback PASS). */
    RUN_TEST(test_lab_cell_unset_fallback_legacy);

    /* Test 9-10: parser-edge defensive coverage. */
    RUN_TEST(test_parser_rejects_empty_cell_id);
    RUN_TEST(test_parser_rejects_long_cell_id);

    TEST_REPORT();
}
