/* test_config_drift_lr_pin.c -- Tests for D-181 LR / training-dynamics pinning
 * AND D-182 binary-side comparator wiring (P-CONFIG-DRIFT-LR-WIRE).
 *
 * D-181 BACKGROUND (PI binding condition #1, 2026-04-24):
 *   Program 2 Phase-2 stat review surfaced a 2× learning-rate confound (Dense-A
 *   LR=2e-3 vs MoE Rev-2 LR=1e-3, hardcoded at scale_experiment.c:714-718 keyed
 *   off `--model`). The structural fields on `spec_invariants.yaml` did NOT
 *   pin `lr_base` / `lr_schedule` / optimizer dynamics. PI condition #1 extended
 *   the manifest with 14 such fields (lr_base, lr_schedule, lr_warmup_steps,
 *   lr_min, optimizer, adam_beta1, adam_beta2, weight_decay, grad_clip_norm,
 *   grad_accum_steps, batch_size, seq_len, max_steps, weight_seed).
 *
 * D-182 / D-190 1i CARRY-FORWARD (P-CONFIG-DRIFT-LR-WIRE):
 *   The D-181 manifest extension landed under the parser's forward-compat
 *   "warn-on-unknown" path (recorded truth, not yet enforced). Phase-3
 *   factorial design (D-190 1i locked) makes silent LR drift verdict-breaking,
 *   so the parser was extended to RECOGNIZE the 14 keys + the comparator now
 *   FIRES on mismatch with arm-prefixed FATAL messages. This test file
 *   exercises both sides:
 *
 *     1. test_manifest_parses_with_lr_pin
 *        — loads the REAL program_2 manifest via LAB_PROGRAM_ROOT, asserts
 *          parse OK + structural fields intact + ALL 14 D-181 fields populated
 *          into the per-arm struct with the exact pinned values.
 *
 *     2. test_manifest_lr_values_expected
 *        — text-level regression guard against accidental in-place edits to
 *          the manifest's pinned LR values.
 *
 *     3. test_runtime_<field>_mismatch_<arm> (15 tests)
 *        — for each of the 14 D-181 fields plus the legacy lr_base differential,
 *          forks a child that constructs a synthetic mismatch and calls
 *          config_drift_assert_or_die; asserts the child exits with rc=1 and
 *          stderr contains "CONFIG-DRIFT FATAL: <arm> <field>".
 *
 *     4. test_runtime_full_match_<arm> (2 tests)
 *        — populate runtime with ALL 14 manifest values exactly; assert
 *          config_drift_assert_or_die returns silently (rc=0 inside child).
 *
 *     5. test_legacy_manifest_no_lr_pin_no_fatal
 *        — synthetic legacy manifest that does NOT declare any LR field;
 *          assert parse OK + assert call returns silently. Defense-in-depth
 *          for Program 1 / older programs.
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

/* ---- mkdir_p / write_file / rm_file fixture helpers ---- */
/* These mirror tests/test_config_drift.c so the legacy-manifest fixture in
 * test #5 below can build a private programs/<name>/spec_invariants.yaml
 * directory without polluting the real programs/ tree. */

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

/* ---- fork-based subprocess runner (mirrors test_config_drift.c) ----
 *
 * config_drift_assert_or_die calls exit(1) on FATAL — testing it requires
 * fork() + waitpid(). See semantic memory § "TDD discipline for fork-based
 * assertion tests" for the two non-obvious gotchas (flush parent stdio
 * pre-fork; redirect child stdout to /dev/null too). */

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

/* ---- canonical "all 14 fields match dense50m" runtime constructor ----
 *
 * The values match programs/program_2_dense_vs_moe_sub100m/spec_invariants.yaml
 * dense50m arm (D-181 pinned). Mismatch tests below mutate ONE field then
 * invoke the assert; full-match tests use this struct unmodified.
 *
 * Static `lr_schedule` / `optimizer` strings — assert struct holds pointer,
 * lifetime is until program exit. */
static void make_dense_runtime_full(ConfigDriftRuntime *rt) {
    memset(rt, 0, sizeof(*rt));
    rt->model              = "dense50m";
    rt->tokenizer_path     = "data/training/tokenizer_32k.bin";
    rt->vocab_size         = 32768;
    rt->param_count_total  = 34619904;
    /* Dense arm: no active count. */
    rt->has_param_count_active = false;
    /* D-181 LR/dynamics fields — must match dense50m manifest section. */
    rt->lr_base            = 0.002;
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
    rt->weight_seed        = 42;
}

static void make_medium_runtime_full(ConfigDriftRuntime *rt) {
    memset(rt, 0, sizeof(*rt));
    rt->model                 = "medium";
    rt->tokenizer_path        = "data/training/tokenizer_32k.bin";
    rt->vocab_size            = 32768;
    rt->param_count_total     = 62988800;
    rt->param_count_active    = 34677248;
    rt->has_param_count_active = true;
    /* Medium arm pinned values. */
    rt->lr_base            = 0.001;
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
    rt->weight_seed        = 42;
}

/* Establish the LAB_PROGRAM env so config_drift_assert_or_die loads the real
 * Program 2 manifest from the repo. LAB_PROGRAM_ROOT defaults to "programs"
 * which is the working-tree relative path the test harness runs from. */
static void enter_program_2_env(void) {
    setenv("LAB_PROGRAM", "program_2_dense_vs_moe_sub100m", 1);
    /* LAB_PROGRAM_ROOT intentionally NOT set: defaults to "programs" =
     * relative path that resolves at test cwd (repo root via `make test`). */
    unsetenv("LAB_PROGRAM_ROOT");
}

/* ---- Test 1: manifest parses cleanly with all 14 D-181 fields ---- */

static void test_manifest_parses_with_lr_pin(void) {
    const char *path =
        "programs/program_2_dense_vs_moe_sub100m/spec_invariants.yaml";

    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    int rc = config_drift_load(path, &mf);
    ASSERT_EQUAL_INT(0, rc);
    /* D-194 schema transition: 4 cell-arm blocks (phase3_cell_A..D) + 2
     * preserved legacy arms (dense50m, medium) = 6 total. The legacy arms
     * retain ALL D-181 14 fields (including weight_seed=42) for Phase-2
     * backward-compat; the cell-aware arms add a `cell_id` discriminator.
     * config_drift_find_arm("dense50m") explicitly skips cell-aware arms
     * (D-194 lookup discipline), so this test continues to verify the
     * legacy dense50m arm's D-181 attestation as before. */
    ASSERT_EQUAL_INT(6, mf.n_arms);

    /* Dense-A structural fields (legacy arm — find_arm skips cell-aware). */
    const ConfigDriftArm *d = config_drift_find_arm(&mf, "dense50m");
    ASSERT_NOT_NULL(d);
    ASSERT_EQUAL_STR("dense50m", d->model);
    ASSERT_EQUAL_STR("tokenizer_32k.bin", d->tokenizer_basename);
    ASSERT_EQUAL_INT(32768, d->vocab_size);
    ASSERT_EQUAL_INT(34619904, d->param_count_total);

    /* D-182: 14 LR/dynamics fields populated for dense50m. */
    ASSERT_TRUE(d->has_lr_base);
    ASSERT_EQUAL_FLOAT(0.002, d->lr_base, 1e-9);
    ASSERT_TRUE(d->has_lr_schedule);
    ASSERT_EQUAL_STR("linear_warmup_cosine_decay", d->lr_schedule);
    ASSERT_TRUE(d->has_lr_warmup_steps);
    ASSERT_EQUAL_INT(200, d->lr_warmup_steps);
    ASSERT_TRUE(d->has_lr_min);
    ASSERT_EQUAL_FLOAT(1e-05, d->lr_min, 1e-12);
    ASSERT_TRUE(d->has_optimizer);
    ASSERT_EQUAL_STR("adamw", d->optimizer);
    ASSERT_TRUE(d->has_adam_beta1);
    ASSERT_EQUAL_FLOAT(0.9, d->adam_beta1, 1e-9);
    ASSERT_TRUE(d->has_adam_beta2);
    ASSERT_EQUAL_FLOAT(0.999, d->adam_beta2, 1e-9);
    ASSERT_TRUE(d->has_weight_decay);
    ASSERT_EQUAL_FLOAT(0.01, d->weight_decay, 1e-9);
    ASSERT_TRUE(d->has_grad_clip_norm);
    ASSERT_EQUAL_FLOAT(1.0, d->grad_clip_norm, 1e-9);
    ASSERT_TRUE(d->has_grad_accum_steps);
    ASSERT_EQUAL_INT(4, d->grad_accum_steps);
    ASSERT_TRUE(d->has_batch_size);
    ASSERT_EQUAL_INT(1, d->batch_size);
    ASSERT_TRUE(d->has_seq_len);
    ASSERT_EQUAL_INT(512, d->seq_len);
    ASSERT_TRUE(d->has_max_steps);
    ASSERT_EQUAL_INT(5000, d->max_steps);
    ASSERT_TRUE(d->has_weight_seed);
    ASSERT_EQUAL_INT(42, d->weight_seed);

    /* MoE arm structural + the key 2× LR differential. */
    const ConfigDriftArm *m = config_drift_find_arm(&mf, "medium");
    ASSERT_NOT_NULL(m);
    ASSERT_EQUAL_STR("medium", m->model);
    ASSERT_EQUAL_STR("tokenizer_32k.bin", m->tokenizer_basename);
    ASSERT_EQUAL_INT(32768, m->vocab_size);
    ASSERT_EQUAL_INT(62988800, m->param_count_total);
    ASSERT_TRUE(m->has_param_count_active);
    ASSERT_EQUAL_INT(34677248, m->param_count_active);
    ASSERT_TRUE(m->has_lr_base);
    ASSERT_EQUAL_FLOAT(0.001, m->lr_base, 1e-9);  /* THE D-181 confound */
    ASSERT_TRUE(m->has_optimizer);
    ASSERT_EQUAL_STR("adamw", m->optimizer);
    ASSERT_TRUE(m->has_weight_seed);
    ASSERT_EQUAL_INT(42, m->weight_seed);

    config_drift_manifest_free(&mf);
}

/* ---- Test 2: manifest LR values literally match the D-181 pin ---- */

static char *slurp_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return NULL; }
    rewind(f);
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[r] = '\0';
    if (len_out) *len_out = r;
    return buf;
}

static const char *section_of(const char *text, const char *arm_name) {
    char marker[128];
    snprintf(marker, sizeof(marker), "arm = %s", arm_name);
    return strstr(text, marker);
}

static const char *next_section(const char *from) {
    const char *next = strstr(from + 1, "\narm = ");
    if (next) return next;
    return from + strlen(from);
}

static bool contains_in_range(const char *start, const char *end,
                              const char *needle) {
    size_t n = strlen(needle);
    for (const char *p = start; p + n <= end; p++) {
        if (memcmp(p, needle, n) == 0) return true;
    }
    return false;
}

static void test_manifest_lr_values_expected(void) {
    const char *path =
        "programs/program_2_dense_vs_moe_sub100m/spec_invariants.yaml";
    size_t len = 0;
    char *text = slurp_file(path, &len);
    ASSERT_NOT_NULL(text);

    const char *d_start = section_of(text, "dense50m");
    ASSERT_NOT_NULL(d_start);
    const char *d_end = next_section(d_start);
    ASSERT_TRUE(contains_in_range(d_start, d_end, "lr_base = 0.002"));
    ASSERT_TRUE(contains_in_range(d_start, d_end,
                                  "lr_schedule = linear_warmup_cosine_decay"));
    ASSERT_TRUE(contains_in_range(d_start, d_end, "optimizer = adamw"));
    ASSERT_TRUE(contains_in_range(d_start, d_end, "weight_seed = 42"));

    const char *m_start = section_of(text, "medium");
    ASSERT_NOT_NULL(m_start);
    const char *m_end = next_section(m_start);
    ASSERT_TRUE(contains_in_range(m_start, m_end, "lr_base = 0.001"));
    ASSERT_TRUE(contains_in_range(m_start, m_end,
                                  "lr_schedule = linear_warmup_cosine_decay"));
    ASSERT_TRUE(contains_in_range(m_start, m_end, "optimizer = adamw"));
    ASSERT_TRUE(contains_in_range(m_start, m_end, "weight_seed = 42"));

    ASSERT_TRUE(strstr(text, "D-181 EXTENSION") != NULL);
    ASSERT_TRUE(strstr(text, "PI-binding condition #1") != NULL);

    free(text);
}

/* ---- Mismatch test scaffolding ----
 *
 * Each mismatch test follows the same pattern: copy the canonical full-match
 * runtime for an arm, mutate ONE field, run config_drift_assert_or_die in a
 * subprocess, expect rc=1 + stderr containing "<arm> <field>" so the FATAL
 * line is unambiguous on the launch console. */

typedef struct {
    bool                use_dense;       /* true=dense50m, false=medium */
    /* Field-mutation closure: applied to a fresh full-match runtime. */
    void (*mutate)(ConfigDriftRuntime *rt);
    const char *expected_arm;
    const char *expected_field;
} MismatchCase;

static int child_run_assert(void *ctx) {
    const MismatchCase *c = (const MismatchCase *)ctx;
    enter_program_2_env();
    ConfigDriftRuntime rt;
    if (c->use_dense) make_dense_runtime_full(&rt);
    else              make_medium_runtime_full(&rt);
    if (c->mutate) c->mutate(&rt);
    config_drift_assert_or_die(&rt);  /* should NOT return on mismatch */
    return 0;  /* full-match path */
}

/* Run the case in a subprocess, expect rc=1 (exit on FATAL) and stderr
 * contains "CONFIG-DRIFT FATAL: <arm> <field>". */
static void expect_mismatch(const MismatchCase *c) {
    char err[4096];
    int rc = run_in_subprocess(child_run_assert, (void *)c, err, sizeof(err));
    if (rc != 1) {
        printf("\n    [DEBUG] expected rc=1, got rc=%d; stderr was:\n%s\n",
               rc, err);
    }
    ASSERT_EQUAL_INT(1, rc);
    char needle[256];
    snprintf(needle, sizeof(needle),
             "CONFIG-DRIFT FATAL: %s %s",
             c->expected_arm, c->expected_field);
    if (!strstr(err, needle)) {
        printf("\n    [DEBUG] needle not found: %s\n    full stderr:\n%s\n",
               needle, err);
    }
    ASSERT_TRUE(strstr(err, needle) != NULL);
}

/* Run the case in a subprocess, expect rc=0 (silent success). */
static void expect_match(const MismatchCase *c) {
    char err[4096];
    int rc = run_in_subprocess(child_run_assert, (void *)c, err, sizeof(err));
    if (rc != 0) {
        printf("\n    [DEBUG] expected rc=0, got rc=%d; stderr was:\n%s\n",
               rc, err);
    }
    ASSERT_EQUAL_INT(0, rc);
}

/* ---- Field-mutation closures (one per field) ---- */

static void mutate_lr_base(ConfigDriftRuntime *rt)         { rt->lr_base = 0.0009; }
static void mutate_lr_schedule(ConfigDriftRuntime *rt)     { rt->lr_schedule = "constant"; }
static void mutate_lr_warmup_steps(ConfigDriftRuntime *rt) { rt->lr_warmup_steps = 100; }
static void mutate_lr_min(ConfigDriftRuntime *rt)          { rt->lr_min = 1e-04; }
static void mutate_optimizer(ConfigDriftRuntime *rt)       { rt->optimizer = "sgd"; }
static void mutate_adam_beta1(ConfigDriftRuntime *rt)      { rt->adam_beta1 = 0.95; }
static void mutate_adam_beta2(ConfigDriftRuntime *rt)      { rt->adam_beta2 = 0.998; }
static void mutate_weight_decay(ConfigDriftRuntime *rt)    { rt->weight_decay = 0.1; }
static void mutate_grad_clip_norm(ConfigDriftRuntime *rt)  { rt->grad_clip_norm = 0.5; }
static void mutate_grad_accum_steps(ConfigDriftRuntime *rt){ rt->grad_accum_steps = 8; }
static void mutate_batch_size(ConfigDriftRuntime *rt)      { rt->batch_size = 2; }
static void mutate_seq_len(ConfigDriftRuntime *rt)         { rt->seq_len = 1024; }
static void mutate_max_steps(ConfigDriftRuntime *rt)       { rt->max_steps = 4999; }
static void mutate_weight_seed(ConfigDriftRuntime *rt)     { rt->weight_seed = 43; }

/* ---- Test 3.x: per-field mismatch FATALs (one per D-181 field) ---- */

/* The D-181 anchor test (un-skipped; was placeholder pre-D-182). */
static void test_runtime_lr_mismatch_reserved(void) {
    /* Anchor case: dense50m arm, lr_base manifest=0.002, runtime=0.001. The
     * historical Phase-2 confound shape. */
    MismatchCase c = {
        .use_dense = true,
        .mutate = mutate_lr_base,
        .expected_arm = "dense50m",
        .expected_field = "lr_base",
    };
    expect_mismatch(&c);
}

/* Below: one rc=1 detection test per field. Dense50m arm chosen as the carrier
 * since most fields are identical between arms. lr_base also tested on medium
 * arm (next test) since the inter-arm differential is the D-181 narrative. */

#define MISMATCH_TEST_DENSE(field)                                              \
    static void test_runtime_##field##_mismatch_dense(void) {                   \
        MismatchCase c = {                                                      \
            .use_dense = true,                                                  \
            .mutate = mutate_##field,                                           \
            .expected_arm = "dense50m",                                         \
            .expected_field = #field,                                           \
        };                                                                      \
        expect_mismatch(&c);                                                    \
    }

MISMATCH_TEST_DENSE(lr_schedule)
MISMATCH_TEST_DENSE(lr_warmup_steps)
MISMATCH_TEST_DENSE(lr_min)
MISMATCH_TEST_DENSE(optimizer)
MISMATCH_TEST_DENSE(adam_beta1)
MISMATCH_TEST_DENSE(adam_beta2)
MISMATCH_TEST_DENSE(weight_decay)
MISMATCH_TEST_DENSE(grad_clip_norm)
MISMATCH_TEST_DENSE(grad_accum_steps)
MISMATCH_TEST_DENSE(batch_size)
MISMATCH_TEST_DENSE(seq_len)
MISMATCH_TEST_DENSE(max_steps)
MISMATCH_TEST_DENSE(weight_seed)

/* lr_base on medium arm (manifest=0.001, runtime mutated to 0.0009). This is
 * the second test the dispatch called out — confirming both arms exercise the
 * critical LR comparison. */
static void test_runtime_lr_base_mismatch_medium(void) {
    MismatchCase c = {
        .use_dense = false,
        .mutate = mutate_lr_base,
        .expected_arm = "medium",
        .expected_field = "lr_base",
    };
    expect_mismatch(&c);
}

/* ---- Test 4: full-match returns silently, both arms ---- */

static void test_runtime_full_match_dense(void) {
    MismatchCase c = {
        .use_dense = true,
        .mutate = NULL,
        .expected_arm = "dense50m",
        .expected_field = "(none)",
    };
    expect_match(&c);
}

static void test_runtime_full_match_medium(void) {
    MismatchCase c = {
        .use_dense = false,
        .mutate = NULL,
        .expected_arm = "medium",
        .expected_field = "(none)",
    };
    expect_match(&c);
}

/* ---- Test 5: legacy-manifest defense-in-depth ----
 *
 * Construct a private programs/<test-prog>/spec_invariants.yaml that has only
 * the structural fields (no LR / training-dynamics keys). Confirm parse OK
 * and that config_drift_assert_or_die returns silently when a runtime that
 * IS populated with LR/dynamics values is checked against it. This mirrors
 * the Program 1 / older-program case where the manifest predates D-181 and
 * has no expectation about training dynamics — the comparator must not
 * spuriously FATAL. */

static int child_run_legacy_assert(void *ctx) {
    const char *root = (const char *)ctx;
    setenv("LAB_PROGRAM", "legacy_no_lr_pin", 1);
    setenv("LAB_PROGRAM_ROOT", root, 1);
    ConfigDriftRuntime rt;
    make_dense_runtime_full(&rt);  /* runtime IS fully populated */
    /* Override the structural side to match the synthetic legacy manifest. */
    rt.model = "legacy_dense";
    rt.vocab_size = 1024;
    rt.param_count_total = 1234567;
    rt.has_param_count_active = false;
    config_drift_assert_or_die(&rt);
    return 0;
}

static void test_legacy_manifest_no_lr_pin_no_fatal(void) {
    const char *root = "/tmp/cd_legacy_test";
    const char *manifest =
        "/tmp/cd_legacy_test/legacy_no_lr_pin/spec_invariants.yaml";
    const char *content =
        "# Synthetic legacy manifest — only structural fields, no D-181 LR pin.\n"
        "arm = legacy_dense\n"
        "model = legacy_dense\n"
        "tokenizer_basename = tokenizer_32k.bin\n"
        "vocab_size = 1024\n"
        "param_count_total = 1234567\n";
    ASSERT_EQUAL_INT(0, write_file(manifest, content));

    /* Verify parse succeeds and zero LR fields are populated. */
    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    ASSERT_EQUAL_INT(0, config_drift_load(manifest, &mf));
    ASSERT_EQUAL_INT(1, mf.n_arms);
    const ConfigDriftArm *a = config_drift_find_arm(&mf, "legacy_dense");
    ASSERT_NOT_NULL(a);
    ASSERT_FALSE(a->has_lr_base);
    ASSERT_FALSE(a->has_lr_schedule);
    ASSERT_FALSE(a->has_optimizer);
    ASSERT_FALSE(a->has_weight_seed);
    config_drift_manifest_free(&mf);

    /* Verify assertion path returns silently — runtime LR values must NOT
     * trigger a FATAL when the manifest has no expectation. */
    char err[4096];
    int rc = run_in_subprocess(child_run_legacy_assert, (void *)root,
                               err, sizeof(err));
    if (rc != 0) {
        printf("\n    [DEBUG] expected rc=0 (legacy passthrough), got rc=%d; "
               "stderr:\n%s\n", rc, err);
    }
    ASSERT_EQUAL_INT(0, rc);

    /* Cleanup. */
    rm_file(manifest);
    rmdir("/tmp/cd_legacy_test/legacy_no_lr_pin");
    rmdir(root);
}

int main(void) {
    printf("=== config_drift LR-pin tests (D-181 manifest + D-182 wiring) ===\n");

    /* Test 1 + 2: parser side. */
    RUN_TEST(test_manifest_parses_with_lr_pin);
    RUN_TEST(test_manifest_lr_values_expected);

    /* Test 3.x: per-field mismatch FATALs (15 cases — 14 fields on dense + 1
     * on medium for lr_base inter-arm coverage). */
    RUN_TEST(test_runtime_lr_mismatch_reserved);     /* anchor: dense lr_base */
    RUN_TEST(test_runtime_lr_base_mismatch_medium);  /* MoE arm lr_base */
    RUN_TEST(test_runtime_lr_schedule_mismatch_dense);
    RUN_TEST(test_runtime_lr_warmup_steps_mismatch_dense);
    RUN_TEST(test_runtime_lr_min_mismatch_dense);
    RUN_TEST(test_runtime_optimizer_mismatch_dense);
    RUN_TEST(test_runtime_adam_beta1_mismatch_dense);
    RUN_TEST(test_runtime_adam_beta2_mismatch_dense);
    RUN_TEST(test_runtime_weight_decay_mismatch_dense);
    RUN_TEST(test_runtime_grad_clip_norm_mismatch_dense);
    RUN_TEST(test_runtime_grad_accum_steps_mismatch_dense);
    RUN_TEST(test_runtime_batch_size_mismatch_dense);
    RUN_TEST(test_runtime_seq_len_mismatch_dense);
    RUN_TEST(test_runtime_max_steps_mismatch_dense);
    RUN_TEST(test_runtime_weight_seed_mismatch_dense);

    /* Test 4: full-match silent success. */
    RUN_TEST(test_runtime_full_match_dense);
    RUN_TEST(test_runtime_full_match_medium);

    /* Test 5: legacy-manifest defense-in-depth. */
    RUN_TEST(test_legacy_manifest_no_lr_pin_no_fatal);

    TEST_REPORT();
}
