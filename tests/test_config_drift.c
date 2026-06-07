/* test_config_drift.c -- Tests for the runtime config-drift assertion (P-CONFIG-DRIFT-DETECTION,
 * D-121, Item 6 of P-PHASE2-REDO).
 *
 * The drift detector compares the *actual* runtime config (model name, tokenizer file,
 * vocab_size, param counts) against a frozen per-program manifest. Activation is opt-in
 * via the LAB_PROGRAM env var:
 *   LAB_PROGRAM=program_2_example  → look for programs/<that>/spec_invariants.yaml
 *   LAB_PROGRAM unset                            → assertion is a no-op
 *
 * On mismatch the assertion prints "CONFIG-DRIFT FATAL: expected X, got Y" and calls
 * exit(1) BEFORE any training step or checkpoint write. This is belt+suspenders against
 * the D-119–D-121 Swiss-cheese class of failure where the launch template silently
 * relied on a binary default tokenizer.
 *
 * Tests:
 *   1. test_parse_valid_manifest      -- key=value parser round-trips a known-good file
 *   2. test_assert_missing_manifest   -- LAB_PROGRAM set + missing file → fatal
 *   3. test_assert_mismatch           -- runtime vocab_size != manifest → fatal w/ message
 *   4. test_assert_match              -- runtime matches manifest → silent PASS
 *   5. test_assert_env_unset          -- LAB_PROGRAM unset → no-op (no error even if missing)
 *
 * The "fatal" tests use fork() + waitpid() to capture the exit code and stderr without
 * killing the test harness itself, since the assertion's contract is exit(1).
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

/* ---- Test fixture helpers ---- */

/* mkdir -p equivalent: create all intermediate dirs in `path` (path is treated
 * as a directory, not a file). Returns 0 on success, -1 only if a non-EEXIST
 * mkdir failure occurs. */
static int mkdir_p(const char *path) {
    char tmp[1024];
    size_t len = strlen(path);
    if (len == 0 || len >= sizeof(tmp)) return -1;
    memcpy(tmp, path, len + 1);
    /* Strip trailing slash (if any) so the loop's terminating mkdir runs. */
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

/* Write a string to a path, creating intermediate dirs as needed.
 * Returns 0 on success, -1 on failure. */
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

/* Remove a file (best-effort). */
static void rm_file(const char *path) {
    unlink(path);
}

/* Fork a child, run the body, capture exit code + stderr. Returns child exit code
 * (0 = clean, 1 = exit(1), >128 = signal). stderr_out is NUL-terminated; cap is
 * the buffer size. */
typedef int (*child_fn_t)(void *ctx);

static int run_in_subprocess(child_fn_t body, void *ctx,
                             char *stderr_out, size_t stderr_cap) {
    int pipefd[2];
    if (pipe(pipefd) != 0) {
        return -1;
    }
    /* Flush parent's stdio buffers BEFORE forking so the child does not inherit
     * pending parent output (would cause duplicate test-banner lines on _exit). */
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        close(pipefd[0]); close(pipefd[1]);
        return -1;
    }
    if (pid == 0) {
        /* Child: redirect stderr to pipe, run body, exit with body's return.
         * Also redirect stdout to /dev/null so the child cannot print into the
         * test harness's stdout (e.g. via printf in any future code path). */
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
    /* Parent: drain pipe, then waitpid. */
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

/* ---- Test 1: parser correctly loads a known-good manifest ---- */

static void test_parse_valid_manifest(void) {
    const char *path = "/tmp/cd_test_valid.yaml";
    /* Two-arm manifest: dense50m + medium. Comments + blank lines exercised. */
    const char *content =
        "# Test manifest for config_drift parser\n"
        "\n"
        "arm = dense50m\n"
        "model = dense50m\n"
        "tokenizer_basename = tokenizer_32k.bin\n"
        "vocab_size = 32768\n"
        "param_count_total = 34619904\n"
        "\n"
        "arm = medium\n"
        "model = medium\n"
        "tokenizer_basename = tokenizer_32k.bin\n"
        "vocab_size = 32768\n"
        "param_count_total = 62994944\n"
        "param_count_active = 34619904\n";
    ASSERT_EQUAL_INT(0, write_file(path, content));

    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    int rc = config_drift_load(path, &mf);
    ASSERT_EQUAL_INT(0, rc);
    ASSERT_EQUAL_INT(2, mf.n_arms);

    /* Look up dense50m. */
    const ConfigDriftArm *a = config_drift_find_arm(&mf, "dense50m");
    ASSERT_NOT_NULL(a);
    ASSERT_EQUAL_STR("dense50m", a->model);
    ASSERT_EQUAL_STR("tokenizer_32k.bin", a->tokenizer_basename);
    ASSERT_EQUAL_INT(32768, a->vocab_size);
    ASSERT_EQUAL_INT(34619904, a->param_count_total);
    ASSERT_FALSE(a->has_param_count_active);  /* Dense: optional field absent */

    /* Look up medium. */
    const ConfigDriftArm *b = config_drift_find_arm(&mf, "medium");
    ASSERT_NOT_NULL(b);
    ASSERT_EQUAL_STR("medium", b->model);
    ASSERT_EQUAL_STR("tokenizer_32k.bin", b->tokenizer_basename);
    ASSERT_EQUAL_INT(32768, b->vocab_size);
    ASSERT_EQUAL_INT(62994944, b->param_count_total);
    ASSERT_TRUE(b->has_param_count_active);
    ASSERT_EQUAL_INT(34619904, b->param_count_active);

    /* Unknown arm → NULL. */
    ASSERT_NULL(config_drift_find_arm(&mf, "100m"));

    config_drift_manifest_free(&mf);
    rm_file(path);
}

/* ---- Test 2: LAB_PROGRAM set + manifest missing → fatal exit(1) ---- */

typedef struct {
    const char *program;
    const char *model;
    const char *tokenizer_path;
    int32_t vocab_size;
    int64_t param_count_total;
    int64_t param_count_active;
    bool has_param_count_active;
} AssertCtx;

static int child_assert(void *ctx_v) {
    AssertCtx *c = (AssertCtx *)ctx_v;
    ConfigDriftRuntime rt = {
        .model              = c->model,
        .tokenizer_path     = c->tokenizer_path,
        .vocab_size         = c->vocab_size,
        .param_count_total  = c->param_count_total,
        .param_count_active = c->param_count_active,
        .has_param_count_active = c->has_param_count_active,
    };
    if (c->program) {
        setenv("LAB_PROGRAM", c->program, 1);
    } else {
        unsetenv("LAB_PROGRAM");
    }
    /* Override the program-root for tests so we don't touch the real tree.
     * The implementation honors LAB_PROGRAM_ROOT for test isolation. */
    setenv("LAB_PROGRAM_ROOT", "/tmp/cd_test_progs", 1);
    /* This call will exit(1) on mismatch / missing manifest, return on success. */
    config_drift_assert_or_die(&rt);
    return 0;  /* assertion passed silently */
}

static void test_assert_missing_manifest(void) {
    /* Ensure the manifest path does NOT exist. */
    rm_file("/tmp/cd_test_progs/program_test_missing/spec_invariants.yaml");
    rmdir("/tmp/cd_test_progs/program_test_missing");

    AssertCtx ctx = {
        .program = "program_test_missing",
        .model = "dense50m",
        .tokenizer_path = "data/training/tokenizer_32k.bin",
        .vocab_size = 32768,
        .param_count_total = 34619904,
        .has_param_count_active = false,
    };
    char err[2048];
    int rc = run_in_subprocess(child_assert, &ctx, err, sizeof(err));
    ASSERT_EQUAL_INT(1, rc);
    /* Error message must mention CONFIG-DRIFT FATAL and the missing manifest. */
    ASSERT_TRUE(strstr(err, "CONFIG-DRIFT FATAL") != NULL);
    ASSERT_TRUE(strstr(err, "spec_invariants") != NULL);
}

/* ---- Test 3: deliberate mismatch (vocab_size) → fatal exit(1) ---- */

static void test_assert_mismatch(void) {
    const char *prog = "program_test_mismatch";
    char path[1024];
    snprintf(path, sizeof(path),
             "/tmp/cd_test_progs/%s/spec_invariants.yaml", prog);
    const char *content =
        "arm = dense50m\n"
        "model = dense50m\n"
        "tokenizer_basename = tokenizer_32k.bin\n"
        "vocab_size = 32768\n"
        "param_count_total = 34619904\n";
    ASSERT_EQUAL_INT(0, write_file(path, content));

    AssertCtx ctx = {
        .program = prog,
        .model = "dense50m",
        .tokenizer_path = "data/training/tokenizer_4k.bin",  /* WRONG */
        .vocab_size = 4096,                                   /* WRONG */
        .param_count_total = 34619904,
        .has_param_count_active = false,
    };
    char err[2048];
    int rc = run_in_subprocess(child_assert, &ctx, err, sizeof(err));
    ASSERT_EQUAL_INT(1, rc);
    /* Must contain the "expected X, got Y" formula. */
    ASSERT_TRUE(strstr(err, "CONFIG-DRIFT FATAL") != NULL);
    ASSERT_TRUE(strstr(err, "expected") != NULL);
    ASSERT_TRUE(strstr(err, "got") != NULL);
    /* And the specific values. */
    ASSERT_TRUE(strstr(err, "32768") != NULL);
    ASSERT_TRUE(strstr(err, "4096") != NULL);

    rm_file(path);
}

/* ---- Test 4: correct match → silent PASS, exit(0) ---- */

static void test_assert_match(void) {
    const char *prog = "program_test_match";
    char path[1024];
    snprintf(path, sizeof(path),
             "/tmp/cd_test_progs/%s/spec_invariants.yaml", prog);
    const char *content =
        "arm = dense50m\n"
        "model = dense50m\n"
        "tokenizer_basename = tokenizer_32k.bin\n"
        "vocab_size = 32768\n"
        "param_count_total = 34619904\n";
    ASSERT_EQUAL_INT(0, write_file(path, content));

    AssertCtx ctx = {
        .program = prog,
        .model = "dense50m",
        .tokenizer_path = "data/training/tokenizer_32k.bin",
        .vocab_size = 32768,
        .param_count_total = 34619904,
        .has_param_count_active = false,
    };
    char err[2048];
    int rc = run_in_subprocess(child_assert, &ctx, err, sizeof(err));
    ASSERT_EQUAL_INT(0, rc);
    /* No FATAL on stderr (informational stderr OK; FATAL is the failure mode). */
    ASSERT_TRUE(strstr(err, "CONFIG-DRIFT FATAL") == NULL);

    rm_file(path);
}

/* ---- Test 5: LAB_PROGRAM unset → assertion is a no-op ---- */

static void test_assert_env_unset(void) {
    AssertCtx ctx = {
        .program = NULL,  /* unset */
        .model = "dense50m",
        .tokenizer_path = "data/training/tokenizer_4k.bin",  /* would be wrong */
        .vocab_size = 4096,                                   /* would be wrong */
        .param_count_total = 999999,                          /* would be wrong */
        .has_param_count_active = false,
    };
    char err[2048];
    int rc = run_in_subprocess(child_assert, &ctx, err, sizeof(err));
    /* No env var → no-op → exit(0) regardless of "wrong" values. */
    ASSERT_EQUAL_INT(0, rc);
    ASSERT_TRUE(strstr(err, "CONFIG-DRIFT FATAL") == NULL);
}

/* ---- Bonus test: tokenizer_basename mismatch (covers tokenizer-path drift,
 * which is the exact D-119–D-121 root cause). ---- */

static void test_assert_tokenizer_basename_mismatch(void) {
    const char *prog = "program_test_tok_mismatch";
    char path[1024];
    snprintf(path, sizeof(path),
             "/tmp/cd_test_progs/%s/spec_invariants.yaml", prog);
    const char *content =
        "arm = dense50m\n"
        "model = dense50m\n"
        "tokenizer_basename = tokenizer_32k.bin\n"
        "vocab_size = 32768\n"
        "param_count_total = 34619904\n";
    ASSERT_EQUAL_INT(0, write_file(path, content));

    AssertCtx ctx = {
        .program = prog,
        .model = "dense50m",
        .tokenizer_path = "data/training/tokenizer_4k.bin",  /* WRONG basename */
        .vocab_size = 32768,                                  /* matches */
        .param_count_total = 34619904,
        .has_param_count_active = false,
    };
    char err[2048];
    int rc = run_in_subprocess(child_assert, &ctx, err, sizeof(err));
    ASSERT_EQUAL_INT(1, rc);
    ASSERT_TRUE(strstr(err, "CONFIG-DRIFT FATAL") != NULL);
    ASSERT_TRUE(strstr(err, "tokenizer") != NULL);
    ASSERT_TRUE(strstr(err, "tokenizer_32k.bin") != NULL);
    ASSERT_TRUE(strstr(err, "tokenizer_4k.bin") != NULL);

    rm_file(path);
}

int main(void) {
    printf("=== config_drift assertion tests ===\n");
    RUN_TEST(test_parse_valid_manifest);
    RUN_TEST(test_assert_missing_manifest);
    RUN_TEST(test_assert_mismatch);
    RUN_TEST(test_assert_match);
    RUN_TEST(test_assert_env_unset);
    RUN_TEST(test_assert_tokenizer_basename_mismatch);
    TEST_REPORT();
}
