/* config_drift.c -- Implementation of the per-program runtime config-drift assertion.
 * See config_drift.h for activation rules, manifest format, and failure semantics.
 *
 * Design notes:
 *   - No external deps: hand-written line-based parser. The `.yaml` extension is a
 *     convention for human readability; the format is intentionally a strict subset
 *     of YAML (flat scalar key=value pairs grouped by `arm = <name>` headers) so
 *     YAML readers can parse it but no libyaml/PyYAML dep is added.
 *   - Strings in ConfigDriftArm are inline char[] (not heap) so manifest_free is
 *     trivial today. Heap-backed extras (e.g. unknown-field warnings) can be added
 *     without breaking the API.
 *   - basename extraction for tokenizer paths: strrchr('/'). On platforms with
 *     backslash separators we'd extend; lab is macOS-only per CLAUDE.md.
 */

#include "config_drift.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- String helpers ---- */

/* In-place trim of leading + trailing whitespace. Returns the (possibly advanced)
 * pointer to the first non-space character; the trailing whitespace is overwritten
 * with NUL. */
static char *str_trim(char *s) {
    if (!s) return s;
    while (*s && isspace((unsigned char)*s)) s++;
    if (!*s) return s;
    char *end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end)) {
        *end = '\0';
        end--;
    }
    return s;
}

/* Return basename(path) — the portion after the last '/'. If no '/', returns path. */
static const char *path_basename(const char *path) {
    if (!path) return "";
    const char *slash = strrchr(path, '/');
    return slash ? (slash + 1) : path;
}

/* Copy `src` into `dst` with bounded length, always NUL-terminating. */
static void str_bounded_copy(char *dst, size_t cap, const char *src) {
    if (cap == 0) return;
    if (!src) {
        dst[0] = '\0';
        return;
    }
    size_t n = strlen(src);
    if (n >= cap) n = cap - 1;
    memcpy(dst, src, n);
    dst[n] = '\0';
}

/* ---- Public: manifest lifecycle ---- */

void config_drift_manifest_init(ConfigDriftManifest *mf) {
    if (!mf) return;
    memset(mf, 0, sizeof(*mf));
}

void config_drift_manifest_free(ConfigDriftManifest *mf) {
    /* No heap state today; reserved for future use. Re-init for safety. */
    if (!mf) return;
    memset(mf, 0, sizeof(*mf));
}

/* ---- Public: load + lookup ---- */

/* ---- D-182 typed-scalar parse helpers (P-CONFIG-DRIFT-LR-WIRE) ----
 *
 * Centralized so the 14 new fields all parse with identical error semantics:
 * empty value → FATAL, range/format error → FATAL with file:lineno:key.
 * `errno = 0` is set BEFORE strtoX so that ERANGE on overflow is detectable
 * (strtoX leaves errno alone on success, which would otherwise carry stale
 * state from earlier syscalls).
 */

/* Parse a strict positive double. errno-checked. `*end` must point at NUL.
 * Returns 0 on success and writes to *out, -1 on any parse failure. */
static int parse_double_strict(const char *value, double *out) {
    if (!value || !*value || !out) return -1;
    errno = 0;
    char *end = NULL;
    double v = strtod(value, &end);
    if (errno != 0 || !end || *end != '\0') return -1;
    *out = v;
    return 0;
}

/* Parse an int32 in range [INT32_MIN, INT32_MAX]. */
static int parse_int32_strict(const char *value, int32_t *out) {
    if (!value || !*value || !out) return -1;
    errno = 0;
    char *end = NULL;
    long v = strtol(value, &end, 10);
    if (errno != 0 || !end || *end != '\0') return -1;
    if (v < INT32_MIN || v > INT32_MAX) return -1;
    *out = (int32_t)v;
    return 0;
}

/* Parse a single key=value line into the current arm. `current` may be NULL if
 * we haven't seen `arm = ...` yet. Returns 0 on success, non-zero on parse error. */
static int parse_kv_line(const char *path, int lineno,
                         const char *key, const char *value,
                         ConfigDriftManifest *mf, ConfigDriftArm **current) {
    /* `arm = <name>` always starts a new section. */
    if (strcmp(key, "arm") == 0) {
        if (mf->n_arms >= CONFIG_DRIFT_MAX_ARMS) {
            fprintf(stderr,
                    "config_drift: %s:%d: too many arms (max %d)\n",
                    path, lineno, CONFIG_DRIFT_MAX_ARMS);
            return -1;
        }
        *current = &mf->arms[mf->n_arms++];
        memset(*current, 0, sizeof(**current));
        str_bounded_copy((*current)->arm_name, CONFIG_DRIFT_STR_MAX, value);
        return 0;
    }

    /* Every non-`arm` key requires a current section. */
    if (!*current) {
        fprintf(stderr,
                "config_drift: %s:%d: key '%s' before any 'arm = ...' header\n",
                path, lineno, key);
        return -1;
    }
    ConfigDriftArm *a = *current;

    if (strcmp(key, "model") == 0) {
        str_bounded_copy(a->model, CONFIG_DRIFT_STR_MAX, value);
    } else if (strcmp(key, "tokenizer_basename") == 0) {
        str_bounded_copy(a->tokenizer_basename, CONFIG_DRIFT_STR_MAX, value);
    } else if (strcmp(key, "vocab_size") == 0) {
        char *end = NULL;
        long v = strtol(value, &end, 10);
        if (!end || *end != '\0' || v <= 0 || v > INT32_MAX) {
            fprintf(stderr, "config_drift: %s:%d: invalid vocab_size '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->vocab_size = (int32_t)v;
    } else if (strcmp(key, "param_count_total") == 0) {
        char *end = NULL;
        long long v = strtoll(value, &end, 10);
        if (!end || *end != '\0' || v <= 0) {
            fprintf(stderr,
                    "config_drift: %s:%d: invalid param_count_total '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->param_count_total = (int64_t)v;
    } else if (strcmp(key, "param_count_active") == 0) {
        char *end = NULL;
        long long v = strtoll(value, &end, 10);
        if (!end || *end != '\0' || v <= 0) {
            fprintf(stderr,
                    "config_drift: %s:%d: invalid param_count_active '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->param_count_active = (int64_t)v;
        a->has_param_count_active = true;
    }
    /* ---- D-182 LR/schedule/dynamics fields (P-CONFIG-DRIFT-LR-WIRE) ----
     * These were warn-on-unknown at D-181 (forward-compat hedge in the parser).
     * Phase-3 factorial design (D-190 1i) makes silent LR drift verdict-breaking,
     * so they're now recognized + typed + has_*-flagged for the comparator. */
    else if (strcmp(key, "lr_base") == 0) {
        if (parse_double_strict(value, &a->lr_base) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid lr_base '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_lr_base = true;
    } else if (strcmp(key, "lr_schedule") == 0) {
        /* String fields: bounded copy. If src exceeds CONFIG_DRIFT_STR_MAX-1 chars
         * we FATAL rather than silently truncate (truncation could mask a typo'd
         * schedule name and let the comparator pass on a substring). */
        if (strlen(value) >= CONFIG_DRIFT_STR_MAX) {
            fprintf(stderr,
                    "config_drift: %s:%d: lr_schedule too long (%zu chars, max %d)\n",
                    path, lineno, strlen(value), CONFIG_DRIFT_STR_MAX - 1);
            return -1;
        }
        str_bounded_copy(a->lr_schedule, CONFIG_DRIFT_STR_MAX, value);
        a->has_lr_schedule = true;
    } else if (strcmp(key, "lr_warmup_steps") == 0) {
        if (parse_int32_strict(value, &a->lr_warmup_steps) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid lr_warmup_steps '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_lr_warmup_steps = true;
    } else if (strcmp(key, "lr_min") == 0) {
        if (parse_double_strict(value, &a->lr_min) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid lr_min '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_lr_min = true;
    } else if (strcmp(key, "optimizer") == 0) {
        if (strlen(value) >= CONFIG_DRIFT_STR_MAX) {
            fprintf(stderr,
                    "config_drift: %s:%d: optimizer too long (%zu chars, max %d)\n",
                    path, lineno, strlen(value), CONFIG_DRIFT_STR_MAX - 1);
            return -1;
        }
        str_bounded_copy(a->optimizer, CONFIG_DRIFT_STR_MAX, value);
        a->has_optimizer = true;
    } else if (strcmp(key, "adam_beta1") == 0) {
        if (parse_double_strict(value, &a->adam_beta1) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid adam_beta1 '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_adam_beta1 = true;
    } else if (strcmp(key, "adam_beta2") == 0) {
        if (parse_double_strict(value, &a->adam_beta2) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid adam_beta2 '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_adam_beta2 = true;
    } else if (strcmp(key, "weight_decay") == 0) {
        if (parse_double_strict(value, &a->weight_decay) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid weight_decay '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_weight_decay = true;
    } else if (strcmp(key, "grad_clip_norm") == 0) {
        if (parse_double_strict(value, &a->grad_clip_norm) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid grad_clip_norm '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_grad_clip_norm = true;
    } else if (strcmp(key, "grad_accum_steps") == 0) {
        if (parse_int32_strict(value, &a->grad_accum_steps) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid grad_accum_steps '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_grad_accum_steps = true;
    } else if (strcmp(key, "batch_size") == 0) {
        if (parse_int32_strict(value, &a->batch_size) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid batch_size '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_batch_size = true;
    } else if (strcmp(key, "seq_len") == 0) {
        if (parse_int32_strict(value, &a->seq_len) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid seq_len '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_seq_len = true;
    } else if (strcmp(key, "max_steps") == 0) {
        if (parse_int32_strict(value, &a->max_steps) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid max_steps '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_max_steps = true;
    } else if (strcmp(key, "weight_seed") == 0) {
        if (parse_int32_strict(value, &a->weight_seed) != 0) {
            fprintf(stderr, "config_drift: %s:%d: invalid weight_seed '%s'\n",
                    path, lineno, value);
            return -1;
        }
        a->has_weight_seed = true;
    } else if (strcmp(key, "cell_id") == 0) {
        /* D-194 / P-PHASE3-CELL-ID: NEW key for the Phase-3 factorial.
         * Bounded to 8 bytes (manifest format pins single-character A/B/C/D
         * for Program 2; future programs may use longer cell tokens up to 7
         * chars + NUL). Empty value is invalid. Truncation is FATAL rather
         * than silent (a typo'd cell_id silently truncated could let the
         * lookup pass on a substring, missing the factorial cell selection).
         */
        if (!value || !*value) {
            fprintf(stderr,
                    "config_drift: %s:%d: empty cell_id value\n", path, lineno);
            return -1;
        }
        if (strlen(value) >= sizeof(a->cell_id)) {
            fprintf(stderr,
                    "config_drift: %s:%d: cell_id too long (%zu chars, max %zu)\n",
                    path, lineno, strlen(value), sizeof(a->cell_id) - 1);
            return -1;
        }
        str_bounded_copy(a->cell_id, sizeof(a->cell_id), value);
        a->has_cell_id = true;
    } else {
        /* Unknown key: warn (forward-compat) but do not fail. */
        fprintf(stderr,
                "config_drift: %s:%d: WARN unknown key '%s' (ignored; "
                "manifest may be from a newer schema)\n",
                path, lineno, key);
    }
    return 0;
}

int config_drift_load(const char *path, ConfigDriftManifest *mf) {
    if (!path || !mf) return -1;
    config_drift_manifest_init(mf);

    FILE *f = fopen(path, "r");
    if (!f) {
        /* Caller decides whether missing-file is fatal. We just report. */
        return -1;
    }

    char line[1024];
    int lineno = 0;
    ConfigDriftArm *current = NULL;
    int rc = 0;

    while (fgets(line, sizeof(line), f)) {
        lineno++;
        /* Strip trailing newline. */
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
            line[--len] = '\0';
        }
        /* Strip comment (# to end-of-line). */
        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';

        char *trimmed = str_trim(line);
        if (*trimmed == '\0') continue;  /* blank line */

        /* Split on first '='. */
        char *eq = strchr(trimmed, '=');
        if (!eq) {
            fprintf(stderr,
                    "config_drift: %s:%d: missing '=' in line\n", path, lineno);
            rc = -1;
            break;
        }
        *eq = '\0';
        char *key   = str_trim(trimmed);
        char *value = str_trim(eq + 1);
        if (*key == '\0') {
            fprintf(stderr, "config_drift: %s:%d: empty key\n", path, lineno);
            rc = -1;
            break;
        }
        if (parse_kv_line(path, lineno, key, value, mf, &current) != 0) {
            rc = -1;
            break;
        }
    }
    fclose(f);
    return rc;
}

const ConfigDriftArm *config_drift_find_arm(const ConfigDriftManifest *mf,
                                            const char *model_name) {
    if (!mf || !model_name) return NULL;
    for (int i = 0; i < mf->n_arms; i++) {
        /* D-194: skip cell-aware arms in the legacy (model-key) lookup so
         * Phase-2 evaluator/replay tooling without LAB_CELL set always hits
         * the legacy arm pair, regardless of file order. Phase-3 factorial
         * cell selection MUST use config_drift_find_arm_by_cell() below. */
        if (mf->arms[i].has_cell_id) continue;
        if (strcmp(mf->arms[i].model, model_name) == 0) {
            return &mf->arms[i];
        }
    }
    return NULL;
}

const ConfigDriftArm *config_drift_find_arm_by_cell(const ConfigDriftManifest *mf,
                                                    const char *cell_id) {
    if (!mf || !cell_id) return NULL;
    for (int i = 0; i < mf->n_arms; i++) {
        /* Only cell-aware arms are eligible — symmetric with find_arm(). */
        if (!mf->arms[i].has_cell_id) continue;
        if (strcmp(mf->arms[i].cell_id, cell_id) == 0) {
            return &mf->arms[i];
        }
    }
    return NULL;
}

/* ---- Public: the assertion ---- */

/* Print a "CONFIG-DRIFT FATAL: <field> expected X, got Y" line to stderr.
 * mismatch_count is incremented by the caller after each call. */
static void print_mismatch_int(const char *field, int64_t expected, int64_t got) {
    fprintf(stderr,
            "CONFIG-DRIFT FATAL: %s expected %lld, got %lld\n",
            field, (long long)expected, (long long)got);
}

static void print_mismatch_str(const char *field, const char *expected,
                               const char *got) {
    fprintf(stderr,
            "CONFIG-DRIFT FATAL: %s expected '%s', got '%s'\n",
            field, expected ? expected : "(null)", got ? got : "(null)");
}

/* D-182 / P-CONFIG-DRIFT-LR-WIRE: arm-prefixed double mismatch printer.
 *
 * The 14 LR/dynamics fields use a richer FATAL message format that includes the
 * arm name + uses %g (loses no significant digits for values like 0.002 / 1e-05
 * but stays readable). Operators see arm context on the front of the launch
 * console without having to grep the surrounding manifest header. */
static void print_mismatch_double_arm(const char *arm, const char *field,
                                      double expected, double got) {
    fprintf(stderr,
            "CONFIG-DRIFT FATAL: %s %s expected %.17g, got %.17g\n",
            arm, field, expected, got);
}

static void print_mismatch_int_arm(const char *arm, const char *field,
                                   int64_t expected, int64_t got) {
    fprintf(stderr,
            "CONFIG-DRIFT FATAL: %s %s expected %lld, got %lld\n",
            arm, field, (long long)expected, (long long)got);
}

static void print_mismatch_str_arm(const char *arm, const char *field,
                                   const char *expected, const char *got) {
    fprintf(stderr,
            "CONFIG-DRIFT FATAL: %s %s expected '%s', got '%s'\n",
            arm, field,
            expected ? expected : "(null)",
            got      ? got      : "(null)");
}

/* Absolute-tolerance equality for double-precision values. Manifest doubles are
 * exact (parsed with strtod) and runtime values are baked-in C constants in
 * scale_experiment.c, so we tolerate only float-→-double round-trip noise.
 * 1e-9 is well below the 1e-5 / 1e-3 / 0.9 / 0.999 magnitudes in play. */
#define CONFIG_DRIFT_DOUBLE_TOL 1e-9
static bool doubles_equal_tol(double a, double b) {
    return fabs(a - b) <= CONFIG_DRIFT_DOUBLE_TOL;
}

void config_drift_assert_or_die(const ConfigDriftRuntime *rt) {
    if (!rt) {
        fprintf(stderr,
                "CONFIG-DRIFT FATAL: internal error — null runtime ptr\n");
        exit(1);
    }

    /* Step 1: gate on LAB_PROGRAM env var. */
    const char *prog = getenv("LAB_PROGRAM");
    if (!prog || prog[0] == '\0') {
        return;  /* no-op: legacy / one-off invocation */
    }

    /* Step 2: build the manifest path. */
    const char *root = getenv("LAB_PROGRAM_ROOT");
    if (!root || root[0] == '\0') root = "programs";
    char path[1024];
    int n = snprintf(path, sizeof(path),
                     "%s/%s/spec_invariants.yaml", root, prog);
    if (n < 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr,
                "CONFIG-DRIFT FATAL: manifest path too long (LAB_PROGRAM=%s)\n",
                prog);
        exit(1);
    }

    /* Step 3: load the manifest. */
    ConfigDriftManifest mf;
    config_drift_manifest_init(&mf);
    if (config_drift_load(path, &mf) != 0) {
        fprintf(stderr,
                "CONFIG-DRIFT FATAL: could not load spec_invariants manifest at "
                "'%s' (LAB_PROGRAM=%s). Required because LAB_PROGRAM is set; "
                "either provide the manifest or unset LAB_PROGRAM.\n",
                path, prog);
        config_drift_manifest_free(&mf);
        exit(1);
    }

    /* Step 4: find the arm. D-194 cell-id discipline:
     *   - LAB_CELL set + non-empty → cell_id-keyed lookup (Phase-3 factorial).
     *     Failure to match is FATAL with a cell-context message; we do NOT
     *     silently fall back to model-key here, because under LAB_CELL the
     *     operator has explicitly opted into cell-aware lookup and a missing
     *     cell signals a manifest/launcher mismatch the operator must fix.
     *   - LAB_CELL unset/empty → legacy model-key lookup (Phase-2 backward-
     *     compat unchanged from D-191). The legacy lookup skips cell-aware
     *     arms (see config_drift_find_arm), so it always resolves to the
     *     legacy arm pair regardless of file order. */
    const char *cell = getenv("LAB_CELL");
    const ConfigDriftArm *arm = NULL;

    if (cell && cell[0] != '\0') {
        arm = config_drift_find_arm_by_cell(&mf, cell);
        if (!arm) {
            fprintf(stderr,
                    "CONFIG-DRIFT FATAL: LAB_CELL='%s' requested but no arm "
                    "in '%s' has cell_id='%s'. Available cells:",
                    cell, path, cell);
            int n_cell_arms = 0;
            for (int i = 0; i < mf.n_arms; i++) {
                if (mf.arms[i].has_cell_id) {
                    fprintf(stderr, " %s(cell_id=%s,model=%s)",
                            mf.arms[i].arm_name,
                            mf.arms[i].cell_id,
                            mf.arms[i].model);
                    n_cell_arms++;
                }
            }
            if (n_cell_arms == 0) {
                fprintf(stderr,
                        " (none — manifest has no cell-aware arms; this "
                        "manifest predates the D-194 cell-id schema or "
                        "LAB_CELL was set against the wrong LAB_PROGRAM)");
            }
            fprintf(stderr, "\n");
            config_drift_manifest_free(&mf);
            exit(1);
        }
    } else {
        arm = config_drift_find_arm(&mf, rt->model);
        if (!arm) {
            fprintf(stderr,
                    "CONFIG-DRIFT FATAL: no arm in '%s' matches runtime "
                    "--model='%s'. Available arms:",
                    path, rt->model ? rt->model : "(null)");
            for (int i = 0; i < mf.n_arms; i++) {
                if (mf.arms[i].has_cell_id) continue;  /* legacy-only listing */
                fprintf(stderr, " %s(model=%s)",
                        mf.arms[i].arm_name, mf.arms[i].model);
            }
            fprintf(stderr,
                    " (cell-aware arms not listed — set LAB_CELL=<X> to "
                    "select one)\n");
            config_drift_manifest_free(&mf);
            exit(1);
        }
    }

    /* Step 5: compare every field. Accumulate mismatches so the operator sees
     * ALL drift on one launch attempt rather than chasing one mismatch at a
     * time across multiple aborted starts. */
    int mismatches = 0;

    /* tokenizer basename */
    const char *runtime_basename = path_basename(rt->tokenizer_path);
    if (strcmp(arm->tokenizer_basename, runtime_basename) != 0) {
        print_mismatch_str("tokenizer", arm->tokenizer_basename, runtime_basename);
        mismatches++;
    }

    /* vocab size */
    if (arm->vocab_size != rt->vocab_size) {
        print_mismatch_int("vocab_size",
                           (int64_t)arm->vocab_size,
                           (int64_t)rt->vocab_size);
        mismatches++;
    }

    /* param_count_total (allow exact match — all our factories are deterministic) */
    if (arm->param_count_total != rt->param_count_total) {
        print_mismatch_int("param_count_total",
                           arm->param_count_total,
                           rt->param_count_total);
        mismatches++;
    }

    /* param_count_active — only checked if both sides have the field. If the
     * manifest specifies it but runtime doesn't (e.g. dense), that's a config
     * mismatch (the manifest is for an MoE arm but runtime is dense). If
     * runtime has it but manifest doesn't, ignore (dense manifest, MoE-style
     * runtime would also fail elsewhere). */
    if (arm->has_param_count_active) {
        if (!rt->has_param_count_active) {
            fprintf(stderr,
                    "CONFIG-DRIFT FATAL: param_count_active expected %lld "
                    "(manifest), got (none) (runtime did not report active count)\n",
                    (long long)arm->param_count_active);
            mismatches++;
        } else if (arm->param_count_active != rt->param_count_active) {
            print_mismatch_int("param_count_active",
                               arm->param_count_active,
                               rt->param_count_active);
            mismatches++;
        }
    }

    /* ---- D-182 LR/schedule/dynamics field comparisons -------------------- */
    /* Each field is checked ONLY when manifest declares it (has_<field>=true).
     * Legacy manifests that don't pin training dynamics produce zero extra
     * checks here (defense-in-depth). Runtime side has no has_* gate — the
     * implementation_engineer_c contract is presence-by-construction:
     * scale_experiment.c init populates all 14 unconditionally before the
     * call to config_drift_assert_or_die. */

    /* Doubles (5 fields): tolerance 1e-9 absolute. */
    if (arm->has_lr_base && !doubles_equal_tol(arm->lr_base, rt->lr_base)) {
        print_mismatch_double_arm(arm->arm_name, "lr_base",
                                  arm->lr_base, rt->lr_base);
        mismatches++;
    }
    if (arm->has_lr_min && !doubles_equal_tol(arm->lr_min, rt->lr_min)) {
        print_mismatch_double_arm(arm->arm_name, "lr_min",
                                  arm->lr_min, rt->lr_min);
        mismatches++;
    }
    if (arm->has_adam_beta1 &&
        !doubles_equal_tol(arm->adam_beta1, rt->adam_beta1)) {
        print_mismatch_double_arm(arm->arm_name, "adam_beta1",
                                  arm->adam_beta1, rt->adam_beta1);
        mismatches++;
    }
    if (arm->has_adam_beta2 &&
        !doubles_equal_tol(arm->adam_beta2, rt->adam_beta2)) {
        print_mismatch_double_arm(arm->arm_name, "adam_beta2",
                                  arm->adam_beta2, rt->adam_beta2);
        mismatches++;
    }
    if (arm->has_weight_decay &&
        !doubles_equal_tol(arm->weight_decay, rt->weight_decay)) {
        print_mismatch_double_arm(arm->arm_name, "weight_decay",
                                  arm->weight_decay, rt->weight_decay);
        mismatches++;
    }
    if (arm->has_grad_clip_norm &&
        !doubles_equal_tol(arm->grad_clip_norm, rt->grad_clip_norm)) {
        print_mismatch_double_arm(arm->arm_name, "grad_clip_norm",
                                  arm->grad_clip_norm, rt->grad_clip_norm);
        mismatches++;
    }

    /* Strings (2 fields): exact strcmp. NULL runtime ptr counted as mismatch. */
    if (arm->has_lr_schedule) {
        const char *got = rt->lr_schedule ? rt->lr_schedule : "";
        if (strcmp(arm->lr_schedule, got) != 0) {
            print_mismatch_str_arm(arm->arm_name, "lr_schedule",
                                   arm->lr_schedule, rt->lr_schedule);
            mismatches++;
        }
    }
    if (arm->has_optimizer) {
        const char *got = rt->optimizer ? rt->optimizer : "";
        if (strcmp(arm->optimizer, got) != 0) {
            print_mismatch_str_arm(arm->arm_name, "optimizer",
                                   arm->optimizer, rt->optimizer);
            mismatches++;
        }
    }

    /* Integers (7 fields): exact equality. */
    if (arm->has_lr_warmup_steps &&
        arm->lr_warmup_steps != rt->lr_warmup_steps) {
        print_mismatch_int_arm(arm->arm_name, "lr_warmup_steps",
                               arm->lr_warmup_steps, rt->lr_warmup_steps);
        mismatches++;
    }
    if (arm->has_grad_accum_steps &&
        arm->grad_accum_steps != rt->grad_accum_steps) {
        print_mismatch_int_arm(arm->arm_name, "grad_accum_steps",
                               arm->grad_accum_steps, rt->grad_accum_steps);
        mismatches++;
    }
    if (arm->has_batch_size && arm->batch_size != rt->batch_size) {
        print_mismatch_int_arm(arm->arm_name, "batch_size",
                               arm->batch_size, rt->batch_size);
        mismatches++;
    }
    if (arm->has_seq_len && arm->seq_len != rt->seq_len) {
        print_mismatch_int_arm(arm->arm_name, "seq_len",
                               arm->seq_len, rt->seq_len);
        mismatches++;
    }
    if (arm->has_max_steps && arm->max_steps != rt->max_steps) {
        print_mismatch_int_arm(arm->arm_name, "max_steps",
                               arm->max_steps, rt->max_steps);
        mismatches++;
    }
    if (arm->has_weight_seed && arm->weight_seed != rt->weight_seed) {
        print_mismatch_int_arm(arm->arm_name, "weight_seed",
                               arm->weight_seed, rt->weight_seed);
        mismatches++;
    }

    if (mismatches > 0) {
        fprintf(stderr,
                "CONFIG-DRIFT FATAL: %d field(s) mismatched against manifest "
                "'%s' (arm '%s', model '%s'). Refusing to start training. "
                "If the manifest is wrong, amend it via the program's formal "
                "amendment procedure; if the launch is wrong, fix the launch "
                "command.\n",
                mismatches, path, arm->arm_name, arm->model);
        config_drift_manifest_free(&mf);
        exit(1);
    }

    /* All fields match: silent success (no banner — keeps stdout clean for
     * existing log-parsing tooling). */
    config_drift_manifest_free(&mf);
}
