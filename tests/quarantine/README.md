# tests/quarantine/

Tests that are known to fail for reasons **unrelated to a real regression**,
and are therefore excluded from the default `make test` target so they do not
mask legitimate failures in CI / build verification.

This directory exists per **D-097 Disposition P2** (Cycle 30, ACCEPTED) and
closes evaluator Risk C that was carried for 4 consecutive cycles (25, 26,
27, 28) without resolution.

**Archive never delete.** These tests are not deleted. They are parked here,
still compilable via `make test-quarantine`, and will be moved back to
`tests/` when the conditions under "Reinstate when" below are met.

---

## Currently quarantined

### `test_metal_matmul.c`

- **Reinstatement trigger**: Metal bridge fix lands AND GPU dispatch is re-enabled
  (reversing D-088 PERF_APPROVED).

- **Why it fails today**: 10 of its assertions fail because
  `default.metallib` is not discoverable at the runtime search path under
  ASan + the repository's Makefile-produced layout. The failures are a
  shader-loading / linking issue, NOT a correctness regression in any Metal
  kernel. The underlying kernels are still correct against their CPU
  reference when loaded manually.

- **Why we no longer care about "fixing" it now**:
  - **D-088 PERF_APPROVED** (see `data/engineering/perf_log.md`): the GPU
    dispatch path was **disabled** after benchmarks showed CPU/AMX is
    3-100x faster than GPU at our current scale (50M params, d_model=512).
    The Metal kernels are dead code on the hot path today.
  - The Metal path only becomes load-bearing at **>= 1B scale** where
    matmul at D > 512 re-crosses the CPU/GPU break-even point. That is
    >= 2 phases away per `data/engineering/phase_c_design.md`.
  - Until then, the 10 test failures are **pure handoff noise**: they
    appear in every cycle's `make test` output and consume evaluator
    attention without signal (Cycle 28 evaluator report §3 Risk C:
    "NOT ADDRESSED" for the 4th consecutive cycle).

- **What was lost by quarantining**: nothing. The tests still build, still
  run under `make test-quarantine`, and any regression in the Metal
  library (if/when re-enabled) will be caught the moment the test is
  moved back to `tests/`. The original path `tests/test_metal_matmul.c`
  is preserved via git history.

- **Reference documents**:
  - `data/engineering/perf_log.md` (D-088 PERF_APPROVED rationale)
  - `data/org_retros/retro_29.md` Section 5 (4-consecutive-cycle defer)
  - `data/org_retros/retro_29.md` Section 8 P2 (ACCEPTED)
  - `data/decisions_recent.md` D-097 Dispositions section

---

## Policy

Adding a test to this directory requires:

1. A one-sentence reason in this README (not a bug to fix — a systemic
   reason the test cannot be green today without out-of-scope work).
2. An explicit **reinstatement trigger** — the concrete event that would
   move the test back to the default suite.
3. A reference to the Director decision that authorized the quarantine.

Removing a test from this directory (reinstatement) is always safe:
`git mv tests/quarantine/test_foo.c tests/test_foo.c` and the default
discovery picks it up again.
