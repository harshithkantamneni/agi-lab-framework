#!/usr/bin/env python3
"""Long-run launcher — session-spanning wrapper for Phase-2 training runs.

Wraps a training command in ``nohup + setsid + caffeinate -i`` so it survives
Director session detach and macOS sleep-throttling. Idempotent: running twice
with the same --run-id detects the live PID and no-ops. On restart, auto-
resumes from the latest ``step_XXXXXX.ckpt`` in the configured checkpoint dir.

Spec: ``data/engineering/long_run_launcher_design.md`` (D-117, PI-approved).

Usage:
    python3 tools/run_long.py --run-id dense_a --cmd 'build/scale_experiment ...'

D-182 P-AUTORESUME-OFF-SPEC-WARNING remediation (PI condition #3, 2026-04-24):
    Option (a) "refuse-to-launch if ckpt-dir non-empty without spec-sentinel"
    chosen over option (b) "short-circuit auto-resume on incompatible shape"
    for three reasons:

      1. It fails fast and loud BEFORE training starts, not mid-run via the
         WARNING → rc=2 fail-fast misfire path at the tail of the old
         launcher. D-155 fired that misfire on Dense-A's clean exit because
         the binary emitted WARNING during auto-resume attempt on a
         leftover ckpt-dir, trained from scratch, exited 0 — but the
         launcher translated 0 → 2 and halted the `&&` chain.

      2. Option (b) requires the launcher to understand the checkpoint
         binary format (HSPAConfig fields, V5 layout) to check shape
         compatibility. That couples the launcher to C-side checkpoint
         internals and creates another silent-drift hole when the format
         changes. Sentinel check is spec-level, not format-level.

      3. Option (a) composes cleanly with the D-122 config_drift detector:
         the sentinel records LAB_PROGRAM + manifest SHA-1 at launch time,
         and on re-launch we compare. If the program or manifest changed
         (e.g. Phase respec with new LR cells) the launcher refuses to
         auto-resume the stale ckpt and tells the operator to archive via
         the P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE procedure.

    The old rc=0+WARNING → rc=2 fail-fast misfire remains as a defense-in-
    depth trailer (it still catches the "binary attempted resume, silently
    started fresh" case from a binary-native warning path the sentinel
    can't see), but the SENTINEL is now the primary guard — in practice the
    pre-launch refusal triggers first and the WARNING scan is rarely
    reached.
"""

import argparse
import hashlib
import os
import re
import shlex
import signal
import subprocess
import sys
import time


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNS_DIR = os.path.join(ROOT, "data", "runs")
CKPT_RE = re.compile(r"^step_(\d{6})\.ckpt$")
# Sentinel filename inside each checkpoint-dir. Records the LAB_PROGRAM at
# launch time plus the manifest SHA-1. Phase-3 ckpt-dirs that pre-date this
# sentinel will have no file; the launcher refuses auto-resume in that case
# and tells the operator to archive per P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE
# (programs/_template → now in data/memories/procedures.md, D-182).
SENTINEL_BASENAME = ".run_long_spec_sentinel"


def log(msg: str) -> None:
    print(f"[run_long {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(pid_file: str):
    try:
        with open(pid_file) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def latest_checkpoint(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None, -1
    best_path, best_step = None, -1
    for name in os.listdir(ckpt_dir):
        m = CKPT_RE.match(name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = os.path.join(ckpt_dir, name)
    return best_path, best_step


def extract_ckpt_dir(cmd: str):
    # Support both `--checkpoint-dir X` (two tokens) and `--checkpoint-dir=X`
    # (single token) forms; argparse / the C CLI both accept the `=` form so
    # the launcher must mirror that or silently lose auto-resume.
    toks = shlex.split(cmd)
    for i, t in enumerate(toks):
        if t == "--checkpoint-dir" and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith("--checkpoint-dir="):
            return t.split("=", 1)[1]
    return None


def inject_resume(cmd: str, ckpt_path: str) -> str:
    # Append --resume only if not already present. shlex-safe quoting.
    toks = shlex.split(cmd)
    if "--resume" in toks:
        return cmd
    toks.extend(["--resume", ckpt_path])
    return " ".join(shlex.quote(t) for t in toks)


# ---- D-182 P-AUTORESUME-OFF-SPEC-WARNING: spec-sentinel helpers ------------

def _manifest_path_for(program: str) -> str:
    """Return the absolute path of the spec manifest for a program, or ''."""
    if not program:
        return ""
    return os.path.join(ROOT, "programs", program, "spec_invariants.yaml")


def compute_spec_fingerprint(program: str) -> str:
    """Return a short fingerprint of the program's spec_invariants manifest.

    Shape: ``<program>:<sha1_hex[:12]>`` so both the program name and the
    manifest-content hash are pinned. If LAB_PROGRAM is unset or the
    manifest is missing, returns ``"<none>:<none>"`` so the sentinel still
    records an explicit "no spec was in effect at launch" marker.
    """
    if not program:
        return "<none>:<none>"
    path = _manifest_path_for(program)
    if not os.path.isfile(path):
        return f"{program}:<missing-manifest>"
    try:
        with open(path, "rb") as f:
            digest = hashlib.sha1(f.read()).hexdigest()[:12]
    except OSError:
        return f"{program}:<unreadable-manifest>"
    return f"{program}:{digest}"


def read_sentinel(ckpt_dir_abs: str):
    """Return the sentinel fingerprint stored in a checkpoint dir, or None."""
    path = os.path.join(ckpt_dir_abs, SENTINEL_BASENAME)
    try:
        with open(path) as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        return None


def write_sentinel(ckpt_dir_abs: str, fingerprint: str) -> None:
    """Write the sentinel fingerprint into the checkpoint dir.

    Safe to call repeatedly; last write wins. Creates the directory if
    absent so a fresh run can still record its fingerprint.
    """
    os.makedirs(ckpt_dir_abs, exist_ok=True)
    path = os.path.join(ckpt_dir_abs, SENTINEL_BASENAME)
    with open(path, "w") as f:
        f.write(fingerprint + "\n")


def ckpt_dir_has_ckpts(ckpt_dir_abs: str) -> bool:
    """Return True iff the ckpt dir contains at least one ``step_*.ckpt``."""
    if not os.path.isdir(ckpt_dir_abs):
        return False
    for name in os.listdir(ckpt_dir_abs):
        if CKPT_RE.match(name):
            return True
    return False


def verify_spec_sentinel(ckpt_dir_abs: str, program: str):
    """Return (ok: bool, reason: str) for the spec-sentinel pre-launch check.

    Rules:
      * No ckpts in the dir → ok=True ("fresh launch"). The caller will
        write the sentinel post-launch so subsequent re-launches can verify.
      * Ckpts present + no sentinel → ok=False ("legacy ckpts without
        sentinel — archive before re-launch"). This is the D-182 refuse-to-
        launch path: P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE requires the
        operator to move legacy ckpts to ``_offspec_<prev_decision>_archive/``
        before the launcher will proceed.
      * Ckpts present + sentinel mismatch → ok=False ("spec changed since
        checkpoints were written"). Same remediation.
      * Ckpts present + sentinel match → ok=True ("resuming compatible
        checkpoint"). The auto-resume path below proceeds.
    """
    if not ckpt_dir_has_ckpts(ckpt_dir_abs):
        return True, "fresh ckpt-dir"
    existing = read_sentinel(ckpt_dir_abs)
    expected = compute_spec_fingerprint(program)
    if existing is None:
        return False, (
            f"checkpoints present but no spec-sentinel in {ckpt_dir_abs} — "
            f"this ckpt-dir pre-dates the D-182 sentinel protocol. "
            f"Archive it per P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE "
            f"(mv {ckpt_dir_abs} {ckpt_dir_abs}_legacy_archive/) then re-launch."
        )
    if existing != expected:
        return False, (
            f"spec-sentinel mismatch in {ckpt_dir_abs}: existing='{existing}' "
            f"vs expected='{expected}'. The program or its spec_invariants "
            f"manifest has changed since checkpoints were written. "
            f"Archive per P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE then re-launch."
        )
    return True, f"sentinel match ({expected})"


_STEP_LINE_RE = re.compile(
    r"^\s*(\d+)\s+\|.*\|\s*(\d+(?:\.\d+)?)\s*$"
)


def _parse_step_times(log_path: str, after_step: int = 0) -> "list[tuple[int, float]]":
    """Parse (step, ms) tuples from scale_experiment's print_step output.

    Format (from src/training/scale_experiment.c print_step):
        '%5d  | %8.4f (...) | %9.2f | %6.3f | %7.5f | %6.4f | %2d | %.0f'
    Last column is step_time in milliseconds. Returns only steps >
    `after_step` so the caller can resume incrementally.
    """
    out: list[tuple[int, float]] = []
    if not os.path.isfile(log_path):
        return out
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                m = _STEP_LINE_RE.match(line)
                if not m:
                    continue
                try:
                    step = int(m.group(1))
                    ms = float(m.group(2))
                except ValueError:
                    continue
                if step > after_step:
                    out.append((step, ms))
    except OSError:
        pass
    return out


def _perf_band_watcher(
    log_path: str,
    child_pid: int,
    band_min_ms: float,
    band_max_ms: float,
    warmup_steps: int,
    window_steps: int,
    drift_flag: dict,
) -> None:
    """Background watcher: poll the child's stdout log for step times.

    Once `warmup_steps + window_steps` steps have been logged, compute the
    mean step time over the post-warmup window and compare against the band.
    If the mean is outside [band_min_ms, band_max_ms], set drift_flag['drift']
    = True with diagnostic context, and SIGTERM the child. The main thread's
    proc.wait() will return non-zero and we surface that as rc=PERF_DRIFT_RC.

    Runs for at most ~3× expected window time. After making a decision (pass
    or kill), exits — does NOT continuously monitor mid-run drift (kernels
    don't drift over time; if the band passed once, it's fine).
    """
    import threading
    poll_interval = 30.0
    needed_steps = warmup_steps + window_steps
    deadline = time.time() + max(900, window_steps * 30)  # generous cap
    while time.time() < deadline:
        if not pid_alive(child_pid):
            return  # child exited on its own; nothing to do
        steps = _parse_step_times(log_path)
        if len(steps) >= needed_steps:
            window = steps[warmup_steps:warmup_steps + window_steps]
            mean_ms = sum(ms for _, ms in window) / len(window)
            min_ms = min(ms for _, ms in window)
            max_ms = max(ms for _, ms in window)
            in_band = band_min_ms <= mean_ms <= band_max_ms
            log(
                f"[perf-band] window steps {window[0][0]}-{window[-1][0]}: "
                f"mean={mean_ms:.0f}ms min={min_ms:.0f}ms max={max_ms:.0f}ms "
                f"band=[{band_min_ms:.0f}, {band_max_ms:.0f}]ms "
                f"verdict={'PASS' if in_band else 'FAIL'}"
            )
            with open(log_path, "a") as f:
                f.write(
                    f"\n=== run_long perf-band check {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
                    f"window: steps {window[0][0]}-{window[-1][0]} (warmup={warmup_steps}, n={window_steps})\n"
                    f"mean_ms={mean_ms:.0f} min_ms={min_ms:.0f} max_ms={max_ms:.0f}\n"
                    f"band=[{band_min_ms:.0f}, {band_max_ms:.0f}]ms\n"
                    f"verdict={'PASS' if in_band else 'FAIL'}\n"
                )
            if not in_band:
                drift_flag["drift"] = True
                drift_flag["mean_ms"] = mean_ms
                drift_flag["band"] = (band_min_ms, band_max_ms)
                drift_flag["window_steps"] = (window[0][0], window[-1][0])
                # SIGTERM the child to break out of proc.wait()
                try:
                    os.kill(child_pid, signal.SIGTERM)
                    log(f"[perf-band] FATAL: mean {mean_ms:.0f}ms outside band; killed child pid {child_pid}")
                except OSError as e:
                    log(f"[perf-band] kill failed: {e}")
            return
        time.sleep(poll_interval)
    log(f"[perf-band] watcher timed out before {needed_steps} steps logged; assumed pass.")


PERF_DRIFT_RC = 4  # exit code reserved for perf-band-fail kills


def main() -> int:
    ap = argparse.ArgumentParser(description="Session-spanning long-run launcher")
    ap.add_argument("--run-id", required=True, help="Unique run identifier (e.g. dense_a)")
    ap.add_argument("--cmd", required=True, help="Full child command line (quoted)")
    ap.add_argument("--foreground", action="store_true", help="Run in foreground (debugging)")
    ap.add_argument("--perf-band-min-ms", type=float, default=None,
                    help="Lower bound on mean step_time (post-warmup window). "
                         "If both --perf-band-min-ms and --perf-band-max-ms are "
                         "given, a watcher kills the child if the post-warmup "
                         "mean step_time is outside [min, max]. Returns rc=4.")
    ap.add_argument("--perf-band-max-ms", type=float, default=None,
                    help="Upper bound on mean step_time (post-warmup window).")
    ap.add_argument("--perf-band-warmup-steps", type=int, default=10,
                    help="Skip first N steps before measuring (default 10; "
                         "first few steps include JIT warmup, alloc, etc.).")
    ap.add_argument("--perf-band-window-steps", type=int, default=50,
                    help="Measurement window size in steps (default 50).")
    args = ap.parse_args()

    run_dir = os.path.join(RUNS_DIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    pid_file = os.path.join(run_dir, "pid")
    log_file = os.path.join(run_dir, "stdout.log")
    done_file = os.path.join(run_dir, "done")

    # Idempotency: live PID → no-op.
    existing = read_pid(pid_file)
    if existing and pid_alive(existing):
        log(f"run '{args.run_id}' already running (pid {existing}); no-op.")
        return 0
    if existing:
        log(f"stale pid file (pid {existing} dead); clearing.")
        try:
            os.remove(pid_file)
        except OSError:
            pass

    # Auto-resume: find latest step_XXXXXX.ckpt in configured --checkpoint-dir.
    # D-182 P-AUTORESUME-OFF-SPEC-WARNING: BEFORE auto-resume, verify the
    # spec-sentinel. If the ckpt-dir has checkpoints from a different spec
    # (or no sentinel at all), REFUSE TO LAUNCH. This is the primary guard
    # added in D-182 — it supersedes the rc=0+WARNING → rc=2 fail-fast
    # misfire that fired on D-155 by catching off-spec ckpt dirs BEFORE the
    # binary starts.
    cmd = args.cmd
    program = os.environ.get("LAB_PROGRAM", "")
    ckpt_dir = extract_ckpt_dir(cmd)
    if ckpt_dir:
        resolved = ckpt_dir if os.path.isabs(ckpt_dir) else os.path.join(ROOT, ckpt_dir)
        sentinel_ok, sentinel_reason = verify_spec_sentinel(resolved, program)
        if not sentinel_ok:
            log(f"FAIL: {sentinel_reason}")
            log(
                "REFUSING TO LAUNCH. P-CHECKPOINT-DIR-ARCHIVE-DISCIPLINE "
                "requires archiving off-spec ckpts before re-launch. Returning "
                "rc=3 so `&&` chains halt loudly."
            )
            return 3
        log(f"spec-sentinel: {sentinel_reason}")
        ckpt_path, ckpt_step = latest_checkpoint(resolved)
        if ckpt_path:
            rel = os.path.relpath(ckpt_path, ROOT)
            log(f"resuming from step {ckpt_step}: {rel}")
            cmd = inject_resume(cmd, rel)
        else:
            log(f"no checkpoint in {ckpt_dir}; starting fresh.")
        # Record the spec fingerprint for this ckpt-dir BEFORE the child
        # launches. If the child fails, the sentinel is harmless; if it
        # succeeds, subsequent re-launches can verify compatibility.
        fingerprint = compute_spec_fingerprint(program)
        try:
            write_sentinel(resolved, fingerprint)
            log(f"spec-sentinel written: {fingerprint}")
        except OSError as exc:
            log(f"warning: failed to write spec-sentinel: {exc}")
    else:
        log("warning: --checkpoint-dir not found in cmd; cannot auto-resume.")

    # Build nohup+caffeinate wrapper. `caffeinate -i` blocks idle sleep.
    # macOS lacks `setsid(1)`, but `start_new_session=True` on the Popen below
    # calls `os.setsid()` in the child, achieving the same session detach so
    # SIGHUP on Director-session-end does not propagate. stdout/stderr tee'd
    # via shell redirection.
    inner = f"exec caffeinate -i /bin/bash -c {shlex.quote(cmd)} >> {shlex.quote(log_file)} 2>&1"
    wrapper = ["/usr/bin/nohup", "/bin/bash", "-c", inner]

    if args.foreground:
        log(f"foreground: {cmd}")
        return subprocess.call(["/bin/bash", "-c", cmd], cwd=ROOT)

    with open(log_file, "a") as f:
        f.write(f"\n=== run_long launch {time.strftime('%Y-%m-%d %H:%M:%S')} run_id={args.run_id} ===\n")
        f.write(f"cmd: {cmd}\n")
        f.flush()
    proc = subprocess.Popen(
        wrapper,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    with open(pid_file, "w") as f:
        f.write(f"{proc.pid}\n")
    log(f"launched '{args.run_id}' pid {proc.pid}; log: {log_file}")

    # Start perf-band watcher if both bounds given. Runs in a daemon thread,
    # measures the mean step_time over a post-warmup window, kills child on
    # out-of-band. Optional: callers without these args get the legacy flow.
    drift_flag: dict = {"drift": False}
    perf_thread = None
    if args.perf_band_min_ms is not None and args.perf_band_max_ms is not None:
        import threading
        perf_thread = threading.Thread(
            target=_perf_band_watcher,
            args=(log_file, proc.pid,
                  args.perf_band_min_ms, args.perf_band_max_ms,
                  args.perf_band_warmup_steps, args.perf_band_window_steps,
                  drift_flag),
            daemon=True,
        )
        perf_thread.start()
        log(
            f"perf-band watcher started: "
            f"band=[{args.perf_band_min_ms:.0f}, {args.perf_band_max_ms:.0f}]ms "
            f"warmup={args.perf_band_warmup_steps} window={args.perf_band_window_steps}"
        )

    # SIGHUP/SIGTERM immunity for the waiter: when a Director tmux/ssh session
    # ends, the kernel sends SIGHUP to every process in the terminal's session
    # group. The child escapes this via setsid (new session), but THIS Python
    # process is still in the original session — default SIGHUP handling would
    # kill the waiter, interrupt proc.wait(), and make the Makefile's
    # `run-phase2-pair` `&&` chain drop MoE Rev-2 even though Dense-A is still
    # running fine. Ignoring SIGHUP keeps the waiter alive until the real child
    # exit. SIGTERM is also ignored because (a) we don't want an operator's
    # stray `kill` to break the serial contract, and (b) if the child needs to
    # die, signalling the child directly is the correct channel.
    # Manual repro (do not actually run during launch):
    #     bash -c 'python3 tools/run_long.py --run-id t --cmd "sleep 3600"' &
    #     WAITER=$!; sleep 2; kill -HUP $WAITER; sleep 2; ps -p $WAITER
    # Expected: waiter still alive, child still alive, no `done` file yet.
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # Block until the child exits so serial `&&` in the Makefile respects order.
    # Director sessions that detach kill this waiter but not the child (setsid).
    try:
        rc = proc.wait()
    except KeyboardInterrupt:
        log("waiter interrupted (SIGINT); child detached via setsid — still running.")
        return 130

    # Distinguish success vs failure vs silent-corrupt-resume. scale_experiment
    # prints "WARNING: could not load checkpoint" on load failure and then
    # starts FRESH — we must not let that masquerade as a successful resume.
    # Window the scan to only lines emitted after the most-recent launch header
    # we just wrote; the log file is append-only across restarts, so scanning
    # the whole file would false-positive on any prior failed run (B4).
    warnings = 0
    if os.path.isfile(log_file):
        try:
            with open(log_file, errors="replace") as f:
                lines = f.readlines()
            # Reverse-scan to the last `=== run_long launch` header, then count
            # warnings only in the tail after it.
            start = 0
            for idx in range(len(lines) - 1, -1, -1):
                if "=== run_long launch" in lines[idx]:
                    start = idx
                    break
            for line in lines[start:]:
                if "WARNING: could not load checkpoint" in line:
                    warnings += 1
        except OSError:
            pass
    if warnings:
        log(f"note: {warnings} 'could not load checkpoint' warning(s) in log (this run).")

    # Fail-fast on fresh-restart false positive (B3-mitigation): if the child
    # exited 0 but we observed a checkpoint-load warning in THIS run's log
    # window, it means scale_experiment silently abandoned the intended resume
    # and trained from scratch. A downstream `&&` would chain to MoE Rev-2 as
    # if Dense-A resumed cleanly — that is silently wrong. Convert to non-zero,
    # skip the `done` file, and let the Makefile halt the pair.
    #
    # D-182 note: with the spec-sentinel pre-launch check added above, this
    # branch is now a DEFENSE-IN-DEPTH trailer — in practice the sentinel
    # catches off-spec ckpt-dirs before the binary ever runs, so this path
    # fires only for binary-native resume failures the sentinel can't see
    # (e.g. a corrupt ckpt file whose spec still matches). Kept to preserve
    # the D-125 safety net for corner cases we haven't yet classified.
    if rc == 0 and warnings > 0:
        log(
            f"FAIL: rc=0 but {warnings} checkpoint-load warning(s) detected — "
            f"child started FRESH instead of resuming. Refusing to mark done; "
            f"returning non-zero to break the serial chain. Inspect {log_file}."
        )
        try:
            os.remove(pid_file)
        except OSError:
            pass
        return 2

    # Perf-band drift takes priority over rc — if the watcher killed the child,
    # the proc.wait rc above is from SIGTERM (typically 143/-15) and the
    # 'real' exit reason is PERF_DRIFT.
    if drift_flag.get("drift"):
        mean_ms = drift_flag.get("mean_ms", 0)
        band = drift_flag.get("band", (0, 0))
        steps = drift_flag.get("window_steps", (0, 0))
        log(
            f"FAIL: PERF_DRIFT — mean step_time {mean_ms:.0f}ms over steps "
            f"{steps[0]}-{steps[1]} outside band [{band[0]:.0f}, {band[1]:.0f}]ms. "
            f"Returning rc={PERF_DRIFT_RC} so the orchestrator stops the factorial."
        )
        with open(log_file, "a") as f:
            f.write(f"PERF_DRIFT: mean={mean_ms:.0f}ms band=[{band[0]:.0f},{band[1]:.0f}]ms\n")
        try:
            os.remove(pid_file)
        except OSError:
            pass
        return PERF_DRIFT_RC

    if rc == 0:
        with open(done_file, "w") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} exit=0\n")
        log(f"run '{args.run_id}' finished OK.")
    else:
        log(f"run '{args.run_id}' FAILED with exit {rc}; see {log_file}")
    try:
        os.remove(pid_file)
    except OSError:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
