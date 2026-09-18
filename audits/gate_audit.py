#!/usr/bin/env python3
"""Did the anti-forgery GATE ever run?

The D-110 remedy has three components: a detector, a runner gate that makes the
detector binding, and a per-phase evaluator audit. The published account credits the
gate with turning the detector from "a warning" into "a control".

This audits the gate the same way the identity audit audits the receipts: against
evidence the Director did not write. The gate lives in run_agi_lab.sh and appends the
detector's stdout+stderr to the session log:

    if [ "$EXIT_REASON" = "GRACEFUL_CHECKPOINT" ] && [ -x tools/verify_signatures.py ]; then
        if ! source .venv/bin/activate && python3 tools/verify_signatures.py >> "$SESSION_LOG" 2>&1; then

So if the gate ran, the detector's output is in the log. Three counts settle it:
  1. how often the gate's precondition (GRACEFUL_CHECKPOINT) was met
  2. how often the detector's own output appears in a session log
  3. how often the gate blocked

`! A && B` parses as `(! A) && B`. When `source .venv/bin/activate` SUCCEEDS, `!`
makes the left side false and `&&` short-circuits, so python3 is never invoked and the
`if` body never runs. The detector therefore executes only when venv activation
*fails* -- and then in the wrong interpreter. `--prove` reproduces this in a scratch
directory rather than asking you to take the parse on faith.
"""
import re, subprocess, sys, tempfile, os
from pathlib import Path

REPO = Path.home() / "mnt/AGI"
LOGS = REPO / "data/infra/session_logs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Strings the detector itself prints (tools/verify_signatures.py). If the gate ran,
# at least one of these is in the log.
DETECTOR_OUTPUT = [
    "[verify_signatures] OK:",
    "[verify_signatures] FORGERY DETECTED",
    "[verify_signatures] no paths to scan",
    "signatures verified",
]
GATE_BLOCK = "BLOCKED: verify_signatures.py detected forged signatures"
PRECONDITION = "GRACEFUL_CHECKPOINT"
# A Director tool call, i.e. the lab grading itself -- not the gate.
MANUAL_CALL = re.compile(r"┌?\s*\$\s*.*python3?\s+\S*verify_signatures\.py")


def prove() -> int:
    """Reproduce the precedence behaviour on a scratch venv and a stub detector."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / ".venv/bin").mkdir(parents=True)
        (d / ".venv/bin/activate").write_text("#!/bin/bash\ntrue\n")
        (d / "det.py").write_text(
            '#!/usr/bin/env python3\nimport sys\nprint("DETECTOR RAN")\nsys.exit(1)\n')
        os.chmod(d / "det.py", 0o755)
        script = r'''
cd "%s"
echo "--- as written in run_agi_lab.sh (activate succeeds, detector would exit 1) ---"
: > out.log
if ! source .venv/bin/activate && python3 det.py >> out.log 2>&1; then
  echo "  gate fired: YES"
else
  echo "  gate fired: NO"
fi
echo "  detector output in log: '$(cat out.log)'"
echo "--- as intended (activate on its own line) ---"
: > out.log
source .venv/bin/activate
if ! python3 det.py >> out.log 2>&1; then
  echo "  gate fired: YES"
else
  echo "  gate fired: NO"
fi
echo "  detector output in log: '$(cat out.log)'"
''' % d
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        print(r.stdout, end="")
        if r.stderr.strip():
            print(r.stderr, file=sys.stderr)
    return 0


def main() -> int:
    if "--prove" in sys.argv:
        return prove()
    if not LOGS.is_dir():
        print(f"[gate] no session logs at {LOGS}", file=sys.stderr)
        return 2
    logs = sorted(LOGS.glob("*.log"))
    pre = out = blocked = manual = 0
    pre_logs, out_logs, manual_logs = [], [], []
    for lf in logs:
        try:
            txt = ANSI.sub("", lf.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        if PRECONDITION in txt:
            pre += 1; pre_logs.append(lf.name)
        if any(s in txt for s in DETECTOR_OUTPUT):
            out += 1; out_logs.append(lf.name)
        if GATE_BLOCK in txt:
            blocked += 1
        if MANUAL_CALL.search(txt):
            manual += 1; manual_logs.append(lf.name)

    print(f"session logs scanned                                 {len(logs)}")
    print(f"sessions reaching the gate's precondition            {pre}")
    print(f"sessions containing the detector's own output        {out}")
    print(f"sessions where the gate blocked                      {blocked}")
    print(f"sessions where the DIRECTOR ran the detector by hand {manual}")
    if manual_logs:
        print(f"  (self-grading, not the gate: {', '.join(manual_logs[:8])}"
              f"{' ...' if len(manual_logs) > 8 else ''})")
    print()
    if pre and not out:
        print(f"VERDICT: the gate never executed. Its precondition was satisfied {pre} "
              f"times\nand the detector's output appears in 0 of {len(logs)} logs, though "
              f"the gate\nredirects that output into the log. Run with --prove for the "
              f"mechanism.")
        return 1
    if out:
        print(f"VERDICT: the detector's output is present in {out} session log(s); "
              f"inspect them\nbefore concluding the gate was live "
              f"({', '.join(out_logs[:5])}).")
        return 0
    print("VERDICT: inconclusive — the precondition was never met either.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
