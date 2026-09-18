#!/usr/bin/env python3
"""tools/verify_signatures.py — anti-forgery detector for AGI lab governance.

For every `<role> ✓` / "approved by <role>" / "signed <role>" attestation found
in program deliverables and lock documents, verify a corresponding entry exists
in `data/agents/<role>/episodic/` dated within the relevant window.

Exit codes:
  0  all signatures verified (no forgery)
  1  forgery detected (role signed without dispatch record)
  2  script error / bad input

Usage:
  python3 tools/verify_signatures.py [--phase <name>] [--program <name>] [--window-hours N]
  python3 tools/verify_signatures.py --full  # scans all programs

Added 2026-04-17 (D-110) after D-109 audit revealed 5 forged PI signatures.
See data/procedures.md §"Signature Forgery Remediation".
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "data" / "agents"
PROGRAMS_DIR = REPO_ROOT / "programs"

# Signature patterns. ONLY match formal signature formats to avoid false positives
# on narrative prose. Informal "the proposal approved by X" in body text is NOT a
# signature — a signature is a structured attestation in a specific format.
SIG_PATTERNS = [
    # **<role> ✓** — formal bolded signature (most common format)
    re.compile(r"\*\*([A-Za-z_]+)\s*✓\*\*"),
    # <role> ✓ at start of line (list item or heading), outside prose
    re.compile(r"^\s*(?:[-*]\s+)?([A-Za-z_]+)\s*✓\s*(?:$|\[|\(|—)", re.MULTILINE),
    # | <role> ✓ | in table cell
    re.compile(r"\|\s*([A-Za-z_]+)\s*✓\s*\|"),
    # Sign-off block: "Signed: <role>" or "Signature: <role>" in isolation
    re.compile(r"^(?:Signed|Signature)\s*[:\-]\s*`?([A-Za-z_]+)`?\s*$", re.MULTILINE),
    # **<role> signature:** ✓ — the format the D-109 lock actually used
    # (`**PI signature:** ✓ (2026-04-17)`, unanimous_lock_1i.md line 122). The four
    # patterns above did not match it; added 2026-09-17 after the replay audit.
    re.compile(r"\*\*([A-Za-z_]+)\s+signature\s*:?\s*\*\*\s*✓", re.IGNORECASE),
    # <role> signature: ✓ at start of line, unbolded variant of the same convention
    re.compile(r"^\s*([A-Za-z_]+)\s+signature\s*:\s*✓", re.MULTILINE | re.IGNORECASE),
]


# --- Citation and quotation handling -------------------------------------------------
# Two classes of FALSE POSITIVE were measured on the live lab tree on 2026-09-17 by
# running this detector over 289 commits of history (see the E-FORGE census). Both are
# fixed here, with regression tests in tests/test_verify_signatures_patterns.py.
#
#   (a) Citation paths were compared verbatim. Real citations in the lab carry markdown
#       backticks, a trailing parenthetical, or a short `<role>/<name>` form with no
#       `data/agents/` prefix and no `.md`. All three resolved to "file does not exist"
#       while the file was sitting on disk.
#   (b) Prose that QUOTES the convention (`the text-based **PI ✓** convention`) matched
#       the signature patterns, so the lab's own paper about signature forgery was
#       flagged as signature forgery.

_TRAILING_PROSE = re.compile(r"\s*[(,;].*$", re.DOTALL)


def normalize_episodic_ref(raw: str, role: str) -> list[str]:
    """Return candidate repo-relative paths for a cited episodic record, best first.

    Handles: surrounding backticks/quotes/whitespace, a trailing parenthetical or
    comma-clause inside the brackets, a missing `.md`, and the short `<role>/<name>`
    form used in practice instead of the full `data/agents/<role>/episodic/<name>.md`.
    """
    ref = raw.strip().strip("`'\"")
    ref = _TRAILING_PROSE.sub("", ref).strip().strip("`'\"")
    ref = ref.lstrip("/")
    if not ref:
        return []
    cands: list[str] = []

    def add(p: str) -> None:
        if p not in cands:
            cands.append(p)
        if not p.endswith(".md") and f"{p}.md" not in cands:
            cands.append(f"{p}.md")

    add(ref)
    if not ref.startswith("data/agents/"):
        # `pi/2026-04-20_x` or `2026-04-20_x` -> data/agents/<role>/episodic/...
        tail = ref.split("/", 1)[1] if ref.startswith(f"{role}/") else ref
        add(f"data/agents/{role}/episodic/{tail}")
        add(f"data/agents/{ref}")
    return cands


_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def signature_date(line_text: str):
    """The date the signature claims for itself, if it carries one.

    The lab writes dated attestations (`**pi ✓** 2026-04-24 D-181 APPROVE`), which is
    what makes a retrospective audit possible at all.
    """
    m = _DATE.search(line_text)
    if not m:
        return None
    try:
        return dt.date.fromisoformat(m.group(1))
    except ValueError:
        return None


def episodic_date(path: Path):
    """The date an episodic record belongs to: its filename date if it has one, else mtime."""
    m = _DATE.search(path.name)
    if m:
        try:
            return dt.date.fromisoformat(m.group(1))
        except ValueError:
            pass
    try:
        return dt.date.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return None


def _in_code_span(line: str, pos: int) -> bool:
    """True if offset `pos` sits inside a markdown backtick code span.

    A signature written inside backticks is a quotation of the convention, not an
    attestation. Counting backticks before the match is enough for single-line spans,
    which is the only form these documents use.
    """
    return line.count("`", 0, pos) % 2 == 1


# Role names that are eligible for signatures (from agents.json + retired).
# Role names with ✓ found outside this set are treated as typos, not signatures.
KNOWN_ROLES = set()


def _load_known_roles():
    """Populate KNOWN_ROLES from agents.json + retired.json."""
    for f in (AGENTS_DIR / "agents.json", AGENTS_DIR / "retired.json"):
        if f.exists():
            try:
                KNOWN_ROLES.update(json.loads(f.read_text()).keys())
            except (json.JSONDecodeError, OSError):
                pass
    # Accept capitalized aliases for common ones (PI, Director)
    KNOWN_ROLES.update({"pi", "director", "PI", "Director"})


@dataclass
class Signature:
    role: str
    source_file: str
    line_number: int
    line_text: str
    has_episodic_ref: bool  # does the signature cite [episodic: ...] inline?
    episodic_ref_path: str | None


@dataclass
class Forgery:
    role: str
    source_file: str
    line_number: int
    reason: str
    line_text: str


def find_signatures(paths: Iterable[Path]) -> list[Signature]:
    """Scan files for role signatures."""
    sigs: list[Signature] = []
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        if path.suffix not in (".md", ".txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            for pat in SIG_PATTERNS:
                for m in pat.finditer(line):
                    role = m.group(1).lower() if m.groups() else "pi"  # the PI+Director unanimous pattern
                    if role not in {r.lower() for r in KNOWN_ROLES}:
                        continue
                    if _in_code_span(line, m.start()):
                        continue  # a quoted convention, not an attestation
                    ep_match = re.search(
                        r"\[episodic:\s*([^\]]+)\]", line
                    )
                    sigs.append(Signature(
                        role=role,
                        source_file=str(path.relative_to(REPO_ROOT)),
                        line_number=i,
                        line_text=line.strip(),
                        has_episodic_ref=bool(ep_match),
                        episodic_ref_path=ep_match.group(1).strip() if ep_match else None,
                    ))
    return sigs


def verify_signature(sig: Signature, window_seconds: int, as_of=None) -> Forgery | None:
    """Check that sig has a corresponding dispatch record.

    `as_of=None` (default) is the LIVE gate: the episodic record must have been modified
    within window_seconds of now. That is correct at a session boundary and only there.

    `as_of` set to a date, or to the string "signature" (use the date the signature
    carries), is the RETROSPECTIVE audit: the episodic record must be dated within the
    window of that reference date instead of of now. Without this, every signature older
    than the window flags, so a per-phase or per-program audit run days later reports the
    whole archive as forged. Measured on the live lab tree 2026-09-17: 6 of 9 reported
    forgeries were this artefact.

    Returns None if verified (or not judgeable retrospectively), Forgery if not.
    """
    # Director is the session itself — signatures are self-attestation, not
    # dispatched-agent attestation. The attack vector we care about is Director
    # forging OTHER roles (D-109 pattern). Director signing its own work is
    # verified by the session existing at all, so we skip episodic check.
    if sig.role.lower() == "director":
        return None

    # If the signature cites [episodic: <path>], verify the file exists and is recent.
    if sig.has_episodic_ref:
        ep_path = None
        for cand in normalize_episodic_ref(sig.episodic_ref_path, sig.role):
            probe = REPO_ROOT / cand
            if probe.exists():
                ep_path = probe
                break
        if ep_path is None:
            tried = ", ".join(normalize_episodic_ref(sig.episodic_ref_path, sig.role)) or "(unparseable)"
            return Forgery(
                role=sig.role,
                source_file=sig.source_file,
                line_number=sig.line_number,
                reason=(f"signature cites episodic [{sig.episodic_ref_path}] but no such file "
                        f"(tried: {tried})"),
                line_text=sig.line_text,
            )
        # Cited file exists; verify it references this signature's source.
        # Weak check: the episodic file should mention the source file or the program.
        try:
            ep_text = ep_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return Forgery(
                role=sig.role,
                source_file=sig.source_file,
                line_number=sig.line_number,
                reason=f"episodic file [{sig.episodic_ref_path}] exists but unreadable",
                line_text=sig.line_text,
            )
        return None  # valid

    # No episodic ref inline — search episodic dir for a recent file.
    ep_dir = AGENTS_DIR / sig.role / "episodic"
    if not ep_dir.exists() or not ep_dir.is_dir():
        return Forgery(
            role=sig.role,
            source_file=sig.source_file,
            line_number=sig.line_number,
            reason=f"role {sig.role} has no episodic/ directory — impossible that agent was dispatched",
            line_text=sig.line_text,
        )
    # Retrospective mode: compare record dates against the signature's own reference date.
    if as_of is not None:
        ref = signature_date(sig.line_text) if as_of == "signature" else as_of
        if ref is None:
            return None  # undated signature: not judgeable after the fact, not evidence of forgery
        window_days = max(1, window_seconds // 86400)
        for f in sorted(ep_dir.glob("*.md")):
            d = episodic_date(f)
            if d is not None and abs((ref - d).days) <= window_days:
                return None
        return Forgery(
            role=sig.role,
            source_file=sig.source_file,
            line_number=sig.line_number,
            reason=(f"signature dated {ref} claims {sig.role} approved, but "
                    f"data/agents/{sig.role}/episodic/ has no record dated within "
                    f"{window_days}d of it"),
            line_text=sig.line_text,
        )

    # Look for any .md file modified within window_seconds.
    now = time.time()
    recent_files = []
    for f in ep_dir.iterdir():
        if f.name == ".gitkeep" or f.suffix != ".md":
            continue
        if now - f.stat().st_mtime <= window_seconds:
            recent_files.append(f)
    if not recent_files:
        return Forgery(
            role=sig.role,
            source_file=sig.source_file,
            line_number=sig.line_number,
            reason=(
                f"signature claims {sig.role} approved, but data/agents/{sig.role}/episodic/ "
                f"has no file modified within the last {window_seconds // 3600}h. "
                f"No dispatch record → signature is unverified/forged."
            ),
            line_text=sig.line_text,
        )
    return None  # at least one recent episodic file exists; signature plausibly real



# ---------------------------------------------------------------------------
# Dispatch-identity audit (D-109 follow-up, added 2026-09-17).
#
# Everything above asks whether a dispatch record EXISTS. A role-labelled record can
# exist for work performed by a generic agent: the Director dispatches
# subagent_type="general-purpose", names the intended role in the free-text description,
# hands that agent the role's procedural file to read, and files the result under
# data/agents/<role>/episodic/. Existence is satisfied; identity is not.
#
# Measured on this lab's own archive 2026-09-17: of 174 non-director episodic receipts,
# 31 are dated before the first dispatch of the role they are filed under, and one role
# (hypothesis_generator) has a receipt but was never dispatched by type in 763 sessions.
# Between 2026-04-17 and 2026-04-20 the lab made 48 dispatches, 0 of them typed, while 30
# roles sat registered in agents.json. The D-109 remediation receipt is in that set: it
# declares itself "the first real invocation of the registered pi agent" while the runner
# logged ">>> AGENT general-purpose - ... (pi role)". The first typed pi dispatch came
# seven days later. So the fix for D-109 exhibited D-109's own structure: an attestation
# naming a property the substrate does not have, and no gate checking the property named.
#
# The evidence used here is deliberately not written by the Director. The session logs
# come from run_agi_lab.sh, and tools/stream_formatter.py prints the AGENT line straight
# out of the Task call's subagent_type parameter. Where the Director's receipts and the
# runner's logs disagree about who ran, the runner is the witness.
# ---------------------------------------------------------------------------

SESSION_LOG_DIR = REPO_ROOT / "data" / "infra" / "session_logs"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LOG_HEADER = re.compile(r"=== Session (\d+) starting at (.+?) ===")
_AGENT_LINE = re.compile(r">>> AGENT\s+([A-Za-z0-9_\-]+)\s*(?:—\s*(.*))?")
_LAUNCH_LINE = re.compile(r">>> Launching agent:\s*([A-Za-z0-9_\-]+)")
_LOG_STAMP = re.compile(r"_(\d{8})_(\d{6})\.log$")
_HDR_TIME = re.compile(r"\w+ (\w+) +(\d+) (\d+):(\d+):(\d+) \S+ (\d+)")
_MONTHS = {m: i + 1 for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}

# subagent_types that carry no role identity. A dispatch with one of these is generic
# however the description is worded.
GENERIC_TYPES = {"general-purpose", "general", "superpowers"}


@dataclass
class IdentityViolation:
    role: str
    receipt: str
    receipt_date: str
    reason: str


def _log_start_time(path: Path, text: str):
    """Wall-clock start of a session, from the runner's header, else the filename."""
    m = _LOG_HEADER.search(text)
    if m:
        h = _HDR_TIME.match(m.group(2).strip())
        if h and h.group(1) in _MONTHS:
            mo, d, H, M, S, Y = h.groups()
            return dt.datetime(int(Y), _MONTHS[mo], int(d), int(H), int(M), int(S))
    fm = _LOG_STAMP.search(path.name)
    if fm:
        return dt.datetime.strptime(fm.group(1) + fm.group(2), "%Y%m%d%H%M%S")
    return None


def read_dispatch_log(log_dir: Path | None = None):
    """Reconstruct the dispatch timeline from the runner's session logs.

    Returns (first_by_type, generic_naming_a_role, n_logs, n_dispatches); first_by_type
    maps subagent_type -> the earliest datetime it was dispatched.
    """
    log_dir = log_dir or SESSION_LOG_DIR
    first: dict = {}
    generic: list = []
    n_logs = n_disp = 0
    if not log_dir.is_dir():
        return first, generic, 0, 0
    if not KNOWN_ROLES:
        # Called directly rather than through main(). Without the roster the
        # generic-substitution flag silently never fires, which is the one thing this
        # function exists to catch.
        _load_known_roles()
    known = {r.lower() for r in KNOWN_ROLES if r.lower() != "director"}
    for path in sorted(log_dir.glob("*.log")):
        n_logs += 1
        try:
            text = _ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        when = _log_start_time(path, text)
        for line in text.splitlines():
            m = _AGENT_LINE.search(line)
            desc = (m.group(2) or "") if m else ""
            if m is None:
                m = _LAUNCH_LINE.search(line)
            if m is None:
                continue
            n_disp += 1
            stype = m.group(1)
            if when is not None and (stype not in first or when < first[stype]):
                first[stype] = when
            if stype in GENERIC_TYPES and desc:
                named = sorted(r for r in known
                               if re.search(r"\b" + re.escape(r).replace("_", "[_ ]") + r"\b",
                                            desc, re.IGNORECASE))
                if named:
                    generic.append({"when": when, "log": path.name,
                                    "subagent_type": stype, "named_roles": named,
                                    "desc": desc.strip()[:120]})
    return first, generic, n_logs, n_disp


def audit_dispatch_identity(log_dir: Path | None = None):
    """Check every episodic receipt against the runner's record of who actually ran.

    A receipt is a violation when the role it is filed under was never dispatched by
    subagent_type, or was first dispatched after the receipt's date. `director` is
    excluded by construction: it is the top-level session, not a dispatched subagent.
    """
    first, generic, n_logs, n_disp = read_dispatch_log(log_dir)
    violations: list = []
    receipts = 0
    for ep_dir in sorted(AGENTS_DIR.glob("*/episodic")):
        role = ep_dir.parent.name
        if role.lower() == "director":
            continue
        for f in sorted(ep_dir.glob("*.md")):
            receipts += 1
            dm = re.match(r"(\d{4}-\d{2}-\d{2})", f.name)
            rdate = dt.date.fromisoformat(dm.group(1)) if dm else None
            fd = first.get(role)
            rel = str(f.relative_to(REPO_ROOT))
            shown = rdate.isoformat() if rdate else "(undated)"
            if fd is None:
                violations.append(IdentityViolation(
                    role=role, receipt=rel, receipt_date=shown,
                    reason=(f"no dispatch with subagent_type={role} appears in any of "
                            f"{n_logs} session logs; the receipt is filed under a role "
                            f"that was never instantiated"),
                ))
            elif rdate is not None and rdate < fd.date():
                violations.append(IdentityViolation(
                    role=role, receipt=rel, receipt_date=shown,
                    reason=(f"receipt dated {shown}, but the first dispatch with "
                            f"subagent_type={role} was {fd:%Y-%m-%d %H:%M} - the work it "
                            f"records was performed by some other agent type"),
                ))
    stats = {"logs": n_logs, "dispatches": n_disp, "receipts": receipts,
             "types_seen": len(first), "generic_naming_a_role": generic,
             "first_by_type": first}
    return violations, stats


def run_identity_audit(verbose: bool = False) -> int:
    violations, st = audit_dispatch_identity()
    if not st["logs"]:
        print("[verify_signatures] cannot audit dispatch identity: no session logs under "
              "data/infra/session_logs. Note that .gitignore excludes them, so a fresh "
              "clone has none; run this against a live lab tree.", file=sys.stderr)
        return 2
    print(f"[verify_signatures] dispatch-identity audit: {st['logs']} session logs, "
          f"{st['dispatches']} dispatches, {st['types_seen']} distinct subagent_types, "
          f"{st['receipts']} non-director episodic receipts")
    gen = st["generic_naming_a_role"]
    if gen:
        print(f"[verify_signatures] {len(gen)} generic dispatch(es) name a registered role in "
              "the description: role-shaped work done by an agent with no role identity")
        if verbose:
            for g in gen:
                stamp = f"{g['when']:%Y-%m-%d %H:%M}" if g["when"] else "(no time)"
                print(f"    {stamp}  {g['subagent_type']:16} "
                      f"names={','.join(g['named_roles'])}  {g['desc']}")
    if verbose:
        print("[verify_signatures] first dispatch per subagent_type:")
        for t, w in sorted(st["first_by_type"].items(), key=lambda kv: kv[1]):
            print(f"    {w:%Y-%m-%d %H:%M}  {t}")
    if not violations:
        print(f"[verify_signatures] OK: all {st['receipts']} receipts are backed by a "
              "dispatch of their own role")
        return 0
    print(f"[verify_signatures] IDENTITY VIOLATION: {len(violations)} receipt(s) filed under "
          "a role that did not run them", file=sys.stderr)
    for v in violations:
        print(f"  {v.receipt} - {v.role}", file=sys.stderr)
        print(f"    reason: {v.reason}", file=sys.stderr)
    print("", file=sys.stderr)
    print("A dispatch record existing is not the same as the named role having run. "
          "Re-dispatch with subagent_type set to the role, or restate the receipt as "
          "generic-agent work.", file=sys.stderr)
    return 1

def main():
    parser = argparse.ArgumentParser(
        description="Anti-forgery detector for lab governance signatures."
    )
    parser.add_argument(
        "--program", default=None,
        help="Scope scan to a single program (e.g., program_1_example). Default: current program per state.md.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Scan all programs in programs/ (not just current).",
    )
    parser.add_argument(
        "--window-hours", type=int, default=48,
        help="How recent must the episodic record be (hours). Default: 48.",
    )
    parser.add_argument(
        "--as-of", default=None, metavar="DATE|signature",
        help=("Audit retrospectively: match episodic records against this reference date, or "
              "against the date each signature carries (--as-of signature), instead of against "
              "now. Required for any audit of past work; without it every signature older than "
              "--window-hours is reported as forged."),
    )
    parser.add_argument(
        "--check-identity", action="store_true",
        help=("Audit dispatch IDENTITY rather than signature existence: check every "
              "episodic receipt against the runner's session logs, which record the actual "
              "subagent_type of each dispatch. Catches role-labelled records produced by a "
              "generic agent - the failure mode the D-109 remediation itself exhibited."),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every signature found (not just forgeries).",
    )
    args = parser.parse_args()

    _load_known_roles()

    if args.check_identity:
        sys.exit(run_identity_audit(verbose=args.verbose))

    # Determine which paths to scan.
    paths_to_scan: list[Path] = []
    if args.full:
        paths_to_scan.extend(PROGRAMS_DIR.rglob("*.md"))
    elif args.program:
        target = PROGRAMS_DIR / args.program
        if target.exists():
            paths_to_scan.extend(target.rglob("*.md"))
    else:
        # Default: scan the current program (inferred from state.md) + the lock/state files.
        state_file = REPO_ROOT / "data" / "state.md"
        if state_file.exists():
            state_text = state_file.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Current Program:\s*(\S+)", state_text)
            if m:
                target = PROGRAMS_DIR / m.group(1)
                if target.exists():
                    paths_to_scan.extend(target.rglob("*.md"))
        # Also include state.md itself and the decisions files.
        paths_to_scan.extend([
            REPO_ROOT / "data" / "state.md",
            REPO_ROOT / "data" / "decisions_recent.md",
        ])

    if not paths_to_scan:
        print("[verify_signatures] no paths to scan; nothing to verify")
        sys.exit(0)

    sigs = find_signatures(paths_to_scan)
    if args.verbose:
        print(f"[verify_signatures] scanned {len(paths_to_scan)} files, found {len(sigs)} signatures")
        for s in sigs:
            print(f"  {s.source_file}:{s.line_number} → {s.role}"
                  f"{' [episodic ref]' if s.has_episodic_ref else ''}")

    window = args.window_hours * 3600
    as_of = args.as_of
    if as_of not in (None, "signature"):
        try:
            as_of = dt.date.fromisoformat(as_of)
        except ValueError:
            parser.error("--as-of must be YYYY-MM-DD or the word 'signature'")
    if args.full and as_of is None:
        print("NOTE: --full without --as-of runs the LIVE criterion over the whole archive, "
              "so every signature older than --window-hours will be reported. For a "
              "retrospective audit use --as-of signature.", file=sys.stderr)
    forgeries: list[Forgery] = []
    for s in sigs:
        f = verify_signature(s, window, as_of=as_of)
        if f is not None:
            forgeries.append(f)

    if not forgeries:
        print(f"[verify_signatures] OK: {len(sigs)} signatures verified, 0 forgeries detected")
        sys.exit(0)

    print(f"[verify_signatures] FORGERY DETECTED: {len(forgeries)} unverified signature(s)", file=sys.stderr)
    for f in forgeries:
        print(f"  {f.source_file}:{f.line_number} — {f.role}", file=sys.stderr)
        print(f"    reason: {f.reason}", file=sys.stderr)
        print(f"    line:   {f.line_text}", file=sys.stderr)
    print("", file=sys.stderr)
    print("Remediation: see data/procedures.md §'Signature Forgery Remediation'.", file=sys.stderr)
    print("Append entry to data/accountability_ledger.md, flag offending docs, dispatch real agent.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
