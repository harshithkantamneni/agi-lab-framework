#!/usr/bin/env python3
"""tools/verify_signatures.py — anti-forgery detector for AGI lab governance.

A sign-off counts only if it is a structured token bound to a receipt that the runner
recorded from a real dispatch (tools/dispatch_ledger.py). The token format and the rule for
roles are in data/agents/_shared/signoff_contract.md. What this does and does not close is
in audits/README.md under "Threat model".

Exit codes:
  0  at least one sign-off or episodic record was checked, and all of them hold
  1  forgery detected: a sign-off with no matching receipt, an unrecognized sign-off shape,
     a role-filed record with no dispatch of that role, or a required sign-off missing
  2  cannot judge: evidence missing, unauthenticated or tampered, or a script error
  3  nothing to verify: the scan ran and found nothing new. Never reported as OK.

Usage:
  python3 tools/verify_signatures.py [--program <name>] [--window-hours N] [--unauthenticated]
  python3 tools/verify_signatures.py --check-identity
  python3 tools/verify_signatures.py --gate --check-identity --run-id R --nonce N   # runner only
  python3 tools/verify_signatures.py --emit-signoff --role pi --target programs/x/lock.md
  python3 tools/verify_signatures.py --init-evidence | --preflight | --rebaseline   # operator
  python3 tools/verify_signatures.py --legacy-episodic --full --as-of signature     # archive
  python3 tools/verify_signatures.py --check-identity --legacy-logs                 # archive

Added 2026-04-17 (D-110) after D-109 audit revealed 5 forged PI signatures.
See data/procedures.md §"Signature Forgery Remediation".
"""
from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# The runner starts this file with `python3 -I`, so a module an agent plants next to it
# cannot shadow the standard library. Appending here keeps that order for the one sibling
# this file needs, and tools/dispatch_ledger.py is frozen for agents by the shipped settings.
if __package__ in (None, ""):
    _TOOLS = str(Path(__file__).resolve().parent)
    if _TOOLS not in sys.path:
        sys.path.append(_TOOLS)
    import dispatch_ledger as dl
else:
    from tools import dispatch_ledger as dl

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

# ---------------------------------------------------------------------------
# Receipt-bound sign-offs (November 2026).
#
# Everything above reads evidence an agent can write: a role's episodic folder, or a log
# that assistant text is printed into. It stays for auditing the archive, behind explicit
# flags. The default check below reads only the runner's ledger.
# ---------------------------------------------------------------------------

EXIT_VERIFIED = 0
EXIT_FORGERY = 1
EXIT_CANNOT_JUDGE = 2
EXIT_EMPTY = 3

# The one recognized sign-off shape. The target is explicit so a line in a decision log can
# cite the lock it refers to, and so a required sign-off can demand target == the document.
_TOKEN_LOOSE = re.compile(r"\[\s*sign[- ]?off\b[^\]\n]*\]", re.IGNORECASE)
_TOKEN_STRICT = re.compile(
    r"\[signoff role=([A-Za-z0-9_\-]+) target=([^\s\]]+) verdict=(APPROVE|MODIFY|REJECT)"
    r" run=([A-Za-z0-9_.\-]+) receipt=([^\s\]]+)\]"
)
SELF_RECEIPT = "self"

# Where sign-offs live. `programs/` must exist: a lab with no programs directory is
# misconfigured, not clean. The two generated briefs quote other files, so they are skipped
# by not being listed; the consolidator moves log entries out to data/archives/, which is
# also outside this list.
REQUIRED_ROOTS = ("programs",)
CORPUS_FILES = ("data/state.md", "data/memories/log.md", "data/memories/current.md")
CORPUS_GLOBS = ("data/decisions*.md",)
REQUIREMENTS_FILE = "data/agents/_shared/signoff_requirements.json"
DRAFT_SCAN_LINES = 30

# The net for gap 4: lines that claim a role's approval without the token. It is deliberately
# wide, and it cannot be complete, because prose is an open set. The hard guarantee is the
# allow-list in REQUIREMENTS_FILE: a document that needs a sign-off is unsigned without a
# verified token, however its text is phrased.
_ATTEST_WORDS = (
    r"approv(?:e|es|ed|al)|sign(?:s|ed)?[ -]?off|signed|signs|co-?sign(?:s|ed)?"
    r"|ratif(?:y|ies|ied)|attest(?:s|ed)?|certif(?:y|ies|ied)|endors(?:e|es|ed)"
    r"|concur(?:s|red)?|agree(?:s|d)?|accept(?:s|ed)|lgtm"
)
_ATTEST = rf"(?:\b(?:{_ATTEST_WORDS})\b|signature\s*[:(\-–—]|[✓✔✅☑])"
_NEG_BEFORE = re.compile(
    r"\b(?:pending|await(?:ing|s)?|requires?|required|needs?|needed|request(?:s|ed|ing)?"
    r"|before|until|unless|if|once|when|without|no|not|missing|lack(?:s|ing)?|for|seek(?:s|ing)?"
    r"|ask(?:s|ing)?|get(?:s|ting)?|obtain(?:s|ing)?|dispatch(?:es|ing)?|whether|must|should"
    r"|shall|will|to)\s+(?:(?:the|a|an)\s+)?$",
    re.IGNORECASE,
)
_NEG_GAP = re.compile(
    r"\b(?:not|never|no|won't|cannot|can't|didn't|doesn't|hasn't|must|should|shall|will|would"
    r"|may|might|to|for|needs?|requires?|refus\w+|declin\w+|reject\w+)\b",
    re.IGNORECASE,
)
_NEG_AFTER = re.compile(
    r"\s*(?:is\s+|was\s+|are\s+)?(?:pending|required?|requires|needed|outstanding|missing|requested"
    r"|blocked|awaited|gate|rule|format|convention|block|requirement|process|flow|step|contract)\b",
    re.IGNORECASE,
)


def _strip_code_spans(line: str) -> str:
    return re.sub(r"`[^`]*`", " ", line)


def _claim_patterns() -> tuple:
    roles = sorted((r.lower() for r in KNOWN_ROLES if r.lower() != "director"), key=len, reverse=True)
    alts = "|".join(re.escape(r).replace("_", "[_ ]") for r in roles) or "pi"
    role_first = re.compile(rf"\b(?P<role>{alts})\b(?P<gap>[^.;!?\n]{{0,30}}?)(?P<attest>{_ATTEST})",
                            re.IGNORECASE)
    word_first = re.compile(
        rf"\b(?:{_ATTEST_WORDS})\b\s*(?:(?:by|from|of)\s+(?:the\s+)?|[:\-–—]\s*(?:the\s+)?)"
        rf"(?P<role>{alts})\b", re.IGNORECASE)
    symbol_first = re.compile(rf"(?:[✓✔✅☑]|/s/)\s*(?:the\s+)?(?P<role>{alts})\b", re.IGNORECASE)
    return role_first, word_first, symbol_first


def _canonical_role(text: str) -> str:
    return text.lower().replace(" ", "_")


def find_claims(visible: str, patterns: tuple) -> set:
    """Roles this line claims approved, outside any token. `visible` has code spans and tokens removed."""
    claims: set = set()
    known = {r.lower() for r in KNOWN_ROLES}
    for pat in SIG_PATTERNS:
        for m in pat.finditer(visible):
            role = m.group(1).lower()
            if role in known and role != "director":
                claims.add(role)
    role_first, word_first, symbol_first = patterns
    for m in role_first.finditer(visible):
        if _NEG_BEFORE.search(visible[: m.start()]) or _NEG_GAP.search(m.group("gap")):
            continue
        if _NEG_AFTER.match(visible[m.end():]):
            continue
        claims.add(_canonical_role(m.group("role")))
    for pat in (word_first, symbol_first):
        for m in pat.finditer(visible):
            if not _NEG_BEFORE.search(visible[: m.start()]):
                claims.add(_canonical_role(m.group("role")))
    return claims


@dataclass
class SignOff:
    role: str
    target: str
    verdict: str
    run: str
    receipt: str


@dataclass
class Hit:
    rel: str
    lineno: int
    line: str
    tokens: list = field(default_factory=list)
    malformed: list = field(default_factory=list)
    claims: set = field(default_factory=set)


def corpus_paths(program: str | None = None) -> list:
    """Files in scope. Raises EvidenceError when the scan could not cover what it must."""
    if program is not None:
        target = REPO_ROOT / "programs" / program
        if not target.is_dir():
            raise dl.EvidenceError(f"program {program!r} not found under programs/")
        roots = [target]
    else:
        missing = [r for r in REQUIRED_ROOTS if not (REPO_ROOT / r).is_dir()]
        if missing:
            raise dl.EvidenceError(f"no paths to scan: {', '.join(missing)} is missing")
        roots = [REPO_ROOT / r for r in REQUIRED_ROOTS]
    paths: list = []
    for root in roots:
        paths.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in (".md", ".txt"))
    if program is None:
        paths.extend(REPO_ROOT / rel for rel in CORPUS_FILES if (REPO_ROOT / rel).is_file())
        for pattern in CORPUS_GLOBS:
            paths.extend(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    return sorted(set(paths))


def scan_for_signoffs(paths: Iterable[Path]) -> list:
    """Every line carrying a token, a malformed token, or a claim of a role's approval."""
    patterns = _claim_patterns()
    hits: list = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise dl.EvidenceError(f"cannot read {path}: {e}") from e
        rel = path.relative_to(REPO_ROOT).as_posix()
        fenced = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if fenced:
                continue
            visible = _strip_code_spans(line)
            tokens, malformed = [], []
            for m in _TOKEN_LOOSE.finditer(visible):
                strict = _TOKEN_STRICT.fullmatch(m.group(0))
                if strict is None:
                    malformed.append(m.group(0))
                else:
                    tokens.append(SignOff(strict.group(1).lower(), strict.group(2), strict.group(3),
                                          strict.group(4), strict.group(5)))
            claims = find_claims(_TOKEN_LOOSE.sub(" ", visible), patterns)
            if tokens or malformed or claims:
                hits.append(Hit(rel, lineno, line.strip(), tokens, malformed, claims))
    return hits


def check_signoff(token: SignOff, ledger: dl.Ledger, now: float, window_seconds: int) -> str | None:
    """None if the token is backed by a matching receipt, else why not."""
    if token.role not in {r.lower() for r in KNOWN_ROLES}:
        return f"role {token.role!r} is not registered"
    target = dl.normalize_target(token.target)
    if target is None or not (REPO_ROOT / target).is_file():
        return f"target {token.target!r} is not a file in this tree"
    if token.role == "director":
        # The Director is the session itself: its own sign-off is self-attestation, and the
        # attack this check exists for is the Director signing for someone else.
        if token.run not in ledger.runs:
            return f"director sign-off names run {token.run!r}, which has no ledger"
        if token.receipt != SELF_RECEIPT:
            return f"director sign-off must use receipt={SELF_RECEIPT}"
        return None
    r = ledger.receipts.get(token.receipt)
    if r is None:
        return f"no receipt {token.receipt!r}: no recorded dispatch produced this sign-off"
    if r.role != token.role:
        return f"receipt {token.receipt} is a dispatch of {r.role!r}, not {token.role!r}"
    if r.run != token.run:
        return f"receipt {token.receipt} belongs to run {r.run!r}, not {token.run!r}"
    if r.parent is not None:
        return f"receipt {token.receipt} is a nested dispatch; only the Director's dispatches count"
    if r.completed_at is None:
        return f"receipt {token.receipt} never completed"
    if r.is_error:
        return f"receipt {token.receipt} ended in an error"
    if [target, token.verdict] not in r.attests:
        return (f"the reply to receipt {token.receipt} does not contain "
                f"'SIGNOFF target={target} verdict={token.verdict}'")
    if [target, token.verdict] in r.dictated:
        return f"the dispatch prompt for receipt {token.receipt} dictated this exact verdict"
    age = now - r.completed_at
    if age > window_seconds:
        return (f"receipt {token.receipt} completed {age / 3600:.1f}h ago, "
                f"outside the {window_seconds // 3600}h window")
    return None


def _load_requirements() -> dict:
    path = REPO_ROOT / REQUIREMENTS_FILE
    if not path.is_file():
        return {"rules": [], "draft_marker": "status: draft"}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise dl.EvidenceError(f"cannot read {REQUIREMENTS_FILE}: {e}") from e


def _is_draft(path: Path, marker: str) -> bool:
    try:
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:DRAFT_SCAN_LINES]
    except OSError:
        return False
    return any(line.strip() == marker for line in head)


@dataclass
class Outcome:
    files: int = 0
    verified: int = 0
    prior: int = 0
    identity_checked: int = 0
    forgeries: list = field(default_factory=list)
    identity: list = field(default_factory=list)
    generic_naming_role: list = field(default_factory=list)
    new_lines: set = field(default_factory=set)
    new_files: set = field(default_factory=set)

    @property
    def violated(self) -> bool:
        return bool(self.forgeries or self.identity)


def check_tree(ledger: dl.Ledger, accepted: dict, window_seconds: int, program: str | None = None,
               scan_signoffs: bool = True, check_identity: bool = False, now: float | None = None) -> Outcome:
    now = time.time() if now is None else now
    out = Outcome()
    if scan_signoffs:
        paths = corpus_paths(program)
        out.files = len(paths)
        hits = scan_for_signoffs(paths)
        good: dict = {}
        for hit in hits:
            key = dl.line_key(hit.rel, hit.line)
            if key in accepted["lines"]:
                out.prior += 1
                # Verified by an earlier gate. It still counts toward a required document
                # that is being signed across sessions.
                for token in hit.tokens:
                    good.setdefault(hit.rel, set()).add((token.role, dl.normalize_target(token.target),
                                                         token.verdict))
                continue
            problems = [f"unrecognized sign-off shape: {m}" for m in hit.malformed]
            backed: set = set()
            for token in hit.tokens:
                reason = check_signoff(token, ledger, now, window_seconds)
                if reason is None:
                    out.verified += 1
                    backed.add(token.role)
                    good.setdefault(hit.rel, set()).add((token.role, dl.normalize_target(token.target),
                                                         token.verdict))
                else:
                    problems.append(reason)
            for role in sorted(hit.claims - backed):
                problems.append(f"claims approval by {role} without a verified [signoff ...] token "
                                f"(unrecognized sign-off shape)")
            if problems:
                roles = sorted({t.role for t in hit.tokens} | hit.claims) or ["?"]
                out.forgeries.append(Forgery(",".join(roles), hit.rel, hit.lineno, "; ".join(problems), hit.line))
            else:
                out.new_lines.add(key)
        requirements = _load_requirements()
        marker = requirements.get("draft_marker", "status: draft")
        for path in paths:
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in accepted["files"]:
                continue
            for rule in requirements.get("rules", []):
                if not fnmatch.fnmatchcase(rel, rule.get("glob", "")) or _is_draft(path, marker):
                    continue
                missing = [r for r in rule.get("roles", []) if (r, rel, "APPROVE") not in good.get(rel, set())]
                if missing:
                    out.forgeries.append(Forgery(
                        ",".join(missing), rel, 0,
                        f"this document needs a verified APPROVE sign-off from {', '.join(missing)} "
                        f"(rule {rule.get('glob')}), or the line '{marker}' in its first "
                        f"{DRAFT_SCAN_LINES} lines", "(required sign-off)"))
                else:
                    out.new_files.add(rel)
    if check_identity:
        known = {r.lower() for r in KNOWN_ROLES if r.lower() != "director"}
        for receipt in ledger.receipts.values():
            if receipt.role in GENERIC_TYPES and receipt.parent is None:
                named = sorted(r for r in known if re.search(
                    r"\b" + re.escape(r).replace("_", "[_ ]") + r"\b", receipt.description, re.IGNORECASE))
                if named:
                    out.generic_naming_role.append((receipt.receipt, named, receipt.description))
        for ep_dir in sorted(AGENTS_DIR.glob("*/episodic")):
            role = ep_dir.parent.name
            if role.lower() == "director":
                continue
            for f in sorted(ep_dir.glob("*.md")):
                rel = f.relative_to(REPO_ROOT).as_posix()
                if rel in accepted["files"]:
                    continue
                out.identity_checked += 1
                backed = any(
                    r.role == role and r.parent is None and r.completed_at is not None
                    and not r.is_error and now - r.completed_at <= window_seconds
                    for r in ledger.receipts.values())
                if backed:
                    out.new_files.add(rel)
                else:
                    out.identity.append(IdentityViolation(
                        role=role, receipt=rel, receipt_date="(new since last accepted gate)",
                        reason=(f"the ledger holds no completed dispatch with subagent_type={role} "
                                f"inside the {window_seconds // 3600}h window")))
    return out


def report(out: Outcome, label_ok: bool = True) -> int:
    for receipt, named, desc in out.generic_naming_role:
        print(f"[verify_signatures] NOTICE: generic dispatch {receipt} names {','.join(named)} in "
              f"its description ({desc[:80]}); it counts as no role's dispatch")
    if out.forgeries:
        print(f"[verify_signatures] FORGERY DETECTED: {len(out.forgeries)} unverified signature(s)", file=sys.stderr)
        for f in out.forgeries:
            print(f"  {f.source_file}:{f.line_number} — {f.role}", file=sys.stderr)
            print(f"    reason: {f.reason}", file=sys.stderr)
            print(f"    line:   {f.line_text}", file=sys.stderr)
    if out.identity:
        print(f"[verify_signatures] IDENTITY VIOLATION: {len(out.identity)} receipt(s) filed under "
              "a role that did not run them", file=sys.stderr)
        for v in out.identity:
            print(f"  {v.receipt} - {v.role}", file=sys.stderr)
            print(f"    reason: {v.reason}", file=sys.stderr)
    if out.violated:
        print("", file=sys.stderr)
        print("Remediation: dispatch the real role, have it return its SIGNOFF line, and cite "
              "the receipt. See data/agents/_shared/signoff_contract.md and "
              "data/procedures.md §'Signature Forgery Remediation'.", file=sys.stderr)
        return EXIT_FORGERY
    if out.verified:
        print(f"[verify_signatures] OK: {out.verified} signatures verified, 0 forgeries detected "
              f"({out.identity_checked} new episodic record(s) backed by a dispatch, "
              f"{out.prior} line(s) accepted earlier, {out.files} file(s) scanned)")
        return EXIT_VERIFIED
    if out.identity_checked:
        # Printing "OK: 0 signatures verified" here would repeat what the audit found in the
        # evaluator reports: a pass that does not say how little it checked.
        print(f"[verify_signatures] OK: {out.identity_checked} new episodic record(s) backed by a "
              f"dispatch. No new sign-offs ({out.prior} line(s) accepted earlier, "
              f"{out.files} file(s) scanned).")
        return EXIT_VERIFIED
    print(f"[verify_signatures] EMPTY: scanned {out.files} file(s), found nothing new to verify "
          f"({out.prior} line(s) accepted earlier). This is not a pass over checked attestations.")
    return EXIT_EMPTY


def _open_evidence(unauthenticated: bool) -> tuple:
    """(directory, key, ledger, accepted). Raises EvidenceError if any of it cannot be trusted."""
    directory = dl.evidence_dir(REPO_ROOT)
    key = None
    if unauthenticated:
        print("[verify_signatures] UNAUTHENTICATED: receipt HMACs not checked. This result is "
              "advisory; only the runner's gate holds the key.", file=sys.stderr)
    else:
        key = dl.load_key(directory)
    return directory, key, dl.read_ledger(directory, key), dl.load_accepted(directory, key)


def run_receipt_check(args) -> int:
    try:
        _d, _k, ledger, accepted = _open_evidence(args.unauthenticated)
        identity_only = args.check_identity and not args.full and not args.program
        if identity_only and not ledger.runs:
            raise dl.EvidenceError("the ledger holds no runs, so dispatch identity cannot be judged")
        out = check_tree(ledger, accepted, args.window_hours * 3600, program=args.program,
                         scan_signoffs=not identity_only, check_identity=args.check_identity)
    except dl.EvidenceError as e:
        print(f"[verify_signatures] CANNOT JUDGE: {e}", file=sys.stderr)
        return EXIT_CANNOT_JUDGE
    return report(out)


def run_gate(args) -> int:
    """The runner's clean-exit check. Prints a nonce-bound verdict only if it got that far."""
    if not args.run_id or not args.nonce:
        print("[verify_signatures] CANNOT JUDGE: --gate needs --run-id and --nonce", file=sys.stderr)
        return EXIT_CANNOT_JUDGE
    try:
        directory, key, ledger, accepted = _open_evidence(False)
        info = ledger.runs.get(args.run_id)
        if info is None:
            raise dl.EvidenceError(f"no ledger for run {args.run_id}: the dispatch recorder did not start")
        if not info.sealed:
            raise dl.EvidenceError(f"ledger for run {args.run_id} has no seal: the dispatch recorder "
                                   "did not reach the end of the stream")
        out = check_tree(ledger, accepted, args.window_hours * 3600, check_identity=args.check_identity)
        rc = report(out)
        status = {EXIT_VERIFIED: "VERIFIED", EXIT_FORGERY: "FORGERY", EXIT_EMPTY: "EMPTY"}[rc]
        missed = [r for r in ledger.runs if r != args.run_id and not dl.verdict_exists(directory, r)]
        for r in missed:
            dl.write_verdict(directory, key, r, status="NEVER_GATED", noticed_by=args.run_id)
        if missed:
            print(f"[verify_signatures] NOTICE: the gate never reached a verdict for {len(missed)} "
                  f"earlier run(s): {', '.join(sorted(missed))}. Their new lines are checked now.")
        # Lines and records that verified stay verified even when another line in the tree
        # failed. Anything that failed is not added, so it keeps blocking until it is fixed.
        dl.save_accepted(directory, key, accepted["lines"] | out.new_lines,
                         accepted["files"] | out.new_files)
        dl.write_verdict(directory, key, args.run_id, status=status, signoffs=out.verified,
                         episodic=out.identity_checked, files=out.files,
                         violations=len(out.forgeries) + len(out.identity), sandbox=info.sandbox)
    except (dl.EvidenceError, OSError) as e:
        print(f"[verify_signatures] CANNOT JUDGE: {e}", file=sys.stderr)
        return EXIT_CANNOT_JUDGE
    if info.sandbox != "on":
        print(f"[verify_signatures] NOTICE: this run recorded sandbox={info.sandbox}, so the key "
              "and ledger were not protected from the agents' shell commands.")
    print(f"[verify_signatures] GATE-VERDICT nonce={args.nonce} status={status} run={args.run_id} "
          f"signoffs={out.verified} episodic={out.identity_checked} files={out.files} sandbox={info.sandbox}")
    return rc


def _all_current_lines() -> tuple:
    """Everything sign-off-shaped now in the tree, for an operator baseline."""
    paths = corpus_paths() if all((REPO_ROOT / r).is_dir() for r in REQUIRED_ROOTS) else []
    hits = scan_for_signoffs(paths)
    lines = {dl.line_key(h.rel, h.line) for h in hits}
    files = {p.relative_to(REPO_ROOT).as_posix() for p in paths}
    files |= {f.relative_to(REPO_ROOT).as_posix() for f in AGENTS_DIR.glob("*/episodic/*.md")}
    return lines, files


def run_init_evidence() -> int:
    directory = dl.evidence_dir(REPO_ROOT)
    if dl.key_path(directory).exists() or (directory / dl.ACCEPTED_NAME).exists():
        print(f"[verify_signatures] CANNOT JUDGE: evidence already exists at {directory}. "
              "--init-evidence never overwrites it.", file=sys.stderr)
        return EXIT_CANNOT_JUDGE
    dl.init_key(directory)
    lines, files = _all_current_lines()
    dl.save_accepted(directory, dl.load_key(directory), lines, files)
    print(f"[verify_signatures] evidence created at {directory}: {len(lines)} existing sign-off "
          f"line(s) and {len(files)} file(s) accepted as a legacy baseline, not verified.")
    return EXIT_VERIFIED


def run_rebaseline() -> int:
    directory = dl.evidence_dir(REPO_ROOT)
    try:
        key = dl.load_key(directory)
    except dl.EvidenceError as e:
        print(f"[verify_signatures] CANNOT JUDGE: {e}", file=sys.stderr)
        return EXIT_CANNOT_JUDGE
    lines, files = _all_current_lines()
    dl.save_accepted(directory, key, lines, files)
    print(f"[verify_signatures] operator baseline: {len(lines)} line(s) and {len(files)} file(s) "
          "accepted without verification.")
    return EXIT_VERIFIED


def run_preflight() -> int:
    """Runner startup. Creates evidence only on a tree with nothing to launder."""
    directory = dl.evidence_dir(REPO_ROOT)
    has_key = dl.key_path(directory).exists()
    has_accepted = (directory / dl.ACCEPTED_NAME).exists()
    try:
        if has_key and has_accepted:
            key = dl.load_key(directory)
            dl.read_ledger(directory, key)
            dl.load_accepted(directory, key)
            print(f"[verify_signatures] preflight OK: evidence at {directory}")
            return EXIT_VERIFIED
        if has_key or has_accepted:
            raise dl.EvidenceError(f"evidence at {directory} is incomplete; an operator must look at it")
        lines, _files = _all_current_lines()
        episodic = [f for f in AGENTS_DIR.glob("*/episodic/*.md")]
        if lines or episodic:
            raise dl.EvidenceError(
                f"this tree holds {len(lines)} sign-off line(s) and {len(episodic)} episodic record(s) "
                f"but no evidence directory. Run once as the operator: "
                f"python3 tools/verify_signatures.py --init-evidence")
        return run_init_evidence()
    except (dl.EvidenceError, OSError) as e:
        print(f"[verify_signatures] CANNOT JUDGE: {e}", file=sys.stderr)
        return EXIT_CANNOT_JUDGE


def run_emit_signoff(args) -> int:
    """Print the exact line to paste. A convenience; the gate does not trust this output."""
    target = dl.normalize_target(args.target or "")
    if not args.role or target is None:
        print("--emit-signoff needs --role and a repo-relative --target", file=sys.stderr)
        return EXIT_CANNOT_JUDGE
    role = args.role.lower()
    try:
        ledger = dl.read_ledger(dl.evidence_dir(REPO_ROOT), None)
    except dl.EvidenceError as e:
        print(f"[verify_signatures] CANNOT JUDGE: {e}", file=sys.stderr)
        return EXIT_CANNOT_JUDGE
    if role == "director":
        run = ledger.latest_run()
        if run is None or args.verdict is None:
            print("--emit-signoff --role director needs --verdict and a ledger run", file=sys.stderr)
            return EXIT_FORGERY
        print(f"[signoff role=director target={target} verdict={args.verdict} run={run} receipt={SELF_RECEIPT}]")
        return EXIT_VERIFIED
    matches = [r for r in ledger.receipts.values()
               if r.role == role and r.parent is None and r.completed_at is not None and not r.is_error
               and any(a[0] == target for a in r.attests)]
    if not matches:
        print(f"no completed dispatch of {role} returned a SIGNOFF line for {target}. Dispatch the "
              "role in the foreground and have it end its reply with that line.", file=sys.stderr)
        return EXIT_FORGERY
    latest = max(matches, key=lambda r: r.completed_at)
    verdict = next(a[1] for a in reversed(latest.attests) if a[0] == target)
    print(f"[signoff role={role} target={target} verdict={verdict} run={latest.run} receipt={latest.receipt}]")
    return EXIT_VERIFIED


def run_legacy_episodic(args, parser) -> int:
    """Audit old free-text signatures against episodic files. Archive use only."""
    print("[verify_signatures] LEGACY MODE: episodic files are written by agents, so this cannot "
          "show a dispatch happened. The gate never uses this mode.", file=sys.stderr)
    paths_to_scan: list = []
    if args.full:
        paths_to_scan.extend(PROGRAMS_DIR.rglob("*.md"))
    elif args.program:
        target = PROGRAMS_DIR / args.program
        if target.exists():
            paths_to_scan.extend(target.rglob("*.md"))
    else:
        state_file = REPO_ROOT / "data" / "state.md"
        if state_file.exists():
            state_text = state_file.read_text(encoding="utf-8", errors="replace")
            m = re.search(r"Current Program:\s*(\S+)", state_text)
            if m:
                target = PROGRAMS_DIR / m.group(1)
                if target.exists():
                    paths_to_scan.extend(target.rglob("*.md"))
        paths_to_scan.extend([REPO_ROOT / "data" / "state.md", REPO_ROOT / "data" / "decisions_recent.md"])
    if not paths_to_scan:
        print("[verify_signatures] no paths to scan; nothing to verify", file=sys.stderr)
        return EXIT_CANNOT_JUDGE
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
    forgeries = [f for f in (verify_signature(s, window, as_of=as_of) for s in sigs) if f is not None]
    if not sigs:
        print(f"[verify_signatures] EMPTY: scanned {len(paths_to_scan)} file(s), 0 signatures found.")
        return EXIT_EMPTY
    if not forgeries:
        print(f"[verify_signatures] OK: {len(sigs)} signatures verified, 0 forgeries detected")
        return EXIT_VERIFIED
    print(f"[verify_signatures] FORGERY DETECTED: {len(forgeries)} unverified signature(s)", file=sys.stderr)
    for f in forgeries:
        print(f"  {f.source_file}:{f.line_number} — {f.role}", file=sys.stderr)
        print(f"    reason: {f.reason}", file=sys.stderr)
        print(f"    line:   {f.line_text}", file=sys.stderr)
    return EXIT_FORGERY


def main():
    parser = argparse.ArgumentParser(
        description="Anti-forgery detector for lab governance signatures."
    )
    parser.add_argument(
        "--program", default=None,
        help="Scope the scan to one program (e.g., program_1_example). Default: the whole corpus.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Scan the whole corpus. This is already the default; kept for existing callers.",
    )
    parser.add_argument(
        "--window-hours", type=int, default=48,
        help="How recently a receipt must have completed for a new sign-off to cite it. Default: 48.",
    )
    parser.add_argument(
        "--check-identity", action="store_true",
        help=("Check each new role-filed episodic record against the ledger's record of who "
              "actually ran. Alone, runs only this check. With --gate, runs both."),
    )
    parser.add_argument("--gate", action="store_true", help="Runner only: clean-exit check with a verdict.")
    parser.add_argument("--run-id", default=None, help="With --gate: the run being closed.")
    parser.add_argument("--nonce", default=None, help="With --gate: echoed in the verdict line.")
    parser.add_argument(
        "--unauthenticated", action="store_true",
        help="For agents, who are denied the key: check chains and bindings, not HMACs. Advisory.",
    )
    parser.add_argument("--emit-signoff", action="store_true", help="Print the token for a recorded attestation.")
    parser.add_argument("--role", default=None, help="With --emit-signoff.")
    parser.add_argument("--target", default=None, help="With --emit-signoff.")
    parser.add_argument("--verdict", choices=dl.VERDICTS, default=None, help="With --emit-signoff --role director.")
    parser.add_argument("--preflight", action="store_true", help="Runner startup check.")
    parser.add_argument("--init-evidence", action="store_true", help="Operator: create the key and baseline once.")
    parser.add_argument("--rebaseline", action="store_true", help="Operator: accept all current lines unverified.")
    parser.add_argument(
        "--legacy-episodic", action="store_true",
        help="Archive audit: old free-text signatures against episodic files. Cannot prove dispatch.",
    )
    parser.add_argument(
        "--as-of", default=None, metavar="DATE|signature",
        help=("With --legacy-episodic: match episodic records against this reference date, or the "
              "date each signature carries (--as-of signature), instead of now."),
    )
    parser.add_argument(
        "--legacy-logs", action="store_true",
        help=("With --check-identity: read the runner's session logs, as the original audit did. "
              "Assistant text is printed into those logs, so this can be fooled. Archive use only."),
    )
    parser.add_argument("--verbose", action="store_true", help="More detail in archive modes.")
    args = parser.parse_args()

    _load_known_roles()

    if args.check_identity and args.legacy_logs:
        sys.exit(run_identity_audit(verbose=args.verbose))
    if args.preflight:
        sys.exit(run_preflight())
    if args.init_evidence:
        sys.exit(run_init_evidence())
    if args.rebaseline:
        sys.exit(run_rebaseline())
    if args.emit_signoff:
        sys.exit(run_emit_signoff(args))
    if args.gate:
        sys.exit(run_gate(args))
    if args.legacy_episodic or args.as_of:
        sys.exit(run_legacy_episodic(args, parser))
    sys.exit(run_receipt_check(args))


if __name__ == "__main__":
    main()
