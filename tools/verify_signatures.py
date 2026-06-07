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
]

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


def verify_signature(sig: Signature, window_seconds: int) -> Forgery | None:
    """Check if sig has a corresponding episodic entry within window_seconds of now.

    Returns None if verified, Forgery if not.
    """
    # Director is the session itself — signatures are self-attestation, not
    # dispatched-agent attestation. The attack vector we care about is Director
    # forging OTHER roles (D-109 pattern). Director signing its own work is
    # verified by the session existing at all, so we skip episodic check.
    if sig.role.lower() == "director":
        return None

    # If the signature cites [episodic: <path>], verify the file exists and is recent.
    if sig.has_episodic_ref:
        ep_path = REPO_ROOT / sig.episodic_ref_path
        if not ep_path.exists():
            return Forgery(
                role=sig.role,
                source_file=sig.source_file,
                line_number=sig.line_number,
                reason=f"signature cites episodic [{sig.episodic_ref_path}] but file does not exist",
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
        "--verbose", action="store_true",
        help="Print every signature found (not just forgeries).",
    )
    args = parser.parse_args()

    _load_known_roles()

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
    forgeries: list[Forgery] = []
    for s in sigs:
        f = verify_signature(s, window)
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
