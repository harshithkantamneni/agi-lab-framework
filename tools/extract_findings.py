#!/usr/bin/env python3
"""tools/extract_findings.py — draft candidate findings from agent episodic records.

Greps recent agent episodic records for insight-bearing phrases ("we learned",
"turns out", "it's important to", "pattern is", "fails when", etc.) and
writes candidate finding drafts under `data/findings/_candidates/`. The
findings_curator reviews each candidate, promotes legitimate ones to the
permanent findings/ directory (or to memories/shared.md once memory refactor
cuts over), and discards spurious ones.

Motivation: insights currently only become findings when an agent remembers
to write a finding file explicitly. Some leak — they land inside an episodic
record's prose and never surface to the cross-program knowledge bus. A
pre-pass grep catches the miss at near-zero cost.

Usage:
    python3 tools/extract_findings.py --since-hours 24
    python3 tools/extract_findings.py --since-hours 72 --min-context 120
    python3 tools/extract_findings.py --since-hours 24 --apply     # write drafts

Dry-run by default — prints the phrases that WOULD become drafts. Pass
--apply to write the candidate files. findings_curator must manually
promote candidates into real findings; this tool does not auto-promote.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = REPO_ROOT / "data" / "agents"
FINDINGS_DIR = REPO_ROOT / "data" / "findings"
CANDIDATES_DIR = FINDINGS_DIR / "_candidates"

# Insight-bearing phrases. Each is a compiled regex matched against a line.
# Ordered by approximate strength-of-signal.
INSIGHT_PATTERNS = [
    # Strong:
    re.compile(r"\bwe (?:learned|found|discovered|confirmed|established)\b", re.IGNORECASE),
    re.compile(r"\b(?:it )?turn(?:s|ed) out(?:,)? that\b", re.IGNORECASE),
    re.compile(r"\b(?:it(?:'s)? important|critical|key) to\b", re.IGNORECASE),
    re.compile(r"\bthe pattern is\b", re.IGNORECASE),
    re.compile(r"\b(?:the )?lesson (?:is|learned)\b", re.IGNORECASE),
    re.compile(r"\b(?:this|that) (?:means|implies|shows) (?:that )?\b", re.IGNORECASE),
    re.compile(r"\bfails? when\b", re.IGNORECASE),
    re.compile(r"\bworks? because\b", re.IGNORECASE),
    # Medium:
    re.compile(r"\bkey (?:finding|insight|takeaway|observation)\b", re.IGNORECASE),
    re.compile(r"\b(?:one|an?) (?:important|key|critical) (?:observation|finding|insight|lesson)\b", re.IGNORECASE),
    re.compile(r"\bsurpris(?:ing|e|ed|ingly)\b", re.IGNORECASE),
    re.compile(r"\bcounter-?intuitiv(?:e|ely)\b", re.IGNORECASE),
    re.compile(r"\b(?:contrary|opposite) to expectat", re.IGNORECASE),
    re.compile(r"\binvariant\s*:?\s*", re.IGNORECASE),
    re.compile(r"\bmeta-?lesson\b", re.IGNORECASE),
    re.compile(r"\bcaveat\s*:?\s*", re.IGNORECASE),
    re.compile(r"\bgotcha\b", re.IGNORECASE),
    # Weak but tagged:
    re.compile(r"\b(?:note|notable) that\b", re.IGNORECASE),
    re.compile(r"\brule of thumb\b", re.IGNORECASE),
]


def collect_recent_episodic(since_seconds: int) -> list[Path]:
    """Return episodic .md files modified within since_seconds."""
    import time
    now = time.time()
    files = []
    for f in AGENTS_DIR.glob("*/episodic/*.md"):
        if f.name == ".gitkeep":
            continue
        try:
            mtime = f.stat().st_mtime
        except OSError:
            continue
        if now - mtime <= since_seconds:
            files.append(f)
    return sorted(files, key=lambda p: -p.stat().st_mtime)


def scan_file(path: Path, min_context_chars: int = 80) -> list[dict]:
    """Scan a single episodic file for insight hits. Returns list of candidates."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = text.splitlines()
    hits: list[dict] = []
    seen_signatures = set()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) < min_context_chars:
            continue
        for pat in INSIGHT_PATTERNS:
            m = pat.search(stripped)
            if not m:
                continue

            # Context: the hit line + up to 2 lines before/after for reviewer clarity
            context_start = max(0, i - 1)
            context_end = min(len(lines), i + 3)
            context_block = "\n".join(lines[context_start:context_end]).strip()

            # Deduplicate per-file: same hit phrase in same line → skip
            sig = hashlib.sha1((str(path) + stripped + pat.pattern).encode()).hexdigest()[:12]
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            hits.append({
                "file": str(path.relative_to(REPO_ROOT)),
                "line_number": i + 1,
                "phrase": m.group(0),
                "line": stripped,
                "context": context_block,
                "sig": sig,
            })
            break  # one hit per line
    return hits


def write_candidate(hit: dict, target_dir: Path) -> Path:
    """Write a candidate finding draft for the reviewer. Returns written path."""
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    source = Path(hit["file"])
    agent = source.parts[2] if len(source.parts) >= 3 else "unknown"
    # Filename includes signature for idempotent writes (re-run doesn't duplicate)
    filename = f"candidate_{stamp}_{agent}_{hit['sig']}.md"
    path = target_dir / filename
    if path.exists():
        return path  # idempotent: already drafted

    body = (
        f"# Candidate finding — {stamp}\n\n"
        f"**Source:** `{hit['file']}:{hit['line_number']}`\n"
        f"**Agent:** `{agent}`\n"
        f"**Triggered by phrase:** `{hit['phrase']}`\n"
        f"**Signature:** `{hit['sig']}` (idempotent — re-running the extractor won't duplicate)\n\n"
        f"## Hit line\n\n"
        f"> {hit['line']}\n\n"
        f"## Surrounding context\n\n"
        f"```\n{hit['context']}\n```\n\n"
        f"## Reviewer action (findings_curator)\n\n"
        f"- [ ] PROMOTE — this is a durable, generalizable finding. Move to `data/findings/<topic>.md` (pre-refactor) or `data/memories/shared.md` (post-refactor); delete this candidate file.\n"
        f"- [ ] DEMOTE — insight is program-specific and belongs in that program's own artifacts. Note the location and delete this candidate.\n"
        f"- [ ] DISCARD — false positive (pattern matched prose that isn't a finding). Delete this candidate.\n"
        f"- [ ] DEFER — needs more context / adjacent-finding synthesis. Leave in place; revisit next closeout.\n"
    )
    path.write_text(body, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Grep episodic records for insight-bearing phrases and draft candidates.")
    parser.add_argument("--since-hours", type=int, default=24, help="Only scan episodic files modified in last N hours (default 24)")
    parser.add_argument("--min-context", type=int, default=80, help="Ignore lines shorter than this many chars (default 80)")
    parser.add_argument("--apply", action="store_true", help="Write candidate drafts to data/findings/_candidates/ (default: dry-run)")
    parser.add_argument("--verbose", action="store_true", help="Print full context blocks, not just phrase summaries")
    args = parser.parse_args()

    since_seconds = args.since_hours * 3600
    files = collect_recent_episodic(since_seconds)
    if not files:
        print(f"No episodic files modified in last {args.since_hours}h.")
        return 0

    all_hits = []
    for f in files:
        all_hits.extend(scan_file(f, min_context_chars=args.min_context))

    if not all_hits:
        print(f"Scanned {len(files)} episodic files, no insight phrases detected.")
        return 0

    print(f"Scanned {len(files)} files, found {len(all_hits)} candidate insight(s).\n")
    for hit in all_hits:
        print(f"  {hit['file']}:{hit['line_number']}")
        print(f"    phrase : {hit['phrase']}")
        print(f"    line   : {hit['line'][:120]}{'...' if len(hit['line']) > 120 else ''}")
        if args.verbose:
            print(f"    context:\n{hit['context']}\n")

    if args.apply:
        written = 0
        for hit in all_hits:
            path = write_candidate(hit, CANDIDATES_DIR)
            written += 1
        print(f"\nWrote {written} candidate files to {CANDIDATES_DIR.relative_to(REPO_ROOT)}/")
        print(f"findings_curator: review each, check a box (PROMOTE/DEMOTE/DISCARD/DEFER), then delete.")
    else:
        print(f"\n(Dry-run. Pass --apply to write candidates to data/findings/_candidates/.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
