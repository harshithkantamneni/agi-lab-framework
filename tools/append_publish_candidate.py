#!/usr/bin/env python3
"""
Append a publish candidate to the repo's data/publish_candidates/<id>.json
based on a closure memo's frontmatter.

Opt-in mechanism: the closure memo (or phase publication file) opts in by
declaring these fields in its YAML frontmatter:

    publish_to_portfolio: true
    public_title: "..."
    public_summary: "..."
    public_source_artifacts:
      - programs/<program>/closure_memo.md
      - programs/<program>/<paper_or_artifact>.md
    public_type: report          # or "note"; defaults to "report"
    public_tags: [agi, envelope-paper, ...]

Without these fields, the script declines (sys.exit 0, not an error).
This is the operator's veto: a memo without the publish opt-in stays inside
the lab.

Idempotent: refuses to overwrite an existing candidate file. If a memo
needs republication (e.g., after revision), use a new public_title (which
slugifies to a new id), or append a version suffix manually.

Schema source of truth:
  ~/Desktop/website/tools/curator/schema/publish_candidate.schema.json
Keep this script's field set in sync with that schema. Inline validation
here is a fast-fail check; the curator re-validates against the full schema.

Usage:
    # Process a single memo
    python3 tools/append_publish_candidate.py programs/program_1_opus47_on_18gb/closure_memo.md

    # Scan every program's closure_memo.md and write candidates for the eligible ones
    python3 tools/append_publish_candidate.py --all

    # See what would happen without writing
    python3 tools/append_publish_candidate.py --all --dry-run
"""
import argparse
import datetime
import json
import re
import sys
from pathlib import Path

LAB_ROOT = Path.home() / "Desktop" / "AGI"
CANDIDATES_DIR = LAB_ROOT / "data" / "publish_candidates"
ID_REGEX = re.compile(r"^[a-z0-9][a-z0-9-]*$")
VALID_TYPES = {"report", "note"}
VALID_CHANNELS = {"website", "hackernews", "linkedin", "buttondown"}


def die(msg: str) -> None:
    print(f"[append] ERROR: {msg}", file=sys.stderr)
    sys.exit(2)


def parse_frontmatter(text: str) -> dict:
    """Tiny YAML parser sufficient for our frontmatter shape.

    Handles:
      - scalar key: value
      - boolean (true/false case-insensitive)
      - inline list:  key: [a, b, c]
      - block list:
          key:
            - item1
            - item2
      - quoted strings (single or double; quotes stripped)

    Does NOT handle: nested dicts, anchors, multiline scalars, etc.
    Our frontmatter doesn't use those — keep this dependency-free.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 4)
    if end == -1:
        return {}
    block = text[4:end]

    out: dict = {}
    current_list_key: str | None = None

    for raw in block.split("\n"):
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - ") or line.startswith("- "):
            item = line.split("-", 1)[1].strip().strip('"').strip("'")
            if current_list_key:
                out.setdefault(current_list_key, []).append(item)
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        current_list_key = None
        if val == "":
            current_list_key = key
            out.setdefault(key, [])
        elif val.lower() == "true":
            out[key] = True
        elif val.lower() == "false":
            out[key] = False
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            items = [s.strip().strip('"').strip("'")
                     for s in inner.split(",") if s.strip()]
            out[key] = items
        else:
            out[key] = val.strip('"').strip("'")
    return out


def slugify(text: str) -> str:
    s = text.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def process(memo_path: Path, dry_run: bool = False) -> bool:
    """Return True if a candidate was (or would be) written."""
    if not memo_path.is_file():
        die(f"memo not found: {memo_path}")

    fm = parse_frontmatter(memo_path.read_text())

    if not fm.get("publish_to_portfolio"):
        print(f"[append] skip: {memo_path} (publish_to_portfolio not true)",
              file=sys.stderr)
        return False

    title = fm.get("public_title") or ""
    summary = fm.get("public_summary") or ""
    src = fm.get("public_source_artifacts") or []
    if isinstance(src, str):
        src = [src]
    typ = fm.get("public_type") or "report"
    tags = fm.get("public_tags") or []
    if isinstance(tags, str):
        tags = [tags]
    channels = fm.get("public_channels") or ["website"]
    if isinstance(channels, str):
        channels = [channels]

    if not title or not summary or not src:
        die(f"{memo_path}: missing public_title, public_summary, "
            f"or public_source_artifacts")
    if typ not in VALID_TYPES:
        die(f"public_type must be 'report' or 'note', got '{typ}'")
    bad_channels = [c for c in channels if c not in VALID_CHANNELS]
    if bad_channels:
        die(f"unknown channel(s): {bad_channels}")

    cid = slugify(title)
    if not ID_REGEX.match(cid):
        die(f"derived id is invalid: '{cid}' (from public_title='{title}')")

    for sa in src:
        if not (LAB_ROOT / sa).exists():
            die(f"source artifact does not resolve: '{sa}'")

    out_path = CANDIDATES_DIR / f"{cid}.json"
    if out_path.exists():
        print(f"[append] idempotent: {out_path} exists; skipping",
              file=sys.stderr)
        return False

    closed_date = fm.get("closed_date") or datetime.date.today().isoformat()
    program = fm.get("program", "")

    if "closure" in str(memo_path).lower():
        ratified_at = f"closure_memo_{closed_date}"
    else:
        ratified_at = closed_date

    payload = {
        "id": cid,
        "lab": "agi",
        "type": typ,
        "status": "ready",
        "title": title,
        "summary": summary,
        "source_artifacts": src,
        "ratified_at": ratified_at,
        "ratified_date": closed_date,
        "tags": tags,
        "channels": channels,
        "program": program,
        "curator_state": "pending",
    }

    if dry_run:
        print(json.dumps(payload, indent=2))
        return True

    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[append] wrote {out_path}", file=sys.stderr)
    return True


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("memo_path", nargs="?",
                   help="path to a closure memo or phase publication file")
    p.add_argument("--all", action="store_true",
                   help="scan all programs/*/closure_memo.md and process eligible ones")
    p.add_argument("--dry-run", action="store_true",
                   help="print payload to stdout, don't write")
    args = p.parse_args()

    if not args.all and not args.memo_path:
        p.error("provide a memo path or --all")

    if args.all:
        wrote = 0
        for memo in sorted((LAB_ROOT / "programs").glob("*/closure_memo.md")):
            if process(memo, dry_run=args.dry_run):
                wrote += 1
        print(f"[append] processed {wrote} eligible memo(s)", file=sys.stderr)
    else:
        process(Path(args.memo_path), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
