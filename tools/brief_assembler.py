"""Deterministic context brief assembler. Reads recent lab state, writes
context_brief.md with curated content for the next Director session.

No LLM in the critical path. Runs in <100ms. Output: 10-20 KB.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# Module-level repo root. Tests can monkeypatch this; production code uses
# assemble_brief(repo_root) directly and this value is not consulted.
REPO: Path = Path(__file__).resolve().parent.parent

SessionType = Literal["routine-monitor", "phase-transition", "user-action", "post-failure"]


def _mtime(path: Path) -> float | None:
    """Return mtime in epoch seconds, or None if file doesn't exist."""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _orchestrator_pid_alive() -> bool:
    """True iff phase3 orchestrator or scale_experiment process is running."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", r"tools/run_phase3_factorial\.py|build/scale_experiment"],
            capture_output=True,
            text=True,
        )
        return out.returncode == 0
    except FileNotFoundError:
        return False


def classify_session(
    repo_root: Path, _orchestrator_alive: bool | None = None
) -> SessionType:
    """Classify the upcoming session by inspecting state files.

    The optional _orchestrator_alive kwarg is a test seam: production callers
    leave it None (we run pgrep), tests pass an explicit bool to control.
    """
    if _orchestrator_alive is None:
        _orchestrator_alive = _orchestrator_pid_alive()

    # phase-transition: orchestrator dead AND run_index newer than log.md
    if not _orchestrator_alive:
        log_mt = _mtime(repo_root / "data/memories/log.md")
        idx_mt = _mtime(repo_root / "data/checkpoints/phase3_factorial/run_index.json")
        if log_mt is not None and idx_mt is not None and idx_mt > log_mt:
            return "phase-transition"

    # user-action: user_notes.md modified within 30 min
    user_notes_mt = _mtime(repo_root / "data/user_notes.md")
    if user_notes_mt is not None:
        if time.time() - user_notes_mt < 30 * 60:
            return "user-action"

    # post-failure: session_exit.md exists with reason != GRACEFUL_CHECKPOINT
    exit_path = repo_root / "data/session_exit.md"
    if exit_path.exists():
        try:
            text = exit_path.read_text()
            m = re.search(r"^reason:\s*(\S+)", text, flags=re.MULTILINE)
            if m and m.group(1) != "GRACEFUL_CHECKPOINT":
                return "post-failure"
        except OSError:
            pass

    return "routine-monitor"


def _extract_current_program_block(repo_root: Path) -> str:
    """Read current.md and extract the §Current Program section.

    Matches a heading line containing "Current Program" (any case, any depth)
    and captures content until the next heading at any depth.
    """
    path = repo_root / "data/memories/current.md"
    if not path.exists():
        return "(no current.md found)"
    text = path.read_text()
    m = re.search(
        r"(^#{1,6}\s*[^\n]*Current Program[^\n]*\n)(.*?)(?=^#{1,6}\s|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if m:
        return (m.group(1) + m.group(2)).strip()
    return "(§Current Program block not found in current.md)"


def _extract_last_n_decisions(repo_root: Path, n: int = 5) -> str:
    r"""Pull last N D-NNN entries (assumes log.md is prepended, so head = newest).

    Real log.md has multiple coexisting formats (P-D427 root cause):
      - Current (post-D-423): ``## D-NNN — date — content``    (markdown heading + em-dash)
      - Older (D-308..D-422): ``D-NNN (timestamp): content``   (bare, parenthetical date, colon)
      - Test fixtures:        ``D-NNN: content``               (bare, colon)
      - Some entries:         ``### D-NNN (timestamp): content`` (deeper heading)

    The decision marker is the ``D-NNN`` token at the start of the heading-or-line content.
    We allow an optional leading ``#{1,6}\s*`` prefix and accept either em-dash (—) or
    colon (``:``) as the separator. The negative-lookahead stop-line for multi-line
    capture must also handle the heading-prefixed continuation case.
    """
    path = repo_root / "data/memories/log.md"
    if not path.exists():
        return "(no log.md)"
    text = path.read_text()
    # Match lines that start with optional markdown heading then D-NNN, then
    # an optional parenthetical timestamp, then either em-dash or colon
    # separator, then the rest of the entry. The full-line capture starts at
    # `D-NNN` (heading hash stripped) so output stays uniform.
    entries = re.findall(
        r"^(?:#{1,6}\s*)?(D-\d+(?:\s*\([^)]*\))?\s*(?:[—:])[^\n]*(?:\n(?!#{1,6}\s*D-|D-\d)[^\n]*)*)",
        text,
        flags=re.MULTILINE,
    )
    if not entries:
        return "(no D-N entries found)"
    out_lines = []
    for entry in entries[:n]:
        # First line only — full entry can be long
        first_line = entry.split("\n", 1)[0]
        out_lines.append(f"- {first_line[:300]}")
    return "\n".join(out_lines)


def _extract_state_delta(repo_root: Path) -> str:
    """Summarize current orchestrator + run_index state."""
    idx_path = repo_root / "data/checkpoints/phase3_factorial/run_index.json"
    if not idx_path.exists():
        return "(no run_index)"
    try:
        data = json.loads(idx_path.read_text())
    except json.JSONDecodeError:
        return "(run_index unparseable)"
    cells = sorted(k for k in data if not k.startswith("_"))
    lines = []
    for cell in cells:
        e = data[cell]
        state = e.get("state", "?")
        idx = e.get("run_index", "?")
        elapsed = e.get("elapsed_hours")
        elapsed_str = f"{elapsed}h" if elapsed else ""
        lines.append(f"- {cell}: {state} (run_index={idx}) {elapsed_str}".rstrip())
    return "\n".join(lines) if lines else "(no cells)"


def _extract_carry_forwards(repo_root: Path) -> str:
    """Find P-* tokens in recent log entries (first 10 KB only)."""
    path = repo_root / "data/memories/log.md"
    if not path.exists():
        return "(none)"
    text = path.read_text()
    # Limit search to first ~10 KB (most recent decisions, log is prepended)
    head = text[:10_000]
    tokens = sorted(set(re.findall(r"\bP-[A-Z][A-Z0-9-]+\b", head)))
    if not tokens:
        return "(none active)"
    return "\n".join(f"- {tok}" for tok in tokens)


_ACTION_HINTS = {
    "phase-transition": "Phase just ended. Read run_index.json + active program directory; start close-out work (mechanism extraction, paper draft, carry-forward processing).",
    "user-action": "Operator wrote to user_notes.md — read Active section, acknowledge directives, log decision IDs.",
    "post-failure": "Previous session exited with non-graceful reason — read session_exit.md, diagnose, acknowledge, recover.",
    "routine-monitor": "Routine monitor tick — verify training health, append minimum-viable-tick to log.md (≤1 KB), exit.",
}


def _action_hint(session_type: SessionType) -> str:
    return _ACTION_HINTS[session_type]


def _extract_active_program_files(repo_root: Path) -> list[str]:
    """List markdown files in the active program directory (relative paths)."""
    cur = repo_root / "data/memories/current.md"
    if not cur.exists():
        return []
    text = cur.read_text()
    m = re.search(r"active_program:\s*(\S+)", text)
    if not m:
        return []
    prog_dir = repo_root / "programs" / m.group(1)
    if not prog_dir.exists():
        return []
    return sorted(str(p.relative_to(repo_root)) for p in prog_dir.glob("*.md"))


def _decision_critical_section(repo_root: Path, session_type: SessionType) -> str:
    if session_type == "phase-transition":
        files = _extract_active_program_files(repo_root)
        if not files:
            return "(none)"
        return "\n".join(f"- {f}" for f in files)
    if session_type == "user-action":
        return "- data/user_notes.md (Active section — read verbatim)"
    if session_type == "post-failure":
        return "- data/session_exit.md (read reason + acknowledge)"
    return "(none — routine tick; expand wiki only on judgment requirement)"


def _wiki_not_loaded(repo_root: Path) -> str:
    """Enumerate on-demand-readable wiki + agent-semantic files.

    Includes: data/memories/*.md (excluding pre-loaded files) AND
    data/agents/<role>/semantic.md (so Director knows its own historical
    patterns are reachable on demand without pre-loading).
    """
    files = []
    wiki_root = repo_root / "data/memories"
    if wiki_root.exists():
        for p in sorted(wiki_root.rglob("*.md")):
            rel = str(p.relative_to(repo_root))
            if rel in (
                "data/memories/log.md",
                "data/memories/current.md",
                "data/memories/session_brief.md",
                "data/memories/context_brief.md",
                "data/memories/INDEX.md",
            ):
                continue
            files.append(rel)

    agents_root = repo_root / "data/agents"
    if agents_root.exists():
        for p in sorted(agents_root.glob("*/semantic.md")):
            files.append(str(p.relative_to(repo_root)))

    return "\n".join(f"- {f}" for f in files) if files else "(none)"


def _check_program_budget(repo_root: Path) -> str:
    """Check if the active program is over its hard budget.

    Reads data/programs/budgets.json + the active program's open date
    (from programs/<name>/ directory mtime as a fallback if no explicit
    open_iso recorded). Returns a notice string if program is over the
    warn threshold (75% by default), tripwire string if over hard limit,
    or empty string if within budget.

    Tripwire forces PI to choose: close, pivot, or extend with reason.
    """
    cur = repo_root / "data/memories/current.md"
    if not cur.exists():
        return ""
    text = cur.read_text()
    m = re.search(r"active_program:\s*(\S+)", text)
    if not m:
        return ""
    program = m.group(1)

    budgets_path = repo_root / "data/programs/budgets.json"
    if not budgets_path.exists():
        return ""
    try:
        budgets = json.loads(budgets_path.read_text())
    except json.JSONDecodeError:
        return ""

    cfg = budgets.get("overrides", {}).get(program) or budgets.get("default", {})
    max_days = cfg.get("max_total_days", 60)
    warn_pct = cfg.get("warn_at_pct", 0.75)

    prog_dir = repo_root / "programs" / program
    if not prog_dir.exists():
        return ""

    # Use directory mtime as program-open proxy (cheap, no extra schema)
    try:
        import time as _t
        open_mt = prog_dir.stat().st_mtime
        # If a program_open_memo.md exists, use ITS mtime (older, more accurate)
        memo = prog_dir / "program_open_memo.md"
        if memo.exists():
            open_mt = min(open_mt, memo.stat().st_mtime)
        days = (_t.time() - open_mt) / 86400
    except OSError:
        return ""

    pct = days / max_days if max_days > 0 else 0
    overrides = repo_root / f"programs/{program}/budget_overrides.md"
    has_override = overrides.exists()

    if pct >= 1.0 and not has_override:
        return (
            f"⚠️ HARD BUDGET TRIPWIRE: {program} has run {days:.1f} days "
            f"(limit {max_days}). PI MUST respond at next gate with: close / "
            f"pivot / documented extension (write to "
            f"`programs/{program}/budget_overrides.md` with reason)."
        )
    if pct >= warn_pct:
        return (
            f"BUDGET WARNING: {program} at {pct*100:.0f}% of {max_days}-day "
            f"budget ({days:.1f} days elapsed). Approaching tripwire."
        )
    return ""


def _build_correction_prompt(reason: str) -> str:
    """Build a CORRECTION PROMPT block at the top of the brief when redispatching."""
    if reason == "schema_violation_previous_session_exit":
        return (
            "## CORRECTION PROMPT — previous session_exit.md was REJECTED\n\n"
            "Your previous session exited with `status=success` (or `partial`) but did NOT "
            "populate `next_action` AND did NOT set `program_complete=true`. The runner's "
            "schema validation (per `data/agents/_shared/session_exit_schema.md` v1.1) rejected "
            "the exit — no log / current / queue mutations were applied from that session.\n\n"
            "**This session, you must either:**\n\n"
            "1. **Populate `next_action`** with the concrete next critical-path item. Read "
            "`current.md`, recent `log.md` entries, your previous session's narrative, and "
            "the active program's plan/dispatch packets. Derive the next work item that should "
            "happen and put it in `session_exit.md`'s JSON `next_action` field.\n\n"
            "2. **OR declare `program_complete: true`** if the program is genuinely done. "
            "Document the reasoning in the `notes` field. The runner accepts this combination "
            "as a valid program-end declaration.\n\n"
            "Refer to `data/agents/_shared/session_exit_schema.md` § 'Required field semantics (v1.1)' "
            "and Example D for the schema contract.\n\n"
            "---\n\n"
        )
    return (
        f"## CORRECTION PROMPT — redispatch reason: {reason}\n\n"
        "Previous session triggered a redispatch. Check session_exit_schema.md "
        "and post_director_telemetry.jsonl for the specific violation.\n\n"
        "---\n\n"
    )


def _pending_calibration_section(repo_root: Path) -> str:
    """Render the pending-calibration nudge (read-only, best-effort).

    Delegates to tools.calibration_pending. Any failure (import error,
    unreadable telemetry) degrades to an empty string so it can never break
    brief assembly. Returns '' when there are no pending claims.
    """
    try:
        from tools.calibration_pending import (
            format_pending_brief,
            list_pending_claims,
        )

        return format_pending_brief(list_pending_claims(repo_root=repo_root))
    except Exception:
        return ""


def assemble_brief(repo_root: Path, _orchestrator_alive: bool | None = None) -> str:
    """Assemble a context brief markdown for the next Director session."""
    session_type = classify_session(repo_root, _orchestrator_alive=_orchestrator_alive)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    budget_notice = _check_program_budget(repo_root)

    # Correction prompt for schema-violation redispatch sessions (L1.3 / B′).
    # When DIRECTOR_REDISPATCH_REASON env var is set, prepend a prominently
    # placed correction block so Director sees exactly what was wrong before
    # reading any other state.
    correction_block = ""
    redispatch_reason = os.environ.get("DIRECTOR_REDISPATCH_REASON")
    if redispatch_reason:
        correction_block = _build_correction_prompt(redispatch_reason)

    sections = [
        f"---\ngenerated_at: {now}\nsession_type: {session_type}\nsize_kb: 0\n---\n",
    ]
    if correction_block:
        sections.append(correction_block)
    if budget_notice:
        sections.append(f"# Program budget\n{budget_notice}\n")
    # Pending-calibration nudge (additive): surface unresolved calibration claims
    # so PI/Director resolves each via score_calibration() when its outcome lands.
    # Placed high (before the large Active-program block, which alone can exceed
    # the 30 KB cap) so this tiny read-only nudge actually survives truncation.
    pending_block = _pending_calibration_section(repo_root)
    if pending_block:
        sections.append(f"# Calibration\n{pending_block}\n")

    sections.extend([
        f"# Active program\n{_extract_current_program_block(repo_root)}\n",
        f"# State delta since last Director session\n{_extract_state_delta(repo_root)}\n",
        f"# Last 5 decisions\n{_extract_last_n_decisions(repo_root, n=5)}\n",
        f"# Active carry-forwards\n{_extract_carry_forwards(repo_root)}\n",
        f"# What this session likely needs to do\n{_action_hint(session_type)}\n",
        f"# Decision-critical files\n{_decision_critical_section(repo_root, session_type)}\n",
        f"# Wiki tier files NOT loaded\n{_wiki_not_loaded(repo_root)}\n",
    ])

    brief = "\n".join(sections)

    # Size cap: hard truncate at 30 KB with breadcrumb
    if len(brief.encode("utf-8")) > 30 * 1024:
        brief = brief.encode("utf-8")[: 30 * 1024 - 200].decode("utf-8", errors="ignore")
        brief += "\n\n# (BRIEF TRUNCATED at 30 KB — see source files for full detail)\n"

    return brief


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="data/memories/context_brief.md",
        help="Output path (default: data/memories/context_brief.md)",
    )
    ap.add_argument(
        "--repo",
        default=".",
        help="Repo root (default: cwd)",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    brief = assemble_brief(repo_root)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(brief)
    print(f"brief_assembler: wrote {len(brief)} bytes to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
