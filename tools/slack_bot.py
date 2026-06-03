#!/usr/bin/env python3
"""tools/slack_bot.py — AGI Lab Slack bot.

Curated notifications from the lab + bidirectional commands from Slack.

Architecture:
  - slack_bolt with Socket Mode (no webhook URL / inbound port needed)
  - Main thread: Socket Mode handler for commands (app_mentions + DMs)
  - Background thread: polls lab state files, detects events, posts to Slack
  - State persisted to data/infra/slack_bot_state.json

Required env vars:
  AGI_LAB_SLACK_BOT_TOKEN   xoxb-... (from OAuth & Permissions)
  AGI_LAB_SLACK_APP_TOKEN   xapp-... (from Basic Information → App-Level Tokens)
  AGI_LAB_SLACK_CHANNEL     channel ID (C...) or #name where updates post
                            (or "DM:<user_id>" for user DMs)

Launch:
  tmux new-session -d -s agi-slack-bot \
    'caffeinate -di python3 tools/slack_bot.py'

See `data/procedures.md` for the Slack bot protocol (future reference).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import pytz
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler


def _mem_or_legacy(new_rel: str, legacy_rel: str):
    """Prefer memory-tier path; fall back to legacy during migration overlap."""
    from pathlib import Path
    np = Path(new_rel)
    if np.exists():
        return np
    return Path(legacy_rel)


# ---- Config ----

REPO = Path(__file__).resolve().parent.parent
STATE_FILE = REPO / "data/infra/slack_bot_state.json"
TZ = pytz.timezone("America/Chicago")
QUIET_START = dtime(23, 0)  # 11:00 PM CDT
QUIET_END = dtime(9, 0)     # 9:00 AM CDT
POLL_INTERVAL_SECONDS = 30
LAB_DEAD_THRESHOLD_MINUTES = 10
DIGEST_SEPARATOR = "─" * 30

# Urgent events bypass quiet hours
URGENT_EVENT_TYPES = {"VICTORY", "CATASTROPHIC", "LAB_DEAD", "SIGNATURE_FORGERY"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("agi-slack-bot")


# ---- Slack app ----

BOT_TOKEN = os.environ.get("AGI_LAB_SLACK_BOT_TOKEN")
APP_TOKEN = os.environ.get("AGI_LAB_SLACK_APP_TOKEN")
CHANNEL = os.environ.get("AGI_LAB_SLACK_CHANNEL")

if not BOT_TOKEN or not APP_TOKEN or not CHANNEL:
    log.error(
        "Missing env vars. Required: AGI_LAB_SLACK_BOT_TOKEN (xoxb-), "
        "AGI_LAB_SLACK_APP_TOKEN (xapp-), AGI_LAB_SLACK_CHANNEL (channel ID "
        "or #name or DM:<user_id>)."
    )
    sys.exit(1)

app = App(token=BOT_TOKEN)

START_TIME = time.time()


# ---- State persistence ----

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            log.warning("state file corrupt; resetting")
    # Initial state
    return {
        "last_session_exit_mtime": 0,
        "last_evaluator_report_mtime": 0,
        "last_evaluator_verdict": None,
        "last_decision_id": None,            # e.g., "D-110"
        "last_ledger_mtime": 0,
        "last_ledger_entry_signature": None,
        "last_rate_limit_resets_at": None,
        "last_seen_session_log": None,
        "last_seen_session_log_mtime": 0,
        "last_state_md_mtime": 0,
        "last_state_status_fingerprint": None,
        "lab_dead_notified": False,
        "digest_queue": [],                  # list of {ts, text, category}
        "last_digest_flush": 0,
        "pause_until_ts": 0,
        # Added 2026-04-25 (post D-202..D-210 incident):
        "user_action_files": {},             # path → {posted_at, decision_state, last_repost_ts}
        "holding_loop_signaled_at": 0,        # ts of last holding-loop post; 0 = not active
        "holding_loop_last_count": 0,        # count at last post (re-post when grows by 5+)
        "last_phase_summary_fp": None,       # fingerprint of "Phase N <state>" line
        "decision_loop_signaled": False,     # suppress NEW_DECISION posts when looping
        "decision_signature_window": [],     # last 5 normalized signatures for de-dupe
        # Added 2026-04-28 (D-288): training-progress detector. Tracks the
        # last checkpoint-step we posted about per cell, so we post on each
        # new checkpoint landing (event-driven) rather than on a polling
        # cadence. Replaces the Director's role for routine monitoring;
        # Director only wakes for events worth waking for.
        "ckpt_progress": {},                 # cell_name → last posted step
        "cell_completed": {},                # cell_name → True once cell hits step==steps
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(STATE_FILE)


# ---- Time helpers ----

def now_tz() -> datetime:
    return datetime.now(TZ)


def in_quiet_hours() -> bool:
    """True if current local time is inside [QUIET_START, QUIET_END) (wrapping midnight)."""
    now_t = now_tz().time()
    if QUIET_START < QUIET_END:
        return QUIET_START <= now_t < QUIET_END
    # wraps midnight
    return now_t >= QUIET_START or now_t < QUIET_END


def next_digest_flush_time() -> datetime:
    """Next time to flush digest — the next occurrence of QUIET_END."""
    now = now_tz()
    candidate = now.replace(hour=QUIET_END.hour, minute=QUIET_END.minute, second=0, microsecond=0)
    if candidate <= now:
        candidate = candidate + timedelta(days=1)
    return candidate


# ---- Posting helpers ----

def resolve_channel() -> str:
    """Map CHANNEL env var to a channel ID understood by chat_postMessage.

    Accepts:
      - Channel ID directly (C... or D...)
      - #name (prefix stripped; Slack accepts names for channels the bot is in)
      - DM:<user_id> — opens/reuses a DM with that user
    """
    if CHANNEL.startswith("DM:"):
        user_id = CHANNEL[3:]
        resp = app.client.conversations_open(users=user_id)
        return resp["channel"]["id"]
    if CHANNEL.startswith("#"):
        return CHANNEL  # Slack API accepts channel names for channels the bot is in
    return CHANNEL


_channel_id_cache = None


def channel_id() -> str:
    global _channel_id_cache
    if _channel_id_cache is None:
        _channel_id_cache = resolve_channel()
    return _channel_id_cache


def post(text: str, *, category: str = "INFO", urgent: bool = False) -> None:
    """Post to Slack, honoring quiet hours unless urgent."""
    state = load_state()
    stamp = now_tz().strftime("%H:%M")
    record = {"ts": time.time(), "time_str": stamp, "category": category, "text": text}

    # Pause window honors the `pause` command — buffer everything as digest during pause
    if state.get("pause_until_ts", 0) > time.time() and not urgent:
        state["digest_queue"].append(record)
        save_state(state)
        log.info("buffered (paused): %s", text[:80])
        return

    if urgent or not in_quiet_hours():
        try:
            app.client.chat_postMessage(channel=channel_id(), text=text, mrkdwn=True)
            log.info("posted: %s", text[:80])
        except Exception as e:
            log.error("post failed: %s", e)
            # Buffer on failure so we don't lose the event
            state["digest_queue"].append(record)
            save_state(state)
    else:
        # Quiet hours, non-urgent → queue
        state["digest_queue"].append(record)
        save_state(state)
        log.info("queued for digest: %s", text[:80])


def flush_digest_if_due() -> None:
    """If quiet hours just ended, post the queued digest."""
    state = load_state()
    queue = state.get("digest_queue", [])
    if not queue:
        return
    # Only flush if we are NOW in work hours (outside quiet window)
    if in_quiet_hours():
        return
    lines = [f"*Overnight digest — {len(queue)} event(s)*", DIGEST_SEPARATOR]
    for item in queue:
        emoji = {
            "INFO": "•",
            "GOOD": "✅",
            "ATTENTION": "⚠️",
            "URGENT": "🚨",
        }.get(item.get("category", "INFO"), "•")
        ts = item.get("time_str") or now_tz().strftime("%H:%M")
        lines.append(f"`{ts}` {emoji} {item.get('text', '')}")
    body = "\n".join(lines)
    try:
        app.client.chat_postMessage(channel=channel_id(), text=body, mrkdwn=True)
        log.info("flushed digest: %d events", len(queue))
        state["digest_queue"] = []
        state["last_digest_flush"] = time.time()
        save_state(state)
    except Exception as e:
        log.error("digest flush failed: %s", e)


# ---- Event detectors ----

def mtime_or_zero(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return 0.0


def read_head(path: Path, n: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:n]
    except (OSError, FileNotFoundError):
        return ""


def _session_log_summary(log_path: Path) -> dict:
    """Extract a summary from a session log: duration (from file mtime - ctime),
    dispatches, session number. No `?` placeholders — missing values are 0 / [].
    """
    summary: dict = {
        "duration_min": 0,         # int minutes, 0 = unknown / sub-minute
        "dispatches": [],
        "rate_limited": False,
        "session_num": 0,           # int, 0 = unknown
        "wrote_files": 0,
    }
    if not log_path.exists():
        return summary
    try:
        st = log_path.stat()
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return summary
    m = re.search(r"session_(\d+)_", log_path.name)
    if m:
        summary["session_num"] = int(m.group(1))
    # Dispatches: only "AGENT <role>" tokens; strip anything not a-z_
    raw = re.findall(r">>>\s*AGENT\s+([a-zA-Z_][a-zA-Z0-9_]*)", text)
    summary["dispatches"] = raw
    # Duration: file lifetime in whole minutes (most reliable signal)
    try:
        delta = max(0, int(st.st_mtime - st.st_birthtime))
    except AttributeError:
        delta = max(0, int(st.st_mtime - st.st_ctime))
    summary["duration_min"] = delta // 60
    summary["rate_limited"] = bool(
        re.search(r"hit your limit|rate limit|resets.*[ap]m", text, re.I)
    )
    summary["wrote_files"] = len(re.findall(r"(?:Write|Edit)\s+\S", text))
    return summary


_PHASE_STATE_KEYWORDS = (
    "PRE-FLIGHT", "EXECUTE", "OPEN", "CLOSED", "COMPLETE",
    "BLOCKED", "PIVOT", "VICTORY",
)


def _current_program_phase() -> tuple[str, str]:
    """Return (program_slug, phase_short) — clean strings, never raw junk.

    program_slug: the lab's canonical program identifier (e.g.
                  `program_2_dense_vs_moe_sub100m`), pulled from the first
                  occurrence of the `program_<slug>` token in current.md.
    phase_short:  e.g. `P3 OPEN` or `P3 EXECUTE` — Phase number + first
                  matching state keyword. Falls back to `P<N>` or
                  `(unknown)` when nothing matches.
    """
    text = read_head(
        _mem_or_legacy(REPO / "data/memories/current.md", REPO / "data/state.md"),
        4000,
    )
    prog_m = re.search(r"\b(program_[a-z0-9_]+)", text, re.IGNORECASE)
    program = prog_m.group(1) if prog_m else "(unknown)"

    phase_short = "(unknown)"
    cp = re.search(
        r"^##\s*Current\s+Phase\s*[:\-]\s*(.+)$",
        text, re.MULTILINE | re.IGNORECASE,
    )
    region = cp.group(1) if cp else text
    pn = re.search(r"Phase\s+(\d+)", region, re.IGNORECASE)
    if pn:
        phase_short = f"P{pn.group(1)}"
        for kw in _PHASE_STATE_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", region, re.IGNORECASE):
                phase_short = f"P{pn.group(1)} {kw}"
                break
    return program, phase_short


# ---- Message composition (post-D-235 cleanup) ----

_SEVERITY_ICON = {"info": "", "good": "✅", "warn": "⚠️", "alert": "🚨"}


def _compose_event(severity: str, headline: str, *,
                   fields: "dict | None" = None,
                   preview: str = "",
                   note: str = "") -> str:
    """Build a Slack message in the lab's standard format.

    Layout:
        <icon> *<headline>*
        Field1: value · Field2: value
        > preview (one line, trimmed)
        _note (italic call-to-action)_

    Empty / `?` / `(unknown)` field values are dropped silently so the
    message stays tight. `severity` ∈ {info, good, warn, alert}.
    """
    icon = _SEVERITY_ICON.get(severity, "")
    head = f"{icon} *{headline.strip()}*" if icon else f"*{headline.strip()}*"
    parts = [head]
    if fields:
        items = []
        for k, v in fields.items():
            if v is None:
                continue
            sv = str(v).strip()
            if not sv or sv in ("?", "(unknown)", "0", "0m"):
                continue
            items.append(f"{k}: {sv}")
        if items:
            parts.append(" · ".join(items))
    if preview:
        clean = " ".join(preview.split())[:280]
        if clean:
            parts.append(f"> {clean}")
    if note:
        parts.append(f"_{note.strip()}_")
    return "\n".join(parts)


def _format_dispatches(dispatches: "list[str]", top_n: int = 5) -> str:
    """Compact dispatch summary like 'pi×1, director×2'. Empty if none."""
    if not dispatches:
        return ""
    from collections import Counter
    counts = Counter(dispatches)
    return ", ".join(f"{role}×{n}" for role, n in counts.most_common(top_n))


_REASON_LINE_RE = re.compile(
    r"^(?:reason|exit[_\- ]reason|status)\s*[:=]\s*(\w[\w_]*)",
    re.MULTILINE | re.IGNORECASE,
)


def _extract_exit_reason(text: str) -> str:
    """Extract the EXIT reason from session_exit.md.

    Looks for an explicit `reason:` / `EXIT_REASON:` / `STATUS:` field
    before falling back to scanning the whole body for known reason
    tokens. Never returns the raw H1 (which used to leak `# Session
    Exit — D-225 (...)` into the channel).
    """
    m = _REASON_LINE_RE.search(text)
    if m:
        return m.group(1).upper()
    upper = text.upper()
    for kw in (
        "VICTORY", "CATASTROPHIC", "SIGNATURE_FORGERY",
        "EVALUATOR_FAIL", "CONTEXT_FULL", "RATE_LIMIT",
        "GRACEFUL_CHECKPOINT",
    ):
        if kw in upper:
            return kw
    return "UNKNOWN"


def detect_session_exit(state: dict) -> list[dict]:
    """Detect a new session_exit.md write — only emit on exceptional exits."""
    events: list[dict] = []
    path = REPO / "data/session_exit.md"
    mtime = mtime_or_zero(path)
    if not (mtime and mtime > state["last_session_exit_mtime"]):
        return events

    text = read_head(path, 1500)
    reason = _extract_exit_reason(text)
    state["last_session_exit_mtime"] = mtime

    prog, phase = _current_program_phase()
    log_dir = REPO / "data/infra/session_logs"
    logs = sorted(
        log_dir.glob("session_*.log"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True,
    )
    sess = _session_log_summary(logs[0]) if logs else {}
    disp = _format_dispatches(sess.get("dispatches", []))
    sess_num = sess.get("session_num") or 0
    dur_min = sess.get("duration_min") or 0
    fields = {
        "Program": prog, "Phase": phase,
        "Session": f"#{sess_num}" if sess_num else "",
        "Duration": f"{dur_min}m" if dur_min else "",
        "Dispatches": disp,
    }

    if reason == "VICTORY":
        events.append({
            "type": "VICTORY", "urgent": True, "category": "URGENT",
            "text": _compose_event(
                "alert", "Victory declared — all benchmarks surpassed",
                fields=fields,
                note="Verify in data/eval/scorecard.md",
            ),
        })
    elif reason == "CATASTROPHIC":
        events.append({
            "type": "CATASTROPHIC", "urgent": True, "category": "URGENT",
            "text": _compose_event(
                "alert", "Catastrophic stop — approach flagged impossible",
                fields=fields,
                note="Read data/memories/log.md and data/session_exit.md",
            ),
        })
    elif reason == "SIGNATURE_FORGERY":
        events.append({
            "type": "SIGNATURE_FORGERY_UNADDRESSED", "urgent": True, "category": "URGENT",
            "text": _compose_event(
                "alert", "Signature forgery — runner forcing re-entry",
                fields={"Program": prog, "Phase": phase},
                note="See data/accountability_ledger.md",
            ),
        })
    elif reason == "EVALUATOR_FAIL":
        events.append({
            "type": "EVAL_FAIL_EXIT", "urgent": True, "category": "URGENT",
            "text": _compose_event(
                "alert", "Evaluator FAIL unaddressed — forcing re-entry",
                fields={"Program": prog, "Phase": phase},
            ),
        })
    elif reason == "CONTEXT_FULL":
        # Quiet info — context compaction is routine
        pass
    elif reason in ("GRACEFUL_CHECKPOINT", "RATE_LIMIT", "UNKNOWN"):
        pass  # routine / handled elsewhere
    else:
        events.append({
            "type": "SESSION_END_OTHER", "urgent": False, "category": "INFO",
            "text": _compose_event(
                "info", f"Session #{sess_num} ended: {reason}",
                fields=fields,
            ),
        })
    return events


def _evaluator_context(text: str) -> str:
    """Extract a short one-line summary of what the evaluator flagged."""
    # Prefer a top-level "Summary" or "Key finding" section
    m = re.search(r"^(?:##\s*Summary|##\s*Key\s*Finding|##\s*Headline)\s*\n+(.+?)(?=\n##|\Z)",
                  text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip().split("\n")[0][:250]
    # Fallback: first FAIL / FLAG checklist item
    m = re.search(r"^\s*(?:\d+\.\s*)?(.+?(?:FAIL|FLAG)\b.+?)$", text, re.MULTILINE | re.IGNORECASE)
    if m:
        return m.group(1).strip()[:250]
    return ""


def detect_evaluator(state: dict) -> list[dict]:
    """Detect an evaluator verdict change — enriched with program/phase + key finding."""
    events: list[dict] = []
    path = REPO / "data/evaluator_report.md"
    mtime = mtime_or_zero(path)
    if not mtime or mtime <= state["last_evaluator_report_mtime"]:
        return events
    text = read_head(path, 6000)
    m = re.search(r"^(?:Overall|Verdict)\s*[:\-]\s*(\w+)", text, re.MULTILINE | re.IGNORECASE)
    verdict = m.group(1).upper() if m else "UNKNOWN"
    if verdict != state["last_evaluator_verdict"]:
        prog, phase = _current_program_phase()
        ctx = _evaluator_context(text)
        fields = {"Program": prog, "Phase": phase}
        if verdict == "PASS":
            events.append({
                "type": "EVALUATOR_PASS", "urgent": False, "category": "GOOD",
                "text": _compose_event(
                    "good", "Evaluator PASS — phase gate cleared",
                    fields=fields, preview=ctx,
                ),
            })
        elif verdict == "PASS_WITH_FLAGS":
            events.append({
                "type": "EVALUATOR_FLAGS", "urgent": False, "category": "ATTENTION",
                "text": _compose_event(
                    "warn", "Evaluator PASS_WITH_FLAGS — closes with concerns",
                    fields=fields, preview=ctx,
                    note="See data/evaluator_report.md",
                ),
            })
        elif verdict == "FAIL":
            events.append({
                "type": "EVALUATOR_FAIL", "urgent": True, "category": "URGENT",
                "text": _compose_event(
                    "alert", "Evaluator FAIL — Director must remediate",
                    fields=fields, preview=ctx,
                    note="Runner blocks GRACEFUL_CHECKPOINT until addressed",
                ),
            })
        else:
            events.append({
                "type": "EVALUATOR_OTHER", "urgent": False, "category": "INFO",
                "text": _compose_event(
                    "info", f"Evaluator verdict: {verdict}",
                    fields=fields,
                ),
            })
        state["last_evaluator_verdict"] = verdict
    state["last_evaluator_report_mtime"] = mtime
    return events


def _normalize_decision_signature(text: str) -> str:
    """Strip volatile tokens (timestamps, session numbers, counters) from a
    decision text so we can detect when the Director is writing the same
    holding-pattern entry over and over (D-202..D-210 incident).
    """
    if not text:
        return ""
    s = text.lower()
    # Strip dates / timestamps / session numbers / decision IDs
    s = re.sub(r"\d{4}-\d{2}-\d{2}", "", s)
    s = re.sub(r"\d{2}:\d{2}(:\d{2})?", "", s)
    s = re.sub(r"session\s*#?\d+", "session", s)
    s = re.sub(r"d-\d+", "d-", s)
    s = re.sub(r"\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth"
               r"|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth"
               r"|eighteenth|nineteenth|twentieth)\b", "Nth", s)
    s = re.sub(r"\b\d+(?:th|st|nd|rd)\b", "Nth", s)
    s = re.sub(r"\b\d+\b", "N", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Take first 400 chars as signature
    return s[:400]


def detect_new_decision(state: dict) -> list[dict]:
    """Detect a new D-NNN decision — enriched with title + What preview.

    De-dupes holding-loop spam (D-202..D-210 pattern, 2026-04-25): if the
    last 3 decisions had near-identical normalized signatures, we suppress
    further NEW_DECISION posts and instead emit a single HOLDING_LOOP alert
    pointing the operator at any pending USER_*.md decision file. The
    suppression resets when the pattern breaks (a real-content decision
    lands).
    """
    events: list[dict] = []
    path = _mem_or_legacy(REPO / "data/memories/log.md", REPO / "data/decisions_recent.md")
    text = read_head(path, 8000)
    ids = re.findall(r"^##?\s*(?:Decision\s*)?(D-\d+)\b", text, re.MULTILINE | re.IGNORECASE)
    if not ids:
        return events
    latest = ids[0]  # file is newest-first per procedures
    if latest == state["last_decision_id"] or state["last_decision_id"] is None:
        state["last_decision_id"] = latest
        return events

    # Extract title
    m = re.search(rf"^##?\s*(?:Decision\s*)?{re.escape(latest)}\s*[:\-—]?\s*(.+)$",
                  text, re.MULTILINE | re.IGNORECASE)
    title = m.group(1).strip() if m else "(no title)"

    # Extract "What" section (first paragraph after ### What); fall back to first body paragraph
    block_m = re.search(
        rf"^##?\s*(?:Decision\s*)?{re.escape(latest)}[\s\S]*?"
        r"###\s*What\s*\n+(.+?)(?=\n###|\n##|\Z)",
        text, re.MULTILINE | re.IGNORECASE
    )
    preview = ""
    body = ""
    if block_m:
        raw = block_m.group(1).strip()
        lines = [ln for ln in raw.split("\n") if ln.strip()]
        body = " ".join(lines)
        preview_text = " ".join(lines[:3])[:400]
        if preview_text:
            preview = f"\n> {preview_text}"
    else:
        # Fallback: take ~200 chars of body after the title for the signature
        body_m = re.search(rf"^##?\s*(?:Decision\s*)?{re.escape(latest)}[^\n]*\n+(.+?)(?=\n##|\Z)",
                           text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if body_m:
            body = body_m.group(1)[:1000]
            lines = [ln for ln in body.split("\n") if ln.strip()][:3]
            if lines:
                preview = f"\n> {' '.join(lines)[:400]}"

    # ---- Holding-loop de-dupe ----
    sig = _normalize_decision_signature(title + " " + body)
    window = state.get("decision_signature_window", [])
    window.append(sig)
    if len(window) > 5:
        window = window[-5:]
    state["decision_signature_window"] = window

    # Loop heuristic: last 3 normalized signatures all overlap ≥80%, OR contain
    # the explicit "consecutive holding" / "Nth holding session" markers.
    looping = False
    if len(window) >= 3:
        recent3 = window[-3:]
        # Cheap overlap: longest common prefix length / mean length
        prefixes = [len(_common_prefix(recent3[0], s)) for s in recent3[1:]]
        mean_len = sum(len(s) for s in recent3) / 3
        if mean_len > 100 and all(p / mean_len >= 0.7 for p in prefixes):
            looping = True
        # Or explicit phrase-based detection
        if any("consecutive holding" in s or "monitor only" in s or "monitor_only" in s
               or "holding session" in s or "no agent dispatches" in s for s in recent3):
            looping = True

    if looping:
        if not state.get("decision_loop_signaled"):
            state["decision_loop_signaled"] = True
            user_files = sorted(REPO.glob("programs/*/USER_*.md"))
            note = ""
            if user_files:
                note = "Likely cause: " + ", ".join(
                    p.relative_to(REPO).as_posix() for p in user_files
                )
            events.append({
                "type": "HOLDING_LOOP",
                "urgent": True,
                "category": "URGENT",
                "text": _compose_event(
                    "alert",
                    f"Director looping — same decision content for 3+ sessions (last: {latest})",
                    fields={"Token cost": "~17K per session"},
                    note=note,
                ),
            })
        state["last_decision_id"] = latest
        return events

    if state.get("decision_loop_signaled"):
        state["decision_loop_signaled"] = False
        events.append({
            "type": "HOLDING_LOOP_END",
            "urgent": False,
            "category": "GOOD",
            "text": _compose_event(
                "good", f"Holding loop broken — {latest} lands with new content",
            ),
        })

    prog, phase = _current_program_phase()
    events.append({
        "type": "NEW_DECISION", "urgent": False, "category": "INFO",
        "text": _compose_event(
            "info", f"{latest} — {title[:160]}",
            fields={"Program": prog, "Phase": phase},
            preview=preview.lstrip("> ").strip() if preview else "",
        ),
    })
    state["last_decision_id"] = latest
    return events


def _common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return a[:i]
    return a[:n]


def detect_accountability(state: dict) -> list[dict]:
    """Detect new accountability_ledger.md entry (e.g., signature forgery)."""
    events: list[dict] = []
    path = REPO / "data/accountability_ledger.md"
    mtime = mtime_or_zero(path)
    if not mtime or mtime <= state["last_ledger_mtime"]:
        state["last_ledger_mtime"] = mtime
        return events
    text = read_head(path, 5000)
    # Look for the newest entry of form `### YYYY-MM-DD — TYPE — ROLE — DECISION`
    m = re.search(r"^###\s+(\d{4}-\d{2}-\d{2})\s*—\s*(\w+)\s*—\s*(\w+)\s*—\s*(D-\d+)",
                  text, re.MULTILINE)
    if m:
        signature = f"{m.group(1)}|{m.group(2)}|{m.group(3)}|{m.group(4)}"
        if signature != state.get("last_ledger_entry_signature"):
            events.append({
                "type": "ACCOUNTABILITY", "urgent": True, "category": "URGENT",
                "text": _compose_event(
                    "alert",
                    f"Accountability ledger entry — {m.group(2)}",
                    fields={"By": m.group(3), "Decision": m.group(4)},
                    note="See data/accountability_ledger.md",
                ),
            })
            state["last_ledger_entry_signature"] = signature
    state["last_ledger_mtime"] = mtime
    return events


def detect_rate_limit(state: dict) -> list[dict]:
    """Detect rate-limit-hit transitions based on rate_limit_resets_at file."""
    events: list[dict] = []
    path = REPO / "data/infra/rate_limit_resets_at"
    if not path.exists():
        return events
    try:
        resets_at = int(path.read_text().strip())
    except (ValueError, OSError):
        return events
    now = time.time()
    if resets_at <= now:
        if state.get("last_rate_limit_resets_at") and state["last_rate_limit_resets_at"] == resets_at:
            events.append({
                "type": "RATE_LIMIT_END", "urgent": False, "category": "GOOD",
                "text": _compose_event(
                    "good", "Rate limit reset — lab resuming",
                ),
            })
            state["last_rate_limit_resets_at"] = None
        return events
    if state.get("last_rate_limit_resets_at") != resets_at:
        reset_dt = datetime.fromtimestamp(resets_at, TZ)
        reset_local = reset_dt.strftime("%H:%M %a")
        wait_min = (resets_at - now) / 60
        wait_str = f"{wait_min/60:.1f}h" if wait_min > 90 else f"{wait_min:.0f}m"
        prog, phase = _current_program_phase()
        note = ""
        if wait_min > 1440:
            note = "Wait > 24h — file timestamp may be stale"
        events.append({
            "type": "RATE_LIMIT", "urgent": False, "category": "ATTENTION",
            "text": _compose_event(
                "warn", "Rate limit hit — lab paused",
                fields={
                    "Program": prog, "Phase": phase,
                    "Resets": f"{reset_local} (in {wait_str})",
                },
                note=note or "Runner auto-resumes when limit clears",
            ),
        })
        state["last_rate_limit_resets_at"] = resets_at
    return events


def detect_lab_liveness(state: dict) -> list[dict]:
    """Detect lab-dead condition: session log silent > LAB_DEAD_THRESHOLD_MINUTES."""
    events: list[dict] = []
    log_dir = REPO / "data/infra/session_logs"
    if not log_dir.exists():
        return events
    logs = sorted(log_dir.glob("session_*.log"), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if not logs:
        return events
    latest = logs[-1]
    mtime = mtime_or_zero(latest)
    state["last_seen_session_log"] = latest.name
    state["last_seen_session_log_mtime"] = mtime
    age_minutes = (time.time() - mtime) / 60
    # Don't alarm if we're in rate-limit wait
    rate_limit_path = REPO / "data/infra/rate_limit_resets_at"
    in_rate_limit = False
    if rate_limit_path.exists():
        try:
            resets_at = int(rate_limit_path.read_text().strip())
            if resets_at > time.time():
                in_rate_limit = True
        except (ValueError, OSError):
            pass

    if age_minutes > LAB_DEAD_THRESHOLD_MINUTES and not in_rate_limit:
        # Check tmux too
        try:
            r = subprocess.run(["tmux", "has-session", "-t", "agi-lab"],
                               capture_output=True, timeout=5)
            tmux_alive = (r.returncode == 0)
        except Exception:
            tmux_alive = False
        if not tmux_alive or age_minutes > 30:
            if not state.get("lab_dead_notified"):
                prog, phase = _current_program_phase()
                sess = _session_log_summary(latest)
                last_dispatch = sess["dispatches"][-1] if sess.get("dispatches") else ""
                tmux_state = "tmux dead" if not tmux_alive else "tmux alive, log silent"
                events.append({
                    "type": "LAB_DEAD", "urgent": True, "category": "URGENT",
                    "text": _compose_event(
                        "alert", f"Lab dead — silent {age_minutes:.0f}m",
                        fields={
                            "Program": prog, "Phase": phase,
                            "State": tmux_state,
                            "Last dispatch": last_dispatch,
                        },
                        note="Reply `resume` to restart",
                    ),
                })
                state["lab_dead_notified"] = True
    else:
        if state.get("lab_dead_notified"):
            prog, phase = _current_program_phase()
            events.append({
                "type": "LAB_ALIVE", "urgent": False, "category": "GOOD",
                "text": _compose_event(
                    "good", "Lab back online — Director resumed",
                    fields={"Program": prog, "Phase": phase},
                ),
            })
            state["lab_dead_notified"] = False
    return events


def detect_state_status(state: dict) -> list[dict]:
    """Detect VICTORY / CATASTROPHIC written to state.md (defense-in-depth)."""
    events: list[dict] = []
    path = _mem_or_legacy(REPO / "data/memories/current.md", REPO / "data/state.md")
    mtime = mtime_or_zero(path)
    if mtime == state.get("last_state_md_mtime"):
        return events
    text = read_head(path, 5000)
    fingerprint_bits = []
    if re.search(r"status.*VICTORY", text, re.IGNORECASE):
        fingerprint_bits.append("VICTORY")
    if re.search(r"status.*CATASTROPHIC", text, re.IGNORECASE):
        fingerprint_bits.append("CATASTROPHIC")
    fp = "|".join(fingerprint_bits)
    if fp and fp != state.get("last_state_status_fingerprint"):
        if "VICTORY" in fp:
            events.append({
                "type": "VICTORY", "urgent": True, "category": "URGENT",
                "text": _compose_event(
                    "alert", "current.md declares VICTORY",
                    note="Verify in data/eval/scorecard.md",
                ),
            })
        if "CATASTROPHIC" in fp:
            events.append({
                "type": "CATASTROPHIC", "urgent": True, "category": "URGENT",
                "text": _compose_event(
                    "alert", "current.md declares CATASTROPHIC — human review needed",
                ),
            })
        state["last_state_status_fingerprint"] = fp
    state["last_state_md_mtime"] = mtime
    return events


def detect_user_action_required(state: dict) -> list[dict]:
    """Detect any `programs/*/USER_*.md` file that needs operator attention.

    Convention: any file in a program directory whose name starts with `USER_`
    is a user-action-required marker (e.g., `USER_GO_NOGO_DECISION.md`).
    Posts on first sighting (URGENT) and re-posts every 6h while the file's
    `DECISION:` field stays at `PENDING` (or the file remains otherwise
    unactioned).

    Added 2026-04-25 after D-202..D-210 incident — Director sat in 16
    holding sessions waiting for user input on a USER_*.md file the bot
    never surfaced.
    """
    events: list[dict] = []
    repost_interval = 6 * 3600  # 6h re-post cadence while pending
    now = time.time()

    user_files_state = state.get("user_action_files", {})
    seen: set[str] = set()

    for path in sorted(REPO.glob("programs/*/USER_*.md")):
        rel = str(path.relative_to(REPO))
        seen.add(rel)
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        mtime = path.stat().st_mtime

        # Look for DECISION: field. Common conventions:
        decision_match = re.search(r"^\s*DECISION:\s*([A-Za-z_<>0-9 ]+)", content, re.MULTILINE)
        decision_value = decision_match.group(1).strip() if decision_match else "(no DECISION field)"

        prior = user_files_state.get(rel)

        # First sight: post immediately
        if prior is None:
            # Compose a body preview from first non-trivia paragraph
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            preview = ""
            for p in paragraphs:
                if p.startswith("#"):
                    continue
                if "DECISION:" in p:
                    continue
                preview = p[:300]
                break

            events.append({
                "type": "USER_ACTION_REQUIRED",
                "urgent": True,
                "category": "URGENT",
                "text": _compose_event(
                    "alert", "User action required",
                    fields={"File": rel, "DECISION": decision_value},
                    preview=preview,
                    note="Set DECISION: GO or NOGO_<reason> in the file",
                ),
            })
            user_files_state[rel] = {
                "first_seen": now,
                "last_post": now,
                "decision_value": decision_value,
                "mtime": mtime,
            }
            continue

        # Subsequent sights: only re-post if DECISION still PENDING and >6h since last post
        is_pending = decision_value.upper().startswith("PENDING") or decision_value == "(no DECISION field)"
        last_post = prior.get("last_post", 0)
        if is_pending and (now - last_post) >= repost_interval:
            hours_pending = (now - prior.get("first_seen", now)) / 3600
            events.append({
                "type": "USER_ACTION_REMINDER",
                "urgent": True,
                "category": "URGENT",
                "text": _compose_event(
                    "alert", f"Still waiting on user — pending {hours_pending:.1f}h",
                    fields={"File": rel},
                    note="Director sessions are blocked. Set DECISION: GO or NOGO_<reason>",
                ),
            })
            user_files_state[rel]["last_post"] = now
            user_files_state[rel]["decision_value"] = decision_value
        elif not is_pending and prior.get("decision_value") != decision_value:
            events.append({
                "type": "USER_ACTION_RESOLVED",
                "urgent": False,
                "category": "GOOD",
                "text": _compose_event(
                    "good", f"User decision set: {decision_value}",
                    fields={"File": rel},
                ),
            })
            user_files_state.pop(rel, None)

        # Update mtime anyway
        if rel in user_files_state:
            user_files_state[rel]["mtime"] = mtime

    # Drop entries for files that no longer exist (e.g., user deleted to revoke)
    for rel in list(user_files_state.keys()):
        if rel not in seen:
            user_files_state.pop(rel, None)

    state["user_action_files"] = user_files_state
    return events


def _training_is_advancing(fresh_seconds: int = 1800) -> bool:
    """Discriminator for detect_holding_loop (added D-251, 2026-04-28).

    Returns True iff a long-running training process is alive AND its
    stdout.log has been written in the last `fresh_seconds` seconds
    (default 30 min). Used to suppress the holding-loop false-positive when
    the Director's short sessions are CORRECT monitoring of a healthy
    training run (D-237..D-247 shape) rather than the D-202..D-225
    stuck-waiting-for-user shape.

    Mirrors the bash _training_is_advancing in run_agi_lab.sh. Cheap:
    one pgrep + one glob. No log parsing.
    """
    try:
        rc = subprocess.run(
            ["pgrep", "-f",
             r"tools/run_long\.py|tools/run_phase3_factorial\.py|build/scale_experiment"],
            capture_output=True, timeout=5,
        ).returncode
    except (OSError, subprocess.TimeoutExpired):
        return False
    if rc != 0:
        return False

    runs_dir = REPO / "data/runs"
    if not runs_dir.exists():
        return False

    now = time.time()
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        # Exclude the orchestrator's own log dir (not a per-cell run dir).
        if run_dir.name == "phase3_factorial_log":
            continue
        log = run_dir / "stdout.log"
        try:
            st = log.stat()
        except OSError:
            continue
        if (now - st.st_mtime) <= fresh_seconds:
            return True
    return False


def detect_holding_loop(state: dict) -> list[dict]:
    """Detect Director-side holding-loop pattern: many short session_*.log
    files in the trailing 2 hours. Same heuristic as run_agi_lab.sh's
    runner-side backoff. Posts ONCE when the loop becomes evident, re-posts
    if the count grows substantially (every +5 short sessions), clears
    when the pattern breaks (one normal-size session lands).

    Added 2026-04-25 (D-202..D-210 incident).

    D-251 amendment (2026-04-28): when a long-running training process is
    alive AND its stdout.log is advancing (mtime <30 min ago), the short
    Director sessions are CORRECT monitoring (template per D-247: one
    pgrep + one tail + one ls). Suppress the alert in that case so the
    Slack channel doesn't get spammed during healthy P8 EXECUTE phases.
    Genuine-stuck path (no live training OR training frozen >30 min) is
    unchanged. See _training_is_advancing().
    """
    events: list[dict] = []
    log_dir = REPO / "data/infra/session_logs"
    if not log_dir.exists():
        return events

    short_threshold = 8192  # bytes — same as runner
    window_seconds = 7200  # 2h
    trigger_count = 5
    repost_step = 5

    now = time.time()
    short_recent = 0
    for f in log_dir.glob("session_*.log"):
        try:
            st = f.stat()
        except OSError:
            continue
        if (now - st.st_mtime) > window_seconds:
            continue
        if st.st_size < short_threshold:
            short_recent += 1

    last_signaled_at = state.get("holding_loop_signaled_at", 0) or 0
    last_count = state.get("holding_loop_last_count", 0) or 0

    # D-251: live-training discriminator. If a training run is alive and
    # advancing, the short sessions are correct monitoring → suppress alert.
    # Treat as the "no-loop" state: clear any prior signaled_at so we re-post
    # cleanly if the pattern becomes a true stuck-pattern later.
    if short_recent >= trigger_count and _training_is_advancing():
        if last_signaled_at:
            # Was previously signaling — emit a clear so the channel knows
            # the alert was a false positive that's now suppressed.
            events.append({
                "type": "HOLDING_LOOP_END",
                "urgent": False,
                "category": "GOOD",
                "text": _compose_event(
                    "good",
                    "Holding loop cleared — live training advancing, "
                    "Director monitoring is correct (D-251 discriminator)",
                ),
            })
        state["holding_loop_signaled_at"] = 0
        state["holding_loop_last_count"] = 0
        return events

    if short_recent >= trigger_count:
        should_post = False
        if last_signaled_at == 0:
            should_post = True  # first detection
        elif (short_recent - last_count) >= repost_step:
            should_post = True  # pattern persisted and grew

        if should_post:
            user_files = sorted(REPO.glob("programs/*/USER_*.md"))
            note = ""
            if user_files:
                note = "Likely cause: " + ", ".join(
                    p.relative_to(REPO).as_posix() for p in user_files
                )
            events.append({
                "type": "HOLDING_LOOP_DETECTED",
                "urgent": True,
                "category": "URGENT",
                "text": _compose_event(
                    "alert",
                    f"Director looping — {short_recent} short sessions in 2h",
                    fields={"Token cost": f"~{short_recent * 17}K + prompt overhead"},
                    note=note,
                ),
            })
            state["holding_loop_signaled_at"] = now
            state["holding_loop_last_count"] = short_recent
    else:
        if last_signaled_at:
            events.append({
                "type": "HOLDING_LOOP_END",
                "urgent": False,
                "category": "GOOD",
                "text": _compose_event(
                    "good",
                    "Holding loop cleared — Director resuming productive work",
                ),
            })
        state["holding_loop_signaled_at"] = 0
        state["holding_loop_last_count"] = 0

    return events


def detect_phase_boundary(state: dict) -> list[dict]:
    """Detect phase-boundary transitions: phase number changed, status keyword
    flipped (PRE-FLIGHT → EXECUTE → CLOSED), program changed, or VICTORY /
    CATASTROPHIC declared. Posts a human-readable transition (e.g. "Phase 2
    closed → Phase 3 open"), not a fingerprint diff.

    Added 2026-04-25; rewritten D-235 for clean output.
    """
    events: list[dict] = []
    program, phase_short = _current_program_phase()
    if program == "(unknown)" and phase_short == "(unknown)":
        return events

    prior_program = state.get("last_phase_summary_program") or ""
    prior_phase = state.get("last_phase_summary_phase") or ""

    if not prior_program:
        # First scan — record baseline, no post
        state["last_phase_summary_program"] = program
        state["last_phase_summary_phase"] = phase_short
        return events

    program_changed = (program != prior_program) and program != "(unknown)"
    phase_changed = (phase_short != prior_phase) and phase_short != "(unknown)"
    if not (program_changed or phase_changed):
        return events

    if program_changed:
        events.append({
            "type": "PHASE_BOUNDARY",
            "urgent": True,
            "category": "URGENT",
            "text": _compose_event(
                "warn", f"Program switched: {prior_program} → {program}",
                fields={"Phase": phase_short},
            ),
        })
    elif phase_changed:
        urgent = any(
            k in phase_short.upper() for k in ("VICTORY", "CATASTROPHIC", "PIVOT")
        )
        events.append({
            "type": "PHASE_BOUNDARY",
            "urgent": urgent,
            "category": "URGENT" if urgent else "ATTENTION",
            "text": _compose_event(
                "alert" if urgent else "warn",
                f"Phase transition: {prior_phase} → {phase_short}",
                fields={"Program": program},
            ),
        })

    state["last_phase_summary_program"] = program
    state["last_phase_summary_phase"] = phase_short
    return events


_CKPT_RE = re.compile(r"step_(\d+)\.ckpt$")


def _parse_latest_step_signals(stdout_path: Path) -> dict:
    """Read the tail of a cell's stdout.log and pull the latest step's
    loss / PPL / step_time. Returns {} on parse failure.

    Format from src/training/scale_experiment.c print_step:
        '%5d  | %8.4f (...) | %9.2f | %6.3f | ... | %.0f'
        step    loss              ppl      grad_norm        ms
    """
    if not stdout_path.exists():
        return {}
    try:
        with open(stdout_path, errors="replace") as f:
            tail = f.readlines()[-200:]
    except OSError:
        return {}
    step_re = re.compile(
        r"^\s*(\d+)\s+\|\s*([\d.]+)\s*\("
        r".*?\)\s*\|\s*([\d.]+)\s*\|.*\|\s*([\d.]+)\s*$"
    )
    last_step = last_loss = last_ppl = last_ms = None
    for line in tail:
        m = step_re.match(line.rstrip("\n"))
        if not m:
            continue
        try:
            last_step = int(m.group(1))
            last_loss = float(m.group(2))
            last_ppl = float(m.group(3))
            last_ms = float(m.group(4))
        except ValueError:
            continue
    if last_step is None:
        return {}
    return {
        "step": last_step,
        "loss": last_loss,
        "ppl": last_ppl,
        "ms": last_ms,
    }


def _get_cell_rss_mb(cell_name: str) -> "float | None":
    """Resolve current RSS (MB) of the scale_experiment subprocess for
    `cell_name` via ps. Returns None if not found / not running.
    """
    try:
        # Find scale_experiment whose --checkpoint-dir contains the cell
        r = subprocess.run(
            ["pgrep", "-f", f"checkpoint-dir.*{cell_name}"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode != 0:
            return None
        for pid in r.stdout.split():
            ps = subprocess.run(
                ["ps", "-p", pid, "-o", "rss=", "-o", "comm="],
                capture_output=True, text=True, timeout=3,
            )
            if ps.returncode != 0:
                continue
            line = ps.stdout.strip()
            if not line or "scale_experiment" not in line:
                continue
            parts = line.split()
            try:
                rss_kb = int(parts[0])
                return rss_kb / 1024.0
            except (ValueError, IndexError):
                continue
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return None


def detect_training_progress(state: dict) -> list[dict]:
    """Watch for new checkpoints landing during long-run training. Posts
    one event per new ckpt (event-driven) so the Slack channel sees
    progress without the Director needing to wake every 14-30 min for
    routine monitoring.

    Added 2026-04-28 (D-288). Replaces the Director's read-only-monitoring
    role for the per-checkpoint observability use case. Director now wakes
    only on real events (orchestrator state change, loss spikes, drift
    alerts, USER_*.md files, decision-log activity) — not on cadence.

    Posts on:
      1. New checkpoint file appears in data/checkpoints/<phase_dir>/<cell>/
      2. Cell completion (step reaches `steps_per_cell`, default 5000)

    State maintained:
      ckpt_progress[<cell>] = last posted step
      cell_completed[<cell>] = True once we have posted the completion
    """
    events: list[dict] = []
    ckpt_root = REPO / "data/checkpoints/phase3_factorial"
    if not ckpt_root.exists():
        return events
    steps_per_cell = 5000  # locked at prereg P6; bound here for the message

    ckpt_progress = state.get("ckpt_progress", {})
    cell_completed = state.get("cell_completed", {})

    for cell_dir in sorted(ckpt_root.iterdir()):
        if not cell_dir.is_dir():
            continue
        cell_name = cell_dir.name
        # Skip non-cell dirs (e.g. backup/archive subdirs)
        if not re.match(r"^[A-Z]\d+$", cell_name):
            continue
        ckpts = sorted(cell_dir.glob("step_*.ckpt"),
                       key=lambda p: p.stat().st_mtime)
        if not ckpts:
            continue
        latest = ckpts[-1]
        m = _CKPT_RE.search(latest.name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except ValueError:
            continue

        last_posted = ckpt_progress.get(cell_name, 0)
        if step <= last_posted:
            continue

        # New checkpoint event. Gather context.
        stdout_path = REPO / f"data/runs/phase3_{cell_name}/stdout.log"
        sig = _parse_latest_step_signals(stdout_path)
        rss_mb = _get_cell_rss_mb(cell_name)

        progress_pct = (step / steps_per_cell) * 100
        steps_left = max(0, steps_per_cell - step)
        eta_h = ""
        if sig.get("ms") and steps_left:
            eta_h = f"{(steps_left * sig['ms'] / 1000) / 3600:.1f}h"

        fields: dict = {"Progress": f"{progress_pct:.0f}%"}
        if sig.get("loss") is not None:
            fields["Loss"] = f"{sig['loss']:.2f}"
        if sig.get("ppl") is not None:
            fields["PPL"] = f"{sig['ppl']:.0f}"
        if rss_mb is not None:
            fields["RSS"] = f"{rss_mb:.0f} MB"
        if eta_h:
            fields["ETA cell"] = eta_h

        # Cell completion = step exactly at (or beyond) steps_per_cell
        if step >= steps_per_cell and not cell_completed.get(cell_name):
            events.append({
                "type": "CELL_COMPLETE", "urgent": False, "category": "GOOD",
                "text": _compose_event(
                    "good",
                    f"Cell {cell_name} complete: {step}/{steps_per_cell} steps",
                    fields=fields,
                ),
            })
            cell_completed[cell_name] = True
        else:
            events.append({
                "type": "CHECKPOINT_LANDED", "urgent": False, "category": "INFO",
                "text": _compose_event(
                    "info",
                    f"Checkpoint: {cell_name} step {step}/{steps_per_cell}",
                    fields=fields,
                ),
            })
        ckpt_progress[cell_name] = step

    state["ckpt_progress"] = ckpt_progress
    state["cell_completed"] = cell_completed
    return events


DETECTORS = [
    detect_session_exit,
    detect_evaluator,
    detect_new_decision,
    detect_accountability,
    detect_rate_limit,
    detect_lab_liveness,
    detect_state_status,
    # Added 2026-04-25 (D-202..D-210 incident):
    detect_user_action_required,
    detect_holding_loop,
    detect_phase_boundary,
    # Added 2026-04-28 (D-288): event-driven training progress; replaces
    # routine Director monitoring during long-run training.
    detect_training_progress,
]


def poll_once() -> None:
    """One full detection + post cycle."""
    state = load_state()
    events: list[dict] = []
    for detector in DETECTORS:
        try:
            events.extend(detector(state))
        except Exception:
            log.exception("detector failed: %s", detector.__name__)
    save_state(state)
    for ev in events:
        post(ev["text"], category=ev.get("category", "INFO"), urgent=ev.get("urgent", False))
    # Flush digest if quiet hours just ended
    flush_digest_if_due()


def poller_thread() -> None:
    log.info("poller thread start — interval=%ss", POLL_INTERVAL_SECONDS)
    while True:
        try:
            poll_once()
        except Exception:
            log.exception("poll_once failed")
        time.sleep(POLL_INTERVAL_SECONDS)


# ---- Commands ----

def _run(cmd: list[str], timeout: float = 15) -> str:
    try:
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, timeout=timeout, text=True)
        return (r.stdout or r.stderr or "").strip()
    except subprocess.TimeoutExpired:
        return "(command timed out)"
    except Exception as e:
        return f"(error: {e})"


def cmd_help(_args: list[str]) -> str:
    return (
        "*AGI Lab bot — commands*\n"
        "*Snapshots:*\n"
        "`summary` comprehensive one-glance snapshot (phone-friendly)\n"
        "`status` lab + program + phase + session + dispatches + verdict\n"
        "`phase` current phase + sub-step checklist\n"
        "`program` program question + status\n"
        "`last` last session + evaluator verdict + decision\n"
        "`recent` last 5 sessions one-liner each\n"
        "*Audit:*\n"
        "`decisions` last 10 D-NNN entries\n"
        "`forgeries` accountability ledger entries\n"
        "`dispatches` agents dispatched this session\n"
        "`commits` last 10 git commits\n"
        "*Control:*\n"
        "`stop` graceful lab halt\n"
        "`resume` restart lab (caffeinate-wrapped)\n"
        "`restart` stop then resume\n"
        "`pause N` buffer notifications for N minutes\n"
        "*Meta:*\n"
        "`ping` liveness check\n"
        "`help` this list\n"
    )


def cmd_ping(_args: list[str]) -> str:
    state = load_state()
    uptime_sec = int(time.time() - START_TIME)
    uptime_str = f"{uptime_sec // 3600}h{(uptime_sec % 3600) // 60}m"
    tmux_r = subprocess.run(["tmux", "has-session", "-t", "agi-lab"], capture_output=True)
    tmux = "alive" if tmux_r.returncode == 0 else "dead"
    last_log = state.get("last_seen_session_log", "?")
    return (f"✅ Alive. Lab tmux: *{tmux}*. Bot uptime: {uptime_str}. "
            f"Last session log: `{last_log}`. Current time: {now_tz().strftime('%H:%M %Z')}.")


_BACKTICK_RE = re.compile(r"`+([^`]+)`+")
_BOLD_RE = re.compile(r"\*\*+([^*]+)\*\*+")


def _strip_md(s: str) -> str:
    """Remove backticks and bold markers — they double-quote in Slack."""
    s = _BACKTICK_RE.sub(r"\1", s)
    s = _BOLD_RE.sub(r"\1", s)
    return s.strip()


def _read_state_block() -> tuple[str, str, str]:
    """Return (program, phase, status) for the command handlers.

    Uses _current_program_phase() for clean program slug + short phase,
    plus a one-line status summary scraped from current.md (falls back
    to "(unknown)" instead of "?").
    """
    program, phase_short = _current_program_phase()
    text = read_head(
        _mem_or_legacy(REPO / "data/memories/current.md", REPO / "data/state.md"),
        4000,
    )
    status = "(unknown)"
    cp = re.search(
        r"^##\s*Current\s+Phase\s*[:\-]\s*(.+)$",
        text, re.MULTILINE | re.IGNORECASE,
    )
    if cp:
        # First sentence (up to em-dash, period, or 200 chars)
        raw = cp.group(1).strip()
        first = re.split(r"\s+(?:—|\.|\(D-)", raw, maxsplit=1)[0]
        status = _strip_md(first)[:240] or "(unknown)"
    return program, phase_short, status


def cmd_status(_args: list[str]) -> str:
    """Lab + program + phase + session + dispatches + verdict, scannable."""
    prog, phase, status = _read_state_block()
    state = load_state()
    mtime = state.get("last_seen_session_log_mtime", 0)
    age_min = int((time.time() - mtime) / 60) if mtime else 0
    r = subprocess.run(["tmux", "has-session", "-t", "agi-lab"], capture_output=True)
    tmux_state = "alive" if r.returncode == 0 else "dead"
    rate_note = ""
    rate_path = REPO / "data/infra/rate_limit_resets_at"
    if rate_path.exists():
        try:
            resets = int(rate_path.read_text().strip())
            if resets > time.time():
                wait_min = int((resets - time.time()) / 60)
                rate_note = f"rate-limited, resets in {wait_min}m"
        except (ValueError, OSError):
            pass
    log_dir = REPO / "data/infra/session_logs"
    logs = sorted(
        log_dir.glob("session_*.log"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True,
    )
    disp_str = ""
    if logs:
        sess = _session_log_summary(logs[0])
        disp_str = _format_dispatches(sess.get("dispatches", []))
    verdict = state.get("last_evaluator_verdict") or ""
    dec_id = state.get("last_decision_id") or ""
    fields = {
        "Lab": tmux_state + (f" ({rate_note})" if rate_note else ""),
        "Program": prog,
        "Phase": phase,
        "Last session": f"{age_min}m ago" if age_min else "",
        "Dispatches": disp_str,
        "Last verdict": verdict,
        "Last decision": dec_id,
    }
    return _compose_event("info", "Lab status", fields=fields, preview=status)


def cmd_phase(_args: list[str]) -> str:
    prog, phase, _ = _read_state_block()
    # Find phase sub-steps if listed in state.md
    text = read_head(_mem_or_legacy(REPO / "data/memories/current.md", REPO / "data/state.md"), 10000)
    # Look for a "Phase N Sub-steps" block
    m = re.search(r"^##\s*Phase\s+\d+\s+Sub-steps.*?\n([\s\S]*?)(?=\n##|\Z)",
                  text, re.MULTILINE | re.IGNORECASE)
    substeps = ""
    if m:
        lines = [ln.strip() for ln in m.group(1).split("\n") if ln.strip().startswith("-")]
        substeps = "\n".join(lines[:15])[:1500]
        if substeps:
            substeps = f"\n*Sub-steps:*\n{substeps}"
    return f"*Program:* {prog}\n*Phase:* {phase}{substeps}"


def cmd_program(_args: list[str]) -> str:
    prog, phase, status = _read_state_block()
    text = read_head(_mem_or_legacy(REPO / "data/memories/current.md", REPO / "data/state.md"), 10000)
    q_match = re.search(r"^##\s*Program\s+Question[\s\S]*?\n([\s\S]*?)(?:\n##|\Z)", text, re.MULTILINE)
    q = q_match.group(1).strip()[:800] if q_match else ""
    return f"*Program:* {prog}\n*Phase:* {phase}\n*Status:* {status[:200]}\n\n*Question:*\n{q}"


def cmd_last(_args: list[str]) -> str:
    """Last evaluator verdict + last session summary + last decision."""
    state = load_state()
    verdict = state.get("last_evaluator_verdict") or "(no evaluator verdict recorded)"
    log_dir = REPO / "data/infra/session_logs"
    logs = sorted(log_dir.glob("session_*.log"),
                  key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    sess_line = "(no session log)"
    if logs:
        sess = _session_log_summary(logs[0])
        from collections import Counter
        disp_count = Counter(sess.get("dispatches", []))
        disp = ", ".join(f"`{r}`×{n}" for r, n in disp_count.most_common(5)) or "none"
        sess_line = (f"`{logs[0].name}` · duration: {sess.get('duration_min', 0):.0f}m · "
                     f"dispatches: {disp}")
    dec = state.get("last_decision_id") or "?"
    return (f"*Last evaluator verdict:* `{verdict}`\n"
            f"*Latest session:* {sess_line}\n"
            f"*Latest decision:* `{dec}`")


def cmd_summary(_args: list[str]) -> str:
    """One-shot snapshot — phone-friendly."""
    prog, phase, status = _read_state_block()
    state = load_state()
    r = subprocess.run(["tmux", "has-session", "-t", "agi-lab"], capture_output=True)
    tmux_state = "running" if r.returncode == 0 else "stopped"
    log_dir = REPO / "data/infra/session_logs"
    logs = sorted(
        log_dir.glob("session_*.log"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True,
    )
    session_str = ""
    if logs:
        sess = _session_log_summary(logs[0])
        age_min = int((time.time() - logs[0].stat().st_mtime) / 60)
        top = _format_dispatches(sess.get("dispatches", []), top_n=3)
        sn = sess.get("session_num") or 0
        bits = []
        if sn:
            bits.append(f"#{sn}")
        if age_min:
            bits.append(f"{age_min}m old")
        if top:
            bits.append(f"top: {top}")
        session_str = " · ".join(bits)
    rate_note = ""
    rate_path = REPO / "data/infra/rate_limit_resets_at"
    if rate_path.exists():
        try:
            resets = int(rate_path.read_text().strip())
            wait_min = int((resets - time.time()) / 60)
            if 0 < wait_min < 1440:
                rate_note = f"rate-limited, resets in {wait_min}m"
        except (ValueError, OSError):
            pass
    queue = len(state.get("digest_queue", []))
    verdict = state.get("last_evaluator_verdict") or ""
    dec = state.get("last_decision_id") or ""
    ledger_path = REPO / "data/accountability_ledger.md"
    forgery_count = 0
    if ledger_path.exists():
        forgery_count = len(re.findall(
            r"^###\s+\d{4}-\d{2}-\d{2}",
            read_head(ledger_path, 10000), re.MULTILINE,
        ))
    fields = {
        "Lab": tmux_state + (f" ({rate_note})" if rate_note else ""),
        "Program": prog,
        "Phase": phase,
        "Session": session_str,
        "Last verdict": verdict,
        "Last decision": dec,
    }
    if queue:
        fields["Digest queued"] = str(queue)
    if forgery_count:
        fields["Governance incidents"] = str(forgery_count)
    severity = "alert" if forgery_count else ("warn" if rate_note else "info")
    return _compose_event(
        severity,
        f"AGI Lab — {now_tz().strftime('%H:%M %Z')}",
        fields=fields,
        preview=status,
    )


def cmd_recent(_args: list[str]) -> str:
    log_dir = REPO / "data/infra/session_logs"
    logs = sorted(log_dir.glob("session_*.log"),
                  key=lambda p: p.stat().st_mtime if p.exists() else 0,
                  reverse=True)[:5]
    if not logs:
        return "(no sessions yet)"
    lines = []
    for p in logs:
        age = (time.time() - p.stat().st_mtime) / 60
        # Try to get first non-empty line
        try:
            with p.open() as f:
                for raw in f:
                    if "Session" in raw and "starting" in raw:
                        lines.append(f"`{p.name}` ({age:.0f}m ago)")
                        break
                else:
                    lines.append(f"`{p.name}` ({age:.0f}m ago)")
        except Exception:
            lines.append(f"`{p.name}`")
    return "*Last 5 sessions:*\n" + "\n".join(lines)


def cmd_decisions(_args: list[str]) -> str:
    text = read_head(_mem_or_legacy(REPO / "data/memories/log.md", REPO / "data/decisions_recent.md"), 8000)
    ids = re.findall(r"^##\s*Decision\s*(D-\d+)\s*[:\-—]?\s*(.*?)$", text, re.MULTILINE | re.IGNORECASE)
    if not ids:
        return "(no decisions found)"
    lines = [f"*{d_id}*: {title.strip()[:180]}" for d_id, title in ids[:10]]
    return "*Last 10 decisions:*\n" + "\n".join(lines)


def cmd_forgeries(_args: list[str]) -> str:
    path = REPO / "data/accountability_ledger.md"
    if not path.exists():
        return "(no ledger file)"
    text = read_head(path, 8000)
    entries = re.findall(r"^###\s+(\d{4}-\d{2}-\d{2})\s*—\s*(\w+)\s*—\s*(\w+)\s*—\s*(D-\d+)",
                        text, re.MULTILINE)
    if not entries:
        return "✅ No governance incidents recorded."
    lines = [f"• {date} *{typ}* — role `{role}` ({dec})" for date, typ, role, dec in entries[:10]]
    return "*Accountability ledger (last 10):*\n" + "\n".join(lines)


def cmd_dispatches(_args: list[str]) -> str:
    state = load_state()
    log_name = state.get("last_seen_session_log")
    if not log_name:
        return "(no session log known yet)"
    log_path = REPO / "data/infra/session_logs" / log_name
    if not log_path.exists():
        return f"(session log {log_name} missing)"
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(unable to read session log)"
    dispatches = re.findall(r">>> AGENT\s+\[?(\w+)", text)
    if not dispatches:
        return "(no dispatches in current session)"
    from collections import Counter
    counts = Counter(dispatches)
    lines = [f"• `{name}` ×{n}" for name, n in counts.most_common(20)]
    total = sum(counts.values())
    return f"*Dispatches this session ({total} total):*\n" + "\n".join(lines)


def cmd_commits(_args: list[str]) -> str:
    out = _run(["git", "log", "-10", "--format=%h %s"])
    return "*Last 10 commits:*\n```\n" + out + "\n```"


def cmd_stop(_args: list[str]) -> str:
    r = subprocess.run(["tmux", "has-session", "-t", "agi-lab"], capture_output=True)
    if r.returncode != 0:
        return "Lab is not running (no tmux session)."
    _run(["tmux", "kill-session", "-t", "agi-lab"])
    return "🛑 Lab stopped (tmux session killed). Run `resume` to restart."


def cmd_resume(_args: list[str]) -> str:
    r = subprocess.run(["tmux", "has-session", "-t", "agi-lab"], capture_output=True)
    if r.returncode == 0:
        return "Lab already running. `stop` first if you want a fresh start."
    out = _run(["make", "lab-start"])
    return f"▶️ Lab restart invoked. Output:\n```\n{out}\n```"


def cmd_restart(_args: list[str]) -> str:
    stop_out = cmd_stop([])
    time.sleep(2)
    resume_out = cmd_resume([])
    return f"🔄 Restart:\n{stop_out}\n{resume_out}"


def cmd_pause(args: list[str]) -> str:
    if not args or not args[0].lstrip("-").isdigit():
        return "Usage: `pause N` where N is minutes (e.g., `pause 30`)."
    minutes = int(args[0])
    state = load_state()
    state["pause_until_ts"] = time.time() + minutes * 60
    save_state(state)
    return f"⏸️ Bot notifications paused for {minutes} min. Events will be queued to digest."


COMMAND_HANDLERS = {
    "help": cmd_help,
    "ping": cmd_ping,
    "summary": cmd_summary,
    "status": cmd_status,
    "phase": cmd_phase,
    "program": cmd_program,
    "last": cmd_last,
    "recent": cmd_recent,
    "decisions": cmd_decisions,
    "forgeries": cmd_forgeries,
    "dispatches": cmd_dispatches,
    "commits": cmd_commits,
    "stop": cmd_stop,
    "resume": cmd_resume,
    "restart": cmd_restart,
    "pause": cmd_pause,
}


def handle_text(text: str) -> str:
    """Route a command message to the right handler."""
    # Strip bot mention if present
    text = re.sub(r"<@U\w+>\s*", "", text).strip()
    if not text:
        return "(empty command — try `help`)"
    parts = text.split()
    cmd = parts[0].lower()
    args = parts[1:]
    handler = COMMAND_HANDLERS.get(cmd)
    if not handler:
        return f"Unknown command: `{cmd}`. Try `help`."
    try:
        return handler(args)
    except Exception as e:
        log.exception("command failed: %s", cmd)
        return f"❌ Command `{cmd}` raised an error: {e}"


@app.event("app_mention")
def on_mention(event, say, logger):
    try:
        response = handle_text(event.get("text", ""))
        say(text=response, thread_ts=event.get("ts"), mrkdwn=True)
    except Exception:
        logger.exception("app_mention handler failed")


@app.event("message")
def on_message(event, say, logger):
    # Only react to direct messages, not channel messages
    if event.get("channel_type") != "im":
        return
    # Ignore messages from bots (including self)
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return
    try:
        response = handle_text(event.get("text", ""))
        say(text=response, mrkdwn=True)
    except Exception:
        logger.exception("dm handler failed")


# ---- Main ----

def main():
    log.info("AGI Lab Slack bot starting — repo=%s channel=%s", REPO, CHANNEL)
    log.info("Quiet hours: %s–%s %s", QUIET_START, QUIET_END, TZ.zone)

    # Hello message
    try:
        post(
            f"👋 *AGI Lab bot live.* Repo: `{REPO.name}`. "
            f"Quiet hours {QUIET_START.strftime('%H:%M')}–{QUIET_END.strftime('%H:%M')} {TZ.zone}. "
            f"Say `help` (DM) or `@bot help` (channel) for commands.",
            category="GOOD", urgent=True,
        )
    except Exception:
        log.exception("hello message failed")

    # Start background poller
    t = threading.Thread(target=poller_thread, daemon=True, name="poller")
    t.start()

    # Main thread: Socket Mode handler for commands
    handler = SocketModeHandler(app, APP_TOKEN)
    handler.start()


if __name__ == "__main__":
    main()
