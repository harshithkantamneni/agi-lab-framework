"""Tests for the dispatch-identity audit (see tools/verify_signatures.py).

The audit's claim is narrow and mechanical: a receipt filed under data/agents/<role>/
episodic/ is backed only if the runner actually dispatched subagent_type=<role> at or
before the receipt's date. These tests pin the three things that can go wrong: reading
the runner's log format, distinguishing a typed dispatch from a generic one whose
description merely names a role, and the date comparison itself.
"""
import datetime as dt
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import verify_signatures as vs  # noqa: E402

ROLES = ["director", "pi", "red_team", "hypothesis_generator", "chief_scientist"]


def _log(tmp: Path, name: str, body: str, header: bool = True) -> Path:
    """Write a session log. The header time is derived from the filename stamp, because
    the audit prefers the header and a shared fixed header would silently collapse every
    fixture log onto one timestamp."""
    tmp.mkdir(parents=True, exist_ok=True)
    p = tmp / name
    hdr = ""
    if header:
        m = re.search(r"_(\d{8})_(\d{6})\.log$", name)
        t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        hdr = f"=== Session 1 starting at {t:%a %b %d %H:%M:%S} CDT {t:%Y} ===\n"
    p.write_text(hdr + body, encoding="utf-8")
    return p


def test_reads_session_header_time(tmp_path):
    d = tmp_path / "logs"
    d.mkdir(parents=True)
    p = d / "session_8_19990101_000000.log"   # filename stamp deliberately wrong
    p.write_text("=== Session 8 starting at Fri Apr 17 14:02:16 CDT 2026 ===\n",
                 encoding="utf-8")
    assert vs._log_start_time(p, p.read_text()) == dt.datetime(2026, 4, 17, 14, 2, 16)


def test_falls_back_to_filename_stamp(tmp_path):
    d = tmp_path / "logs"
    d.mkdir(parents=True)
    p = d / "session_72_20260424_125038.log"
    p.write_text("no header here\n>>> AGENT pi - review\n", encoding="utf-8")  # noqa
    assert vs._log_start_time(p, p.read_text()) == dt.datetime(2026, 4, 24, 12, 50, 38)


def test_strips_ansi_before_matching():
    """The runner colourises the role name, so the literal string ">>> AGENT pi" never
    appears in the raw bytes. A grep-level audit finds nothing; this must not."""
    assert vs._ANSI.sub("", "\x1b[35m>>> AGENT\x1b[0m \x1b[1mpi\x1b[0m") == ">>> AGENT pi"


def test_typed_dispatch_is_recorded(tmp_path):
    vs.KNOWN_ROLES = set(ROLES)
    d = tmp_path / "logs"
    _log(d, "session_72_20260424_125038.log",
         "\x1b[35m>>> AGENT\x1b[0m \x1b[1mpi\x1b[0m — Phase 2 co-sign\n")
    first, generic, n_logs, n_disp = vs.read_dispatch_log(d)
    assert n_logs == 1 and n_disp == 1
    assert "pi" in first
    assert generic == []


def test_generic_dispatch_naming_a_role_is_flagged_not_credited(tmp_path):
    """The D-109 remediation, verbatim in shape: subagent_type=general-purpose with the
    role named in the description. It must not count as a pi dispatch.

    Note: KNOWN_ROLES is deliberately NOT populated here. read_dispatch_log must load
    the roster itself, or the flag silently never fires for a direct caller."""
    d = tmp_path / "logs"
    _log(d, "session_8_20260417_140216.log",
         ">>> AGENT general-purpose — Phase 1 lock + SQ1 amendment review (pi role)\n")
    first, generic, _, n_disp = vs.read_dispatch_log(d)
    assert n_disp == 1
    assert "pi" not in first
    assert "general-purpose" in first
    assert len(generic) == 1 and generic[0]["named_roles"] == ["pi"]


def test_generic_dispatch_without_a_role_name_is_not_flagged(tmp_path):
    vs.KNOWN_ROLES = set(ROLES)
    d = tmp_path / "logs"
    _log(d, "session_2_20260417_095931.log",
         ">>> AGENT general-purpose — 1d — Tractability check\n")
    _, generic, _, _ = vs.read_dispatch_log(d)
    assert generic == []


def test_launching_agent_form_also_counts(tmp_path):
    vs.KNOWN_ROLES = set(ROLES)
    d = tmp_path / "logs"
    _log(d, "session_1_20260501_010101.log", ">>> Launching agent: red_team\n")
    first, _, _, n_disp = vs.read_dispatch_log(d)
    assert n_disp == 1 and "red_team" in first


def _lab(tmp_path, receipts, logs):
    """Build a miniature lab tree and point the module at it."""
    root = tmp_path / "lab"
    for role, names in receipts.items():
        ep = root / "data" / "agents" / role / "episodic"
        ep.mkdir(parents=True, exist_ok=True)
        for n in names:
            (ep / n).write_text("# receipt\n", encoding="utf-8")
    ld = root / "data" / "infra" / "session_logs"
    for name, body in logs.items():
        _log(ld, name, body)
    vs.REPO_ROOT = root
    vs.AGENTS_DIR = root / "data" / "agents"
    vs.SESSION_LOG_DIR = ld
    vs.KNOWN_ROLES = set(ROLES)
    return root


def test_receipt_before_first_typed_dispatch_is_a_violation(tmp_path):
    _lab(tmp_path,
         {"pi": ["2026-04-17_program_1_phase_1_and_2.md"]},
         {"session_8_20260417_140216.log":
              ">>> AGENT general-purpose — Phase 1 lock review (pi role)\n",
          "session_72_20260424_125038.log": ">>> AGENT pi — real co-sign\n"})
    v, st = vs.audit_dispatch_identity()
    assert st["receipts"] == 1
    assert len(v) == 1
    assert "first dispatch with subagent_type=pi was 2026-04-24" in v[0].reason


def test_receipt_on_or_after_first_typed_dispatch_is_backed(tmp_path):
    _lab(tmp_path,
         {"pi": ["2026-04-25_phase_3_lock.md"]},
         {"session_72_20260424_125038.log": ">>> AGENT pi — real co-sign\n"})
    v, _ = vs.audit_dispatch_identity()
    assert v == []


def test_role_never_dispatched_is_a_violation(tmp_path):
    _lab(tmp_path,
         {"hypothesis_generator": ["2026-04-18_program_2_phase_1.md"]},
         {"session_1_20260418_002820.log": ">>> AGENT general-purpose — ideas\n"})
    v, _ = vs.audit_dispatch_identity()
    assert len(v) == 1
    assert "never instantiated" in v[0].reason


def test_director_receipts_are_excluded(tmp_path):
    """The Director is the top-level session, not a dispatched subagent, so its own
    receipts can never be backed by a dispatch line and must not be reported."""
    _lab(tmp_path,
         {"director": ["2026-04-20_program_2_phase_2_launch.md"]},
         {"session_1_20260420_000000.log": ">>> AGENT general-purpose — work\n"})
    v, st = vs.audit_dispatch_identity()
    assert st["receipts"] == 0
    assert v == []


def test_no_logs_is_reported_as_inconclusive_not_clean(tmp_path):
    """.gitignore excludes session_logs, so a fresh clone has none. Absent evidence must
    exit 2 (cannot judge), never 0 (verified)."""
    root = _lab(tmp_path, {"pi": ["2026-04-17_x.md"]}, {})
    vs.SESSION_LOG_DIR = root / "data" / "infra" / "session_logs_missing"
    assert vs.run_identity_audit() == 2
