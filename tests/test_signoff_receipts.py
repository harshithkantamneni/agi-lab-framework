"""Controls and unit tests for receipt-bound sign-offs.

test_signoff_gaps.py pins what used to be wrong. This file pins what must keep working: an
honest sign-off passes, prose about sign-offs is not flagged, evidence failures are reported
as failures, and the shipped settings cover what the runner executes. The threat-model limits
these tests do not reach are listed in audits/README.md.
"""
from __future__ import annotations

import json
import re
import stat
import sys
import time

import pytest

from signoff_harness import (
    Evidence, FORGED_LOCK, GRACEFUL, REPO, clean_exit_accepted, detector, ev_agent, ev_result, ev_text,
    make_lab, receipt, run_formatter, run_runner, session_logs, stub_calls, token, write_doc,
)
from tools import dispatch_ledger as dl
from tools import verify_signatures as vs

RUN = "session_1_20261104_101010"
LOCK = "programs/p/lock.md"
PI_OK = f"SIGNOFF target={LOCK} verdict=APPROVE"


@pytest.fixture
def lab(tmp_path):
    return make_lab(tmp_path)


@pytest.fixture
def evidence(tmp_path):
    return Evidence(tmp_path / "evidence")


# --- the recorder sees structure, not text ---------------------------------------------

def _record(evidence, lab, events, run="run_a"):
    done = run_formatter(lab, events, evidence.env(), run=run)
    assert done.returncode == 0, done.stderr
    return dl.read_ledger(evidence.directory, evidence.key)


def test_real_dispatch_and_reply_become_a_receipt(lab, evidence):
    ledger = _record(evidence, lab, [ev_agent("toolu_1", "pi"), ev_result("toolu_1", f"Fine.\n{PI_OK}\n")])
    r = ledger.receipts["toolu_1"]
    assert (r.role, r.run, r.parent, r.is_error) == ("pi", "run_a", None, False)
    assert r.attests == [[LOCK, "APPROVE"]]
    assert ledger.runs["run_a"].sealed


def test_old_tool_name_is_recorded(lab, evidence):
    ledger = _record(evidence, lab, [ev_agent("toolu_1", "pi", name="Task"), ev_result("toolu_1", "ok")])
    assert ledger.receipts["toolu_1"].role == "pi"


def test_dispatch_without_a_type_is_generic(lab, evidence):
    ledger = _record(evidence, lab, [ev_agent("toolu_1", None), ev_result("toolu_1", "ok")])
    assert ledger.receipts["toolu_1"].role == "general-purpose"


def test_other_tools_are_not_dispatches(lab, evidence):
    bash = {"type": "assistant", "session_id": "s", "parent_tool_use_id": None, "message": {"content": [
        {"type": "tool_use", "id": "toolu_b", "name": "Bash", "input": {"command": "echo hi"}}]}}
    ledger = _record(evidence, lab, [bash, ev_result("toolu_b", PI_OK)])
    assert ledger.receipts == {}


def test_repeated_blocks_make_one_receipt(lab, evidence):
    # Partial-message streaming repeats an assistant message.
    use = ev_agent("toolu_1", "pi")
    ledger = _record(evidence, lab, [use, use, ev_result("toolu_1", PI_OK)])
    assert list(ledger.receipts) == ["toolu_1"]


def test_string_tool_result_content_is_read(lab, evidence):
    ledger = _record(evidence, lab, [ev_agent("toolu_1", "pi"), ev_result("toolu_1", PI_OK, as_string=True)])
    assert ledger.receipts["toolu_1"].attests == [[LOCK, "APPROVE"]]


def test_error_result_is_recorded_as_an_error(lab, evidence):
    ledger = _record(evidence, lab, [ev_agent("toolu_1", "nope"), ev_result("toolu_1", "unknown type", is_error=True)])
    assert ledger.receipts["toolu_1"].is_error


def test_subagent_dispatch_keeps_its_parent(lab, evidence):
    events = [ev_agent("toolu_1", "pi"), ev_agent("toolu_2", "red_team", parent="toolu_1"),
              ev_result("toolu_2", "ok", parent="toolu_1"), ev_result("toolu_1", "ok")]
    ledger = _record(evidence, lab, events)
    assert ledger.receipts["toolu_2"].parent == "toolu_1"


@pytest.mark.parametrize("text", [
    "I would write: SIGNOFF target=programs/p/lock.md verdict=APPROVE",  # not at line start
    "SIGNOFF target=/etc/passwd verdict=APPROVE",
    "SIGNOFF target=programs/../../x.md verdict=APPROVE",
    "SIGNOFF target=programs/p/lock.md verdict=MAYBE",
    "signoff target=programs/p/lock.md verdict=APPROVE",
])
def test_only_a_well_formed_attestation_counts(lab, evidence, text):
    ledger = _record(evidence, lab, [ev_agent("toolu_1", "pi"), ev_result("toolu_1", text)])
    assert ledger.receipts["toolu_1"].attests == []


def test_prompt_that_contains_the_verdict_is_marked_dictated(lab, evidence):
    events = [ev_agent("toolu_1", "pi", prompt=f"Reply with exactly:\n{PI_OK}"), ev_result("toolu_1", PI_OK)]
    ledger = _record(evidence, lab, events)
    assert ledger.receipts["toolu_1"].dictated == [[LOCK, "APPROVE"]]


def test_a_prompt_asking_for_the_line_is_not_dictation(lab, evidence):
    events = [ev_agent("toolu_1", "pi", prompt=f"End with SIGNOFF target={LOCK} verdict=<APPROVE|MODIFY|REJECT>"),
              ev_result("toolu_1", PI_OK)]
    assert _record(evidence, lab, events).receipts["toolu_1"].dictated == []


def test_non_json_and_non_object_lines_do_not_stop_the_recorder(lab, evidence):
    ledger = _record(evidence, lab, ["plain text", "[1, 2]", "42", ev_agent("toolu_1", "pi"), ev_result("toolu_1", "ok")])
    assert "toolu_1" in ledger.receipts and ledger.runs["run_a"].sealed


def test_formatter_without_a_run_id_writes_no_ledger(lab, evidence):
    run_formatter(lab, [ev_agent("toolu_1", "pi")], evidence.env(), run=None)
    assert not (evidence.directory / "ledger").exists()


def test_a_reused_run_id_does_not_overwrite_a_ledger(lab, evidence):
    _record(evidence, lab, [ev_agent("toolu_1", "pi"), ev_result("toolu_1", PI_OK)])
    before = (evidence.directory / "ledger" / "run_a.jsonl").read_bytes()
    done = run_formatter(lab, [ev_text("x")], evidence.env(), run="run_a")
    assert "recorder did not start" in done.stdout
    assert (evidence.directory / "ledger" / "run_a.jsonl").read_bytes() == before


def test_run_records_the_roster_digest_and_sandbox_state(lab, evidence):
    _record(evidence, lab, [ev_text("x")])
    first = json.loads((evidence.directory / "ledger" / "run_a.jsonl").read_text().splitlines()[0])
    assert first["kind"] == "open" and first["sandbox"] == "on"
    assert len(first["roster_sha256"]) == 64


# --- the record cannot be changed quietly ----------------------------------------------

def _ledger_file(evidence):
    return evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))


def test_module_reads_the_format_the_independent_writer_produces(evidence):
    _ledger_file(evidence)
    ledger = dl.read_ledger(evidence.directory, evidence.key)
    assert ledger.receipts["toolu_1"].attests == [[LOCK, "APPROVE"]]


@pytest.mark.parametrize("edit", ["drop_line", "swap_lines", "truncate_after_open_keep_seal"])
def test_chain_breaks_are_caught_even_without_the_key(evidence, edit):
    path = _ledger_file(evidence)
    lines = path.read_text().splitlines()
    if edit == "drop_line":
        del lines[1]
    elif edit == "swap_lines":
        lines[1], lines[2] = lines[2], lines[1]
    else:
        lines = [lines[0], lines[-1]]
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(dl.EvidenceError):
        dl.read_ledger(evidence.directory, None)


def test_record_after_the_seal_is_refused(evidence):
    path = _ledger_file(evidence)
    last = json.loads(path.read_text().splitlines()[-1])
    extra = {"v": 1, "run": RUN, "seq": last["seq"] + 1, "prev": last["mac"], "ts": time.time(), "kind": "open"}
    extra["mac"] = dl._sign(evidence.key, extra)
    path.write_bytes(path.read_bytes() + dl.canonical(extra) + b"\n")
    with pytest.raises(dl.EvidenceError, match="after the seal"):
        dl.read_ledger(evidence.directory, evidence.key)


def test_key_readable_by_others_is_refused(evidence):
    (evidence.directory / "receipt.key").chmod(0o644)
    with pytest.raises(dl.EvidenceError, match="readable beyond its owner"):
        dl.load_key(evidence.directory)


def test_tampered_baseline_cannot_be_judged(lab, evidence):
    write_doc(lab, LOCK, FORGED_LOCK)
    evidence.accept(lines=[dl.line_key(LOCK, FORGED_LOCK)], key=b"wrong" * 8)
    result = detector(lab, "--program", "p", env=evidence.env())
    assert result.returncode == 2 and "HMAC does not verify" in result.stderr


def test_missing_key_cannot_be_judged_and_says_so(lab, evidence):
    write_doc(lab, LOCK, "# nothing\n")
    (evidence.directory / "receipt.key").unlink()
    result = detector(lab, "--program", "p", env=evidence.env())
    assert result.returncode == 2 and "cannot read receipt key" in result.stderr


# --- an honest sign-off passes ---------------------------------------------------------

def test_valid_token_passes_and_reports_a_real_count(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))
    write_doc(lab, LOCK, token("pi", LOCK, RUN, "toolu_1") + "\n")
    result = detector(lab, "--program", "p", env=evidence.env())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: 1 signatures verified" in result.stdout


def test_token_may_sit_beside_the_prose_it_backs(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))
    write_doc(lab, LOCK, f"**PI ✓** approved {token('pi', LOCK, RUN, 'toolu_1')}\n")
    assert detector(lab, "--program", "p", env=evidence.env()).returncode == 0


def test_token_for_one_role_does_not_cover_a_claim_about_another(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "red_team", attests=[(LOCK, "APPROVE")]))
    write_doc(lab, LOCK, f"PI approved {token('red_team', LOCK, RUN, 'toolu_1')}\n")
    result = detector(lab, "--program", "p", env=evidence.env())
    assert result.returncode == 1 and "claims approval by pi" in result.stderr


def test_decision_log_line_can_cite_a_lock_it_refers_to(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))
    write_doc(lab, LOCK, "# Lock\n")
    write_doc(lab, "data/memories/log.md", f"D-1: PI approved the lock {token('pi', LOCK, RUN, 'toolu_1')}\n")
    assert detector(lab, env=evidence.env()).returncode == 0


def test_a_non_approve_verdict_is_recordable(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "MODIFY")]))
    write_doc(lab, LOCK, token("pi", LOCK, RUN, "toolu_1", verdict="MODIFY") + "\n")
    assert detector(lab, "--program", "p", env=evidence.env()).returncode == 0


def test_window_is_configurable(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")], age_hours=2))
    write_doc(lab, LOCK, token("pi", LOCK, RUN, "toolu_1") + "\n")
    assert detector(lab, "--program", "p", "--window-hours", "3", env=evidence.env()).returncode == 0
    assert detector(lab, "--program", "p", "--window-hours", "1", env=evidence.env()).returncode == 1


def test_director_self_attestation_needs_a_real_run_and_no_receipt(lab, evidence):
    evidence.write_run(RUN, [])
    write_doc(lab, LOCK, f"[signoff role=director target={LOCK} verdict=APPROVE run={RUN} receipt=self]\n")
    assert detector(lab, "--program", "p", env=evidence.env()).returncode == 0
    write_doc(lab, LOCK, f"[signoff role=director target={LOCK} verdict=APPROVE run=nope receipt=self]\n")
    assert detector(lab, "--program", "p", env=evidence.env()).returncode == 1


def test_required_document_passes_with_both_verified_sign_offs(lab, evidence):
    memo = "programs/p/closure_memo.md"
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(memo, "APPROVE")]))
    write_doc(lab, memo, (f"# Closure\n{token('pi', memo, RUN, 'toolu_1')}\n"
                          f"[signoff role=director target={memo} verdict=APPROVE run={RUN} receipt=self]\n"))
    assert detector(lab, env=evidence.env()).returncode == 0


def test_required_document_needs_every_listed_role(lab, evidence):
    memo = "programs/p/closure_memo.md"
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(memo, "APPROVE")]))
    write_doc(lab, memo, f"# Closure\n{token('pi', memo, RUN, 'toolu_1')}\n")
    result = detector(lab, env=evidence.env())
    assert result.returncode == 1 and "from director" in result.stderr


def test_required_document_can_be_signed_across_sessions(lab, evidence):
    memo = "programs/p/closure_memo.md"
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(memo, "APPROVE")]))
    write_doc(lab, memo, f"# Closure\n{token('pi', memo, RUN, 'toolu_1')}\n")
    first = detector(lab, "--gate", "--run-id", RUN, "--nonce", "n", env=evidence.env())
    assert first.returncode == 1 and "from director" in first.stderr
    evidence.write_run("session_2_x", [])
    with open(lab / memo, "a", encoding="utf-8") as f:
        f.write(f"[signoff role=director target={memo} verdict=APPROVE run=session_2_x receipt=self]\n")
    second = detector(lab, "--gate", "--run-id", "session_2_x", "--nonce", "n", env=evidence.env())
    assert second.returncode == 0, second.stdout + second.stderr


def test_draft_marker_exempts_a_required_document(lab, evidence):
    write_doc(lab, "programs/p/closure_memo.md", "---\nstatus: draft\n---\n# Closure\n")
    assert detector(lab, env=evidence.env()).returncode == 3


def test_unauthenticated_mode_is_advisory_and_needs_no_key(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))
    write_doc(lab, LOCK, token("pi", LOCK, RUN, "toolu_1") + "\n")
    (evidence.directory / "receipt.key").unlink()
    result = detector(lab, "--program", "p", "--unauthenticated", env=evidence.env())
    assert result.returncode == 0 and "UNAUTHENTICATED" in result.stderr


def test_emit_signoff_prints_a_token_the_check_accepts(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))
    done = detector(lab, "--emit-signoff", "--role", "pi", "--target", LOCK, env=evidence.env())
    assert done.returncode == 0, done.stderr
    write_doc(lab, LOCK, done.stdout)
    assert detector(lab, "--program", "p", env=evidence.env()).returncode == 0


def test_emit_signoff_refuses_when_the_role_never_returned_a_line(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[]))
    done = detector(lab, "--emit-signoff", "--role", "pi", "--target", LOCK, env=evidence.env())
    assert done.returncode == 1 and done.stdout == ""


# --- prose about sign-offs is not a sign-off -------------------------------------------

@pytest.mark.parametrize("line", [
    "the proposal is awaiting PI approval",
    "BLOCKED pending PI co-sign",
    "before requesting PI sign-off",
    "PI co-sign requires BOTH content approval AND a clean build.",
    "PI did not approve this.",
    "Dispatch PI for approval.",
    "If PI approves, proceed.",
    "PI must sign off.",
    "PI approval gate",
    "PI approval is pending",
    "Director approved the PI's plan.",
    "PI signature block follows below",
    "The PI reviewed the data and asked for more runs.",
    "see pi_notes.md for what was approved",
    "Replace the text-based `**PI ✓**` convention with an HMAC.",
    "the token is `[signoff role=pi target=x verdict=APPROVE run=r receipt=t]`",
    "**Director ✓** 2026-11-04",
])
def test_prose_is_not_flagged(lab, evidence, line):
    write_doc(lab, "programs/p/notes.md", f"# Notes\n{line}\n")
    result = detector(lab, "--program", "p", env=evidence.env())
    assert result.returncode == 3, f"{line!r}: {result.stdout}{result.stderr}"


def test_fenced_block_is_quotation(lab, evidence):
    write_doc(lab, "programs/p/notes.md", "# Notes\n```\n**PI ✓**\n[signoff pi]\n```\nend\n")
    assert detector(lab, "--program", "p", env=evidence.env()).returncode == 3


def test_a_line_that_was_rejected_in_a_fence_still_counts_after_it(lab, evidence):
    write_doc(lab, "programs/p/notes.md", "```\nx\n```\nPI approved.\n")
    assert detector(lab, "--program", "p", env=evidence.env()).returncode == 1


# --- the baseline ----------------------------------------------------------------------

def test_baselined_legacy_line_is_not_reverified(lab, evidence):
    write_doc(lab, LOCK, FORGED_LOCK)
    evidence.accept(lines=[dl.line_key(LOCK, FORGED_LOCK)])
    result = detector(lab, "--program", "p", env=evidence.env())
    assert result.returncode == 3 and "1 line(s) accepted earlier" in result.stdout


def test_baselined_line_copied_to_another_document_is_new(lab, evidence):
    write_doc(lab, LOCK, FORGED_LOCK)
    write_doc(lab, "programs/p/forged.md", FORGED_LOCK)
    evidence.accept(lines=[dl.line_key(LOCK, FORGED_LOCK)])
    result = detector(lab, "--program", "p", env=evidence.env())
    assert result.returncode == 1 and "programs/p/forged.md" in result.stderr


def test_init_evidence_baselines_existing_lines_and_never_overwrites(lab, tmp_path):
    write_doc(lab, LOCK, FORGED_LOCK)
    env = {"PATH": "/usr/bin:/bin", "LAB_EVIDENCE_DIR": str(tmp_path / "fresh"), "HOME": str(tmp_path)}
    first = detector(lab, "--init-evidence", env=env)
    assert first.returncode == 0 and "not verified" in first.stdout
    assert stat.S_IMODE((tmp_path / "fresh" / "receipt.key").stat().st_mode) == 0o600
    key = (tmp_path / "fresh" / "receipt.key").read_bytes()
    assert detector(lab, "--init-evidence", env=env).returncode == 2
    assert (tmp_path / "fresh" / "receipt.key").read_bytes() == key
    assert detector(lab, "--program", "p", env=env).returncode == 3


def test_preflight_creates_evidence_only_when_there_is_nothing_to_launder(lab, tmp_path):
    env = {"PATH": "/usr/bin:/bin", "LAB_EVIDENCE_DIR": str(tmp_path / "fresh"), "HOME": str(tmp_path)}
    write_doc(lab, LOCK, FORGED_LOCK)
    refused = detector(lab, "--preflight", env=env)
    assert refused.returncode == 2 and "--init-evidence" in refused.stderr
    assert not (tmp_path / "fresh" / "receipt.key").exists()
    (lab / LOCK).unlink()
    assert detector(lab, "--preflight", env=env).returncode == 0
    assert (tmp_path / "fresh" / "receipt.key").exists()


def test_preflight_refuses_half_present_evidence(lab, evidence):
    (evidence.directory / "accepted.json").unlink()
    assert detector(lab, "--preflight", env=evidence.env()).returncode == 2


def test_rebaseline_needs_the_key(lab, evidence):
    write_doc(lab, LOCK, FORGED_LOCK)
    (evidence.directory / "receipt.key").unlink()
    assert detector(lab, "--rebaseline", env=evidence.env()).returncode == 2


def test_rebaseline_accepts_current_lines(lab, evidence):
    write_doc(lab, LOCK, FORGED_LOCK)
    assert detector(lab, "--rebaseline", env=evidence.env()).returncode == 0
    assert detector(lab, "--program", "p", env=evidence.env()).returncode == 3


# --- the gate command ------------------------------------------------------------------

def _gate(lab, evidence, run=RUN, nonce="abc123"):
    return detector(lab, "--gate", "--check-identity", "--run-id", run, "--nonce", nonce, env=evidence.env())


def test_gate_echoes_the_nonce_and_records_a_verdict(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))
    write_doc(lab, LOCK, token("pi", LOCK, RUN, "toolu_1") + "\n")
    result = _gate(lab, evidence)
    assert result.returncode == 0, result.stderr
    assert "GATE-VERDICT nonce=abc123 status=VERIFIED" in result.stdout
    verdict = json.loads((evidence.directory / "verdicts" / f"{RUN}.json").read_text())
    assert verdict["status"] == "VERIFIED" and verdict["signoffs"] == 1


def test_gate_accepts_a_verified_line_for_good(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))
    write_doc(lab, LOCK, token("pi", LOCK, RUN, "toolu_1") + "\n")
    assert _gate(lab, evidence).returncode == 0
    evidence.write_run("session_2_x", [])
    later = _gate(lab, evidence, run="session_2_x")
    assert later.returncode == 3 and "1 line(s) accepted earlier" in later.stdout


def test_gate_keeps_blocking_on_a_failed_line_but_keeps_the_good_ones(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi", attests=[(LOCK, "APPROVE")]))
    write_doc(lab, LOCK, token("pi", LOCK, RUN, "toolu_1") + "\n")
    write_doc(lab, "programs/p/forged.md", FORGED_LOCK)
    assert _gate(lab, evidence).returncode == 1
    evidence.write_run("session_2_x", [])
    later = _gate(lab, evidence, run="session_2_x")
    assert later.returncode == 1
    assert "1 line(s) accepted earlier" not in later.stdout  # not printed when blocking
    assert "programs/p/forged.md" in later.stderr and LOCK not in later.stderr


def test_gate_needs_a_sealed_ledger_for_its_own_run(lab, evidence):
    evidence.write_run(RUN, [], sealed=False)
    result = _gate(lab, evidence)
    assert result.returncode == 2 and "no seal" in result.stderr
    assert "GATE-VERDICT" not in result.stdout


def test_gate_needs_a_ledger_for_its_own_run(lab, evidence):
    result = _gate(lab, evidence)
    assert result.returncode == 2 and "recorder did not start" in result.stderr


def test_gate_notices_an_earlier_run_that_was_never_gated_and_says_so_once(lab, evidence):
    evidence.write_run("session_0_x", [])
    evidence.write_run(RUN, [])
    first = _gate(lab, evidence)
    assert "gate never reached a verdict" in first.stdout and "session_0_x" in first.stdout
    evidence.write_run("session_2_x", [])
    assert "session_0_x" not in _gate(lab, evidence, run="session_2_x").stdout


def test_gate_prints_when_a_run_had_no_sandbox(lab, evidence):
    evidence.write_run(RUN, [], sandbox="off")
    result = _gate(lab, evidence)
    assert "sandbox=off" in result.stdout and "not protected" in result.stdout


def test_identity_accepts_a_new_record_when_the_role_ran(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "pi"))
    write_doc(lab, "data/agents/pi/episodic/2026-11-04_x.md", "# real\n")
    result = _gate(lab, evidence)
    assert result.returncode == 0 and "1 new episodic record(s)" in result.stdout


def test_a_pass_with_no_sign_offs_never_prints_ok_zero(lab, evidence):
    # "OK: 0 signatures verified" is the string the audit found recorded as a pass.
    evidence.write_run(RUN, receipt("toolu_1", "pi"))
    write_doc(lab, "data/agents/pi/episodic/2026-11-04_x.md", "# real\n")
    assert "OK: 0" not in _gate(lab, evidence).stdout


def test_junk_fields_in_a_ledger_cannot_be_judged(lab, evidence):
    path = evidence.write_run(RUN, [])
    record = json.loads(path.read_text().splitlines()[0])
    record["ts"] = "not a number"
    path.write_text(json.dumps(record) + "\n")
    write_doc(lab, LOCK, "# x\n")
    result = detector(lab, "--program", "p", "--unauthenticated", env=evidence.env())
    assert result.returncode == 2 and "malformed record" in result.stderr


def test_identity_skips_director_records(lab, evidence):
    evidence.write_run(RUN, [])
    write_doc(lab, "data/agents/director/episodic/2026-11-04_x.md", "# self\n")
    assert _gate(lab, evidence).returncode == 3


def test_identity_without_any_run_cannot_be_judged(lab, evidence):
    write_doc(lab, "data/agents/pi/episodic/2026-11-04_x.md", "# x\n")
    assert detector(lab, "--check-identity", env=evidence.env()).returncode == 2


def test_generic_dispatch_naming_a_role_is_noticed(lab, evidence):
    evidence.write_run(RUN, receipt("toolu_1", "general-purpose", description="lock review (pi role)"))
    assert "names pi" in _gate(lab, evidence).stdout


# --- archive modes keep working, labelled --------------------------------------------

def test_legacy_episodic_mode_is_labelled_and_still_reads_signatures(lab, evidence):
    write_doc(lab, "data/agents/pi/episodic/2026-11-04_x.md", "# x\n")
    write_doc(lab, LOCK, FORGED_LOCK)
    done = detector(lab, "--legacy-episodic", "--program", "p", env=evidence.env())
    assert done.returncode == 0 and "LEGACY MODE" in done.stderr


def test_legacy_episodic_empty_scan_is_not_a_pass(lab, evidence):
    write_doc(lab, "programs/p/notes.md", "# none\n")
    assert detector(lab, "--legacy-episodic", "--program", "p", env=evidence.env()).returncode == 3


def test_legacy_logs_mode_is_reachable_for_the_archive(lab, evidence):
    assert detector(lab, "--check-identity", "--legacy-logs", env=evidence.env()).returncode == 2


# --- the real runner, end to end, with a stub harness ----------------------------------

def _honest_session(receipt_id="toolu_real_1", target=LOCK):
    return [
        {"write": [target, "# Lock\nThe locked items.\n"]},
        {"stream": ev_agent(receipt_id, "pi")},
        {"stream": ev_result(receipt_id, f"Reviewed.\nSIGNOFF target={target} verdict=APPROVE\n")},
        {"emit_signoff": {"role": "pi", "target": target, "into": target}},
        {"exit_reason": GRACEFUL},
    ]


def test_runner_honest_session_passes_the_gate(tmp_path, lab):
    log = run_runner(lab, [_honest_session()], tmp_path / "e")
    assert "result=verified" in log
    assert clean_exit_accepted(log)
    assert "[signoff role=pi" in (lab / LOCK).read_text()


def test_runner_session_after_an_honest_one_is_empty_not_verified(tmp_path, lab):
    run_runner(lab, [_honest_session(), [{"exit_reason": GRACEFUL}]], tmp_path / "e")
    logs = session_logs(lab)
    assert "result=verified" in logs and "result=empty" in logs


def test_runner_block_clears_once_the_session_fixes_it(tmp_path, lab):
    forged = [{"write": [LOCK, FORGED_LOCK]}, {"exit_reason": GRACEFUL}]
    log = run_runner(lab, [forged, _honest_session()], tmp_path / "e")
    assert session_logs(lab).count("result=blocked") == 1
    assert "result=verified" in log and clean_exit_accepted(log)


def test_runner_refuses_to_start_with_a_detector_that_cannot_run(tmp_path, lab):
    (lab / "tools" / "verify_signatures.py").write_text("def broken(:\n")
    log = run_runner(lab, [[{"exit_reason": GRACEFUL}]], tmp_path / "e")
    assert "not starting" in log and stub_calls(lab) == []


def test_runner_detector_that_breaks_during_a_session_blocks(tmp_path, lab):
    log = run_runner(lab, [[{"write": ["tools/verify_signatures.py", "def broken(:\n"]},
                            {"exit_reason": GRACEFUL}]], tmp_path / "e")
    assert "trust_base_changed" in log and not clean_exit_accepted(log)


def test_runner_refuses_to_start_on_broken_evidence(tmp_path, lab):
    evidence = Evidence(tmp_path / "e")
    (evidence.directory / "accepted.json").unlink()
    log = run_runner(lab, [[{"exit_reason": GRACEFUL}]], tmp_path / "e")
    assert "not starting" in log and stub_calls(lab) == []


def test_runner_refuses_an_unknown_sandbox_mode(tmp_path, lab):
    log = run_runner(lab, [[{"exit_reason": GRACEFUL}]], tmp_path / "e", extra_env={"LAB_SANDBOX": "maybe"})
    assert "LAB_SANDBOX must be on or off" in log and stub_calls(lab) == []


@pytest.mark.parametrize("module", ["hmac", "json", "hashlib"])
def test_runner_planted_stdlib_shadow_does_not_reach_the_gate(tmp_path, lab, module):
    # Without -I, python3 puts tools/ first on sys.path and this file would run in the gate.
    planted = "import sys\nprint('PLANTED RAN')\nsys.exit(0)\n"
    log = run_runner(lab, [[{"write": [f"tools/{module}.py", planted]},
                            {"write": [LOCK, FORGED_LOCK]}, {"exit_reason": GRACEFUL}]], tmp_path / "e")
    assert "PLANTED RAN" not in session_logs(lab)
    assert "result=blocked" in log


def test_runner_passes_settings_to_the_director_and_hides_the_run_id(tmp_path, lab):
    run_runner(lab, [[{"exit_reason": GRACEFUL}]], tmp_path / "e")
    call = stub_calls(lab)[0]
    assert call["has_settings"] and call["run_id_visible"] is None
    settings = json.loads(call["settings"])
    assert settings["sandbox"]["enabled"] is True
    assert settings["sandbox"]["allowUnsandboxedCommands"] is False


def test_runner_sandbox_off_is_recorded_and_announced(tmp_path, lab):
    log = run_runner(lab, [[{"exit_reason": GRACEFUL}]], tmp_path / "e", extra_env={"LAB_SANDBOX": "off"})
    assert "WARNING: LAB_SANDBOX=off" in log and "sandbox=off" in log
    assert "sandbox" not in json.loads(stub_calls(lab)[0]["settings"])


def test_consolidator_runs_under_the_same_settings():
    runner = (REPO / "run_agi_lab.sh").read_text()
    calls = re.findall(r"claude[^\n]*\\\n(?:[^\n]*\\\n)*?\s*--dangerously-skip-permissions[^\n]*\\\n([^\n]*)", runner)
    assert calls, "expected claude invocations"
    assert all("--settings" in c for c in calls), calls


# --- the shipped settings --------------------------------------------------------------

@pytest.fixture
def settings(tmp_path):
    return dl.render_settings(REPO, tmp_path / "ev", sandbox=True)


def test_settings_turn_the_escape_hatches_off(settings):
    box = settings["sandbox"]
    assert box["enabled"] is True and box["failIfUnavailable"] is True
    assert box["allowUnsandboxedCommands"] is False
    assert "excludedCommands" not in box


def test_settings_deny_the_key_to_file_tools_and_to_shell_commands(settings, tmp_path):
    key = str(tmp_path / "ev" / "receipt.key")
    assert f"Read(/{key})" in settings["permissions"]["deny"]
    assert key in settings["sandbox"]["filesystem"]["denyRead"]


def test_settings_keep_the_evidence_out_of_write_scope(settings, tmp_path):
    ev = str(tmp_path / "ev")
    assert f"Edit(/{ev}/**)" in settings["permissions"]["deny"]
    assert ev in settings["sandbox"]["filesystem"]["denyWrite"]


def test_settings_are_rendered_with_absolute_paths_and_no_placeholders(settings):
    text = json.dumps(settings)
    assert "{{" not in text and "_comment" not in settings
    for entry in settings["sandbox"]["filesystem"]["denyWrite"] + settings["sandbox"]["filesystem"]["denyRead"]:
        assert entry.startswith("/"), entry


def test_rendering_with_the_sandbox_off_keeps_the_deny_rules(tmp_path):
    settings = dl.render_settings(REPO, tmp_path / "ev", sandbox=False)
    assert "sandbox" not in settings and settings["permissions"]["deny"]


def _runner_executed_paths() -> set:
    runner = (REPO / "run_agi_lab.sh").read_text()
    code = "\n".join(line for line in runner.splitlines() if not line.lstrip().startswith("#"))
    found = set(re.findall(r"(?:python3?|\.venv/bin/python)(?:\s+-[Iu])*\s+(tools/[\w./-]+\.py)", code))
    found |= {m.replace(".", "/") + ".py" for m in re.findall(r"-m\s+(tools[\w.]+)", code)}
    found |= {m.replace(".", "/") + ".py" for m in re.findall(r"from\s+(tools\.[\w.]+)\s+import", code)}
    assert "tools/post_director.py" in found and "tools/work_queue.py" in found, found
    return found


def test_everything_the_runner_executes_is_frozen(settings):
    # The runner is outside the sandbox. A file it runs that an agent could edit would be a
    # way out, so this fails if a new runner line starts executing something unfrozen.
    for rel in _runner_executed_paths() | {".venv/bin/python", "run_agi_lab.sh"}:
        assert dl.covered_by_deny_write(settings, str(REPO / rel)), rel


def test_a_new_top_level_module_cannot_be_planted_where_python_would_find_it(settings):
    for rel in ("tools/json.py", "json.py", "tools/retrieval/json.py", "tools/__pycache__/x.pyc", ".venv/lib/x.py"):
        assert dl.covered_by_deny_write(settings, str(REPO / rel)), rel


def test_agent_output_paths_stay_writable(settings):
    for rel in ("programs/p/x.md", "data/memories/log.md", "data/infra/x.log", "tools/lab/x.py", "src/a.c"):
        assert not dl.covered_by_deny_write(settings, str(REPO / rel)), rel


def test_every_pinned_file_is_also_frozen(settings):
    runner = (REPO / "run_agi_lab.sh").read_text()
    block = re.search(r"TRUST_BASE_FILES=\((.*?)\)", runner, re.DOTALL).group(1)
    pinned = block.split()
    assert "tools/verify_signatures.py" in pinned and "tools/dispatch_ledger.py" in pinned
    for rel in pinned:
        assert (REPO / rel).is_file(), rel
        assert dl.covered_by_deny_write(settings, str(REPO / rel)), rel


def test_gate_and_recorder_are_started_isolated():
    runner = (REPO / "run_agi_lab.sh").read_text()
    assert "python3 -I tools/verify_signatures.py --gate" in runner
    assert "python3 -I -u tools/stream_formatter.py" in runner
    assert "2>&1 | python3" not in runner, "stderr must not reach the recorder"


def test_detector_gate_path_uses_only_the_standard_library():
    for rel in ("tools/verify_signatures.py", "tools/dispatch_ledger.py"):
        tree = (REPO / rel).read_text()
        imported = set(re.findall(r"^\s*(?:from|import)\s+([A-Za-z_]+)", tree, re.MULTILINE))
        assert imported <= set(sys.stdlib_module_names) | {"tools", "dispatch_ledger"}, imported


# --- the next Director is told -------------------------------------------------------

def test_correction_prompt_for_a_blocked_gate_is_accurate():
    from tools.brief_assembler import _build_correction_prompt
    text = _build_correction_prompt("signoff_gate_blocked")
    assert "signoff_gate_last.txt" in text and "already applied" in text
    assert "signoff_contract.md" in text


def test_contract_documents_the_formats_the_code_accepts():
    contract = (REPO / "data/agents/_shared/signoff_contract.md").read_text()
    token_line = re.search(r"\[signoff role=[^\]]*\]", contract).group(0)
    assert vs._TOKEN_STRICT.fullmatch(token_line), token_line
    attest = re.search(r"^SIGNOFF target=\S+ verdict=[A-Z]+$", contract, re.MULTILINE).group(0)
    assert dl.ATTEST_LINE.search(attest), attest
