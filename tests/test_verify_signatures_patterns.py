"""Regression tests for the signature patterns in tools/verify_signatures.py.

The D-109 lock document signed itself with `**PI signature:** ✓ (2026-04-17)`, a format
the original four patterns did not match (found 2026-09-17 while building the replay
harness for the incident). These tests pin every format the detector is expected to
catch, and the prose it must not catch.
"""
from __future__ import annotations

from tools import verify_signatures as vs


def _matches(line: str) -> list[str]:
    roles = []
    for pat in vs.SIG_PATTERNS:
        for m in pat.finditer(line):
            roles.append(m.group(1).lower())
    return roles


def test_d109_lock_signature_line_is_detected():
    assert "pi" in _matches("**PI signature:** ✓ (2026-04-17)")


def test_unbolded_signature_line_is_detected():
    assert "pi" in _matches("PI signature: ✓")


def test_canonical_formats_still_detected():
    assert "pi" in _matches("**PI ✓**")
    assert "pi" in _matches("- PI ✓ [episodic: data/agents/pi/episodic/x.md]")
    assert "pi" in _matches("| PI ✓ |")
    assert "director" in _matches("Signed: director")


def test_narrative_prose_is_not_a_signature():
    assert _matches("the proposal approved by pi was filed") == []
    assert _matches("PI signature block follows below") == []
    assert _matches("I (PI) agree that the five locked items are correct.") == []


# --- Citation resolution, quotation guard, retrospective window ----------------------
# All three behaviours were added 2026-09-17 after running the shipped detector over the
# live lab archive: `--full` reported 23 unverified signatures and every one was a false
# positive (8 from citation parsing and quoted prose, 15 from the wall-clock window).

import datetime as dt


def test_citation_short_form_resolves():
    assert "data/agents/pi/episodic/2026-04-20_x.md" in vs.normalize_episodic_ref("pi/2026-04-20_x", "pi")


def test_citation_backticks_are_stripped():
    cands = vs.normalize_episodic_ref("`data/agents/pi/episodic/2026-04-24_x.md`", "pi")
    assert cands[0] == "data/agents/pi/episodic/2026-04-24_x.md"


def test_citation_trailing_prose_inside_brackets_is_dropped():
    cands = vs.normalize_episodic_ref(
        "`data/agents/lab_architect/episodic/2026-04-24_x.md` (written at invocation close)",
        "lab_architect")
    assert cands[0] == "data/agents/lab_architect/episodic/2026-04-24_x.md"


def test_citation_missing_extension_is_tried():
    assert any(c.endswith(".md") for c in vs.normalize_episodic_ref("pi/2026-04-20_x", "pi"))


def test_quoted_convention_is_not_an_attestation():
    # The lab's own paper about signature forgery quotes the pattern in backticks.
    line = "Replace the text-based `**PI ✓**` convention with an HMAC."
    assert vs._in_code_span(line, line.index("**PI"))


def test_real_attestation_is_not_treated_as_quoted():
    line = "- **PI ✓** (2026-04-17)"
    assert not vs._in_code_span(line, line.index("**PI"))


def test_signature_date_is_read_from_the_line():
    assert vs.signature_date("> **pi ✓** 2026-04-24 D-181 APPROVE") == dt.date(2026, 4, 24)
    assert vs.signature_date("> **pi ✓** APPROVE") is None


def test_episodic_date_prefers_filename_over_mtime(tmp_path):
    f = tmp_path / "2026-04-20_methodology_paper_cosign.md"
    f.write_text("x")
    assert vs.episodic_date(f) == dt.date(2026, 4, 20)
