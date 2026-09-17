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
