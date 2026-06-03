"""Calibration telemetry — logs PI's confidence claims for organic learning.

Every PI gate verdict (program selection, phase close, paper approval, etc.)
attaches a probability ("I'm 70% confident this is the right target"). We
log (decision_id, claim, confidence, reasoning) here. When outcomes are
known later (program closed successfully or not, paper accepted/rejected,
sub-step held or broke), we score the calibration via tools/calibration_rollup.py.

Over many gate verdicts, this loop tells PI:
- "you said 70% on these 10 things, 5 hit = 50% actual" (overconfident at 70%)
- "you said 30% on these 8 things, 6 hit = 75% actual" (underconfident at 30%)

PI uses that to recalibrate without external prescription. This is the highest-
leverage organic-learning mechanism in the literature (per the spec at
docs/superpowers/specs/2026-05-04-pi-self-correction.md).
"""
from __future__ import annotations
import datetime
import json
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent


def log_calibration(
    decision_id: str,
    claim: str,
    confidence: float,
    reasoning: str,
    role: str = "pi",
    gate_type: str = "",
) -> None:
    """Append a single calibration record to data/infra/calibration_telemetry.jsonl.

    Called by PI (or any judgment-tier role) at every gate verdict, before
    the verdict is finalized. The verifier (lab_architect) checks that
    confidence is present + numerical + has accompanying reasoning before
    accepting the verdict.

    Args:
        decision_id: D-NNN identifier for the gate verdict (links to log.md)
        claim: short text of what's being asserted (e.g., "this paper is
               publishable at ICLR workshop")
        confidence: probability in [0, 1] that the claim is correct
        reasoning: why this confidence and not higher/lower (1-3 sentences)
        role: typically "pi"; can be other judgment-tier roles
        gate_type: program_selection | phase_close | paper_approval |
                   sub_step | mechanism | other
    """
    if not (0 <= confidence <= 1):
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "decision_id": decision_id,
        "role": role,
        "gate_type": gate_type,
        "claim": claim,
        "confidence": confidence,
        "reasoning": reasoning,
        "outcome": None,        # filled in later by score_calibration()
        "outcome_known_at": None,
    }
    log_path = _REPO / "data/infra/calibration_telemetry.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def score_calibration(decision_id: str, outcome: bool) -> bool:
    """Mark the outcome of a previously-logged calibration claim.

    Called when the outcome of a claim becomes known (e.g., paper got
    accepted, program closed successfully, sub-step held). Updates the
    matching record in-place by rewriting the jsonl. Idempotent: if the
    record already has an outcome, no-op.

    Returns True if a matching record was found and updated, False otherwise.
    """
    log_path = _REPO / "data/infra/calibration_telemetry.jsonl"
    if not log_path.exists():
        return False
    lines = log_path.read_text().splitlines()
    found = False
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        if rec.get("decision_id") == decision_id and rec.get("outcome") is None:
            rec["outcome"] = bool(outcome)
            rec["outcome_known_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(timespec="seconds")
            new_lines.append(json.dumps(rec))
            found = True
        else:
            new_lines.append(line)
    if found:
        log_path.write_text("\n".join(new_lines) + "\n")
    return found
