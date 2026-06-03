"""Surface unresolved calibration claims so the PI/Director resolves them.

Companion to tools/calibration_logger.py. log_calibration() appends a claim
with outcome=None; score_calibration(decision_id, outcome) marks the outcome
later. Nothing in the lab loop currently calls score_calibration, so claims
pile up unresolved forever and the calibration learning loop never closes.

This module is the SURFACING NUDGE: it reads the telemetry, finds the claims
that still have outcome=None, and renders a short markdown brief reminding the
PI/Director to resolve each one (with evidence-based judgment, via
score_calibration) when its outcome lands. It is strictly read-only on the
jsonl — it never writes, guesses, or fabricates any outcome value.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parent.parent

_TELEMETRY_REL = "data/infra/calibration_telemetry.jsonl"


def _telemetry_path(repo_root: Optional[Path] = None) -> Path:
    root = Path(repo_root) if repo_root is not None else _REPO
    return root / _TELEMETRY_REL


def _compute_age_days(ts: object) -> Optional[float]:
    """Days between the record ts and now (UTC). None if ts is unparseable."""
    if not isinstance(ts, str):
        return None
    raw = ts.strip()
    if not raw:
        return None
    # Normalize a trailing 'Z' to an explicit UTC offset for fromisoformat.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None
    # Treat naive timestamps as UTC so the subtraction is well-defined.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    now = datetime.datetime.now(datetime.timezone.utc)
    delta = now - parsed
    return delta.total_seconds() / 86400.0


def list_pending_claims(repo_root: Optional[Path] = None) -> list[dict]:
    """Return calibration claims whose outcome is still None, oldest-first.

    Reads data/infra/calibration_telemetry.jsonl (read-only). Each returned
    dict has: decision_id, role, gate_type, confidence, claim, ts, age_days.
    age_days is the number of days between ts and now (None if ts unparseable).
    Missing file -> []. Blank / malformed lines are skipped.
    """
    path = _telemetry_path(repo_root)
    if not path.exists():
        return []

    pending: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        if rec.get("outcome") is not None:
            continue
        ts = rec.get("ts")
        pending.append(
            {
                "decision_id": rec.get("decision_id"),
                "role": rec.get("role"),
                "gate_type": rec.get("gate_type"),
                "confidence": rec.get("confidence"),
                "claim": rec.get("claim"),
                "ts": ts,
                "age_days": _compute_age_days(ts),
            }
        )

    # Sort oldest-first by age (largest age_days first). Records with an
    # unparseable ts (age_days=None) sort last to keep ordering deterministic.
    pending.sort(key=lambda c: (c["age_days"] is None, -(c["age_days"] or 0.0)))
    return pending


def _truncate(text: object, limit: int = 100) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\n", " ").strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1].rstrip() + "…"


def _fmt_age(age_days: Optional[float]) -> str:
    if age_days is None:
        return "?"
    return f"{age_days:.0f}"


def format_pending_brief(claims: list[dict]) -> str:
    """Render a short markdown block for the pending calibration claims.

    Empty list -> '' (caller should not emit an empty section).
    """
    if not claims:
        return ""
    n = len(claims)
    lines = [
        f"### Pending calibration claims ({n}) — resolve with "
        f"score_calibration() when the outcome lands",
    ]
    for c in claims:
        lines.append(
            f"- {c.get('decision_id')} [{c.get('gate_type')}] "
            f"conf={c.get('confidence')} age={_fmt_age(c.get('age_days'))}d "
            f"— {_truncate(c.get('claim'))}"
        )
    return "\n".join(lines)


def main() -> int:
    print(format_pending_brief(list_pending_claims()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
