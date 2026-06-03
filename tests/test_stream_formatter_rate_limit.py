"""Gate-file behavior of tools/stream_formatter.py on rate_limit_event messages.

The runner's pre-spawn gate sleeps whenever data/infra/rate_limit_resets_at holds
a future timestamp. Only an *actual* rejection (status == "rejected") should write
that file. An "allowed_warning" means the request went through — it must NOT gate,
or the lab sleeps for hours despite never being blocked.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "tools" / "stream_formatter.py"
GATE_REL = Path("data/infra/rate_limit_resets_at")
FUTURE = 9999999999  # year 2286; safely in the future


def _run(stdin_lines, cwd):
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        input="".join(line + "\n" for line in stdin_lines),
        capture_output=True,
        text=True,
        cwd=str(cwd),
        timeout=30,
    )


def _event(status, resets_at=FUTURE):
    return json.dumps(
        {"type": "rate_limit_event", "rate_limit_info": {"status": status, "resetsAt": resets_at}}
    )


def test_allowed_warning_does_not_write_gate(tmp_path):
    _run([_event("allowed_warning")], tmp_path)
    gate = tmp_path / GATE_REL
    assert not gate.exists(), "allowed_warning must NOT write the rate-limit gate file"


def test_rejected_writes_gate(tmp_path):
    _run([_event("rejected")], tmp_path)
    gate = tmp_path / GATE_REL
    assert gate.exists(), "rejected must write the rate-limit gate file"
    assert gate.read_text().strip() == str(FUTURE)


def test_warning_then_rejection_writes_rejection(tmp_path):
    # A warning early in a session must not gate; a later rejection must.
    _run([_event("allowed_warning", FUTURE - 100), _event("rejected", FUTURE)], tmp_path)
    gate = tmp_path / GATE_REL
    assert gate.exists(), "rejection after a warning must write the gate file"
    assert gate.read_text().strip() == str(FUTURE)
