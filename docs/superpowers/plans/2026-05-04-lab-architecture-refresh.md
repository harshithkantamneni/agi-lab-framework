# Lab Architecture Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement heterogeneous model tiering + deterministic context assembler + async consolidator + verifier loops + telemetry, in 7 staged commits.

**Architecture:** See `docs/superpowers/specs/2026-05-04-lab-architecture-refresh.md`. TL;DR: replace synchronous LLM curator with deterministic Python brief assembler; tier 30 agents across haiku/sonnet/opus with self-escalation contracts + verifier loops; track escalation/rejection rates for organic tier adjustment.

**Tech Stack:** Python 3.14, pytest, bash (run_agi_lab.sh), JSON (agents.json), Markdown (procedurals).

**Stage gates:** User review after Stage 1, Stage 3, Stage 5. Stages 2, 4, 6, 7 proceed automatically once their predecessor passes.

---

## Stage 1 — `brief_assembler.py` (purely additive, safe with running Director)

### Task 1: Test scaffolding for brief_assembler

**Files:**
- Create: `tests/test_brief_assembler.py`
- Create: `tools/brief_assembler.py` (empty stub)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_brief_assembler.py
import json
import os
import tempfile
from pathlib import Path
import pytest

from tools.brief_assembler import assemble_brief, classify_session


def test_assemble_brief_returns_markdown_with_required_sections(tmp_path):
    # Set up a minimal fake lab state in tmp_path
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text(
        "D-100: minimal entry / STATUS=success / SUMMARY=test\n"
    )
    (tmp_path / "data/memories/current.md").write_text("# Current\nactive_program: test_prog\n")
    (tmp_path / "data/checkpoints/phase3_factorial").mkdir(parents=True)
    (tmp_path / "data/checkpoints/phase3_factorial/run_index.json").write_text(
        '{"_meta": {}, "A42": {"state": "completed"}}'
    )

    brief = assemble_brief(repo_root=tmp_path)

    assert brief.startswith("---\ngenerated_at:")
    assert "session_type:" in brief
    assert "# Active program" in brief
    assert "# Last 5 decisions" in brief
    assert "# Active carry-forwards" in brief
    assert "# Decision-critical files" in brief
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd <repo>
source .venv/bin/activate
pytest tests/test_brief_assembler.py -v
```

Expected: FAIL with `ImportError: cannot import name 'assemble_brief' from 'tools.brief_assembler'` (or AttributeError).

- [ ] **Step 3: Write minimal implementation passing structure-only check**

```python
# tools/brief_assembler.py
"""Deterministic context brief assembler. Reads recent lab state, writes
context_brief.md with curated content for the next Director session.

No LLM in the critical path. Runs in <100ms. Output: 10-20 KB.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Literal

SessionType = Literal["routine-monitor", "phase-transition", "user-action", "post-failure"]


def classify_session(repo_root: Path) -> SessionType:
    """Classify the upcoming session based on file mtimes and content."""
    return "routine-monitor"  # placeholder, refined in Task 2


def assemble_brief(repo_root: Path) -> str:
    """Assemble a context brief markdown for the next Director session."""
    session_type = classify_session(repo_root)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sections = [
        f"---\ngenerated_at: {now}\nsession_type: {session_type}\nsize_kb: 0\n---\n",
        "# Active program\n(placeholder — Task 3)\n",
        "# State delta since last Director session\n(placeholder — Task 4)\n",
        "# Last 5 decisions\n(placeholder — Task 5)\n",
        "# Active carry-forwards\n(placeholder — Task 6)\n",
        "# What this session likely needs to do\n(placeholder — Task 7)\n",
        "# Decision-critical files\n(placeholder — Task 7)\n",
        "# Wiki tier files NOT loaded\n(placeholder — Task 7)\n",
    ]
    return "\n".join(sections)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_brief_assembler.py::test_assemble_brief_returns_markdown_with_required_sections -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_brief_assembler.py tools/brief_assembler.py
git commit -m "feat(brief-assembler): scaffold + structural test"
```

### Task 2: Session classification

**Files:**
- Modify: `tests/test_brief_assembler.py`
- Modify: `tools/brief_assembler.py`

- [ ] **Step 1: Write failing tests for each session_type branch**

```python
def test_classify_phase_transition_when_orchestrator_dead_and_run_index_newer(tmp_path):
    # log.md older than run_index.json AND no orchestrator process
    log_path = tmp_path / "data/memories/log.md"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("D-100: old\n")
    run_idx = tmp_path / "data/checkpoints/phase3_factorial/run_index.json"
    run_idx.parent.mkdir(parents=True)
    run_idx.write_text("{}")
    # Make run_idx newer than log.md
    import os, time
    old = time.time() - 86400
    os.utime(log_path, (old, old))
    
    assert classify_session(tmp_path, _orchestrator_alive=False) == "phase-transition"


def test_classify_user_action_when_user_notes_recent(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("x\n")
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data/user_notes.md").write_text("DIRECTIVE\n")
    # user_notes is fresh by default
    
    assert classify_session(tmp_path, _orchestrator_alive=True) == "user-action"


def test_classify_post_failure_when_session_exit_says_so(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("x\n")
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data/session_exit.md").write_text("reason: RATE_LIMIT\n")

    assert classify_session(tmp_path, _orchestrator_alive=True) == "post-failure"


def test_classify_routine_monitor_default(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("x\n")

    assert classify_session(tmp_path, _orchestrator_alive=True) == "routine-monitor"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_brief_assembler.py -v -k classify
```

Expected: 4 FAILs — current `classify_session` always returns "routine-monitor".

- [ ] **Step 3: Implement classification logic**

```python
# In tools/brief_assembler.py — replace classify_session

import os
import subprocess


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _orchestrator_pid_alive() -> bool:
    """True iff phase3 orchestrator or scale_experiment process is running."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "tools/run_phase3_factorial\\.py|build/scale_experiment"],
            capture_output=True,
            text=True,
        )
        return out.returncode == 0
    except FileNotFoundError:
        return False


def classify_session(
    repo_root: Path, _orchestrator_alive: bool | None = None
) -> SessionType:
    """Classify the upcoming session by inspecting state files."""
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
        import time as _t
        if _t.time() - user_notes_mt < 30 * 60:
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_brief_assembler.py -v -k classify
```

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_brief_assembler.py tools/brief_assembler.py
git commit -m "feat(brief-assembler): session classification"
```

### Task 3: Active program block

**Files:**
- Modify: `tests/test_brief_assembler.py`
- Modify: `tools/brief_assembler.py`

- [ ] **Step 1: Write failing test**

```python
def test_active_program_block_extracted_from_current_md(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("x\n")
    (tmp_path / "data/memories/current.md").write_text(
        "# Current\n"
        "## §Current Program\n"
        "active_program: program_2_dense_vs_moe_sub100m\n"
        "phase: 3\n"
        "status_line: B42 step 400/5000 plateau-center\n"
        "## Some other section\n"
        "irrelevant content\n"
    )
    brief = assemble_brief(tmp_path)
    assert "program_2_dense_vs_moe_sub100m" in brief
    assert "phase: 3" in brief
    assert "B42 step 400/5000 plateau-center" in brief
```

- [ ] **Step 2: Run test, verify FAIL**

```bash
pytest tests/test_brief_assembler.py::test_active_program_block_extracted_from_current_md -v
```

- [ ] **Step 3: Implement extraction**

```python
# In tools/brief_assembler.py

def _extract_current_program_block(repo_root: Path) -> str:
    """Read current.md and extract the §Current Program section."""
    path = repo_root / "data/memories/current.md"
    if not path.exists():
        return "(no current.md found)"
    text = path.read_text()
    # Match a heading line containing "Current Program" (any case, any heading depth)
    # and capture content until the next heading at same or shallower depth.
    m = re.search(
        r"(^#{1,6}\s*[^\n]*Current Program[^\n]*\n)(.*?)(?=^#{1,6}\s|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    if m:
        return (m.group(1) + m.group(2)).strip()
    return "(§Current Program block not found in current.md)"
```

Then update `assemble_brief` to substitute the placeholder:
```python
sections = [
    # ... unchanged front-matter ...
    f"# Active program\n{_extract_current_program_block(repo_root)}\n",
    # ... rest ...
]
```

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(brief-assembler): extract active program block from current.md"
```

### Task 4: State delta + last decisions

**Files:**
- Modify: `tests/test_brief_assembler.py`
- Modify: `tools/brief_assembler.py`

- [ ] **Step 1: Write failing tests**

```python
def test_last_5_decisions_extracted_from_log_md(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text(
        "D-307: latest / STATUS=success / KEY_FINDING=X / SUMMARY=lorem\n"
        "\n"
        "D-306: prior / STATUS=success / KEY_FINDING=Y / SUMMARY=ipsum\n"
        "\n"
        "D-305: more / STATUS=success / KEY_FINDING=Z / SUMMARY=dolor\n"
        "\n"
        "D-304: even / STATUS=success / KEY_FINDING=W / SUMMARY=sit\n"
        "\n"
        "D-303: oldest in window / STATUS=success / KEY_FINDING=V / SUMMARY=amet\n"
        "\n"
        "D-302: should NOT appear / SUMMARY=outside-window\n"
    )
    brief = assemble_brief(tmp_path)
    for did in ("D-307", "D-306", "D-305", "D-304", "D-303"):
        assert did in brief
    assert "D-302" not in brief

def test_state_delta_lists_run_index_changes(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("D-100: x\n")
    (tmp_path / "data/checkpoints/phase3_factorial").mkdir(parents=True)
    (tmp_path / "data/checkpoints/phase3_factorial/run_index.json").write_text(
        json.dumps({
            "_meta": {},
            "A42": {"state": "completed", "elapsed_hours": 12.5},
            "B42": {"state": "in_progress", "run_index": 3},
        })
    )
    brief = assemble_brief(tmp_path)
    assert "A42" in brief
    assert "B42" in brief
    assert "completed" in brief
    assert "in_progress" in brief
```

- [ ] **Step 2: Run tests, verify FAIL**

- [ ] **Step 3: Implement extractors**

```python
# In tools/brief_assembler.py

def _extract_last_n_decisions(repo_root: Path, n: int = 5) -> str:
    """Pull last N D-NNN entries (assumes log.md prepended, so head = newest)."""
    path = repo_root / "data/memories/log.md"
    if not path.exists():
        return "(no log.md)"
    text = path.read_text()
    # Each decision starts with "D-NNN" at line start
    entries = re.findall(r"^(D-\d+:[^\n]*(?:\n[^D\n][^\n]*)*)", text, flags=re.MULTILINE)
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
```

Update `assemble_brief` to substitute:
```python
f"# State delta since last Director session\n{_extract_state_delta(repo_root)}\n",
f"# Last 5 decisions\n{_extract_last_n_decisions(repo_root, n=5)}\n",
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(brief-assembler): state delta + last 5 decisions"
```

### Task 5: Active carry-forwards + heuristic next-action

**Files:**
- Modify: `tests/test_brief_assembler.py`
- Modify: `tools/brief_assembler.py`

- [ ] **Step 1: Write failing test**

```python
def test_carry_forwards_extracted_from_log_md(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text(
        "D-200: P-FOO-BAR carry-forward opened — needs follow-up\n"
        "D-199: P-BAZ-QUX queued for next phase\n"
        "D-150: P-OLD resolved at D-180\n"
    )
    brief = assemble_brief(tmp_path)
    assert "P-FOO-BAR" in brief
    assert "P-BAZ-QUX" in brief

def test_routine_monitor_session_includes_brief_action_hint(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("D-1: x\n")
    brief = assemble_brief(tmp_path)
    assert "monitor" in brief.lower() or "tick" in brief.lower()
```

- [ ] **Step 2: Run tests, verify FAIL**

- [ ] **Step 3: Implement**

```python
# In tools/brief_assembler.py

def _extract_carry_forwards(repo_root: Path) -> str:
    """Find P-* tokens in recent log entries."""
    path = repo_root / "data/memories/log.md"
    if not path.exists():
        return "(none)"
    text = path.read_text()
    # Limit search to first ~10 KB (most recent decisions)
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
```

Update `assemble_brief`:
```python
f"# Active carry-forwards\n{_extract_carry_forwards(repo_root)}\n",
f"# What this session likely needs to do\n{_action_hint(session_type)}\n",
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(brief-assembler): carry-forwards + per-session-type action hint"
```

### Task 6: Decision-critical file pointers + size guard

**Files:**
- Modify: `tests/test_brief_assembler.py`
- Modify: `tools/brief_assembler.py`

- [ ] **Step 1: Write failing tests**

```python
def test_decision_critical_files_for_phase_transition(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("D-1: x\n")
    (tmp_path / "data/memories/current.md").write_text(
        "active_program: program_2_dense_vs_moe_sub100m\n"
    )
    (tmp_path / "programs/program_2_dense_vs_moe_sub100m").mkdir(parents=True)
    (tmp_path / "programs/program_2_dense_vs_moe_sub100m/phase3_close_carry_forwards.md").write_text("x")
    (tmp_path / "programs/program_2_dense_vs_moe_sub100m/spec.md").write_text("x")
    # Force phase-transition
    (tmp_path / "data/checkpoints/phase3_factorial").mkdir(parents=True)
    idx = tmp_path / "data/checkpoints/phase3_factorial/run_index.json"
    idx.write_text("{}")
    log = tmp_path / "data/memories/log.md"
    import os, time
    old = time.time() - 86400
    os.utime(log, (old, old))

    brief = assemble_brief(tmp_path, _orchestrator_alive=False)
    assert "phase3_close_carry_forwards.md" in brief
    assert "spec.md" in brief

def test_brief_size_under_30kb(tmp_path):
    (tmp_path / "data/memories").mkdir(parents=True)
    # Big log to test size cap
    big_log = "\n".join(f"D-{i}: lorem ipsum dolor " * 20 for i in range(500))
    (tmp_path / "data/memories/log.md").write_text(big_log)
    brief = assemble_brief(tmp_path)
    assert len(brief.encode("utf-8")) < 30 * 1024
```

- [ ] **Step 2: Run tests, verify FAIL**

- [ ] **Step 3: Implement**

```python
# In tools/brief_assembler.py

def _extract_active_program_files(repo_root: Path) -> list[str]:
    """List markdown files in the active program directory."""
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
    wiki_root = repo_root / "data/memories"
    if not wiki_root.exists():
        return "(none)"
    files = []
    for p in sorted(wiki_root.rglob("*.md")):
        rel = str(p.relative_to(repo_root))
        if rel in ("data/memories/log.md", "data/memories/current.md", "data/memories/session_brief.md", "data/memories/INDEX.md"):
            continue
        files.append(rel)
    return "\n".join(f"- {f}" for f in files) if files else "(none)"
```

Then in `assemble_brief`, replace the corresponding placeholders:
```python
sections = [
    # ... all the prior sections ...
    f"# Decision-critical files\n{_decision_critical_section(repo_root, session_type)}\n",
    f"# Wiki tier files NOT loaded\n{_wiki_not_loaded(repo_root)}\n",
]
brief = "\n".join(sections)

# Size cap: hard truncate at 30 KB with breadcrumb
if len(brief.encode("utf-8")) > 30 * 1024:
    brief = brief.encode("utf-8")[: 30 * 1024 - 200].decode("utf-8", errors="ignore")
    brief += "\n\n# (BRIEF TRUNCATED at 30 KB — see source files for full detail)\n"
return brief
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(brief-assembler): decision-critical pointers + 30KB size cap"
```

### Task 7: CLI + integration with runner

**Files:**
- Modify: `tools/brief_assembler.py` (add CLI)
- Modify: `run_agi_lab.sh` (call brief_assembler)
- Create: `tests/test_brief_assembler_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_brief_assembler_integration.py
"""Integration test: run brief_assembler against real lab state, verify output."""
import subprocess
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def test_cli_writes_context_brief(tmp_path):
    out_path = tmp_path / "context_brief.md"
    result = subprocess.run(
        ["python3", "tools/brief_assembler.py", "--out", str(out_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_path.exists()
    content = out_path.read_text()
    assert content.startswith("---\ngenerated_at:")
    assert "session_type:" in content
    # Real state should classify as phase-transition right now (May 4 post-factorial)
    # but we don't enforce that — just structural validity
    size_kb = len(content.encode("utf-8")) / 1024
    assert size_kb < 30, f"brief too large: {size_kb:.1f} KB"
```

- [ ] **Step 2: Run test, verify FAIL** (no CLI yet)

- [ ] **Step 3: Add CLI**

```python
# Append to tools/brief_assembler.py

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/memories/context_brief.md",
                    help="Output path (default: data/memories/context_brief.md)")
    ap.add_argument("--repo", default=".", help="Repo root (default: cwd)")
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
```

- [ ] **Step 4: Run test, verify PASS**

- [ ] **Step 5: Wire into runner**

In `run_agi_lab.sh`, find `write_session_brief()` (~line 100) and add a call to brief_assembler immediately after the existing brief is written:

```bash
# After the existing mv "${brief_tmp}" "${brief_final}" line, add:

# Generate context_brief.md (deterministic, no LLM, ~50ms).
# Director's procedural reads this in addition to session_brief.md.
if [ -x tools/brief_assembler.py ] || [ -f tools/brief_assembler.py ]; then
    python3 tools/brief_assembler.py --out data/memories/context_brief.md \
        2>>data/infra/brief_assembler.log || \
        echo "warning: brief_assembler failed (non-fatal); Director will fall back" \
            >>data/infra/brief_assembler.log
fi
```

- [ ] **Step 6: Smoke-test in place**

```bash
cd <repo>
python3 tools/brief_assembler.py
ls -la data/memories/context_brief.md
head -30 data/memories/context_brief.md
```

Expected: file exists, ~10-20 KB, well-formed sections.

- [ ] **Step 7: Commit**

```bash
git add tools/brief_assembler.py tests/test_brief_assembler_integration.py run_agi_lab.sh
git commit -m "feat(brief-assembler): CLI + runner integration"
```

### Task 8: STAGE 1 GATE — User review

- [ ] Brief output looks correct on real lab state (size, structure, content)
- [ ] All 7 brief_assembler tests pass
- [ ] Runner change syntax-checks (`bash -n run_agi_lab.sh`)
- [ ] User has reviewed the generated brief and approves Stage 2

---

## Stage 2 — agents.json tiering audit + apply

### Task 9: Audit agents.json

**Files:**
- Read: `data/agents/agents.json`
- Create: `docs/superpowers/plans/2026-05-04-agent-tier-audit.md` (output of audit)

- [ ] **Step 1: Read all 30 agent definitions**

```bash
cd <repo>
python3 -c "
import json
with open('data/agents/agents.json') as f:
    agents = json.load(f)
for name, defn in agents.items():
    print(f'{name}: model={defn.get(\"model\", \"?\")}  desc={defn.get(\"description\", \"\")[:80]}')
"
```

- [ ] **Step 2: For each agent, write tier rationale**

Save to `docs/superpowers/plans/2026-05-04-agent-tier-audit.md` with rows:
```
| role | proposed_tier | reasoning |
| pi | A (opus-4-7) | scientific judgment, program selection, paper approval |
| director | A | program execution coordination, dispatches |
| code_reviewer | A | catches subtle issues; downgrade risk too high |
...
```

Apply heuristic:
- Tier A: produces decisions, judgments, scientific verdicts, papers
- Tier B: implements, integrates, retrieves, fetches, summarizes
- Tier C: archives, formats, indexes, file-ops, mechanical extraction

- [ ] **Step 3: Commit audit**

```bash
git add docs/superpowers/plans/2026-05-04-agent-tier-audit.md
git commit -m "docs(tier-audit): per-role tier proposal for 30 agents"
```

### Task 10: Apply tiering to agents.json

**Files:**
- Modify: `data/agents/agents.json`
- Create: `tests/test_agents_json_tiering.py`

- [ ] **Step 1: Write test**

```python
# tests/test_agents_json_tiering.py
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_agents_json_uses_supported_models():
    with open(REPO / "data/agents/agents.json") as f:
        agents = json.load(f)
    valid_models = {"claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"}
    for name, defn in agents.items():
        assert defn.get("model") in valid_models, \
            f"{name} has invalid model: {defn.get('model')}"


def test_judgment_roles_remain_opus():
    """Non-negotiable opus-4-7 roles per spec."""
    must_be_opus = {"pi", "director", "code_reviewer", "evaluator", "paper_writer", "lab_architect"}
    with open(REPO / "data/agents/agents.json") as f:
        agents = json.load(f)
    for role in must_be_opus:
        if role in agents:  # only check roles that exist
            assert agents[role]["model"] == "claude-opus-4-7", \
                f"{role} must be opus-4-7"


def test_no_role_lacks_model():
    with open(REPO / "data/agents/agents.json") as f:
        agents = json.load(f)
    for name, defn in agents.items():
        assert "model" in defn, f"{name} missing model"
```

- [ ] **Step 2: Run tests** — first should PASS (currently all opus), second should PASS (judgment roles already opus), third should PASS (all already have model). Then we WILL change tiering and re-run.

- [ ] **Step 3: Apply audit results to agents.json**

Open `data/agents/agents.json` and update each role's `model` field per the audit table from Task 9. Use Edit tool, one role at a time, preserving JSON formatting.

- [ ] **Step 4: Run tests, verify PASS**

```bash
pytest tests/test_agents_json_tiering.py -v
```

- [ ] **Step 5: Commit**

```bash
git add data/agents/agents.json tests/test_agents_json_tiering.py
git commit -m "feat(agents): heterogeneous tier assignment per audit"
```

### Task 11: STAGE 2 internal — applies once Director session settles

(No user gate; auto-proceeds to Stage 3 after Director's current session completes.)

---

## Stage 3 — `dispatch_helper.py` + Director procedural

### Task 12: Build dispatch_helper

**Files:**
- Create: `tools/dispatch_helper.py`
- Create: `tests/test_dispatch_helper.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_dispatch_helper.py
from tools.dispatch_helper import dispatch


def test_uses_role_default_when_no_signals():
    result = dispatch("findings_curator", "archive last week's log entries")
    assert result["model"] == "claude-haiku-4-5"


def test_upgrades_to_opus_on_review_keywords():
    result = dispatch("findings_curator", "review and propose changes to the procedural")
    assert result["model"] == "claude-opus-4-7"


def test_upgrades_to_sonnet_on_implement_keywords():
    result = dispatch("findings_curator", "implement the fix from yesterday's diagnostic")
    assert result["model"] == "claude-sonnet-4-6"


def test_keeps_haiku_on_extract_keywords():
    result = dispatch("tooling_engineer", "extract last 5 D-N entries into a json file")
    assert result["model"] == "claude-haiku-4-5"


def test_unknown_role_falls_back_to_sonnet():
    result = dispatch("nonexistent_role", "do a thing")
    assert result["model"] == "claude-sonnet-4-6"
```

- [ ] **Step 2: Run tests, verify FAIL**

- [ ] **Step 3: Implement dispatch_helper**

```python
# tools/dispatch_helper.py
"""Decision-table dispatcher: given (role, task), return (model, effort).

Reads role's default_model from data/agents/agents.json. Applies override
keywords to upgrade or downgrade. Returns explicit dict so callers can log.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import TypedDict


_REPO = Path(__file__).resolve().parent.parent
_AGENTS_JSON = _REPO / "data/agents/agents.json"


class DispatchResult(TypedDict):
    model: str
    effort: str
    reasoning: str


# Keywords that force a specific tier
_OPUS_KEYWORDS = re.compile(r"\b(review|propose|verdict|approve|paper|close-out|sign-off|judge)\b", re.IGNORECASE)
_SONNET_KEYWORDS = re.compile(r"\b(implement|fix|wire|integrate|build|design)\b", re.IGNORECASE)
_HAIKU_KEYWORDS = re.compile(r"\b(extract|archive|format|plot|fetch|index|list|tally)\b", re.IGNORECASE)

_TIER_ORDER = ["claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"]


def _load_role_default(role: str) -> str:
    if not _AGENTS_JSON.exists():
        return "claude-sonnet-4-6"
    try:
        agents = json.loads(_AGENTS_JSON.read_text())
    except json.JSONDecodeError:
        return "claude-sonnet-4-6"
    return agents.get(role, {}).get("model", "claude-sonnet-4-6")


def dispatch(role: str, task: str) -> DispatchResult:
    """Return (model, effort) for dispatching this task to this role."""
    default = _load_role_default(role)

    if _OPUS_KEYWORDS.search(task):
        return {"model": "claude-opus-4-7", "effort": "max",
                "reasoning": f"task signals judgment-tier (default was {default})"}
    if _SONNET_KEYWORDS.search(task):
        # Upgrade haiku to sonnet, leave sonnet/opus alone
        if default == "claude-haiku-4-5":
            return {"model": "claude-sonnet-4-6", "effort": "high",
                    "reasoning": f"task signals integration (upgraded from {default})"}
        return {"model": default, "effort": "high",
                "reasoning": f"task signals integration; role default {default} retained"}
    if _HAIKU_KEYWORDS.search(task):
        # Don't downgrade opus-tier roles, but allow downgrade for sonnet roles
        if default == "claude-opus-4-7":
            return {"model": default, "effort": "low",
                    "reasoning": f"task is mechanical but role is judgment-tier; effort low"}
        return {"model": "claude-haiku-4-5", "effort": "low",
                "reasoning": f"task is mechanical (downgraded from {default})"}

    # No signal: trust role default with default effort
    return {"model": default, "effort": "high",
            "reasoning": f"no override signal; using role default {default}"}
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add tools/dispatch_helper.py tests/test_dispatch_helper.py
git commit -m "feat(dispatch-helper): decision-table dispatcher with override keywords"
```

### Task 13: Director procedural — point at brief + dispatch_helper

**Files:**
- Modify: `data/agents/director/procedural.md`

- [ ] **Step 1: Read current procedural**

```bash
wc -l data/agents/director/procedural.md
head -100 data/agents/director/procedural.md
```

- [ ] **Step 2: Locate the "Before doing anything, read" section**

Find the block that lists files Director reads at startup. Replace with:

```markdown
## Before Doing Anything, Read

1. `data/memories/session_brief.md` — runner-written pre-session pointer
2. `data/memories/context_brief.md` — DETERMINISTIC PRE-CURATED brief for THIS session (~10-20 KB, written by `tools/brief_assembler.py` immediately before Director launch). Always read this; it's tailored to your session_type.
3. The wiki-tier files NOT pre-loaded by context_brief — expand only when:
   - context_brief flagged them under "Decision-critical files"
   - your judgment requires content not in the brief
   - you need to write a substantive change to a wiki entry
   Use `Read` directly with explicit paths. Do not load the full wiki tier.

## Dispatching Agents

When dispatching a subagent, use `tools/dispatch_helper.py`:

```python
from tools.dispatch_helper import dispatch
result = dispatch(role="findings_curator", task="archive last 30 D-N entries to dated archive")
# result["model"] = "claude-haiku-4-5", result["effort"] = "low"
```

Pass `result["model"]` and `result["effort"]` to the Agent dispatch call. Override only when you have a specific judgment reason; log the override + reason in your decision entry.
```

- [ ] **Step 3: Commit**

```bash
git add data/agents/director/procedural.md
git commit -m "docs(director): read context_brief, dispatch via dispatch_helper"
```

### Task 14: STAGE 3 GATE — User review

- [ ] Director procedural change reviewed
- [ ] dispatch_helper test suite passes
- [ ] Brief actually loaded by next Director session (verify via session_log inspection — Director references context_brief.md)
- [ ] Startup-read size on next routine tick measured (telemetry: `startup_kb` field)
- [ ] User approves Stage 4

---

## Stage 4 — Self-escalation contracts

### Task 15: Add escalation contract template

**Files:**
- Create: `data/agents/_shared/self_escalation_contract.md`
- Modify: every agent's `data/agents/<role>/procedural.md`

- [ ] **Step 1: Write the contract template**

```markdown
# Self-Escalation Contract

You are running on tier {YOUR_TIER}. Your role default is {YOUR_DEFAULT_MODEL}.

On reading your assigned task, assess: does this task fit your tier?

Tier C (haiku-4-5): mechanical execution. File ops, archiving, formatting,
mechanical plotting, structured extraction. NOT scientific judgment, NOT
nuanced analysis, NOT design tradeoffs.

Tier B (sonnet-4-6): substantive engineering. Implementation, integration,
literature search + synthesis, mechanical-but-non-trivial extraction. NOT
program-level judgment, NOT paper-quality writing, NOT scientific verdicts.

Tier A (opus-4-7): judgment, taste, nuance. Scientific verdicts, paper
drafting, code review for subtle issues, program selection.

If the task before you exceeds your tier, return immediately:
- status: BLOCKED
- key_finding: "task exceeds tier {YOUR_TIER}"
- suggest_model: "{higher tier model id}"
- summary: <1-line description of the specific judgment / nuance the task requires>

DO NOT produce shallow output. False BLOCKED → Director re-dispatches at the
suggested tier (~5K wasted tokens, recoverable). Confident shallow output →
downstream consumers act on incorrect content (NOT recoverable).

Tier C special clause: when in doubt, escalate. The cost of false escalation
is much lower than the cost of a confident shallow answer.
```

- [ ] **Step 2: Append contract to every agent procedural**

```bash
for proc in data/agents/*/procedural.md; do
    role=$(basename $(dirname "$proc"))
    if grep -q "Self-Escalation Contract" "$proc"; then
        echo "skipping $role (already has contract)"
        continue
    fi
    cat >> "$proc" <<'EOF'

---

## Self-Escalation Contract

(See `data/agents/_shared/self_escalation_contract.md` for full text.)

If your assigned task exceeds your model tier, return BLOCKED with
suggest_model set to a higher tier instead of producing shallow output.
EOF
    echo "appended to $role"
done
```

- [ ] **Step 3: Commit**

```bash
git add data/agents/_shared/self_escalation_contract.md data/agents/*/procedural.md
git commit -m "feat(agents): self-escalation contract appended to all procedurals"
```

---

## Stage 5 — Verifier loops on opus-tier output

### Task 16: Define verifier pairs in `data/agents/_shared/verifier_pairs.json`

**Files:**
- Create: `data/agents/_shared/verifier_pairs.json`

- [ ] **Step 1: Write the pairs**

```json
{
  "pi": {
    "verifier_role": "lab_architect",
    "verifier_model": "claude-sonnet-4-6",
    "trigger": "any PI proposal or program-selection verdict",
    "max_iterations": 2
  },
  "paper_writer": {
    "verifier_role": "code_reviewer",
    "verifier_model": "claude-opus-4-7",
    "trigger": "every paper draft revision",
    "max_iterations": 2,
    "note": "different opus instance — checks technical claims independently"
  },
  "code_reviewer": {
    "verifier_role": "evaluator",
    "verifier_model": "claude-sonnet-4-6",
    "trigger": "every code review verdict",
    "max_iterations": 1
  },
  "evaluator": {
    "verifier_role": null,
    "trigger": "no verifier (evaluator IS a verifier)"
  }
}
```

- [ ] **Step 2: Update Director procedural to enforce pairs**

Add to Director procedural:

```markdown
## Verifier Loops on Opus-Tier Output

For every opus-tier deliverable, dispatch a verifier per
`data/agents/_shared/verifier_pairs.json`. Verifier must approve before the
output is considered final. Up to N iterations (per pairs file); after N,
escalate to user via session_exit.md as "verifier-rejection-loop".
```

- [ ] **Step 3: Commit**

```bash
git add data/agents/_shared/verifier_pairs.json data/agents/director/procedural.md
git commit -m "feat(agents): verifier pairs config + Director enforcement clause"
```

### Task 17: STAGE 5 GATE — User review

- [ ] Self-escalation contract appended to all procedurals
- [ ] Verifier pairs file exists + Director knows how to use it
- [ ] User approves Stage 6

---

## Stage 6 — Async consolidator agent + trigger

### Task 18: Define consolidator agent in agents.json

**Files:**
- Modify: `data/agents/agents.json`
- Create: `data/agents/consolidator/procedural.md`

- [ ] **Step 1: Add consolidator entry to agents.json**

```json
"consolidator": {
  "description": "Async memory consolidator — runs between Director sessions to archive old log entries, refresh wiki summaries, flag stale assumptions.",
  "model": "claude-haiku-4-5",
  "prompt": "..."
}
```

- [ ] **Step 2: Write procedural**

```markdown
# Consolidator — Async Memory Maintenance

You run BETWEEN Director sessions, off the critical path. Your job is to
keep memory tiers healthy without blocking the Director.

## Triggers
You are dispatched by `run_agi_lab.sh` when ANY of:
- ≥6 hours since last consolidator run
- ≥10 new D-N entries in `data/memories/log.md` since last run
- Phase boundary just crossed (orchestrator exited)

## Tasks (in order)
1. Read `data/memories/log.md`. Identify D-N entries older than 7 days.
2. Archive them to `data/archives/<YYYY-MM-DD>/log_consolidated.md` with a
   breadcrumb at the original location:
   `[archived to data/archives/<date>/log_consolidated.md]`
3. Update `data/memories/INDEX.md` with archive pointers.
4. Flag stale assumptions in `data/diagnostics/stale_assumptions.md`:
   - Assumptions in wiki tier referencing files that no longer exist
   - Carry-forwards (P-*) older than 14 days that aren't resolved
5. Append a one-line entry to `data/infra/consolidator.log` with timestamp,
   trigger reason, and what was done.

## Self-Escalation Contract
See `data/agents/_shared/self_escalation_contract.md`. If the consolidation
requires judgment (e.g., resolving a contradiction between two wiki
entries), return BLOCKED with suggest_model=claude-sonnet-4-6.
```

- [ ] **Step 3: Wire trigger into runner**

In `run_agi_lab.sh`, after the skip-when-stable check (line ~462), add:

```bash
# Consolidator trigger (D-LATER, this stage). Runs BEFORE Director session
# spawns, so wiki tier is fresh when Director reads it.
_should_run_consolidator() {
    local last_run_file="data/infra/consolidator.last_run"
    local now=$(date +%s)
    local last=0
    [ -f "$last_run_file" ] && last=$(cat "$last_run_file")
    local hours_since=$(( (now - last) / 3600 ))

    # Trigger 1: ≥6 hours since last
    [ "$hours_since" -ge 6 ] && return 0

    # Trigger 2: phase just ended
    _phase_just_ended && return 0

    # Trigger 3: ≥10 new D-N entries since last run
    if [ -f data/memories/log.md ] && [ -f "$last_run_file" ]; then
        local new_entries
        new_entries=$(awk -v last="$last" 'BEGIN{n=0} /^D-/{n++} END{print n}' data/memories/log.md)
        [ "$new_entries" -ge 10 ] && return 0
    fi

    return 1
}

if _should_run_consolidator; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [CONSOLIDATOR] Triggering async consolidator." \
        | tee -a data/infra/consolidator.log
    # Dispatch via claude with consolidator role (~5 min haiku run)
    claude --print --model claude-haiku-4-5 --effort high \
           --agents "$(jq '.consolidator' data/agents/agents.json)" \
           --prompt "Run consolidator procedural per data/agents/consolidator/procedural.md" \
           >>data/infra/consolidator.log 2>&1 || \
        echo "consolidator dispatch failed (non-fatal)" >>data/infra/consolidator.log
    date +%s >data/infra/consolidator.last_run
fi
```

- [ ] **Step 4: Commit**

```bash
git add data/agents/agents.json data/agents/consolidator/procedural.md run_agi_lab.sh
git commit -m "feat(consolidator): async memory consolidator + runner trigger"
```

---

## Stage 7 — Telemetry + weekly rollup

### Task 19: dispatch_telemetry hook

**Files:**
- Modify: `tools/dispatch_helper.py`
- Create: `tools/dispatch_rollup.py`
- Create: `tests/test_dispatch_telemetry.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_dispatch_telemetry.py
import json
import os
from pathlib import Path
from tools.dispatch_helper import dispatch, log_outcome


def test_dispatch_appends_to_telemetry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data/agents").mkdir(parents=True)
    (tmp_path / "data/agents/agents.json").write_text("{}")
    (tmp_path / "data/infra").mkdir(parents=True)

    result = dispatch("findings_curator", "archive last week")
    log_outcome(role="findings_curator", model=result["model"],
                escalated=False, verifier_pass=None)

    log_path = tmp_path / "data/infra/dispatch_telemetry.jsonl"
    assert log_path.exists()
    line = log_path.read_text().strip()
    obj = json.loads(line)
    assert obj["role"] == "findings_curator"
    assert obj["escalated"] is False
```

- [ ] **Step 2: Implement `log_outcome`**

```python
# Append to tools/dispatch_helper.py

def log_outcome(role: str, model: str, escalated: bool,
                verifier_pass: bool | None, task_class: str = "") -> None:
    """Append a single dispatch outcome to telemetry."""
    import datetime
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "role": role,
        "model_dispatched": model,
        "task_class": task_class,
        "escalated": escalated,
        "verifier_pass": verifier_pass,
    }
    log_path = _REPO / "data/infra/dispatch_telemetry.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
```

- [ ] **Step 3: Run test, verify PASS**

- [ ] **Step 4: Write rollup script**

```python
# tools/dispatch_rollup.py
"""Weekly rollup of dispatch telemetry. Per-role escalation + verifier-pass rates."""
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    log_path = REPO / "data/infra/dispatch_telemetry.jsonl"
    if not log_path.exists():
        print("no telemetry yet")
        return

    counts = defaultdict(lambda: {"total": 0, "escalated": 0, "verifier_fail": 0})
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        role = rec["role"]
        counts[role]["total"] += 1
        if rec.get("escalated"):
            counts[role]["escalated"] += 1
        if rec.get("verifier_pass") is False:
            counts[role]["verifier_fail"] += 1

    out_path = REPO / "data/infra/dispatch_rollup.md"
    lines = ["# Dispatch rollup\n", "| role | total | escalation_rate | verifier_fail_rate | flag |", "|---|---|---|---|---|"]
    for role in sorted(counts):
        c = counts[role]
        esc = c["escalated"] / max(c["total"], 1)
        vfail = c["verifier_fail"] / max(c["total"], 1)
        flag = ""
        if esc > 0.30:
            flag = "ESCALATION HIGH (consider tier upgrade)"
        elif vfail > 0.20:
            flag = "VERIFIER FAIL HIGH (consider model change or procedural fix)"
        lines.append(f"| {role} | {c['total']} | {esc:.1%} | {vfail:.1%} | {flag} |")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"rollup: wrote {len(counts)} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Add weekly trigger in runner** (or cron — simplest = once a day check at runner top)

In `run_agi_lab.sh` near the consolidator trigger:

```bash
# Weekly rollup (idempotent — overwrites)
if [ -f data/infra/dispatch_telemetry.jsonl ]; then
    python3 tools/dispatch_rollup.py >>data/infra/dispatch_rollup.log 2>&1 || true
fi
```

- [ ] **Step 6: Commit**

```bash
git add tools/dispatch_helper.py tools/dispatch_rollup.py tests/test_dispatch_telemetry.py run_agi_lab.sh
git commit -m "feat(telemetry): dispatch_telemetry + weekly rollup"
```

### Task 20: Final review

- [ ] All 7 stages complete + tested
- [ ] Director session post-deployment shows reduced startup_kb (telemetry)
- [ ] Telemetry fires on real dispatches; rollup readable
- [ ] User signs off on full architecture refresh

---

## Total: 20 tasks across 7 stages

Approximate effort: ~4-6 hours of mechanical implementation + ~2 hours user review at gates.
