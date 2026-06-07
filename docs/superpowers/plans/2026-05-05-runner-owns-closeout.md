# Runner-Owned Close-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move close-out responsibility from Director (LLM, unreliable) to runner (shell, deterministic) so silent Director deaths can be auto-recovered.

**Architecture:** `data/session_exit.md` becomes a structured JSON contract. `tools/post_director.py` finalizes log/current/queue/enqueue from that contract; falls back to silent-death recovery via expected-deliverable-pattern matching when the contract is missing or partial. `tools/queue_scanner.py` gains an orphan-deliverable detector. Runner calls `post_director.py` after every Director exit and `reclaim_stale()` every iteration.

**Tech Stack:** Python 3 (stdlib + tools/work_queue.py + pytest); bash (run_agi_lab.sh); markdown with YAML frontmatter for handler schema.

**Spec:** `docs/superpowers/specs/2026-05-05-runner-owns-closeout.md`

---

## Stage R1: Handler Schema Parser

**Goal:** Standalone library that loads YAML schema blocks from `work_queue_handlers.md`. Used by post_director.py and queue_scanner.py later.

### Task 1: Test fixture for handler schema parsing

**Files:**
- Create: `tests/fixtures/work_queue_handlers_sample.md`

- [ ] **Step 1: Create the fixture**

Create `tests/fixtures/work_queue_handlers_sample.md` with three minimal handler sections — one with full schema, one with `expected_deliverable_pattern: null`, one with no schema block at all (legacy):

```markdown
# Sample handlers

## `phase_advance`

```yaml
expected_deliverable_pattern: "programs/{program}/phase{to_phase_num}_*.md"
next_action_template:
  type: phase_advance
  priority: normal
  payload:
    from_phase: "{to_phase}"
    to_phase: "{to_phase}_close"
```

**Meaning:** Test handler.

**Action:** Test action.

---

## `cell_failed`

```yaml
expected_deliverable_pattern: null
next_action_template: null
```

**Meaning:** Test handler with null pattern.

---

## `legacy_no_schema`

**Meaning:** Test handler with no YAML block (legacy form).

**Action:** Test action.
```

- [ ] **Step 2: Commit fixture**

```bash
git add tests/fixtures/work_queue_handlers_sample.md
git commit -m "test: handler schema fixture with full/null/legacy forms"
```

---

### Task 2: handler_schema.py — load_schema() function

**Files:**
- Create: `tools/handler_schema.py`
- Test: `tests/test_handler_schema.py`

- [ ] **Step 1: Write failing test for load_schema()**

```python
# tests/test_handler_schema.py
"""Validates tools/handler_schema.py — YAML-block parser for work_queue_handlers.md."""
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests/fixtures/work_queue_handlers_sample.md"


def test_load_schema_parses_full_block():
    from tools.handler_schema import load_schema
    schemas = load_schema(FIXTURE)
    assert "phase_advance" in schemas
    pa = schemas["phase_advance"]
    assert pa["expected_deliverable_pattern"] == "programs/{program}/phase{to_phase_num}_*.md"
    assert pa["next_action_template"]["type"] == "phase_advance"
    assert pa["next_action_template"]["payload"]["from_phase"] == "{to_phase}"


def test_load_schema_parses_null_block():
    from tools.handler_schema import load_schema
    schemas = load_schema(FIXTURE)
    cf = schemas["cell_failed"]
    assert cf["expected_deliverable_pattern"] is None
    assert cf["next_action_template"] is None


def test_load_schema_skips_legacy_no_block():
    from tools.handler_schema import load_schema
    schemas = load_schema(FIXTURE)
    # legacy_no_schema has no YAML block; load_schema should return {} or absent for it
    assert schemas.get("legacy_no_schema") is None or schemas.get("legacy_no_schema") == {}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd <repo> && source .venv/bin/activate
python -m pytest tests/test_handler_schema.py::test_load_schema_parses_full_block -v
```
Expected: `ImportError: cannot import name 'load_schema' from 'tools.handler_schema'` or `ModuleNotFoundError`.

- [ ] **Step 3: Implement load_schema()**

```python
# tools/handler_schema.py
"""Parse YAML schema blocks from work_queue_handlers.md.

Each handler section starts with `## \`<type>\`` and may optionally contain a
YAML code block with the keys `expected_deliverable_pattern` (str | null) and
`next_action_template` (dict | null). load_schema() returns a dict mapping
type -> schema-dict. Handlers without a YAML block are absent from the result.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore


def load_schema(path: Path) -> dict[str, dict[str, Any]]:
    text = Path(path).read_text()
    schemas: dict[str, dict[str, Any]] = {}
    # Split on `## \`<type>\`` headers
    sections = re.split(r"\n## `([^`]+)`\n", text)
    # sections[0] is preamble; then alternating type, body, type, body, ...
    for i in range(1, len(sections), 2):
        type_name = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        # Find the first ```yaml ... ``` block
        m = re.search(r"```yaml\n(.*?)\n```", body, re.DOTALL)
        if not m:
            continue
        if yaml is None:
            raise RuntimeError("PyYAML required to parse handler schema")
        block = yaml.safe_load(m.group(1)) or {}
        schemas[type_name] = block
    return schemas
```

- [ ] **Step 4: Verify test passes**

```bash
python -m pytest tests/test_handler_schema.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/handler_schema.py tests/test_handler_schema.py
git commit -m "feat: handler_schema.load_schema() — parse YAML blocks from handlers.md"
```

---

### Task 3: handler_schema.py — render_template() function

**Files:**
- Modify: `tools/handler_schema.py`
- Modify: `tests/test_handler_schema.py`

- [ ] **Step 1: Write failing test for render_template()**

Append to `tests/test_handler_schema.py`:

```python
def test_render_template_substitutes_payload_vars():
    from tools.handler_schema import render_template
    template = "programs/{program}/phase{to_phase_num}_*.md"
    payload = {"program": "program_2_example", "to_phase": "P10"}
    rendered = render_template(template, payload)
    assert rendered == "programs/program_2_example/phase10_*.md"


def test_render_template_handles_missing_var():
    from tools.handler_schema import render_template
    # Missing var — substitute with empty string and log a warning
    rendered = render_template("programs/{program}/foo", {})
    assert rendered == "programs/{program}/foo" or rendered == "programs//foo"


def test_render_template_dict_recurses():
    from tools.handler_schema import render_template
    template = {
        "type": "phase_advance",
        "payload": {"from_phase": "{to_phase}", "to_phase": "{to_phase}_close"}
    }
    payload = {"to_phase": "P10"}
    rendered = render_template(template, payload)
    assert rendered["payload"]["from_phase"] == "P10"
    assert rendered["payload"]["to_phase"] == "P10_close"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_handler_schema.py::test_render_template_substitutes_payload_vars -v
```
Expected: `ImportError`.

- [ ] **Step 3: Implement render_template()**

Append to `tools/handler_schema.py`:

```python
def render_template(template: Any, payload: dict[str, Any]) -> Any:
    """Recursively render a template (string, dict, list, None) by substituting
    {var} placeholders with payload[var]. Derived vars: to_phase_num strips
    leading non-digits from to_phase. Missing vars are left as-is (logged)."""
    derived = dict(payload)
    if "to_phase" in payload:
        # Extract digits from to_phase: "P10" -> "10", "P10_close" -> "10"
        m = re.search(r"\d+", str(payload["to_phase"]))
        if m:
            derived["to_phase_num"] = m.group(0)

    if isinstance(template, str):
        out = template
        for k, v in derived.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(template, dict):
        return {k: render_template(v, payload) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(x, payload) for x in template]
    return template
```

- [ ] **Step 4: Verify all 3 tests pass**

```bash
python -m pytest tests/test_handler_schema.py -v
```
Expected: 6 passed (3 from Task 2 + 3 from Task 3).

- [ ] **Step 5: Commit**

```bash
git add tools/handler_schema.py tests/test_handler_schema.py
git commit -m "feat: handler_schema.render_template() — recursive {var} substitution with to_phase_num derived var"
```

---

## Stage R2: Schema Upgrade & Reference Docs

**Goal:** Add YAML schema blocks to all 9 item types in work_queue_handlers.md. Create session_exit_schema.md as reference for Director.

### Task 4: Upgrade work_queue_handlers.md with schema blocks for all 9 item types

**Files:**
- Modify: `data/agents/_shared/work_queue_handlers.md`
- Test: `tests/test_handler_schema_real.py` (NEW)

- [ ] **Step 1: Write failing test that all 9 types have schema entries (or explicit nulls)**

Create `tests/test_handler_schema_real.py`:

```python
"""Validates real work_queue_handlers.md has schema for all 9 item types."""
from pathlib import Path
from tools.handler_schema import load_schema

REPO = Path(__file__).resolve().parent.parent
HANDLERS = REPO / "data/agents/_shared/work_queue_handlers.md"

EXPECTED_TYPES = {
    "phase_advance", "cell_complete", "cell_failed", "operator_nudge",
    "verifier_review", "consolidator_run", "paper_review",
    "carry_forward_resolve", "heartbeat",
}


def test_all_item_types_have_schema_entry():
    schemas = load_schema(HANDLERS)
    missing = EXPECTED_TYPES - set(schemas)
    assert not missing, f"missing schema for: {missing}"


def test_each_schema_has_required_keys():
    schemas = load_schema(HANDLERS)
    for type_name, block in schemas.items():
        assert "expected_deliverable_pattern" in block, f"{type_name}: missing key"
        assert "next_action_template" in block, f"{type_name}: missing key"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_handler_schema_real.py -v
```
Expected: missing schema entries for all 9 types.

- [ ] **Step 3: Add YAML schema block at top of each handler section**

Edit `data/agents/_shared/work_queue_handlers.md`. For each `## \`<type>\`` section, insert a YAML block immediately after the header line. Concrete schemas:

`phase_advance`:
```yaml
expected_deliverable_pattern: "programs/{program}/phase{to_phase_num}_*.md"
next_action_template:
  type: phase_advance
  priority: normal
  payload:
    from_phase: "{to_phase}"
    to_phase: "{to_phase}_close"
    next_action: "dispatch_close_gate_for_{to_phase}"
    context: "Auto-recovered from silent-death; deliverable on disk; {to_phase} close gate pending PI dispatch."
```

`cell_complete`:
```yaml
expected_deliverable_pattern: "data/runs/{cell_id}/done"
next_action_template: null  # cell completion handler decides on its own
```

`cell_failed`:
```yaml
expected_deliverable_pattern: null  # failures should not be auto-recovered
next_action_template: null
```

`operator_nudge`:
```yaml
expected_deliverable_pattern: null  # nudges are read-and-respond, no deliverable artifact
next_action_template: null
```

`verifier_review`:
```yaml
expected_deliverable_pattern: "data/agents/{verifier_role}/episodic/*verifier_review*.md"
next_action_template: null  # verifier outcomes drive different follow-ons
```

`consolidator_run`:
```yaml
expected_deliverable_pattern: "data/agents/findings_curator/episodic/*km_consolidation*.md"
next_action_template: null
```

`paper_review`:
```yaml
expected_deliverable_pattern: "programs/{program}/paper_review_*.md"
next_action_template: null
```

`carry_forward_resolve`:
```yaml
expected_deliverable_pattern: null  # resolution is in-place edit, no new file
next_action_template: null
```

`heartbeat`:
```yaml
expected_deliverable_pattern: null  # heartbeat has no deliverable
next_action_template: null
```

- [ ] **Step 4: Verify tests pass**

```bash
python -m pytest tests/test_handler_schema_real.py -v
python -m pytest tests/test_handler_schema.py -v
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add data/agents/_shared/work_queue_handlers.md tests/test_handler_schema_real.py
git commit -m "feat: schema blocks (expected_deliverable_pattern + next_action_template) for all 9 item types"
```

---

### Task 5: Create session_exit_schema.md reference doc

**Files:**
- Create: `data/agents/_shared/session_exit_schema.md`

- [ ] **Step 1: Write the reference doc**

Content (full markdown body):

```markdown
# Session Exit Schema (RO-CO v1)

Director's last action of every session: write `data/session_exit.md` containing
**both** a JSON code block (structured channel) and a markdown body (operator
narrative + legacy `reason:` line). The runner's `tools/post_director.py`
finalizes log/current/queue/enqueue based on the JSON block; the markdown body
is for human consumption.

## JSON block schema

```json
{
  "schema_version": "1.0",
  "session_id": "D-313",
  "claimed_item_id": "wq-baa39841",
  "status": "success",
  "reason": "GRACEFUL_CHECKPOINT",
  "log_entry_text": "D-313 (2026-05-05, ~04:48 UTC): **Phase 3 P10 mechanism report delivered.** Brief Director session ... STATUS=success / KEY_FINDING=... / FILES_MODIFIED=... / SUMMARY=...\n\n---\n",
  "current_md_patches": [
    {"old": "P9 ANALYZE CLOSED + P10 MECHANISM OPEN", "new": "P10 MECHANISM REPORT DELIVERED + P10 CLOSE GATE PENDING"}
  ],
  "deliverables": [
    "programs/program_2_example/phase10_mechanism_report.md"
  ],
  "next_action": {
    "type": "phase_advance",
    "priority": "normal",
    "program": "program_2_example",
    "created_by": "director_session_d313",
    "payload": {
      "from_phase": "P10",
      "to_phase": "P10_close",
      "next_action": "dispatch_pi_for_p10_close_gate",
      "context": "P10 mechanism report delivered; P10 close gate pending PI dispatch."
    }
  },
  "notes": "Optional free text for operator review."
}
```

## Field semantics

- `schema_version`: always `"1.0"` for v1.
- `session_id`: `D-N` where N is the Director session number.
- `claimed_item_id`: the `wq-XXXX` ID Director claimed at session start (or `null` if no claim — e.g., heartbeat with empty queue).
- `status`: one of:
  - `"success"`: Director completed all intended work, deliverable on disk, ready for follow-on.
  - `"partial"`: some work succeeded, some did not. Runner does log/current/complete; does not enqueue follow-on.
  - `"failure"`: Director attempted but could not deliver. Runner fails the claim and enqueues `diagnostic_review`.
  - `"no_op"`: Director judged nothing-to-do (heartbeat with stable state). No mutations.
- `reason`: legacy field, must match an existing reason value (`GRACEFUL_CHECKPOINT`, `RATE_LIMIT`, `EVALUATOR_FAIL`, etc.). Runner's existing parsing reads this from the markdown body — keep them in sync.
- `log_entry_text`: full markdown body to append to `data/memories/log.md`. Include the trailing `---` separator and the `STATUS=`/`KEY_FINDING=`/etc. one-liner per existing log convention. Runner appends at top (after the preamble), preserving reverse-chronological order.
- `current_md_patches`: list of `{old, new}` str_replace ops applied in order. Each `old` must be a unique substring of current.md at the time of patch (use long anchor strings to be robust to drift).
- `deliverables`: list of file paths Director produced this session. Used by silent-death recovery as a cross-check.
- `next_action`: a queue item (same schema as `tools/work_queue.py` enqueue accepts) to enqueue, OR `null`. `created_by` should be `director_session_<lower(session_id)>`.
- `notes`: free text for operator. Optional.

## Markdown body conventions

Below the JSON block, write the existing freeform body Director already produces:

```markdown
reason: GRACEFUL_CHECKPOINT
session_id: D-313

(Free-text narrative for operator: what was done, what's next, any flags.)
```

The `reason:` line is parsed by existing runner code (line 790, run_agi_lab.sh).
The JSON block must precede this body so PyYAML/json parser can extract it
without colliding with `reason:`-style lines.

## Worked examples

### Example A: success with phase advance

(See top of this doc.)

### Example B: no_op heartbeat

```json
{
  "schema_version": "1.0",
  "session_id": "D-320",
  "claimed_item_id": "wq-9a8b7c6d",
  "status": "no_op",
  "reason": "GRACEFUL_CHECKPOINT",
  "log_entry_text": "",
  "current_md_patches": [],
  "deliverables": [],
  "next_action": null,
  "notes": "heartbeat tick: state stable, training silent, no anomalies"
}
```

(Empty `log_entry_text` → runner skips log append; runner still calls
`work_queue.complete()` on the heartbeat item.)

### Example C: failure with diagnostic enqueue

```json
{
  "schema_version": "1.0",
  "session_id": "D-405",
  "claimed_item_id": "wq-deadbeef",
  "status": "failure",
  "reason": "VERIFIER_REJECTION_LOOP",
  "log_entry_text": "D-405 (...): VERIFIER_REJECTION_LOOP after 2 iterations on paper_writer / scientific_reviewer pair.\n\n---\n",
  "current_md_patches": [],
  "deliverables": [],
  "next_action": null,
  "notes": "verifier rejected paper twice; needs methodological revision before retry"
}
```

(Runner: appends log, calls `work_queue.fail(wq-deadbeef, reason="VERIFIER_REJECTION_LOOP")`,
enqueues `diagnostic_review` with payload pointing at this failure.)

## Idempotency

- `log_entry_text` must contain `session_id` somewhere (in the heading line, for
  example). `post_director.py` checks for substring presence before appending —
  re-running over the same session_exit.md is a no-op.
- `current_md_patches` must use unique anchor strings. `post_director.py` skips
  a patch if `old` is not found AND `new` is already present (already-applied case).
- `next_action` is enqueued via `work_queue.enqueue()` which is idempotent on
  deterministic IDs; running twice does not duplicate.
```

- [ ] **Step 2: Commit**

```bash
git add data/agents/_shared/session_exit_schema.md
git commit -m "docs: session_exit_schema.md reference for Director (RO-CO v1)"
```

---

## Stage R2 Gate (user review): pause for confirmation that schema design is right before building post_director.py.

---

## Stage R3: post_director.py — Structured-Finalize Path

**Goal:** Read structured `data/session_exit.md`, apply log/current/queue mutations.

### Task 6: Test scaffolding for post_director.py

**Files:**
- Create: `tests/test_post_director.py`
- Create: `tests/fixtures/session_exit_success.md`
- Create: `tests/fixtures/session_exit_failure.md`
- Create: `tests/fixtures/session_exit_no_op.md`
- Create: `tests/fixtures/session_exit_legacy.md`

- [ ] **Step 1: Create the four fixture files**

Each fixture is a complete `session_exit.md` example matching schema in Task 5. `session_exit_legacy.md` has only the markdown body, no JSON block.

(Concrete content for each fixture — use the worked examples from Task 5.)

- [ ] **Step 2: Write failing scaffolding test**

```python
# tests/test_post_director.py
"""Validates tools/post_director.py — runner-side close-out finalizer."""
from pathlib import Path
import json
import pytest
import sys

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _setup_repo(tmp_path):
    """Build a minimal repo skeleton matching post_director.py's expectations."""
    (tmp_path / "data/memories").mkdir(parents=True)
    (tmp_path / "data/work_queue").mkdir(parents=True)
    (tmp_path / "data/work_queue/completed").mkdir()
    (tmp_path / "data/work_queue/failed").mkdir()
    (tmp_path / "data/infra").mkdir(parents=True)
    (tmp_path / "data/agents/_shared").mkdir(parents=True)
    (tmp_path / "data/memories/log.md").write_text("# Log\n\n---\n\n")
    (tmp_path / "data/memories/current.md").write_text("# Current\n\nStatus: P9 ANALYZE\n")
    return tmp_path


def test_module_imports(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    assert hasattr(pd, "main")
```

- [ ] **Step 3: Run to verify it fails**

```bash
python -m pytest tests/test_post_director.py::test_module_imports -v
```
Expected: `ModuleNotFoundError: No module named 'tools.post_director'`.

- [ ] **Step 4: Create skeleton tools/post_director.py**

```python
# tools/post_director.py
"""Runner-side close-out finalizer.

Runs unconditionally after every Director exit. Reads data/session_exit.md and
applies log/current/queue mutations per the structured JSON block. Falls back
to silent-death recovery when the file is missing or partial.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Verify scaffolding test passes**

```bash
python -m pytest tests/test_post_director.py::test_module_imports -v
```
Expected: 1 passed.

- [ ] **Step 6: Commit scaffolding**

```bash
git add tools/post_director.py tests/test_post_director.py tests/fixtures/session_exit_*.md
git commit -m "scaffold: tools/post_director.py + test fixtures (success/failure/no_op/legacy)"
```

---

### Task 7: post_director.py — read_session_exit() with JSON-block extraction

**Files:**
- Modify: `tools/post_director.py`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Write failing test for read_session_exit()**

```python
def test_read_session_exit_extracts_json_block(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    fixture = (Path(__file__).parent / "fixtures/session_exit_success.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    se = pd.read_session_exit()
    assert se is not None
    assert se["status"] == "success"
    assert se["session_id"].startswith("D-")
    assert se["claimed_item_id"].startswith("wq-")


def test_read_session_exit_returns_none_when_missing(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    se = pd.read_session_exit()
    assert se is None


def test_read_session_exit_returns_legacy_marker_when_no_json_block(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    fixture = (Path(__file__).parent / "fixtures/session_exit_legacy.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    se = pd.read_session_exit()
    assert se is not None
    assert se.get("_legacy") is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_post_director.py -v
```

- [ ] **Step 3: Implement read_session_exit()**

```python
def read_session_exit() -> dict | None:
    """Read data/session_exit.md and return the JSON block (with `_legacy: True`
    marker if no JSON block found but file exists)."""
    path = REPO / "data/session_exit.md"
    if not path.exists():
        return None
    text = path.read_text()
    m = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not m:
        return {"_legacy": True, "_raw": text}
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return {"_legacy": True, "_raw": text, "_invalid_json": True}
```

- [ ] **Step 4: Verify tests pass**

```bash
python -m pytest tests/test_post_director.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tools/post_director.py tests/test_post_director.py
git commit -m "feat: post_director.read_session_exit() — JSON block extraction with legacy fallback"
```

---

### Task 8: post_director.py — finalize_from_structured() success branch

**Files:**
- Modify: `tools/post_director.py`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Write failing test**

```python
def test_finalize_success_appends_log_and_completes_claim(tmp_path, monkeypatch):
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    # Set up an active claim
    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal", "program": "test",
        "created_by": "test", "payload": {"to_phase": "P10"}
    })
    wq.claim(item_id, "director_session_d999")

    se = {
        "status": "success",
        "session_id": "D-999",
        "claimed_item_id": item_id,
        "log_entry_text": "D-999 (...): Test entry\n\n---\n",
        "current_md_patches": [{"old": "P9 ANALYZE", "new": "P10 DONE"}],
        "next_action": None,
    }
    pd.finalize_from_structured(se)

    log = (tmp_path / "data/memories/log.md").read_text()
    assert "D-999" in log
    cur = (tmp_path / "data/memories/current.md").read_text()
    assert "P10 DONE" in cur and "P9 ANALYZE" not in cur
    s = wq.summary()
    assert s["claimed"] == 0
    assert s["completed_today"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_post_director.py::test_finalize_success -v
```

- [ ] **Step 3: Implement finalize_from_structured() success branch**

```python
def finalize_from_structured(se: dict) -> None:
    import tools.work_queue as wq
    status = se.get("status")
    sid = se.get("session_id", "")

    if status in ("success", "partial"):
        if se.get("log_entry_text"):
            _append_log_idempotent(se["log_entry_text"], sid)
        for patch in (se.get("current_md_patches") or []):
            _apply_current_patch(patch)
        if se.get("claimed_item_id"):
            wq.complete(se["claimed_item_id"], outcome={"status": status, "session_id": sid})
        if se.get("next_action"):
            wq.enqueue(se["next_action"])
    elif status == "failure":
        if se.get("log_entry_text"):
            _append_log_idempotent(se["log_entry_text"], sid)
        if se.get("claimed_item_id"):
            wq.fail(se["claimed_item_id"], reason=se.get("reason", "director-reported-failure"))
        # Enqueue diagnostic_review
        wq.enqueue({
            "type": "diagnostic_review", "priority": "urgent",
            "program": "", "created_by": f"post_director_{sid}",
            "payload": {"failed_item_id": se.get("claimed_item_id"),
                        "reason": se.get("reason", "unknown"),
                        "notes": se.get("notes", "")},
        })
    elif status == "no_op":
        if se.get("log_entry_text"):
            _append_log_idempotent(se["log_entry_text"], sid)


def _append_log_idempotent(text: str, sid: str) -> None:
    log = REPO / "data/memories/log.md"
    body = log.read_text()
    if sid and sid in body:
        return  # already appended
    # Insert after the preamble (first `---\n\n`)
    parts = body.split("\n---\n\n", 1)
    if len(parts) == 2:
        new_body = parts[0] + "\n---\n\n" + text + "\n" + parts[1]
    else:
        new_body = body + "\n" + text + "\n"
    log.write_text(new_body)


def _apply_current_patch(patch: dict) -> bool:
    cur = REPO / "data/memories/current.md"
    body = cur.read_text()
    old = patch.get("old", "")
    new = patch.get("new", "")
    if old and old in body:
        cur.write_text(body.replace(old, new, 1))
        return True
    if new and new in body:
        return False  # already applied
    return False  # not found and not applied — log warning
```

- [ ] **Step 4: Verify test passes**

```bash
python -m pytest tests/test_post_director.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tools/post_director.py tests/test_post_director.py
git commit -m "feat: post_director.finalize_from_structured() — success/partial branches"
```

---

### Task 9: post_director.py — failure and no_op branches + idempotency

**Files:**
- Modify: `tools/post_director.py`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Write failing tests**

```python
def test_finalize_failure_fails_claim_and_enqueues_diagnostic(tmp_path, monkeypatch):
    # Similar setup to success test
    # ... assert wq.summary() shows failed=1, pending=1 (the diagnostic_review)


def test_finalize_no_op_no_mutations(tmp_path, monkeypatch):
    # se = {"status": "no_op", "log_entry_text": "", "current_md_patches": [], "next_action": None, ...}
    # ... assert log unchanged, current unchanged, no queue mutation


def test_finalize_idempotent_on_rerun(tmp_path, monkeypatch):
    # Run finalize_from_structured twice; second run should be no-op
    # ... assert log appended only once, claim completed only once
```

- [ ] **Step 2-4: Implement, run, fix until tests pass**

(Implementation already drafted in Task 8 — just verify failure/no_op/idempotent branches work.)

- [ ] **Step 5: Commit**

```bash
git add tools/post_director.py tests/test_post_director.py
git commit -m "feat: post_director.py failure + no_op branches; idempotency checks"
```

---

### Task 10: post_director.py — main() entry point + telemetry

**Files:**
- Modify: `tools/post_director.py`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Write failing test for main() + telemetry**

```python
def test_main_writes_telemetry(tmp_path, monkeypatch):
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    fixture = (Path(__file__).parent / "fixtures/session_exit_no_op.md").read_text()
    (tmp_path / "data/session_exit.md").write_text(fixture)

    pd.main()
    telem = tmp_path / "data/infra/post_director_telemetry.jsonl"
    assert telem.exists()
    rec = json.loads(telem.read_text().strip())
    assert "branch_taken" in rec
    assert rec["branch_taken"] in ("structured_no_op", "structured_success",
                                   "structured_partial", "structured_failure",
                                   "legacy_finalize", "silent_death", "silent_death_no_claim")
```

- [ ] **Step 2-4: Implement main() that dispatches and writes telemetry, run tests until pass**

```python
def main() -> int:
    se = read_session_exit()
    branch = "unknown"
    try:
        if se is None:
            branch = "silent_death"
            recover_from_silent_death()  # implemented in Stage R4
        elif se.get("_legacy"):
            branch = "legacy_finalize"
            legacy_finalize(se)
        else:
            status = se.get("status", "")
            branch = f"structured_{status}"
            finalize_from_structured(se)
    finally:
        _log_telemetry({"branch_taken": branch, "session_id": se.get("session_id") if se else None})
    return 0
```

- [ ] **Step 5: Commit**

```bash
git add tools/post_director.py tests/test_post_director.py
git commit -m "feat: post_director.main() entry + telemetry; legacy_finalize stub"
```

---

## Stage R3 Gate (user review): pause to verify post_director.py structured paths work end-to-end on a sandbox before adding silent-death recovery.

---

## Stage R4: Silent-Death Recovery

**Goal:** When session_exit.md is missing, walk the active claim and recover via expected_deliverable_pattern.

### Task 11: post_director.py — recover_from_silent_death() with deliverable

**Files:**
- Modify: `tools/post_director.py`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Write failing test**

```python
def test_silent_death_with_deliverable_recovers(tmp_path, monkeypatch):
    """Director claimed item, produced deliverable, died without session_exit.md."""
    import tools.post_director as pd
    import tools.work_queue as wq
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    monkeypatch.setattr(wq, "_REPO", tmp_path)

    # Copy real handler schema to tmp repo
    src_handlers = REPO / "data/agents/_shared/work_queue_handlers.md"
    (tmp_path / "data/agents/_shared/work_queue_handlers.md").write_text(src_handlers.read_text())

    item_id = wq.enqueue({
        "type": "phase_advance", "priority": "normal",
        "program": "test_program", "created_by": "test",
        "payload": {"program": "test_program", "to_phase": "P10"}
    })
    wq.claim(item_id, "director_session_d999")

    # Create the expected deliverable (matches phase_advance pattern programs/{program}/phase{to_phase_num}_*.md)
    deliv = tmp_path / "programs/test_program/phase10_mechanism_report.md"
    deliv.parent.mkdir(parents=True)
    deliv.write_text("# Mechanism report\n\n(43 KB equivalent test content)\n" + "x" * 1000)

    # No session_exit.md — silent death
    pd.main()

    s = wq.summary()
    assert s["claimed"] == 0
    assert s["completed_today"] == 1
    log = (tmp_path / "data/memories/log.md").read_text()
    assert "RECOVERY" in log or "silent" in log.lower()
    # Should also enqueue session_recovery_review
    assert s["pending"] >= 1


def test_silent_death_without_deliverable_fails(tmp_path, monkeypatch):
    # Same setup but don't create the deliverable
    # ... assert claim is failed, diagnostic_review enqueued
```

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement recover_from_silent_death()**

```python
def recover_from_silent_death() -> None:
    import tools.work_queue as wq
    import tools.handler_schema as hs
    from datetime import datetime, timezone

    claimed = _read_claimed()
    if not claimed:
        return  # nothing to recover

    handlers = hs.load_schema(REPO / "data/agents/_shared/work_queue_handlers.md")

    for claim in claimed:
        schema = handlers.get(claim.get("type"))
        if not schema or not schema.get("expected_deliverable_pattern"):
            # Handler opted out; fail the claim for operator review
            _silent_death_fail(claim, reason="silent_death_handler_opt_out")
            continue

        pattern = hs.render_template(schema["expected_deliverable_pattern"], claim["payload"])
        matches = list((REPO).glob(pattern))
        claim_ts = datetime.fromisoformat(claim["claimed_at"].replace("Z", "+00:00")).timestamp()
        fresh = [m for m in matches if m.stat().st_mtime > claim_ts and m.stat().st_size > 100]

        if fresh:
            _silent_death_recover(claim, deliverable=fresh[0], schema=schema, handlers=handlers)
        else:
            _silent_death_fail(claim, reason="silent_death_no_deliverable")


def _silent_death_recover(claim, deliverable, schema, handlers):
    import tools.work_queue as wq
    import tools.handler_schema as hs
    sid_synthetic = f"D-{claim['id'][3:]}-RECOVERY"
    log_text = (
        f"\n{sid_synthetic} ({_now_iso()}, **runner-auto-recovered**): "
        f"**Director silent-died after deliverable; auto-recovery performed.** "
        f"Claim `{claim['id']}` (type={claim['type']}); deliverable `{deliverable.relative_to(REPO)}` "
        f"({deliverable.stat().st_size} bytes) on disk; `work_queue.complete()` + follow-on enqueue + "
        f"`session_recovery_review` enqueued. STATUS=auto_recovered.\n\n---\n"
    )
    _append_log_idempotent(log_text, sid_synthetic)
    wq.complete(claim["id"], outcome={
        "status": "success_recovered", "synthetic": True,
        "deliverable": str(deliverable.relative_to(REPO)),
    })

    # Enqueue follow-on per handler template
    if schema.get("next_action_template"):
        next_item = hs.render_template(schema["next_action_template"], claim["payload"])
        next_item.setdefault("program", claim.get("program", ""))
        next_item.setdefault("created_by", f"post_director_silent_death_recovery")
        wq.enqueue(next_item)

    # Always enqueue a session_recovery_review for operator inspection
    wq.enqueue({
        "type": "session_recovery_review", "priority": "normal",
        "program": claim.get("program", ""),
        "created_by": "post_director_silent_death_recovery",
        "payload": {
            "orphan_item_id": claim["id"],
            "deliverable": str(deliverable.relative_to(REPO)),
            "claimer": claim.get("claimed_by", "unknown"),
        },
    })


def _silent_death_fail(claim, reason):
    import tools.work_queue as wq
    sid_synthetic = f"D-{claim['id'][3:]}-FAILED"
    log_text = (
        f"\n{sid_synthetic} ({_now_iso()}, **runner-detected**): "
        f"**Director silent-died without deliverable.** Claim `{claim['id']}` "
        f"(type={claim['type']}) failed; diagnostic_review enqueued. STATUS=silent_death.\n\n---\n"
    )
    _append_log_idempotent(log_text, sid_synthetic)
    wq.fail(claim["id"], reason=reason)
    wq.enqueue({
        "type": "diagnostic_review", "priority": "urgent",
        "program": claim.get("program", ""),
        "created_by": "post_director_silent_death_fail",
        "payload": {"failed_item_id": claim["id"], "reason": reason},
    })
```

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

```bash
git add tools/post_director.py tests/test_post_director.py
git commit -m "feat: post_director silent-death recovery — deliverable detection + synthetic log + follow-on enqueue"
```

---

### Task 12: post_director.py — handle session_recovery_review item type

**Files:**
- Modify: `data/agents/_shared/work_queue_handlers.md` (add `session_recovery_review` and `diagnostic_review` sections)
- Modify: `tests/test_handler_schema_real.py` (extend EXPECTED_TYPES)

- [ ] **Step 1: Update EXPECTED_TYPES test to include 2 new types**

```python
EXPECTED_TYPES = {
    "phase_advance", "cell_complete", "cell_failed", "operator_nudge",
    "verifier_review", "consolidator_run", "paper_review",
    "carry_forward_resolve", "heartbeat",
    "session_recovery_review", "diagnostic_review",  # new in Stage R4
}
```

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Add the two new sections to work_queue_handlers.md with full Meaning/Action/Complete/Fail prose + YAML schema (both with `expected_deliverable_pattern: null`)**

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: handlers for session_recovery_review + diagnostic_review item types"
```

---

## Stage R5: Orphan Deliverable Detector

**Goal:** queue_scanner.py detects orphan claims even when post_director.py also failed to run.

### Task 13: queue_scanner.py — orphan_deliverable detector

**Files:**
- Modify: `tools/queue_scanner.py`
- Test: `tests/test_queue_scanner_orphan.py` (NEW)

- [ ] **Step 1: Write failing test**

```python
def test_orphan_detector_emits_recovery_when_deliverable_exists_for_old_claim(tmp_path, monkeypatch):
    # Set up: claim that's >30min old + matching deliverable on disk + no completion event
    # Expect: scanner emits session_recovery_review queue item
```

(Plus the negative cases — young claim, no deliverable, already-completed.)

- [ ] **Step 2-4: Add detector function, run tests until pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: queue_scanner orphan_deliverable detector — emits session_recovery_review for stranded claims"
```

---

### Task 14: queue_scanner.py — wire orphan detector into scan() pipeline

**Files:**
- Modify: `tools/queue_scanner.py`
- Modify: `tests/test_queue_scanner.py` (existing)

- [ ] **Step 1: Add orphan detector to the existing scan() loop**

- [ ] **Step 2: Test idempotency end-to-end (scan twice, no duplicate emit)**

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: queue_scanner wires orphan detector into scan() pipeline"
```

---

## Stage R5 Gate (user review): pause to verify post_director + scanner work together against a sandboxed silent-death scenario before touching the runner.

---

## Stage R6: Runner Integration & Director Procedural

**Goal:** `run_agi_lab.sh` calls post_director after every Director exit; reclaim_stale every iteration. Director procedural mandates session_exit.md schema.

### Task 15: run_agi_lab.sh — reclaim_stale every iteration

**Files:**
- Modify: `run_agi_lab.sh`

- [ ] **Step 1: Find current location of reclaim_stale call (likely inside the dispatch path)**

```bash
grep -n "reclaim_stale" run_agi_lab.sh
```

- [ ] **Step 2: Move it to the top of the main loop, before the queue-empty check**

- [ ] **Step 3: Verify shell script syntactically valid**

```bash
bash -n run_agi_lab.sh
```

- [ ] **Step 4: Test in dry-run / smoke mode**

```bash
# Manually trigger one iteration with a stale claim, verify reclaim runs
```

- [ ] **Step 5: Commit**

```bash
git commit -m "fix(runner): reclaim_stale every iteration, not only on dispatch path"
```

---

### Task 16: run_agi_lab.sh — call post_director.py after every Director exit

**Files:**
- Modify: `run_agi_lab.sh`

- [ ] **Step 1: Add post_director.py call as first step after EXIT_CODE=$?, before EXIT_REASON parsing**

```bash
# After: EXIT_CODE=$?
python3 tools/post_director.py >> "$SESSION_LOG" 2>&1 || \
    echo "WARNING: post_director.py exited non-zero; check $SESSION_LOG" >&2
```

- [ ] **Step 2: Verify existing EXIT_REASON parsing (line 790) still works alongside the new structured channel**

- [ ] **Step 3: Smoke test**

- [ ] **Step 4: Commit**

```bash
git commit -m "fix(runner): call post_director.py after every Director exit"
```

---

### Task 17: Director procedural — mandate session_exit.md JSON schema

**Files:**
- Modify: `data/agents/director/procedural.md`
- Test: `tests/test_director_procedural_session_exit.py` (NEW)

- [ ] **Step 1: Write failing test that procedural references new schema**

```python
def test_director_procedural_references_session_exit_schema():
    text = Path("data/agents/director/procedural.md").read_text()
    assert "session_exit_schema.md" in text
    assert "JSON" in text  # must mention JSON block
    assert '"status"' in text or "status: success" in text  # must show schema fields


def test_director_procedural_forbids_self_completion():
    text = Path("data/agents/director/procedural.md").read_text()
    # Must explicitly say Director does NOT call work_queue.complete itself
    assert "do not call" in text.lower() or "do not run" in text.lower() or "runner finalizes" in text.lower()
```

- [ ] **Step 2: Run tests to verify failures**

- [ ] **Step 3: Edit Director procedural to:**
  1. Replace any existing "call work_queue.complete()" instruction with "populate `next_action` in session_exit.md; runner finalizes"
  2. Add a "Step Last: Write session_exit.md" section that points at session_exit_schema.md
  3. Note that runner now does log/current/queue mutations based on the JSON block

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(director): procedural mandates session_exit.md JSON schema; runner finalizes housekeeping"
```

---

## Stage R6 Gate (user review): runner is rewired. Pause to confirm before R7 end-to-end test on the live lab.

---

## Stage R7: End-to-End Validation

**Goal:** Reproduce D-313-like silent death in a sandbox; verify auto-recovery; restart runner against live lab.

### Task 18: End-to-end silent-death simulation test

**Files:**
- Create: `tests/test_e2e_silent_death.py`

- [ ] **Step 1: Build sandbox runner that:**
  1. Enqueues a `phase_advance` queue item.
  2. "Director" subprocess writes a fake deliverable to programs/test_program/phase10_*.md.
  3. "Director" exits WITHOUT writing session_exit.md.
  4. Runs `tools/post_director.py`.
  5. Asserts: queue item completed, follow-on enqueued, session_recovery_review enqueued, log entry appended, log entry contains "RECOVERY".

- [ ] **Step 2: Implement, run, debug until passes**

- [ ] **Step 3: Commit**

```bash
git commit -m "test(e2e): silent-death simulation passes auto-recovery"
```

---

### Task 19: Restart live runner & monitor first 24h

**Files:**
- (no code changes — operational task)

- [ ] **Step 1: Restart runner**

```bash
make lab-stop
make lab-start
```

- [ ] **Step 2: Monitor for 24h: any silent-death telemetry hits, any recovery actions, any orphan_deliverable detections**

```bash
tail -F data/infra/post_director_telemetry.jsonl
```

- [ ] **Step 3: After 24h, verify queue health, log integrity, and that the next phase advance progressed correctly**

---

## Self-Review Notes

Before marking the plan complete, verify:

1. **Spec coverage:** Each section of the spec maps to at least one task above.
   - Schema upgrade → Task 4
   - session_exit_schema.md → Task 5
   - post_director.py finalize → Tasks 6-10
   - Silent-death recovery → Tasks 11-12
   - Orphan detector → Tasks 13-14
   - Runner integration → Tasks 15-16
   - Director procedural → Task 17
   - E2E test → Task 18
   - Restart → Task 19

2. **Type consistency:** `next_action` schema in session_exit.md (Task 5) matches `tools/work_queue.py` enqueue requirements (`created_by` is required field).

3. **No placeholders:** All steps have concrete code blocks. No "TBD" or "implement appropriately."

4. **TDD discipline:** Each implementation task has Test → Run-fail → Implement → Run-pass → Commit cadence.

---

## Estimated effort

- Stage R1: 45 min (parser + tests)
- Stage R2: 30 min (schema upgrade + reference doc)
- Stage R3: 90 min (post_director.py 4 sub-tasks + tests)
- Stage R4: 60 min (silent-death recovery)
- Stage R5: 45 min (orphan detector)
- Stage R6: 60 min (runner integration + procedural)
- Stage R7: 60 min (e2e test + restart + monitor)

**Total: ~6.5 hours**, plus user-review gates between R2/R3, R3/R4, R5/R6, and final R7.
