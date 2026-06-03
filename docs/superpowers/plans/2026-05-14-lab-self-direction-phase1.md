# Lab Self-Direction Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the queue structurally non-empty when an active program has remaining work. Phase 1 = B′ + C′ + D′ from the v2 spec.

**Architecture:** Three components built in parallel TDD-style:
- **L1 (B′)**: schema-enforced `next_action` in session_exit.md + post_director redispatch on violation
- **L2 (C′)**: artifact-presence queue projection — workspace IS the queue
- **L3 (D′)**: Agent Contracts on Director — formal success criteria

**Tech Stack:** Python 3 stdlib + PyYAML (already in venv) + pytest. No external deps.

**Spec:** `docs/superpowers/specs/2026-05-14-lab-self-direction-v2.md`
**Research basis:** `docs/superpowers/notes/2026-05-14-agentic-architecture-research.md`

---

## Stage L1: Schema-Enforced next_action (B′)

**Goal:** `session_exit.md` JSON schema requires `next_action` (or explicit `program_complete: true`). Runner rejects violations and immediately redispatches Director with correction prompt.

### Task L1.1: Update session_exit_schema.md to v1.1

**Files:**
- Modify: `data/agents/_shared/session_exit_schema.md`

- [ ] **Step 1: Add the validation section + worked failure example**

Update the schema reference doc to:
- Mark `next_action` as **REQUIRED** when `status ∈ {success, partial}` AND `program_complete != true`
- Add `program_complete` as optional boolean field (default false). When true with `next_action: null`, that's the ONE valid combination for "this program is done."
- Add a worked **schema-violation** example showing the runner's correction prompt response.

The doc should explicitly say:
> If `status=success|partial` AND `next_action=null` AND `program_complete != true`, the runner rejects the session_exit and immediately re-dispatches Director with a correction prompt asking it to either declare program_complete OR populate next_action.

- [ ] **Step 2: Commit**

```bash
git add data/agents/_shared/session_exit_schema.md
git commit -m "docs: session_exit_schema v1.1 — next_action required + program_complete field + violation example"
```

### Task L1.2: post_director.py validation + redispatch logic

**Files:**
- Modify: `tools/post_director.py`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Write failing tests**

```python
def test_finalize_rejects_success_without_next_action(tmp_path, monkeypatch):
    """status=success + next_action=None + program_complete=false → schema_violation branch."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    se = {
        "status": "success",
        "session_id": "D-TEST-1",
        "claimed_item_id": "wq-test",
        "log_entry_text": "D-TEST-1: ...",
        "current_md_patches": [],
        "next_action": None,
        # program_complete NOT set (defaults to false)
    }
    result = pd.finalize_from_structured(se)
    # Should return a violation marker (not silently succeed)
    assert result is None or result.get("violation") is True


def test_finalize_accepts_success_with_program_complete(tmp_path, monkeypatch):
    """status=success + next_action=None + program_complete=true → valid (program closing)."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    se = {
        "status": "success",
        "session_id": "D-TEST-2",
        "claimed_item_id": "wq-test",
        "log_entry_text": "D-TEST-2: program complete\n",
        "current_md_patches": [],
        "next_action": None,
        "program_complete": True,
    }
    # Should NOT raise; valid program-complete declaration
    pd.finalize_from_structured(se)


def test_finalize_no_op_does_not_require_next_action(tmp_path, monkeypatch):
    """status=no_op exempt from next_action requirement (heartbeat case)."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    se = {
        "status": "no_op",
        "session_id": "D-TEST-3",
        "claimed_item_id": None,
        "log_entry_text": "",
        "current_md_patches": [],
        "next_action": None,
    }
    pd.finalize_from_structured(se)  # should not raise


def test_main_violation_writes_violation_telemetry(tmp_path, monkeypatch):
    """When schema violation detected, branch_taken=schema_violation in telemetry."""
    import tools.post_director as pd
    monkeypatch.setattr(pd, "REPO", _setup_repo(tmp_path))
    # Write invalid session_exit.md
    invalid = {"status": "success", "session_id": "D-V", "claimed_item_id": "wq-v",
               "log_entry_text": "x", "current_md_patches": [], "next_action": None}
    import json as _j
    (tmp_path / "data/session_exit.md").write_text(
        f"```json\n{_j.dumps(invalid)}\n```\n\nreason: GRACEFUL_CHECKPOINT\n"
    )
    pd.main()
    telem_path = tmp_path / "data/infra/post_director_telemetry.jsonl"
    last = _j.loads(telem_path.read_text().strip().split("\n")[-1])
    assert last["branch_taken"] in ("schema_violation", "structured_violation_redispatched")
```

- [ ] **Step 2: Run tests fail → implement validation function**

Add to `tools/post_director.py`:

```python
def _validate_session_exit_schema(se: dict) -> tuple[bool, str | None]:
    """Validate session_exit JSON against v1.1 schema.
    
    Returns (is_valid, violation_reason). Reason is None when valid.
    """
    status = se.get("status")
    if status not in ("success", "partial", "failure", "no_op"):
        return False, f"invalid status: {status!r}"
    
    # next_action requirement: success/partial MUST have next_action OR program_complete
    if status in ("success", "partial"):
        has_next_action = se.get("next_action") is not None
        program_complete = se.get("program_complete") is True
        if not has_next_action and not program_complete:
            return False, (
                "status=success/partial requires either next_action populated "
                "OR program_complete=true; got both null/false"
            )
    
    return True, None
```

Modify `finalize_from_structured()` to call validator FIRST and return early on violation:

```python
def finalize_from_structured(se: dict) -> dict | None:
    is_valid, violation = _validate_session_exit_schema(se)
    if not is_valid:
        # Write violation event to telemetry; do NOT apply mutations
        return {"violation": True, "reason": violation}
    
    # ... existing success/partial/failure/no_op dispatch logic
```

Modify `main()` to set branch_taken=schema_violation when finalize returns violation marker. Also: write a redispatch trigger file (`data/session_exit_redispatch_pending.flag`) so the runner sees it and re-spawns Director.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(post_director): schema-enforced next_action; violations → redispatch trigger"
```

### Task L1.3: run_agi_lab.sh — redispatch on schema violation flag

**Files:**
- Modify: `run_agi_lab.sh`
- Modify: `tests/test_runner_reclaim_stale.py` (extend with new test)

- [ ] **Step 1: Add to runner main loop after post_director call**

After post_director.py call in the runner, check for the redispatch flag:

```bash
# Schema violation triggers immediate redispatch (L1.3 / B′)
if [ -f data/session_exit_redispatch_pending.flag ]; then
    echo "$(date) [SCHEMA-VIOLATION] post_director flagged redispatch; spawning Director with correction prompt"
    rm -f data/session_exit_redispatch_pending.flag
    # Set environment variable that Director's session_brief picks up
    export DIRECTOR_REDISPATCH_REASON="schema_violation_previous_session_exit"
    # Fall through to normal Director spawn (the env var triggers correction prompt at brief assembly)
fi
```

- [ ] **Step 2: brief_assembler.py — honor DIRECTOR_REDISPATCH_REASON**

When `DIRECTOR_REDISPATCH_REASON` env var is set, inject a "CORRECTION PROMPT" block at top of `context_brief.md` explaining what was wrong and what Director must do this session.

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(runner): redispatch Director on schema violation with correction prompt"
```

## Stage L1 Gate: pause for user review before L2.

---

## Stage L2: Artifact-Presence Queue Projection (C′)

**Goal:** Define canonical artifacts per program phase. Scanner inspects filesystem and emits queue items deterministically.

### Task L2.1: artifact_schema.yaml format + sample for program_2 P11

**Files:**
- Create: `programs/program_2_dense_vs_moe_sub100m/artifact_schema.yaml`
- Create: `docs/superpowers/notes/artifact-schema-format.md` (operator reference)

- [ ] **Step 1: Define the schema format (operator-facing doc)**

```yaml
# artifact_schema.yaml format v1.0
program: program_2_dense_vs_moe_sub100m
current_phase: P11
phases:
  - id: P11
    artifacts:
      - id: P11.apparatus_methodology_compliant
        path: "programs/{program}/phase11_apparatus_methodology_compliant.md"
        description: "Apparatus uses lm_eval.evaluator.simple_evaluate, NOT subprocess wrapper"
        handler:
          type: apparatus_build
          priority: normal
          payload_template:
            scope_doc: "programs/{program}/phase11_apparatus_methodology_gap.md"
            next_action: "tooling_engineer adapter rewrite to subclass lm_eval.api.model.LM"
            context: "[from spec] Required-before-launch artifact for P11 real-data scoring"
      - id: P11.smoke_pass_post_rewrite
        path: "data/eval/p11/smoke_a42_post_rewrite.json"
        prereqs: [P11.apparatus_methodology_compliant]
        handler:
          type: apparatus_build
          priority: normal
          payload_template:
            next_action: "Re-run A42 HSPA smoke test against the rewritten adapter"
      - id: P11.real_data_results
        path: "programs/{program}/phase11_real_data_results.md"
        prereqs: [P11.smoke_pass_post_rewrite]
        handler:
          type: phase_advance
          priority: normal
          payload_template:
            from_phase: P11_apparatus_complete
            to_phase: P11_real_data_scoring
            next_action: "dispatch_measurement_theorist_and_statistical_reviewer for 4-benchmark real-data scoring"
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: artifact_schema.yaml for program_2 P11 + format reference doc"
```

### Task L2.2: tools/artifact_queue_projector.py

**Files:**
- Create: `tools/artifact_queue_projector.py`
- Create: `tests/test_artifact_queue_projector.py`

- [ ] **Step 1: Write failing tests**

Tests cover:
- Empty repo (no schema file) → empty projection
- Schema with no missing artifacts → empty projection
- Missing leaf artifact (no prereqs) → 1 item emitted
- Missing artifact with prereqs satisfied → 1 item emitted
- Missing artifact with unsatisfied prereqs → 0 items (waits for prereqs)
- Idempotency: same artifact state → same item IDs (compute_id determinism)
- Multi-program: schema per program; only emit for active program

- [ ] **Step 2: Implement projector**

```python
"""Artifact-presence queue projector — workspace IS the queue.

Reads programs/<active>/artifact_schema.yaml + filesystem state.
Emits queue items for missing artifacts whose prereqs are satisfied.
Deterministic — zero LLM calls.

Per spec: docs/superpowers/specs/2026-05-14-lab-self-direction-v2.md (C′).
"""
from __future__ import annotations
import json, sys, yaml
from pathlib import Path

# sys.path bootstrap (same pattern as queue_scanner.py)
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_ON_PATH = str(_TOOLS_DIR.parent)
if _REPO_ROOT_ON_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_ON_PATH)

REPO = Path(__file__).resolve().parent.parent


def project_artifacts(active_program: str, repo_root: Path | None = None) -> list[dict]:
    """Read artifact_schema.yaml for active program; emit queue items for
    missing artifacts whose prereqs are satisfied.
    
    Returns list of queue item dicts ready for work_queue.enqueue().
    """
    root = repo_root or REPO
    schema_path = root / "programs" / active_program / "artifact_schema.yaml"
    if not schema_path.exists():
        return []
    
    schema = yaml.safe_load(schema_path.read_text())
    items: list[dict] = []
    
    # Build artifact-id → exists map
    artifact_exists: dict[str, bool] = {}
    artifact_handlers: dict[str, dict] = {}
    artifact_prereqs: dict[str, list[str]] = {}
    
    for phase in schema.get("phases", []):
        for artifact in phase.get("artifacts", []):
            aid = artifact["id"]
            path_template = artifact["path"]
            path = root / path_template.format(program=active_program)
            artifact_exists[aid] = path.exists()
            artifact_handlers[aid] = artifact.get("handler", {})
            artifact_prereqs[aid] = artifact.get("prereqs", [])
    
    # Emit items for missing artifacts with satisfied prereqs
    for aid, exists in artifact_exists.items():
        if exists:
            continue
        prereqs = artifact_prereqs[aid]
        if not all(artifact_exists.get(p, False) for p in prereqs):
            continue
        handler = artifact_handlers[aid]
        payload = dict(handler.get("payload_template", {}))
        # Render {program} in payload values
        for k, v in payload.items():
            if isinstance(v, str):
                payload[k] = v.format(program=active_program)
        payload["_artifact_id"] = aid  # deterministic dedup key
        items.append({
            "type": handler.get("type", "apparatus_build"),
            "priority": handler.get("priority", "normal"),
            "program": active_program,
            "created_by": "artifact_queue_projector",
            "payload": payload,
        })
    
    return items
```

- [ ] **Step 3: Run tests pass + commit**

```bash
git commit -m "feat: artifact_queue_projector — workspace IS the queue (C′)"
```

### Task L2.3: Wire projector into queue_scanner.py

**Files:**
- Modify: `tools/queue_scanner.py`
- Modify: `tests/test_queue_scanner.py`

- [ ] **Step 1: Add projector as 8th detector**

In `scan()` function, after the 7 existing detectors:

```python
try:
    from tools.artifact_queue_projector import project_artifacts
    active_program = _active_program(root)
    if active_program:
        artifact_items = project_artifacts(active_program, repo_root=root)
except Exception as exc:
    _warn(f"artifact_queue_projector failed: {exc}")
    artifact_items = []
```

Add `+ artifact_items` to the `all_items` aggregation.

- [ ] **Step 2: Idempotency test**

Two scans against same state → same item IDs (compute_id deterministic since payload._artifact_id is stable).

- [ ] **Step 3: Commit**

```bash
git commit -m "feat: queue_scanner wires artifact_queue_projector as 8th detector"
```

## Stage L2 Gate: pause for user review before L3.

---

## Stage L3: Agent Contracts on Director (D′)

**Goal:** Director has formal success criteria. Contract violation = detectable structural failure, triggers redispatch.

### Task L3.1: agent_contracts.json schema + Director contract

**Files:**
- Create: `data/agents/_shared/agent_contracts.json`

- [ ] **Step 1: Define contracts file**

```json
{
  "schema_version": "1.0",
  "contracts": {
    "director": {
      "max_tokens_per_session": 500000,
      "max_cycles_between_invocations_seconds": 14400,
      "success_criteria": [
        {
          "id": "session_exit_present",
          "check": "file_exists",
          "path": "data/session_exit.md"
        },
        {
          "id": "session_exit_schema_valid",
          "check": "session_exit_schema_v1.1"
        },
        {
          "id": "next_action_or_program_complete",
          "check": "next_action_populated_or_program_complete_true"
        }
      ],
      "violation_handler": "immediate_redispatch_with_correction_prompt",
      "max_redispatches_per_session": 3,
      "escalation_on_max_redispatches": "schedule_operator_review"
    }
  }
}
```

- [ ] **Step 2: Commit**

```bash
git commit -m "feat: agent_contracts.json — Director resource bounds + success criteria"
```

### Task L3.2: post_director.py contract validation

**Files:**
- Modify: `tools/post_director.py`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Write failing tests**

Tests cover:
- Contract loads from JSON cleanly
- Contract check fires after schema validation
- Contract violation → redispatch flag set + telemetry records `contract_violated_redispatched`
- Max redispatches (3) → escalation flag set instead
- Contract success path → existing behavior unchanged

- [ ] **Step 2: Implement contract check**

```python
def _check_contract(se: dict, role: str = "director") -> tuple[bool, str | None]:
    """Check session against agent contract. Returns (passed, violation_reason)."""
    contracts_path = REPO / "data" / "agents" / "_shared" / "agent_contracts.json"
    if not contracts_path.exists():
        return True, None  # no contract = no enforcement
    
    contracts = json.loads(contracts_path.read_text()).get("contracts", {})
    contract = contracts.get(role)
    if not contract:
        return True, None
    
    for criterion in contract.get("success_criteria", []):
        cid = criterion["id"]
        check_type = criterion["check"]
        if check_type == "file_exists":
            if not (REPO / criterion["path"]).exists():
                return False, f"contract criterion {cid!r}: file missing"
        elif check_type == "session_exit_schema_v1.1":
            ok, reason = _validate_session_exit_schema(se)
            if not ok:
                return False, f"contract criterion {cid!r}: {reason}"
        elif check_type == "next_action_populated_or_program_complete_true":
            if se.get("status") in ("success", "partial"):
                if not se.get("next_action") and not se.get("program_complete"):
                    return False, f"contract criterion {cid!r}: violated"
    
    return True, None
```

Integrate into `finalize_from_structured()`:

```python
def finalize_from_structured(se: dict) -> dict | None:
    # 1. Schema validation
    ok, reason = _validate_session_exit_schema(se)
    if not ok:
        return {"violation": True, "reason": reason, "violation_type": "schema"}
    
    # 2. Contract check
    ok, reason = _check_contract(se)
    if not ok:
        return {"violation": True, "reason": reason, "violation_type": "contract"}
    
    # 3. Apply mutations (existing logic)
    ...
```

- [ ] **Step 3: Run tests pass + commit**

```bash
git commit -m "feat(post_director): contract validation (D′) — Director success criteria enforced"
```

### Task L3.3: Redispatch ceiling + operator escalation

**Files:**
- Modify: `tools/post_director.py`
- Modify: `run_agi_lab.sh`
- Modify: `tests/test_post_director.py`

- [ ] **Step 1: Track redispatch count**

Add to `data/work_queue/last_seen.json` (already a state file):

```json
{
  "director_redispatch_count": 0,
  "director_redispatch_session_id_last": null
}
```

When session_id changes, reset count to 0. Each violation increments.

- [ ] **Step 2: Escalation at max (3)**

When count >= 3 for same session_id (consecutive failures):
- Write `data/operator_review_pending.md` with the failure history
- Send macOS notification: "AGI Lab: Director contract violation 3× — operator review needed"
- Runner exits the redispatch loop and falls through to skip-when-stable

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(post_director): redispatch ceiling + operator escalation at 3× contract violations"
```

## Stage L3 Gate: pause for final review.

---

## Stage L4: End-to-End Validation

### Task L4.1: Full pipeline test

- [ ] Simulate D-313 scenario: Director writes session_exit with `status=success, next_action=None, program_complete=false`
- [ ] Verify: schema_violation telemetry → contract_violated → redispatch flag → runner redispatches Director with correction prompt
- [ ] Verify: artifact_queue_projector picks up `phase11_apparatus_methodology_compliant.md` missing + emits apparatus_build queue item

### Task L4.2: Documentation + sunsetting

- [ ] Update `data/user_notes.md` with NOTICE about Phase 1 deployment
- [ ] Mark the 2026-05-13 informal B+C spec as superseded
- [ ] Commit

```bash
git commit -m "validate: Phase 1 (B′+C′+D′) end-to-end against D-313 scenario; operator notice published"
```

---

## Self-review checklist

- [ ] Every section of the v2 spec maps to a task
- [ ] No placeholders (TBD, TODO without concrete content)
- [ ] TDD discipline maintained: tests first, then implementation
- [ ] User-review gates after L1, L2, L3 — operator can pause/redirect

## Estimated effort

- **Stage L1**: ~6h (schema + post_director enforcement + runner redispatch)
- **Stage L2**: ~6h (yaml schema + projector + scanner integration)
- **Stage L3**: ~4h (contracts file + validation + redispatch ceiling)
- **Stage L4**: ~2h (e2e + docs)

**Total: ~18h via subagent-driven-development, plus user-review gates between stages.**

Phase 1 alone closes both observed failure modes (D-313 silent-death + D-326..D-328 narrative-only) with architectural guarantees instead of behavioral hopes.
