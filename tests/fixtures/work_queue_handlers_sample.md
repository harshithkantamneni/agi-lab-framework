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
