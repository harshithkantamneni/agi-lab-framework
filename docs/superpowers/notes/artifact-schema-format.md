# Artifact Schema Format (v1.0)

The `artifact_schema.yaml` file at `programs/<program>/artifact_schema.yaml` describes the canonical set of artifacts a program produces per phase. The `tools/artifact_queue_projector.py` tool reads this file and emits queue items for missing artifacts whose prerequisites are satisfied.

## Top-level structure

```yaml
program: <program_id>
current_phase: <phase_id>       # e.g., "P11"
schema_version: "1.0"
phases:
  - id: <phase_id>
    artifacts:
      - id: <artifact_id>
        path: <relative_path>
        description: <human-readable>
        prereqs: [<artifact_id>, ...]   # optional
        handler:
          type: <queue_item_type>
          priority: urgent | normal | low
          payload_template:
            <key>: <value>             # {program} interpolated at projection time
```

## Field semantics

- **program**: identifier matching the directory name (e.g., `program_2_dense_vs_moe_sub100m`)
- **current_phase**: the phase the projector should consider "active." Artifacts in other phases are ignored unless the operator updates this field.
- **artifact.id**: unique within the schema. Used as dedup key by `compute_id()` so re-projection produces the same queue-item ID.
- **artifact.path**: relative to repo root. Supports `{program}` interpolation. Existence check is `Path(path.format(program=program)).exists()`.
- **artifact.prereqs**: list of artifact IDs that must exist before this artifact's missing-state should fire. Empty/omitted = no prereqs.
- **artifact.handler.type**: one of the valid queue item types in `work_queue_handlers.md` (apparatus_build, phase_advance, etc.)
- **artifact.handler.priority**: priority hint for the emitted queue item.
- **artifact.handler.payload_template**: dict of key→value pairs that become the queue item's payload. The string `{program}` is interpolated. The `_artifact_id` key is auto-added by the projector for dedup determinism.

**Missing `current_phase` field**: if `current_phase` is omitted or null, the projector silently emits no items (because no phase is considered "active"). If your schema seems to produce no work even though artifacts are missing, check that `current_phase` is set correctly.

## Projector behavior

For each artifact in the schema:
1. If artifact's path exists → skip (no work needed)
2. If any prereq artifact's path does NOT exist → skip (wait for prereqs)
3. Otherwise → emit a queue item using handler.{type, priority, payload_template}

This is a pure file-system inspection. Zero LLM calls. Deterministic — same filesystem state produces the same queue items.

## Operator workflow

When closing a phase or starting a new one:
1. Update `current_phase` in `artifact_schema.yaml`
2. Add new artifacts for the new phase (if not already defined)
3. Set their prereqs correctly (artifacts from the prior phase that gate the new work)

The projector handles the rest automatically — no manual queue management.

## Out of scope (v1)

- Cross-program artifacts (each schema is single-program)
- Conditional artifacts (e.g., "this artifact required only if MoE arch enabled")
- Time-based artifacts (e.g., "weekly status report") — those are queue_scanner detectors, not artifacts
- Auto-detecting when an artifact becomes stale and needs regeneration

These can be added in v2 if needed.
