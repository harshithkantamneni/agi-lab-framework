"""Artifact-presence queue projector — workspace IS the queue.

Reads programs/<active>/artifact_schema.yaml + inspects filesystem state.
Emits queue items for missing artifacts whose prereqs are satisfied.
Deterministic — zero LLM calls.

Per spec: docs/superpowers/specs/2026-05-14-lab-self-direction-v2.md (C′ pattern).
"""
from __future__ import annotations
import json
import sys
import yaml
from pathlib import Path

# sys.path bootstrap (mirrors queue_scanner.py / post_director.py / cost_rollup.py pattern)
_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_ON_PATH = str(_TOOLS_DIR.parent)
if _REPO_ROOT_ON_PATH not in sys.path:
    sys.path.insert(0, _REPO_ROOT_ON_PATH)

REPO = Path(__file__).resolve().parent.parent


def load_artifact_schema(program: str, repo_root: Path | None = None) -> dict | None:
    """Read programs/<program>/artifact_schema.yaml. Return parsed dict or None if missing/malformed."""
    root = repo_root or REPO
    path = root / "programs" / program / "artifact_schema.yaml"
    if not path.exists():
        return None
    try:
        return yaml.safe_load(path.read_text())
    except yaml.YAMLError:
        return None


def project_artifacts(
    program: str,
    repo_root: Path | None = None,
) -> list[dict]:
    """Read the artifact schema for `program` and inspect filesystem state.

    Returns a list of queue item dicts ready for tools.work_queue.enqueue().
    Each item:
    - Missing artifact whose path doesn't exist
    - All prereq artifacts exist
    - Handler block from schema is rendered with {program} substitution

    Idempotent: same filesystem state → same queue items (compute_id stable
    because payload._artifact_id is the unique key).
    """
    root = repo_root or REPO
    schema = load_artifact_schema(program, root)
    if schema is None:
        return []

    current_phase = schema.get("current_phase")
    phases = schema.get("phases", []) or []

    # Build artifact-id → exists map across ALL phases (so prereqs in prior phases work)
    artifact_exists: dict[str, bool] = {}
    artifact_handlers: dict[str, dict] = {}
    artifact_prereqs: dict[str, list[str]] = {}
    artifact_active_phase: dict[str, bool] = {}

    for phase in phases:
        is_active_phase = phase.get("id") == current_phase
        for artifact in phase.get("artifacts", []) or []:
            aid = artifact.get("id")
            if not aid:
                continue
            path_template = artifact.get("path", "")
            resolved_path = root / path_template.format(program=program)
            artifact_exists[aid] = resolved_path.exists()
            artifact_handlers[aid] = artifact.get("handler", {}) or {}
            artifact_prereqs[aid] = artifact.get("prereqs", []) or []
            artifact_active_phase[aid] = is_active_phase

    items: list[dict] = []
    for aid, exists in artifact_exists.items():
        if exists:
            continue
        # Only emit for active-phase artifacts
        if not artifact_active_phase.get(aid, False):
            continue
        # Skip if prereqs not satisfied
        prereqs = artifact_prereqs[aid]
        if not all(artifact_exists.get(p, False) for p in prereqs):
            continue

        handler = artifact_handlers[aid]
        if not handler.get("type"):
            continue  # malformed schema — skip silently

        # Render payload_template via the shared handler_schema.render_template.
        # That helper also auto-derives to_phase_num from to_phase if present,
        # so artifact_schema.yaml templates can reference {to_phase_num} if they
        # set to_phase in payload — useful when artifact paths or payloads
        # depend on numeric phase IDs.
        from tools.handler_schema import render_template as _render
        payload_raw = handler.get("payload_template", {}) or {}
        payload = _render(payload_raw, {"program": program})
        # Add deterministic dedup keys — _artifact_id for human readability,
        # _dedup_key so compute_id hashes only (type, program, _dedup_key) and
        # payload text edits in payload_template do NOT orphan pending items.
        payload["_artifact_id"] = aid
        payload["_dedup_key"] = aid

        items.append({
            "type": handler["type"],
            "priority": handler.get("priority", "normal"),
            "program": program,
            "created_by": "artifact_queue_projector",
            "payload": payload,
        })

    return items


def main(argv: list[str] | None = None) -> int:
    """CLI inspection mode — print items that would be emitted.

    Production runs via tools/queue_scanner.py integration (which calls
    project_artifacts() directly and enqueues via work_queue.enqueue).
    This CLI is for manual inspection only.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Artifact queue projector (inspection mode).")
    parser.add_argument("program", help="program identifier (matches directory name)")
    args = parser.parse_args(argv)

    items = project_artifacts(args.program)
    print(json.dumps(items, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
