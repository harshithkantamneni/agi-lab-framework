"""tests/test_director_queue_procedural.py — Stage Q4 structural tests.

Verifies that:
1. data/agents/_shared/work_queue_handlers.md exists and covers all 9 item types.
2. data/agents/director/procedural.md contains the Work Queue section with correct
   references to tools.work_queue and the handler doc.
3. The procedural warns against anti-patterns (one item per session, fail() for
   out-of-scope items).
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

HANDLERS_DOC = REPO_ROOT / "data" / "agents" / "_shared" / "work_queue_handlers.md"
DIRECTOR_PROCEDURAL = REPO_ROOT / "data" / "agents" / "director" / "procedural.md"

# All 9 item types defined in the v1 spec.
ITEM_TYPES_V1 = [
    "phase_advance",
    "cell_complete",
    "cell_failed",
    "operator_nudge",
    "verifier_review",
    "consolidator_run",
    "paper_review",
    "carry_forward_resolve",
    "heartbeat",
]


# ---------------------------------------------------------------------------
# 1. handlers doc exists and lists all item types
# ---------------------------------------------------------------------------


def test_handlers_doc_exists_and_lists_all_item_types() -> None:
    """work_queue_handlers.md must exist and mention every v1 item type."""
    assert HANDLERS_DOC.exists(), (
        f"Missing handler reference doc: {HANDLERS_DOC}\n"
        "Stage Q4 requires data/agents/_shared/work_queue_handlers.md."
    )

    content = HANDLERS_DOC.read_text(encoding="utf-8")

    missing = [t for t in ITEM_TYPES_V1 if t not in content]
    assert not missing, (
        f"work_queue_handlers.md is missing sections for item types: {missing}\n"
        "All 9 v1 types must be documented."
    )


# ---------------------------------------------------------------------------
# 2. Director procedural references the work queue
# ---------------------------------------------------------------------------


def test_director_procedural_references_work_queue() -> None:
    """Director procedural must contain the Work Queue section with key references."""
    assert DIRECTOR_PROCEDURAL.exists(), (
        f"Director procedural not found: {DIRECTOR_PROCEDURAL}"
    )

    content = DIRECTOR_PROCEDURAL.read_text(encoding="utf-8")

    # Section heading
    assert "## Work Queue" in content, (
        "Director procedural is missing '## Work Queue' section."
    )

    # Must reference tools.work_queue module
    assert "tools.work_queue" in content, (
        "Director procedural does not reference 'tools.work_queue'."
    )

    # Must reference the four core operations
    for fn in ("peek", "claim", "complete", "fail"):
        assert fn in content, (
            f"Director procedural does not mention '{fn}' from tools.work_queue."
        )

    # Must reference the handler doc
    assert "work_queue_handlers.md" in content, (
        "Director procedural does not reference 'data/agents/_shared/work_queue_handlers.md'."
    )


# ---------------------------------------------------------------------------
# 3. Procedural warns against anti-patterns
# ---------------------------------------------------------------------------


def test_director_procedural_warns_against_anti_patterns() -> None:
    """Director procedural must warn: one item per session + fail() for out-of-scope."""
    content = DIRECTOR_PROCEDURAL.read_text(encoding="utf-8")

    # "one item per session" or equivalent phrasing
    one_item_phrases = [
        "one item per session",
        "One item per session",
        "claim multiple items",
        "Don't claim multiple",
    ]
    assert any(p in content for p in one_item_phrases), (
        "Director procedural must warn against claiming multiple items per session."
    )

    # fail() for out-of-scope items
    out_of_scope_phrases = [
        "out-of-scope",
        "out of scope",
        "handler-doc says is out-of-scope",
    ]
    assert any(p in content for p in out_of_scope_phrases), (
        "Director procedural must instruct Director to fail() out-of-scope items."
    )
