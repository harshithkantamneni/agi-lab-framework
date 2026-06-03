"""Validates Director's procedural mandates the RO-CO session_exit.md schema."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROCEDURAL = REPO / "data/agents/director/procedural.md"


def test_procedural_references_session_exit_schema():
    """Procedural must point at the schema reference doc."""
    text = PROCEDURAL.read_text()
    assert "session_exit_schema.md" in text, \
        "Director procedural must reference data/agents/_shared/session_exit_schema.md"


def test_procedural_mandates_json_block():
    """Procedural must instruct Director to write a structured JSON block."""
    text = PROCEDURAL.read_text()
    # Must mention the JSON code block convention or specific schema fields
    has_json_keyword = "json" in text.lower() and "session_exit" in text.lower()
    assert has_json_keyword, "Director procedural must mention JSON in context of session_exit"
    # Must reference at least one of the structured-channel field names
    assert any(field in text for field in (
        "log_entry_text", "current_md_patches", "next_action", "claimed_item_id"
    )), "Director procedural must mention the structured-channel fields"


def test_procedural_states_runner_finalizes():
    """Procedural must explicitly say runner does the housekeeping, not Director."""
    text = PROCEDURAL.read_text().lower()
    # Look for any explicit handoff statement
    handoff_phrases = (
        "runner finalizes",
        "runner does the",
        "runner handles",
        "post_director",
        "runner-owned close-out",
        "runner will finalize",
        "runner-side finalizer",
    )
    assert any(p in text for p in handoff_phrases), \
        "Director procedural must explicitly say runner does close-out housekeeping"


def test_procedural_forbids_direct_log_edits_at_session_end():
    """Procedural must instruct Director NOT to edit log.md / current.md at session end —
    populate the JSON patches/log_entry_text fields and let the runner apply them."""
    text = PROCEDURAL.read_text().lower()
    forbidden_patterns = (
        "do not edit log",
        "do not write log.md",
        "do not call work_queue.complete",
        "do not run work_queue.complete",
        "runner finalizes log",
        "do not directly edit log",
        "runner appends log",
        "runner writes log",
    )
    assert any(p in text for p in forbidden_patterns), (
        "Director procedural must explicitly forbid direct log/current edits at session end "
        "(populate session_exit.md and let runner apply)"
    )


def test_procedural_step_last_section_exists():
    """Procedural must have a clearly labeled 'Step Last' or 'Final step' or
    'Before exiting' section that points at session_exit.md."""
    text = PROCEDURAL.read_text()
    last_step_anchors = (
        "Step Last",
        "Final step",
        "Last step",
        "Before exit",
        "End of session",
        "Session end",
        "Exit protocol",
    )
    matched = [a for a in last_step_anchors if a in text]
    assert matched, (
        "Director procedural must have a labeled final-step section "
        "(one of: Step Last / Final step / Last step / Before exit / End of session / "
        "Session end / Exit protocol)"
    )
    # And the matched section must mention session_exit.md
    # Find the position of the first matched anchor
    first_anchor_idx = min(text.find(a) for a in matched if a in text)
    after_anchor = text[first_anchor_idx: first_anchor_idx + 2000]
    assert "session_exit.md" in after_anchor, \
        "the final-step section must reference session_exit.md"
