"""Validates mechanism_extractor procedural references training digest pipeline."""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROCEDURAL = REPO / "data/agents/mechanism_extractor/procedural.md"


def test_procedural_references_phase_summary_path():
    text = PROCEDURAL.read_text()
    assert "data/digests/training/" in text, \
        "mechanism_extractor procedural must reference data/digests/training/ path"
    assert "phase" in text.lower() and "summary" in text.lower(), \
        "must reference the phase-level summary digest"


def test_procedural_specifies_digests_first_then_raw_stdout():
    """Procedural must explicitly state digests are read first, raw stdout is fallback."""
    text = PROCEDURAL.read_text().lower()
    # Must have an order-implying word
    has_order = any(w in text for w in ("first", "before", "preferred", "fallback"))
    assert has_order, "must explicitly order: digests first, raw stdout fallback"
    # Must mention raw stdout as a fallback (not the primary read)
    assert "stdout" in text and ("fallback" in text or "only when" in text or "raw evidence" in text), \
        "must mark raw stdout as fallback evidence"


def test_procedural_lists_when_to_read_raw_stdout():
    """Must enumerate the cases where raw stdout still needs reading (verbatim, anomaly without line ref, mtime mismatch)."""
    text = PROCEDURAL.read_text().lower()
    # At least one of these markers should appear
    markers = ("verbatim", "quote", "line ref", "mtime", "stale", "anomaly")
    matches = [m for m in markers if m in text]
    assert len(matches) >= 2, \
        f"procedural should enumerate raw-stdout-fallback triggers; got matches: {matches}"


def test_procedural_references_per_cell_digest_path():
    """Per-cell digest path also referenced (data/digests/training/<cell_id>.md)."""
    text = PROCEDURAL.read_text()
    # Must reference the per-cell pattern (with <cell_id> placeholder OR a literal example like A42.md)
    has_pattern = "data/digests/training/" in text
    assert has_pattern
    # Must mention "per-cell" or specific cell file naming
    assert "per-cell" in text.lower() or ".md" in text or "cell" in text.lower()
