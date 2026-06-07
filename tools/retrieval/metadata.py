"""Single-valued metadata derivation for the dense/bm25 retrieval columns.

Backfills/derives memories.{program_id, role, phase, deliverable_type} so scoped
retrieval (search --program/--role/--phase, exact match) actually works. Reuses
tools/retrieval/graph.py extractors so this column agrees with the token graph's
canonical vocabulary, mapping graph's FULL program tokens to the SHORT ids the
search layer filters on.

Design notes (see docs/superpowers/specs + the design+red-team workflow):
  * program_id: SHORT form for numbered programs (program_0..3), matching the
    1,566 already-tagged rows (search is exact-match). program_3_brainstorm and
    the methodology_* programs keep their full directory name; the program_3 vs
    program_3_brainstorm collision is avoided by deriving from the directory name
    via an explicit table, never a `program_(\\d+)` regex.
  * Content-based program tagging uses a WHITELIST of the 5 real full program
    tokens, so graph noise (program_open_memo, program_complete, program_id, ...)
    is ignored, and assigns a program ONLY when the WHOLE-FILE text names exactly
    one real program. Callers MUST pass whole-file text (all chunks of a
    source_path), never a single chunk — otherwise a chronological log slice that
    mentions one program in passing gets mis-tagged.
  * role = the agent role the content belongs to, derivable from path
    (data/agents/<role>/ and the legacy governance dirs). Program-directory
    research is NOT given a producer-role tag (it is not a single agent's output
    and is not consistently recoverable). Those rows still carry program_id, so
    they remain discoverable by program filter.
  * phase is a best-effort weak signal only.
"""
from __future__ import annotations
import os
import re

from tools.retrieval.graph import (
    extract_program_refs,
    extract_phase_refs,
    _load_role_vocabulary,
)

# programs/<dir> -> canonical program_id (SHORT for numbered programs).
_PROGRAM_DIR_TO_ID = {
    "program_0_retrospective": "program_0",
    "program_1_example": "program_1",
    "program_2_example": "program_2",
    "program_3_example": "program_3",
    "program_3_brainstorm": "program_3_brainstorm",
    "methodology_structural_anti_forgery": "methodology_structural_anti_forgery",
    "methodology_tiered_memory": "methodology_tiered_memory",
}

# Whitelist: only these FULL tokens emitted by extract_program_refs are real
# programs. Everything else it matches (program_open_memo, program_complete,
# program_id, program_closure_*, program_4_, program_5_*, ...) is noise.
_FULL_TOKEN_TO_ID = {
    "program_0_retrospective": "program_0",
    "program_1_example": "program_1",
    "program_2_example": "program_2",
    "program_3_example": "program_3",
    "program_3_brainstorm": "program_3_brainstorm",
}

_ROLE_VOCAB = _load_role_vocabulary()

# Anchored on a leading boundary so 'metaphase_2' / 'paraphrase_9' do NOT match.
_PHASE_PATH_RE = re.compile(r"(?:^|[_\s/-])phase[_\s]?(\d{1,3})", re.IGNORECASE)

_GLOBAL_SINGLE_FILES = {
    "data/shared_knowledge.md",
    "data/killed_ideas.md",
    "data/bibliography.md",
}


def canonical_program_from_text(text: str) -> str:
    """Return a program_id only when WHOLE-FILE text names exactly one real
    (whitelisted) program. Conservative by design — '' is the honest answer for
    global / cross-program files. Pass whole-file text, never a single chunk."""
    canon = {
        _FULL_TOKEN_TO_ID[r]
        for r in extract_program_refs(text)
        if r in _FULL_TOKEN_TO_ID
    }
    return next(iter(canon)) if len(canon) == 1 else ""


def _derive_phase(fname: str, text: str) -> str:
    """Best-effort weak signal: filename 'phase_N'/'phaseN' -> 'Phase N', else
    exactly one extract_phase_refs hit, else ''."""
    m = _PHASE_PATH_RE.search(fname)
    if m:
        return f"Phase {int(m.group(1))}"
    phs = extract_phase_refs(text)
    return next(iter(phs)) if len(phs) == 1 else ""


def derive_metadata(source_path: str, text: str) -> dict:
    """Derive {program_id, role, phase, deliverable_type} for ONE source file.

    `text` MUST be the whole-file content (all of a source_path's chunks), not a
    single chunk. Every field defaults to '' (the column default) when unknown.
    """
    program_id = role = phase = deliverable_type = ""
    parts = source_path.split("/")
    fname = parts[-1] if parts else source_path
    stem = os.path.splitext(fname)[0]
    parent = parts[-2] if len(parts) >= 2 else ""

    if len(parts) >= 2 and parts[0] == "programs":
        program_id = _PROGRAM_DIR_TO_ID.get(parts[1], "")
        if stem == "decisions_archive":
            deliverable_type = "decisions"
        elif parent == "summaries" and stem.startswith("cycle_"):
            deliverable_type = f"summary_{stem}"
        elif parent == "org_retros":
            deliverable_type = "org_retro"
        else:
            deliverable_type = stem
        # programs/ research is not a single agent's output -> no producer role.

    elif len(parts) >= 2 and parts[0] == "legacy":
        program_id = "legacy"
        # Role-by-convention for the legacy governance log (matches the original
        # tagger: director for the decisions/summaries, lab_architect for retros).
        if stem == "decisions_archive":
            deliverable_type, role = "decisions", "director"
        elif parent == "summaries" and stem.startswith("cycle_"):
            deliverable_type, role = f"summary_{stem}", "director"
        elif parent == "org_retros":
            deliverable_type, role = "org_retro", "lab_architect"
        elif stem == "rolling_summary":
            deliverable_type, role = "rolling_summary", "director"
        else:
            deliverable_type = stem

    elif len(parts) >= 3 and parts[0] == "data" and parts[1] == "agents":
        seg = parts[2]  # role dir, or _shared / a stray top-level file
        if seg in _ROLE_VOCAB:
            role = seg
        deliverable_type = "memory" if stem in ("semantic", "procedural") else stem
        program_id = canonical_program_from_text(text)

    elif source_path in _GLOBAL_SINGLE_FILES:
        program_id, role, deliverable_type = "global", "findings_curator", stem

    else:
        # data/memories, data/archives, data/engineering, docs/**, misc single
        # files: global / cross-program. Whole-file single-program tag only.
        program_id = canonical_program_from_text(text)

    if not phase:
        phase = _derive_phase(fname, text)

    return {
        "program_id": program_id,
        "role": role,
        "phase": phase,
        "deliverable_type": deliverable_type,
    }
