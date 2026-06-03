"""Unit tests for tools.retrieval.metadata.derive_metadata + ingest wiring.

derive_metadata is FILE-LEVEL: `text` must be the whole-file content, never a
single chunk. canonical_program_from_text returns '' on ambiguity (zero or 2+
whitelisted programs) and the SHORT id when exactly one whitelisted program is
named.
"""
from __future__ import annotations

import pytest

from tools.retrieval.metadata import canonical_program_from_text, derive_metadata


# ---- program_id from the programs/ directory (explicit table, not regex) ----

def test_program_2_short_id():
    m = derive_metadata("programs/program_2_dense_vs_moe_sub100m/x.md", "body")
    assert m["program_id"] == "program_2"


def test_program_3_brainstorm_keeps_full_name_not_program_3():
    m = derive_metadata("programs/program_3_brainstorm/x.md", "body")
    assert m["program_id"] == "program_3_brainstorm"
    assert m["program_id"] != "program_3"


def test_methodology_program_keeps_full_name():
    m = derive_metadata("programs/methodology_tiered_memory/x.md", "body")
    assert m["program_id"] == "methodology_tiered_memory"


# ---- role / deliverable from data/agents/<role>/ ----

def test_agent_director_semantic_role_and_memory_deliverable():
    m = derive_metadata("data/agents/director/semantic.md", "body")
    assert m["role"] == "director"
    assert m["deliverable_type"] == "memory"


def test_agent_shared_dir_not_in_role_vocab():
    m = derive_metadata("data/agents/_shared/x.md", "body")
    assert m["role"] == ""


# ---- legacy governance log conventions ----

def test_legacy_cycle_summary_program_role_deliverable():
    m = derive_metadata("legacy/cycles_1_31/summaries/cycle_5.md", "body")
    assert m["program_id"] == "legacy"
    assert m["role"] == "director"
    assert m["deliverable_type"] == "summary_cycle_5"


def test_legacy_org_retro_role_lab_architect():
    m = derive_metadata("legacy/cycles_1_31/org_retros/x.md", "body")
    assert m["program_id"] == "legacy"
    assert m["role"] == "lab_architect"


# ---- global single files ----

def test_global_shared_knowledge_program_role():
    m = derive_metadata("data/shared_knowledge.md", "body")
    assert m["program_id"] == "global"
    assert m["role"] == "findings_curator"


# ---- content-based program tagging (whole-file text) ----

def test_two_programs_in_text_is_ambiguous_empty():
    text = "program_1_opus47_on_18gb and also program_2_dense_vs_moe_sub100m"
    assert canonical_program_from_text(text) == ""
    m = derive_metadata("data/memories/log.md", text)
    assert m["program_id"] == ""


def test_exactly_one_whitelisted_program_in_text():
    text = "we focus on program_2_dense_vs_moe_sub100m for the dense baselines"
    assert canonical_program_from_text(text) == "program_2"
    m = derive_metadata("data/memories/log.md", text)
    assert m["program_id"] == "program_2"


# ---- phase derivation: anchored regex, no substring misfire ----

def test_phase_from_filename_anchored():
    m = derive_metadata("data/memories/agent_phase_3_close.md", "body")
    assert m["phase"] == "Phase 3"


def test_metaphase_does_not_misfire():
    m = derive_metadata("data/memories/metaphase_2.md", "body with no phase tokens")
    assert m["phase"] == ""


# ---- global file with cross-program text stays untagged ----

def test_global_file_cross_program_text_is_empty_program():
    text = (
        "mission spanning program_1_opus47_on_18gb and "
        "program_3_alt_grad_qat_100m across the lab"
    )
    m = derive_metadata("data/memories/mission.md", text)
    assert m["program_id"] == ""


# ---- ingestion wiring: ingest auto-derives metadata when caller omits it ----

def test_ingest_auto_tags_program_from_path(tmp_path, monkeypatch):
    """Ingesting a programs/program_2 file into a temp LabMemory should tag
    program_id='program_2' without the caller passing it. We stub the embedder
    so the test never loads a model / touches the network."""
    import numpy as np

    from tools import lab_memory
    from tools.lab_memory import LabMemory, EMBEDDING_DIM

    # Build a file whose RELATIVE path (as derive_metadata sees it) is a
    # programs/program_2 path. derive_metadata splits on '/', so we pass a
    # source_path string with that prefix while pointing the reader at a real
    # temp file via a tiny shim.
    prog_file = tmp_path / "x.md"
    prog_file.write_text("dense vs moe sub-100m experiment notes\n", encoding="utf-8")

    db = tmp_path / "tiny.db"
    lm = LabMemory(str(db))
    lm.init()

    # Stub the embedder: deterministic unit vectors, no model load.
    def _fake_embed(self, texts):
        return np.ones((len(texts), EMBEDDING_DIM), dtype="float32")

    monkeypatch.setattr(LabMemory, "_embed", _fake_embed, raising=True)

    # derive_metadata keys off the source_path string. Open the real temp file
    # for content but record the canonical relative path as source_path by
    # symlinking the expected directory structure under tmp_path.
    src_rel = "programs/program_2_dense_vs_moe_sub100m/x.md"
    nested = tmp_path / "programs" / "program_2_dense_vs_moe_sub100m"
    nested.mkdir(parents=True)
    (nested / "x.md").write_text(
        "dense vs moe sub-100m experiment notes\n", encoding="utf-8"
    )

    # Ingest using a path whose split yields the programs/program_2 prefix.
    # We chdir into tmp_path so the relative path both reads the file AND is
    # what derive_metadata sees.
    monkeypatch.chdir(tmp_path)
    n = lm.ingest(src_rel)
    assert n >= 1

    import sqlite3

    rows = sqlite3.connect(str(db)).execute(
        "SELECT DISTINCT program_id, deliverable_type FROM memories"
    ).fetchall()
    assert rows, "expected at least one ingested row"
    assert all(r[0] == "program_2" for r in rows)


def test_ingest_explicit_metadata_wins(tmp_path, monkeypatch):
    """A caller passing explicit non-empty metadata overrides derivation."""
    import numpy as np

    from tools.lab_memory import LabMemory, EMBEDDING_DIM

    src_rel = "programs/program_2_dense_vs_moe_sub100m/x.md"
    nested = tmp_path / "programs" / "program_2_dense_vs_moe_sub100m"
    nested.mkdir(parents=True)
    (nested / "x.md").write_text("body text\n", encoding="utf-8")

    db = tmp_path / "tiny2.db"
    lm = LabMemory(str(db))
    lm.init()
    monkeypatch.setattr(
        LabMemory,
        "_embed",
        lambda self, texts: np.ones((len(texts), EMBEDDING_DIM), dtype="float32"),
        raising=True,
    )
    monkeypatch.chdir(tmp_path)
    lm.ingest(src_rel, program_id="program_0", role="pi", phase="P9",
              deliverable_type="manual")

    import sqlite3

    rows = sqlite3.connect(str(db)).execute(
        "SELECT DISTINCT program_id, role, phase, deliverable_type FROM memories"
    ).fetchall()
    assert rows == [("program_0", "pi", "P9", "manual")]
