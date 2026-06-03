"""Token graph: schema initialization."""
import sqlite3
from pathlib import Path


def test_init_creates_tables(tmp_path):
    from tools.retrieval.graph import TokenGraph
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    with sqlite3.connect(db) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "nodes" in tables
    assert "edges" in tables


def test_init_is_idempotent(tmp_path):
    from tools.retrieval.graph import TokenGraph
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    g.init()  # should not raise


def test_node_types_constraint():
    from tools.retrieval.graph import NODE_TYPES
    assert NODE_TYPES == {"decision", "carry_forward", "program", "phase", "role", "file", "chunk"}


def test_extract_decision_tokens_simple():
    from tools.retrieval.graph import extract_decision_tokens
    text = "Recall D-420 (closing-out) and D-87 (older). Also see D-114."
    tokens = extract_decision_tokens(text)
    assert tokens == {"D-420", "D-87", "D-114"}


def test_extract_decision_tokens_with_heading():
    from tools.retrieval.graph import extract_decision_tokens
    text = "### D-432 (2026-05-19, ~21:43 UTC): **headline**\n\nBody references D-87."
    tokens = extract_decision_tokens(text)
    assert tokens == {"D-432", "D-87"}


def test_extract_carry_forward_tokens():
    from tools.retrieval.graph import extract_carry_forward_tokens
    text = "Raises P-D417-ARTIFACT-PROJECTOR-DEDUP-GAP and P-D360-BRIEF-STATE."
    tokens = extract_carry_forward_tokens(text)
    assert "P-D417-ARTIFACT-PROJECTOR-DEDUP-GAP" in tokens
    assert "P-D360-BRIEF-STATE" in tokens


def test_extract_program_phase_role():
    from tools.retrieval.graph import extract_program_refs, extract_phase_refs, extract_role_refs
    text = (
        "In program_2_dense_vs_moe_sub100m Phase 3 P11, the director dispatched "
        "findings_curator and lab_architect."
    )
    progs = extract_program_refs(text)
    assert "program_2_dense_vs_moe_sub100m" in progs
    phases = extract_phase_refs(text)
    assert any("Phase 3" in p or "P11" in p for p in phases)
    roles = extract_role_refs(text)
    assert "director" in roles
    assert "findings_curator" in roles


def test_extract_decision_headers():
    from tools.retrieval.graph import extract_decision_headers
    text = (
        "### D-420 (2026-05-19, ~10:30 UTC): **P15 SECOND STAGE LANDED**\n\n"
        "Body cites D-419 and D-87.\n\n"
        "### D-421 (2026-05-19, ~13:00 UTC): **PROGRAM 2 CLOSED**\n"
    )
    headers = extract_decision_headers(text)
    assert headers == {"D-420", "D-421"}


def test_build_graph_from_corpus_extracts_decisions(tmp_path):
    from tools.retrieval.graph import TokenGraph, build_graph_from_corpus
    import sqlite3
    fixture = Path(__file__).parent / "fixtures/sample_corpus"
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    stats = build_graph_from_corpus(g, fixture)

    assert stats["decisions"] >= 3
    assert stats["carry_forwards"] >= 2

    with sqlite3.connect(db) as conn:
        decision_ids = {r[0] for r in conn.execute(
            "SELECT id FROM nodes WHERE type='decision'"
        )}
    assert "D-432" in decision_ids
    assert "D-431" in decision_ids
    assert "D-420" in decision_ids


def test_build_graph_extracts_cites_edge(tmp_path):
    from tools.retrieval.graph import TokenGraph, build_graph_from_corpus
    import sqlite3
    fixture = Path(__file__).parent / "fixtures/sample_corpus"
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    build_graph_from_corpus(g, fixture)
    with sqlite3.connect(db) as conn:
        edges = [r for r in conn.execute(
            "SELECT src, dst, edge_type FROM edges WHERE src='D-431' AND edge_type='cites'"
        )]
    cited = {dst for src, dst, et in edges}
    assert "D-87" in cited
    assert "D-114" in cited


def test_build_graph_does_not_fabricate_header_to_header_cites(tmp_path):
    """A decision header must not 'cite' another decision that is itself a header
    defined in the same file. Two co-located headers (D-432, D-431, D-420 in the
    fixture) merely share a file; that is not a citation. cites should only point
    at decisions referenced-but-not-defined-here (true outbound references)."""
    from tools.retrieval.graph import TokenGraph, build_graph_from_corpus
    import sqlite3
    fixture = Path(__file__).parent / "fixtures/sample_corpus"
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    build_graph_from_corpus(g, fixture)
    with sqlite3.connect(db) as conn:
        cites = {(src, dst) for src, dst in conn.execute(
            "SELECT src, dst FROM edges WHERE edge_type='cites'"
        )}
    defined = {"D-432", "D-431", "D-420"}  # the three headers in the fixture log.md
    fabricated = {(s, d) for (s, d) in cites if s in defined and d in defined}
    assert not fabricated, f"header-to-header cites fabricated: {fabricated}"
    # Real outbound citations to referenced-not-defined decisions still exist.
    assert ("D-431", "D-87") in cites


def test_build_graph_is_idempotent(tmp_path):
    """Running build twice must not double-insert edges/nodes."""
    from tools.retrieval.graph import TokenGraph, build_graph_from_corpus
    import sqlite3
    fixture = Path(__file__).parent / "fixtures/sample_corpus"
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    build_graph_from_corpus(g, fixture)
    with sqlite3.connect(db) as conn:
        n1 = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        e1 = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    build_graph_from_corpus(g, fixture)
    with sqlite3.connect(db) as conn:
        n2 = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
        e2 = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    assert n1 == n2, f"nodes doubled: {n1} → {n2}"
    assert e1 == e2, f"edges doubled: {e1} → {e2}"


def _build_corpus(tmp_path, files):
    """Write {relpath-under-data/memories: content} into a tmp corpus and build."""
    from tools.retrieval.graph import TokenGraph, build_graph_from_corpus
    root = tmp_path / "corpus"
    for rel, content in files.items():
        p = root / "data" / "memories" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    db = tmp_path / "g.db"
    g = TokenGraph(db)
    g.init()
    build_graph_from_corpus(g, root)
    return db


def test_phase_nodes_qualified_by_program_no_collision(tmp_path):
    """'Phase 3' in two DIFFERENT programs must not collapse to one node (#3b)."""
    import sqlite3
    db = _build_corpus(tmp_path, {
        "a.md": "### D-1 (x): body\n\nIn program_2_dense_vs_moe_sub100m Phase 3.\n",
        "b.md": "### D-2 (x): body\n\nIn program_3_alt_grad_qat_100m Phase 3.\n",
    })
    with sqlite3.connect(db) as conn:
        pids = {r[0] for r in conn.execute("SELECT id FROM nodes WHERE type='phase'")}
        cols = conn.execute(
            "SELECT program, phase FROM nodes WHERE type='phase'").fetchall()
    assert "phase:program_2_dense_vs_moe_sub100m:Phase 3" in pids
    assert "phase:program_3_alt_grad_qat_100m:Phase 3" in pids
    assert "phase:Phase 3" not in pids, "phase node collapsed across programs"
    assert all(prog and ph for prog, ph in cols), "program/phase columns not populated"


def test_phase_and_role_nodes_are_not_orphans(tmp_path):
    """phase/role nodes must have >=1 edge (in_file) — orphan nodes can't diffuse
    PPR mass and are dead weight (#3c)."""
    import sqlite3
    db = _build_corpus(tmp_path, {
        "a.md": "### D-1 (x): body\n\nprogram_2_dense_vs_moe_sub100m Phase 3; "
                "the director dispatched findings_curator.\n",
    })
    with sqlite3.connect(db) as conn:
        for ntype in ("phase", "role"):
            for (nid,) in conn.execute("SELECT id FROM nodes WHERE type=?", (ntype,)):
                deg = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE src=? OR dst=?", (nid, nid)
                ).fetchone()[0]
                assert deg > 0, f"orphan {ntype} node with no edges: {nid}"


def test_in_program_only_on_single_program_files(tmp_path):
    """A multi-program file must NOT fan out in_program edges (ambiguous); a
    single-program file links its decisions to that one program (#3d)."""
    import sqlite3
    db = _build_corpus(tmp_path, {
        "multi.md": "### D-10 (x): body\n\nMentions program_2_dense_vs_moe_sub100m "
                    "and program_3_alt_grad_qat_100m.\n",
        "single.md": "### D-20 (x): body\n\nOnly program_3_alt_grad_qat_100m here.\n",
    })
    with sqlite3.connect(db) as conn:
        in_prog = {(s, d) for s, d in conn.execute(
            "SELECT src, dst FROM edges WHERE edge_type='in_program'")}
    assert not any(s == "D-10" for s, d in in_prog), f"fan-out on multi-program file: {in_prog}"
    assert ("D-20", "program_3_alt_grad_qat_100m") in in_prog


def test_personalized_pagerank_basic(tmp_path):
    """Seed=D-420 should rank D-419 (cites) and P-D420-DEDUP (raises) high."""
    from tools.retrieval.graph import TokenGraph, build_graph_from_corpus, personalized_pagerank
    fixture = Path(__file__).parent / "fixtures/sample_corpus"
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    build_graph_from_corpus(g, fixture)
    ranking = personalized_pagerank(g, seeds={"D-420": 1.0}, top_k=10)
    assert isinstance(ranking, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in ranking)
    top_ids = [n for n, _ in ranking[:3]]
    assert "D-420" in top_ids
    assert any("P-D420" in n for n, _ in ranking)


def test_personalized_pagerank_multi_seed(tmp_path):
    from tools.retrieval.graph import TokenGraph, build_graph_from_corpus, personalized_pagerank
    fixture = Path(__file__).parent / "fixtures/sample_corpus"
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    build_graph_from_corpus(g, fixture)
    ranking = personalized_pagerank(g, seeds={"D-420": 0.5, "D-432": 0.5}, top_k=10)
    top_ids = [n for n, _ in ranking[:5]]
    assert "D-420" in top_ids
    assert "D-432" in top_ids


def test_personalized_pagerank_handles_unknown_seed(tmp_path):
    from tools.retrieval.graph import TokenGraph, build_graph_from_corpus, personalized_pagerank
    fixture = Path(__file__).parent / "fixtures/sample_corpus"
    db = tmp_path / "graph.db"
    g = TokenGraph(db)
    g.init()
    build_graph_from_corpus(g, fixture)
    ranking = personalized_pagerank(g, seeds={"D-NONEXISTENT": 1.0}, top_k=10)
    assert isinstance(ranking, list)
