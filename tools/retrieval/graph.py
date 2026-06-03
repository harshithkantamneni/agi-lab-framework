"""Token graph: persistent SQLite store of canonical lab tokens + edges.

Nodes:
    decision      D-N entries (e.g. "D-420")
    carry_forward P-* tokens (e.g. "P-D417-ARTIFACT-PROJECTOR-DEDUP-GAP")
    program       Program identifiers (e.g. "program_2_dense_vs_moe_sub100m")
    phase         Phase IDs scoped per program (e.g. "Phase 3", "P11")
    role          Agent role names (e.g. "director", "pi", "lab_architect")
    file          Source markdown file paths
    chunk         Chunks from tools/lab_memory.py (linked by chunk id)

Edges (directed) — ACTUALLY EMITTED by build_graph_from_corpus:
    cites           decision header A references decision B (B not headed here)
    raises          decision header A co-occurs with carry-forward B
    in_program      decision belongs to program (file co-occurrence)
    in_file         decision/carry_forward lives in file

RESERVED (declared for a future richer extractor; NOT emitted yet — do not
filter on these expecting matches): precedes, resolves, dispatched_by, in_phase.
Tracked for a deliberate graph-semantics change, not this hardening pass.

Built by tools/retrieval/graph.py:build_graph_from_corpus which regex-parses
the lab's markdown corpus. No LLM extraction — token vocabulary is canonical.
"""
from __future__ import annotations
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


NODE_TYPES = {"decision", "carry_forward", "program", "phase", "role", "file", "chunk"}
# Edge types actually emitted by build_graph_from_corpus.
EDGE_TYPES = {"cites", "raises", "in_program", "in_file"}
# Declared but not yet emitted; kept as a roadmap for a future richer extractor.
RESERVED_EDGE_TYPES = {"precedes", "resolves", "dispatched_by", "in_phase"}


_DECISION_RE = re.compile(r"\bD-\d{1,4}\b")
_CARRY_FORWARD_RE = re.compile(r"\bP-[A-Z][A-Z0-9_]+(?:-[A-Z0-9_]+)*\b")
_PROGRAM_RE = re.compile(r"\bprogram_[0-9a-z_]+\b")
_PHASE_RE = re.compile(r"\b(?:Phase\s+\d+|P\d{1,3})\b")
_DECISION_HEADER_RE = re.compile(r"^(?:###\s+)?\*?\*?(D-\d{1,4})\b", re.MULTILINE)


def extract_decision_tokens(text: str) -> set[str]:
    return set(_DECISION_RE.findall(text))


def extract_carry_forward_tokens(text: str) -> set[str]:
    return set(_CARRY_FORWARD_RE.findall(text))


def extract_program_refs(text: str) -> set[str]:
    return set(_PROGRAM_RE.findall(text))


def extract_phase_refs(text: str) -> set[str]:
    return set(_PHASE_RE.findall(text))


def extract_role_refs(text: str, role_vocabulary: set[str] | None = None) -> set[str]:
    if role_vocabulary is None:
        role_vocabulary = _load_role_vocabulary()
    found = set()
    for role in role_vocabulary:
        if re.search(r"\b" + re.escape(role) + r"\b", text):
            found.add(role)
    return found


def extract_decision_headers(text: str) -> set[str]:
    return set(_DECISION_HEADER_RE.findall(text))


def _load_role_vocabulary() -> set[str]:
    import json
    candidates = [Path(__file__).resolve().parent.parent.parent / "data/agents/agents.json"]
    for p in candidates:
        if p.exists():
            try:
                return set(json.loads(p.read_text()).keys())
            except (OSError, json.JSONDecodeError):
                continue
    return {
        "pi", "director", "unanimous_compromise_mediator",
        "chief_scientist", "math_theorist", "experimental_methodologist",
        "hypothesis_generator", "mechanism_extractor", "measurement_theorist",
        "infrastructure_architect", "implementation_engineer_c", "sota_scout",
        "tooling_engineer", "reproducibility_engineer",
        "profiler", "kernel_specialist", "memory_optimizer",
        "scientific_reviewer", "statistical_reviewer", "red_team",
        "pre_reg_auditor", "code_reviewer",
        "literature_hunter", "paper_digester", "findings_curator",
        "paper_writer", "figure_generator",
        "lab_architect", "grant_reviewer", "evaluator",
    }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    label TEXT,
    program TEXT,
    phase TEXT,
    first_seen_ts INTEGER,
    last_seen_ts INTEGER,
    chunk_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    first_seen_ts INTEGER,
    PRIMARY KEY (src, dst, edge_type),
    FOREIGN KEY (src) REFERENCES nodes(id),
    FOREIGN KEY (dst) REFERENCES nodes(id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
CREATE INDEX IF NOT EXISTS idx_nodes_program ON nodes(program);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
"""


@dataclass
class TokenGraph:
    db_path: Path

    def init(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()


import time


def build_graph_from_corpus(g: TokenGraph, repo_root: Path) -> dict:
    """Walk the corpus, extract canonical tokens, write to the graph DB.

    Idempotent: re-running on the same corpus produces the same graph
    (INSERT OR IGNORE). Returns stats dict for telemetry.
    """
    stats = {"files_scanned": 0, "decisions": 0, "carry_forwards": 0,
             "programs": 0, "phases": 0, "roles": 0, "edges_cite": 0,
             "edges_raise": 0, "edges_in_program": 0}
    role_vocab = _load_role_vocabulary()
    now_ts = int(time.time())

    roots = [
        repo_root / "data/memories",
        repo_root / "programs",
        repo_root / "data/agents",
        repo_root / "data/engineering",
        repo_root / "docs",
    ]

    with sqlite3.connect(g.db_path) as conn:
        cur = conn.cursor()

        for root in roots:
            if not root.exists():
                continue
            repo_resolved = repo_root.resolve()
            for md_path in root.rglob("*.md"):
                if not md_path.is_file():
                    continue
                # Symlink containment: skip any .md that resolves outside the
                # repo. A symlinked file (or a symlinked directory in the tree)
                # could otherwise pull off-repo content into the searchable graph.
                try:
                    md_path.resolve().relative_to(repo_resolved)
                except (ValueError, OSError):
                    continue
                stats["files_scanned"] += 1
                try:
                    text = md_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue

                defined = extract_decision_headers(text)
                referenced = extract_decision_tokens(text)
                carry_forwards = extract_carry_forward_tokens(text)
                programs = extract_program_refs(text)
                phases = extract_phase_refs(text)
                roles = extract_role_refs(text, role_vocab)

                file_id = f"file:{md_path.relative_to(repo_root)}"
                _upsert_node(cur, file_id, "file", str(md_path.relative_to(repo_root)), now_ts)

                for d in defined:
                    _upsert_node(cur, d, "decision", d, now_ts)
                    stats["decisions"] += 1
                    _upsert_edge(cur, d, file_id, "in_file", now_ts)
                for d in referenced - defined:
                    _upsert_node(cur, d, "decision", d, now_ts)

                for cf in carry_forwards:
                    _upsert_node(cur, cf, "carry_forward", cf, now_ts)
                    stats["carry_forwards"] += 1
                    _upsert_edge(cur, cf, file_id, "in_file", now_ts)

                program_list = sorted(programs)
                for p in program_list:
                    _upsert_node(cur, p, "program", p, now_ts)
                    stats["programs"] += 1
                # in_program: link a decision to a program only when the file is
                # UNAMBIGUOUS (mentions exactly one program). A multi-program file
                # (e.g. a cross-program memory log) would otherwise fan out
                # |decisions| x |programs| spurious edges. (audit #3d)
                if len(program_list) == 1:
                    only = program_list[0]
                    for d in defined:
                        _upsert_edge(cur, d, only, "in_program", now_ts)
                        stats["edges_in_program"] += 1

                # phase nodes are qualified by program so 'Phase 3' in two
                # different programs do NOT collapse to one node (#3b); each phase
                # node also gets an in_file edge so it is not an orphan (#3c).
                # Populate the program/phase columns.
                for ph in phases:
                    for prog in (program_list or [None]):
                        node_id = f"phase:{prog}:{ph}" if prog else f"phase:{ph}"
                        _upsert_node(cur, node_id, "phase", ph, now_ts,
                                     program=prog, phase=ph)
                        _upsert_edge(cur, node_id, file_id, "in_file", now_ts)
                    stats["phases"] += 1

                # role nodes also get an in_file edge (non-orphan). (#3c)
                for r in roles:
                    node_id = f"role:{r}"
                    _upsert_node(cur, node_id, "role", r, now_ts)
                    _upsert_edge(cur, node_id, file_id, "in_file", now_ts)
                    stats["roles"] += 1

                # cites: a header points only at decisions referenced but NOT
                # themselves defined (headed) in this file. Two co-located headers
                # merely share a file — that is not a citation. (`referenced -
                # defined` also makes the old `src == dst` guard redundant.)
                for src in defined:
                    for dst in referenced - defined:
                        _upsert_edge(cur, src, dst, "cites", now_ts)
                        stats["edges_cite"] += 1

                for src in defined:
                    for cf in carry_forwards:
                        _upsert_edge(cur, src, cf, "raises", now_ts)
                        stats["edges_raise"] += 1

        conn.commit()
    return stats


def _upsert_node(cur, node_id: str, node_type: str, label: str, ts: int,
                 program: str | None = None, phase: str | None = None) -> None:
    cur.execute(
        """INSERT INTO nodes (id, type, label, program, phase, first_seen_ts, last_seen_ts)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
               last_seen_ts = excluded.last_seen_ts,
               program = COALESCE(nodes.program, excluded.program),
               phase = COALESCE(nodes.phase, excluded.phase)""",
        (node_id, node_type, label, program, phase, ts, ts),
    )


def _upsert_edge(cur, src: str, dst: str, edge_type: str, ts: int) -> None:
    cur.execute(
        """INSERT INTO edges (src, dst, edge_type, first_seen_ts)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(src, dst, edge_type) DO NOTHING""",
        (src, dst, edge_type, ts),
    )


import networkx as nx


def personalized_pagerank(
    g: TokenGraph,
    seeds: dict[str, float],
    top_k: int = 20,
    alpha: float = 0.85,
    max_iter: int = 100,
) -> list[tuple[str, float]]:
    """Run Personalized PageRank with given seed weights over the token graph.

    `seeds` maps node_id → personalization weight (normalized internally).
    Returns top_k nodes by PPR score, descending.

    Unknown seeds are silently dropped. If no known seeds remain, returns [].
    """
    nx_graph = _build_nx_graph(g)
    if nx_graph.number_of_nodes() == 0:
        return []

    known_seeds = {s: w for s, w in seeds.items() if s in nx_graph}
    if not known_seeds:
        return []

    total = sum(known_seeds.values())
    if total == 0:
        return []
    personalization = {n: known_seeds.get(n, 0.0) / total for n in nx_graph.nodes()}

    try:
        pr = nx.pagerank(
            nx_graph, alpha=alpha, personalization=personalization,
            max_iter=max_iter, tol=1e-6,
        )
    except nx.PowerIterationFailedConvergence:
        pr = nx.pagerank(nx_graph, alpha=alpha, personalization=personalization,
                         max_iter=max_iter, tol=1e-4)

    return sorted(pr.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def _build_nx_graph(g: TokenGraph) -> nx.DiGraph:
    nx_graph = nx.DiGraph()
    with sqlite3.connect(g.db_path) as conn:
        for row in conn.execute("SELECT id FROM nodes"):
            nx_graph.add_node(row[0])
        for src, dst, etype, weight in conn.execute(
            "SELECT src, dst, edge_type, weight FROM edges"
        ):
            if nx_graph.has_edge(src, dst):
                nx_graph[src][dst]["weight"] += weight
            else:
                nx_graph.add_edge(src, dst, weight=weight, edge_type=etype)
    return nx_graph


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Token graph build + query CLI.")
    parser.add_argument("--db", default="tools/lab_graph.db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build graph from corpus")
    p_build.add_argument("--db", default=argparse.SUPPRESS)
    p_build.add_argument("--root", default=".")

    p_ppr = sub.add_parser("ppr", help="Run Personalized PageRank query")
    p_ppr.add_argument("--db", default=argparse.SUPPRESS)
    p_ppr.add_argument("--seed", action="append", required=True,
                       help="seed node id (repeatable); equal weights")
    p_ppr.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args(argv)

    g = TokenGraph(Path(args.db))
    g.init()

    if args.cmd == "build":
        stats = build_graph_from_corpus(g, Path(args.root))
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return 0

    if args.cmd == "ppr":
        seeds = {s: 1.0 / len(args.seed) for s in args.seed}
        ranking = personalized_pagerank(g, seeds, top_k=args.top_k)
        for node_id, score in ranking:
            print(f"  [{score:.4f}] {node_id}")
        return 0

    return 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
