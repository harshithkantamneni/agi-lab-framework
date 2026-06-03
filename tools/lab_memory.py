#!/usr/bin/env python3
"""tools/lab_memory.py -- local semantic memory for the AGI research lab.

Replaces goodmem (cloud-hosted) with a fully local stack:
- SQLite (stdlib) for metadata
- sqlite-vec extension for vector storage + cosine similarity
- sentence-transformers/all-MiniLM-L6-v2 (~80MB, 384 dim) for embeddings

Primary API:
    lm = LabMemory("tools/lab_memory.db")
    lm.init()
    lm.ingest(path="programs/program_0_retrospective/findings.md",
              program_id="program_0", phase="P15", role="findings_curator",
              deliverable_type="findings")
    hits = lm.search("router entropy collapse", program_id=None, top_k=5)

See spec: docs/superpowers/specs/2026-04-17-scientific-research-lab-overhaul.md §7.1
"""
import argparse
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    import sqlite_vec
except ModuleNotFoundError as _err:
    # Common pitfall: invoked with system Python instead of the project venv.
    # sqlite-vec and sentence-transformers live in .venv/ (PEP 668 blocks a
    # system-Python pip install on macOS). See `make lab-memory-check` and
    # `make lab-memory-install` for the canonical invocation.
    sys.stderr.write(
        "\nERROR: sqlite_vec not found.\n"
        "lab_memory.py must run inside the project venv. Run via one of:\n"
        "    .venv/bin/python tools/lab_memory.py ...\n"
        "    make lab-memory-check      # smoke-test\n"
        "    make lab-memory-test       # full test suite\n"
        "    make lab-memory-install    # (re)install deps into .venv/\n"
        f"(interpreter in use: {sys.executable})\n\n"
    )
    raise

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking ---

CHUNK_SIZE = 800  # characters; roughly 150-200 tokens for MiniLM
CHUNK_OVERLAP = 100  # overlap so context at boundaries isn't cleaved


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> Iterator[str]:
    """Split text into overlapping chunks of roughly `size` chars.

    Prefers paragraph-boundary splits. Falls back to hard-sized chunks if
    paragraphs are enormous.
    """
    # First pass: split on blank-line paragraph boundaries.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) + 2 <= size:
            buf = f"{buf}\n\n{para}" if buf else para
        else:
            if buf:
                yield buf
            # If a single paragraph is too big, hard-chunk it with overlap.
            if len(para) > size:
                i = 0
                while i < len(para):
                    yield para[i:i + size]
                    i += size - overlap
                buf = ""
            else:
                buf = para
    if buf:
        yield buf


@dataclass
class Hit:
    """A single search result."""
    id: int
    program_id: str
    phase: str
    role: str
    deliverable_type: str
    source_path: str
    chunk_text: str
    timestamp: int
    distance: float  # cosine distance; smaller = more similar


class LabMemory:
    """Local semantic memory for the AGI lab.

    Thin wrapper over SQLite + sqlite-vec + sentence-transformers.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._model = None  # lazy-loaded

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def init(self) -> None:
        """Create the schema. Idempotent."""
        with self._conn() as conn:
            conn.executescript(f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    program_id TEXT NOT NULL DEFAULT '',
                    phase TEXT NOT NULL DEFAULT '',
                    role TEXT NOT NULL DEFAULT '',
                    deliverable_type TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    timestamp INTEGER NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(
                    embedding float[{EMBEDDING_DIM}]
                );
                CREATE INDEX IF NOT EXISTS idx_memories_program ON memories(program_id);
                CREATE INDEX IF NOT EXISTS idx_memories_role ON memories(role);
                CREATE INDEX IF NOT EXISTS idx_memories_phase ON memories(phase);
            """)
            conn.commit()

    def _get_model(self):
        """Lazy-load the sentence-transformers model (pinned revision for repro)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            from tools.retrieval.model_pins import pinned_revision
            self._model = SentenceTransformer(
                EMBEDDING_MODEL, revision=pinned_revision(EMBEDDING_MODEL)
            )
        return self._model

    def _embed(self, texts: list[str]):
        """Return an array of EMBEDDING_DIM-float normalized vectors."""
        model = self._get_model()
        return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def ingest(
        self,
        source_path: str,
        program_id: str = "",
        phase: str = "",
        role: str = "",
        deliverable_type: str = "",
    ) -> int:
        """Read a file, chunk it, embed chunks, and persist. Return chunk count.

        Metadata defaults to '' but is auto-derived from the whole-file text when
        the caller does not supply it, so ingest_incremental auto-tags. An
        explicit non-empty value from the caller always wins (backward compatible).
        """
        with open(source_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        chunks = list(chunk_text(text))
        if not chunks:
            return 0

        # Auto-derive metadata file-level when the caller left a field empty.
        # derive_metadata requires whole-file text, which is exactly `text`.
        if not (program_id and role and phase and deliverable_type):
            from tools.retrieval.metadata import derive_metadata
            derived = derive_metadata(source_path, text)
            program_id = program_id or derived["program_id"]
            role = role or derived["role"]
            phase = phase or derived["phase"]
            deliverable_type = deliverable_type or derived["deliverable_type"]

        embeddings = self._embed(chunks)
        now = int(time.time())

        with self._conn() as conn:
            cur = conn.cursor()
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    """INSERT INTO memories
                       (program_id, phase, role, deliverable_type, source_path,
                        chunk_text, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (program_id, phase, role, deliverable_type,
                     source_path, chunk, now),
                )
                rowid = cur.lastrowid
                cur.execute(
                    "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                    (rowid, emb.tobytes()),
                )
            conn.commit()

        return len(chunks)

    def _last_ingest_timestamp(self) -> int:
        """Return the max timestamp from the memories table (0 if empty)."""
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(timestamp) FROM memories").fetchone()
            return int(row[0]) if row and row[0] is not None else 0

    def ingest_incremental(self, roots: list) -> int:
        """Re-ingest *.md files under roots whose mtime > last ingest timestamp.

        Returns number of files (re-)ingested.
        """
        since = self._last_ingest_timestamp()
        count = 0
        for root in roots:
            root = Path(root)
            if not root.exists():
                continue
            for f in root.rglob("*.md"):
                if not f.is_file():
                    continue
                if int(f.stat().st_mtime) <= since:
                    continue
                self.ingest(str(f))
                count += 1
        return count

    def search(
        self,
        query: str,
        program_id: str | None = None,
        role: str | None = None,
        phase: str | None = None,
        deliverable_type: str | None = None,
        top_k: int = 10,
    ) -> list[Hit]:
        """Semantic search. Optional metadata filters."""
        if not query.strip():
            return []

        q_emb = self._embed([query])[0]

        # Build WHERE clause for metadata filters.
        where_clauses: list[str] = []
        params: list = []
        if program_id is not None:
            where_clauses.append("m.program_id = ?")
            params.append(program_id)
        if role is not None:
            where_clauses.append("m.role = ?")
            params.append(role)
        if phase is not None:
            where_clauses.append("m.phase = ?")
            params.append(phase)
        if deliverable_type is not None:
            where_clauses.append("m.deliverable_type = ?")
            params.append(deliverable_type)
        where_sql = (" AND " + " AND ".join(where_clauses)) if where_clauses else ""

        sql = f"""
            SELECT m.id, m.program_id, m.phase, m.role, m.deliverable_type,
                   m.source_path, m.chunk_text, m.timestamp, v.distance
            FROM memories_vec v
            JOIN memories m ON m.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?{where_sql}
            ORDER BY v.distance
        """
        params_full = [q_emb.tobytes(), top_k * 4] + params  # over-fetch, then trim

        with self._conn() as conn:
            rows = conn.execute(sql, params_full).fetchall()

        return [
            Hit(id=r[0], program_id=r[1], phase=r[2], role=r[3],
                deliverable_type=r[4], source_path=r[5], chunk_text=r[6],
                timestamp=r[7], distance=r[8])
            for r in rows[:top_k]
        ]

    def list(
        self,
        program_id: str | None = None,
        role: str | None = None,
        phase: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List records (metadata only, no vectors)."""
        where_clauses: list[str] = []
        params: list = []
        if program_id is not None:
            where_clauses.append("program_id = ?")
            params.append(program_id)
        if role is not None:
            where_clauses.append("role = ?")
            params.append(role)
        if phase is not None:
            where_clauses.append("phase = ?")
            params.append(phase)
        where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""SELECT id, program_id, phase, role, deliverable_type,
                         source_path, chunk_text, timestamp
                  FROM memories{where_sql}
                  ORDER BY timestamp DESC LIMIT ?"""
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        cols = ["id", "program_id", "phase", "role", "deliverable_type",
                "source_path", "chunk_text", "timestamp"]
        return [dict(zip(cols, r)) for r in rows]

    def get(self, record_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT id, program_id, phase, role, deliverable_type,
                          source_path, chunk_text, timestamp
                   FROM memories WHERE id = ?""",
                (record_id,),
            ).fetchone()
        if row is None:
            return None
        cols = ["id", "program_id", "phase", "role", "deliverable_type",
                "source_path", "chunk_text", "timestamp"]
        return dict(zip(cols, row))

    def delete(self, record_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (record_id,))
            conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (record_id,))
            conn.commit()
            return cur.rowcount > 0


# --- CLI ---

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_memory.db")


def _cmd_init(args):
    lm = LabMemory(args.db)
    lm.init()
    print(f"Initialized {args.db}")


def _cmd_ingest(args):
    lm = LabMemory(args.db)
    lm.init()  # idempotent
    if args.incremental:
        roots = args.roots or ["data/memories", "programs", "data", "docs"]
        count = lm.ingest_incremental([Path(r) for r in roots])
        print(f"ingested {count} files (incremental)")
        return
    if args.path is None:
        raise SystemExit("ingest: path required unless --incremental is set")
    n = lm.ingest(
        args.path,
        program_id=args.program or "",
        phase=args.phase or "",
        role=args.role or "",
        deliverable_type=args.type or "",
    )
    print(f"Ingested {n} chunks from {args.path}")


def _cmd_search(args):
    """Search the lab's semantic memory.

    Default: 4-layer hybrid (graph + bm25 + dense + rerank) via tools.retrieval.search.
    Use --legacy for the old pure-vector path.
    """
    if getattr(args, "legacy", False):
        lm = LabMemory(args.db)
        hits = lm.search(
            args.query,
            program_id=args.program, role=args.role, phase=args.phase,
            deliverable_type=args.type, top_k=args.top_k,
        )
        for h in hits:
            print(f"[{h.distance:.3f}] {h.program_id}/{h.phase}/{h.role} — {h.source_path}")
            snippet = h.chunk_text[:200].replace("\n", " ")
            print(f"    {snippet}{'...' if len(h.chunk_text) > 200 else ''}")
            print()
        return

    import os
    from pathlib import Path
    repo_root = Path(args.db).resolve().parent.parent
    # When invoked as `python tools/lab_memory.py`, the `tools` package is not
    # on sys.path. Add repo root so `from tools.retrieval...` resolves both as a
    # script and as an imported module.
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    use_graph = not getattr(args, "no_graph", False)
    use_rerank = not getattr(args, "no_rerank", False)

    def _print(results):
        for r in results:
            score = r.get("rerank_score") or r.get("rrf_score") or 0.0
            snip = r.get("chunk_text", "")[:200].replace("\n", " ")
            print(f"[{score:.3f}] {r.get('source_path', '?')} — "
                  f"{snip}{'...' if len(r.get('chunk_text', '')) > 200 else ''}")
            print()

    def _inprocess():
        # The original committed path: build everything in-process. Correct but
        # cold (loads the embedder + reranker per invocation).
        from tools.retrieval.search import search
        _print(search(
            query=args.query, repo_root=repo_root, top_k=args.top_k,
            program=args.program, role=args.role, phase=args.phase,
            use_graph=use_graph, use_rerank=use_rerank,
        ))

    # Try the warm persistent server first; fall back to in-process on ANY
    # failure (server down/slow/broken) so the lab never hangs or errors.
    if getattr(args, "no_server", False):
        _inprocess()
        return
    try:
        from tools.retrieval.client import query_server, ServerUnavailable
        results = query_server(
            query=args.query, repo=repo_root, top_k=args.top_k,
            program=args.program, role=args.role, phase=args.phase,
            use_graph=use_graph, use_rerank=use_rerank,
            connect_timeout=float(os.environ.get("RETRIEVAL_CONNECT_TIMEOUT", "0.5")),
            read_timeout=float(os.environ.get("RETRIEVAL_READ_TIMEOUT", "20")),
        )
        _print(results)
    except Exception as e:  # ServerUnavailable + any import/parse issue
        sys.stderr.write(f"retrieval server unavailable: {e}; in-process fallback\n")
        _inprocess()


def _cmd_list(args):
    lm = LabMemory(args.db)
    rows = lm.list(
        program_id=args.program,
        role=args.role,
        phase=args.phase,
        limit=args.limit,
    )
    for r in rows:
        print(f"#{r['id']} {r['program_id']}/{r['phase']}/{r['role']} "
              f"({r['deliverable_type']}) — {r['source_path']}")


def _cmd_delete(args):
    lm = LabMemory(args.db)
    ok = lm.delete(args.id)
    print("deleted" if ok else "not found")


def main():
    parser = argparse.ArgumentParser(description="Local semantic memory for the AGI lab.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="Initialize schema")
    p_init.set_defaults(func=_cmd_init)

    p_ing = sub.add_parser("ingest", help="Ingest a file")
    p_ing.add_argument("path", nargs="?", default=None)
    p_ing.add_argument("--program", default="")
    p_ing.add_argument("--phase", default="")
    p_ing.add_argument("--role", default="")
    p_ing.add_argument("--type", default="")
    p_ing.add_argument("--incremental", action="store_true",
                       help="Re-ingest files modified since last ingest (overrides positional path)")
    p_ing.add_argument("--roots", nargs="+", default=None,
                       help="Roots to scan for --incremental mode (default: data/memories programs data docs)")
    p_ing.set_defaults(func=_cmd_ingest)

    p_srch = sub.add_parser("search", help="Semantic search")
    p_srch.add_argument("query")
    p_srch.add_argument("--program", default=None)
    p_srch.add_argument("--role", default=None)
    p_srch.add_argument("--phase", default=None)
    p_srch.add_argument("--type", default=None)
    p_srch.add_argument("--top-k", type=int, default=10)
    p_srch.add_argument("--legacy", action="store_true",
                        help="Use the pre-2026-05-20 pure-vector path (ablation)")
    p_srch.add_argument("--no-graph", action="store_true",
                        help="Skip L0 token graph seed expansion")
    p_srch.add_argument("--no-rerank", action="store_true",
                        help="Skip L4 cross-encoder reranking (~10x faster)")
    p_srch.add_argument("--no-server", action="store_true",
                        help="Force in-process retrieval, skip the persistent server")
    p_srch.set_defaults(func=_cmd_search)

    p_list = sub.add_parser("list", help="List records")
    p_list.add_argument("--program", default=None)
    p_list.add_argument("--role", default=None)
    p_list.add_argument("--phase", default=None)
    p_list.add_argument("--limit", type=int, default=100)
    p_list.set_defaults(func=_cmd_list)

    p_del = sub.add_parser("delete", help="Delete a record by id")
    p_del.add_argument("id", type=int)
    p_del.set_defaults(func=_cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
