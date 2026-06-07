# Memory Retrieval Upgrade — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the lab's flat-vector semantic search to a 4-layer hybrid retrieval system (token graph + BM25 + dense + reranker) anchored on the lab's canonical token vocabulary (D-N decisions, P-* carry-forwards, programs, phases, roles). Land two orthogonal pain fixes (brief staleness, concurrency timeouts) along the way. Keep the existing `tools/lab_memory.py` API working as a strict subset of the new system so adoption is incremental.

**Architecture:** Additive, not replacement. New code lives in `tools/retrieval/` package. Existing `tools/lab_memory.py` becomes a thin shim that calls the new system; old CLI invocations continue working. Each layer is independently testable. Persistent storage in SQLite + JSON (no pickle for untrusted-content safety; no Neo4j; regex-derived token graph, no LLM extraction).

**Tech Stack:** Python 3.14 + sqlite + sqlite-vec (existing) + sentence-transformers (existing) + networkx (existing for PPR) + rank_bm25 (new, ~50 KB) + bge-reranker-v2-m3 (~568 MB cross-encoder via HF).

**Methodology contribution:** The lab's D-N / P-* canonical-token discipline is genuinely ahead of what's published in agent-memory research (per 2026-05-20 SOTA synthesis). Implementing L0 (token graph + PPR) properly opens the door to a future methodology paper on "agent-lab memory via canonical-token graphs". Keep the implementation clean, documented, and reproducible to preserve that option.

**Scope discipline:**
- ✅ Build: L0 token graph, L2 BM25, L3 RRF fusion, L4 cross-encoder reranker
- ✅ Build: concurrency isolation for lab_memory.py, brief-staleness fix
- ❌ Do NOT build: entity extraction via LLM (no need — tokens are already canonical)
- ❌ Do NOT build: community detection / Leiden clustering (premature)
- ❌ Do NOT migrate to Neo4j (sqlite is fine)
- ❌ Do NOT build reflection / community summarization (closure memos already do this)
- ❌ Do NOT rip out anything currently working
- ❌ Do NOT use pickle for index persistence (security-hook flagged; use JSON + rebuild)

---

## File Structure

New package — keeps responsibilities separated, each module testable independently:

```
tools/retrieval/
├── __init__.py              # Re-exports search() top-level API
├── graph.py                 # L0: token graph + Personalized PageRank
├── bm25.py                  # L2: BM25 sparse retrieval over chunks
├── hybrid.py                # L3: RRF fusion of BM25 + dense
├── rerank.py                # L4: cross-encoder reranker
├── search.py                # Orchestrator — runs L0/L1/L2/L3/L4 in sequence
└── concurrency.py           # Subprocess-isolated query worker

tools/lab_memory.py          # Existing — becomes thin shim around tools/retrieval/

tests/retrieval/
├── __init__.py
├── test_graph.py
├── test_bm25.py
├── test_hybrid.py
├── test_rerank.py
├── test_search.py
├── test_concurrency.py
└── fixtures/                # Small test corpus
```

**Existing files that get touched (modify, don't rewrite):**
- `tools/lab_memory.py` — wrap new system; existing API preserved
- `tools/brief_assembler.py` — root-cause brief-staleness fix (Task 0.2)
- `run_agi_lab.sh` — add `_run_retrieval_reindex` next to `_run_lab_memory_reindex`
- `data/agents/director/procedural.md` — surface new search capabilities in dispatch CCA section
- `.venv/` — add `rank_bm25`

---

## Stage 0 — Foundations (test infra + small fixes)

Three tasks. Lands prerequisites: pip deps, test fixtures, concurrency fix, brief-staleness fix. Each is independently shippable.

### Task 0.1: Add `rank_bm25` to `.venv` + verify all deps importable

**Files:**
- Modify: `Makefile` (add to `lab-memory-install` target if it exists, else create)
- Create: `tools/retrieval/__init__.py` (empty package marker)

- [ ] **Step 1: Install rank_bm25 into .venv**

```bash
.venv/bin/pip install 'rank_bm25==0.2.2'
```

- [ ] **Step 2: Verify importable**

```bash
.venv/bin/python -c "
import rank_bm25
import networkx as nx
import sentence_transformers
import sqlite_vec
print('rank_bm25', rank_bm25.__version__)
print('networkx', nx.__version__)
print('sentence_transformers', sentence_transformers.__version__)
print('sqlite_vec OK')
"
```

Expected: 4 lines, all version numbers print, no ImportError.

- [ ] **Step 3: Pin in Makefile**

Find the existing `lab-memory-install` target in `Makefile`. Add `rank_bm25==0.2.2` to the pip install line. If no such target exists, create one.

- [ ] **Step 4: Create package markers**

```bash
mkdir -p tools/retrieval tests/retrieval tests/retrieval/fixtures
touch tools/retrieval/__init__.py tests/retrieval/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add Makefile tools/retrieval/__init__.py tests/retrieval/__init__.py
git commit -m "feat(retrieval): scaffold tools/retrieval/ package + add rank_bm25 dep"
```

---

### Task 0.2: Brief-staleness root-cause fix (`tools/brief_assembler.py`)

**Context:** P-D427-BRIEF-LOG-CONSISTENCY-GAP fired across D-405..D-419 (11+ sessions). Brief showed `Last decision: D-358` while reality was D-414..D-419. Root cause (per phase3 retro §3.2): `_extract_last_n_decisions` regex parses correctly but the brief's `last_decision` field uses a separate stale path. Need to trace + fix.

**Files:**
- Modify: `tools/brief_assembler.py`
- Test: `tests/test_brief_assembler.py` (existing — add new test cases)

- [ ] **Step 1: Read the existing brief_assembler to identify the actual stale code path**

Read `tools/brief_assembler.py` end-to-end. Look for where `Last decision:` gets written into the brief. Trace back to its data source. There are two candidate sources:

a) `_extract_last_n_decisions(repo_root, n=5)` — returns last N decisions from log.md head
b) A separate `last_decision` variable populated from another file (state.md? session_state.md? a cache?)

Determine which one is stale. If both exist, the stale one is the bug.

- [ ] **Step 2: Write the failing test**

In `tests/test_brief_assembler.py`, add:

```python
def test_last_decision_matches_log_md_head(tmp_path, monkeypatch):
    """The `Last decision` field in the brief must reflect log.md head,
    not a cached/stale snapshot. Regression test for P-D427."""
    repo = tmp_path / "repo"
    (repo / "data/memories").mkdir(parents=True)

    log_md = repo / "data/memories/log.md"
    log_md.write_text(
        "Log tier — recent decisions.\n\n---\n\n"
        "### D-999 (2026-05-20, ~14:00 UTC): **FRESHEST DECISION — should appear in brief.**\n"
        "Lorem ipsum.\n\n"
        "### D-998 (2026-05-20, ~13:00 UTC): **Older decision.**\n"
    )
    (repo / "data/memories/current.md").write_text(
        "## Current Program: foo\nCurrent Phase: bar\n"
    )

    from tools.brief_assembler import _extract_last_n_decisions
    out = _extract_last_n_decisions(repo, n=2)
    assert "D-999" in out, f"expected D-999 in last-decisions, got: {out!r}"
    assert "FRESHEST" in out, f"expected newest decision body, got: {out!r}"

    # Now write the full brief and check the rendered Last decision line
    import tools.brief_assembler as ba
    brief_text = ba.assemble_brief(repo)
    last_dec_line = next(
        (l for l in brief_text.splitlines() if l.startswith("Last decision:") or "D-999" in l),
        None,
    )
    assert last_dec_line is not None, f"no Last decision line in brief; brief was: {brief_text[:500]!r}"
    assert "D-999" in last_dec_line, f"Last decision line stale; got: {last_dec_line!r}"
```

- [ ] **Step 3: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_brief_assembler.py::test_last_decision_matches_log_md_head -xvs
```

Expected: FAIL (either AssertionError about stale value, or field-name mismatch).

- [ ] **Step 4: Fix the root cause in `brief_assembler.py`**

Based on Step 1's investigation, fix whichever code path is producing the stale `last_decision`. Likely fix paths:

a) If the stale source is a cache file (e.g., `data/memories/.last_decision_cache`), DELETE the cache read; always derive from log.md head.

b) If the stale source is a different regex/parse path, route to `_extract_last_n_decisions(repo, n=1)` and take the first line.

Apply the fix as a surgical str_replace, not a wholesale rewrite.

- [ ] **Step 5: Verify the new test passes + existing tests still pass**

```bash
.venv/bin/python -m pytest tests/test_brief_assembler.py -xvs
```

- [ ] **Step 6: Commit**

```bash
git add tools/brief_assembler.py tests/test_brief_assembler.py
git commit -m "fix(brief_assembler): P-D427 brief staleness — derive Last decision from log.md head, not cache"
```

---

### Task 0.3: Concurrency isolation for `tools/lab_memory.py` (subprocess worker)

**Context:** PI brainstorm 2026-05-19 hit 5+ minute timeout on 3 parallel `lab_memory.py search` queries. The sentence-transformers model load happens in-process per query — when 3 invocations race, they contend on the model cache + sqlite-vec connection, then all hang.

**Files:**
- Create: `tools/retrieval/concurrency.py`
- Test: `tests/retrieval/test_concurrency.py`

- [ ] **Step 1: Write the failing test**

```python
"""Concurrency isolation for lab_memory.py search."""
import subprocess
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

def test_concurrent_queries_complete_within_budget():
    """3 parallel CLI queries must complete within 60s (not the 5+min PI saw)."""
    import concurrent.futures
    import time

    queries = [
        "router entropy collapse",
        "iPC kill verdict",
        "MoE active parameters",
    ]
    def run_one(q):
        out = subprocess.run(
            [str(REPO / ".venv/bin/python"), "tools/lab_memory.py", "search", q,
             "--top-k", "3", "--legacy"],
            capture_output=True, text=True, timeout=90, cwd=str(REPO),
        )
        return out.returncode, len(out.stdout)

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(run_one, queries))
    elapsed = time.time() - start

    assert all(rc == 0 for rc, _ in results), f"queries failed: {results}"
    assert elapsed < 60, f"3 parallel queries took {elapsed:.1f}s > 60s budget"
    assert all(out_len > 0 for _, out_len in results), "queries returned empty stdout"


def test_concurrency_worker_handles_n_queries():
    """The new RetrievalWorker spawn-once-serve-many path works for a single subprocess."""
    from tools.retrieval.concurrency import RetrievalWorker
    w = RetrievalWorker.spawn(REPO)
    try:
        r1 = w.query("router entropy", top_k=2)
        r2 = w.query("dense vs MoE", top_k=2)
        assert isinstance(r1, list) and len(r1) > 0, f"r1 empty: {r1!r}"
        assert isinstance(r2, list) and len(r2) > 0, f"r2 empty: {r2!r}"
    finally:
        w.shutdown()
```

- [ ] **Step 2: Run tests to verify failure**

```bash
.venv/bin/python -m pytest tests/retrieval/test_concurrency.py -xvs
```

Expected: `test_concurrency_worker_handles_n_queries` FAILs with ImportError.

- [ ] **Step 3: Implement `tools/retrieval/concurrency.py`**

```python
"""Subprocess-isolated retrieval worker.

Loads sentence-transformers model ONCE per worker subprocess, then serves
queries via stdin/stdout JSON-RPC. Cheap to spawn (~2s); cheap to query
(~50ms after first). Use when multiple in-process callers need to share
a model instance OR when concurrent CLI invocations would otherwise race
on the model cache.
"""
from __future__ import annotations
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_WORKER_SCRIPT = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path('REPO_PATH_PLACEHOLDER').resolve()))
from tools.lab_memory import LabMemory

lm = LabMemory(str(Path('REPO_PATH_PLACEHOLDER') / 'tools/lab_memory.db'))
print(json.dumps({"event": "ready"}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        print(json.dumps({"event": "error", "msg": str(e)}), flush=True)
        continue
    if req.get("cmd") == "shutdown":
        break
    if req.get("cmd") == "search":
        try:
            hits = lm.search(
                req["query"],
                program_id=req.get("program"),
                role=req.get("role"),
                phase=req.get("phase"),
                top_k=req.get("top_k", 10),
            )
            payload = [
                {"id": h.id, "program_id": h.program_id, "phase": h.phase,
                 "role": h.role, "source_path": h.source_path,
                 "chunk_text": h.chunk_text, "distance": h.distance}
                for h in hits
            ]
            print(json.dumps({"event": "result", "hits": payload}), flush=True)
        except Exception as e:
            print(json.dumps({"event": "error", "msg": str(e)}), flush=True)
"""


@dataclass
class RetrievalWorker:
    proc: subprocess.Popen
    repo: Path

    @classmethod
    def spawn(cls, repo: Path) -> "RetrievalWorker":
        script = _WORKER_SCRIPT.replace("REPO_PATH_PLACEHOLDER", str(repo))
        venv_py = repo / ".venv/bin/python"
        proc = subprocess.Popen(
            [str(venv_py), "-c", script],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        line = proc.stdout.readline()
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            stderr = proc.stderr.read()
            proc.kill()
            raise RuntimeError(f"worker startup failed: line={line!r}, stderr={stderr!r}") from e
        if msg.get("event") != "ready":
            stderr = proc.stderr.read()
            proc.kill()
            raise RuntimeError(f"worker did not signal ready: {msg!r}, stderr={stderr!r}")
        return cls(proc=proc, repo=repo)

    def query(self, q: str, top_k: int = 10, **filters: Any) -> list[dict]:
        req = {"cmd": "search", "query": q, "top_k": top_k, **filters}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        msg = json.loads(line)
        if msg.get("event") == "error":
            raise RuntimeError(f"worker error: {msg.get('msg')}")
        return msg.get("hits", [])

    def shutdown(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        self.proc.wait(timeout=5)
```

- [ ] **Step 4: Verify tests pass**

```bash
.venv/bin/python -m pytest tests/retrieval/test_concurrency.py -xvs
```

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/concurrency.py tests/retrieval/test_concurrency.py
git commit -m "feat(retrieval): subprocess-isolated RetrievalWorker — fixes parallel-query timeouts"
```

---

## Stage 1 — L0 Token Graph (highest leverage)

Six tasks. Builds a persistent graph of D-N decisions, P-* tokens, programs, phases, and roles. Queryable via Personalized PageRank.

### Task 1.1: Schema design + table creation (`tools/retrieval/graph.py` skeleton)

**Files:**
- Create: `tools/retrieval/graph.py`
- Test: `tests/retrieval/test_graph.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
.venv/bin/python -m pytest tests/retrieval/test_graph.py -xvs
```

- [ ] **Step 3: Implement skeleton**

```python
"""Token graph: persistent SQLite store of canonical lab tokens + edges.

Nodes:
    decision      D-N entries (e.g. "D-420")
    carry_forward P-* tokens (e.g. "P-D417-ARTIFACT-PROJECTOR-DEDUP-GAP")
    program       Program identifiers (e.g. "program_2_example")
    phase         Phase IDs scoped per program (e.g. "Phase 3", "P11")
    role          Agent role names (e.g. "director", "pi", "lab_architect")
    file          Source markdown file paths
    chunk         Chunks from tools/lab_memory.py (linked by chunk id)

Edges (directed):
    cites           A cites B (D-X cites D-Y)
    precedes        A precedes B in decision sequence
    resolves        A resolves B (D-420 resolves P-D417-...)
    raises          A raises carry-forward B
    dispatched_by   role/decision A dispatched chunk/decision B
    in_program      decision/phase/chunk belongs to program
    in_phase        decision/chunk belongs to phase
    in_file         chunk lives in file

Built by tools/retrieval/graph.py:build_graph_from_corpus which regex-parses
the lab's markdown corpus. No LLM extraction — token vocabulary is canonical.
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass
from pathlib import Path


NODE_TYPES = {"decision", "carry_forward", "program", "phase", "role", "file", "chunk"}
EDGE_TYPES = {"cites", "precedes", "resolves", "raises", "dispatched_by",
              "in_program", "in_phase", "in_file"}


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
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/graph.py tests/retrieval/test_graph.py
git commit -m "feat(retrieval/graph): schema for token graph (nodes + edges)"
```

---

### Task 1.2: Regex extraction of D-N and P-* tokens from corpus

**Files:**
- Modify: `tools/retrieval/graph.py`
- Test: `tests/retrieval/test_graph.py`

- [ ] **Step 1: Write the failing tests**

```python
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
        "In program_2_example Phase 3 P11, the director dispatched "
        "findings_curator and lab_architect."
    )
    progs = extract_program_refs(text)
    assert "program_2_example" in progs
    phases = extract_phase_refs(text)
    assert any("Phase 3" in p or "P11" in p for p in phases)
    roles = extract_role_refs(text)
    assert "director" in roles
    assert "findings_curator" in roles


def test_extract_decision_headers():
    """A markdown `### D-N` header marks D-N as DEFINED in this file."""
    from tools.retrieval.graph import extract_decision_headers
    text = (
        "### D-420 (2026-05-19, ~10:30 UTC): **P15 SECOND STAGE LANDED**\n\n"
        "Body cites D-419 and D-87.\n\n"
        "### D-421 (2026-05-19, ~13:00 UTC): **PROGRAM 2 CLOSED**\n"
    )
    headers = extract_decision_headers(text)
    assert headers == {"D-420", "D-421"}
```

- [ ] **Step 2: Run tests, verify fail**

- [ ] **Step 3: Implement extractors**

Add to `tools/retrieval/graph.py`:

```python
import re

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
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/graph.py tests/retrieval/test_graph.py
git commit -m "feat(retrieval/graph): regex extractors for D-N, P-*, program, phase, role tokens"
```

---

### Task 1.3: Corpus walker that builds the graph

**Files:**
- Modify: `tools/retrieval/graph.py`
- Test: `tests/retrieval/test_graph.py` + `tests/retrieval/fixtures/sample_corpus/`

- [ ] **Step 1: Create test fixtures**

```bash
mkdir -p tests/retrieval/fixtures/sample_corpus/data/memories
mkdir -p tests/retrieval/fixtures/sample_corpus/programs/program_2_example
```

Write `tests/retrieval/fixtures/sample_corpus/data/memories/log.md`:

```markdown
*Log tier — recent decisions.*

---

### D-432 (2026-05-19, ~21:43 UTC): **Program 3 lock 1i closed.**

Carries forward P-D417-ARTIFACT-PROJECTOR-DEDUP-GAP. Resolves nothing this session.
Dispatched: lab_architect, findings_curator. In program_2_example Phase 3 P11.

### D-431 (2026-05-19, ~20:30 UTC): **Director synthesis delivered.**

Cites D-87 and D-114.

### D-420 (2026-05-19, ~10:30 UTC): **P15 second stage landed.**

Raises P-D420-WORK-QUEUE-DEDUP-FIX.
```

- [ ] **Step 2: Write failing tests**

```python
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
```

- [ ] **Step 3: Run tests, verify fail**

- [ ] **Step 4: Implement `build_graph_from_corpus`**

Add to `tools/retrieval/graph.py`:

```python
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
            for md_path in root.rglob("*.md"):
                if not md_path.is_file():
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

                for p in programs:
                    _upsert_node(cur, p, "program", p, now_ts)
                    stats["programs"] += 1
                    for d in defined:
                        _upsert_edge(cur, d, p, "in_program", now_ts)
                        stats["edges_in_program"] += 1

                for ph in phases:
                    _upsert_node(cur, f"phase:{ph}", "phase", ph, now_ts)
                    stats["phases"] += 1

                for r in roles:
                    _upsert_node(cur, f"role:{r}", "role", r, now_ts)
                    stats["roles"] += 1

                for src in defined:
                    for dst in referenced:
                        if src == dst:
                            continue
                        _upsert_edge(cur, src, dst, "cites", now_ts)
                        stats["edges_cite"] += 1

                for src in defined:
                    for cf in carry_forwards:
                        _upsert_edge(cur, src, cf, "raises", now_ts)
                        stats["edges_raise"] += 1

        conn.commit()
    return stats


def _upsert_node(cur, node_id: str, node_type: str, label: str, ts: int) -> None:
    cur.execute(
        """INSERT INTO nodes (id, type, label, first_seen_ts, last_seen_ts)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET last_seen_ts = excluded.last_seen_ts""",
        (node_id, node_type, label, ts, ts),
    )


def _upsert_edge(cur, src: str, dst: str, edge_type: str, ts: int) -> None:
    cur.execute(
        """INSERT INTO edges (src, dst, edge_type, first_seen_ts)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(src, dst, edge_type) DO NOTHING""",
        (src, dst, edge_type, ts),
    )
```

- [ ] **Step 5: Run tests, verify pass**

- [ ] **Step 6: Commit**

```bash
git add tools/retrieval/graph.py tests/retrieval/test_graph.py tests/retrieval/fixtures/
git commit -m "feat(retrieval/graph): build_graph_from_corpus — regex parse + idempotent upserts"
```

---

### Task 1.4: Personalized PageRank query API

**Files:**
- Modify: `tools/retrieval/graph.py`
- Test: `tests/retrieval/test_graph.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement PPR using networkx**

Add to `tools/retrieval/graph.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/graph.py tests/retrieval/test_graph.py
git commit -m "feat(retrieval/graph): Personalized PageRank over token graph (networkx)"
```

---

### Task 1.5: CLI for graph inspection + build command

**Files:**
- Modify: `tools/retrieval/graph.py` (add `main()`)
- Test: `tests/retrieval/test_graph_cli.py`

- [ ] **Step 1: Write failing test**

```python
"""CLI smoke tests for tools/retrieval/graph.py."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

def test_graph_build_cli(tmp_path):
    db = tmp_path / "test_graph.db"
    out = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "tools.retrieval.graph", "build",
         "--db", str(db), "--root", str(REPO / "tests/retrieval/fixtures/sample_corpus")],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, f"stderr: {out.stderr}"
    assert "decisions" in out.stdout.lower() or "decisions" in out.stderr.lower()
    assert db.exists() and db.stat().st_size > 0


def test_graph_ppr_cli(tmp_path):
    db = tmp_path / "test_graph.db"
    subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "tools.retrieval.graph", "build",
         "--db", str(db), "--root", str(REPO / "tests/retrieval/fixtures/sample_corpus")],
        check=True, cwd=str(REPO),
    )
    out = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "tools.retrieval.graph", "ppr",
         "--db", str(db), "--seed", "D-420", "--top-k", "5"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, f"stderr: {out.stderr}"
    assert "D-420" in out.stdout
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement CLI**

Add to bottom of `tools/retrieval/graph.py`:

```python
def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Token graph build + query CLI.")
    parser.add_argument("--db", default="tools/lab_graph.db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build graph from corpus")
    p_build.add_argument("--root", default=".")

    p_ppr = sub.add_parser("ppr", help="Run Personalized PageRank query")
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
```

- [ ] **Step 4: Verify pass**

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/graph.py tests/retrieval/test_graph_cli.py
git commit -m "feat(retrieval/graph): CLI for build + ppr query"
```

---

### Task 1.6: Wire graph reindex into runner

**Files:**
- Modify: `run_agi_lab.sh`
- Test: `tests/test_runner_reclaim_stale.py` (add new test)

- [ ] **Step 1: Add tests for the runner hook**

```python
def test_runner_defines_retrieval_graph_reindex():
    text = RUNNER.read_text()
    assert "_run_retrieval_graph_reindex()" in text or "_run_retrieval_graph_reindex ()" in text, \
        "runner must define _run_retrieval_graph_reindex helper"


def test_runner_calls_retrieval_graph_reindex_post_director():
    text = RUNNER.read_text()
    pd_idx = text.find("tools/post_director.py")
    assert pd_idx > 0
    after_pd = text[pd_idx:]
    assert "_run_retrieval_graph_reindex" in after_pd, \
        "_run_retrieval_graph_reindex must be called after post_director.py"


def test_runner_retrieval_graph_reindex_non_fatal():
    text = RUNNER.read_text()
    import re
    m = re.search(r"_run_retrieval_graph_reindex\(\)\s*\{(.*?)\n\}", text, re.DOTALL)
    assert m
    body = m.group(1)
    assert "||" in body or "2>&1" in body, \
        "_run_retrieval_graph_reindex must handle non-zero exit non-fatally"
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Add the helper to `run_agi_lab.sh`**

After `_run_lab_memory_reindex()` definition, add:

```bash
_run_retrieval_graph_reindex() {
    # Rebuild the token graph (D-N, P-*, program, phase, role) from the corpus.
    # Idempotent — INSERT OR IGNORE on every node/edge. Cheap (~5-30s).
    # Hard 90s cap. Failure non-fatal.
    if [ -f tools/retrieval/graph.py ] && [ -x .venv/bin/python ]; then
        timeout 90 .venv/bin/python -m tools.retrieval.graph build \
            --db tools/lab_graph.db --root . \
            >> data/infra/retrieval_graph.log 2>&1 || \
            echo "retrieval_graph reindex failed/timed-out (non-fatal); will retry next iteration" \
                >> data/infra/retrieval_graph.log
    fi
}
```

Call it after `_run_lab_memory_reindex` in the main loop:

```bash
_run_lab_memory_reindex
_run_retrieval_graph_reindex
```

- [ ] **Step 4: Verify tests + bash syntax**

```bash
bash -n run_agi_lab.sh && echo "SYNTAX-OK"
.venv/bin/python -m pytest tests/test_runner_reclaim_stale.py -xvs
```

- [ ] **Step 5: Commit**

```bash
git add run_agi_lab.sh tests/test_runner_reclaim_stale.py
git commit -m "feat(runner): wire _run_retrieval_graph_reindex (post-Director hook, non-fatal)"
```

---

## Stage 2 — L2 BM25 (sparse retrieval over chunks)

Two tasks. BM25 over the same chunks as `tools/lab_memory.py`, persisted as JSON (no pickle).

### Task 2.1: BM25 index builder with JSON persistence

**Files:**
- Create: `tools/retrieval/bm25.py`
- Test: `tests/retrieval/test_bm25.py`

- [ ] **Step 1: Write failing tests**

```python
def test_bm25_index_build_and_query(tmp_path):
    from tools.retrieval.bm25 import BM25Index
    chunks = [
        {"id": 1, "source_path": "a.md", "chunk_text": "D-420 closed Program 2 phase 3"},
        {"id": 2, "source_path": "b.md", "chunk_text": "router entropy collapse mechanism"},
        {"id": 3, "source_path": "c.md", "chunk_text": "P-D417 carry forward dedup gap"},
        {"id": 4, "source_path": "d.md", "chunk_text": "dense vs MoE comparison at sub-100M"},
    ]
    idx = BM25Index.from_chunks(chunks)
    hits = idx.search("D-420 Program 2", top_k=2)
    assert len(hits) > 0
    assert hits[0]["id"] == 1
    hits = idx.search("P-D417 dedup", top_k=2)
    assert hits[0]["id"] == 3


def test_bm25_persists_and_reloads_via_json(tmp_path):
    """BM25 state is persisted as JSON-serialized tokenized corpus.
    On load, BM25Okapi is rebuilt from the corpus — no pickle anywhere."""
    from tools.retrieval.bm25 import BM25Index
    chunks = [{"id": 1, "source_path": "a.md", "chunk_text": "alpha beta gamma"}]
    idx_path = tmp_path / "bm25.json"
    idx = BM25Index.from_chunks(chunks)
    idx.save(idx_path)
    assert idx_path.exists()
    # Verify the file is JSON (not pickle)
    import json
    data = json.loads(idx_path.read_text())
    assert "chunks" in data
    assert "tokenized" in data
    idx2 = BM25Index.load(idx_path)
    hits = idx2.search("alpha", top_k=1)
    assert hits[0]["id"] == 1


def test_bm25_build_from_lab_memory_db(tmp_path):
    from tools.retrieval.bm25 import BM25Index
    import sqlite3
    db = tmp_path / "lab_memory.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY,
                program_id TEXT DEFAULT '',
                phase TEXT DEFAULT '',
                role TEXT DEFAULT '',
                deliverable_type TEXT DEFAULT '',
                source_path TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                timestamp INTEGER NOT NULL
            );
            INSERT INTO memories (source_path, chunk_text, timestamp) VALUES
                ('a.md', 'alpha beta gamma', 1),
                ('b.md', 'delta epsilon zeta', 2);
        """)
    idx = BM25Index.from_lab_memory_db(db)
    hits = idx.search("alpha", top_k=2)
    assert any(h["chunk_text"] == "alpha beta gamma" for h in hits)
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `tools/retrieval/bm25.py` with JSON persistence**

```python
"""BM25 sparse retrieval over the same chunks as tools/lab_memory.py.

Uses rank_bm25 (BM25Okapi). Persistence: JSON file containing tokenized
corpus + chunk metadata. On load, BM25Okapi is rebuilt from tokens.
JSON instead of pickle for safety (no arbitrary code execution risk on
load). Rebuild cost: ~50-200ms for 11K chunks — negligible.

Tokenization preserves canonical lab identifiers (D-420, P-D417-FOO,
program_2_example) by keeping hyphen-joined alphanumeric runs intact.
"""
from __future__ import annotations
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z0-9_]+)*|[0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that preserves canonical identifiers."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Index:
    bm25: BM25Okapi
    chunks: list[dict]
    tokenized: list[list[str]]  # cached for JSON save without re-tokenizing

    @classmethod
    def from_chunks(cls, chunks: list[dict]) -> "BM25Index":
        if not chunks:
            raise ValueError("BM25Index.from_chunks requires at least one chunk")
        tokenized = [tokenize(c["chunk_text"]) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25=bm25, chunks=chunks, tokenized=tokenized)

    @classmethod
    def from_lab_memory_db(cls, db_path: Path) -> "BM25Index":
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                """SELECT id, program_id, phase, role, deliverable_type,
                          source_path, chunk_text, timestamp
                   FROM memories"""
            ).fetchall()
        cols = ["id", "program_id", "phase", "role", "deliverable_type",
                "source_path", "chunk_text", "timestamp"]
        chunks = [dict(zip(cols, r)) for r in rows]
        return cls.from_chunks(chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        program: str | None = None,
        role: str | None = None,
        phase: str | None = None,
    ) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        indexed = []
        for i, score in enumerate(scores):
            c = self.chunks[i]
            if program is not None and c.get("program_id") != program:
                continue
            if role is not None and c.get("role") != role:
                continue
            if phase is not None and c.get("phase") != phase:
                continue
            indexed.append((score, c))
        indexed.sort(key=lambda x: x[0], reverse=True)
        return [{**c, "bm25_score": float(s)} for s, c in indexed[:top_k] if s > 0]

    def save(self, path: Path) -> None:
        """Persist as JSON. On load, BM25Okapi is rebuilt from tokenized."""
        data = {
            "chunks": self.chunks,
            "tokenized": self.tokenized,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(path.read_text())
        chunks = data["chunks"]
        tokenized = data["tokenized"]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25=bm25, chunks=chunks, tokenized=tokenized)
```

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/bm25.py tests/retrieval/test_bm25.py
git commit -m "feat(retrieval/bm25): rank_bm25 sparse index with JSON persistence (no pickle)"
```

---

### Task 2.2: BM25 reindex hook + CLI

**Files:**
- Modify: `tools/retrieval/bm25.py` (add main)
- Modify: `run_agi_lab.sh` (extend `_run_retrieval_graph_reindex`)
- Test: `tests/retrieval/test_bm25.py` (CLI smoke test)

- [ ] **Step 1: Add CLI test**

```python
def test_bm25_cli_build_and_search(tmp_path):
    import subprocess
    REPO = Path(__file__).resolve().parent.parent.parent
    db = REPO / "tools/lab_memory.db"
    if not db.exists():
        import pytest
        pytest.skip("lab_memory.db not present; skipping CLI integration")
    # Build into a tmp path so we don't disturb the live index
    idx_path = tmp_path / "bm25_test.json"
    out = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "tools.retrieval.bm25", "build",
         "--index", str(idx_path), "--lab-memory-db", str(db)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, f"build failed: {out.stderr}"
    assert idx_path.exists()

    out = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "tools.retrieval.bm25", "search",
         "router entropy", "--index", str(idx_path), "--top-k", "3"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert out.returncode == 0, f"search failed: {out.stderr[-500:]}"
    assert len(out.stdout) > 0
```

- [ ] **Step 2: Implement CLI**

Add to `tools/retrieval/bm25.py`:

```python
def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="BM25 sparse retrieval CLI.")
    parser.add_argument("--index", default="tools/lab_bm25.json")
    parser.add_argument("--lab-memory-db", default="tools/lab_memory.db")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="(Re)build the BM25 index from lab_memory.db")

    p_search = sub.add_parser("search", help="Search the index")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--program", default=None)
    p_search.add_argument("--role", default=None)
    p_search.add_argument("--phase", default=None)

    args = parser.parse_args(argv)
    index_path = Path(args.index)

    if args.cmd == "build":
        idx = BM25Index.from_lab_memory_db(Path(args.lab_memory_db))
        idx.save(index_path)
        print(f"Built BM25 index with {len(idx.chunks)} chunks at {index_path}")
        return 0

    if args.cmd == "search":
        if not index_path.exists():
            idx = BM25Index.from_lab_memory_db(Path(args.lab_memory_db))
            idx.save(index_path)
        else:
            idx = BM25Index.load(index_path)
        hits = idx.search(args.query, top_k=args.top_k, program=args.program,
                          role=args.role, phase=args.phase)
        for h in hits:
            snip = h["chunk_text"][:140].replace("\n", " ")
            print(f"[{h['bm25_score']:.3f}] {h['source_path']} — {snip}{'...' if len(h['chunk_text']) > 140 else ''}")
        return 0

    return 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
```

- [ ] **Step 3: Wire into runner**

Extend `_run_retrieval_graph_reindex` to also rebuild BM25:

```bash
_run_retrieval_graph_reindex() {
    if [ -f tools/retrieval/graph.py ] && [ -x .venv/bin/python ]; then
        timeout 90 .venv/bin/python -m tools.retrieval.graph build \
            --db tools/lab_graph.db --root . \
            >> data/infra/retrieval_graph.log 2>&1 || \
            echo "retrieval_graph reindex failed/timed-out (non-fatal)" \
                >> data/infra/retrieval_graph.log
    fi
    if [ -f tools/retrieval/bm25.py ] && [ -x .venv/bin/python ]; then
        timeout 60 .venv/bin/python -m tools.retrieval.bm25 build \
            --index tools/lab_bm25.json --lab-memory-db tools/lab_memory.db \
            >> data/infra/retrieval_bm25.log 2>&1 || \
            echo "bm25 reindex failed/timed-out (non-fatal)" \
                >> data/infra/retrieval_bm25.log
    fi
}
```

- [ ] **Step 4: Verify tests + bash syntax**

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/bm25.py tests/retrieval/test_bm25.py run_agi_lab.sh
git commit -m "feat(retrieval/bm25): CLI + runner reindex hook"
```

---

## Stage 3 — L3 RRF Fusion (hybrid search)

One task. Combines BM25 + dense via Reciprocal Rank Fusion.

### Task 3.1: RRF fusion + top-level hybrid search API

**Files:**
- Create: `tools/retrieval/hybrid.py`
- Test: `tests/retrieval/test_hybrid.py`

- [ ] **Step 1: Write failing tests**

```python
def test_rrf_fuses_two_rankings():
    from tools.retrieval.hybrid import reciprocal_rank_fusion
    dense_ranking = ["doc1", "doc2", "doc3", "doc4"]
    bm25_ranking = ["doc4", "doc3", "doc1", "doc2"]
    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking], k=60)
    ids = [d for d, _ in fused]
    assert set(ids) == {"doc1", "doc2", "doc3", "doc4"}
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_handles_disjoint_rankings():
    from tools.retrieval.hybrid import reciprocal_rank_fusion
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "d"]], k=60)
    ids = [d for d, _ in fused]
    assert set(ids) == {"a", "b", "c", "d"}


def test_hybrid_search_returns_unioned_hits(tmp_path):
    from tools.retrieval.hybrid import hybrid_search
    from tools.retrieval.bm25 import BM25Index
    chunks = [
        {"id": 1, "source_path": "a.md", "chunk_text": "D-420 closed Program 2"},
        {"id": 2, "source_path": "b.md", "chunk_text": "router entropy collapse"},
        {"id": 3, "source_path": "c.md", "chunk_text": "dense MoE comparison"},
    ]
    bm25 = BM25Index.from_chunks(chunks)
    bm25_path = tmp_path / "bm25.json"
    bm25.save(bm25_path)

    def stub_dense_search(query, top_k):
        return [
            {"id": 2, "source_path": "b.md", "chunk_text": "router entropy collapse", "distance": 0.1},
            {"id": 1, "source_path": "a.md", "chunk_text": "D-420 closed Program 2", "distance": 0.3},
        ]

    results = hybrid_search(
        query="D-420",
        bm25_index_path=bm25_path,
        dense_search_fn=stub_dense_search,
        top_k=5,
    )
    assert len(results) > 0
    top_ids = [r["id"] for r in results[:2]]
    assert 1 in top_ids
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `tools/retrieval/hybrid.py`**

```python
"""L3: RRF fusion of BM25 (sparse) + dense (vector) retrieval.

Reciprocal Rank Fusion is score-free: it ranks each retriever's hits, then
sums 1/(k+rank) across retrievers per document. Robust to score scale
differences between dense (cosine distance) and sparse (BM25). k=60 is the
literature default.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Iterable


def reciprocal_rank_fusion(
    rankings: Iterable[list[Any]],
    k: int = 60,
) -> list[tuple[Any, float]]:
    """Fuse multiple rankings via RRF. Each ranking is a list of doc-IDs in
    descending preference order. Returns list of (doc_id, fused_score) sorted
    by score descending.
    """
    scores: dict[Any, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def hybrid_search(
    query: str,
    bm25_index_path: Path,
    dense_search_fn: Callable[[str, int], list[dict]],
    top_k: int = 20,
    fetch_per_retriever: int = 50,
    rrf_k: int = 60,
    program: str | None = None,
    role: str | None = None,
    phase: str | None = None,
) -> list[dict]:
    """Run both retrievers, fuse via RRF, return top_k merged chunks."""
    from tools.retrieval.bm25 import BM25Index
    if bm25_index_path.exists():
        bm25 = BM25Index.load(bm25_index_path)
        bm25_hits = bm25.search(query, top_k=fetch_per_retriever,
                                program=program, role=role, phase=phase)
    else:
        bm25_hits = []
    dense_hits = dense_search_fn(query, fetch_per_retriever)

    bm25_ids = [h["id"] for h in bm25_hits]
    dense_ids = [h["id"] for h in dense_hits]

    fused = reciprocal_rank_fusion([bm25_ids, dense_ids], k=rrf_k)

    by_id_dense = {h["id"]: h for h in dense_hits}
    by_id_bm25 = {h["id"]: h for h in bm25_hits}

    results = []
    for doc_id, rrf_score in fused[:top_k]:
        rec = by_id_dense.get(doc_id) or by_id_bm25.get(doc_id) or {"id": doc_id}
        results.append({
            **rec,
            "rrf_score": rrf_score,
            "from_dense": doc_id in by_id_dense,
            "from_bm25": doc_id in by_id_bm25,
        })
    return results
```

- [ ] **Step 4: Verify tests pass**

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/hybrid.py tests/retrieval/test_hybrid.py
git commit -m "feat(retrieval/hybrid): RRF fusion of BM25 + dense retrieval"
```

---

## Stage 4 — L4 Cross-Encoder Reranker

Two tasks. Drop-in cross-encoder on top-30 → top-5.

### Task 4.1: Reranker model wrapper

**Files:**
- Create: `tools/retrieval/rerank.py`
- Test: `tests/retrieval/test_rerank.py`

- [ ] **Step 1: Write failing tests**

```python
def test_reranker_lazy_loads_model():
    from tools.retrieval.rerank import CrossEncoderReranker
    r = CrossEncoderReranker()
    assert r._model is None


def test_reranker_orders_pairs_by_relevance():
    from tools.retrieval.rerank import CrossEncoderReranker
    r = CrossEncoderReranker()
    query = "Program 2 closed with verdict O4"
    candidates = [
        {"id": 1, "chunk_text": "Random unrelated text about cats and dogs."},
        {"id": 2, "chunk_text": "Program 2 dense vs MoE closed at D-421 with verdict O4 'below the measurement floor'."},
        {"id": 3, "chunk_text": "Program 1 envelope paper on Opus-4.7 18GB infeasibility."},
    ]
    reranked = r.rerank(query, candidates, top_k=3)
    top_ids = [c["id"] for c in reranked[:2]]
    assert 2 in top_ids, f"reranker missed relevant chunk; order: {[c['id'] for c in reranked]}"


def test_reranker_truncates_top_k():
    from tools.retrieval.rerank import CrossEncoderReranker
    r = CrossEncoderReranker()
    candidates = [{"id": i, "chunk_text": f"chunk {i}"} for i in range(10)]
    reranked = r.rerank("query", candidates, top_k=3)
    assert len(reranked) == 3
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `tools/retrieval/rerank.py`**

```python
"""L4: Cross-encoder reranker over hybrid retrieval top-K.

Uses sentence-transformers' CrossEncoder with `bge-reranker-v2-m3` by default
(SOTA listwise reranker; ~568 MB; multilingual; first-load ~10s, subsequent
~50-100ms for 30 pairs). Falls back to `ms-marco-MiniLM-L-12-v2` (~120 MB;
faster, English-only) if the default fails to download.
"""
from __future__ import annotations
from dataclasses import dataclass


DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
FALLBACK_RERANKER = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@dataclass
class CrossEncoderReranker:
    model_name: str = DEFAULT_RERANKER
    _model: object | None = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = CrossEncoder(FALLBACK_RERANKER)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        text_key: str = "chunk_text",
    ) -> list[dict]:
        if not candidates:
            return []
        model = self._get_model()
        pairs = [(query, c.get(text_key, "")) for c in candidates]
        scores = model.predict(pairs)
        scored = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored[:top_k]
```

- [ ] **Step 4: Verify tests pass**

Note: first-run model download will take ~30-60s (HF download of ~568MB). Subsequent runs use the cache.

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "feat(retrieval/rerank): cross-encoder reranker (bge-reranker-v2-m3)"
```

---

### Task 4.2: Orchestrator — `tools/retrieval/search.py`

**Files:**
- Create: `tools/retrieval/search.py`
- Test: `tests/retrieval/test_search.py`

- [ ] **Step 1: Write failing test**

```python
def test_search_returns_ranked_chunks_end_to_end():
    from tools.retrieval.search import search
    from pathlib import Path
    REPO = Path(__file__).resolve().parent.parent.parent

    if not (REPO / "tools/lab_memory.db").exists():
        import pytest
        pytest.skip("lab_memory.db missing; integration test skipped")

    results = search(
        query="Program 2 D-421 closure_memo",
        repo_root=REPO,
        top_k=5,
        use_graph=True,
        use_rerank=True,
    )
    assert isinstance(results, list)
    if results:
        assert "chunk_text" in results[0]
        assert "rerank_score" in results[0] or "rrf_score" in results[0]


def test_search_works_without_rerank():
    from tools.retrieval.search import search
    from pathlib import Path
    REPO = Path(__file__).resolve().parent.parent.parent
    if not (REPO / "tools/lab_memory.db").exists():
        import pytest
        pytest.skip("lab_memory.db missing")
    results = search(
        query="router entropy",
        repo_root=REPO,
        top_k=3,
        use_graph=False,
        use_rerank=False,
    )
    assert isinstance(results, list)


def test_search_extracts_seeds_from_query():
    from tools.retrieval.search import _extract_seed_tokens
    seeds = _extract_seed_tokens("Tell me about D-420 and P-D417-DEDUP")
    assert "D-420" in seeds
    assert "P-D417-DEDUP" in seeds
```

- [ ] **Step 2: Run, verify fail**

- [ ] **Step 3: Implement `tools/retrieval/search.py`**

```python
"""Top-level orchestrator for the 4-layer retrieval system.

Pipeline:
    1. Extract canonical seed tokens from query (D-N, P-*)
    2. Run PPR on the token graph; produce related-node set
    3. Run BM25 search and dense search in parallel
    4. RRF fusion
    5. Cross-encoder rerank top-30 to top-K
    6. Return ranked chunks with full metadata
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

from tools.retrieval.graph import (
    TokenGraph, personalized_pagerank,
    extract_decision_tokens, extract_carry_forward_tokens,
)
from tools.retrieval.bm25 import BM25Index
from tools.retrieval.hybrid import hybrid_search
from tools.retrieval.rerank import CrossEncoderReranker


def _extract_seed_tokens(query: str) -> set[str]:
    """Pull canonical tokens from the query for graph PPR seeding."""
    decisions = extract_decision_tokens(query)
    carry_forwards = extract_carry_forward_tokens(query)
    return decisions | carry_forwards


def search(
    query: str,
    repo_root: Path,
    top_k: int = 5,
    fetch_per_retriever: int = 50,
    rerank_pool: int = 30,
    use_graph: bool = True,
    use_rerank: bool = True,
    program: str | None = None,
    role: str | None = None,
    phase: str | None = None,
) -> list[dict]:
    """Run the full 4-layer retrieval pipeline."""
    # ---- L0: Token graph seed expansion ----
    graph_related: set[str] = set()
    if use_graph:
        seeds = _extract_seed_tokens(query)
        if seeds:
            graph_db = repo_root / "tools/lab_graph.db"
            if graph_db.exists():
                g = TokenGraph(graph_db)
                seed_weights = {s: 1.0 / len(seeds) for s in seeds}
                ranking = personalized_pagerank(g, seed_weights, top_k=20)
                graph_related = {node_id for node_id, _ in ranking}

    # ---- L1 + L2: dense + BM25 ----
    bm25_path = repo_root / "tools/lab_bm25.json"
    lab_memory_db = repo_root / "tools/lab_memory.db"

    def dense_search(q: str, k: int) -> list[dict]:
        import sys as _sys
        if str(repo_root) not in _sys.path:
            _sys.path.insert(0, str(repo_root))
        from tools.lab_memory import LabMemory
        lm = LabMemory(str(lab_memory_db))
        hits = lm.search(q, program_id=program, role=role, phase=phase, top_k=k)
        return [
            {"id": h.id, "program_id": h.program_id, "phase": h.phase, "role": h.role,
             "source_path": h.source_path, "chunk_text": h.chunk_text, "distance": h.distance}
            for h in hits
        ]

    fused = hybrid_search(
        query=query,
        bm25_index_path=bm25_path,
        dense_search_fn=dense_search,
        top_k=rerank_pool if use_rerank else top_k,
        fetch_per_retriever=fetch_per_retriever,
        program=program, role=role, phase=phase,
    )

    # Graph boost annotation (does not re-rank — keeps RRF authoritative)
    if graph_related:
        for chunk in fused:
            file_node = f"file:{chunk.get('source_path', '')}"
            chunk["graph_boost"] = file_node in graph_related

    # ---- L4: rerank ----
    if use_rerank:
        rr = CrossEncoderReranker()
        final = rr.rerank(query, fused, top_k=top_k)
    else:
        final = fused[:top_k]

    return final


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="4-layer hybrid retrieval CLI.")
    parser.add_argument("query")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--program", default=None)
    parser.add_argument("--role", default=None)
    parser.add_argument("--phase", default=None)
    args = parser.parse_args(argv)

    results = search(
        query=args.query,
        repo_root=Path(args.repo),
        top_k=args.top_k,
        use_graph=not args.no_graph,
        use_rerank=not args.no_rerank,
        program=args.program,
        role=args.role,
        phase=args.phase,
    )

    for r in results:
        score = r.get("rerank_score") or r.get("rrf_score") or 0.0
        boost = " +graph" if r.get("graph_boost") else ""
        sources = ("D" if r.get("from_dense") else " ") + ("B" if r.get("from_bm25") else " ")
        snip = r.get("chunk_text", "")[:160].replace("\n", " ")
        print(f"[{score:.3f}][{sources}{boost}] {r.get('source_path', '?')} — {snip}{'...' if len(r.get('chunk_text', '')) > 160 else ''}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
```

- [ ] **Step 4: Verify tests pass**

```bash
.venv/bin/python -m pytest tests/retrieval/ -xvs
```

- [ ] **Step 5: Commit**

```bash
git add tools/retrieval/search.py tests/retrieval/test_search.py
git commit -m "feat(retrieval/search): top-level 4-layer orchestrator"
```

---

## Stage 5 — Integration + Documentation

Three tasks. Wire into existing tools, update agent prompts, write the methodology doc.

### Task 5.1: Update `tools/lab_memory.py` to call new system

**Files:**
- Modify: `tools/lab_memory.py`

- [ ] **Step 1: Verify existing tests pass before edit**

```bash
.venv/bin/python -m pytest tests/test_lab_memory.py -xvs
```

- [ ] **Step 2: Modify `_cmd_search` to delegate to new system**

Edit `_cmd_search` in `tools/lab_memory.py`:

```python
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

    from pathlib import Path
    from tools.retrieval.search import search
    repo_root = Path(args.db).resolve().parent.parent
    results = search(
        query=args.query,
        repo_root=repo_root,
        top_k=args.top_k,
        program=args.program, role=args.role, phase=args.phase,
        use_graph=not getattr(args, "no_graph", False),
        use_rerank=not getattr(args, "no_rerank", False),
    )
    for r in results:
        score = r.get("rerank_score") or r.get("rrf_score") or 0.0
        snip = r.get("chunk_text", "")[:200].replace("\n", " ")
        print(f"[{score:.3f}] {r.get('source_path', '?')} — {snip}{'...' if len(r.get('chunk_text', '')) > 200 else ''}")
        print()
```

Add the new flags to argparse:

```python
p_srch.add_argument("--legacy", action="store_true",
                    help="Use the pre-2026-05-20 pure-vector path (ablation)")
p_srch.add_argument("--no-graph", action="store_true",
                    help="Skip L0 token graph seed expansion")
p_srch.add_argument("--no-rerank", action="store_true",
                    help="Skip L4 cross-encoder reranking (~10× faster)")
```

- [ ] **Step 3: Verify existing tests still pass + smoke**

```bash
.venv/bin/python -m pytest tests/test_lab_memory.py tests/retrieval/ -xvs
.venv/bin/python tools/lab_memory.py search "Program 2 D-421" --top-k 3
.venv/bin/python tools/lab_memory.py search "Program 2 D-421" --legacy --top-k 3
```

- [ ] **Step 4: Commit**

```bash
git add tools/lab_memory.py
git commit -m "feat(lab_memory): delegate search to 4-layer retrieval by default (--legacy keeps old path)"
```

---

### Task 5.2: Update Director procedural to surface the new capabilities

**Files:**
- Modify: `data/agents/director/procedural.md` (and via symlink, `data/agents/director_prompt.md`)

- [ ] **Step 1: Locate the existing `lab_memory.py search` reference**

```bash
grep -n "lab_memory" data/agents/director/procedural.md | head -5
```

- [ ] **Step 2: Update the retrieval-tools description**

Find this line:
```
**Retrieval tools available:** `Grep`, `Glob`, `Read`, and `.venv/bin/python tools/lab_memory.py search "<query>"` ...
```

Replace with:
```
**Retrieval tools available:**
- `Grep`, `Glob`, `Read` — exact-match + file walks
- `.venv/bin/python tools/lab_memory.py search "<query>"` — **4-layer hybrid retrieval** (token graph + BM25 + dense + reranker). Default for "what did we try / decide / surface before?" queries. Pass `--program <name> --role <role> --phase <phase>` to filter. Pass `--legacy` for the old pure-vector path if you suspect the hybrid is misbehaving (rare). Pass `--no-rerank` for ~10× faster results when you trust BM25+RRF alone.
- `.venv/bin/python -m tools.retrieval.graph ppr --seed D-N --top-k 20` — **token graph traversal**. Use when you want "all decisions/carry-forwards related to D-X" without text-matching them. The graph encodes cites/precedes/resolves/raises edges; PPR diffuses from the seed.
- `.venv/bin/python -m tools.retrieval.bm25 search "<query>"` — sparse-only retrieval. Use when the query is dominated by canonical identifiers (D-N, P-*, program names) and you don't want semantic ranking to interfere.

Index auto-refreshes after each Director exit. Corpus is ~11K+ chunks across 347+ files; token graph is ~10K nodes / 50K edges.
```

- [ ] **Step 3: Verify**

```bash
grep -n "tools/retrieval" data/agents/director/procedural.md
```

- [ ] **Step 4: Commit**

```bash
git add data/agents/director/procedural.md
git commit -m "docs(director): surface 4-layer retrieval (hybrid + graph PPR + sparse-only) in CCA section"
```

---

### Task 5.3: Methodology doc + substrate changelog entry

**Files:**
- Create: `docs/retrieval/architecture.md`
- Modify: `SUBSTRATE_CHANGELOG.md`

- [ ] **Step 1: Write `docs/retrieval/architecture.md`**

A ~2000-word doc explaining:
- 4-layer architecture with diagram
- Design rationale: why canonical-token graph instead of LLM extraction
- Performance characteristics: rebuild cost, query latency per layer, storage
- Comparison to SOTA: HippoRAG2, LightRAG, Microsoft GraphRAG — what we adopted, what we deliberately skipped, why
- Future work for methodology paper: community detection, reflection layer, write-side hallucination detection

Use the SOTA synthesis as input. Cite arXiv IDs:
- 2404.16130 (Microsoft GraphRAG)
- 2410.05779 (LightRAG)
- 2405.14831 (HippoRAG)
- 2502.12110 (A-MEM)
- 2511.03506 (HaluMem)
- 2304.03442 (Generative Agents memory)
- 2309.02427 (CoALA)

- [ ] **Step 2: Append substrate changelog entry**

```
## 2026-05-20 HH:MM | <sha>+uncommitted | tools/retrieval/* (new package) + tools/lab_memory.py:_cmd_search + data/agents/director/procedural.md:Retrieval-tools + run_agi_lab.sh:_run_retrieval_graph_reindex | landed 4-layer hybrid retrieval: token graph (L0) + BM25 (L2) + dense (L1, existing) + cross-encoder rerank (L4), fused via RRF (L3) | replaces flat-vector-only retrieval; closes the 30%-success / 5-min-timeout symptoms documented in 2026-05-20 audit; the canonical-token graph (D-N, P-*, program, phase, role) is regex-built — no LLM-extraction cost. Concurrency-isolation fix via RetrievalWorker (subprocess); brief-staleness fix landed separately (Task 0.2) | new tools: `tools/lab_memory.py search` is now hybrid by default (--legacy for old path); `.venv/bin/python -m tools.retrieval.graph ppr --seed D-N` for graph traversal; `.venv/bin/python -m tools.retrieval.bm25 search` for sparse-only. Index refresh wired into runner after each Director exit. Director prompt updated.
```

- [ ] **Step 3: Final integration test**

```bash
.venv/bin/python -m tools.retrieval.graph build --db tools/lab_graph.db --root .
.venv/bin/python -m tools.retrieval.bm25 build --index tools/lab_bm25.json --lab-memory-db tools/lab_memory.db

.venv/bin/python tools/lab_memory.py search "P-D417 dedup gap" --top-k 5
.venv/bin/python -m tools.retrieval.graph ppr --seed D-420 --seed D-421 --top-k 10

.venv/bin/python -m pytest tests/retrieval/ tests/test_lab_memory.py tests/test_brief_assembler.py -xvs
```

Expected: all green; the query returns the actual P-D417 retro section as top hit; PPR over D-420+D-421 surfaces related decisions and carry-forwards.

- [ ] **Step 4: Commit**

```bash
git add docs/retrieval/architecture.md SUBSTRATE_CHANGELOG.md
git commit -m "docs(retrieval): architecture document + substrate changelog entry"
```

---

## Final Review Checklist

After all stages land, run:

- [ ] `bash -n run_agi_lab.sh` — syntax clean
- [ ] `.venv/bin/python -m pytest tests/retrieval/ tests/test_lab_memory.py tests/test_brief_assembler.py tests/test_runner_reclaim_stale.py` — all green
- [ ] `.venv/bin/python tools/lab_memory.py search "test query"` — returns results (smoke test)
- [ ] `.venv/bin/python -m tools.retrieval.graph build --db tools/lab_graph.db --root .` — builds without error
- [ ] Lab restart: `make lab-start` — runner starts, new tools work for first Director session
- [ ] One full Director session runs with the new retrieval — verify no regressions in log.md head

---

## What's intentionally NOT in this plan

- **Entity extraction via LLM** — token vocabulary is canonical; no need.
- **Community detection / Leiden clustering on the graph** — premature until corpus is 10× bigger.
- **Reflection layer** — closure memos already serve this; pre-mature to automate.
- **Write-side hallucination detection** — HaluMem 2025 paper makes this a real gap, but it's a Program-scale follow-on, not a memory-tool task. Track as a separate carry-forward.
- **Neo4j / graph DB migration** — sqlite handles 10K nodes / 100K edges trivially.
- **Switching to Ollama for embeddings** — `nomic-embed-text` would be a swap of the dense layer; doesn't change architecture. Defer.
- **Pickle anywhere** — JSON persistence everywhere (security: untrusted-content safety).

These are listed so the next architect retro can revisit them deliberately.

---

## Estimated effort

- Stage 0 (foundations + 2 small fixes): ~1 day
- Stage 1 (token graph + PPR + runner hook): ~2 days
- Stage 2 (BM25): ~0.5 day
- Stage 3 (RRF fusion): ~0.25 day
- Stage 4 (reranker + orchestrator): ~1 day
- Stage 5 (integration + docs): ~0.5 day

**Total: ~5 days of focused work** if executed via subagent-driven-development. Each task is independently shippable; safe to pause between stages.

If executed inline (single session per task), expect more like 8-10 days due to context-switching.

---

## Rollback plan

Every commit is independent. If any layer misbehaves in production:

- **L4 (reranker) misbehaves** → `--no-rerank` flag bypasses it
- **L0 (graph) misbehaves** → `--no-graph` flag bypasses; old vector-only path remains via `--legacy`
- **L2 (BM25) misbehaves** → delete `tools/lab_bm25.json`; hybrid_search falls back to dense-only
- **L1 (dense) misbehaves** → already exists pre-this-plan; revert is just `git revert`

No commit deletes or rewrites existing data; the only persistent state added is `tools/lab_graph.db` and `tools/lab_bm25.json`, both regenerable.

---

End of plan.
