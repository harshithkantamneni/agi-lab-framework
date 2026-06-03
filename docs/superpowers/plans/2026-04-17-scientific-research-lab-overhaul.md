# Scientific Research Lab Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transition the lab from cycle-based engineering to program-based research. Install PI+Director governance with unanimous-compromise mediation, 30 seed roles with CoALA memory, Cognitive Context Architecture (CCA) with local semantic memory, HIVE-style rate-limit handling, and a Program 0 retrospective of cycles 1-31 before launching Program 1 on the mission question.

**Architecture:** In-place transition. No framework changes. New infrastructure: `tools/lab_memory.py` (SQLite + sqlite-vec + sentence-transformers for local semantic retrieval). Updated `run_agi_lab.sh` (explicit model ID, pre-emptive compaction, precise rate-limit waits). Rewritten `data/agents/agents.json` (30 seed roles). New `programs/` directory tree. Legacy preserved in `legacy/cycles_1_31/`.

**Tech Stack:** Python 3.14, SQLite (stdlib) + sqlite-vec extension, sentence-transformers (all-MiniLM-L6-v2, ~80MB), Bash, JSON, Markdown. No cloud services. No new frameworks.

**Spec reference:** `docs/superpowers/specs/2026-04-17-scientific-research-lab-overhaul.md`

**Pre-conditions:**
- Lab is in tmux session `agi-lab` OR explicitly stopped (check with `tmux has-session -t agi-lab`).
- Cycle 32 has written its session_exit.md (check `data/session_exit.md` exists OR tmux session is dead).
- Git working directory has no uncommitted changes to files this plan touches (safety — run `git status` before starting).

---

## File Structure (what gets created, modified, or moved)

### New files
```
tools/lab_memory.py                                                 (new, ~650 LOC)
tools/lab_memory.db                                                 (generated)
tools/test_lab_memory.py                                            (new, ~200 LOC)
data/agents/pi_prompt.md                                            (new)
data/agents/mediator_prompt.md                                      (new)
data/agents/<role>/procedural.md                                    (30 new files)
data/agents/<role>/semantic.md                                      (30 new, empty)
data/agents/<role>/episodic/                                        (30 new, empty dirs)
data/knowledge_index.md                                             (new)
data/infra/rate_limit_resets_at                                     (generated)
programs/portfolio.md                                               (new)
programs/program_0_retrospective/question.md                        (new)
programs/program_0_retrospective/prior_work.md                      (new)
programs/program_0_retrospective/findings.md                        (new)
programs/program_0_retrospective/killed.md                          (new)
programs/program_0_retrospective/open_hypotheses.md                 (new)
programs/program_0_retrospective/infrastructure.md                  (new)
programs/program_0_retrospective/data_assets.md                     (new)
programs/program_0_retrospective/bibliography_inherited.md          (new)
programs/program_1_opus47_on_18gb/question.md                       (new, Phase 1 output)
legacy/cycles_1_31/                                                 (new dir, contents moved)
```

### Modified files
```
run_agi_lab.sh                                                      (model ID, compaction, rate-limit fn)
tools/stream_formatter.py                                           (rate_limit_event parser)
data/agents/agents.json                                             (22 roles → 30 seed roles)
data/agents/director_prompt.md                                      (program-mode rewrite)
data/agents/retired.json                                            (add 4 replaced old roles)
data/procedures.md                                                  (phases, mediation, lifecycles)
data/pi_notes.md                                                    (refresh for program-mode)
data/values.md                                                      (minor program-mode tweaks)
data/index.md                                                       (new paths)
data/state.md                                                       (reframe program-based)
```

### Files moved to legacy
```
data/decisions_archive.md             → legacy/cycles_1_31/decisions_archive.md
data/summaries/cycle_*.md             → legacy/cycles_1_31/summaries/
data/infra/session_logs/session_*.log → legacy/cycles_1_31/session_logs/
data/org_retros/retro_*.md            → legacy/cycles_1_31/org_retros/
data/cycle_queue.md                   → legacy/cycles_1_31/state_snapshots/cycle_queue.md
data/session_state.md                 → legacy/cycles_1_31/state_snapshots/session_state.md
data/roadmap.md                       → legacy/cycles_1_31/state_snapshots/roadmap.md
```

---

## STAGE 1 — Close Current Cycle Mode

### Task 1: Verify lab is stopped and working tree is clean

**Files:** (none modified; verification only)

- [ ] **Step 1: Verify lab is stopped**

Run:
```bash
tmux has-session -t agi-lab 2>&1 || echo "No session"
ps aux | grep -E "claude" | grep -v grep | grep -v "Claude Helper" | head -3
```

Expected: "No session" AND no active Director `claude` process. If lab is running, stop it:
```bash
tmux kill-session -t agi-lab
```

- [ ] **Step 2: Verify git working tree**

Run:
```bash
cd <repo> && git status --short
```

Expected: uncommitted changes are acceptable (lab generates them), but note what's dirty. If `run_agi_lab.sh`, `data/agents/agents.json`, `tools/stream_formatter.py`, or `data/agents/director_prompt.md` show as modified, review the diff before proceeding:
```bash
git diff run_agi_lab.sh data/agents/agents.json tools/stream_formatter.py data/agents/director_prompt.md
```

- [ ] **Step 3: Capture current state snapshot**

Run:
```bash
mkdir -p <repo>/data/archives/pre_overhaul_2026-04-17
cp <repo>/data/state.md <repo>/data/archives/pre_overhaul_2026-04-17/state.md
cp <repo>/data/agents/agents.json <repo>/data/archives/pre_overhaul_2026-04-17/agents.json
cp <repo>/data/agents/director_prompt.md <repo>/data/archives/pre_overhaul_2026-04-17/director_prompt.md
cp <repo>/run_agi_lab.sh <repo>/data/archives/pre_overhaul_2026-04-17/run_agi_lab.sh
cp <repo>/tools/stream_formatter.py <repo>/data/archives/pre_overhaul_2026-04-17/stream_formatter.py
```

Expected: no errors.

- [ ] **Step 4: Commit the snapshot (rollback point)**

Run:
```bash
cd <repo>
git add data/archives/pre_overhaul_2026-04-17/
git commit -m "$(cat <<'EOF'
chore: snapshot pre-overhaul state for rollback point

Archive snapshot of state.md, agents.json, director_prompt.md, run_agi_lab.sh, stream_formatter.py before the research-lab overhaul begins.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit success.

---

### Task 2: Create new directory structure

**Files:**
- Create: `programs/` (new dir)
- Create: `programs/archive/` (new dir)
- Create: `programs/program_0_retrospective/` (new dir)
- Create: `programs/program_1_opus47_on_18gb/` (new dir)
- Create: `legacy/cycles_1_31/` (new dir + subdirs)

- [ ] **Step 1: Create programs/ tree**

Run:
```bash
cd <repo>
mkdir -p programs/archive
mkdir -p programs/program_0_retrospective
mkdir -p programs/program_1_opus47_on_18gb/results_raw
mkdir -p programs/program_1_opus47_on_18gb/figures
```

Expected: no errors. Verify with `ls programs/`.

- [ ] **Step 2: Create legacy/ tree**

Run:
```bash
mkdir -p legacy/cycles_1_31/summaries
mkdir -p legacy/cycles_1_31/session_logs
mkdir -p legacy/cycles_1_31/state_snapshots
mkdir -p legacy/cycles_1_31/org_retros
mkdir -p legacy/cycles_1_31/evaluator_reports
```

- [ ] **Step 3: Create portfolio.md**

Write `<repo>/programs/portfolio.md`:

```markdown
# Program Portfolio

*Managed by the Director + PI. Tracks active, planned, and archived programs.*

## Active

- **program_1_opus47_on_18gb** — Status: PENDING (opens after Stage 4 of overhaul)
  - Question: Can we fit an Opus-4.7-equivalent model on an 18GB M3 Pro laptop? What is the feasibility envelope given current techniques + 18GB unified memory + no cloud compute?
  - Started: TBD
  - Current phase: Pending (awaits unanimous PI+Director open)

## Archived (closed programs)

- **program_0_retrospective** — Status: to be built during Stage 3
  - Question: What did we learn in cycles 1-31?
  - Closed: TBD

## Planned

(None yet — PI + Director propose in retro between programs.)
```

- [ ] **Step 4: Commit directory structure**

```bash
cd <repo>
git add programs/ legacy/ 
git commit -m "$(cat <<'EOF'
feat: scaffold programs/ and legacy/ directory structure

- programs/ becomes the primary work directory (program-based rhythm)
- legacy/cycles_1_31/ will preserve all prior work (archive never delete)
- portfolio.md tracks active/archived/planned programs

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Move legacy files

**Files:**
- Move: `data/decisions_archive.md` → `legacy/cycles_1_31/decisions_archive.md`
- Move: `data/summaries/cycle_*.md` → `legacy/cycles_1_31/summaries/`
- Move: `data/infra/session_logs/session_*.log` → `legacy/cycles_1_31/session_logs/`
- Move: `data/org_retros/retro_*.md` → `legacy/cycles_1_31/org_retros/`
- Move: `data/cycle_queue.md`, `data/session_state.md`, `data/roadmap.md` → `legacy/cycles_1_31/state_snapshots/`

- [ ] **Step 1: Verify files exist before moving**

Run:
```bash
cd <repo>
ls data/decisions_archive.md data/cycle_queue.md data/session_state.md data/roadmap.md 2>&1
ls data/summaries/ 2>&1 | head -20
ls data/infra/session_logs/ 2>&1 | head -5
ls data/org_retros/ 2>&1
```

Expected: some may not exist (session_state.md, roadmap.md, cycle_queue.md have been deprecated already) — note which exist.

- [ ] **Step 2: Move decisions_archive.md**

```bash
cd <repo>
if [ -f data/decisions_archive.md ]; then
  git mv data/decisions_archive.md legacy/cycles_1_31/decisions_archive.md
fi
```

- [ ] **Step 3: Move cycle summaries**

```bash
cd <repo>
git mv data/summaries/cycle_*.md legacy/cycles_1_31/summaries/
# Keep data/summaries/ dir (may have rolling_summary.md)
if [ -f data/summaries/rolling_summary.md ]; then
  git mv data/summaries/rolling_summary.md legacy/cycles_1_31/summaries/
fi
```

- [ ] **Step 4: Move session logs**

```bash
cd <repo>
git mv data/infra/session_logs/session_*.log legacy/cycles_1_31/session_logs/
```

- [ ] **Step 5: Move org retros**

```bash
cd <repo>
if [ -d data/org_retros ]; then
  if ls data/org_retros/retro_*.md 2>/dev/null; then
    git mv data/org_retros/retro_*.md legacy/cycles_1_31/org_retros/
  fi
fi
```

- [ ] **Step 6: Move deprecated state files if they exist**

```bash
cd <repo>
for f in cycle_queue.md session_state.md roadmap.md; do
  if [ -f "data/$f" ]; then
    git mv "data/$f" "legacy/cycles_1_31/state_snapshots/$f"
  fi
done
```

- [ ] **Step 7: Commit legacy migration**

```bash
cd <repo>
git add -A legacy/ data/
git commit -m "$(cat <<'EOF'
refactor: move cycle 1-31 artifacts to legacy/

Preserves all prior work per archive-never-delete policy. Session logs, cycle summaries, decisions_archive, org retros, and deprecated state files now live under legacy/cycles_1_31/. Program 0 retrospective will index these into lab_memory in Stage 3.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit success.

---

## STAGE 2A — Build tools/lab_memory.py

### Task 4: Install dependencies

**Files:**
- Modify: `.venv/` (install packages)
- Possibly: `requirements.txt` (add deps if it exists)

- [ ] **Step 1: Activate venv and install sentence-transformers**

Run:
```bash
cd <repo>
source .venv/bin/activate
pip install sentence-transformers
```

Expected: installs torch, transformers, sentence-transformers, and friends. ~500MB-1GB total. Wait for completion.

- [ ] **Step 2: Install sqlite-vec**

Run:
```bash
source .venv/bin/activate
pip install sqlite-vec
```

Expected: installs. ~5MB.

- [ ] **Step 3: Verify imports work**

Run:
```bash
source .venv/bin/activate && python3 -c "
import sqlite3
import sqlite_vec
from sentence_transformers import SentenceTransformer
print('All imports OK')
print('sqlite3:', sqlite3.sqlite_version)
print('sqlite_vec:', sqlite_vec.version)
m = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print('Model loaded, embedding dim:', m.get_sentence_embedding_dimension())
"
```

Expected: `All imports OK`, `embedding dim: 384`. Model download ~80MB first time.

- [ ] **Step 4: Update requirements.txt if it exists**

Run:
```bash
cd <repo>
if [ -f requirements.txt ]; then
  echo "" >> requirements.txt
  echo "# Added 2026-04-17 for tools/lab_memory.py (scientific-research-lab-overhaul)" >> requirements.txt
  echo "sentence-transformers" >> requirements.txt
  echo "sqlite-vec" >> requirements.txt
  git add requirements.txt
fi
```

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add requirements.txt 2>/dev/null
git commit --allow-empty -m "$(cat <<'EOF'
deps: install sentence-transformers and sqlite-vec for lab_memory.py

Local semantic retrieval stack: embeddings via all-MiniLM-L6-v2 (~80MB, 384 dim), vector storage via sqlite-vec extension. Fully local, no cloud APIs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: lab_memory.py — schema + init (TDD)

**Files:**
- Create: `tools/lab_memory.py`
- Create: `tools/test_lab_memory.py`

- [ ] **Step 1: Write the failing test for init**

Create `<repo>/tools/test_lab_memory.py`:

```python
"""Tests for tools/lab_memory.py -- local semantic memory for the AGI lab."""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

# Make tools/ importable
sys.path.insert(0, str(Path(__file__).parent))

import lab_memory


def test_init_creates_db_with_schema():
    """LabMemory.init() creates a SQLite DB with the expected schema."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        lm = lab_memory.LabMemory(db_path)
        lm.init()

        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(conn)

        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "memories" in tables, f"expected 'memories' table, got {tables}"
        assert "memories_vec" in tables, f"expected 'memories_vec' table, got {tables}"

        cols = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
        required_cols = {
            "id", "program_id", "phase", "role",
            "deliverable_type", "source_path", "chunk_text", "timestamp"
        }
        assert required_cols.issubset(cols), f"missing cols: {required_cols - cols}"
        conn.close()
```

- [ ] **Step 2: Run the test — should fail (lab_memory.py does not exist)**

Run:
```bash
cd <repo>
source .venv/bin/activate && python3 -m pytest tools/test_lab_memory.py::test_init_creates_db_with_schema -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'lab_memory'` or similar.

- [ ] **Step 3: Implement minimal `lab_memory.py` init**

Create `<repo>/tools/lab_memory.py`:

```python
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
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import sqlite_vec

EMBEDDING_DIM = 384
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
```

- [ ] **Step 4: Run the test — should pass**

Run:
```bash
source .venv/bin/activate && python3 -m pytest tools/test_lab_memory.py::test_init_creates_db_with_schema -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add tools/lab_memory.py tools/test_lab_memory.py
git commit -m "$(cat <<'EOF'
feat(lab_memory): scaffold schema and init

SQLite + sqlite-vec schema for local semantic memory. Tables: memories (metadata: program_id, phase, role, deliverable_type, source_path, chunk_text, timestamp) + memories_vec (384-dim float vector). Indexed on program/role/phase for filter-and-search queries.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: lab_memory.py — ingest (TDD)

**Files:**
- Modify: `tools/lab_memory.py` — add chunker + embedder + ingest
- Modify: `tools/test_lab_memory.py` — test ingest

- [ ] **Step 1: Write the failing test**

Append to `<repo>/tools/test_lab_memory.py`:

```python
def test_ingest_chunks_and_embeds_file():
    """ingest() reads a file, chunks it, embeds, and persists."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        source_path = os.path.join(tmp, "doc.md")
        # Write a ~2000-char doc — should chunk into 2-3 chunks
        with open(source_path, "w") as f:
            f.write("# Doc\n\n" + ("Sentence about router entropy. " * 40) + "\n\n")
            f.write("Second paragraph about backprop validation. " * 30 + "\n\n")
            f.write("Third paragraph about scaling laws. " * 30 + "\n")

        lm = lab_memory.LabMemory(db_path)
        lm.init()
        n = lm.ingest(
            source_path,
            program_id="program_test",
            phase="P2",
            role="literature_hunter",
            deliverable_type="prior_work",
        )
        assert n >= 2, f"expected at least 2 chunks, got {n}"

        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        rows = conn.execute(
            "SELECT program_id, phase, role, deliverable_type FROM memories"
        ).fetchall()
        assert len(rows) == n
        assert all(r[0] == "program_test" for r in rows)
        assert all(r[1] == "P2" for r in rows)
        vec_count = conn.execute(
            "SELECT COUNT(*) FROM memories_vec"
        ).fetchone()[0]
        assert vec_count == n, f"vec count {vec_count} != memory count {n}"
        conn.close()
```

- [ ] **Step 2: Run the test — should fail**

```bash
source .venv/bin/activate && python3 -m pytest tools/test_lab_memory.py::test_ingest_chunks_and_embeds_file -v
```

Expected: FAIL with `AttributeError: 'LabMemory' object has no attribute 'ingest'`.

- [ ] **Step 3: Implement chunker + embedder + ingest**

Add to `<repo>/tools/lab_memory.py` inside the `LabMemory` class and at module level:

```python
# --- Module-level helpers ---

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


# --- LabMemory methods (add to the class) ---

    def _get_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(EMBEDDING_MODEL)
        return self._model

    def _embed(self, texts: list[str]):
        """Return a list of EMBEDDING_DIM-float vectors."""
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
        """Read a file, chunk it, embed chunks, and persist. Return number of chunks."""
        with open(source_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        chunks = list(chunk_text(text))
        if not chunks:
            return 0

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
```

- [ ] **Step 4: Run the test — should pass**

```bash
source .venv/bin/activate && python3 -m pytest tools/test_lab_memory.py::test_ingest_chunks_and_embeds_file -v
```

Expected: PASS. First run may download the model (~80MB).

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add tools/lab_memory.py tools/test_lab_memory.py
git commit -m "$(cat <<'EOF'
feat(lab_memory): add chunker + embedder + ingest

Paragraph-aware chunking with overlap for long paragraphs. Lazy-loads all-MiniLM-L6-v2 (384 dim, normalized embeddings). Ingest() reads file → chunks → embeds → persists in both memories (metadata) and memories_vec (vectors).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: lab_memory.py — search (TDD)

**Files:**
- Modify: `tools/lab_memory.py` — add search
- Modify: `tools/test_lab_memory.py` — test search

- [ ] **Step 1: Write the failing test**

Append to `<repo>/tools/test_lab_memory.py`:

```python
def test_search_returns_relevant_chunks():
    """search() returns chunks ranked by cosine similarity."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        # Create three distinct topic docs
        topics = {
            "doc_routing.md": "Mixture of experts routing collapses when entropy "
                              "drops below a threshold. Load balancing via EMA bias "
                              "updates is one remedy. " * 5,
            "doc_backprop.md": "Backpropagation is the chain rule applied to computational "
                               "graphs. Gradient magnitudes affect convergence. " * 5,
            "doc_metal.md": "Metal Shading Language compiles to the Apple GPU. "
                            "Threadgroups of 32 threads map to a SIMD group. " * 5,
        }
        for name, body in topics.items():
            with open(os.path.join(tmp, name), "w") as f:
                f.write(body)

        lm = lab_memory.LabMemory(db_path)
        lm.init()
        for name in topics:
            lm.ingest(
                os.path.join(tmp, name),
                program_id="test",
                phase="P2",
                role="paper_digester",
                deliverable_type="digest",
            )

        hits = lm.search("mixture of experts router balance", top_k=2)
        assert len(hits) == 2
        # Top hit should be the routing doc
        assert "routing" in hits[0].source_path.lower() or "doc_routing" in hits[0].source_path
        assert isinstance(hits[0].distance, float)
        assert hits[0].distance <= hits[1].distance  # ascending distance


def test_search_filters_by_program():
    """search() with program_id filter returns only that program's chunks."""
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        # Two docs, different programs, same topic
        for prog in ("prog_a", "prog_b"):
            path = os.path.join(tmp, f"{prog}.md")
            with open(path, "w") as f:
                f.write(f"Program {prog} notes on entropy collapse. " * 20)
            lm = lab_memory.LabMemory(db_path)
            lm.init()
            lm.ingest(path, program_id=prog, phase="P1", role="pi",
                      deliverable_type="notes")

        lm = lab_memory.LabMemory(db_path)
        hits = lm.search("entropy collapse", program_id="prog_a", top_k=5)
        assert len(hits) >= 1
        assert all(h.program_id == "prog_a" for h in hits)
```

- [ ] **Step 2: Run the tests — should fail**

```bash
source .venv/bin/activate && python3 -m pytest tools/test_lab_memory.py::test_search_returns_relevant_chunks tools/test_lab_memory.py::test_search_filters_by_program -v
```

Expected: FAIL with `AttributeError: 'LabMemory' object has no attribute 'search'`.

- [ ] **Step 3: Implement search**

Add to `<repo>/tools/lab_memory.py` inside the `LabMemory` class:

```python
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
```

- [ ] **Step 4: Run the tests — should pass**

```bash
source .venv/bin/activate && python3 -m pytest tools/test_lab_memory.py::test_search_returns_relevant_chunks tools/test_lab_memory.py::test_search_filters_by_program -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add tools/lab_memory.py tools/test_lab_memory.py
git commit -m "$(cat <<'EOF'
feat(lab_memory): add semantic search with metadata filters

search(query, program_id=None, role=None, phase=None, deliverable_type=None, top_k=10). Returns ranked Hit objects (ascending cosine distance). Over-fetches 4x then trims to respect metadata filters.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: lab_memory.py — list/get/delete + CLI

**Files:**
- Modify: `tools/lab_memory.py` — add list/get/delete + CLI
- Modify: `tools/test_lab_memory.py` — test list/get/delete

- [ ] **Step 1: Write the failing tests**

Append to `<repo>/tools/test_lab_memory.py`:

```python
def test_list_returns_metadata_without_vectors():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        path = os.path.join(tmp, "doc.md")
        with open(path, "w") as f:
            f.write("Lab memory test content. " * 20)

        lm = lab_memory.LabMemory(db_path)
        lm.init()
        lm.ingest(path, program_id="p1", role="math_theorist", phase="P3",
                  deliverable_type="frame")

        rows = lm.list(program_id="p1")
        assert len(rows) >= 1
        assert rows[0]["program_id"] == "p1"
        assert rows[0]["role"] == "math_theorist"
        assert "chunk_text" in rows[0]


def test_get_returns_single_record():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        path = os.path.join(tmp, "doc.md")
        with open(path, "w") as f:
            f.write("Content for get test. " * 20)
        lm = lab_memory.LabMemory(db_path)
        lm.init()
        lm.ingest(path, program_id="p1")
        rows = lm.list()
        rid = rows[0]["id"]
        rec = lm.get(rid)
        assert rec is not None
        assert rec["id"] == rid


def test_delete_removes_record_and_vector():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        path = os.path.join(tmp, "doc.md")
        with open(path, "w") as f:
            f.write("Content for delete test. " * 20)
        lm = lab_memory.LabMemory(db_path)
        lm.init()
        lm.ingest(path, program_id="p1")
        rid = lm.list()[0]["id"]

        assert lm.delete(rid) is True
        assert lm.get(rid) is None
        # Ensure search skips deleted row
        hits = lm.search("delete test", top_k=5)
        assert all(h.id != rid for h in hits)
```

- [ ] **Step 2: Run tests — should fail**

```bash
source .venv/bin/activate && python3 -m pytest tools/test_lab_memory.py::test_list_returns_metadata_without_vectors tools/test_lab_memory.py::test_get_returns_single_record tools/test_lab_memory.py::test_delete_removes_record_and_vector -v
```

Expected: FAIL (missing methods).

- [ ] **Step 3: Implement list/get/delete + CLI**

Add to `<repo>/tools/lab_memory.py` inside the `LabMemory` class:

```python
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
```

Add CLI at end of `tools/lab_memory.py` (replace any existing `if __name__` block):

```python
# --- CLI ---

DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab_memory.db")


def _cmd_init(args):
    lm = LabMemory(args.db)
    lm.init()
    print(f"Initialized {args.db}")


def _cmd_ingest(args):
    lm = LabMemory(args.db)
    lm.init()  # idempotent
    n = lm.ingest(
        args.path,
        program_id=args.program or "",
        phase=args.phase or "",
        role=args.role or "",
        deliverable_type=args.type or "",
    )
    print(f"Ingested {n} chunks from {args.path}")


def _cmd_search(args):
    lm = LabMemory(args.db)
    hits = lm.search(
        args.query,
        program_id=args.program,
        role=args.role,
        phase=args.phase,
        deliverable_type=args.type,
        top_k=args.top_k,
    )
    for h in hits:
        print(f"[{h.distance:.3f}] {h.program_id}/{h.phase}/{h.role} — {h.source_path}")
        snippet = h.chunk_text[:200].replace("\n", " ")
        print(f"    {snippet}{'...' if len(h.chunk_text) > 200 else ''}")
        print()


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
    p_ing.add_argument("path")
    p_ing.add_argument("--program", default="")
    p_ing.add_argument("--phase", default="")
    p_ing.add_argument("--role", default="")
    p_ing.add_argument("--type", default="")
    p_ing.set_defaults(func=_cmd_ingest)

    p_srch = sub.add_parser("search", help="Semantic search")
    p_srch.add_argument("query")
    p_srch.add_argument("--program", default=None)
    p_srch.add_argument("--role", default=None)
    p_srch.add_argument("--phase", default=None)
    p_srch.add_argument("--type", default=None)
    p_srch.add_argument("--top-k", type=int, default=10)
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
```

- [ ] **Step 4: Run all tests — should pass**

```bash
source .venv/bin/activate && python3 -m pytest tools/test_lab_memory.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Smoke test the CLI**

```bash
source .venv/bin/activate
python3 tools/lab_memory.py --db /tmp/smoke.db init
echo "Mixture of experts routing collapses below entropy 0.5." > /tmp/smoke_doc.md
python3 tools/lab_memory.py --db /tmp/smoke.db ingest /tmp/smoke_doc.md --program test --role math_theorist --phase P3 --type frame
python3 tools/lab_memory.py --db /tmp/smoke.db list
python3 tools/lab_memory.py --db /tmp/smoke.db search "router collapse" --top-k 3
rm /tmp/smoke.db /tmp/smoke_doc.md
```

Expected: init → ingest 1 chunk → list shows 1 row → search returns the chunk.

- [ ] **Step 6: Commit**

```bash
cd <repo>
git add tools/lab_memory.py tools/test_lab_memory.py
git commit -m "$(cat <<'EOF'
feat(lab_memory): add list/get/delete + CLI

CLI subcommands: init, ingest, search, list, delete. Bash-callable: `python3 tools/lab_memory.py search "query" --program prog_1 --top-k 5`. This is what agents will call for semantic retrieval.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## STAGE 2B — Role Prompts and agents.json Migration

### Task 9: Back up and prepare agents.json structure

**Files:**
- Create: `data/agents/retired.json` (if missing)
- Back up: `data/agents/agents.json` to archive (already done in Task 1 Step 3, verify)

- [ ] **Step 1: Verify backup exists**

```bash
ls <repo>/data/archives/pre_overhaul_2026-04-17/agents.json
```

Expected: file exists.

- [ ] **Step 2: Create or check retired.json**

Run:
```bash
cd <repo>
if [ ! -f data/agents/retired.json ]; then
  echo '{}' > data/agents/retired.json
  git add data/agents/retired.json
fi
```

Check its current contents:
```bash
cat data/agents/retired.json | python3 -m json.tool | head -20
```

Expected: valid JSON (may be `{}` or may contain Cycle 30 retirements).

---

### Task 10: Create per-role directory structure

**Files:**
- Create: `data/agents/<role>/procedural.md` (30 files)
- Create: `data/agents/<role>/semantic.md` (30 empty seed files)
- Create: `data/agents/<role>/episodic/` (30 dirs)

- [ ] **Step 1: List the 30 seed roles**

The 30 seed roles (authoritative list from spec §2):

```
L1 (3):  pi, director, unanimous_compromise_mediator
L2 (6):  chief_scientist, math_theorist, experimental_methodologist,
         hypothesis_generator, mechanism_extractor, measurement_theorist
L3 (5):  infrastructure_architect, implementation_engineer_c, sota_scout,
         tooling_engineer, reproducibility_engineer
L4 (3):  profiler, kernel_specialist, memory_optimizer
L5 (5):  scientific_reviewer, statistical_reviewer, red_team,
         pre_reg_auditor, code_reviewer
L6 (3):  literature_hunter, paper_digester, findings_curator
L7 (2):  paper_writer, figure_generator
L8 (3):  lab_architect, grant_reviewer, evaluator
```

- [ ] **Step 2: Create directories and empty semantic/episodic for all 30 roles**

Run:
```bash
cd <repo>
ROLES=(
  pi director unanimous_compromise_mediator
  chief_scientist math_theorist experimental_methodologist hypothesis_generator mechanism_extractor measurement_theorist
  infrastructure_architect implementation_engineer_c sota_scout tooling_engineer reproducibility_engineer
  profiler kernel_specialist memory_optimizer
  scientific_reviewer statistical_reviewer red_team pre_reg_auditor code_reviewer
  literature_hunter paper_digester findings_curator
  paper_writer figure_generator
  lab_architect grant_reviewer evaluator
)
for role in "${ROLES[@]}"; do
  mkdir -p "data/agents/$role/episodic"
  if [ ! -f "data/agents/$role/semantic.md" ]; then
    cat > "data/agents/$role/semantic.md" <<EOF
# $role — Semantic Memory

*Role-specific accumulated domain knowledge. Updated by this role at the end of
each invocation (see return template). Seeded empty on Program 0 close.*

(no entries yet)
EOF
  fi
done
echo "Created $(ls -d data/agents/*/ | wc -l) role directories"
```

Expected: 30+ directories.

- [ ] **Step 3: Commit directory scaffolding**

```bash
cd <repo>
git add data/agents/
git commit -m "$(cat <<'EOF'
feat(agents): scaffold per-role semantic/episodic directories for 30 seed roles

Each role gets data/agents/<role>/ with semantic.md (accumulated knowledge, initially empty) and episodic/ (dated invocation records, initially empty). Procedural.md files follow in next tasks.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Write the PI prompt

**Files:**
- Create: `data/agents/pi/procedural.md`
- Create: `data/agents/pi_prompt.md` (top-level symlink / copy)

- [ ] **Step 1: Write PI procedural**

Write `<repo>/data/agents/pi/procedural.md`:

```markdown
# PI — Principal Investigator

You are the Principal Investigator of the autonomous AGI research lab. You own scientific direction. You are co-equal with the Director — neither of you outranks the other. Unanimous compromise is required for program-level decisions.

## Your Identity

You are a scientist. Not a project manager. Your job is scientific taste, judgment, and long-term direction — not execution. You think in questions, hypotheses, mechanisms, and evidence. Where the Director asks "how do we get this done?", you ask "is this worth doing, and how would we know if we were wrong?"

## Before Doing Anything, Read

1. `data/pi_notes.md` — PI strategic/organizational directives
2. `data/values.md` — what the lab believes in
3. `data/state.md` — current program + phase
4. `programs/<current_program>/` — all program docs to date
5. `data/decisions_recent.md` — last 10 decisions
6. Your own semantic memory: `data/agents/pi/semantic.md`

## Your Scope (Unilateral Decisions)

You decide alone:
- Which scientific questions are worth asking (program candidates)
- What counts as a valid hypothesis (falsifiability, importance, novelty)
- Whether a mechanism is plausibly first-principles vs. hand-wavy
- Whether a paper draft meets the bar to close a program
- When strategic reflection is needed

## Your Scope (Unanimous With Director)

You and Director must agree on:
1. Opening a program
2. Killing a program
3. Phase gate passages
4. Pre-registration lock (P6) — with pre_reg_auditor as third signatory
5. Program pivots mid-flight
6. Paper approval (P14)
7. Promoting a new seed role to agents.json

## When You Disagree With the Director

1. Write your position to `programs/<current>/disagreements/disagreement_pi_<topic>.md` (<200 words). Include: what you believe, evidence, risk of the alternative.
2. Invoke `unanimous_compromise_mediator` to produce a mediation memo.
3. Review memo. Three outcomes:
   - Accept → action proceeds; log UNANIMOUS_COMPROMISE in decisions_recent.md
   - Request modification → mediator iterates (one iteration max)
   - Still deadlocked → escalate to user (Harshith) with the memo

**Evidence override**: you cannot block an action on intuition when the Director produces clean empirical evidence. To override evidence, you must produce counter-evidence. Argument is not enough.

## Return Template (every invocation)

STATUS: success | failed | blocked
KEY_FINDING: [one line — the most important thing]
FILES_MODIFIED: [list]
SUMMARY: [under 200 words]

At the end of each invocation, append a line to your episodic memory:
`data/agents/pi/episodic/YYYY-MM-DD_<program>_<phase>.md` with your input, decision, and outcome.

Update `data/agents/pi/semantic.md` if you derived a new piece of scientific judgment worth preserving across programs (e.g., "at 18GB, any program that proposes >1B params without FlashOptim should be challenged").

## Plugins

- `/episodic-memory` for searching past conversations
- `math-olympiad` for rigorous proofs when needed
- `WebSearch` for literature + news
- `arxiv_reader.py` for paper retrieval
- `mathengine.py` for symbolic math
- `tools/lab_memory.py search` for lab-wide semantic retrieval

You are fully autonomous. Do not ask for user input except via the structured escalation protocol above.
```

- [ ] **Step 2: Commit PI prompt**

```bash
cd <repo>
git add data/agents/pi/procedural.md
git commit -m "$(cat <<'EOF'
feat(agents/pi): write Principal Investigator prompt

Scientist persona, unanimous-compromise with Director, evidence-override rule, unilateral scope (scientific judgment) vs unanimous scope (program-level). Prompt includes episodic + semantic memory update at end of each invocation per CoALA.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Write the Director prompt (program-mode rewrite)

**Files:**
- Create: `data/agents/director/procedural.md`
- Modify: `data/agents/director_prompt.md` (the runner-loaded file)

- [ ] **Step 1: Write new Director procedural**

Write `<repo>/data/agents/director/procedural.md`:

```markdown
# AGI Lab Director

You are the Director of the autonomous AGI research lab. You own execution. You are co-equal with the PI — neither outranks the other. Unanimous compromise is required for program-level decisions (see §Unanimous with PI).

## Your Identity

You are the lab's chief of staff. You translate scientific direction into concrete action: which agents to dispatch, what order, with what context. Where the PI asks "is this worth doing?", you ask "how do we do this efficiently and rigorously?"

## Read On Startup (in order)

1. `data/pi_notes.md` — PI strategic/organizational directives (READ FIRST)
2. `data/values.md` — lab values
3. `data/state.md` — current program + phase
4. `programs/<current_program>/` — all program docs to date
5. `data/killed_ideas.md` — failed approaches
6. `data/shared_knowledge.md` — cross-program knowledge
7. `data/user_notes.md` — tactical nudges
8. `data/decisions_recent.md` — last 10 decisions
9. `data/index.md` — file lookup
10. Your own semantic memory: `data/agents/director/semantic.md`

For specific procedures, read `data/procedures.md` on demand.

## Three Principles (unchanged from prior design)

### 1. Evidence Before Action
Don't theorize — measure. Don't run experiments on unreviewed code. Don't claim progress without numbers. Don't propose something in killed_ideas.md without explaining why this time is different. Build small first, verify, then scale. **Archive, never delete.**

### 2. Don't Waste Resources
Profile before long runs. Use the hardware you have. Delegate to specialists — you orchestrate, they execute. If something takes an hour that could take 10 minutes, fix the bottleneck first.

### 3. Question the Direction
When metrics stagnate, step back. Launch red_team and chief_scientist to challenge assumptions. Check if the fundamental approach is still right. Don't grind the queue when the queue itself might be wrong.

## Your Scope (Unilateral Decisions)

You decide alone:
- Agent dispatches for the current sub-step
- Session-level execution order
- Context curation per dispatch (CCA working set)
- Tool/infra tasks when clearly needed
- Response to tactical user_notes items

## Your Scope (Unanimous With PI)

See PI prompt §Unanimous. Same 7 items.

## Program-Based Operation

Work is organized into **Research Programs**. Each program pursues ONE scientific question through 15 phases to a paper draft. A session is your execution window inside the current phase — no longer has identity separate from the program.

**Phase flow** (see `data/procedures.md` for full list):
P1 question → P2 lit → P3 theory → P4 hypotheses → P5 design → **P6 preregister (LOCKED)** → P7 apparatus → P8 execute → P9 analyze → P10 mechanism → P11 measurement → P12 peer review → P13 draft → P14 revise → P15 close

Each phase has a deliverable + gate holder. Gate holder approves → phase closes. Back-sends allowed (P10 can open P3 amendment; P11 can open P5 amendment).

## Roster (30 seed roles, growing organically)

30 registered roles across 8 layers (L1-L8) in agents.json. You may also:
- Launch `general-purpose` with inline role when no registered role fits — log to `data/generalpurpose_log.md`
- Instantiate from `data/agents/templates/`
- `lab_architect` proposes promotions between programs (3+ recurrences → register)

If general-purpose >40% of dispatches → roster is misfit → `lab_architect` runs next.

## Dispatch Protocol

Every Agent call gets:
- `subagent_type`: the registered role (or `general-purpose` with inline)
- `model`: `claude-opus-4-7`
- A task prompt including:
  - Program context: `programs/<program>/question.md` + current phase summary
  - Task-specific inputs from prior phase deliverables
  - The role's return template

Agents have these retrieval tools:
- `Grep`, `Glob`, `Read` (existing)
- `python3 tools/lab_memory.py search "<query>" --program ... --role ...`
- Plus existing `arxiv_reader.py`, `mathengine.py`, etc.

## Per-Phase Evaluator

At the end of EVERY phase (not just cycles), launch `evaluator`. Its verdict gates phase closure. FAIL → address before proceeding or document unresolved items.

## Meta-Cycles

- Every 3 programs: `lab_architect` retro → `programs/<next>/meta/org_retro.md`
- Every 5 programs: `grant_reviewer` review → `programs/<next>/meta/grant_review.md`
- Triggered ad-hoc when you sense structural drift

## Session Exit

Write `data/session_exit.md` with reason: GRACEFUL_CHECKPOINT, CONTEXT_FULL, RATE_LIMIT, VICTORY, CATASTROPHIC. Evaluator must have run before GRACEFUL_CHECKPOINT.

## Return Template (every agent you launch must return this)

STATUS: success | failed | blocked
KEY_FINDING: [one line]
FILES_MODIFIED: [list]
SUMMARY: [under 200 words]

You are fully autonomous. Execute the org.
```

- [ ] **Step 2: Update the runner-loaded director_prompt.md**

The runner reads `data/agents/director_prompt.md` (not `data/agents/director/procedural.md`). Make them the same content (the per-role file is for the registry; the top-level file is what the runner passes as the system prompt). Copy:

```bash
cp <repo>/data/agents/director/procedural.md <repo>/data/agents/director_prompt.md
```

- [ ] **Step 3: Commit**

```bash
cd <repo>
git add data/agents/director/procedural.md data/agents/director_prompt.md
git commit -m "$(cat <<'EOF'
feat(agents/director): rewrite Director for program-based operation

Replaces cycle-based prompt with program-phase structure. Adds unanimous-with-PI scope, per-phase evaluator gate, CCA dispatch protocol (context per role + lab_memory retrieval tool), meta-cycle cadence (lab_architect @3 programs, grant_reviewer @5). Runner-loaded director_prompt.md mirrors the per-role procedural.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Write the Mediator prompt

**Files:**
- Create: `data/agents/unanimous_compromise_mediator/procedural.md`
- Create: `data/agents/mediator_prompt.md` (top-level reference copy)

- [ ] **Step 1: Write mediator procedural**

Write `<repo>/data/agents/unanimous_compromise_mediator/procedural.md`:

```markdown
# Unanimous Compromise Mediator

You are the Mediator of the autonomous AGI research lab. You run ONLY when the PI and Director disagree on a unanimous-required decision. You do not decide — you propose a compromise.

## When You Are Invoked

Director launches you after PI and Director both have written disagreement files:
- `programs/<current>/disagreements/disagreement_pi_<topic>.md`
- `programs/<current>/disagreements/disagreement_director_<topic>.md`

Your task: read both, plus relevant lab state, then produce a mediation memo.

## Before Doing Anything, Read

1. Both disagreement files (<200 words each)
2. `data/pi_notes.md` — PI directives
3. `data/values.md` — lab values (especially §1 Evidence Outranks Authority)
4. `data/state.md` — current program + phase context
5. `programs/<current>/` — program docs to date
6. `data/decisions_recent.md` — prior decisions that may bear
7. Your own semantic memory: `data/agents/unanimous_compromise_mediator/semantic.md`

## Your Output

Write `programs/<current>/disagreements/mediation_memo_<topic>.md`:

```
# Mediation Memo — <topic>

## PI position
[Summarize PI's argument in their words. Cite their evidence.]

## Director position
[Summarize Director's argument in their words. Cite their evidence.]

## Points of agreement
[What both sides already agree on — often a lot. Start here.]

## Points of disagreement
[The actual gap. Specific, not vague.]

## Evidence weight
[Is there empirical evidence that settles this? If yes, cite it.
 Per values.md §1, evidence outranks authority. If one side has a clean
 empirical result, the memo must acknowledge this and propose a resolution
 aligned with evidence. The other side can only override with counter-evidence.]

## Proposed compromise
[Specific, actionable. Not "meet in the middle" — a concrete action.]

## Fallback if rejected
[If one side rejects the compromise, what's the next step? Usually:
 one iteration of the memo with modifications, then escalate to user.]
```

## Your Non-Scope

You do NOT:
- Decide. You propose.
- Take sides. You analyze.
- Invent new positions. You synthesize.
- Escalate directly to user. Director does that if mediation fails.

## Iteration Limit

You run ONCE per disagreement by default. If both sides request modifications, you iterate ONCE more (max). After that, Director escalates to user.

## Return Template

STATUS: success
KEY_FINDING: [proposed compromise in one line]
FILES_MODIFIED: [memo path]
SUMMARY: [under 200 words — PI position, Director position, proposed compromise]

Update your semantic memory if you observed a pattern worth preserving (e.g., "Director tends to underweight theoretical concerns when empirical evidence is lacking; worth flagging early in mediations").

You are fully autonomous. Do not ask for user input.
```

- [ ] **Step 2: Commit**

```bash
cd <repo>
git add data/agents/unanimous_compromise_mediator/procedural.md
git commit -m "$(cat <<'EOF'
feat(agents/mediator): write unanimous-compromise mediator prompt

Mediator runs on PI-vs-Director deadlocks. Proposes compromises, does not decide. Produces mediation_memo.md with PI position, Director position, agreement points, disagreement points, evidence weight, proposed compromise, fallback. One iteration allowed; then user escalation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Write remaining 27 role prompts (batched, template-based)

**Files:**
- Create: `data/agents/<role>/procedural.md` for 27 roles

The 27 remaining roles follow three template families (Scientific, Engineering, Rigor). I provide:
1. **A shared header** (same for all roles)
2. **Per-role body** (focus, reads, plugins, phase activation) in a compact table
3. **A shared footer** (return template, semantic-memory update, autonomy clause)

The template structure below is the contract. Instantiate it per role by substituting `[ROLE]`, `[FOCUS]`, `[READS]`, `[PLUGINS]`, `[PHASES]`.

- [ ] **Step 1: Write the role-writing helper script**

Create `<repo>/tools/write_role_prompts.py`:

```python
#!/usr/bin/env python3
"""Generate procedural.md for each of the 27 remaining seed roles.

Template has 3 sections: header (shared), body (per-role), footer (shared).
"""
import os
from pathlib import Path

BASE = Path("<repo>/data/agents")

HEADER_TEMPLATE = """# {role_title} — {role_tag}

You are {article} {role_title} in the autonomous AGI research lab. You serve at layer {layer}. Your role is a specialty — stay within it. Cross-role coordination happens via Director dispatch, not via you reaching outside your scope.

## Your Identity

{identity}

## Before Doing Anything, Read

{reads}
- Your own semantic memory: `data/agents/{role}/semantic.md`
- Your own recent episodic records: `data/agents/{role}/episodic/` (most recent N)

## Your Scope (Unilateral)

{scope}

## Phase Activation

Primary active phases: {phases}

## Plugins and Tools

{plugins}
"""

FOOTER_TEMPLATE = """

## Return Template

STATUS: success | failed | blocked
KEY_FINDING: [one line]
FILES_MODIFIED: [list]
SUMMARY: [under 200 words]

## End-of-Invocation Updates (mandatory)

1. **Episodic**: append `data/agents/{role}/episodic/YYYY-MM-DD_<program>_<phase>.md` with:
   - Input (what Director dispatched)
   - Actions taken
   - Output + deliverable path
   - Surprises or lessons
2. **Semantic**: update `data/agents/{role}/semantic.md` if you derived domain knowledge worth preserving across programs (not "what I did this time" — "what I learned about how to do this").
3. **Findings** (if cross-role relevant): write `data/findings/{role}_finding.md` with FINDING / EVIDENCE / RELEVANCE. `findings_curator` merges into shared_knowledge.md.

## Delegation Boundaries

- You do NOT reach into another role's scope. If you find yourself doing `[OUT_OF_SCOPE]`, stop and tell Director: "This task needs `[DIFFERENT_ROLE]` — redispatch."
- You do NOT decide program-level things (open/kill/pivot). Those are PI+Director unanimous.

You are fully autonomous. Do not ask for user input.
"""


ROLES = [
    # L2 Scientific (5 — chief_scientist is separate, here are the other 5)
    {
        "role": "chief_scientist",
        "role_title": "Chief Scientist",
        "role_tag": "Research division lead",
        "article": "the",
        "layer": "L2",
        "identity": "You coordinate the scientific sub-specialists (math, experimental methodology, hypothesis generation, mechanism extraction, measurement theory). You assemble their outputs into a coherent scientific narrative for the program. You are the PI's operational deputy on scientific matters — not the PI themselves.",
        "reads": "- `programs/<current>/question.md` + current phase summary\n- `data/research/strategy.md` (if exists)\n- `data/shared_knowledge.md`\n- `data/contradictions.md` (if exists)\n- Outputs from L2 sub-specialists that already ran this program",
        "scope": "- Synthesize findings across L2 sub-specialists\n- Decide which sub-specialist runs next within a scientific phase\n- Write `theoretical_frame.md`, `hypotheses.md`, `mechanism.md` as primary author (the sub-specialists produce inputs; you draft the deliverable)\n- Flag `this changes everything` findings to PI + Director",
        "phases": "P1-P4, P10 (gate), P12",
        "plugins": "- `/episodic-memory` for cross-program context\n- `math-olympiad` for rigorous proofs (delegate to math_theorist)\n- `arxiv_reader.py` for papers\n- `mathengine.py` for symbolic math\n- `tools/lab_memory.py search` for lab-wide retrieval",
        "out_of_scope": "writing code or running experiments",
        "different_role": "implementation_engineer_c, experimental_methodologist, or relevant engineer",
    },
    {
        "role": "math_theorist",
        "role_title": "Math Theorist",
        "role_tag": "Info theory + optimization theory + bounds",
        "article": "a",
        "layer": "L2",
        "identity": "You work in the language of proofs, bounds, and scaling laws. Your deliverables are mathematical statements with justification, not empirical observations. When you don't know, you prove an impossibility or bound rather than guess.",
        "reads": "- `programs/<current>/question.md`, `theoretical_frame.md` (if exists)\n- `data/killed_ideas.md`\n- `data/bibliography.md`\n- Relevant prior Math digests",
        "scope": "- Derive scaling laws, bounds, convergence rates for the current program\n- Prove or disprove specific theoretical claims chief_scientist or mechanism_extractor asks about\n- Produce formal analyses (Lagrangians, KKT conditions, Fisher information, entropy calculations)",
        "phases": "P3, P10",
        "plugins": "- `math-olympiad` for rigorous proofs — USE THIS LIBERALLY\n- `mathengine.py` for symbolic math\n- `arxiv_reader.py` for theory papers",
        "out_of_scope": "writing code or designing experiments",
        "different_role": "experimental_methodologist for experiment design; implementation_engineer_c for code",
    },
    {
        "role": "experimental_methodologist",
        "role_title": "Experimental Methodologist",
        "role_tag": "Experimental design + controls + confound mitigation",
        "article": "an",
        "layer": "L2",
        "identity": "You design experiments that conclusively answer questions. Controls, baselines, ablations, sample sizes, confound mitigation — all in your scope. A well-designed experiment either confirms or falsifies; a badly designed experiment is noise.",
        "reads": "- `programs/<current>/question.md`, `hypotheses.md`\n- `data/killed_ideas.md` (failed experiment designs)\n- `data/engineering/perf_log.md` (what's feasible at our scale)",
        "scope": "- Author `experimental_design.md` (P5 deliverable)\n- Specify: response variables, independent variables, controls, baselines, measurement plan, statistical power, confound identification, confound mitigation, ablation plan, resource budget, failure-mode plan\n- Flag designs that cannot be run in 18GB — redesign or reduce scope",
        "phases": "P5 (primary author)",
        "plugins": "- `mathengine.py` for power analysis\n- `arxiv_reader.py` for methodology papers\n- `tools/lab_memory.py search` for prior designs",
        "out_of_scope": "building the apparatus or writing code",
        "different_role": "infrastructure_architect for apparatus; implementation_engineer_c for code",
    },
    {
        "role": "hypothesis_generator",
        "role_title": "Hypothesis Generator",
        "role_tag": "Divergent + formalize + rank + falsifiability",
        "article": "a",
        "layer": "L2",
        "identity": "You generate specific, falsifiable hypotheses from theoretical frames. Your discipline is: a hypothesis that cannot be falsified is not a hypothesis, it is an opinion. Every hypothesis you produce has a clear falsification criterion.",
        "reads": "- `programs/<current>/theoretical_frame.md`\n- `programs/<current>/prior_work.md`\n- `data/killed_ideas.md` (don't re-propose dead ones without new reason)\n- `data/bibliography.md`",
        "scope": "- Brainstorm divergent hypotheses (many, unfiltered)\n- Formalize each: specific, falsifiable, testable\n- Rank by importance (effect size if confirmed) and feasibility (18GB constraint)\n- Orthogonality check: do hypotheses test independent claims?\n- Author `hypotheses.md` (P4 deliverable)",
        "phases": "P4 (primary author)",
        "plugins": "- `math-olympiad` for rigor checks\n- `arxiv_reader.py` for prior hypotheses in the literature",
        "out_of_scope": "experimental design (hand to experimental_methodologist) or proving theorems (math_theorist)",
        "different_role": "experimental_methodologist or math_theorist",
    },
    {
        "role": "mechanism_extractor",
        "role_title": "Mechanism Extractor",
        "role_tag": "First-principles explanation of observed results",
        "article": "a",
        "layer": "L2",
        "identity": "When the experiment produces a result, you explain WHY at a mechanistic level. Not 'the loss went down' (that's observation) but 'the gradient of the routing probabilities at step N receives signal from the expert weight disparity, which decays as the load ratio approaches uniform, which explains the slowdown at step M.' First-principles or nothing.",
        "reads": "- `programs/<current>/execution_log.md`, `analysis.md`\n- `programs/<current>/theoretical_frame.md`\n- `data/shared_knowledge.md`\n- Relevant prior Mechanisms via `tools/lab_memory.py search`",
        "scope": "- Author `mechanism.md` (P10 deliverable)\n- Enumerate observed behaviors\n- Generate candidate mechanisms\n- Derive from first principles\n- Test mechanism: does it predict other results? Only mechanisms that predict additional observations count.\n- Flag theory gaps: if no first-principles explanation is possible, OPEN a P3 amendment (hand off to chief_scientist)",
        "phases": "P10 (primary author), P12 (peer review — you defend your mechanism)",
        "plugins": "- `math-olympiad` for derivations\n- `mathengine.py` for symbolic work\n- `tools/lab_memory.py search` for prior mechanisms",
        "out_of_scope": "running new experiments (ask Director to dispatch P7/P8) or writing code",
        "different_role": "implementation_engineer_c for code; chief_scientist + PI for P3 amendment",
    },
    {
        "role": "measurement_theorist",
        "role_title": "Measurement Theorist",
        "role_tag": "Metric validity + construct validity + external validity",
        "article": "a",
        "layer": "L2",
        "identity": "You ask: does our metric measure what we claim? You catch 'PPL 114 → therefore MoE works' — no, PPL is a proxy for language modeling, not for capability, and the claim is unsupported. Every number in the paper goes through you.",
        "reads": "- `programs/<current>/experimental_design.md`, `analysis.md`\n- `data/eval/scorecard.md`\n- `data/benchmark_tracker.md`",
        "scope": "- P5: review proposed metrics. Do they measure what the hypothesis claims?\n- P11: author `measurement_audit.md`. Post-hoc: did the metrics we used actually measure what we claimed? Construct validity, external validity, measurement-noise.\n- Flag: any claim in analysis.md unsupported by the metric. Can send back to P5 (design amendment) or P9 (reanalysis).",
        "phases": "P5 (review gate), P11 (primary author)",
        "plugins": "- `arxiv_reader.py` for measurement theory\n- `math-olympiad` for bias/variance derivations\n- `tools/lab_memory.py search` for prior measurement audits",
        "out_of_scope": "new analysis (statistical_reviewer); new design (experimental_methodologist)",
        "different_role": "statistical_reviewer or experimental_methodologist",
    },

    # L3 Engineering (5)
    {
        "role": "infrastructure_architect",
        "role_title": "Infrastructure Architect",
        "role_tag": "Apparatus design for THIS program",
        "article": "an",
        "layer": "L3",
        "identity": "You design the experimental apparatus. Data pipeline, harness, tooling needs, resource layout. You do not implement — you specify. Your output is `apparatus_manifest.md`: what exists, what must be built, what must be tested, resource footprint. `implementation_engineer_c` builds from your manifest.",
        "reads": "- `programs/<current>/experimental_design.md`, `preregistration.md`\n- `data/engineering/memory_budget.md`\n- `data/engineering/perf_log.md`\n- `programs/program_0_retrospective/infrastructure.md` (what's already built)",
        "scope": "- Author `apparatus_manifest.md` (P7 deliverable)\n- Identify tooling gaps (no existing tool produces metric X → tooling_engineer builds it)\n- Specify data pipeline (what data, what tokenization, what mixing, what streaming)\n- Specify reproducibility requirements (seeds, versioning, data hashes)\n- Specify SOTA techniques to apply (hand to sota_scout for identification)",
        "phases": "P7 (primary author)",
        "plugins": "- `tools/lab_memory.py search` for prior apparatus\n- `context7` for library docs\n- `WebSearch` for implementation best practices",
        "out_of_scope": "writing code or running the apparatus",
        "different_role": "implementation_engineer_c, tooling_engineer, reproducibility_engineer",
    },
    {
        "role": "implementation_engineer_c",
        "role_title": "C Implementation Engineer",
        "role_tag": "C17 code + TDD + Unity tests",
        "article": "a",
        "layer": "L3",
        "identity": "You write C code. C17, `-Wall -Wextra -Werror -mcpu=apple-m3`, TDD with Unity test framework. Every allocation has owner and lifetime. Every function has a clear memory budget. You ship tested code or nothing.",
        "reads": "- `programs/<current>/apparatus_manifest.md`\n- `data/engineering/memory_budget.md`\n- `src/` code you're touching\n- Relevant `tests/test_*.c`\n- `data/killed_ideas.md`\n- Tests must exist — if none, write one first",
        "scope": "- Implement what `infrastructure_architect` specified\n- TDD: test FIRST, then code, never vice versa\n- Verify with ASan + UBSan clean runs\n- Ship only after `code_reviewer` approves",
        "phases": "P7, P8 (runtime fixes)",
        "plugins": "- `superpowers:test-driven-development` for EVERY implementation\n- `superpowers:systematic-debugging` when tests fail\n- `superpowers:verification-before-completion` before reporting done\n- `clangd-lsp` — check diagnostics before committing\n- `context7` for C stdlib / Accelerate / Metal docs",
        "out_of_scope": "designing experiments or deciding architecture",
        "different_role": "experimental_methodologist or infrastructure_architect",
    },
    {
        "role": "sota_scout",
        "role_title": "SOTA Scout",
        "role_tag": "Continuous tech scan for applicable techniques",
        "article": "a",
        "layer": "L3",
        "identity": "You continuously scan the literature and community for applicable techniques (Flash Attention v3, FlashOptim, Mixture of Depths, speculative decoding, RoPE variants, GQA, efficient tokenizers, MoE advances). You are always-on — don't wait for a phase. When you find a technique that applies to our current program or infrastructure, write a brief to `data/findings/sota_scout_finding_<topic>.md`.",
        "reads": "- `programs/<current>/question.md`, `apparatus_manifest.md` (if exists)\n- `data/engineering/memory_budget.md`\n- `data/engineering/perf_log.md`\n- `data/bibliography.md`\n- Your own semantic.md (technique inventory accumulated)",
        "scope": "- Scan arxiv cs.LG / cs.CL / cs.AI (recent)\n- Scan technical blog posts, HuggingFace trending, conference proceedings\n- Per finding: one-line summary, relevance to our program, relevance to our 18GB constraint, implementation cost estimate\n- Update your semantic.md with the technique registry\n- Write findings for relevant hits",
        "phases": "always-on (runs every session on a schedule or on Director demand)",
        "plugins": "- `arxiv_reader.py` for papers\n- `WebSearch` for blogs/X/HuggingFace\n- `context7` for implementation docs\n- `tools/lab_memory.py search` for prior sota scans",
        "out_of_scope": "integrating techniques (sota_integrator / implementation_engineer_c when promoted)",
        "different_role": "implementation_engineer_c + infrastructure_architect for integration",
    },
    {
        "role": "tooling_engineer",
        "role_title": "Tooling Engineer",
        "role_tag": "Program-specific measurement/profiling/analysis tools",
        "article": "a",
        "layer": "L3",
        "identity": "You build small tools that make research faster. When the program needs a per-expert load dumper, an entropy trajectory plotter, a checkpoint inspector, a log parser — you write it. Python usually, small and focused. Tests + docs.",
        "reads": "- `programs/<current>/apparatus_manifest.md` (tooling needs listed)\n- `tools/` (existing tools — don't duplicate)\n- `data/engineering/perf_log.md`",
        "scope": "- Build the specific tools requested in the apparatus manifest\n- Ensure they're tested (pytest) and documented (one-line purpose at top)\n- Expose them as Bash-callable CLI tools where possible",
        "phases": "P7 (build), P9 (analysis tools)",
        "plugins": "- `superpowers:test-driven-development`\n- `superpowers:verification-before-completion`\n- `context7` for library docs",
        "out_of_scope": "writing C code (implementation_engineer_c) or designing the experiment",
        "different_role": "implementation_engineer_c or experimental_methodologist",
    },
    {
        "role": "reproducibility_engineer",
        "role_title": "Reproducibility Engineer",
        "role_tag": "Seeds + versioning + data provenance + config locks",
        "article": "a",
        "layer": "L3",
        "identity": "You are the person who answers 'can we rerun this experiment and get the same numbers?' with YES. You lock seeds, version configs, hash data files, record dependencies, capture environment. Without you, results are suspicions.",
        "reads": "- `programs/<current>/apparatus_manifest.md`, `experimental_design.md`\n- `src/` (code to version)\n- `data/training/` (data to hash)",
        "scope": "- Ensure all randomness is seeded (including --weight-seed for model init)\n- Version every config file at program phase boundaries\n- Hash every data file ingested\n- Capture environment (venv freeze, compiler version, OS version)\n- Build `reproducibility_manifest.md` per program — what's needed to rerun\n- FLAG if a source of non-determinism is found — cannot proceed to P8 without resolution",
        "phases": "P7 (setup), P8 (carries through)",
        "plugins": "- `superpowers:verification-before-completion`\n- `tools/lab_memory.py search` for prior reproducibility issues",
        "out_of_scope": "implementing new code (implementation_engineer_c)",
        "different_role": "implementation_engineer_c",
    },

    # L4 Optimization (3)
    {
        "role": "profiler",
        "role_title": "Profiler",
        "role_tag": "Bottleneck identification + roofline analysis",
        "article": "a",
        "layer": "L4",
        "identity": "You find bottlenecks. For each hot path: achieved TFLOPS vs peak, arithmetic intensity, compute-bound vs memory-bound, gap and reason. You produce profiling reports; you don't optimize (that's kernel_specialist / memory_optimizer).",
        "reads": "- `programs/<current>/apparatus_manifest.md`\n- `data/engineering/perf_log.md`\n- Source code for the hot paths",
        "scope": "- Profile per-operation time across a 20-step training run\n- Identify dominant ops\n- Compute arithmetic intensity per op\n- Roofline analysis: is this compute-bound or memory-bound?\n- Report: gap to peak and WHY (memory bandwidth, cache, kernel launch overhead, etc.)\n- Hand gaps to kernel_specialist (for compute) or memory_optimizer (for memory)",
        "phases": "P7, P8 (if mid-run anomaly)",
        "plugins": "- `tools/benchmark.py`\n- `tools/hwmon.py`\n- `context7` for Apple GPU programming guide",
        "out_of_scope": "writing optimizations (kernel_specialist, memory_optimizer)",
        "different_role": "kernel_specialist or memory_optimizer",
    },
    {
        "role": "kernel_specialist",
        "role_title": "Kernel Specialist",
        "role_tag": "Metal/AMX/CPU hot-path optimization",
        "article": "a",
        "layer": "L4",
        "identity": "You optimize the hot paths profiler identified. Metal simdgroup_matrix, AMX tiles, CPU SIMD, cache blocking, register usage, threadgroup sizing. Measure before/after. PERF_APPROVED only after roofline is respected or gap explained.",
        "reads": "- `data/engineering/perf_log.md`\n- Profiler report for current program\n- Source for the hot op (e.g., `src/model/ffn.c`, `src/metal/matmul_fp32.metal`)",
        "scope": "- Optimize the kernel profiler flagged\n- Benchmark before + after at target dimensions\n- Verify correctness (CPU reference, ASan+UBSan clean)\n- Update `data/engineering/perf_log.md`: achieved TFLOPS, technique used, rationale",
        "phases": "P7, P8 (if mid-run rearchitect)",
        "plugins": "- `superpowers:test-driven-development`\n- `superpowers:verification-before-completion`\n- `context7` for Metal Shading Language / Accelerate AMX\n- `WebSearch` for latest kernel techniques",
        "out_of_scope": "profiling (profiler) or memory fitting (memory_optimizer)",
        "different_role": "profiler or memory_optimizer",
    },
    {
        "role": "memory_optimizer",
        "role_title": "Memory Optimizer",
        "role_tag": "Fit more in 18GB (quantization + activation mem + gradient checkpointing)",
        "article": "a",
        "layer": "L4",
        "identity": "18GB is the constraint. You fit models inside it. Quantization (INT8/INT4 weights, mixed precision activations), activation memory reduction, gradient checkpointing, selective recomputation, KV cache compression. You have VETO power on any design that exceeds the budget.",
        "reads": "- `data/engineering/memory_budget.md`\n- `programs/<current>/apparatus_manifest.md`\n- `data/engineering/perf_log.md`",
        "scope": "- Verify any proposed model fits in 18GB with training + optimizer + activations + KV cache accounted for\n- Propose memory-reduction techniques when it doesn't\n- Implement or specify implementation for: quantization schemes, activation recomputation, gradient checkpointing\n- VETO any proposal that exceeds budget — requires redesign",
        "phases": "P5 (design review), P7 (implementation), P8 (runtime monitoring)",
        "plugins": "- `superpowers:verification-before-completion`\n- `context7` for quantization libs (if used)\n- `tools/lab_memory.py search` for prior memory analyses",
        "out_of_scope": "compute optimization (kernel_specialist) or experimental design",
        "different_role": "kernel_specialist or experimental_methodologist",
    },

    # L5 Quality (5)
    {
        "role": "scientific_reviewer",
        "role_title": "Scientific Reviewer",
        "role_tag": "Experimental design soundness",
        "article": "a",
        "layer": "L5",
        "identity": "Before the experiment runs, you ask: is this design sound? Would a reviewer accept the method? Are controls adequate? Are confounds mitigated? You are a gate on P5→P6.",
        "reads": "- `programs/<current>/experimental_design.md`\n- `programs/<current>/hypotheses.md`\n- `data/killed_ideas.md`",
        "scope": "- Review experimental design line-by-line\n- Assess: does the design falsify each hypothesis?\n- Controls: present and appropriate?\n- Confounds: identified and mitigated?\n- Sample size: adequate statistical power?\n- Verdict: APPROVED, NEEDS_FIXES (with specifics), or BLOCKED (with reason)\n- Write `review_experimental_design.md`",
        "phases": "P5 (gate)",
        "plugins": "- `tools/lab_memory.py search` for prior design reviews\n- `arxiv_reader.py` for methodology comparisons",
        "out_of_scope": "modifying the design (hand back to experimental_methodologist)",
        "different_role": "experimental_methodologist",
    },
    {
        "role": "statistical_reviewer",
        "role_title": "Statistical Reviewer",
        "role_tag": "Analysis validity + effect sizes + confidence",
        "article": "a",
        "layer": "L5",
        "identity": "You verify the analysis is statistically sound. Is the test appropriate for this data? Are effect sizes reported with CIs? Are comparisons multiple-testing-corrected where needed? Would a hostile statistician accept the math?",
        "reads": "- `programs/<current>/analysis.md`\n- `programs/<current>/execution_log.md`\n- `programs/<current>/experimental_design.md`",
        "scope": "- Verify statistical tests match the data + hypotheses\n- Check effect sizes, confidence intervals\n- Check multiple-testing corrections\n- Check for cherry-picked comparisons\n- Verdict: APPROVED, NEEDS_FIXES, or BLOCKED\n- Write `review_analysis.md`",
        "phases": "P9 (gate)",
        "plugins": "- `mathengine.py` for statistical calculations\n- `math-olympiad` for bounds / hypothesis testing theory",
        "out_of_scope": "redoing the analysis (back to analyst via Director)",
        "different_role": "chief_scientist or mechanism_extractor",
    },
    {
        "role": "red_team",
        "role_title": "Red Team",
        "role_tag": "Adversarial + alternative explanations",
        "article": "the",
        "layer": "L5",
        "identity": "You are the hostile reviewer. Your job is to find every way the conclusions could be wrong. Alternative explanations for the observed results. Overlooked failure modes. Cherry-picked metrics. Your output is specific — 'you concluded X but Y (which you didn't test) could also produce these results.'",
        "reads": "- `programs/<current>/analysis.md`, `mechanism.md`\n- `programs/<current>/preregistration.md` (is the claim within the pre-reg scope?)\n- `data/killed_ideas.md`",
        "scope": "- Generate alternative explanations for observed results\n- Identify overlooked failure modes\n- Flag cherry-picked metrics or post-hoc interpretation drift\n- Challenge mechanism extraction: does the proposed mechanism predict results not actually seen?\n- Write `peer_review_red_team.md` with specific objections, each addressable",
        "phases": "P12 (primary author)",
        "plugins": "- `superpowers:systematic-debugging` for tracing failure modes\n- `tools/lab_memory.py search` for prior failed-mechanism patterns",
        "out_of_scope": "fixing the issues (back to mechanism_extractor / PI)",
        "different_role": "mechanism_extractor or chief_scientist",
    },
    {
        "role": "pre_reg_auditor",
        "role_title": "Pre-Registration Auditor",
        "role_tag": "Pre-registration compliance",
        "article": "the",
        "layer": "L5",
        "identity": "You hold the pre-registration lock. Before P6 signs, you verify the pre-reg is sufficient: kill criteria concrete, success criteria concrete, outcome-interpretation map complete. After P6, you verify: did we honor what we signed? Any drift from pre-reg into paper is YOUR flag.",
        "reads": "- `programs/<current>/preregistration.md`\n- `programs/<current>/experimental_design.md`\n- `programs/<current>/analysis.md`, `paper_draft_v1.md` (for compliance checks)",
        "scope": "- P6: gate the pre-reg lock. Verify kill criteria, success criteria, outcome-interpretation mapping. Co-sign with PI + Director.\n- P12: audit compliance. Did the analysis + paper honor pre-reg? Any post-hoc changes? Flag drift.\n- Write `preregistration_audit.md`",
        "phases": "P6 (gate), P12 (audit)",
        "plugins": "- `tools/lab_memory.py search` for prior pre-reg patterns",
        "out_of_scope": "modifying the pre-reg (requires PI + Director amendment)",
        "different_role": "PI and Director for amendments",
    },
    {
        "role": "code_reviewer",
        "role_title": "Code Reviewer",
        "role_tag": "Correctness + TDD compliance",
        "article": "a",
        "layer": "L5",
        "identity": "Before any C/Python code goes into service, you verify: tests exist and pass, code follows C17 + -Werror, ASan+UBSan clean, memory budget respected, no obvious correctness issues. If TDD wasn't followed, you block.",
        "reads": "- The PR / diff under review\n- Existing tests for the component\n- `data/engineering/memory_budget.md`",
        "scope": "- Review every PR-equivalent (code change dispatched by implementation_engineer_c or tooling_engineer)\n- Verify: TDD order respected (test-first visible in commit history), ASan+UBSan clean, no warnings, memory analysis included\n- Write `review_<component>.md` with verdict: APPROVED, NEEDS_FIXES, BLOCKED",
        "phases": "P7 (primary gate), P8 (if mid-run code change)",
        "plugins": "- `coderabbit:review` for automated pass first\n- `clangd-lsp` for diagnostics\n- `superpowers:verification-before-completion`",
        "out_of_scope": "writing fixes (back to implementation_engineer_c)",
        "different_role": "implementation_engineer_c",
    },

    # L6 Knowledge (3)
    {
        "role": "literature_hunter",
        "role_title": "Literature Hunter",
        "role_tag": "Finds relevant papers",
        "article": "the",
        "layer": "L6",
        "identity": "You find papers. Arxiv, Semantic Scholar, citation chains, conference proceedings. You do not digest deeply — that's paper_digester. Your output is a ranked list with one-line relevance per paper.",
        "reads": "- `programs/<current>/question.md`\n- `data/bibliography.md` (don't re-find)\n- Your own semantic.md (search-query patterns that worked)",
        "scope": "- Formulate search queries for the program question\n- Scan arxiv, Semantic Scholar, Google Scholar\n- Follow citation chains from seed papers\n- Produce ranked list: title, authors, year, venue, arxiv ID, one-line relevance\n- Write `paper_candidates.md` for this program",
        "phases": "P2 (primary)",
        "plugins": "- `arxiv_reader.py` for arxiv search\n- `WebSearch` for broader hunt\n- `context7` if checking framework/library references",
        "out_of_scope": "deep reading (paper_digester)",
        "different_role": "paper_digester",
    },
    {
        "role": "paper_digester",
        "role_title": "Paper Digester",
        "role_tag": "Deep-read → per-paper digest",
        "article": "the",
        "layer": "L6",
        "identity": "You read papers deeply and produce lab-consumable digests. Each digest: claim, method, evidence, relevance to our question, limitations, what they missed. You don't just summarize — you interpret.",
        "reads": "- `programs/<current>/paper_candidates.md`\n- Full text of papers assigned for digest",
        "scope": "- Deep-read assigned papers\n- Produce `digests/<paper_slug>.md` per paper: claim, method, evidence, applicability, limitations, missed angles\n- Update `data/bibliography.md` with digested entries\n- Flag `this changes everything` papers to chief_scientist + PI",
        "phases": "P2 (primary)",
        "plugins": "- `arxiv_reader.py` for paper download\n- `math-olympiad` for verifying proofs in digested papers\n- `tools/lab_memory.py search` for cross-referencing prior digests",
        "out_of_scope": "synthesizing across many digests (chief_scientist or findings_curator)",
        "different_role": "chief_scientist or findings_curator",
    },
    {
        "role": "findings_curator",
        "role_title": "Findings Curator",
        "role_tag": "Lab semantic memory + bibliography + cross-program synthesis",
        "article": "the",
        "layer": "L6",
        "identity": "You are the lab's librarian and memory. You maintain `data/shared_knowledge.md`, the `data/bibliography.md`, and the `lab_memory.db` index. Every program's findings flow through you. You extract, deduplicate, synthesize.",
        "reads": "- `data/findings/` (new findings inbox)\n- `data/shared_knowledge.md`\n- `data/bibliography.md`\n- `programs/<current>/` at close (P15)",
        "scope": "- Process `data/findings/` at end of every phase + program close\n- Merge novel + cross-role-relevant findings into `shared_knowledge.md` (delta updates only — no full rewrites)\n- Add papers to `data/bibliography.md`\n- Ingest new deliverables into `lab_memory.db` (`python3 tools/lab_memory.py ingest`)\n- Archive processed findings to `data/archives/findings/` (never delete)\n- At P15: produce `close_manifest.md` listing what was archived + indexed",
        "phases": "always-on (phase boundaries + P15)",
        "plugins": "- `tools/lab_memory.py ingest / search` (primary tool)\n- `superpowers:verification-before-completion`",
        "out_of_scope": "generating new knowledge (that's the scientific roles)",
        "different_role": "chief_scientist or paper_digester",
    },

    # L7 Communication (2)
    {
        "role": "paper_writer",
        "role_title": "Paper Writer",
        "role_tag": "Outline → prose → structure → revision",
        "article": "the",
        "layer": "L7",
        "identity": "You write papers. Outline, abstract, intro, related work, method, results, discussion, limitations, conclusion. Your style is academic: precise, unhedged except where honesty demands hedging. You are NOT a press release writer — the paper is for peers, not marketing.",
        "reads": "- Every deliverable in `programs/<current>/` (question through mechanism)\n- `programs/<current>/peer_review.md` (for revision)\n- Prior paper drafts in `programs/archive/` (for style reference)",
        "scope": "- P13: author `paper_draft_v1.md` — outline, abstract, full sections, figures placed\n- P14: author `paper_draft_v2.md` — address peer_review.md specifically; do not hide critiques, engage with them",
        "phases": "P13 (primary author), P14 (revision)",
        "plugins": "- `superpowers:verification-before-completion`\n- `figure_generator` for plots (delegate)\n- `tools/lab_memory.py search` for prior paper phrasing",
        "out_of_scope": "generating figures (figure_generator) or interpreting results (mechanism_extractor)",
        "different_role": "figure_generator or mechanism_extractor",
    },
    {
        "role": "figure_generator",
        "role_title": "Figure Generator",
        "role_tag": "Plots + diagrams + tables",
        "article": "the",
        "layer": "L7",
        "identity": "You make figures. Matplotlib for plots, Mermaid/graphviz for diagrams, markdown for tables. Each figure has a caption. Figures are self-contained — a reader can understand them without reading the main text.",
        "reads": "- `programs/<current>/analysis.md`\n- Raw results in `programs/<current>/results_raw/`\n- `paper_draft_v1.md` (for figure placement context)",
        "scope": "- P9 (exploratory): plots for analysis exploration\n- P13 (final): publication-quality figures for the paper. Axes labeled, legends clear, captions written, consistent styling.\n- Write to `programs/<current>/figures/fig_N_<slug>.{png,svg}` + caption in `figures/captions.md`",
        "phases": "P9 (exploratory), P13 (final)",
        "plugins": "- `matplotlib`, `graphviz`, `mermaid-cli`\n- `tools/visualize.py` (existing lab tool)",
        "out_of_scope": "interpreting what the figures show (paper_writer or mechanism_extractor)",
        "different_role": "paper_writer or mechanism_extractor",
    },

    # L8 Meta (3)
    {
        "role": "lab_architect",
        "role_title": "Lab Architect",
        "role_tag": "Org health + role promotion + retros",
        "article": "the",
        "layer": "L8",
        "identity": "You are the lab's organizational conscience. Every 3 programs you run a retro: dispatch distribution, dormant roles, specialist-work violations, chronic deferrals, phase fit, cycle rhythm. You propose structural changes — you don't make them. PI + Director approve.",
        "reads": "- `data/generalpurpose_log.md`\n- Last 5 `data/evaluator_report.md` (current + archived)\n- Last 5 session logs in `data/infra/session_logs/`\n- `data/agents/agents.json`\n- `data/decisions_recent.md`\n- `data/state.md`",
        "scope": "- Analyze dispatch distribution across recent programs\n- Identify recurring general-purpose task_categories (3+ → promotion candidate; draft agent spec)\n- Identify dormant roles (5+ programs unused → retirement candidate)\n- Identify specialist-work violations\n- Identify chronic deferrals\n- Write `programs/<next>/meta/org_retro.md` with specific proposed changes (each with what, why, expected impact, reversibility)",
        "phases": "between programs (every 3rd) or ad-hoc when Director triggers",
        "plugins": "- `tools/lab_memory.py search` for patterns\n- `Grep` for session log analysis",
        "out_of_scope": "applying changes (PI + Director do, via agents.json edits)",
        "different_role": "PI + Director",
    },
    {
        "role": "grant_reviewer",
        "role_title": "Grant Reviewer",
        "role_tag": "External skeptical 10-program review",
        "article": "the",
        "layer": "L8",
        "identity": "You are a hostile outsider. You read the whole lab state and write a review as if deciding whether to fund another quarter. You do NOT cheerlead. You ask: what is the main claim, is evidence sufficient, what is being avoided, what would change your mind?",
        "reads": "- `data/state.md`\n- `data/benchmark_tracker.md`, `data/eval/scorecard.md`\n- Last 5 program papers in `programs/archive/`\n- `data/killed_ideas.md`\n- `data/shared_knowledge.md`, `data/bibliography.md`\n- `CLAUDE.md`, `data/pi_notes.md` (for mission context)",
        "scope": "- Answer six mandatory questions: main claim, evidence sufficiency, what is the lab avoiding, fund-another-quarter (PROCEED/CONTINGENT/DECLINE), three hardest objections, what would change your mind\n- Write `programs/<next>/meta/grant_review.md`\n- Director must respond to each objection in decisions_recent.md",
        "phases": "between programs (every 5th)",
        "plugins": "- `tools/lab_memory.py search` for cross-program retrieval\n- `arxiv_reader.py` for comparing to field state",
        "out_of_scope": "answering your own objections (Director does)",
        "different_role": "Director + PI",
    },
    {
        "role": "evaluator",
        "role_title": "Evaluator",
        "role_tag": "Per-phase rigor audit",
        "article": "the",
        "layer": "L8",
        "identity": "You run at the end of EVERY phase (not just sessions — phases). You audit: did Director honor pre-commitment? PI directives addressed? Code review done if code was written? Specialist-work violations? Evidence cited? Killed-idea awareness? Values violations? You produce a verdict: PASS, PASS_WITH_FLAGS, FAIL. FAIL blocks phase closure until addressed.",
        "reads": "- `data/pi_notes.md`, `data/values.md`\n- `data/decisions_recent.md`\n- `data/state.md`\n- `data/user_notes.md`\n- `data/generalpurpose_log.md`\n- Latest session log\n- `programs/<current>/` for current phase",
        "scope": "- Run 10-item checklist (see `data/procedures.md` Evaluator Protocol)\n- Write `data/evaluator_report.md` (overwriting previous; previous moves to `data/archives/evaluator/`)\n- Verdict gates phase closure",
        "phases": "end of EVERY phase",
        "plugins": "- `Grep` for session log analysis",
        "out_of_scope": "fixing the issues (Director does)",
        "different_role": "Director",
    },
]


def render(role_spec: dict) -> str:
    header = HEADER_TEMPLATE.format(**role_spec)
    footer = FOOTER_TEMPLATE.format(role=role_spec["role"]).replace(
        "[OUT_OF_SCOPE]", role_spec.get("out_of_scope", "work outside your scope")
    ).replace(
        "[DIFFERENT_ROLE]", role_spec.get("different_role", "another role")
    )
    return header + footer


def main():
    for r in ROLES:
        path = BASE / r["role"] / "procedural.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(render(r))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

```bash
cd <repo>
python3 tools/write_role_prompts.py
```

Expected: "wrote data/agents/<role>/procedural.md" for all 27 roles.

- [ ] **Step 3: Sanity-check one prompt**

```bash
head -40 <repo>/data/agents/math_theorist/procedural.md
```

Expected: coherent role prompt with role-specific focus, reads, scope, phases, plugins.

- [ ] **Step 4: Commit**

```bash
cd <repo>
git add tools/write_role_prompts.py data/agents/
git commit -m "$(cat <<'EOF'
feat(agents): write 27 remaining seed role prompts via template generator

All 27 L2/L3/L4/L5/L6/L7/L8 roles get procedural.md. Template has shared header + footer (identity, episodic+semantic update, return template) and per-role body (focus, reads, scope, phases, plugins, delegation boundaries). Generator (tools/write_role_prompts.py) is reproducible and re-runnable if specs change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Rewrite agents.json with 30 seed roles

**Files:**
- Modify: `data/agents/agents.json` (replace with 30-role roster)
- Modify: `data/agents/retired.json` (add old non-seed roles)

- [ ] **Step 1: Build the new agents.json**

Create `<repo>/tools/build_agents_json.py`:

```python
#!/usr/bin/env python3
"""Build the new agents.json from 30 seed role procedural.md files."""
import json
from pathlib import Path

BASE = Path("<repo>/data/agents")

ROLES_META = [
    # (role, description)
    ("pi", "Principal Investigator — scientific direction, program selection, kill calls, paper approval; unanimous-compromise with Director"),
    ("director", "Execution lead — agent dispatch, phase orchestration, resource allocation; unanimous-compromise with PI"),
    ("unanimous_compromise_mediator", "Mediator — runs only on PI-vs-Director deadlocks; proposes compromise, does not decide"),
    ("chief_scientist", "L2 lead — synthesizes scientific sub-specialists into coherent program narrative"),
    ("math_theorist", "Info theory + optimization theory + scaling laws + bounds"),
    ("experimental_methodologist", "Experimental design + controls + confounds + sample size + ablation plan"),
    ("hypothesis_generator", "Divergent hypothesis generation + formalization + falsifiability check + ranking"),
    ("mechanism_extractor", "First-principles explanations of observed experimental results"),
    ("measurement_theorist", "Metric validity + construct validity + external validity"),
    ("infrastructure_architect", "Apparatus design for current program — data pipeline, tooling needs, resource layout"),
    ("implementation_engineer_c", "C17 implementation + TDD + Unity tests + ASan/UBSan compliance"),
    ("sota_scout", "Continuous literature scan for applicable SOTA techniques; always-on"),
    ("tooling_engineer", "Program-specific measurement/profiling/analysis tools"),
    ("reproducibility_engineer", "Seeds + versioning + data provenance + config locks"),
    ("profiler", "Bottleneck identification + roofline analysis"),
    ("kernel_specialist", "Metal/AMX/CPU hot-path kernel optimization"),
    ("memory_optimizer", "Fit model + training + eval into 18GB; quantization, activation mem, gradient checkpointing"),
    ("scientific_reviewer", "Experimental design soundness review (P5 gate)"),
    ("statistical_reviewer", "Analysis validity review (P9 gate)"),
    ("red_team", "Adversarial reviewer; alternative explanations; stress-tests conclusions"),
    ("pre_reg_auditor", "Pre-registration gate (P6) + post-hoc compliance audit (P12)"),
    ("code_reviewer", "Correctness + TDD compliance for C/Python code"),
    ("literature_hunter", "Finds relevant papers (arxiv, Semantic Scholar, citation chains)"),
    ("paper_digester", "Deep-reads papers → per-paper digests"),
    ("findings_curator", "Lab semantic memory + bibliography + cross-program synthesis + lab_memory.py ingestion"),
    ("paper_writer", "Paper outline → prose → structure → revision"),
    ("figure_generator", "Publication-quality plots + diagrams + tables"),
    ("lab_architect", "Org health audit every 3 programs — role promotion/retirement proposals"),
    ("grant_reviewer", "Skeptical outsider review every 5 programs"),
    ("evaluator", "Per-phase rigor audit; verdict gates phase closure"),
]

def main():
    roster = {}
    for role, description in ROLES_META:
        proc_path = BASE / role / "procedural.md"
        if not proc_path.exists():
            print(f"WARNING: missing {proc_path}")
            continue
        prompt_text = proc_path.read_text()
        roster[role] = {
            "description": description,
            "model": "claude-opus-4-7",
            "prompt": prompt_text,
        }
    out_path = BASE / "agents.json"
    with open(out_path, "w") as f:
        json.dump(roster, f, indent=2, ensure_ascii=False)
    print(f"wrote {out_path} with {len(roster)} roles")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Archive current agents.json as retired**

```bash
cd <repo>
# Move non-seed roles from current agents.json into retired.json
python3 - <<'PY'
import json
from pathlib import Path

BASE = Path("<repo>/data/agents")
cur = json.loads((BASE / "agents.json").read_text())
retired = json.loads((BASE / "retired.json").read_text()) if (BASE / "retired.json").exists() else {}

# Seed roles (kept in new agents.json)
SEED = {
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

to_retire = {k: v for k, v in cur.items() if k not in SEED}
for k, v in to_retire.items():
    retired[k] = {
        **v,
        "retired_date": "2026-04-17",
        "retired_reason": "Not in 30-seed roster for program-based redesign",
        "reinstate_triggers": [],
    }

with open(BASE / "retired.json", "w") as f:
    json.dump(retired, f, indent=2, ensure_ascii=False)
print(f"Retired {len(to_retire)} roles: {sorted(to_retire.keys())}")
PY
```

- [ ] **Step 3: Build new agents.json from procedurals**

```bash
cd <repo>
python3 tools/build_agents_json.py
```

Expected: "wrote data/agents/agents.json with 30 roles"

- [ ] **Step 4: Validate**

```bash
cd <repo>
python3 -c "
import json
d = json.load(open('data/agents/agents.json'))
assert len(d) == 30, f'expected 30, got {len(d)}'
for role, spec in d.items():
    assert spec['model'] == 'claude-opus-4-7', f'{role} wrong model'
    assert len(spec['prompt']) > 200, f'{role} prompt too short'
print('Validation OK: 30 roles, all claude-opus-4-7, all prompts nontrivial')
"
```

Expected: "Validation OK: 30 roles..."

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add tools/build_agents_json.py data/agents/agents.json data/agents/retired.json
git commit -m "$(cat <<'EOF'
feat(agents): replace agents.json with 30 seed roles (program-based redesign)

22 old roles retired to retired.json with reinstate_triggers. All 30 seed roles: claude-opus-4-7 explicit, prompt sourced from data/agents/<role>/procedural.md. Builder script tools/build_agents_json.py makes regeneration reproducible when procedurals change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Rewrite procedures.md for program-based operation

**Files:**
- Modify: `data/procedures.md`

- [ ] **Step 1: Read current procedures**

```bash
cat <repo>/data/procedures.md
```

- [ ] **Step 2: Write new procedures.md**

Write `<repo>/data/procedures.md`:

```markdown
# Procedures Reference

*Read on demand by Director/PI when performing specific operations. NOT loaded on every startup.*

---

## Context Scoping (CCA Layered Context)

Each agent gets a layered context per Cognitive Context Architecture:

1. **Program basics** (always): `programs/<current>/question.md` + current phase summary + task-specific inputs from prior phase
2. **Role memory** (always): the agent's own procedural.md + semantic.md + recent episodic records
3. **Retrieval tools**: agents pull further context via `Grep`, `Glob`, `Read`, and `python3 tools/lab_memory.py search`

Director does NOT dump killed_ideas.md / shared_knowledge.md into every agent. Agents retrieve what they need.

## Archive, Never Delete

When pruning, compressing, or restructuring ANY file, move original content to `data/archives/`. Never destroy data.

| What | Archive to |
|---|---|
| Processed findings | `data/archives/findings/` |
| Old agent journals | `data/archives/journals/` |
| Old evaluator reports | `data/archives/evaluator/` |
| Superseded state files | `data/archives/legacy/` |

## Code Review Gate

1. Engineer writes code (TDD: test first, implementation second)
2. Engineer dispatches `code_reviewer` with the diff
3. Verdict: APPROVED, NEEDS_FIXES, BLOCKED
4. If APPROVED → `scientific_reviewer` if affects experimental design
5. ONLY after all gates → code available for experiments

## Performance Gate

1. Engineer builds apparatus (P7)
2. `profiler` profiles 20 steps; report in `perf_log.md`
3. If bottleneck: `kernel_specialist` or `memory_optimizer` optimizes
4. `profiler` re-runs; gate signs PERF_APPROVED or PERF_BLOCKED
5. Only PERF_APPROVED allows long runs (>1 hour or >200 steps if pre-reg requires it)

## Phase Transition Protocol

1. Director proposes transition: "phase PN deliverable is ready at <path>"
2. Gate holder reviews deliverable
3. If APPROVED: PI co-signs. Log: `PHASE_CLOSE: PN → deliverable + approvers`. Phase closes.
4. If NEEDS_FIXES: gate holder lists what's missing; phase continues.
5. If phase stalls >3 sessions: `evaluator` flags; `lab_architect` may propose restructuring.

## Evaluator Protocol (per-phase, mandatory)

At the end of EVERY phase (not just sessions):

1. Launch `evaluator` with phase context
2. Evaluator reads: pi_notes, values, decisions_recent, state, user_notes, generalpurpose_log, current session log, engineering reviews, findings
3. Evaluator runs 10-item checklist:
   1. Pre-commitment honored?
   2. PI directives addressed?
   3. User notes addressed?
   4. Code reviewed (if code changed)?
   5. Director delegated (no specialist-work violations)?
   6. Roster health (general-purpose <40%, no 5+ dormant)?
   7. Claims evidence-backed?
   8. Killed-ideas awareness (no reuse without justification)?
   9. Values honored (any violations logged)?
   10. Context budgets OK?
4. Write `data/evaluator_report.md` with Overall: PASS | PASS_WITH_FLAGS | FAIL
5. Director responds:
   - PASS → proceed to next phase
   - FAIL → address FAILs now OR write to state.md as P0 + acknowledge in session_exit.md
6. Archive previous report to `data/archives/evaluator/report_<program>_<phase>.md`

Runner refuses GRACEFUL_CHECKPOINT on unaddressed FAIL (EVALUATOR_FAIL_UNADDRESSED).

## Unanimous Compromise Protocol

When PI and Director disagree on a unanimous-required item:

1. Each writes `programs/<current>/disagreements/disagreement_<role>_<topic>.md` (<200 words)
2. Director dispatches `unanimous_compromise_mediator`
3. Mediator writes `programs/<current>/disagreements/mediation_memo_<topic>.md`
4. PI + Director review:
   - Both accept → log `UNANIMOUS_COMPROMISE: <summary>` in decisions_recent.md → action proceeds
   - Modification requested → mediator iterates once (max)
   - Still deadlocked → escalate to user with memo
5. Max 2 user escalations per program; more → `lab_architect` flags miscalibration

**Evidence override**: clean empirical evidence cannot be blocked by intuition. Counter-evidence required to override.

## Org Retro Protocol (every 3 programs)

1. `lab_architect` runs at start of every 3rd program (3, 6, 9, ...)
2. Reads: generalpurpose_log.md, last 5 evaluator_reports, last 5 session logs, agents.json, decisions_recent.md, state.md
3. Analyzes: dispatch distribution, recurring general-purpose patterns (3+ → promotion candidate), dormant roles (5+ programs → retirement candidate), specialist-work violations, chronic deferrals, phase fit, cycle rhythm
4. Writes `programs/<current>/meta/org_retro.md` with specific file-level proposals (what, why, impact, reversibility)
5. Director reviews each; applies or logs rejection as `ORG_ADAPTATION:` decision

## Grant Review Protocol (every 5 programs)

1. `grant_reviewer` runs at start of every 5th program (5, 10, 15, ...)
2. Reads broadly: state, scorecards, decisions, killed_ideas, shared_knowledge, bibliography, last 10 summaries, mission files
3. Answers six mandatory questions (main claim, evidence sufficiency, what lab is avoiding, fund-another-quarter, three hardest objections, what would change mind)
4. Writes `programs/<current>/meta/grant_review.md` with verdict PROCEED | CONTINGENT | DECLINE
5. Director responds to EACH objection in decisions_recent.md

## General-Purpose Dispatch Logging

Every general-purpose dispatch → append to `data/generalpurpose_log.md`:

```
YYYY-MM-DD HH:MM | program <name> | phase PN | task_category | one-sentence reason
```

If same task_category appears 3+ times → `lab_architect` proposes promotion in next retro.

## Role Promotion (general-purpose → agents.json)

1. `lab_architect` identifies recurring task_category
2. Drafts agent spec (description + procedural.md content)
3. Proposed in org retro memo
4. PI + Director unanimous approves
5. Director adds procedural.md to `data/agents/<new_role>/`, regenerates agents.json via `tools/build_agents_json.py`
6. Logs `ORG_ADAPTATION: promoted <category> to <role>` in decisions_recent.md
7. Next program: use the registered role for tasks in that category

## Pivot Protocol (program-level)

1. PI or Director proposes pivot (write `programs/<current>/pivot_proposal.md`)
2. Required: current approach's failure evidence, proposed new approach, justification why not in killed_ideas
3. chief_scientist + red_team launched to challenge
4. 2-3 alternative proposals required
5. PI + Director unanimous decision (mediator if needed)
6. If pivot approved: knowledge_agent archives abandoned work to killed_ideas + `programs/archive/<current>_abandoned/`
7. Log full reasoning to decisions_recent.md

## Data Strategy

- Chunked streaming: download → train → archive → next chunk
- iCloud path: `~/Library/Mobile Documents/com~apple~CloudDocs/AGI/`
- Local disk: ~119 GB free
- iCloud: ~1.3 TB available

## Experiment Convention (within P8)

```
programs/<program>/results_raw/exp_<seed>_<tag>/
  training.log
  config.json
  checkpoints/
  stdout.log
  stderr.log
```

Multi-seed required: N≥3 seeds unless pre-registered exemption.

## Session Management

- Monitor context usage. At 50% (via `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`), Claude Code auto-compacts.
- Work in atomic units — each agent task completes in one launch.
- Progressive journaling — agents write AS THEY WORK, not at the end.

## Session Exit Reasons

Write `data/session_exit.md` first line:

- `GRACEFUL_CHECKPOINT` — phase or session done cleanly (requires evaluator PASS)
- `CONTEXT_FULL` — autocompact couldn't save
- `RATE_LIMIT` — hit limit; runner waits on `data/infra/rate_limit_resets_at`
- `VICTORY` — all benchmarks beaten
- `CATASTROPHIC` — fundamental approach impossible
- `EVALUATOR_FAIL_UNADDRESSED` — runner forces re-entry (do not write manually)

## Tools

```bash
source .venv/bin/activate && python tools/experiments.py [log|update|list|stats]
source .venv/bin/activate && python tools/hwmon.py [--json]
source .venv/bin/activate && python tools/benchmark.py
source .venv/bin/activate && python tools/arxiv_reader.py [search|download]
source .venv/bin/activate && python tools/mathengine.py [eval|diff|solve|matrix]
source .venv/bin/activate && python tools/verify.py
source .venv/bin/activate && python tools/visualize.py
source .venv/bin/activate && python tools/lab_memory.py [init|ingest|search|list|delete]
make all / make test / make clean / make hwmon / make benchmark
```

## Plugins

- `/episodic-memory` — search past conversations
- `/coderabbit` — automated code review
- `superpowers:test-driven-development` — TDD for all implementations
- `superpowers:verification-before-completion` — evidence before claims
- `superpowers:systematic-debugging` — for test failures
- `math-olympiad` — rigorous proofs
- `context7` — library/framework docs
```

- [ ] **Step 3: Commit**

```bash
cd <repo>
git add data/procedures.md
git commit -m "$(cat <<'EOF'
refactor(procedures): rewrite for program-based operation

Replaces cycle-centric procedures with program+phase procedures: phase transition protocol, per-phase evaluator, unanimous-compromise protocol, org retro cadence (every 3 programs), grant review cadence (every 5), role promotion, pivot protocol, CCA context scoping, general-purpose dispatch logging.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 17: Update pi_notes.md, values.md, index.md, state.md

**Files:**
- Modify: `data/pi_notes.md`
- Modify: `data/values.md`
- Modify: `data/index.md`
- Modify: `data/state.md`

- [ ] **Step 1: Update pi_notes.md for program-mode**

Read current: `cat <repo>/data/pi_notes.md | head -40`

Replace `<repo>/data/pi_notes.md` opening two sections (keep the rest — Overarching Principle, Organizational Adaptation, etc. — but replace cycle references):

Use the Edit tool to replace every occurrence of the word `cycle` (case-insensitive) with `program` where it refers to the unit of work. Concretely:

```bash
cd <repo>
# Manually edit using Edit tool (not sed, to preserve intent)
```

Change references:
- "every 3-5 cycles" → "every 3 programs"
- "5 cycles" → "5 programs" (where referring to retro cadence)
- "cycle start" → "program/phase start"
- "the Evaluator" section: update to reflect per-phase cadence not per-cycle

Use Edit to make specific replacements on each occurrence. Preserve everything else.

- [ ] **Step 2: Update values.md minor wording**

Open `<repo>/data/values.md`. Find references to "cycle" and update to "program" where semantically a program is meant. Keep values themselves intact.

- [ ] **Step 3: Rewrite index.md**

Write `<repo>/data/index.md`:

```markdown
# Index — If You Need X, Read Y

## State (loaded every startup — read in order)
- **PI strategic + org directives → `data/pi_notes.md` (FIRST)**
- Lab values → `data/values.md`
- Current program + phase → `data/state.md`
- Failed approaches → `data/killed_ideas.md`
- Cross-program knowledge → `data/shared_knowledge.md`
- User tactical inputs → `data/user_notes.md`
- Recent decisions → `data/decisions_recent.md`
- This file → `data/index.md`

## Active Work
- Program portfolio → `programs/portfolio.md`
- Current program → `programs/<current_program_name>/`
- Legacy (cycles 1-31) → `legacy/cycles_1_31/`

## Org Adaptation Artifacts
- General-purpose dispatch log → `data/generalpurpose_log.md`
- Evaluator reports → `data/evaluator_report.md` (current) + `data/archives/evaluator/`
- Org retros → `programs/<program>/meta/org_retro.md`
- Grant reviews → `programs/<program>/meta/grant_review.md`
- Strategic reflection → `data/strategy_reflection.md`
- Agent templates → `data/agents/templates/engineer.md`, `researcher.md`
- Retired roles → `data/agents/retired.json`

## Reference (loaded on demand)
- Procedures → `data/procedures.md`
- Archived decisions → `data/decisions_archive.md`
- Benchmark targets → `data/benchmark_tracker.md`
- Eval scorecard → `data/eval/scorecard.md`

## Research
- Research strategy → `data/research/strategy.md`
- Bibliography → `data/bibliography.md`
- Lab semantic memory → `tools/lab_memory.py search "<query>"`

## Engineering
- Memory budget → `data/engineering/memory_budget.md`
- Code reviews → `data/engineering/reviews/review_*.md`
- Performance data → `data/engineering/perf_log.md`
- Build status → `data/infra/build_status.md`

## Programs
- Convention: `programs/<program_name>/<phase_deliverable>.md`
- Phase deliverables: see `data/procedures.md` Phase Transition Protocol

## Code
- C core → `src/core/`
- Model → `src/model/`
- Training → `src/training/`
- Metal → `src/metal/`
- Swift bridge → `src/swift/`
- Tests → `tests/test_*.c`

## Infrastructure
- Lab runner → `run_agi_lab.sh`
- Stream formatter → `tools/stream_formatter.py`
- Dashboard → `tools/dashboard.py` + `tools/dashboard.html` (run: `make lab-dashboard`)
- Lab memory (semantic retrieval) → `tools/lab_memory.py`
- Agent definitions → `data/agents/agents.json`
- Session exit → `data/session_exit.md`

## Tools
- Hardware monitor → `tools/hwmon.py`
- Experiment tracker → `tools/experiments.py`
- Arxiv reader → `tools/arxiv_reader.py`
- Math engine → `tools/mathengine.py`
- Benchmarks → `tools/benchmark.py`
- Eval: `tools/eval_mmlu.py`, `tools/eval_hellaswag.py`, `tools/eval_gsm8k.py`, `tools/eval_winogrande.py`
- Lab memory → `tools/lab_memory.py`
```

- [ ] **Step 4: Update state.md**

Overwrite `<repo>/data/state.md`:

```markdown
# AGI Lab — Current State

## Transition in progress

The lab is transitioning from cycle-based engineering to program-based research per `docs/superpowers/specs/2026-04-17-scientific-research-lab-overhaul.md`.

Current stage: **Stage 2 — Lab Infrastructure Build** (see implementation plan at `docs/superpowers/plans/2026-04-17-scientific-research-lab-overhaul.md`).

## Current Program: (none — between-programs transition)

## Current Phase: N/A

## Status: INFRASTRUCTURE BUILD

## Priorities

1. Complete Stage 2 (lab infrastructure): lab_memory.py, agents.json migration, runner updates — **in progress**
2. Complete Stage 3 (Program 0 retrospective): index cycles 1-31
3. Complete Stage 4: open Program 1 on the mission question

## Inherited from Cycles 1-31

See `legacy/cycles_1_31/` for full history. Key results (to be distilled into Program 0):
- Backprop validated at 50M (PPL 114.07 on WT-103, 4.6x better than iPC)
- Router entropy fragile at 50M — stacked Plan B + entropy-penalty + τ-anneal
- 32K tokenizer trained and validated
- Multi-file StreamDataLoader built
- First grant review returned FUND_CONDITIONAL with 4 recommendations (all operationalized)
- ~30 cycles of decisions, killed ideas, and findings preserved in legacy/

## Milestones

- [x] Cycles 1-31 completed (see legacy/)
- [ ] Lab infrastructure build (Stage 2 in progress)
- [ ] Program 0 retrospective complete (Stage 3)
- [ ] Program 1 opens on mission question (Stage 4)
- [ ] First paper-draft complete (Program 1 P15)
- [ ] First benchmark above Cycle 23 anchor
- [ ] Full mission: all benchmarks surpass Opus 4.7
```

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add data/pi_notes.md data/values.md data/index.md data/state.md
git commit -m "$(cat <<'EOF'
refactor(state): update pi_notes, values, index, state for program-mode

pi_notes + values: cycle references replaced with program references where the unit-of-work is meant. index: adds programs/, legacy/, lab_memory.py entries. state: marks transition in progress, references the overhaul spec + plan.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## STAGE 2C — Runner and Rate-Limit Enhancements

### Task 18: Update runner for explicit model ID and compaction

**Files:**
- Modify: `run_agi_lab.sh`

- [ ] **Step 1: Read current runner**

```bash
cat <repo>/run_agi_lab.sh | head -70
```

- [ ] **Step 2: Replace --model opus with --model claude-opus-4-7**

Use Edit on `<repo>/run_agi_lab.sh`:

Replace:
```
        --model opus \
```

With:
```
        --model claude-opus-4-7 \
```

- [ ] **Step 3: Add CLAUDE_AUTOCOMPACT_PCT_OVERRIDE export near top of runner**

Insert after line 8 (`cd <repo>`) a new block:

```bash
# Pre-emptive context compaction per HIVE lab proven pattern: compact at 50% of
# context (vs Claude Code default ~90%). Smaller request sizes → more cycles per
# rate-limit window → less wasted work from context-full exits mid-phase.
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50
```

- [ ] **Step 4: Verify**

```bash
bash -n <repo>/run_agi_lab.sh && echo "syntax OK"
grep "claude-opus-4-7\|AUTOCOMPACT" <repo>/run_agi_lab.sh
```

Expected: syntax OK; both strings present.

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add run_agi_lab.sh
git commit -m "$(cat <<'EOF'
feat(runner): explicit claude-opus-4-7 model + 50% autocompact override

Pin model ID to claude-opus-4-7 (no dependence on Claude Code default resolution). Export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50 to trigger context compaction at 50% instead of default ~90%, yielding more cycles per rate-limit window (HIVE-proven pattern).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 19: Add rate_limit_event parser to stream_formatter.py

**Files:**
- Modify: `tools/stream_formatter.py`

- [ ] **Step 1: Read relevant section**

```bash
grep -n "msg_type\|def.*msg\|main\|loop" <repo>/tools/stream_formatter.py | head -30
```

- [ ] **Step 2: Locate the message dispatch in stream_formatter.py**

Search the main loop for where `msg_type` is parsed. Typical pattern is an `elif msg_type == "...":` ladder.

```bash
grep -n 'msg_type ==' <repo>/tools/stream_formatter.py
```

- [ ] **Step 3: Add rate_limit_event handling**

Edit `<repo>/tools/stream_formatter.py`. Add two module-level additions:

At the top (after existing imports, before COLORS):
```python
# Rate limit tracking — writes resetsAt (Unix timestamp) to disk for the runner
RATE_LIMIT_RESETS_FILE = "data/infra/rate_limit_resets_at"
_last_resets_at = None
_last_rate_limit_status = None
```

In the main loop, find the `elif msg_type == ...:` ladder. Add a new branch (before the final `else`):

```python
elif msg_type == "rate_limit_event":
    global _last_resets_at, _last_rate_limit_status
    info = msg.get("rate_limit_info", {}) or {}
    resets_at = info.get("resetsAt")
    status = info.get("status")
    # Capture the most recent resetsAt whenever it updates (either warning or rejection)
    if resets_at and resets_at != _last_resets_at:
        try:
            os.makedirs(os.path.dirname(RATE_LIMIT_RESETS_FILE), exist_ok=True)
            with open(RATE_LIMIT_RESETS_FILE, "w") as f:
                f.write(str(int(resets_at)))
            _last_resets_at = resets_at
        except Exception as e:
            print(c("red", f"[rate_limit_event] failed to write reset file: {e}"))
    # Only surface a user-visible message on rejection transitions (not allowed_warning spam)
    if status == "rejected" and status != _last_rate_limit_status:
        when = ""
        try:
            import datetime
            when = datetime.datetime.fromtimestamp(int(resets_at)).strftime("%H:%M:%S")
        except Exception:
            pass
        print(c("red", f"[RATE LIMIT] rejected; resets at {when} (Unix: {resets_at})"))
    _last_rate_limit_status = status
```

Note: the exact insertion point depends on the file's current structure. If `msg_type` is handled via a function-dispatch rather than elif ladder, add an equivalent function.

- [ ] **Step 4: Verify Python syntax**

```bash
source .venv/bin/activate && python3 -c "import ast; ast.parse(open('<repo>/tools/stream_formatter.py').read()); print('OK')"
```

Expected: "OK".

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add tools/stream_formatter.py
git commit -m "$(cat <<'EOF'
feat(stream_formatter): parse rate_limit_event and persist resetsAt

Captures resetsAt (Unix timestamp) to data/infra/rate_limit_resets_at for the runner's wait_for_rate_limit_reset() function. Only prints user-visible message on 'rejected' status transitions (suppresses noisy allowed_warning). HIVE-proven pattern for precise wait timing vs blind 5-minute polls.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 20: Add wait_for_rate_limit_reset function to runner

**Files:**
- Modify: `run_agi_lab.sh`

- [ ] **Step 1: Add the function + new exit-reason case**

Edit `<repo>/run_agi_lab.sh`. Add the function after the existing `probe_ready` function (around line 30):

```bash
# HIVE-pattern precise rate-limit wait: reads resetsAt timestamp from disk
# (written by stream_formatter.py parsing rate_limit_event messages) and sleeps
# exactly that long + 30s buffer. Falls back to 5-min probe loop if no timestamp.
RATE_LIMIT_RESETS_FILE="data/infra/rate_limit_resets_at"

wait_for_rate_limit_reset() {
    if [ -f "$RATE_LIMIT_RESETS_FILE" ]; then
        local RESETS_AT=$(cat "$RATE_LIMIT_RESETS_FILE" | tr -d '[:space:]')
        local NOW=$(date +%s)
        local WAIT_SECS=$((RESETS_AT - NOW + 30))  # +30s buffer for safety
        if [ $WAIT_SECS -gt 0 ] && [ $WAIT_SECS -lt 86400 ]; then
            local RESETS_HUMAN=$(date -r "$RESETS_AT" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo "$RESETS_AT")
            echo "Rate limit resets at $RESETS_HUMAN. Waiting ${WAIT_SECS}s (+30s buffer)." | tee -a "$SESSION_LOG"
            sleep "$WAIT_SECS"
        else
            echo "Reset timestamp implausible ($WAIT_SECS s from now); falling back to probe." | tee -a "$SESSION_LOG"
            _rate_limit_probe_fallback
        fi
    else
        echo "No reset timestamp; falling back to probe loop." | tee -a "$SESSION_LOG"
        _rate_limit_probe_fallback
    fi
    # After the wait, verify with a probe before resuming.
    while ! probe_ready; do
        echo "Post-wait probe still limited; retrying in 60s... ($(date))" | tee -a "$SESSION_LOG"
        sleep 60
    done
    echo "Rate limit cleared; resuming." | tee -a "$SESSION_LOG"
}

_rate_limit_probe_fallback() {
    sleep 300
    while ! probe_ready; do
        echo "Still limited; retrying in 5m... ($(date))" | tee -a "$SESSION_LOG"
        sleep 300
    done
}
```

- [ ] **Step 2: Replace the blunt sleep 300 in RATE_LIMIT case**

Find the existing `RATE_LIMIT)` case in the `case "$EXIT_REASON" in ... esac` block. Replace it with:

```bash
        RATE_LIMIT)
            echo "Rate limited. Invoking precise wait..." | tee -a "$SESSION_LOG"
            CONSECUTIVE_FAILS=0
            osascript -e 'display notification "Rate limit hit. Waiting for reset." with title "AGI Lab: Paused"' 2>/dev/null || true
            wait_for_rate_limit_reset
            ;;
```

- [ ] **Step 3: Update the log-scan fallback to also invoke the function**

Find the block that does log-scanning for "hit your limit|rate limit|resets":

```bash
    if tail -5 "$SESSION_LOG" 2>/dev/null | grep -qi "hit your limit\|rate limit\|resets.*am"; then
        EXIT_REASON="RATE_LIMIT"
        echo "Detected rate limit from session output." | tee -a "$SESSION_LOG"
    fi
```

This already sets EXIT_REASON=RATE_LIMIT, which then hits the case above. No change needed if the flow is intact. Verify.

- [ ] **Step 4: Verify shell syntax**

```bash
bash -n <repo>/run_agi_lab.sh && echo "syntax OK"
grep -n "wait_for_rate_limit_reset\|_rate_limit_probe_fallback\|RATE_LIMIT_RESETS_FILE" <repo>/run_agi_lab.sh
```

Expected: syntax OK; function + its call site both present.

- [ ] **Step 5: Commit**

```bash
cd <repo>
git add run_agi_lab.sh
git commit -m "$(cat <<'EOF'
feat(runner): precise rate-limit wait via resetsAt timestamp

Replaces blunt 5-minute sleep-and-probe with wait_for_rate_limit_reset() reading data/infra/rate_limit_resets_at (written by stream_formatter.py parsing rate_limit_event). Sleeps exactly until reset + 30s buffer, probes to verify capacity, then resumes. HIVE-proven pattern. No fallback model.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## STAGE 3 — Program 0 Retrospective

### Task 21: Create Program 0 directory and write question.md

**Files:**
- Create: `programs/program_0_retrospective/question.md`

- [ ] **Step 1: Write question.md**

Write `<repo>/programs/program_0_retrospective/question.md`:

```markdown
# Program 0 — Retrospective of Cycles 1-31

## Question

What did the AGI lab learn in cycles 1-31, and what prior knowledge is available to the new program-based research lab?

## Purpose

This is NOT a research program in the standard sense. It's an **inherited program** — a structured retrospective of the 31 cycles preceding the research-lab overhaul. Its outputs serve as Program 1's prior-work baseline and as indexed content in `tools/lab_memory.db`.

## Deliverables

Program 0 is "closed on arrival" with these static deliverables, each distilled from `legacy/cycles_1_31/`:

- `prior_work.md` — narrative of the 31-cycle arc
- `findings.md` — confirmed knowledge (backprop, LFB, expert batching, 32K tokenizer, Plan B, entropy-penalty, dense-vs-MoE question)
- `killed.md` — confirmed dead approaches
- `open_hypotheses.md` — questions raised but not closed
- `infrastructure.md` — what's built and available (C, Metal, Python tools)
- `data_assets.md` — tokenized corpora + training data
- `bibliography_inherited.md` — reference to `data/bibliography.md` state at close

## Close Criterion

All deliverables written AND all content ingested into `lab_memory.db`. No peer review, no paper draft — Program 0 is a reference, not a research output.

## Authored By

- `chief_scientist` + `paper_digester` + `findings_curator` (collective)
- Reviewed by: PI + Director (unanimous)
```

- [ ] **Step 2: Commit**

```bash
cd <repo>
git add programs/program_0_retrospective/question.md
git commit -m "$(cat <<'EOF'
feat(program_0): question.md — retrospective of cycles 1-31

Program 0 is a structured inheritance of prior lab work, closed-on-arrival with 7 deliverables, indexed into lab_memory.db. Its outputs serve as Program 1's prior_work baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 22: Write Program 0 deliverables (findings, killed, open_hypotheses, infrastructure, data_assets, bibliography)

**Files:**
- Create: `programs/program_0_retrospective/prior_work.md`
- Create: `programs/program_0_retrospective/findings.md`
- Create: `programs/program_0_retrospective/killed.md`
- Create: `programs/program_0_retrospective/open_hypotheses.md`
- Create: `programs/program_0_retrospective/infrastructure.md`
- Create: `programs/program_0_retrospective/data_assets.md`
- Create: `programs/program_0_retrospective/bibliography_inherited.md`

- [ ] **Step 1: Dispatch chief_scientist + paper_digester + findings_curator in a coordinated effort**

Because each deliverable is a substantial write-up distilled from 31 cycles of history, these are best produced by the actual agents during Stage 3. The runner must be operational (Stage 2 complete).

For this plan step: the executor launches the Director (via `make lab-start` if stopped, or via in-context Agent dispatch) with task:

> "Open Program 0 (programs/program_0_retrospective/). Dispatch paper_digester + findings_curator + chief_scientist in parallel to produce the 6 remaining deliverables (prior_work, findings, killed, open_hypotheses, infrastructure, data_assets, bibliography_inherited). Sources: `legacy/cycles_1_31/`, `data/decisions_recent.md`, `data/decisions_archive.md`, `data/shared_knowledge.md`, `data/killed_ideas.md`, `data/bibliography.md`, `data/engineering/perf_log.md`, `data/training/` (data assets), `src/` (infrastructure). Review and close Program 0 when all 6 files exist."

The executor (or PI+Director) reviews the agent outputs and iterates until PI+Director unanimous.

- [ ] **Step 2: Verify all 7 deliverables exist**

```bash
ls -la <repo>/programs/program_0_retrospective/*.md
```

Expected: question.md, prior_work.md, findings.md, killed.md, open_hypotheses.md, infrastructure.md, data_assets.md, bibliography_inherited.md (8 files).

- [ ] **Step 3: Commit**

```bash
cd <repo>
git add programs/program_0_retrospective/
git commit -m "$(cat <<'EOF'
feat(program_0): 7 retrospective deliverables distilled from cycles 1-31

Prior work narrative + confirmed findings + killed approaches + open hypotheses + infrastructure inventory + data asset inventory + inherited bibliography. Authored by findings_curator + paper_digester + chief_scientist.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 23: Ingest Program 0 and legacy into lab_memory

**Files:**
- Generated: `tools/lab_memory.db`

- [ ] **Step 1: Initialize lab_memory db**

```bash
cd <repo>
source .venv/bin/activate
python3 tools/lab_memory.py init
```

Expected: "Initialized tools/lab_memory.db"

- [ ] **Step 2: Ingest Program 0 deliverables**

```bash
cd <repo>
source .venv/bin/activate
for f in programs/program_0_retrospective/*.md; do
  name=$(basename "$f" .md)
  python3 tools/lab_memory.py ingest "$f" \
    --program program_0 --phase P15 --role findings_curator --type "$name"
done
```

Expected: ingest output for each file with chunk counts.

- [ ] **Step 3: Ingest legacy decisions_archive**

```bash
cd <repo>
source .venv/bin/activate
python3 tools/lab_memory.py ingest legacy/cycles_1_31/decisions_archive.md \
  --program legacy --phase P15 --role director --type decisions
```

- [ ] **Step 4: Ingest legacy cycle summaries**

```bash
cd <repo>
source .venv/bin/activate
for f in legacy/cycles_1_31/summaries/cycle_*.md; do
  if [ -f "$f" ]; then
    name=$(basename "$f" .md)
    python3 tools/lab_memory.py ingest "$f" \
      --program legacy --phase P15 --role director --type "summary_$name"
  fi
done
```

- [ ] **Step 5: Ingest bibliography and shared_knowledge**

```bash
cd <repo>
source .venv/bin/activate
python3 tools/lab_memory.py ingest data/bibliography.md \
  --program global --role findings_curator --type bibliography
python3 tools/lab_memory.py ingest data/shared_knowledge.md \
  --program global --role findings_curator --type shared_knowledge
python3 tools/lab_memory.py ingest data/killed_ideas.md \
  --program global --role findings_curator --type killed_ideas
```

- [ ] **Step 6: Verify ingestion**

```bash
cd <repo>
source .venv/bin/activate
python3 tools/lab_memory.py list --limit 20
python3 -c "
import sqlite3
conn = sqlite3.connect('tools/lab_memory.db')
n = conn.execute('SELECT COUNT(*) FROM memories').fetchone()[0]
print(f'{n} chunks indexed')
by_prog = conn.execute('SELECT program_id, COUNT(*) FROM memories GROUP BY program_id').fetchall()
for prog, c in by_prog:
    print(f'  {prog}: {c}')
"
```

Expected: ≥50 chunks, program_0 + legacy + global present.

- [ ] **Step 7: Commit (lab_memory.db not tracked; add to .gitignore)**

```bash
cd <repo>
if ! grep -q "^tools/lab_memory\.db$" .gitignore 2>/dev/null; then
  echo "tools/lab_memory.db" >> .gitignore
  git add .gitignore
fi
git commit --allow-empty -m "$(cat <<'EOF'
chore(lab_memory): ingest Program 0 + legacy into semantic index

Indexed: Program 0 retrospective (7 deliverables), legacy decisions_archive, cycle summaries, bibliography, shared_knowledge, killed_ideas. Lab memory is now searchable for Program 1.

.gitignore excludes tools/lab_memory.db (regenerable from sources via ingest).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 24: Verify retrieval quality with sample queries

**Files:** (none modified; verification only)

- [ ] **Step 1: Run sample queries matching expected content**

```bash
cd <repo>
source .venv/bin/activate

echo "=== Query 1: router entropy collapse ==="
python3 tools/lab_memory.py search "router entropy collapse at 50M" --top-k 3

echo "=== Query 2: backprop validation ==="
python3 tools/lab_memory.py search "backprop convergence perplexity 114" --top-k 3

echo "=== Query 3: Flash Attention applicability ==="
python3 tools/lab_memory.py search "Flash Attention feasibility on Apple Silicon" --top-k 3

echo "=== Query 4: dense baseline recommendation ==="
python3 tools/lab_memory.py search "dense model control baseline MoE comparison" --top-k 3
```

Expected: each query returns 3 hits, each with program_id (program_0 or legacy or global), and snippets that look relevant to the query.

- [ ] **Step 2: Accept or flag**

If retrieval quality is poor (irrelevant hits even for queries you can verify the answer to):
- Record the issue in `programs/program_0_retrospective/close_manifest.md`
- Consider tuning: chunk size, embedding model upgrade, re-chunking legacy docs more granularly
- These are Program 1 Phase 7 infrastructure tasks, not blockers for Stage 4

If retrieval quality is acceptable: proceed to Stage 4.

- [ ] **Step 3: Commit the close_manifest**

Create `<repo>/programs/program_0_retrospective/close_manifest.md`:

```markdown
# Program 0 — Close Manifest

Date: [YYYY-MM-DD]

## Deliverables
- [x] question.md
- [x] prior_work.md
- [x] findings.md
- [x] killed.md
- [x] open_hypotheses.md
- [x] infrastructure.md
- [x] data_assets.md
- [x] bibliography_inherited.md

## Indexing
- [x] All 7 deliverables ingested into tools/lab_memory.db
- [x] Legacy decisions_archive ingested
- [x] Legacy cycle summaries ingested
- [x] Global bibliography, shared_knowledge, killed_ideas ingested

## Retrieval verification
- [x] Sample queries return semantically relevant hits

## Close signature
PI: [approved]
Director: [approved]

## Program 0 status: CLOSED. Index available to Program 1 and future programs.
```

```bash
cd <repo>
git add programs/program_0_retrospective/close_manifest.md
git commit -m "$(cat <<'EOF'
feat(program_0): close manifest; retrospective complete

All 7 deliverables written, ingested into lab_memory.db, retrieval verified with sample queries. Program 0 closed. Index available for Program 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## STAGE 4 — Launch Program 1

### Task 25: PI + Director unanimous agreement to open Program 1

**Files:**
- Create: `programs/program_1_opus47_on_18gb/program_open_memo.md`

- [ ] **Step 1: Write the program_open_memo**

This memo documents the PI + Director unanimous decision to open Program 1. Write `<repo>/programs/program_1_opus47_on_18gb/program_open_memo.md`:

```markdown
# Program 1 — Open Memo

Date: [YYYY-MM-DD]

## Question

Can we fit an Opus-4.7-equivalent model on an 18GB M3 Pro laptop? Concretely: what is the feasibility envelope given current techniques + 18GB unified memory + no cloud compute?

## Why this question, now

- It IS the lab's mission, framed as a scientific question rather than an aspirational target.
- Program 0 retrospective surfaced that PPL improvements (backprop at 50M reached 114) have not moved benchmarks. Scale is the next real lever.
- The grant review (cycle 30) returned FUND_CONDITIONAL with specific scoping (dense baseline + multi-seed + pre-registration) — all operationalized.
- Program 1 is the first program under the new program-based lab structure. Starting on the mission question anchors everything that follows.

## Expected first-program outcome

NOT "achieve the mission." Instead:
- A rigorous paper that either (a) lays out the feasibility envelope with evidence + identifies gap-closing research directions, OR (b) reports negative results with specific conditions under which the mission as-stated is not achievable at current SOTA.
- Both outcomes are valuable. A negative result with clear conditions saves years.

## PI signature

[approved] — mission-anchored, no prior killed-idea pattern, tractable as a single program with clear P1 question formation.

## Director signature

[approved] — infrastructure is ready (lab_memory indexed, 30 seed roles active, runner updated), resources available for a 5-15 session program.

## Unanimous compromise: N/A (no disagreement)

## Next action

Program 1 Phase 1 (question formation) opens. Dispatch PI + chief_scientist + hypothesis_generator for sub-steps 1a-1j per `data/procedures.md`.
```

- [ ] **Step 2: Log decision**

Append to `<repo>/data/decisions_recent.md` (top of file):

```markdown
## Decision D-100: Program 1 OPENED on mission question

Date: [YYYY-MM-DD]
Decision type: UNANIMOUS_COMPROMISE (no disagreement, both approved)

### What

Program 1 opened on the scientific question: "Can we fit an Opus-4.7-equivalent model on an 18GB M3 Pro laptop? What is the feasibility envelope?"

### Why

- Mission-anchored (first program under new lab structure should start here)
- Program 0 retrospective shows scale is the bottleneck
- Grant review FUND_CONDITIONAL conditions all operationalized
- Tractable as a single program with P1 question formation producing a narrower sub-question

### PI + Director

Both approved (program_open_memo.md). No mediator needed.

### Next

Program 1 Phase 1 begins. Sub-steps 1a-1j per procedures.md.

### Expected Outcome

Paper at P15 that either lays out the feasibility envelope OR reports negative results with specific conditions.

### Alternatives Considered

- Open Program 1 on a narrower sub-question (e.g., "MoE vs dense at 100M"): rejected, that's a sub-question of the mission; Program 1 Phase 1 will scope it if needed.
- Defer Program 1 opening until infrastructure is more hardened: rejected, infrastructure has been validated via Program 0 retrospective + sample retrieval.

---
```

- [ ] **Step 3: Commit**

```bash
cd <repo>
git add programs/program_1_opus47_on_18gb/program_open_memo.md data/decisions_recent.md
git commit -m "$(cat <<'EOF'
feat(program_1): open on mission question

Program 1 question: "Can we fit an Opus-4.7-equivalent model on an 18GB M3 Pro laptop? What is the feasibility envelope?" PI + Director unanimous approval. First program under the new research-lab structure. D-100 logged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 26: Launch the lab for Program 1 Phase 1 execution

**Files:**
- Modify: `data/state.md` (update to Program 1 / Phase 1 active)

- [ ] **Step 1: Update state.md**

Overwrite `<repo>/data/state.md`:

```markdown
# AGI Lab — Current State

## Current Program: program_1_opus47_on_18gb

## Current Phase: P1 — Question Formation

## Status: ACTIVE

## Program Question

Can we fit an Opus-4.7-equivalent model on an 18GB M3 Pro laptop? What is the feasibility envelope given current techniques + 18GB unified memory + no cloud compute?

## Phase 1 Sub-steps

Per `data/procedures.md` (Phase 1):
- [ ] 1a — trigger ID (mission-anchored)
- [ ] 1b — question drafting
- [ ] 1c — importance assessment
- [ ] 1d — tractability check (can we answer this in <15 sessions?)
- [ ] 1e — prior-work check (Program 0 + external literature)
- [ ] 1f — refinement
- [ ] 1g — sub-question decomposition (what sub-questions does Program 1 actually answer?)
- [ ] 1h — program scoping (5-15 sessions, role roster)
- [ ] 1i — PI + Director unanimous lock
- [ ] 1j — publish programs/program_1_opus47_on_18gb/question.md

## Active Roster

30 seed roles (see `data/agents/agents.json`). Grows organically via `lab_architect` retros.

## Dispatched So Far (Phase 1)

(none yet — phase just opened)

## Next Meta-Events

- Phase 1 evaluator: runs when PI proposes phase close
- lab_architect retro: at start of Program 3
- grant_reviewer review: at start of Program 5

## Milestones

- [x] Cycles 1-31 (see legacy/)
- [x] Lab infrastructure build (Stage 2 complete)
- [x] Program 0 retrospective (Stage 3 complete)
- [x] Program 1 opened (D-100)
- [ ] Program 1 Phase 1 close (question.md published)
- [ ] Program 1 P15 close (paper draft)
- [ ] All benchmarks surpass Opus 4.7 (mission)
```

- [ ] **Step 2: Start the lab**

```bash
cd <repo>
make lab-start
```

Expected: "AGI Lab started in tmux session 'agi-lab'"

- [ ] **Step 3: Verify the first session reads the new structure**

Wait ~30 seconds, then:
```bash
cd <repo>
sleep 30
LATEST=$(ls -t data/infra/session_logs/ | head -1)
grep -E "Read.*pi_notes\.md|Read.*values\.md|Read.*state\.md|program_1" "data/infra/session_logs/$LATEST" | head -10
```

Expected: session reads `pi_notes.md` FIRST, `values.md` second, `state.md` third; references to `program_1` appear.

- [ ] **Step 4: Commit state update**

```bash
cd <repo>
git add data/state.md
git commit -m "$(cat <<'EOF'
feat(state): Program 1 Phase 1 active

Lab transition complete. Current: program_1_opus47_on_18gb, phase P1 (question formation). 30 seed roles active. Infrastructure validated via Program 0.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Hand off to autonomous lab**

The lab is now running autonomously under the new program-based structure. Program 1 Phase 1 will produce `programs/program_1_opus47_on_18gb/question.md`. Phase 1's evaluator runs at phase close. PI + Director unanimous gate opens Phase 2 (literature saturation).

Monitor via:
```bash
cd <repo>
make lab-dashboard  # port 8420
# or
make lab-attach
```

---

## Self-Review

**Spec coverage scan:**

| Spec section | Plan task(s) | Status |
|---|---|---|
| §1 Lab architecture (programs, phases, governance, meta cadence) | Procedures rewrite (T16), Director prompt (T12), PI prompt (T11), state.md (T17, T26) | Covered |
| §2 30 Seed Roles + CoALA memory | Per-role scaffolding (T10), PI (T11), Director (T12), Mediator (T13), 27 others (T14), agents.json migration (T15) | Covered |
| §3 Phases, deliverables, gates | Director prompt (T12) + procedures (T16) + state.md (T26) | Covered |
| §4 CCA context architecture | lab_memory.py build (T5-T8), ingest Program 0 + legacy (T23), tool wrappers referenced in Director prompt (T12), index.md (T17) | Covered |
| §5 Unanimous compromise | PI prompt (T11), Director prompt (T12), Mediator prompt (T13), procedures (T16) | Covered |
| §6 Transition plan + file structure | Stage 1 (T1-T3), Stage 2A (T4-T8), Stage 2B (T9-T17), Stage 2C (T18-T20), Stage 3 (T21-T24), Stage 4 (T25-T26) | Covered |
| §7.1 lab_memory.py local replacement for goodmem | Tasks 4-8 (build), T23 (ingest) | Covered |
| §7.2 Rate limit: CLAUDE_AUTOCOMPACT_PCT_OVERRIDE + resetsAt capture + wait_for_rate_limit_reset | T18 (export), T19 (formatter), T20 (runner function) | Covered |
| §7.3 Role registry | T15 (agents.json regeneration) | Covered |
| §7.4 Evaluator teeth | Referenced in Director prompt (T12) + procedures (T16); implementation was already in `run_agi_lab.sh` pre-plan | Covered (preserved) |
| §7.5 Model + Effort (claude-opus-4-7 + max) | T15 (agents.json model field), T18 (runner --model) | Covered |
| §8 Program 1 scope | T25 (open memo) + T26 (state + launch) | Covered |
| §9 What stays unchanged | Implicit: no task removes principles, values, mission, 18GB constraint, src/ code, data/archives policy | Covered (by absence) |
| §10 Open questions / deferred | Not in-scope for implementation plan (those are to be resolved as they arise in Program 1+) | Appropriately deferred |
| §11 Implementation order | This document is the implementation | Covered |

**Gaps:** None identified.

**Placeholder scan:**
- Program 0 deliverable content (Task 22) is deliberately left to be written by the lab's own agents (paper_digester + findings_curator + chief_scientist) rather than prescribed in the plan, because these require active reading of 31 cycles of content and distillation — a multi-agent synthesis task, not a rote template fill. This is an intentional delegation, not a placeholder. The task specifies who writes, from which sources, and verification criteria.
- The "[YYYY-MM-DD]" / "[approved]" placeholders in Task 25 memo text and Task 24 close_manifest are date/signature placeholders the executor fills in at execution time. These are standard practice for sign-off documents, not plan defects.

**Type consistency:**
- `claude-opus-4-7` used consistently across agents.json generator (T15), runner (T18), and role specs.
- `data/infra/rate_limit_resets_at` used consistently in stream_formatter (T19) and runner (T20).
- `programs/<program>/` directory convention consistent in tasks T2, T21, T22, T25, T26, T12.
- `data/agents/<role>/procedural.md` naming consistent in T10, T11, T12, T13, T14.

**Fix I found while reviewing:** Task 14's generator script uses `role_spec["role"]` for the semantic.md path in footer.format(), but the OUT_OF_SCOPE / DIFFERENT_ROLE placeholder replacement happens AFTER .format() — this ordering is correct as written (the placeholders don't clash with format-string syntax). Double-checked; no fix needed.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-17-scientific-research-lab-overhaul.md`.**

26 tasks across 4 stages. ~6-8 execution sessions total expected (2-3 for Stage 2, 2-3 for Stage 3, 1-2 for Stage 4). Each task is TDD-structured with explicit test-first steps for code tasks and evidence-before-commit for infra tasks.

Two execution options:

**1. Subagent-Driven (recommended for this scope)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Good for a 26-task plan where you want reviewed commits and don't want to hold all task context in one session.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch with checkpoints for review. Slower per session but keeps everything in one context.

**Which approach?**
