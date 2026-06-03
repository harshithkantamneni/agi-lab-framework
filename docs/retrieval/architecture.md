# Retrieval Architecture

**Status:** shipped 2026-05-21 (Plan `docs/superpowers/plans/2026-05-20-memory-retrieval-upgrade.md`, T0.1–T5.3)
**Owner:** lab substrate (no single agent role; all roles read it, runner re-indexes it)
**Scope:** how agents in this lab find prior work — decisions, programs, episodic memory, engineering reviews, closure memos.

This document explains the 4-layer hybrid retrieval system that replaced flat dense semantic search across the lab's ~509-file, ~47K-chunk corpus. It also explains the choices we deliberately did NOT make (LLM entity extraction, community detection, Neo4j) and why those omissions are principled rather than incidental.

---

## 1. Overview

Retrieval is structured as four layers (L0–L3) plus an optional cross-encoder rerank (L4), orchestrated by `tools/retrieval/search.py`.

```
                    query string
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
        +-------+    +-------+    +--------+
        |  L0   |    |  L1   |    |   L2   |
        | graph |    | dense |    |  BM25  |
        |  PPR  |    | MiniLM|    | rank   |
        +-------+    +-------+    +--------+
            |             |             |
            |             +------+------+
            |                    v
            |              +-----------+
            |              |    L3     |
            |              |   RRF     |
            |              |  fusion   |
            |              +-----------+
            |                    |
            +-------> seed <-----+
                       |
                       v
                 +-----------+
                 |    L4     |
                 |  rerank   |
                 |  bge-v2   |
                 +-----------+
                       |
                       v
                  top-K results
```

| Layer | What it does                                                            | Substrate                  | Latency  | Index size |
| ----- | ----------------------------------------------------------------------- | -------------------------- | -------- | ---------- |
| L0    | Personalized PageRank over canonical-token graph (D-N, P-*, phases…)    | SQLite + networkx          | ~200 ms  | ~5 MB      |
| L1    | Dense MiniLM-L6-v2 (384-dim) ANN over chunks                            | sqlite-vec                 | ~50 ms   | ~150 MB    |
| L2    | BM25 lexical match over identical chunk set                             | rank\_bm25, JSON persisted | ~30 ms   | ~10 MB     |
| L3    | Reciprocal Rank Fusion (k=60) over L1+L2                                | pure Python                | <5 ms    | -          |
| L4    | Cross-encoder rerank top-30 → top-K (bge-reranker-v2-m3, ms-marco fallback) | sentence-transformers      | ~10–12 s warm | 568 MB on disk |

Full-pipeline warm latency: **~12 s** (dominated by L4 rerank on CPU). Without L4: **<300 ms** end-to-end.

The orchestrator is intentionally thin: each layer is independently invokable from the CLI (`python -m tools.retrieval.graph ppr …`, `python -m tools.retrieval.bm25 search …`, `python -m tools.retrieval.search …`) so agents can sidestep rerank latency when they want recall over precision.

---

## 2. Design Rationale: Canonical-Token Graph Instead of LLM Entity Extraction

The single most consequential decision in this design is **using regex-extracted canonical tokens as the graph's node vocabulary instead of LLM-extracted entities**. This is what separates us from Microsoft GraphRAG and its descendants.

The lab's substrate already produces a structured vocabulary as a byproduct of normal operation:

- `D-N` decision IDs (e.g. `D-420`, `D-446`) — referenced everywhere in `log.md`, retros, closure memos
- `P-*` carry-forward IDs (e.g. `P-D420-WORK-QUEUE-DEDUP-FIX`) — pinned at the head of `current.md` until resolved
- Program names (`Program 1`, `P15`, `Phase 11`) — used in dispatch, deliverables, papers
- Agent role names (`Director`, `lab_architect`, `engineering_lead`, …) — author attributions and reviewer pairings

These tokens form a graph **for free**. Two regex parsers — one for the token surface forms, one for adjacency within a chunk — produce nodes and edges in roughly the time it takes to walk the corpus on disk. The token graph rebuild for our current 509-file / ~47K-chunk corpus produces **4,533 nodes and 4,011 edges in ~100 s** of single-threaded Python, fully deterministic, with zero LLM cost.

Contrast with the alternative: Microsoft GraphRAG ([arXiv:2404.16130](https://arxiv.org/abs/2404.16130)) runs an LLM over every chunk to extract entities and relationships at index time. Reported cost at scale: roughly **\$10 per megabyte of corpus** for GPT-4-class extraction, with re-extraction needed whenever the schema or prompt changes. Our corpus is ~50 MB of markdown; that would price the first index build at ~\$500 and every reindex at the same. Our entire L0 reindex is free, deterministic, and finishes during the runner's housekeeping window.

The trade-off is that we cannot extract relationships the lab has not already named. We lose, for instance, "this paragraph implicitly references the same root cause as that paragraph" — the kind of latent link an LLM extractor would surface. But the lab's discipline of *naming everything that matters* (D-N for decisions, P-* for unresolved threads, named carry-forwards in retros) means most latent links *are* already explicit. The vocabulary discipline pays for itself twice: once when humans read the log, once when retrieval walks the graph.

---

## 3. Performance Characteristics

| Operation               | Latency      | Notes                                                  |
| ----------------------- | ------------ | ------------------------------------------------------ |
| L0 graph rebuild        | ~100 s       | 509 files → 4.5K nodes, 4K edges; runs in runner hook  |
| L0 PPR query            | ~200 ms      | networkx personalized PageRank, restart α=0.15         |
| L1 vector rebuild       | ~6 min       | 47K chunks @ MiniLM-L6-v2 on M3 Pro CPU                |
| L1 ANN query            | ~50 ms       | sqlite-vec brute-force (no ANN index yet — corpus small) |
| L2 BM25 rebuild         | ~1.4 s       | rank\_bm25 over 47K chunks                             |
| L2 BM25 query           | ~30 ms       | pure-Python ranking                                    |
| L3 RRF fusion           | <5 ms        | k=60, no I/O                                           |
| L4 rerank (CPU, warm)   | ~10–12 s     | bge-reranker-v2-m3 over top-30 → top-K                 |
| L4 rerank (CPU, cold)   | +3–10 s once | first-time model load                                  |
| Full L0+L1+L2+L3+L4     | ~12 s        | warm; dominated by L4                                  |
| Full L0+L1+L2+L3 (no L4)| <300 ms      | when recall > precision                                |

Storage footprint (all on-laptop, no cloud):

| Index            | File                                       | Size    |
| ---------------- | ------------------------------------------ | ------- |
| Token graph DB   | `data/retrieval/graph.db`                  | ~5 MB   |
| BM25 JSON        | `data/retrieval/bm25.json`                 | ~10 MB  |
| Vector DB        | `data/retrieval/vectors.db` (sqlite-vec)   | ~150 MB |
| Reranker model   | `~/.cache/huggingface/.../bge-reranker-v2-m3` | 568 MB  |
| **Total**        |                                            | ~730 MB |

All within the 18 GB envelope with plenty of headroom for the active training/eval workload.

The runner's `_run_retrieval_reindex` hook re-builds L0+L1+L2 incrementally after every Director session exit. L2 (BM25) rebuilds from scratch each time because rank\_bm25 has no incremental API and the rebuild is sub-2 s anyway. L0 (graph) rebuilds nodes touched by changed files. L1 (dense) reuses the existing `lab_memory.py` chunk-mtime delta logic.

---

## 4. Comparison to SOTA

The 2026-05-20 SOTA synthesis (`data/research/retrieval_sota_2026-05-20.md`) surveyed the seven systems below. Our adoption decisions are summarised here.

| System          | Year     | Core idea                                                                | Adopted? | Notes                                                                              |
| --------------- | -------- | ------------------------------------------------------------------------ | -------- | ---------------------------------------------------------------------------------- |
| GraphRAG        | Apr 2024 | LLM-extracted entity/relation KG + community summaries (Leiden)          | Partial  | Adopted PPR concept. Skipped LLM extraction (we use canonical tokens) and community detection (premature at 47K chunks). |
| LazyGraphRAG    | Jun 2025 | GraphRAG with community summaries deferred to query time                 | No       | Still incurs LLM-extraction cost at index time. Same objection as GraphRAG.        |
| LightRAG        | Oct 2024 | Dual-level (low / high keyword) retrieval over lighter graph             | Partial  | Adopted the "cheap graph" philosophy. Skipped LLM keyword extraction.              |
| HippoRAG2       | Oct 2024 | Phrase-passage graph + PPR seeded by phrase+dense matches (NeurIPS 2024) | Yes      | Closest analog. Adopted PPR-over-graph. Difference: HippoRAG2 extracts phrases via OpenIE; we use canonical tokens (zero LLM cost). |
| A-MEM           | Feb 2025 | Zettelkasten-style agentic memory with dynamic backlinks                 | Concept  | Adopted relational thinking. Different scope (per-agent vs lab-wide). The lab's per-role CoALA memory already does this at the agent level. |
| HaluMem         | Nov 2025 | Write-side hallucination detection over memory writes                    | No       | Flagged for future work (§6.B). Worth doing once corpus growth exposes real false-positive memory writes. |
| CoALA           | Sep 2023 | Working / Episodic / Semantic / Procedural memory taxonomy (ICLR 2024)  | Yes      | Already followed per-agent (`data/agents/<role>/{procedural.md,episodic/,semantic.md}`). Retrieval rides on top. |

The closest analog to our system is **HippoRAG2**: phrase-seeded PPR over a corpus graph, fused with dense retrieval. The substantive difference is that HippoRAG2 pays an OpenIE extraction pass at index time to build the phrase vocabulary; we extract the vocabulary for free because the lab's authoring discipline already produces it. On a sufficiently disciplined corpus, the canonical-token approach strictly dominates LLM extraction on cost and matches it on link density (subject to the discipline holding — see §5).

---

## 5. What We Deliberately Did NOT Build

Listing the unbuilt options is as important as the built ones — these are choices, not oversights.

- **LLM-based entity extraction.** Reason: canonical tokens already give us the graph for free, deterministically, at zero LLM cost. Re-evaluate only if the corpus grows past ~100K chunks and we start seeing meaningful information loss from un-named entities.
- **Community detection / Leiden clustering / summarisation.** Reason: 4.5K nodes is too small for community structure to be informative. Re-evaluate at >50K nodes.
- **Reflection layer (agent self-critique on memory writes).** Reason: out of scope for retrieval; belongs in agent policy. Director session-end log already serves this function.
- **Write-side hallucination detection (HaluMem-style).** Reason: not yet operationally needed — the corpus is small enough that bad writes are caught at PR review or by the next agent that reads the file. Flagged for §6.B.
- **Migration to Neo4j or a real graph DB.** Reason: SQLite + a `nodes` / `edges` schema + networkx PPR handles 5K-node graphs in 200 ms. Operational complexity of Neo4j (separate process, separate backup, separate auth) is not justified at this scale.
- **Switching the embedding model to Ollama or a larger sentence-transformer.** Reason: MiniLM at 384 dims is fast, fits on CPU, and the dense layer is already only one of three retrieval signals — increasing its quality has diminishing returns inside RRF + rerank.
- **A reranker hosted as a long-running server.** Reason: the concurrency-isolated `RetrievalWorker` subprocess (T0.3) already loads the model once and serves N queries via JSON-RPC stdin/stdout. Adding a network hop would only matter for cross-machine deployment, which isn't our target.

---

## 6. Future Work — Methodology Paper Track

If the lab decides to publish *"Agent-Lab Memory via Canonical-Token Graphs"* as a methodology contribution (one of the four output types from the 2026-05-19 reframed mission), the next four pieces are pre-scoped:

**A. Reflection / community summarisation at >100K chunks.** Once the corpus passes 100K chunks, run Leiden on the canonical-token graph and generate single-sentence community summaries (one LLM call per community, cached). Use those summaries as additional L0 seeds. The cost remains bounded because communities are O(\sqrt{N}) not O(N).

**B. Write-side contradiction detection (HaluMem-style).** Run a small contradiction-detection pass over each memory write against the top-K retrieved chunks from the existing corpus. Flag (don't block) writes that contradict prior canon. Useful especially for `current.md` and `decisions_recent.md` where authoring drift produces silent inconsistency.

**C. Benchmark against MemBench / MemoryAgentBench / HaluMem.** None of these benchmarks targets agent-lab memory specifically, but the relevant axes — multi-hop retrieval, temporal reasoning, contradiction detection — translate. The publishable claim would be a Pareto frontier point: comparable recall at orders-of-magnitude lower index cost.

**D. Open-source the canonical-token discipline.** The lab's D-N / P-* / program / phase / role vocabulary discipline is, per the 2026-05-20 SOTA synthesis, genuinely ahead of what's published in the agent-memory literature. The methodology contribution is not the retrieval stack — it's the *authoring discipline* that makes a free retrieval stack viable. The paper writes itself: "If you make your agents name things consistently, you don't need LLMs to extract entities."

---

## 7. Module Map

For navigation:

| Module                              | LOC  | Role                                                     |
| ----------------------------------- | ---- | -------------------------------------------------------- |
| `tools/retrieval/graph.py`          | 344  | L0 — SQLite token graph, regex extractors, PPR query API |
| `tools/retrieval/bm25.py`           | 154  | L2 — BM25 index with JSON persistence, search CLI        |
| `tools/retrieval/hybrid.py`         | 66   | L3 — RRF fusion + hybrid\_search() helper                |
| `tools/retrieval/rerank.py`         | 47   | L4 — cross-encoder reranker with ms-marco fallback       |
| `tools/retrieval/search.py`         | 134  | top-level orchestrator + CLI entrypoint                  |
| `tools/retrieval/concurrency.py`    | 160  | RetrievalWorker subprocess for model-load isolation      |
| `tools/lab_memory.py`               | (shim) | Now delegates to `tools/retrieval/search.py` for back-compat |

**Total new code:** ~900 LOC across 7 modules. **Total new index storage:** ~165 MB on disk. **Total new model storage:** 568 MB (one-time download). **Total LLM cost at index time:** \$0.

---

*Last updated: 2026-05-21 (Task 5.3 of `2026-05-20-memory-retrieval-upgrade.md`).*
