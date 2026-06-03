# Memory Tool + Wiki Refactor — Design Spec

**Date:** 2026-04-19
**Author:** Director (post-Program-1 reframe, under PI DIRECTIVE D-114)
**Status:** DRAFT — awaiting user review
**Implements:** Three-tier context discipline + Anthropic-protocol memory tool + session_brief runner hook
**Supersedes (partially):** Free-for-all reads on `data/state.md`, `data/decisions_recent.md`, `data/director_log.md`

---

## 1. Goal

One sentence: **cut Director session-startup reads from ~30–50 KB down to ≤10 KB, raise sessions-per-compaction-window from 2-3 to 5-7, without losing any durable knowledge.**

The lab's rigor (pre-registration, unanimous compromise, anti-forgery, 30-role roster) is all functioning. The one remaining systemic drag is **context thrashing**: the Director re-reads large orchestration files on every session restart, burning the 200K context window in 2-3 sessions and forcing compaction. That compaction is lossy (the user has seen it), and the churn compounds.

The fix is to separate three kinds of information the orchestration layer uses, cap each one with a size budget, route all edits through a single tool API that matches Anthropic's forthcoming `memory_20250818` protocol, and pre-compute a 500-token session brief before each Director start so the Director can orient without scanning.

This is the same architectural move Karpathy documented for his LLM wiki and Anthropic codified in their Memory tool. HIVE (the sibling lab) is also doing it. We're doing a version calibrated to our existing governance machinery.

---

## 2. Background and motivation

### 2.1 The problem, measured

Current session-startup read-footprint (baseline, 2026-04-19):

| File | Size | Read-frequency |
|---|---|---|
| `CLAUDE.md` | 8 KB | Every session (Claude Code auto-load) |
| `data/state.md` | 32 KB | Every session, Director first action |
| `data/pi_notes.md` | 12 KB | Every session |
| `data/decisions_recent.md` | 164 KB | Selective (often ≥20 KB read) |
| `data/director_log.md` | 184 KB | Selective (often ≥20 KB read) |
| `data/shared_knowledge.md` | 44 KB | Selective |
| `data/index.md` | 4 KB | Every session |
| Agent procedural prompts (dispatched) | 2–10 KB each | Per dispatch |

Typical unavoidable startup read: **30–50 KB**. Total conversational context budget (Opus 4.7 1M mode): ~1M tokens ≈ 2 MB of raw text, but **effective working window** (the part where attention stays sharp and compaction doesn't kick in) is ~200 K tokens ≈ 400 KB of text. Two to three normal sessions exhaust it.

### 2.2 Prior art referenced

- **Anthropic Memory tool** (`memory_20250818`) — client-side file-based memory with `view / create / str_replace / insert / delete / rename` as the operation set; mounts at `/memories`
- **Karpathy LLM Wiki** pattern — INDEX.md + topic files, fetched on demand by name, claimed ~70× more context-efficient than RAG for this scale
- **HIVE's three-tier plan** — Hot state (40 KB) / Wiki (50 KB) / Log (30 KB) with Knowledge Manager closeouts
- **CoALA 4-memory framework** (already used at agent-role level) — procedural / working / semantic / episodic

This spec is the union of the three organization-level tools (Anthropic + Karpathy + HIVE) fitted to AGI's existing CoALA agent-role memory.

### 2.3 What this does NOT change

- Agent-role procedural / episodic / semantic memory (`data/agents/<role>/`) — untouched
- Program deliverables (papers, closure memos under `programs/*/`) — untouched
- `tools/lab_memory.py` semantic search (sqlite-vec + sentence-transformers) — kept, extended
- `data/values.md`, the three principles, PI directives — unchanged
- Program-based phase structure, unanimous compromise, anti-forgery — unchanged
- `data/agents/agents.json` roster — unchanged
- `CLAUDE.md` auto-memory at `~/.claude/projects/.../memory/` — **separate system**, not touched by this spec

---

## 3. Scope (explicit in and out)

### 3.1 In scope

Files that govern Director-level orchestration state and move into the tiered model:

- `data/state.md` → **hot state tier**
- `data/decisions_recent.md` → **log tier canonical**
- `data/decisions_archive.md` → **log-rotation target**
- `data/director_log.md` → merged into **log tier**
- `data/pi_notes.md` → **wiki tier** (may split if it grows past 10 KB section)
- `data/shared_knowledge.md` → **wiki tier** (may split by topic)
- `data/killed_ideas.md` → **wiki tier**
- `data/values.md` → **wiki tier**
- `data/procedures.md` → **wiki tier** (on-demand; doesn't count against default startup read)
- `data/index.md` → replaced by **`data/memories/INDEX.md`** (regenerated)
- `data/mission_reframe_2026-04-18.md` → **wiki tier**
- `data/checkpoints/ARCHIVED.md` → **wiki tier**

New files created:

- `data/memories/INDEX.md` — single entry point, all other memories reachable from here
- `data/memories/current.md` — hot state (replaces `data/state.md`)
- `data/memories/mission.md` — aspirational + real mission statements (lifted from `CLAUDE.md` + reframe doc)
- `data/memories/governance.md` — three principles, values, PI directives pointer
- `data/memories/roster.md` — 30-role map (summary of `agents.json`, with pointers)
- `data/memories/procedures.md` — moved from `data/procedures.md`
- `data/memories/programs/active.md` — current program's state (references `programs/<active>/question.md` etc.)
- `data/memories/programs/portfolio.md` — moved from `programs/portfolio.md`
- `data/memories/log.md` — canonical rolling log (merges `decisions_recent.md` + `director_log.md`)
- `data/memories/history.md` — one-line-per-session index into archived logs
- `data/memories/killed.md` — moved from `data/killed_ideas.md`
- `data/memories/shared.md` — moved from `data/shared_knowledge.md` (may split)
- `tools/memory.py` — Anthropic-protocol wrapper with AGI extensions
- `data/infra/memory_telemetry.log` — startup-read size telemetry
- `data/archives/<YYYY-MM-DD>/<tier>/<file>` — rotation / delete-override target
- `data/archives/snapshots/<timestamp>/<file>` — pre-edit defensive snapshots

### 3.2 Out of scope (explicitly stays as-is)

| Domain | Location | Why out |
|---|---|---|
| Agent-private memory | `data/agents/<role>/procedural.md`, `episodic/`, `semantic.md` | Per-role CoALA memory, not shared orchestration state. Modified by role dispatches, not by Director's hot loop. |
| Program deliverables | `programs/<program>/*.md` (papers, closure memos, phase reports) | Research artifacts — never compacted, never rotated. |
| Runtime infra state | `data/infra/rate_limit_resets_at`, `build_status.md`, `thermal_log.md`, `perf_log.md` | Runtime telemetry, small, ephemeral, separate contract. |
| Training / experiment data | `data/experiments.db`, `data/training/`, `data/checkpoints/` | Raw scientific data, governed by Values §4 archive-never-delete already. |
| Specs and plans | `docs/superpowers/specs/`, `docs/superpowers/plans/` | Design docs, indexed by `lab_memory.py` for search but not in Director's hot path. |
| Tooling source | `tools/*.py`, `*.c`, `*.metal`, `*.swift` | Code, not memory. |
| Claude Code auto-memory | `~/.claude/projects/.../memory/MEMORY.md` + files | Separate persistence layer owned by Claude Code itself. |
| ~~`OBSIDIAN_IMPLEMENTATION.md` at repo root~~ | (removed) | Orphan from unrelated audio project; moved to `~/Desktop/Obsidian/` on 2026-04-19. No longer in this repo. |

---

## 4. The three-tier model

### 4.1 Tier definitions

| Tier | Size cap | Canonical file(s) | Purpose | Update frequency |
|---|---|---|---|---|
| **Hot** | 40 KB | `data/memories/current.md` | What is active RIGHT NOW: current program, current phase, active blocker, next step, rate-limit state, open questions | Multiple times per session (phase transitions) |
| **Wiki** | 50 KB **total across all files** | `mission.md`, `governance.md`, `roster.md`, `procedures.md`, `killed.md`, `shared.md`, `programs/portfolio.md`, reframes | Durable slow-changing truth: mission, values, roster, lessons, killed ideas, cross-program knowledge | Rarely (program open, mission reframe, roster change) |
| **Log** | 30 KB rolling | `data/memories/log.md`, `data/memories/history.md` | Append-only recent session trail. When `log.md` exceeds 30 KB, oldest entries rotate to `data/archives/<YYYY-MM-DD>/log.md` and a breadcrumb line lands in `history.md`. | Every session end |

### 4.2 Tier routing rules

When the Director or an agent needs to write state, these rules decide which tier:

```
if fact decays within one session (e.g., "phase 2 step 3 in progress"):
    -> hot/current.md
elif fact is a decision, event, or dispatch outcome:
    -> log/log.md
elif fact is a principle, role, mission statement, or killed approach:
    -> wiki/<appropriate file>
elif fact is a research artifact (paper, closure memo):
    -> programs/<program>/... (NOT a memory tier)
```

### 4.3 Size calibration

The 40 / 50 / 30 numbers are HIVE's starting caps. Our baseline data says we're landing 32 KB in hot, ~50 KB across wiki candidates, and the log is currently unbounded. After 5 sessions under the new system, `lab_architect` recalibrates:

- If a tier is chronically below 50% of cap → tighten
- If a tier is chronically against the cap → widen, or split the files that are bloating it
- Commit the new caps to this spec as an addendum (append-only)

---

## 5. Tool layer (`tools/memory.py`)

### 5.1 Protocol fidelity

`tools/memory.py` implements Anthropic's `memory_20250818` tool protocol verbatim for the following commands:

| Command | Signature | Semantics |
|---|---|---|
| `view` | `view(path, view_range=None)` | Return file contents, optionally line-ranged |
| `create` | `create(path, file_text)` | Create or overwrite file at path |
| `str_replace` | `str_replace(path, old_str, new_str)` | Exact-match substring replace; errors if `old_str` not unique |
| `insert` | `insert(path, insert_line, insert_text)` | Insert text at line number |
| `delete` | `delete(path)` | **Override (see §5.3)** |
| `rename` | `rename(old_path, new_path)` | Move within memory tree |

Directory-valued `view` returns children (like `ls`). All paths are rooted at `data/memories/`.

When Anthropic exposes the tool natively to Claude Code, we rebind `data/memories/` → `/memories` at the mount and retain the same directory structure. Zero further migration.

### 5.2 AGI extensions

Two commands added, namespaced to avoid protocol collision:

| Command | Signature | Purpose |
|---|---|---|
| `search` | `search(query, k=5)` | Semantic search over memories + program artifacts via `tools/lab_memory.py`. Returns file paths + excerpts. |
| `snapshot` | `snapshot(path)` | Defensive copy of a file into `data/archives/snapshots/<timestamp>/<path>` before a risky edit. Returns snapshot path. |

These are **not** sent to Anthropic when the native tool is adopted — they stay as local CLI subcommands.

### 5.3 Protocol deviation: `delete`

**Anthropic spec:** `delete(path)` removes the file.
**AGI deviation:** `delete(path)` moves the file to `data/archives/deletes/<YYYY-MM-DD>/<path>` and leaves a one-line breadcrumb at the original location:

```markdown
<!-- archived 2026-05-01 → data/archives/deletes/2026-05-01/memories/shared.md -->
```

**Reason:** Values §4 archive-never-delete. All raw lab state is preserved; destruction is never the right verb.

Document this as an **explicit protocol deviation** in the tool's `--help` output and in this spec. If the native Anthropic tool is later adopted, we wrap it or override it at the CLI level to preserve this invariant.

### 5.4 CLI shape

```bash
# view
python3 tools/memory.py view memories/current.md
python3 tools/memory.py view memories/programs/           # directory listing
python3 tools/memory.py view memories/log.md --lines 1-50

# edit
python3 tools/memory.py create memories/note.md --text "..."
python3 tools/memory.py str-replace memories/current.md --old "..." --new "..."
python3 tools/memory.py insert memories/log.md --line 1 --text "..."
python3 tools/memory.py delete memories/obsolete.md       # -> archives + breadcrumb
python3 tools/memory.py rename memories/a.md memories/b.md

# extensions
python3 tools/memory.py search "checkpoint archival policy" -k 5
python3 tools/memory.py snapshot memories/current.md

# admin (used by runner + findings_curator)
python3 tools/memory.py index                             # regenerate INDEX.md
python3 tools/memory.py audit                             # check caps, report breaches
python3 tools/memory.py rotate-log --cap-kb 30            # run log rotation
```

### 5.5 Library shape

```python
from tools.memory import Memory
m = Memory(root="data/memories")
m.view("current.md")
m.str_replace("current.md", old="Phase 2 step 3", new="Phase 2 step 4")
m.search("entropy floor")
```

Called from the runner (`run_agi_lab.sh`), from agent subprocesses, and from `findings_curator` closeouts.

---

## 6. Session lifecycle

### 6.1 Pre-session (runner writes `session_brief.md`)

Before the Director launches, `run_agi_lab.sh` composes **`data/memories/session_brief.md`** (≤500 tokens, ≤4 KB). Sources:

| Field | Source |
|---|---|
| Timestamp | `date -u +%FT%TZ` |
| Active program | `data/memories/programs/active.md` head |
| Current phase & step | `current.md` "State" section |
| Last decision ID | last `D-NNN` in `log.md` |
| Blockers | `current.md` "Blockers" section |
| Rate-limit status | `data/infra/rate_limit_resets_at` (if exists) |
| Pending evaluator action | last `data/agents/evaluator/episodic/*.md` |
| Memory telemetry last session | last line of `data/infra/memory_telemetry.log` |

Written **atomically** (`tmp` + `rename`) with an ISO-8601 header line:

```markdown
---
generated_at: 2026-04-19T14:32:18Z
runner_pid: 54821
runner_session: s-1234
---
# Session brief
Active program: program_2_dense_vs_moe_sub100m (Phase 2 pre-flight, blocked on P-METAL-1)
Last decision: D-116 (Phase 2 pre-flight close, 2026-04-18)
Next step: resolve P-METAL-1 (metallib build) → dispatch engineer
Rate-limit: clean
Evaluator pending: none (last report passed all 18 checks)
Memory last session: startup 8.2 KB / 18 checks passed / log 22 KB
```

### 6.2 Director start sequence (new)

The Director's **first action** on session start:

1. `memory.view memories/session_brief.md`
2. **Timestamp check:** if `generated_at` is >10 minutes old, treat brief as stale and fall back to direct `memory.view memories/INDEX.md`
3. Otherwise: `memory.view memories/INDEX.md` for the lookup table
4. Selective `memory.view` on files the brief or INDEX says are relevant to the next step
5. Proceed to phase work

Target startup read: **≤10 KB** (brief ~4 KB + INDEX ~2 KB + 1-2 selective files ~4 KB).

### 6.3 During-session edits

- Phase transitions: `memory.str_replace` on `current.md` — no full-file rewrite
- New decisions: `memory.insert` at top of `log.md`, tagged with `D-NNN`
- Wiki edits: rare, via `memory.str_replace` or `memory.insert`
- Snapshot before large edits: `memory.snapshot memories/current.md` — written to `data/archives/snapshots/`

### 6.4 End-of-session (findings_curator closeout)

At phase close, **not** at session end (sessions are shorter than phases now), the Director dispatches `findings_curator` with:

1. **Findings inbox**: promote new facts from log/hot to wiki where they're durable
2. **INDEX maintenance**: run `memory.py index` — verify no broken pointers, no orphan files
3. **Cap enforcement**: run `memory.py audit`; if any tier >90% cap OR any wiki file exceeds 15 KB soft target, act:
   - Hot >36 KB: compress verbose sections, move completed items to log
   - Wiki total >45 KB (hard cap 50 KB): split the largest file along topic boundaries
   - Wiki per-file >15 KB (soft target): inspect for natural topic split; if one exists, split; if the file is genuinely one cohesive topic, leave it and note in closeout record
   - Log >27 KB: run `memory.py rotate-log` (oldest entries → `data/archives/<date>/log.md`; one-line breadcrumb in `history.md`)
4. **Semantic reindex**: `python3 tools/lab_memory.py ingest --incremental` over changed files
5. **Breadcrumb audit**: verify every archive-move left a valid breadcrumb
6. **Telemetry line**: append summary to `data/infra/memory_telemetry.log`

### 6.5 Subagent briefing pattern (unchanged from today)

When Director dispatches an Agent subagent (e.g., `engineer`, `red_team`, `pi`), **subagents do not read `memories/` directly**. The Director curates and passes relevant content as part of the dispatch prompt — same pattern used today. The tiered memory is the Director's working space; subagents get targeted briefs.

**Reason:** preserves the curation layer; prevents agents from drifting via direct INDEX access; keeps the memory surface small.

---

## 7. Knowledge manager (KM) closeout — responsibilities for `findings_curator`

The existing `findings_curator` role gains explicit tier obligations. Its procedural prompt gets a new section §KM-Closeout with the six tasks from §6.4. Invocation trigger: phase close (Evaluator's phase-gate sub-step).

The KM closeout runs **before** Evaluator's phase gate, so Evaluator sees the cleaned state. If KM fails (caps still breached after closeout), Evaluator's memory-discipline checks (§8) fail, and the phase blocks.

---

## 8. Evaluator discipline checks (7 new, added to the 11-item checklist)

Current Evaluator checklist is 11 items (1–10 program/phase validity + #11 signature integrity). Adding 7 memory-discipline items → **18-item checklist**.

| # | Check | Fail condition | Fix path |
|---|---|---|---|
| 12 | Hot tier size | `current.md` > 40 KB | KM closeout compresses or routes to log |
| 13 | Wiki tier total | sum of wiki files > 50 KB | KM closeout splits largest file |
| 14 | Log tier size | `log.md` > 30 KB | KM closeout runs rotation |
| 15 | INDEX integrity | Any `INDEX.md` pointer resolves to missing file, OR any file in `memories/` not reachable from INDEX | `memory.py index` regenerates |
| 16 | No full-file rewrites | Git diff shows a full-file replacement of any tier file in last session (>80% line change + no rename) | Investigate, revert, use `str_replace` or `insert` |
| 17 | KM closeout ran | If any tier ≥90% cap at phase close and KM closeout episodic record absent | Block phase close until KM dispatched |
| 18 | Session brief consumed | Director's first tool call in session was NOT `memory.view memories/session_brief.md` OR `memory.view memories/INDEX.md` | Note in evaluator report; 3 consecutive violations → ORG_ADAPTATION |

Failures are surfaced via the existing evaluator report template with a new `§memory_discipline` section.

---

## 9. Cross-tier references and stable links

Wiki files will mention events (decisions, dispatches) that live in the log. Logs rotate. To keep wiki links stable across rotation:

**Reference format:**

```markdown
[log:2026-04-19#D-117]
[session:2026-04-19T14:32]
[episodic:pi/2026-04-18_program_2_phase_1]
```

Resolver rules:

- `[log:YYYY-MM-DD#D-NNN]` — resolves to live `log.md` section if within window, else to `data/archives/<YYYY-MM-DD>/log.md#D-NNN`
- `[session:timestamp]` — resolves to session_brief archive or `history.md` entry
- `[episodic:<role>/<file>]` — resolves to `data/agents/<role>/episodic/<file>.md` (existing convention from `verify_signatures.py`)

`tools/memory.py` gains a `resolve <reference>` command that returns the absolute path.

A CI-lite check (part of Evaluator check #15) scans wiki files for broken references.

---

## 10. File unification map (what merges, splits, or moves)

### 10.1 Merges

| From | To | Rationale |
|---|---|---|
| `data/director_log.md` (184 KB) | `data/memories/log.md` + rotated archives | 184 KB is the biggest single source of bloat. Rotation immediately retires ~90% of it. |
| `data/decisions_recent.md` (164 KB) | `data/memories/log.md` (merged) | Same log stream, duplicated today. |
| `data/decisions_archive.md` | `data/archives/decisions/<date>.md` | Archive is already archive; move to canonical archive root. |

### 10.2 Moves (verbatim, with rename)

| From | To |
|---|---|
| `data/state.md` | `data/memories/current.md` |
| `data/pi_notes.md` | `data/memories/governance/pi_notes.md` (keep the wiki governance section; pi_notes itself links in) |
| `data/values.md` | `data/memories/governance/values.md` |
| `data/procedures.md` | `data/memories/procedures.md` |
| `data/shared_knowledge.md` | `data/memories/shared.md` |
| `data/killed_ideas.md` | `data/memories/killed.md` |
| `data/index.md` | regenerated as `data/memories/INDEX.md` |
| `data/mission_reframe_2026-04-18.md` | `data/memories/mission.md` + archive the reframe doc as history |
| `data/checkpoints/ARCHIVED.md` | `data/memories/checkpoints_archived.md` |
| `programs/portfolio.md` | `data/memories/programs/portfolio.md` (symlink back if external readers expect old path) |

### 10.3 Splits (deferred; two independent triggers)

**Trigger 1 — hard total cap (load-bearing):** wiki total exceeds 50 KB → KM splits the largest file along topic boundaries at next closeout. Non-optional.

**Trigger 2 — soft per-file target (ergonomic):** any individual wiki file exceeds 15 KB → KM inspects for a natural topic split at next closeout. If a clean boundary exists, split; if the file is one cohesive topic, leave it and note reasoning in the closeout record. This is a *judgment check*, not a forced action.

Rationale: total cap catches runaway growth; per-file target catches monolithic files that are technically under the total but ergonomically unwieldy for Director's focused reads.

| File | Split plan when triggered |
|---|---|
| `shared.md` (currently 44 KB) | By topic: `shared/architecture.md`, `shared/data.md`, `shared/eval.md`, `shared/infra.md` |
| `pi_notes.md` (currently 12 KB, growing) | By directive era: `pi_notes.md` keeps only active directives; archived as `pi_notes_archive_<YYYY>.md` |

**Note:** `shared.md` at 44 KB already exceeds the 15 KB per-file soft target. Migration (§12.2) will split it during bootstrap rather than leaving a knowingly-oversized file to trigger the check on session 1.

---

## 11. Integration touchpoints (files that change outside memories/)

| File | Change |
|---|---|
| `CLAUDE.md` | Update state-file references from `data/state.md` → `data/memories/INDEX.md` and `data/memories/current.md`; note the memory tier rules |
| `data/agents/director/procedural.md` | Prepend: "session start = view `memories/session_brief.md` first, then `memories/INDEX.md`, then selective"; detail the 18-item evaluator checklist expectations |
| `data/agents/pi/procedural.md` | Add: "reads `memories/governance/pi_notes.md`; directives go via `memory.insert` on that file" |
| `data/agents/evaluator/procedural.md` | Add 7 checks §memory_discipline; expand checklist template to 18 items |
| `data/agents/findings_curator/procedural.md` | Add §KM-Closeout with the six §6.4 tasks; trigger = phase close |
| `run_agi_lab.sh` | Add `write_session_brief` function before each Director launch; after graceful checkpoint, log startup read size to `data/infra/memory_telemetry.log` |
| `tools/dashboard.py` | Path updates: `data/state.md` → `data/memories/current.md`; add tier-size panel |
| `tools/dashboard.html` | Surface tier caps + current size in health panel |
| `tools/slack_bot.py` | Path updates for `/status`, `/phase`, `/program`, `/last`, `/recent` commands |
| `tools/lab_memory.py` | Index `data/memories/` explicitly; add `--incremental` mode used by KM closeout |
| `Makefile` | `lab-status` reads new path; add `memory-audit`, `memory-index`, `memory-rotate-log` targets |

---

## 12. Migration plan (phase-gated, reversible)

### 12.1 Migration gate

Program 2 is mid-phase (Phase 2 pre-flight, blocked on P-METAL-1 build). **Do not migrate live.** The migration gate is:

> Phase 2 closes (P-METAL-1 resolved OR pivoted to Plan B) AND Evaluator signs off on phase close → migration may begin.

Until then, this spec is written and queued but not executed.

### 12.2 Migration steps (one-time)

1. **Snapshot.** `python3 tools/memory.py snapshot-all data/` → `data/archives/snapshots/pre-memory-refactor/`. Include `state.md`, `decisions_recent.md`, `director_log.md`, `pi_notes.md`, `shared_knowledge.md`, `killed_ideas.md`, `values.md`, `procedures.md`, `index.md`, `mission_reframe_2026-04-18.md`.
2. **Build `data/memories/`.** `mkdir -p data/memories/{governance,programs,shared}`
3. **Run bootstrap script** (§13) to seed files from sources
4. **Verify tiers under cap** via `python3 tools/memory.py audit`
5. **Regenerate INDEX** via `python3 tools/memory.py index`
6. **Install `tools/memory.py`** (TDD, unit tests for each of the 6 protocol commands + 2 extensions + delete-override)
7. **Rewrite Director prompt** to new session-start sequence
8. **Add evaluator checks 12–18**
9. **Add KM-Closeout to findings_curator prompt**
10. **Runner hook**: `write_session_brief` in `run_agi_lab.sh`
11. **Telemetry init**: create `data/infra/memory_telemetry.log` with header line
12. **Update dashboard/slack/Makefile** paths
13. **Semantic reindex**: `python3 tools/lab_memory.py ingest --full`
14. **Dry-run**: start a test Director session with `CLAUDE_DRY_RUN=1` environment flag; verify brief is consumed first and startup read ≤10 KB
15. **Cutover**: next Program 2 Phase 3 opens under new regime; old files kept read-only for 3 sessions as fallback

### 12.3 Post-migration validation window

Three consecutive phases run cleanly (no memory-discipline evaluator failures) before we:

- Delete read-only old-path fallbacks (replaced by breadcrumbs pointing to new paths)
- Bump `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` from 50 → 70
- Add a new MEMORY.md auto-memory entry noting the refactor succeeded

If validation fails at any phase, see §14 rollback.

---

## 13. Bootstrap script (one-time seed)

`tools/migrate_to_memories.py` (ephemeral — deleted after migration + two sessions, committed to git history for audit).

Inputs: existing `data/*.md` files enumerated in §10.
Outputs: populated `data/memories/` + INDEX.

Algorithm:

```python
# Pseudocode
ensure("data/memories/", "governance/", "programs/", "shared/")
snapshot_all("data/archives/snapshots/pre-memory-refactor/")

# Hot
current = extract_current_state(read("data/state.md"))
write("data/memories/current.md", current)

# Wiki
write("data/memories/mission.md", extract_mission_from(CLAUDE_md, reframe_md))
write("data/memories/governance/pi_notes.md", read("data/pi_notes.md"))
write("data/memories/governance/values.md", read("data/values.md"))
write("data/memories/procedures.md", read("data/procedures.md"))
write("data/memories/shared.md", read("data/shared_knowledge.md"))
write("data/memories/killed.md", read("data/killed_ideas.md"))
write("data/memories/roster.md", summarize_roster("data/agents/agents.json"))
write("data/memories/checkpoints_archived.md", read("data/checkpoints/ARCHIVED.md"))

# Programs
write("data/memories/programs/portfolio.md", read("programs/portfolio.md"))
write("data/memories/programs/active.md", summarize_active_program())

# Log (merged + compressed)
log_content = merge_and_dedupe(
    read("data/director_log.md"),
    read("data/decisions_recent.md"),
)
recent_30kb, older = split_by_size(log_content, 30_000)
write("data/memories/log.md", recent_30kb)
archive(older, "data/archives/<date>/log.md")
write("data/memories/history.md", one_line_index_of(older))

# INDEX
run("python3 tools/memory.py index")

# Audit
run("python3 tools/memory.py audit")  # must pass
```

The script itself is tested (TDD) with a fixture directory so we can dry-run before running on real data.

---

## 14. Rollback protocol

### 14.1 Triggers (any ONE of these):

- 3 consecutive phases with ≥2 evaluator memory-discipline failures (checks 12–18)
- Director's startup read size ≥20 KB (2× target) for 5 consecutive sessions
- `memory.py` crashes or corrupts a file that can't be recovered from the archive snapshot
- User explicitly requests rollback

### 14.2 Rollback steps:

1. Copy `data/archives/snapshots/pre-memory-refactor/` back into `data/`
2. Restore runner to use `state.md` path (remove `write_session_brief` hook)
3. Restore Director prompt to pre-migration version (kept in `data/agents/director/archives/`)
4. Disable evaluator checks 12–18 (revert to 11-item list)
5. Log rollback event to `data/accountability_ledger.md` with reasoning
6. `lab_architect` runs ad-hoc retro: why did it fail? (miscalibrated caps, underspecified responsibilities, premature cutover, etc.)
7. Write addendum to this spec documenting failure and what would need to change before retry

Rollback is **reversible**: all original files are preserved under archive; the new `data/memories/` tree is retained (renamed to `data/memories_attempt_1/` for the retro).

---

## 15. Telemetry and success criteria

### 15.1 Metrics collected

Written to `data/infra/memory_telemetry.log`, one session per line:

```
2026-04-22T10:15:03Z | startup_kb=8.2 | brief_age_s=35 | eval_checks_passed=18 | log_kb=22.4 | wiki_kb=46.1 | hot_kb=34.8 | km_closeout=phase_close_only
```

### 15.2 Targets (measured over a 10-session rolling window)

| Metric | Baseline | Target | Time to hit |
|---|---|---|---|
| Director startup read | 30–50 KB | ≤10 KB | Session 1 post-migration |
| Sessions per compaction window | 2–3 | 5–7 | Session 3 post-migration |
| Evaluator discipline pass rate (checks 12–18) | — | ≥95% | Session 5 post-migration |
| Log tier rotation frequency | — | ≤1×/phase | Ongoing |
| Wiki tier growth rate | — | ≤5 KB/month | Ongoing |

### 15.3 Review cadence

- **Every phase close:** Evaluator reports discipline status
- **Every 3 programs:** `lab_architect` runs memory-architecture retro (includes telemetry aggregation)
- **Program 3 open** (first retro after migration): full memory-system audit; recalibrate caps; consider Phase 2.5 / Phase 3 escalations (FTS5, native Anthropic tool binding) if warranted

---

## 16. Open questions (all resolved 2026-04-19)

All five questions below were discussed and resolved with the user before implementation planning. Spec sections above reflect the resolutions; this section is retained as an audit trail.



1. ~~**`OBSIDIAN_IMPLEMENTATION.md`**~~ — **RESOLVED 2026-04-19:** orphan file; moved to `~/Desktop/Obsidian/`. Out of repo entirely. No longer relevant to this spec.
2. ~~**`data/memories/` vs native `/memories` mount**~~ — **RESOLVED 2026-04-19: rebind.** When Claude Code exposes the native Memory tool, `tools/memory.py` is retired and the native tool is configured to mount at `data/memories/`. Zero file migration (same filenames, same locations, different implementation). No dual-run — single memory surface maintained throughout.
3. ~~**Compaction bump timing**~~ — **RESOLVED 2026-04-19: conservative.** Hold `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=50` through migration cutover. Bump to 70 after 3 consecutive clean phases (no memory-discipline evaluator failures, startup reads consistently ≤10 KB). Rationale: migration is a fragile many-moving-parts change; validate the refactor delivers before relaxing safeguards. Rollback protocol §14 also reverts this bump if triggered.
4. ~~**Telemetry log retention**~~ — **RESOLVED 2026-04-19: pure append, no rotation.** `data/infra/memory_telemetry.log` grows linearly at ~150 bytes/session (~300 KB/year max). Storage impact negligible; grep-friendly as one file; archive-never-delete satisfied by keeping the file. No rotation logic to maintain.
5. ~~**Wiki-split thresholds**~~ — **RESOLVED 2026-04-19: both (C).** Hard total cap 50 KB (load-bearing, triggers forced split of largest file). Soft per-file target 15 KB (ergonomic, triggers KM inspection for natural topic boundary; split only if one exists). Evaluator check #13 watches the total; per-file is a KM closeout judgment item (§6.4, §10.3). `shared.md` will be pre-split during migration since it already exceeds the per-file target.

---

## 17. Non-goals

- **Not** a full rewrite of the knowledge graph / RAG system. Semantic search stays in `lab_memory.py`.
- **Not** a new database. SQLite-vec is already there; we reuse.
- **Not** auto-summarization of logs (HIVE discussed this; defer). Rotation is deterministic byte-level.
- **Not** a UI. Dashboard gets tier-size panel; no new interactive surface.
- **Not** a replacement for agent-private CoALA memory. Those stay per-role.

---

## 18. Dependencies and ordering

| Depends on | Provides for |
|---|---|
| Program 2 Phase 2 close | Migration gate |
| `lab_memory.py` (exists) | Semantic search extension |
| `verify_signatures.py` (exists) | Episodic reference format reuse |
| `run_agi_lab.sh` rate-limit logic (exists) | Session brief timing |
| Evaluator 11-item checklist (exists) | Extension to 18 |

No new external libraries. Python stdlib + `sqlite-vec` + `sentence-transformers` already installed.

---

## 19. Estimated implementation size

Rough task count for the plan document that follows this spec:

- **Tool**: `tools/memory.py` — 6 protocol commands + 2 extensions + delete-override + CLI + tests ≈ 8 tasks
- **Bootstrap**: `tools/migrate_to_memories.py` + fixture + dry-run ≈ 3 tasks
- **Agent prompt updates**: Director, PI, Evaluator, findings_curator ≈ 4 tasks
- **Runner hook**: `write_session_brief` + telemetry ≈ 2 tasks
- **Integration**: dashboard, slack_bot, Makefile, CLAUDE.md, lab_memory.py reindex ≈ 5 tasks
- **Migration execution**: snapshot, bootstrap run, verify, cutover, validation ≈ 5 tasks
- **Post-validation**: compaction bump, old-path cleanup, retro line ≈ 2 tasks

**~29 tasks across ~4 stages**, roughly the size of the 2026-04-17 overhaul spec. Each task bite-sized (2–15 minutes).

---

## 20. Appendices

### A. Anthropic `memory_20250818` protocol reference

Full spec: (to be fetched via context7 at implementation time).
Mount path: `/memories` (config-rebindable).
Command surface: `view / create / str_replace / insert / delete / rename` — identical to our §5.1 table.

### B. Reference format BNF

```
ref         := "[" tag ":" body "]"
tag         := "log" | "session" | "episodic"
body        := date ("#" anchor)? | path
date        := YYYY-MM-DD ("T" HH:MM)?
anchor      := identifier
path        := role "/" filename
```

### C. File-unification map (tabular summary)

See §10 — three tables cover merges, moves, and conditional splits.

### D. Delete-override examples

```markdown
# Before
data/memories/shared.md (44 KB)

# After: python3 tools/memory.py delete memories/shared.md
data/memories/shared.md:
  <!-- archived 2026-05-01 → data/archives/deletes/2026-05-01/memories/shared.md -->
data/archives/deletes/2026-05-01/memories/shared.md  (full original content)
```

### E. Session_brief.md example (populated)

See §6.1.

### F. Telemetry line schema

```
<ISO8601> | startup_kb=<float> | brief_age_s=<int> | eval_checks_passed=<int> | log_kb=<float> | wiki_kb=<float> | hot_kb=<float> | km_closeout=<enum: none|phase_close_only|cap_triggered>
```

---

## 21. Sign-off

| Role | Required? | Notes |
|---|---|---|
| User (Harshith) | **Yes** — reviews this spec before writing-plans generates the implementation plan | Gate step per `superpowers:brainstorming` protocol |
| PI | On program-level adoption | Via `data/agents/pi/episodic/` dispatch at migration gate |
| Director | Executes migration post-PI co-sign | Normal dispatch pattern |

---

*End of spec. Next step per `superpowers:writing-plans` protocol: await user review → on approval, draft implementation plan at `docs/superpowers/plans/2026-04-19-memory-tool-and-wiki-refactor.md`.*
