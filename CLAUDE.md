# AGI Project

## What This Is
A fully autonomous AI agent organization running rigorous small-model ML research on a MacBook Pro M3 Pro (18GB unified RAM). Two missions — one aspirational, one operational — both active, both load-bearing.

## Mission (two tiers)

### Aspirational Mission (star polaris)
Beat Claude Opus 4.7 on all standardized benchmarks (MMLU, HumanEval, ARC-AGI, GSM8K/MATH, HellaSwag, TruthfulQA, WinoGrande, BigBench-Hard) on 18GB laptop hardware with no cloud compute.

**Status (post-Program 1 D-114)**: formally *not achievable at current open-weights SOTA*. Program 1's Alt-D envelope paper (2026-04-18) establishes this with evidence (3-of-5 primary-floor hard-fails, GPQA Diamond gap −43.7pp at ≤70B open-weight scale). This mission remains fixed as the **star polaris** — a forcing function that keeps the research rigorous and frontier-facing. It is not revised downward. Programs measure progress *relative to* it without being expected to *reach* it.

### Real Mission (what programs actually target)
**Produce rigorous research at any scale that fits on 18 GB hardware.** What rigorous research *produces* depends on what the evidence supports — positive findings where an approach works, honest envelope characterization where it doesn't. The lab is not biased toward either outcome; both are valid scientific contributions. Programs are designed to be falsifiable, and the pre-registered decision rules decide which output type the artifact becomes.

**Scale is hardware-bounded, not arbitrarily capped.** Practical range: from ≤100M params trained from scratch (full bidirectional reach), up to ~9B FP16 or larger with 4-/2-bit quantization for inference-time or partial-fine-tuning work. Program selection should choose the smallest scale that can actually answer the question, but never refuse a question because "the model would be too big" without first checking whether quantization, offload, gradient-checkpointing, or LoRA make it tractable.

The lab's legitimate output, month over month, is a mix of:
- **Positive-result papers** on architecture, training, and efficiency at any tractable scale (e.g., dense vs MoE comparisons, recipe transfers, post-training experiments, novel training schedules, sparsity / quantization recipes that actually move the needle). Target: publishable contributions at ICLR/NeurIPS workshop tier.
- **Methodology / governance contributions** on autonomous-agent research labs (anti-forgery, pre-registration, unanimous compromise, RO-CO, Phase-1 B'+C'+D' agent contracts) — the lab itself as research artifact.
- **Reproducibility reports** attempting to replicate small-model claims from current papers. Successful replications are publishable evidence; failed replications are honest negative-results.
- **Envelope / negative-result papers** when a question's evidence lands negative (Program 1 is the first; future programs may or may not produce more). These are *one of four* output types, not the lab's defining purpose.

Program selection at PI + Director unanimous gate should favor questions that fit this real-mission scope. The aspirational mission informs *which* questions matter (those that test scaling / efficiency / capability gaps); it does not require each program to *attempt* the aspirational mission directly. The PI's selection bias should be toward questions where the answer is genuinely *unknown* in the published literature — that's where both positive findings and honest negative results carry the most scientific value.

### Reframe rationale (D-114, 2026-04-18)
Full context in `data/mission_reframe_2026-04-18.md`. TL;DR: Program 1's envelope paper formally established the aspirational mission is not reachable at current SOTA on 18GB. Rather than pivot the aspirational mission (which would weaken the lab's rigor), the lab adds a Real Mission layer that captures what it is *actually* producing: rigorous research artifacts at tractable scale. Both missions coexist; the aspirational one forces quality, the real one defines output.

## Constraints
- 18GB unified RAM — HARD LIMIT. Everything must fit.
- No cloud compute, no external APIs, no frameworks
- Everything from scratch: tensor library, model architecture, tokenizer, training, inference
- C17 + Metal + Swift toolchain
- iCloud Drive available for storage/backups/checkpoint overflow
- Time is not a constraint — quality over everything
- Chunked data streaming: download, process, train, delete, next chunk. No dataset size limit.

## Code Standards
- C: C17, `-Wall -Wextra -Werror -mcpu=apple-m3`, formatted with clang-format
- Metal: optimized for M3 Pro's 18 GPU cores, SIMD width 32
- Swift: `-O -target arm64-apple-macosx14.0`
- Python: type hints, pytest for tests
- TDD always: write test first, then implementation
- Every allocation has a clear owner and lifetime

## Build
```bash
make all        # Build C core + Metal shaders
make test       # Run all tests (C + Python)
make benchmark  # Run performance benchmarks
make hwmon      # Check hardware status
make lab-start  # Start the autonomous AGI lab
make lab-status # Check lab progress
```

## Tools (in tools/)
- `experiments.py` — experiment tracker (SQLite)
- `research_digest.py` — cross-stream knowledge synthesis
- `hwmon.py` — hardware monitor (CPU/GPU/memory/thermal)
- `benchmark.py` — performance benchmarks
- `arxiv_reader.py` — arxiv paper search + download
- `mathengine.py` — symbolic math via SymPy
- `verify.py` — C-vs-SymPy numerical verification
- `visualize.py` — plotting and visualization
- `preflight.py` — pre-training resource check

## Agent Organization
This project is run by an autonomous agent lab (30 seed roles across Research, Engineering, Evaluation, Quality, Knowledge, Communication, and Meta layers — L1-L8). All agents run on `claude-opus-4-7` at max effort. See `docs/superpowers/specs/2026-04-17-scientific-research-lab-overhaul.md` for the full program-based lab design (Cycles→Programs transition in Cycle 31-32 / D-109 onward).

## State Files (post-D-117 memory refactor)

Primary entry (read first, always):
- `data/memories/session_brief.md` — runner-written pre-session snapshot
- `data/memories/INDEX.md` — directory of all memory files

Tiered memory content:
- `data/memories/current.md` — hot tier, active state (cap 40 KB)
- `data/memories/log.md` — log tier, recent decisions (cap 30 KB rolling; overflow archived)
- `data/memories/mission.md`, `governance/`, `shared.md`, `killed.md`, `procedures.md`, `programs/` — wiki tier (50 KB total cap, 15 KB soft per-file)

Tool: `tools/memory.py` implements Anthropic memory_20250818 protocol. See spec at `docs/superpowers/specs/2026-04-19-memory-tool-and-wiki-refactor.md`.

Agent and program state:
- `data/agents/<role>/{procedural.md,episodic/,semantic.md}` — per-role CoALA memory (NOT part of memory tiers)
- `programs/<program>/` — program deliverables, papers, closure memos (NOT part of memory tiers)

Legacy paths (archived with breadcrumbs at their old locations post-migration):
- `data/state.md`, `data/decisions_recent.md`, `data/director_log.md`, `data/pi_notes.md`, `data/shared_knowledge.md`, `data/killed_ideas.md`, `data/procedures.md`, `data/index.md`, `data/mission_reframe_2026-04-18.md`, `data/checkpoints/ARCHIVED.md`, `programs/portfolio.md`

## Rules for All Agents
- Think on disk, not in context. Write to journal progressively.
- All work in atomic units that complete within ~10% of session capacity.
- Log every decision with reasoning.
- If you modify shared state, note what you changed and why.
- Never blow the 18GB memory budget. Optimization team has veto power.
- Use the plugins and skills specified in your agent prompt.

## Publication (added 2026-05-12)

Programs whose closure memos opt in via `publish_to_portfolio: true` (plus the `public_title` / `public_summary` / `public_source_artifacts` / `public_type` / `public_tags` frontmatter fields) become public pages on the operator's portfolio site. See `PUBLISHING.md` in the AGI repo root for the opt-in field set and discipline invariants.

The Director (or whoever commits a closure memo) should run, as part of the close ritual:

```bash
python3 tools/append_publish_candidate.py --all
```

This scans every `programs/*/closure_memo.md` and writes manifests for the eligible ones. The script is idempotent and safe to invoke repeatedly. Non-eligible memos (no `publish_to_portfolio` flag) are silently skipped — that's the operator's default veto.

Publication is NOT a substitute for the lab's existing ratification discipline. PI + Director co-sign on the closure memo is the first gate; the website-side curator's quality pipeline is the second.
