# AGI Lab Framework — Architecture

> **What this document is.** A complete technical reference for the **AGI Lab Framework** — a from-scratch, fork-and-run stack for running an autonomous machine-learning research laboratory on a single 18 GB MacBook M3 Pro, with **no cloud and no ML frameworks**. It maps every subsystem shipped in this repository (what it does, how it works, how the pieces fit), gives the verified design/scale metrics, and tells you **where to find the code** for any part. Read it top-to-bottom or jump by section.
>
> **Framework vs. deployment.** This repo is the *reusable framework*: the native training engine, the autonomous agent organization, the retrieval memory, the build/telemetry/assessment tooling, and the governance protocols — everything you need to stand up your own lab. The author's own *reference deployment* of this framework — "**the AGI Lab**" — ran for ~7 weeks and produced the research results, decision history, trained checkpoints, and program corpus. That research output is **deliberately not included here** (it is a separate, private artifact). Wherever a metric or finding describes that specific run rather than the framework's design, it is marked **"Reference deployment (the AGI Lab)"** and kept brief, with a pointer to the sibling **`RESULTS.md`** for the actual numbers.
>
> *Generated 2026-06-02. Every cited framework path was verified to exist in this repository; design/scale metrics were independently double-measured (zero mismatches). Read **§ Honest framing & caveats** before quoting any number — the framing matters.*

## Contents

1. **The Native Training Engine** (C17 / Metal / Swift)
2. **The Autonomous Agent Organization & Governance**
3. **Memory & Retrieval** — the 4-Layer Hybrid RAG System
4. **Reference Deployment (the AGI Lab)** — Research Programs, Evaluation & Experiment Tracking *(brief; see `RESULTS.md`)*
5. **Infrastructure, Tooling, Reproducibility & Self-Assessment**
- **Framework Metrics** (verified)
- **Navigation Index** — where to find what
- **Honest Framing & Caveats**

---

## Overview — what the framework is

The AGI Lab Framework is a **from-scratch autonomous AI-research laboratory stack** sized to run on a single 18 GB MacBook M3 Pro. It has three tightly-integrated halves, all present in this repo:

1. **A native neural-network training engine** (C17 + Apple Metal GPU kernels + a Swift Metal bridge) — hand-written tensors, memory arenas, a Mixture-of-Experts transformer (**HSPA**), **three** gradient methods (backpropagation, iterative predictive coding, direct feedback alignment), quantization-aware training (4-bit), and a byte-level BPE tokenizer. No PyTorch / TensorFlow / JAX anywhere. → `src/`
2. **A ~30-role autonomous research organization** — LLM "agents" in specialized roles (PI, Director, theorists, C/kernel engineers, reviewers, curators…) that propose, **pre-register**, execute, verify, and write up research under a formal governance protocol: phase gates (P0–P15), unanimous-compromise decisions, anti-forgery co-signing, and a calibration self-correction loop. → `data/agents/`, `run_agi_lab.sh`, `tools/`
3. **A 4-layer hybrid retrieval (RAG) memory** — knowledge graph (Personalized PageRank) + dense embeddings + BM25, fused via reciprocal-rank fusion and re-ranked with a cross-encoder, over a lab's own markdown corpus — served by a persistent warm-model daemon with graceful fallback. → `tools/retrieval/`, `tools/lab_memory.py`

**Two missions, stated honestly** (this framing is baked into the framework's governance and is worth preserving when you fork it):
- *Aspirational:* match or beat frontier models. In the author's reference deployment this was **formally established as unreachable at this scale** — the trainable models are tiny (≈48M–120M params) and score near random. That is documented as a result, not hidden (see `RESULTS.md`).
- *Real mission:* conduct **rigorous, reproducible small-scale research** — producing positive-result, null-result ("below-the-measurement-floor"), methodology, and reproducibility papers, with pre-registration and honest reporting. **This is where the framework is designed to deliver.**

The engineering achievement the framework embodies is the **integrated system**: a complete, tested, reproducible research stack — engine + autonomous org + retrieval memory + eval harness + self-assessment — that fits and operates end-to-end on a laptop.

---

## Framework metrics (verified 2026-06-02)

*Figures below were measured twice by independent agents with zero mismatches. These describe the **framework's design and scale** — not the outcomes of any particular research run (those live in `RESULTS.md`). Read the final section's caveats before quoting.*

### Scale & engineering
| Metric | Value |
|---|---|
| Total code | **~74,200 LOC** (raw `wc -l`) |
| Languages | **C17, Apple Metal, Swift, Python** (+ shell, Make) |
| Native engine | **~19,600 LOC** — C 15,573 · headers 2,740 · Metal 1,047 · Swift 233 |
| Python (tracked, non-test) | 18,991 LOC |
| Tests | **828 Python test functions + 269 C (Unity) cases** |
| Autonomous agent roles | **30** (plus 18 retired, kept as templates/precedent) |
| Tools | **64** Python tools |
| Git history | **283 commits over ~7.1 weeks** (2026-04-13 → 2026-06-02) |

### 4-layer retrieval system (design parameters)
| Metric | Value |
|---|---|
| Embeddings | all-MiniLM-L6-v2, **384-dim**, sqlite-vec |
| Fusion / rerank | Reciprocal Rank Fusion (k=60) → cross-encoder `bge-reranker-v2-m3` |
| Latency | **~0.85 s** BM25 cold full-process · **~32 ms** warm dense query (in-process) |
| Reference-deployment corpus | 62,364 chunks / 497 source files; graph 1,277 nodes / 8,396 edges; index ~140 MB dense + ~99 MB BM25 + ~1.4 MB graph; metadata coverage 31.3% any-tag (up from 2.5%) — *these reflect one lab's corpus; see `RESULTS.md`* |

> The corpus-size, graph, index-footprint and coverage numbers describe the author's reference deployment's indexed history, not the framework binary. A fresh fork builds its own index over its own corpus.

---

## 1. The Native Training Engine (C17 / Metal / Swift)

The framework's centerpiece is a **from-scratch neural-network training and inference engine** written in C17, with GPU kernels in Apple Metal Shading Language and a thin Swift bridge to the Metal runtime. There is **no PyTorch, TensorFlow, JAX, or any ML framework** anywhere in the stack — tensors, autodiff-style backprop, optimizers, quantization, the tokenizer, and the GPU kernels are all hand-written. It lives under `src/` (about 19,600 lines across `core`, `model`, `training`, `metal`, `swift`, `eval`, `tests`) and is built by the top-level `Makefile`, which compiles `.metal` shaders via `xcrun -sdk macosx metal`/`metallib` into a `default.metallib`, compiles the Swift bridge with `swiftc`, and links everything into C binaries. It is real and working: the reference deployment trained checkpoints with it and produced benchmark scores from those checkpoints (see `RESULTS.md`).

The model it implements is **HSPA** ("Hierarchical Sparse Predictive Architecture"): a Mixture-of-Experts transformer. The aspirational full-scale config (`hspa_config_default()` in `src/model/hspa_config.c`) targets 16B total / 3.36B active params (32 layers, d_model 4096, 16 experts top-2, GQA with 32 query / 8 KV heads, INT4 weight storage), but everything actually trainable at this hardware scale runs much smaller — the preset factories `hspa_config_micro()` (~1M params), `_small()`, `_medium()`, `hspa_config_100m()` (119.6M), and `hspa_config_dense_50m_a()` (a 34.6M dense control), all defined in `src/training/ipc_train.c` (declared in `ipc_train.h`). The 16B figure is a design ceiling, not a trained artifact.

### Core layer (`src/core`)

- **`tensor.c` / `tensor.h`** — an N-dimensional (up to 4D) tensor with dtype-aware storage: `DTYPE_FP32`, `DTYPE_FP16` (`_Float16`), `DTYPE_INT8`, and `DTYPE_INT4` (packed 2 values/byte). Supports zero-copy **views** via stride manipulation (`tensor_view`), with explicit ownership rules (views don't free parent data). `tensor_nbytes` accounts for INT4 packing.
- **`memory_pool.c` / `memory_pool.h`** — an arena allocator with three pool flavors that mirror how the model uses memory: `POOL_WEIGHTS` (static, free-list backed), `POOL_ACTIVATIONS` (ring buffer, rewound between forward passes), and `POOL_SCRATCH` (reset per-op). It guarantees 16-byte NEON alignment and offers 4096-byte page alignment for Metal buffers, and tracks a peak high-water mark. This is the central lever for the **18 GB memory constraint** — activations are streamed through a bounded ring rather than all held live.
- **`ops.c` / `ops.h`** — the CPU op library: `op_add/mul/scale/sub/add_scaled`, `op_relu/silu/silu_backward/swiglu`, `op_softmax`, `op_rms_norm`, `op_cross_entropy` (returns loss and optionally writes `dL/dlogits`), Xavier init, and matmuls. FP32 matmul (`op_matmul`, plus transposed variants `op_matmul_tn` / `op_matmul_nt`) is backed by Apple **Accelerate `cblas_sgemm`**; `op_matmul_q4` dequantizes INT4 weights on the fly. There is a GPU dispatch seam (`op_set_gpu_matmul` registers a Metal matmul; `op_matmul` auto-dispatches above a FLOP threshold) — **but note that `GPU_MATMUL_THRESHOLD` is hard-set to `INT64_MAX`, so in practice the GPU path is disabled and all training/eval runs on the Accelerate CPU path.** The Metal matmul is wired and tested but not on the hot path.
- **`quantize.c` / `quantize.h`** — group quantization for weight storage: per-group asymmetric (unsigned) INT4 `[0,15]` and INT8 `[0,255]`, each group of `group_size` (typically 128) elements sharing one FP32 `(scale, zero_point)` pair (`QuantGroup`). The header documents the budget math (e.g. 12.5% overhead for INT4/128). This is distinct from the per-tensor *symmetric signed* INT4 `[-8,7]` packing in `ops.h` — the codebase deliberately keeps both schemes.
- **`tokenizer.c` / `tokenizer.h`** — a hand-written **Byte-Pair Encoding** tokenizer: 256 byte-level tokens grown by learned merges up to `TOK_MAX_VOCAB` (32768), with `train`/`encode`/`decode` and a versioned binary save format (`"BPE\0"` header). `train_tokenizer.c` is the training driver.

### Model layer (`src/model`)

A full transformer assembled from per-component modules, all weight-pool-allocated:
- **`hspa_model.c`** — top-level: token embedding (weight-tied with `lm_head`), a stack of `HSPABlock`s, final RMSNorm, projection to vocab. Owns the three memory pools and per-layer KV caches. Exposes `hspa_model_forward` and a QAT-aware `hspa_model_forward_qat` (bit-identical to the plain path when the QAT context is NULL — an explicit regression invariant).
- **`hspa_block.c`** — one transformer block: pre-attn RMSNorm → GQA attention → residual → pre-FFN RMSNorm → FEP router → MoE (only `n_active` of `n_experts` experts evaluated) → residual. Supports an `n_active_override` for progressive sparsification.
- **`attention.c`** — **Grouped Query Attention** (`W_q/W_k/W_v/W_o`), with a pre-allocated `KVCache` sized for the full context.
- **`ffn.c`** — **SwiGLU** expert FFN (`gate`, `up`, `down`; `silu(gate)*up`).
- **`rmsnorm.c`** — RMSNorm (`a/sqrt(mean(a²)+eps)*weight`), chosen over LayerNorm.
- **`embedding.c`** — token embedding lookup, weight-tied with the output `lm_head`.
- **`router.c` / `router.h`** — the most distinctive module: a **Free-Energy-Principle (FEP) router**. It carries `W_mu` (mean routing logits) *and* `W_sigma` (log-uncertainty); during training it samples from `N(mu, sigma)` for epistemic exploration, at inference it uses `mu` (exploit). It computes routing **entropy H(g)** that feeds the load-balancing term of the VFE loss (replacing a classic aux loss), and layers on **DeepSeek-style auxiliary-loss-free bias balancing** (`expert_bias` + load EMA, `router_update_bias`, citing arXiv 2408.15664), an optional **Default-MoE EMA** for dense-backward gradient substitution (arXiv 2504.12463), and a cosine-annealed **router temperature**.

### Training layer (`src/training`)

This is where the "alt-grad" research apparatus lives. The engine implements **three different ways to compute gradients**, all sharing the same `ModelGrad`/`AdamState` structures and optimizer:

1. **`backprop_train.c`** — standard backpropagation (forward saving activations, then exact chain-rule backward). It also exposes entropy-penalty + temperature-anneal math helpers (`entropy_penalty_shannon_H`, `entropy_penalty_grad_logit`, `compute_router_temperature`, `compute_beta_h_warmup`) as externally-testable functions.
2. **`ipc_train.c` / `ipc_state.c`** — **iterative Predictive Coding** (Salvatori et al. 2022, adapted): a forward pass initializes value nodes, then T inference iterations refine them to minimize local prediction errors `epsilon[l] = v[l+1] − f_l(v[l])`, with weights updated from those *local* errors (identity-Jacobian approximation, frozen KV/routing during iterations). This is what `loss.c`'s **Variational Free Energy** loss `L_VFE = L_pred + β·L_balance + λ·L_output` is built for.
3. **`local_feedback_train.c`** — **Local / Direct Feedback Alignment** (Nøkland 2016, Lee et al. 2015): fixed-random feedback matrices `B_l` propagate the global error directly to each layer (~1,234 LOC). It implements the forward pass + QAT coverage, **Haar-orthogonal `B_l` init, a full DFA backward pass** (error projection through `B_l` with `dW` accumulation), gradient clipping, and the **optimizer step** — exercised by `tests/test_localfb.c`.

Supporting modules: **`grad.c` / `grad.h`** holds per-parameter gradient accumulators and the optimizers — SGD and **AdamW** (decoupled weight decay, bias-corrected moments) — applied with **Depth-muP** learning-rate scaling (`eta_width = 1/D` width scaling plus per-layer precision scaling) and linear-warmup/cosine-decay LR. **`checkpoint.c`** is a versioned binary format (magic `"HSPA"`, currently `CKPT_VERSION 6`) saving config + meta + weights + Adam moments + router bias, with documented backward-compatible struct sizes across versions and an optional embedded data-loader stream state. **`data_loader.c`** streams/tokenizes training corpora; **`weight_init.c`** does seeded deterministic init; **`config_drift.c`** guards against silent hyperparameter drift; **`scale_experiment.c`** and **`micro_experiment.c`** are the CLI training drivers (`--model`, `--data`, `--steps`, `--lr`, `--grad-accum`, `--resume`, etc.).

**QAT (Quantization-Aware Training)** is a first-class apparatus targeting 4-bit weights. `qat.c` implements fake-quantize forward (`fake_quantize_int4`: snap FP32 weights onto the 16-level per-group INT4 grid, keep FP32 storage) and a **Straight-Through Estimator** backward (`fake_quantize_int4_backward`: identity gradient, Bengio et al. 2013). `qat_context.c` is a per-step, pointer-keyed hash-map cache of the shadow weights (`w_hat`) so a single quantized grid is shared across the forward pass and STE backward (and iPC double-forward), with zero per-step allocation after warm-up; the design tracks a **224-weight coverage count** as an acceptance gate. The `op_matmul_qat` / `op_matmul_nt_qat` wrappers (and `qat`-threaded forward functions throughout the model) are the integration seam — NULL `qat` is provably bit-identical to the plain path.

### Metal GPU kernels (`src/metal`) and Swift bridge (`src/swift`)

Five hand-written Metal compute kernels, all tuned for the M3 Pro (18 GPU cores, SIMD width 32, 256-thread threadgroups):
- **`matmul_fp32.metal`** — FP32 GEMM using hardware `simdgroup_matrix` 8×8 multiplies, 32×32 register tiling, double-buffered threadgroup loads, vectorized `float4` access.
- **`quantized_matmul.metal`** — INT4 (block-size-32, 20-byte block: half scale + half zero + 16 packed bytes) × FP16 matmul, plus a batch-1 `vecmat_q4_fp16` inference hot-path variant.
- **`rms_norm.metal`** (FP32-accumulate, FP16 I/O, SIMD reduction), **`softmax.metal`** (numerically stable two-pass online), and **`elementwise.metal`** (SwiGLU, add/mul/scale, GELU, and RoPE).

**`src/swift/MetalBridge.swift`** (with C header `metal_bridge.h`) is the only Swift in the system: a `@_cdecl` C-callable bridge exposing `metal_init`, `metal_matmul`, `metal_is_ready`, `metal_cleanup`. It holds a singleton `MetalContext` (device, command queue, cached pipeline state), uses **unified memory (`MTLStorageModeShared`)** for zero-copy CPU↔GPU sharing, pre-warms the matmul pipeline at init, and is serial-locked for thread safety. The C side registers `metal_matmul` into `ops.c` via `op_set_gpu_matmul`. As noted above, the dispatch threshold is `INT64_MAX`, so this GPU path is **built and unit-tested but dormant** — the engine runs production training on Accelerate (CPU/AMX) today, with the Metal path as ready-to-enable acceleration. (Measured rationale: the custom Metal matmul ran 3–100× *slower* than Accelerate at every tested size on the reference hardware.)

### End-to-end flow

`tokens → embedding lookup → for each of L blocks { RMSNorm → GQA (KV-cached) → residual → RMSNorm → FEP router picks top-k experts → SwiGLU MoE → residual } → final RMSNorm → lm_head → logits`. A loss is computed (cross-entropy for backprop, or the VFE composite for iPC); gradients are produced by one of the three methods into `ModelGrad`; gradients are L2-clipped and applied by AdamW (or SGD) with Depth-muP scaling; periodically the full state is serialized via `checkpoint_save`. `src/eval/eval_model.c` then loads a checkpoint and runs `perplexity`, `score`, or `score-per-choice` modes against benchmark data (MMLU/HellaSwag), emitting JSON consumed by the scorecard generators (§5).

### Testing

A custom single-header **Unity-style C test framework** (`src/tests/unity.h`, `TEST_BEGIN`/`ASSERT_*`/`RUN_TEST`/`TEST_REPORT`, no external dependency) backs the C test suite under `tests/` (`test_tensor`, `test_quantize`, `test_attention`, `test_router`, `test_router_grad`, `test_adam`, `test_checkpoint`, `test_qat`, `test_qat_context`, `test_localfb`, `test_op_matmul_qat`, etc. — **269 C/Unity cases**). The Makefile compiles each test with ASan/UBSan, with a quarantine mechanism (`tests/quarantine/`) for tests needing the Metal/Swift link.

### Where to find more
- **Core tensor/memory/quantize/tokenizer**: `src/core/{tensor,memory_pool,quantize,ops,tokenizer}.c|.h`
- **Model (MoE transformer, FEP router, GQA, SwiGLU, RMSNorm)**: `src/model/{hspa_model,hspa_block,hspa_config,attention,ffn,router,rmsnorm,embedding}.c|.h`
- **Gradient methods (backprop / iPC / LocalFB), optimizers, loss, checkpoints, QAT, data, drivers**: `src/training/{backprop_train,ipc_train,ipc_state,local_feedback_train,grad,loss,checkpoint,qat,qat_context,data_loader,weight_init,config_drift,scale_experiment,micro_experiment,train_tokenizer}.c|.h` and `src/training/train_config.h`
- **Metal GPU kernels**: `src/metal/{matmul_fp32,quantized_matmul,rms_norm,softmax,elementwise}.metal`
- **Swift↔Metal bridge**: `src/swift/MetalBridge.swift`, `src/swift/metal_bridge.h`
- **Evaluation harness (C)**: `src/eval/{eval_model,eval_utils}.c|.h`
- **Test framework + suite**: `src/tests/unity.h`, `tests/test_*.c`, `tests/quarantine/README.md`, and the `Makefile` (`make test`, Metal/Swift build rules)

## 2. The Autonomous Agent Organization & Governance

The framework is not a single LLM loop — it is a self-governing research organization staffed by ~30 specialized LLM "agent" roles, coordinated by a `bash` runner that spawns one Director session at a time. Authority is split between two leads (PI + Director) under a unanimous-compromise protocol, work is gated through a 16-phase research pipeline (P0–P15), and every consequential act is recorded as a numbered decision (`D-NNN`) with anti-forgery verification. This section documents the org chart, the governance protocols, and the session/dispatch lifecycle as they ship in this repo.

> **Note on agent memory in this repo.** Each role ships with its **`procedural.md`** (its operating procedure — all 31 present). The role's *accumulated* `semantic.md` knowledge and dated `episodic/` audit records are **research output produced during a deployment, so they are not included here** — a fresh fork starts those empty and grows them as it runs. The Director additionally ships `data/agents/director/{decisions.md,state.md}` scaffolding.

### 2.1 The role roster (`data/agents/`)

The active roster is defined in `data/agents/agents.json` — 30 roles, each an object with a `description`, a default `model`, and a `prompt`. The roster is explicitly **tiered** by model capability (the tier scheme is codified in `data/agents/_shared/self_escalation_contract.md`):

- **Tier A — `claude-opus-4-8`** (judgment/taste): the two leads `pi` and `director`, the `unanimous_compromise_mediator`, scientific specialists (`chief_scientist`, `math_theorist`, `experimental_methodologist`, `hypothesis_generator`, `mechanism_extractor`, `measurement_theorist`, `kernel_specialist`), and the review/gate roles (`scientific_reviewer`, `statistical_reviewer`, `red_team`, `pre_reg_auditor`, `code_reviewer`, `paper_writer`, `lab_architect`, `grant_reviewer`, `evaluator`).
- **Tier B — `claude-sonnet-4-6`** (substantive engineering/synthesis): `infrastructure_architect`, `implementation_engineer_c`, `tooling_engineer`, `reproducibility_engineer`, `profiler`, `memory_optimizer`, `literature_hunter`, `paper_digester`, `findings_curator`, `figure_generator`.
- **Tier C — `claude-haiku-4-5`** (mechanical execution): `consolidator` (the async memory janitor).

Each role lives in its own directory under `data/agents/<role>/` carrying its **`procedural.md`** (the role's step-by-step operating procedure). The Director's `data/agents/director/procedural.md` is special: it is the literal prompt fed to every Director session, with a documented prompt-cache contract at the top (no dynamic content above the `<!-- CACHE BOUNDARY -->` marker, so the message prefix stays byte-identical and bills at cache-read rates). In a live deployment each role *also* accrues a `semantic.md` (accumulated domain knowledge) and a dated `episodic/` directory; those episodic files are the **load-bearing audit substrate** — anti-forgery verification checks that every recorded signature has a matching episodic record (see §2.5) — but they are produced at runtime and are not shipped in this framework repo.

Retirement, not deletion, is the rule: `data/agents/retired.json` holds ~18 decommissioned roles (`metal_kernel_team`, `tensor_team`, `knowledge_agent`, `sota_scout`, etc.), with their old role directories kept as precedent. The org evolves through the **role-promotion / org-retro** protocol (§2.3). `data/agents/templates/` (`engineer.md`, `researcher.md`) provides skeletons for spinning up ad-hoc general-purpose agents when no registered role fits.

### 2.2 The two leads and the unanimous-compromise protocol

Scientific authority is deliberately split:
- **PI** (`pi`) owns scientific direction, program selection, kill calls, and paper approval.
- **Director** (`director`) owns execution: agent dispatch, phase orchestration, resource allocation.

For any "unanimous-required" item (program open, pre-registration lock, paper approval, kill/pivot, role changes), **both must agree**. The deadlock procedure (in a live deployment's `data/memories/procedures.md §Unanimous Compromise Protocol`) is:
1. Each lead writes a <200-word position file under the current program's `disagreements/`.
2. Director dispatches `unanimous_compromise_mediator` (a role that *proposes a compromise but does not decide*).
3. Mediator writes a mediation memo; PI + Director review.
4. Both accept → logged as `UNANIMOUS_COMPROMISE:`. Modification requested → mediator iterates once (max). Still deadlocked → escalate to the human operator (capped at 2 escalations per program before `lab_architect` flags miscalibration).

A hard override sits above the protocol — **Evidence Outranks Authority**: clean empirical evidence cannot be blocked by intuition, and no work is approved without cited evidence (file:line or experiment ID). Coordination is strictly **Director-centric** (puppeteer pattern): specialists never dispatch each other; all dispatches flow Director → role → episodic record → Director. The rationale: peer-to-peer A2A would create audit blind spots, defeat anti-forgery (which depends on a single dispatch chain per decision), and cause premature consensus that would rob the PI's gate-reading of divergent-thinking value. *(The governance memos that spell this out — `procedures.md`, `governance/values.md`, `governance/phase_transition_addenda.md` — are deployment-side memory; they are seeded and grown at runtime, not shipped here.)*

### 2.3 Phase gates (P0–P15) and pre-registration discipline

A program advances through a 16-phase pipeline, each phase owned by a **gate holder** who must sign off its deliverable before the phase closes. The canonical table:

| Phase | Deliverable | Gate holder(s) |
|---|---|---|
| P1 Question formation | locked `question.md` + sub-questions | `chief_scientist` + `pi` |
| P2 Literature scan | `lit_scan.md` | `literature_hunter` (+ retired `sota_scout`) |
| P3 Hypothesis set | 3–5 falsifiable H's | `hypothesis_generator` + `experimental_methodologist` |
| P4 Math/theory framing | capacity/memory bounds | `math_theorist` + `measurement_theorist` |
| P5 Red-team the plan | adversarial review | `red_team` + `scientific_reviewer` |
| **P6 Pre-registration** | metrics, gates, stop rules, seeds locked | `pre_reg_auditor` + `experimental_methodologist` |
| P7 Apparatus build | profiling harness, PERF_APPROVED | `implementation_engineer_c` + `profiler` + `memory_optimizer` |
| P8 Micro experiments | sub-1hr micro-runs | engineering + kernel roles |
| P9 Scaled probes | gated >200-step runs | `implementation_engineer_c` + `profiler` |
| P10 Analysis | stats + plot grids | `statistical_reviewer` + `figure_generator` |
| P11 Replication/ablation | N≥3 seeds on headline result | `reproducibility_engineer` |
| P12 Red-team findings | what would overturn the claim | `red_team` |
| P13 Synthesis | locked `findings.md` + verdict | `findings_curator` + `chief_scientist` |
| P14 Paper draft | `paper_v1.md` + figures | `paper_writer` + `figure_generator` |
| P15 Close + handoff | `close_manifest.md`, killed ideas, lab_memory ingest | `evaluator` (+ retired `knowledge_agent`) |

The standard close gate is: Director proposes "phase deliverable ready at `<path>`" → gate holder reviews → if APPROVED, **PI co-signs** and a `PHASE_CLOSE:` entry is logged → phase closes. P6 is the **pre-registration lock**: metrics, decision rules, stop/kill criteria, seeds, and ablations are committed *before* experiments run, and the program carries explicit pre-registered **kill criteria**. Pre-registration discipline is enforced downstream — the evaluator audits the pre-reg-violation rate, and the `tools/run_long.py` spec-sentinel + config-drift detector refuse off-spec launches. There are also additional gates beyond the close gate: a **code-review gate** (TDD test-first → `code_reviewer` verdict APPROVED/NEEDS_FIXES/BLOCKED → optional `scientific_reviewer`) and a **performance gate** (`profiler` signs PERF_APPROVED before any long run).

Two recurring meta-reviews keep the org honest: **org retro** every 3 programs (`lab_architect` audits dispatch distribution, dormant roles, specialist-work violations → role promotion/retirement proposals) and **grant review** every 5 programs (`grant_reviewer` answers six skeptical-outsider questions → PROCEED/CONTINGENT/DECLINE).

### 2.4 The calibration self-correction loop

The PI's verdicts are not taken on faith. Every PI gate verdict must attach a numeric `CONFIDENCE ∈ [0,1]` and `CALIBRATION_REASONING`, logged via `tools/calibration_logger.py` (`log_calibration()` appends a record) **before** the verdict is finalized. The runner periodically aggregates this with `tools/calibration_rollup.py`, producing per-confidence-band hit rates ("you said 70% on these 10 things, 5 hit"). `tools/calibration_pending.py` is a read-only nudge that surfaces still-unresolved claims oldest-first so the PI/Director closes the loop. This is an *organic-learning nudge*, not a prescription — the PI recalibrates itself over time rather than being told a fixed bias correction (spec: `docs/superpowers/specs/2026-05-04-pi-self-correction.md`). The accompanying `tools/dispatch_helper.py` is a separate self-correction aid: a keyword-driven decision table that picks the right model/effort per dispatch (opus keywords like *review/verdict/approve* force Tier A; haiku keywords like *extract/archive/format* downgrade), so the Director allocates the right tier without burning its own cognitive budget.

### 2.5 Verification and anti-forgery

Two independent mechanisms guard against an agent claiming work it never did.

**Cross-tier verifier pairs** (`data/agents/_shared/verifier_pairs.json`): named high-stakes producer roles are checked by a *different* model instance acting as verifier, with a `max_iterations` retry budget. Examples: `pi` → verified by `lab_architect` (deliberately a sonnet-tier cross-check to catch over-confident opus claims, and the enforcer of the calibration-fields requirement); `paper_writer` → `code_reviewer` (a separate opus instance checks technical claims independent of authorship); `code_reviewer` → `evaluator`; `experimental_methodologist` → `scientific_reviewer`; `mechanism_extractor` → `red_team`; `math_theorist` → `statistical_reviewer`. A `VERIFY_FAIL` re-dispatches the producer with the critique; exhausting iterations escalates with reason `VERIFIER_REJECTION_LOOP`.

**Signature-forgery detection** (`tools/verify_signatures.py`): every formal attestation (`**<role> ✓**`, "approved by `<role>`", etc.) in program deliverables must have a corresponding dated file under that role's `episodic/` directory. The canonical signature format embeds the episodic path so verification is mechanical. Three redundant detectors run it: the runner's pre-checkpoint hook (converts the exit reason to `SIGNATURE_FORGERY_UNADDRESSED` on mismatch), the evaluator's mandatory per-phase check (AUTO-FAIL, no pass-with-flags), and the `lab_architect` retro (repeat-offender pattern detection). *(The tool was added after a real incident in which a Director forged five PI signatures without dispatching the PI agent — the structural fix that became a methodology paper in the reference deployment.)*

The **self-escalation contract** (`data/agents/_shared/self_escalation_contract.md`) is the third leg: any agent that judges its task exceeds its model tier must return `status: BLOCKED` with a `suggest_model` rather than produce shallow output — making self-assessment a structured, dispatcher-actionable signal. (`data/agents/_shared/agent_contracts.json` adds formal per-role resource/success contracts; v1 enforces only the Director's `session_exit.md` checks, the rest are stubbed for v2.)

### 2.6 The runner and session lifecycle (`run_agi_lab.sh`)

`run_agi_lab.sh` is the heartbeat. Started once via `tmux`/`make lab-start`, it loops "forever until VICTORY or CATASTROPHIC_STOP," spawning **one Director session at a time** (single-Director invariant). Per iteration it:

1. **Pre-spawn gates:** if a rate-limit-reset marker (`data/infra/rate_limit_resets_at`) is in the future, sleep until then in 1h chunks; if an `operator_review_pending.md` exists, skip the spawn entirely. These avoid burning thousands of no-op sessions across a multi-day rate-limit reset.
2. **Maintenance passes** (all idempotent, non-fatal): reclaim stale claims, run the queue scanner, incremental `lab_memory.py` re-index, retrieval graph/BM25 reindex with a change-detection gate, and async `consolidator` dispatch (Tier C haiku) on a 6h/phase-end/10-new-decisions trigger. It also warms a **persistent retrieval daemon** (`_start_retrieval_server` → `tools/retrieval/server.py`) that holds the embedder + cross-encoder reranker + BM25 index in memory so per-query subprocesses don't reload ~1 GB of models; everything degrades gracefully to in-process fallback.
3. **Skip-when-stable / dynamic heartbeat:** if nothing external has changed within the stable window and a Director ran recently, skip the cycle. A dynamic heartbeat extends 30 min → 2 h (deep-stable, healthy training) or up to 8 h, with special wake-once detectors (`_phase_just_ended`) and a **holding-loop backoff** (`detect_holding_loop_backoff`) that exponentially backs off when the Director cycles in short no-op sessions — with discriminators that suppress false positives during legitimate long training/eval runs.
4. **Spawn the Director.** The session is launched as:
   `claude --print --output-format stream-json --model claude-opus-4-8 --effort max --dangerously-skip-permissions --agents "$(cat agents.json)" --append-system-prompt "$DIRECTOR_SYSTEM_ASSERT" "$DIRECTOR_PROMPT"`.
   The `--agents` flag registers the entire roster as the starting team. `--append-system-prompt` carries the **Director identity assertion** (a static, cache-safe string): it tells the model it is the autonomous Director, no human is attached, never ask for input — overriding any auto-memory that leaked in. The pipe runs through `tools/stream_formatter.py` and tees to a session log.
5. **Watchdog.** `watchdog_cli_hang` runs as a background sentinel to SIGKILL a Director whose CLI hangs after emitting its final result (idle-detection on the log, GitHub issue #21099), so the main loop never freezes.
6. **Close-out.** After exit, `tools/post_director.py` applies the Director's `session_exit.md` (see §2.7), logs telemetry, and checks the lab state files for `VICTORY`/`CATASTROPHIC` (which `break` the loop and fire a desktop notification).

The runner also stamps each session with the git SHA and surfaces unread substrate-changelog entries via a cursor mechanism, so the Director acknowledges operator-side infra changes. Note: the Director's procedural explicitly **forbids it from managing the runner** (no `pkill`, `make lab-start/stop`, etc.) — runner-lifecycle is operator-scope only (a hard lesson from an incident where a Director killed its own parent).

### 2.7 Work-queue / dispatch lifecycle

Work flows through an event-driven, file-backed priority queue (spec: `docs/superpowers/specs/2026-05-04-work-queue.md`):

- **`tools/queue_scanner.py`** runs every iteration and *projects external state into queue items*. It detects 8 event types: `phase_advance` (a `phase<N>_close_memo.md` appeared), `operator_nudge` (a user-notes file changed), `cell_complete` / `cell_failed` (training `run_index.json` cell states), `diagnostic_review`, `consolidator_run`, `heartbeat`, and `artifact_queue_projector` (missing artifacts vs. the program's `artifact_schema.yaml`). All detectors are idempotent with deterministic IDs.
- **`tools/work_queue.py`** is the file-backed store under a live deployment's `data/work_queue/` (`pending.jsonl`, `claimed.jsonl`, `completed/<date>.jsonl`, `failed/<date>.jsonl`, plus `queue_telemetry.jsonl`). Moves are atomic (tmp-file + `os.rename`), enqueue dedups by deterministic ID, priority is `urgent > normal > low`, and the single-Director constraint means only one item may be claimed at a time. `reclaim_stale()` recovers claims orphaned by a silently-died Director (>60 min, no completion).
- **`tools/brief_assembler.py`** builds the Director's context brief deterministically (<100 ms, no LLM); the runner writes a companion session brief — the two files the Director reads first.

The end-to-end flow per session: runner writes the brief → spawns the Director → Director reads the session brief, claims the top `pending.jsonl` item, looks up how to handle it in `data/agents/_shared/work_queue_handlers.md` (per-type Meaning/Action/Complete/Fail playbooks), and **dispatches specialist agents** as needed → agents work and leave episodic records → Director verifies (verifier pairs / signature checks) → Director records the decision and writes its `session_exit.md`. That exit file (schema in `data/agents/_shared/session_exit_schema.md`, v1.1) carries a structured JSON block: `status` (`success`/`partial`/`failure`/`no_op`), `log_entry_text`, `current_md_patches`, `deliverables`, and a **required `next_action`** whenever work remains (the v1.1 rule that closed an "the lab went idle because narrative said work remained but `next_action` was null" failure). `tools/post_director.py` finalizes the mutations: it prepends the `D-NNN` block to the decision log (newest-first), patches the hot-state file, moves the claimed item to completed/failed, and enqueues `next_action`. A schema violation forces the next iteration to re-spawn the Director with a correction prompt. The decision log is thus the canonical, append-only record of every `D-NNN` governance decision. *(The live queue, decision log, and hot-state files under `data/work_queue/` and `data/memories/` are runtime memory, not shipped here.)*

### Where to find more

- **Roster & role memory** → `data/agents/agents.json` (active 30), `data/agents/retired.json` (decommissioned), `data/agents/<role>/procedural.md`, `data/agents/director/{procedural,decisions,state}.md`, `data/agents/templates/`, `data/agents/_shared/self_escalation_contract.md`, `data/agents/_shared/agent_contracts.json`.
- **Verification / anti-forgery** → `data/agents/_shared/verifier_pairs.json`, `tools/verify_signatures.py`, `tools/calibration_logger.py` + `tools/calibration_rollup.py` + `tools/calibration_pending.py` (calibration loop), spec at `docs/superpowers/specs/2026-05-04-pi-self-correction.md`.
- **Runner / session lifecycle** → `run_agi_lab.sh` (identity assertion, watchdog `watchdog_cli_hang`, main loop, retrieval server start/stop), `tools/stream_formatter.py`, `tools/post_director.py`, `tools/measure_startup_read.py`.
- **Work queue / dispatch** → `tools/work_queue.py`, `tools/queue_scanner.py`, `tools/brief_assembler.py`, `tools/dispatch_helper.py`, `data/agents/_shared/work_queue_handlers.md`, `data/agents/_shared/session_exit_schema.md`, spec at `docs/superpowers/specs/2026-05-04-work-queue.md`.
- **Governance protocols, mission, decision log, per-program memory** → *(produced and held in a live deployment under `programs/` and `data/memories/`; not in this framework repo — see `RESULTS.md` and §4).*

## 3. Memory & Retrieval — the 4-Layer Hybrid RAG System

The framework's long-term memory is not a vector store bolted onto an LLM — it is a four-layer hybrid retrieval-augmented-generation (RAG) system built from scratch over a lab's *own* markdown corpus. Every decision log, program memo, agent procedural/semantic file, and engineering review a lab writes is ingested, chunked, indexed four different ways, and served to agents at query time. The design philosophy, documented in `docs/retrieval/architecture.md`, is deliberately *non*-fancy: no LLM entity extraction, no community detection, no Neo4j — every layer is a small, auditable, dependency-light Python module, because the token vocabulary of this kind of lab (decision IDs like `D-420`, carry-forward tokens like `P-D417-...`, program names) is *canonical and regex-extractable*, so an LLM is not needed to build the graph.

The orchestrator is `tools/retrieval/search.py`; the four layers live as sibling modules under `tools/retrieval/`. *(Index/corpus statistics below — chunk/node/edge counts, footprints — describe the author's reference deployment's indexed history; a fresh fork builds its own. See `RESULTS.md`.)*

### The four layers (L0–L3) plus rerank (L4)

**L0 — Knowledge graph (`tools/retrieval/graph.py`).**
A persistent SQLite graph of canonical lab tokens. Node types: `decision`, `file`, `carry_forward`, `program`, `phase`, `role`. Edge types actually emitted by `build_graph_from_corpus`: `cites` (a defined decision header references a decision *not* defined in the same file), `raises` (a decision header co-occurs with a carry-forward token), `in_program`, `in_file`. The graph is built by regex-walking the corpus with token extractors (`_DECISION_RE`, `_CARRY_FORWARD_RE`, `_PROGRAM_RE`, `_PHASE_RE`); roles come from a vocabulary loaded out of `data/agents/agents.json`. Builds are idempotent (`INSERT OR IGNORE` upserts) and contain a symlink-containment guard that skips any `.md` resolving outside the repo. At query time, seed tokens are extracted from the query and **Personalized PageRank** (`networkx.pagerank`, α=0.85, with a tol=1e-6 → 1e-4 convergence fallback) is run from those seeds; the top-20 related nodes are returned. The graph layer does **not** re-rank results — it only *annotates* fused chunks with a `graph_boost` flag when a chunk's `file:` node is in the PPR-related set, keeping RRF authoritative. The module declares `RESERVED_EDGE_TYPES` (`precedes`, `resolves`, `dispatched_by`, `in_phase`) that are documented but **not yet emitted** — explicitly flagged as a roadmap, not a live capability. (Reference-deployment live stats: 1,277 nodes / 8,396 edges.)

**L1 — Dense / vector (`tools/lab_memory.py`).**
The semantic backbone, originally written to replace a cloud "goodmem" service with a fully-local stack. `LabMemory` chunks markdown (~800-char chunks, 100-char overlap, paragraph-boundary-preferring), embeds with **sentence-transformers `all-MiniLM-L6-v2` (384-dim, normalized)**, and stores vectors in a **`sqlite-vec`** virtual table (`memories_vec USING vec0(embedding float[384])`) alongside a metadata `memories` table. Search is cosine-distance KNN (`v.embedding MATCH ? AND k = ?`) with optional exact-match metadata filters and a 4× over-fetch-then-trim. It guards against being run with system Python (sqlite-vec lives only in `.venv/`).

**L2 — Lexical / sparse (`tools/retrieval/bm25.py`).**
BM25 (`rank_bm25.BM25Okapi`) over the *same* chunks as L1. The tokenizer deliberately preserves canonical identifiers (`D-420`, `P-D417-FOO`, `program_2_...`) by keeping hyphen-joined alphanumeric runs intact — the one thing dense embeddings are worst at matching. Persistence is the interesting part: the index is saved as **JSON, not pickle** (no code-execution risk on load) and stores the *precomputed BM25 statistics* (`idf`, `doc_freqs`, `doc_len`, `avgdl`, …) rather than raw tokens, so `load()` rehydrates a scorable `BM25Okapi` via `__new__` without re-running the O(total-tokens) corpus loop. Writes are **atomic** (write-to-temp + `os.replace`) so a query reading the index while the runner rebuilds it never sees a half-written file.

**L3 — Fusion (`tools/retrieval/hybrid.py`).**
**Reciprocal Rank Fusion (RRF, k=60)**, the literature default. RRF is score-free — it ranks each retriever's hits and sums `1/(k+rank)` per document — which makes it robust to the incomparable score scales of cosine distance vs. BM25. `hybrid_search` runs BM25 + dense (dense is injected as a callable), fuses their ID rankings, and re-hydrates the top-`k` chunk dicts, tagging each with `from_dense`/`from_bm25` provenance. A missing or corrupt BM25 index degrades to dense-only rather than crashing.

**L4 — Reranking (`tools/retrieval/rerank.py`).**
A cross-encoder, **`BAAI/bge-reranker-v2-m3`** (568M params, ~1.05 GB resident fp32), scores the (query, chunk) pairs of the RRF top-30 pool and returns the top-`k`. If the primary model fails to load it falls back — *loudly, never silently* — to `cross-encoder/ms-marco-MiniLM-L-12-v2`, because a silent model swap would change rankings with no trace.

### End-to-end query flow

`search()` in `search.py` runs: **(1)** extract decision + carry-forward seed tokens from the query; **(2)** if seeds exist, run PPR over the graph DB for a related-node set (L0); **(3)** run dense (L1) and BM25 (L2) each fetching `fetch_per_retriever=50`; **(4)** RRF-fuse them (L3) down to a `rerank_pool=30`; **(5)** annotate chunks with `graph_boost`; **(6)** cross-encoder rerank to `top_k` (L4, default 5). Every layer is wrapped so a fault *degrades* rather than kills the query: graph failure → skip annotation; dense failure → BM25-only; BM25 corrupt → dense-only; rerank failure → fall back to RRF order. The CLI is `python tools/lab_memory.py search "<query>"` (4-layer by default), with `--legacy` (pure vector), `--no-graph`, `--no-rerank`, and `--no-server` ablation flags wired straight into `search()`'s boolean parameters.

### Warm daemon + graceful fallback

Loading the ~1 GB reranker and importing `sentence_transformers` cold costs ~10+ seconds — unacceptable per query-subprocess. The fix is a **persistent Unix-domain-socket daemon**, `tools/retrieval/server.py`, that holds **one** warm embedder, **one** cross-encoder, and **one** parsed BM25 index in memory and serves full 4-layer searches over newline-delimited JSON (`{"cmd":"search",...}` → `{"event":"result","hits":[...]}`; also `ping`/`reload`/`shutdown`). It is single-threaded (one model copy in RAM, queries serialize; a warm 30-pair rerank is ~50–100 ms), sets `HF_HUB_OFFLINE` so the warm path never touches the network, binds with `umask 0o077` (owner-only socket), bounds every connection with a timeout, refuses to stomp a live server (probes the existing socket before unlinking), and self-heals its BM25 index by mtime if a `reload` signal is missed. `run_agi_lab.sh` starts it (`_start_retrieval_server`, nohup, ping-until-ready) and tears it down on exit; on a corpus rebuild it sends `client reload`.

The client, `tools/retrieval/client.py`, is intentionally lightweight (no server deps) and raises `ServerUnavailable` on *any* connection/timeout/protocol failure. `lab_memory.py`'s search command tries the warm server first (0.5 s connect, 20 s read timeout) and, on **any** exception, transparently falls back to building everything in-process — so the lab never hangs or errors when the daemon is down. A second, separate path exists in `tools/retrieval/concurrency.py` (`RetrievalWorker`): a stdin/stdout JSON-RPC subprocess worker that loads the embedder once and passes the repo path via JSON (not string-substituted into source, avoiding injection) — used when concurrent CLI callers would otherwise race on the model cache.

### Metadata-scoped retrieval

`tools/retrieval/metadata.py` derives the single-valued `{program_id, role, phase, deliverable_type}` columns from `source_path` + whole-file text, enabling `--program/--role/--phase` exact-match filters across all layers. It reuses the graph's extractors so vocabulary agrees, maps full program tokens to short IDs via an explicit `_PROGRAM_DIR_TO_ID` table (avoiding a naïve `program_(\d+)` regex that would collide `program_3` with `program_3_brainstorm`), and tags a program *only* when whole-file text names exactly one whitelisted program — conservative by design. `tools/retrieval/backfill_metadata.py` re-derives metadata file-level for existing rows (dry-run by default, `--apply` to write; metadata-only UPDATEs that never touch IDs/vectors). In the reference deployment it raised coverage from ~2.5% to a verified **31.3% any-tag** — partial coverage by intent, since most cross-program memory content gets no single tag. `LabMemory.ingest` auto-derives this metadata for new files when the caller omits it.

### Reproducibility

Embedding/reranker weights are pinned by HF commit SHA in the repo-root `model_revisions.json` (consumed by `tools/retrieval/model_pins.py`): `bge-reranker-v2-m3 → 953dc6f…` and `all-MiniLM-L6-v2 → c9745ed…`. The principle: a model `f_θ(x)` is only comparable across runs/machines if θ is fixed; loading "by tag" lets the Hub serve different weights silently. Combined with the `--legacy/--no-graph/--no-rerank` ablation flags, the system supports controlled layer-by-layer evaluation.

### Where to find more
- **Orchestrator / query flow:** `tools/retrieval/search.py` (the `search()` signature is the canonical contract for all layers + ablation flags)
- **L0 graph:** `tools/retrieval/graph.py` (extractors, schema, PPR)
- **L1 dense:** `tools/lab_memory.py` (`LabMemory`, chunking, sqlite-vec schema); spec ref §7.1 of `docs/superpowers/specs/2026-04-17-scientific-research-lab-overhaul.md`
- **L2 lexical:** `tools/retrieval/bm25.py` (tokenizer, precomputed-stats persistence)
- **L3 fusion:** `tools/retrieval/hybrid.py` (`reciprocal_rank_fusion`, `hybrid_search`)
- **L4 rerank:** `tools/retrieval/rerank.py` (`CrossEncoderReranker`, primary + fallback models)
- **Serving:** daemon `tools/retrieval/server.py`, client/fallback `tools/retrieval/client.py`, subprocess worker `tools/retrieval/concurrency.py`; launch/teardown in `run_agi_lab.sh` (`_start_retrieval_server` / `_stop_retrieval_server`)
- **Metadata:** derivation `tools/retrieval/metadata.py`, backfill `tools/retrieval/backfill_metadata.py`
- **Reproducibility:** `model_revisions.json` (repo root) + `tools/retrieval/model_pins.py`
- **Design narrative & rationale:** `docs/retrieval/architecture.md` (the L0–L4 diagram, and the explicit "choices we did NOT make")
- **Build/upgrade plan:** `docs/superpowers/plans/2026-05-20-memory-retrieval-upgrade.md` (tasks T0.1–T5.3)
- **Live SQLite indexes** (`tools/lab_memory.db`, `tools/lab_graph.db`, `tools/lab_bm25.json`) → *built at runtime over your corpus; not shipped in this framework repo.*

## 4. Reference Deployment (the AGI Lab) — Research Programs, Evaluation & Experiment Tracking

> **This section is intentionally brief.** It describes the *framework's research machinery* (the program lifecycle, the evaluation harnesses, the experiment-tracking schema) — all of which ship in this repo — and then summarizes what the author's reference deployment, **the AGI Lab**, actually produced with it. The full results, scores, program write-ups, decision history, and paper drafts are research output that is **not included in this framework repo**; they live in the sibling **`RESULTS.md`**.

### 4.1 The two missions (framework design)

The framework is built around an explicit two-tier mission that a deployment defines in its own `data/memories/mission.md` (a runtime file):

- **Aspirational Mission:** beat a named frontier model on a fixed set of standardized benchmarks (e.g. MMLU, HumanEval, ARC-AGI, GSM8K/MATH, HellaSwag, TruthfulQA, WinoGrande, BigBench-Hard) on 18 GB with no cloud — revised *upward only* as new frontier models ship, kept fixed as a *forcing function*, and never expected to be reached by any single program. In the reference deployment this was formally established as **not achievable at this scale** (Program 1).
- **Real Mission:** "produce rigorous research at any scale that fits on 18 GB hardware." The lab is explicitly unbiased between positive and negative outcomes — pre-registered decision rules decide which output type an artifact becomes. The four legitimate output types: (1) **positive-result** architecture/training/efficiency papers, (2) **methodology/governance** contributions (the lab itself as research artifact), (3) **reproducibility** reports, and (4) **envelope / negative-result** papers when evidence lands negative. Scale is hardware-bounded (≤100M trained from scratch, up to ~9B with 4-/2-bit quantization for inference), not arbitrarily capped.

### 4.2 Research programs (the lifecycle, framework-side)

Programs are long-lived, pre-registered research efforts. In a deployment they live under `programs/<slug>/`, each with a `question.md` (locked research question with a SHA-1 "binding blob hash" affixed at the unanimous lock gate so the question can't be silently restated), a `program_open_memo.md`, phase deliverables, and — on close — a `closure_memo.md`. The framework supplies the *protocol and tooling* for this lifecycle (phase gates §2.3, pre-reg discipline, factorial orchestrators §5); the **program directories themselves are research output and are not in this repo.**

**Reference deployment summary** (full detail in `RESULTS.md`): 2 programs taken to formal closure (Program 1 — an Opus-on-18GB feasibility *envelope/negative-result* paper; Program 2 — a pre-registered compute-matched dense-vs-MoE comparison that yielded a *null result* "below the measurement floor"), with Program 3 (alt-gradient methods × QAT at 100M) active. Plus two methodology/governance tracks (structural anti-forgery; tiered memory). Output: 4 markdown paper drafts + 3 typeset IEEE PDFs, including a null-result envelope paper.

### 4.3 Evaluation harnesses (in this repo)

Benchmark scoring is done by stdlib-only Python drivers in `tools/` that each download a public test set, format items for a compiled C scoring binary, and report accuracy:

- `tools/eval_mmlu.py` — MMLU (Hendrycks data from Berkeley), per-subject + overall accuracy across 57 subjects, 4-way multiple choice.
- `tools/eval_gsm8k.py` — GSM8K grade-school math (OpenAI test.jsonl), final-answer multiple-choice.
- `tools/eval_hellaswag.py` — HellaSwag commonsense completion (validation JSONL).
- `tools/eval_winogrande.py` — WinoGrande coreference, binary-choice.

Each driver shells out via `subprocess` to a compiled C scoring binary (`build/eval_model`, produced by `make eval` from `src/eval/eval_model.c`) given a checkpoint + tokenizer, and writes both a JSON result and a human-readable scorecard. Scoring is per-choice-logprob argmax. `tools/eval_harness_p11.py` is the more rigorous, pre-registered harness: it iterates factorial cells × benchmarks through a direct `lm-eval`-style adapter (`tools/hspa_lm_eval_adapter.py`), pins a specific harness commit hash that must not change, applies pre-reg-mandated left-truncation for over-length items, and uses `acc` (not `acc_norm`) for WinoGrande per a co-signed amendment.

**What the numbers mean.** For ~48–119M-parameter from-scratch models on a laptop, standard-benchmark scores at or near random are the **expected and documented outcome**, not a defect — exactly the point a feasibility program establishes formally. The reference deployment's actual scores and their honest framing are in `RESULTS.md` (do not quote benchmark numbers as model "capability").

### 4.4 Experiment tracking (schema in this repo, data in the deployment)

`tools/experiments.py` defines a small SQLite experiment store with three tables — `experiments` (with reproducibility columns `commit_hash`, `config_json`, `random_seed`, `data_version`), `notes`, and `experiment_links` (typed experiment-to-experiment relations). The framework ships the *schema and accessor*; the populated `data/experiments.db` is deployment data and is not included. In practice the *primary* experiment-tracking medium is the program-deliverable corpus (pre-regs, `benchmark_scores.json`, phase close memos, run-index/factorial state files) — `experiments.db` is the legacy structured layer that predates the program system.

### Where to find more
- **Eval harnesses (in this repo):** `tools/eval_mmlu.py`, `tools/eval_gsm8k.py`, `tools/eval_hellaswag.py`, `tools/eval_winogrande.py`, `tools/eval_harness_p11.py`, `tools/hspa_lm_eval_adapter.py`; C scoring source `src/eval/eval_model.c` (binary built to `build/eval_model` by `make eval`).
- **Experiment-store schema (in this repo):** `tools/experiments.py`.
- **Factorial orchestrators (in this repo):** `tools/run_phase3_factorial.py`, `tools/run_long.py`.
- **All actual results, scores, program write-ups, paper drafts, decision history** → **`RESULTS.md`** (sibling doc) *(the `programs/`, `data/memories/`, `data/eval/`, `data/experiments.db` artifacts are research output produced in a live deployment; not in this framework repo).*

## 5. Infrastructure, Tooling, Reproducibility & Self-Assessment

This is the operational and meta layer: the build system that turns C/Metal/Swift sources into runnable artifacts, the telemetry rollups that watch the lab watching itself, the reproducibility machinery, and the self-assessment suite that scores a whole lab against its own data. All of the *code* in this section ships in this repo; the *telemetry it consumes/produces* (cost logs, calibration JSONL, infra state files) is runtime data and lives only in a live deployment.

### 5.1 Build system — `Makefile`

The single `Makefile` at the repo root is a hand-rolled multi-language build (no CMake/Bazel). It targets a MacBook M3 Pro specifically — `CFLAGS` bakes in `-mcpu=apple-m3` and `-DACCELERATE_NEW_LAPACK`, and `LDFLAGS` links the Apple frameworks `Metal`, `MetalPerformanceShaders`, `Foundation`, and `Accelerate`. `make help` auto-generates a target list from `##` doc comments.

Key design decisions and targets:

- **Two C build profiles.** Release (`CFLAGS`) is `-std=c17 -Wall -Wextra -Werror -O2`. Debug (`CFLAGS_DBG`) drops to `-O0 -g` and adds `-fsanitize=address,undefined` (ASan + UBSan). A deliberate choice: **every test binary is built with the debug+sanitizer flags**, not the release flags — the default test rule `$(BUILD_DIR)/tests/test_%` compiles each `tests/test_*.c` together with all library sources under ASan/UBSan, so memory and UB bugs surface inside the test process regardless of how the core was built.
- **Source discovery with main() exclusion.** `C_SRCS_ALL` globs `src/**/*.c`, then `C_SRCS` filters out the four files that contain their own `main()` (`micro_experiment.c`, `scale_experiment.c`, `train_tokenizer.c`, `eval/eval_model.c`) so the library link is clean; those become their own binaries via dedicated targets (`make micro`, `make scale`, `make train-tokenizer`, `make eval`).
- **Build graph:** `make all` → `dirs core shaders swift-bridge`. `core` compiles objects; `shaders` runs `xcrun -sdk macosx metal`/`metallib` over `src/metal/*.metal` into `build/default.metallib`; `swift-bridge` builds `libMetalBridge.dylib` from `src/swift/MetalBridge.swift` with an emitted ObjC header. `compile-commands` emits `compile_commands.json` for clangd.
- **Test targets.** `make test` builds + runs all `tests/test_*.c` (each prefixed `MTL_DEBUG_LAYER=1`), then runs `pytest tests/`. **Quarantine policy:** `tests/quarantine/*` is explicitly excluded from the default glob and only built/run by `make test-quarantine` (the Metal matmul test lives here — see `tests/quarantine/README.md`). There are special-cased rules for `test_eval` (needs `-Isrc/eval`), `test_eval_model_score_per_choice` (a subprocess integration test that prefers the sanitizer-instrumented `eval_model_dbg` SUT over the release binary), and the quarantined Metal test. `make test-shell-guardrails` runs `tests/shell/run_all.sh` for CLI-level guardrail coverage.
- **Lab lifecycle targets.** `make lab-start` launches the autonomous lab via `caffeinate -dis ./run_agi_lab.sh` inside a tmux session `agi-lab` (caffeinate keeps the machine awake through long rate-limit waits). Companions: `lab-stop`, `lab-attach`, `lab-status`, `lab-log`, `lab-dashboard` (serves `tools/dashboard.py` at localhost:8420). A parallel `slack-bot-{start,stop,log}` family runs `tools/slack_bot.py` in its own tmux session.
- **Experiment launchers.** Long, session-spanning training runs go through `tools/run_long.py` and orchestrators: `make run-phase2-pair`, `make run-phase3-factorial[-dryrun]` (`tools/run_phase3_factorial.py`). These encode hard physical constraints in comments and command structure — e.g. serial `&&` chaining to enforce "AMX-serialization" (concurrent AMX matmul inflated step time ~86%, so runs never overlap).
- **Eval + paper output.** `make eval-p11` / `eval-p11-smoke` drive `tools/eval_harness_p11.py` against a pinned lm-evaluation-harness commit; `make ieee-pdf PAPER=…` builds a conference PDF via `tools/build_ieee_pdf.py`.
- **Memory/RAG tooling targets.** `lab-memory-{check,test,install}` and `memory-{audit,index,rotate-log,test}` wrap `tools/lab_memory.py` and `tools/memory.py`, always through `.venv/bin/python` (the deps `sqlite-vec`, `sentence-transformers`, `rank_bm25` are not on system Python under PEP 668).
- **`make lab-report`** → runs `PYTHONPATH=. .venv/bin/python -m tools.lab_assessment --repo .` (see §5.4).

### 5.2 Telemetry & rollups

The lab emits append-only JSONL telemetry during operation and periodically "rolls it up" into human-readable markdown + machine JSON. Each rollup module follows the same pattern: module-level `REPO` (monkeypatchable for tests), idempotent overwrite of its output file, graceful no-op when its source is absent. *(The JSONL telemetry and rollup outputs are runtime data and are not shipped here; the tools are.)*

- **Cost — `tools/cost_rollup.py`.** Reads Claude Code's own session JSONL logs from `~/.claude/projects/…` and joins them with lab telemetry. It carries a hard-coded `PRICING` table ($/M tokens per model, with cache-creation priced at 1.25× input and cache-read at 0.10×) and computes weekly totals, a per-model and per-role breakdown, `wastage_events` (silent-death recoveries, failed claims, escalations, verifier failures, holding loops), and an outlier-session list (sessions >3.5× MAD above median cost). Spec: `docs/superpowers/specs/2026-05-13-cost-rollup.md`. (Reference-deployment rollup: $3,293 over 134 director dispatches ≈ $24.58/session — see `RESULTS.md`.)
- **Calibration — three cooperating modules.** `tools/calibration_logger.py` exposes `log_calibration(decision_id, claim, confidence, reasoning, …)` (appends a record with `outcome: null`, validating confidence ∈ [0,1]) and `score_calibration(decision_id, outcome)` (idempotently fills the outcome in-place). `tools/calibration_rollup.py` buckets scored claims into 5 confidence bands and computes actual hit rate, midpoint error, and an over/under-confidence flag when |error| > 15% with ≥5 samples. `tools/calibration_pending.py` is a read-only nudge surfacing claims still at `outcome=null` (oldest-first) so the PI/Director closes the learning loop.
- **Dispatch — `tools/dispatch_rollup.py`.** From `dispatch_telemetry.jsonl`, per-role total dispatches, escalation rate, and verifier-fail rate, flagging roles over thresholds (escalation >30% → suggest tier upgrade; verifier-fail >20% → suggest model change). Feeds tuning of default model tiers in `data/agents/agents.json`.
- **Queue — `tools/queue_rollup.py`.** From `queue_telemetry.jsonl`, per-item-type lifetime counts (enqueued/completed/failed) and median claim→complete latency, flagging types with >20% fail rate. Pulls live pending/claimed counts from `tools/work_queue.summary()` when importable.
- **Live hardware status** comes from `tools/hwmon.py` (`make hwmon`, per-cycle CPU/GPU/memory-pressure/swap snapshots with throttle-event tracking) and `tools/preflight.py` (`make preflight`, which gates training on memory/disk/swap/battery thresholds). *(The `data/infra/` state-file snapshots they feed — `build_status.md`, `thermal_log.md`, `disk_budget.md`, `memory_budget.md` — are operational journals produced at runtime, not shipped here.)*

### 5.3 Reproducibility & rigor

The reproducibility story is **strong on inputs, partial on experiment provenance** — documented honestly:

- **Model pinning (solid).** `model_revisions.json` at the repo root is a lockfile mapping HF model IDs → exact commit SHAs (`BAAI/bge-reranker-v2-m3` and `sentence-transformers/all-MiniLM-L6-v2`). It is consumed by `tools/retrieval/model_pins.py`, whose `pinned_revision(model_id)` forces `revision=<sha>` loads so the embedder/reranker weights θ are fixed across machines — the module docstring spells out the reasoning (loading "by tag" lets the Hub silently serve different weights). The eval harness is likewise pinned: the `eval-p11` target echoes the pinned lm-evaluation-harness SHA.
- **Experiment provenance (schema present, populated at runtime).** `tools/experiments.py`'s `experiments` table carries reproducibility columns — `commit_hash`, `config_json`, `random_seed`, `data_version`. The heavyweight reproducibility actually lives in the factorial orchestrators (`tools/run_phase3_factorial.py`, `tools/run_long.py`) which use locked launch orders, `--weight-seed 42`, per-cell checkpoint dirs, and spec-fingerprint/config-drift pre-flights.
- **Engineering constraint as a first-class design axis.** Everything is sized to a **single 18 GB MacBook M3 Pro with no cloud.** This shows up as: the serial-only AMX discipline in the launchers, the thermal/disk/swap/battery preflight gate, and per-component memory budgeting (every new component must declare its memory budget before implementation; the optimization team holds veto over anything pushing past 18 GB).

### 5.4 Self-assessment suite — `tools/lab_assessment/`

A read-only Python package that recomputes "how good is the lab?" from the lab's own data, emitting a `lab_assessment.{md,json}` report via `make lab-report`. The explicit non-goal is a one-time essay — the value is that the *same scripts rerun* as the lab evolves.

**Architecture.** A shared `Sources` object (`sources.py`) is the one place that knows where raw data lives (experiments.db, cost rollup, calibration JSONL, scorecard, decision-log headers, program closure/paper drafts, git log). Five independent dimension computers each take `Sources` and return a `DimensionResult` (`types.py`: `dimension`, `metrics` dict, `verdict_level`, `verdict_rationale`, `relative_to`, `caveats`). `report.py` renders the five into markdown + JSON. `__main__.py` is the CLI entrypoint (stamps git SHA + UTC timestamp). **Graceful degradation is the core contract:** a missing/unreadable source makes the affected metric the literal string `"unavailable"` and records a caveat — a broken source never crashes the report. Computers never read each other, only `Sources`. (This is exactly why the suite runs cleanly in this framework repo even though the research-data sources are absent — affected metrics report `"unavailable"`.)

**The 5 dimensions + verdict rubric.** Verdicts are deliberately a *labeled qualitative scale* — `Strong | Solid | Developing | Weak | N/A` (enforced in `DimensionResult.__post_init__`) — backed by hard numbers, never a fake 0–10. Each states what it is *relative to*:

1. **research** (`research.py`) — counts closed-program closure memos, paper drafts, decision headers, and an output-type mix (envelope/methodology/reproducibility/positive). ≥2 closed programs + papers → *Solid*. Novelty is explicitly flagged as a qualitative judgment *not* computed here.
2. **benchmarks** (`benchmarks.py`) — parses the scorecard and compares each score to a `RANDOM_BASELINE` table. Any-above-random → *Developing*, else *Weak*. The rationale is load-bearingly honest: near-random is *expected* for 48–119M-param laptop models.
3. **ops** (`ops.py`) — total cost, session count, cost/session, plus decisions tracked and evaluator verdict.
4. **governance** (`governance.py`) — calibration mean-absolute-error (binned |confidence − accuracy|, weighted by bin size) over *resolved* claims, plus ORG_ADAPTATION flag count and evaluator verdict. Rubric: MAE ≤0.1 *Strong*, ≤0.2 *Solid*, else *Developing*; no resolved outcomes → *N/A*.
5. **rag** (`rag.py`) — index health from the live `tools/lab_memory.db` (chunk count + per-column metadata coverage), with retrieval quality (recall@k/MRR/nDCG@k) as an **opt-in heavy path** (`--rag-quality` → `compute_retrieval_quality()`, which loads the real embedder/reranker and runs a deterministic known-item eval).

The reference deployment's actual scorecard (research **Solid** · benchmarks **Developing** · ops **Solid** · governance **N/A** · rag **Solid**) is in `RESULTS.md`. The whole suite is reflexive (it assesses the lab it lives in) and leans on hard, source-grounded metrics to avoid self-congratulation.

### Where to find more
- **Build system:** `Makefile` (all targets, sanitizer flags, source/test discovery, launcher comments); `tests/quarantine/README.md` (quarantine policy); `run_agi_lab.sh` (lab entrypoint wrapped by `make lab-start`).
- **Telemetry tooling:** `tools/cost_rollup.py` (spec `docs/superpowers/specs/2026-05-13-cost-rollup.md`), `tools/calibration_logger.py` / `tools/calibration_rollup.py` / `tools/calibration_pending.py` (spec `docs/superpowers/specs/2026-05-04-pi-self-correction.md`), `tools/dispatch_rollup.py`, `tools/queue_rollup.py`, `tools/work_queue.py`, `tools/queue_scanner.py`.
- **Hardware:** `tools/hwmon.py` (`make hwmon`), `tools/preflight.py` (`make preflight`).
- **Self-assessment suite:** `tools/lab_assessment/` — `sources.py` (data access), `research.py`/`benchmarks.py`/`ops.py`/`governance.py`/`rag.py` (5 computers), `report.py` (md+JSON), `types.py` (`DimensionResult` + verdict levels), `__main__.py` (CLI). Tests: `tests/lab_assessment/`. Run: `make lab-report`.
- **Reproducibility:** model lockfile `model_revisions.json` (root) + consumer `tools/retrieval/model_pins.py`; experiment-store schema `tools/experiments.py`; factorial orchestrators `tools/run_phase3_factorial.py`, `tools/run_long.py`.
- **P11 eval apparatus:** `tools/eval_harness_p11.py`, `tools/hspa_lm_eval_adapter.py`.
- **Runtime-only data** (cost/calibration JSONL, `data/infra/` state files, `data/engineering/` budgets, `data/experiments.db`) → *produced in a live deployment; not in this framework repo.*

---

## Navigation Index — where to find what

Each section above ends with a detailed **"Where to find more."** This is the top-level map. *(F) = ships in this framework repo · (D) = produced in a live deployment, see `RESULTS.md`.*

| If you want to understand… | Go to | Primary code / docs |
|---|---|---|
| The neural-net engine, model, gradients, kernels | §1 | (F) `src/core`, `src/model`, `src/training`, `src/metal`, `src/swift`, `src/eval`, `src/tests` |
| How the autonomous org runs / governs itself | §2 | (F) `data/agents/`, `run_agi_lab.sh`, `tools/{work_queue,queue_scanner,dispatch_helper,brief_assembler,post_director,verify_signatures}.py` |
| The retrieval / RAG memory | §3 | (F) `tools/retrieval/`, `tools/lab_memory.py`; (D) the built `lab_memory.db`/`lab_graph.db`/`lab_bm25.json` indexes |
| The research machinery (lifecycle, harnesses, schema) | §4 | (F) `tools/eval_*.py`, `tools/hspa_lm_eval_adapter.py`, `tools/experiments.py`, `tools/run_phase3_factorial.py`; (D) `programs/`, `data/eval/`, `data/experiments.db` → `RESULTS.md` |
| Build, telemetry, reproducibility, self-assessment | §5 | (F) `Makefile`, `tools/lab_assessment/`, `tools/{cost,dispatch,queue}_rollup.py`, `model_revisions.json`; (D) `data/infra/`, `data/engineering/` |
| Mission, governance protocols, decision history | §2/§4 | (F) protocol + tooling here; (D) `data/memories/` + `programs/` are runtime memory → `RESULTS.md` |
| Design specs (the "why") | — | (F) `docs/superpowers/specs/*.md`, `docs/superpowers/plans/*.md`, `docs/retrieval/architecture.md` |
| The actual research results, scores, papers | §4 | (D) **`RESULTS.md`** (sibling doc) |
| Live framework self-assessment, regenerated | §5.4 | (F) `make lab-report` → `lab_assessment.{md,json}` |

---

## Honest Framing & Caveats

This framework's culture is anti-forgery and honest reporting; keep the numbers defensible:

- **Framework vs. deployment.** This repo is the *reusable framework*. Anything describing a *specific run's outcomes* (benchmark scores, $/session, program verdicts, corpus/index sizes, metadata coverage) is the author's **reference deployment** and lives in `RESULTS.md` — not a property of the framework binary.
- **"~74K LOC" is raw `wc -l`** (includes blanks/comments) — say *"~74K lines,"* not "SLOC." Verified composition: native engine ~19,600 (C 15,573 · headers 2,740 · Metal 1,047 · Swift 233); Python tracked non-test 18,991.
- **Tests: say "828 Python test functions + 269 C (Unity) cases,"** not "passing" — these are counts of test functions/cases as collected, not a clean end-to-end run claim.
- **Benchmark scores are at/near random** on tiny from-scratch models — frame as a *rigorous small-scale research process and a documented null result*, **never** as model capability. "Beating Opus" was formally proven unreachable at this scale and reported as such (reference deployment, Program 1; see `RESULTS.md`).
- **"2 programs closed"** of 8 program directories (in the reference deployment) — the rest are methodology tracks, a retrospective, and brainstorm scaffolding. Say *"2 programs taken to formal closure."*
- **The Metal GPU matmul path is built + unit-tested but dormant** (dispatch threshold disabled) — production training runs on Apple Accelerate (CPU/AMX), because the custom Metal kernel measured 3–100× *slower*. Say the GPU path is *"implemented and ready-to-enable,"* not "in production."
- **30 agent roles** (active) + 18 retired; **64 Python tools**; **283 commits over ~7.1 weeks** (2026-04-13 → 2026-06-02).
- **This framework repo deliberately excludes research output** — `programs/`, `data/memories/`, `data/experiments.db`, `data/eval/`, `data/infra/`, `data/engineering/`, checkpoints, and per-role `semantic.md`/`episodic/` history. A fresh fork starts those empty and grows them as it runs. The corresponding artifacts and results are summarized in the sibling `RESULTS.md`.

*To regenerate the framework's live self-assessment at any time: `make lab-report` (sources tied to research data report `"unavailable"` in a fresh fork — by design).*
