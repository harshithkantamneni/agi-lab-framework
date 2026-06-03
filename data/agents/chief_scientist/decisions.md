# chief_scientist -- Decision Ledger

*Every decision logged with: what, why, alternatives considered, reversibility.*
*Future sessions read this to understand past reasoning.*

---

## D001: Primary Architecture Selection -- HSPA
**Date:** 2026-04-13
**Decision:** Adopt Hierarchical Sparse Predictive Architecture (HSPA) as primary research direction, combining Mixture of Experts, Predictive Coding, and Hyperdimensional Computing.

**Why:**
- Dense transformers cannot achieve frontier quality in 18GB. This is a hard physical constraint.
- MoE is the only proven path to frontier scores (>90% MMLU) in constrained memory. GPT-OSS-20B achieves mid-80s MMLU with 3.6B active params on 16GB.
- Predictive coding offers local learning rules (no full backprop graph needed), which reduces memory pressure during training -- critical for 18GB.
- HDC offers ultra-compact input representations that could reduce embedding memory by 32x.
- M3 Pro unified memory eliminates the zero-copy overhead that makes MoE expensive on cloud GPUs.

**Alternatives considered:**
1. Small dense transformer (rejected: cannot fit frontier quality in 18GB)
2. Pure predictive coding network (rejected: unproven at LLM scale)
3. Pure HDC system (rejected: tops out at ~85% on complex tasks)
4. Standard MoE transformer with backprop (kept as fallback)

**Reversibility:** HIGH. This is a research direction, not a code commitment. If Cycle 1 hypotheses falsify the approach, we pivot to fallback (standard MoE transformer) with no sunk cost beyond research time.

---

## D002: Target Score Estimation Methodology
**Date:** 2026-04-13
**Decision:** Use web-sourced estimates for Opus 4.6 scores as initial targets while Literature team verifies exact numbers.

**Preliminary estimates:**
- MMLU: ~91.3%, HumanEval: ~92.8%, GSM8K: ~96.4%
- Others: estimated 85-95% range pending verification

**Why:** We need targets to set research priorities now. Waiting for perfect data would block all research teams. Literature team (H-L1) will refine these.

**Reversibility:** HIGH. Targets will be updated when Literature team reports.

---

## D003: Cycle 1 Hypothesis Assignments
**Date:** 2026-04-13
**Decision:** Assigned 15 hypotheses across 5 teams (3 per team).

**Assignment rationale:**
- Math team gets foundational theory work (iPC convergence, compression, loss function unification) because everything else depends on the mathematical framework being sound.
- Neuro team gets routing and architecture design because expert routing IS the core innovation -- if we just use standard top-k gating, we are just building another MoE transformer.
- Physics team gets thermodynamic analysis because understanding the energy landscape of sparse activation tells us whether HSPA can even converge.
- Novel Compute gets memory fitting and hardware optimization because if it does not physically fit in 18GB, nothing else matters.
- Literature team gets benchmark verification and survey work to ground all other teams.

**Priority ordering:**
- P0 (blocking): H-NC1 (memory fit), H-M1 (iPC convergence), H-L1 (benchmark scores), H-M3 (unified loss)
- P1 (important): H-P1 (thermodynamic cost), H-N1 (FEP routing), H-NC2 (HDC encoding), H-NC3 (switching latency)
- P2 (if capacity allows): H-N2 (Thousand Brains), H-P2 (power-law), others

**Reversibility:** HIGH. Hypotheses are designed to be falsifiable. Negative results redirect, they do not waste work.

---

## D004: Fallback Strategy Definition
**Date:** 2026-04-13
**Decision:** If H-M1 (iPC convergence), H-NC1 (memory fit), or H-NC3 (switching latency) are falsified, we pivot to standard MoE transformer with backprop.

**Why:** These three hypotheses test the three pillars of HSPA. If any pillar fails, the hybrid is unsound. The fallback (standard MoE) is well-understood and proven.

**Reversibility:** N/A (this is a contingency plan, not an action).

---

## D005: Unified Memory as Structural Advantage
**Date:** 2026-04-13
**Decision:** Treat M3 Pro unified memory as our primary competitive advantage over cloud-trained models, not a limitation.

**Why:**
- Cloud GPUs must copy expert parameters across PCIe (16-32 GB/s) for MoE routing.
- M3 Pro unified memory provides 150 GB/s bandwidth with zero-copy access from both CPU and GPU.
- This means we can keep ALL experts in memory simultaneously and route at the token level without the latency penalty that makes fine-grained MoE impractical on cloud hardware.
- Net effect: we can use more experts with finer granularity, potentially achieving better coverage of the knowledge space per parameter.

**Alternatives considered:**
1. Treat memory as pure constraint (rejected: misses the opportunity)
2. Use iCloud for expert overflow (kept as backup for training, not inference)

**Reversibility:** LOW impact if wrong -- even if the advantage is smaller than expected, unified memory never hurts.
