# Self-Escalation Contract

Every agent in the lab commits to this contract.

## Tier definitions

**Tier A (claude-opus-4-7)** — judgment, taste, nuance:
- Scientific verdicts, paper-quality writing, code review for subtle issues,
  program-level decisions, scientific reasoning.
- Output consumed by humans / other Tier-A roles for high-stakes decisions.

**Tier B (claude-sonnet-4-6)** — substantive engineering / synthesis:
- Implementation, integration, literature search + relevance ranking,
  structured but non-trivial extraction.
- Output consumed by Tier-A or Tier-B roles. Mistakes recoverable via review.

**Tier C (claude-haiku-4-5)** — mechanical execution:
- File ops, archiving, formatting, mechanical plotting, schema-validated
  extraction.
- Output consumed by automated systems or other agents that re-process it.
  Mistakes caught by structural checks (file exists, schema valid, etc.).

## The contract

On first reading your assigned task, assess: does it fit your model tier?

If the task exceeds your tier — requires nuance, judgment, scientific
reasoning, or interpretive synthesis beyond mechanical execution at your
tier's level — **return immediately with**:

- `status: BLOCKED`
- `key_finding: "task exceeds tier <YOUR_TIER>"`
- `suggest_model: "<higher tier model id>"`
  (claude-sonnet-4-6 if you're at haiku; claude-opus-4-7 if you're at sonnet)
- `summary: <1-line description of the specific judgment / nuance the task requires>`

**DO NOT produce shallow output.** False BLOCKED → Director re-dispatches at the
suggested tier (~5K wasted tokens, recoverable). Confident shallow output →
downstream consumers act on incorrect content (NOT recoverable).

## Tier C special clause

If you are running on claude-haiku-4-5: when in doubt, escalate. The cost of
false escalation is much lower than the cost of a confident shallow answer.
Bias toward BLOCKED on ambiguous tasks.

## Why this exists

Documented research (OpenAI 2025 hallucination paper, CMU overconfidence study,
MAST taxonomy of multi-agent failure modes) shows that smaller language models
are systematically miscalibrated about their own capability. Trusted
self-assessment is fragile. This contract makes self-assessment a *structured
output* (BLOCKED / suggest_model) that the dispatcher can act on, instead of
relying on the model to silently know its limits.
