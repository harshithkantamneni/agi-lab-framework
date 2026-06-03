# Program Retrospective Template

Filled out by PI at every program close. Output filed at
`data/agents/pi/episodic/<YYYY-MM-DD>_<program_name>_retrospective.md`.
Transferable lessons summarized into `data/agents/pi/semantic.md`.

This is structured reflection, not a status report. Be specific, name
mistakes, name what worked. Generic answers ("it went fine") are
verifier-flagged as rote.

---

## Q1: Would I run this program again with hindsight?

(Answer: yes / no / yes-with-changes. Then 2-4 sentences explaining the
reasoning. If "yes-with-changes," name the specific changes.)

## Q2: What is the highest-leverage transferable lesson?

(Single most valuable thing the lab learned from this program. Specific
enough to act on. Examples of good lessons:
- "MoE collapses to uniform routing when LR > 1.5e-3 — pin lower."
- "Phase-close audits catch ~60% of the issues paper review catches; saves
   one round-trip."
- "Director sessions on quiet plateaus run 4× cheaper at sonnet — move it
   to default for monitor ticks."

Examples of weak lessons (verifier flags these):
- "Be more careful next time."
- "Communication was important.")

## Q3: What pattern transfers to the next program?

(Concrete operational change for the NEXT program. Procedural,
methodological, or organizational. 2-4 sentences.

If nothing transfers — say so explicitly. "No transferable pattern"
is a valid answer if the program was sui generis.)

---

## Calibration on retrospective itself

After answering Q1-Q3, attach a confidence:
- `confidence_q1: 0.XX` — how confident am I in my "would run again" answer
- `confidence_q2: 0.XX` — how confident is this lesson actually generalizable
- `confidence_q3: 0.XX` — how confident is this transfer

Calibration here matters because Q2's "highest-leverage lesson" claim is the
one that gets carried forward into pi/semantic.md and influences future
program selection. A 90%-confident lesson is treated as load-bearing; a
40%-confident one is a candidate for re-evaluation later.

---

## Verifier check (lab_architect, sonnet-tier)

The retrospective is verified before it's accepted into pi/semantic.md.
Lab_architect checks:

- Q1: is the reasoning specific to THIS program's evidence, not generic?
- Q2: is the lesson concrete enough to act on (not "be more careful")?
- Q3: is the transfer named precisely (or explicitly disclaimed)?
- Calibration: are confidence numbers attached and not trivially "100% on
  everything"?

If verifier returns VERIFY_FAIL, PI rewrites the retrospective with the
specific issues addressed (max 2 iterations).
