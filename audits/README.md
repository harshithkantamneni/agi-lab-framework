# Audits of this framework's own verification layer

These scripts audit the anti-forgery remedy that ships in this repository. They were
written in September 2026 against the operational archive of the lab that ran it —
289 commits, 763 runner-written session logs, 195 dispatch receipts, seven weeks — and
they found that all three components of the remedy were defective for that entire period.

No model calls. Every count is reproducible from a lab tree plus git history.

## What they found

| component | defect |
|---|---|
| **Detector** (`tools/verify_signatures.py`) | Its original four patterns did not match the signature format of the incident that prompted it (`**PI signature:** ✓`). Run over the whole archive it reported 23 forgeries, **all 23 false positives** in three mechanical classes: citation parsing, prose quoting the convention, and a 48h window anchored to wall-clock rather than to the signature's own date. |
| **Gate** (`run_agi_lab.sh`) | **Never executed.** `if ! source .venv/bin/activate && python3 tools/verify_signatures.py` parses as `(! source) && python3`, so when the virtualenv activated normally the `&&` short-circuited and the detector was never invoked. Precondition satisfied **270** times across 763 sessions; detector output in **0** logs; blocked **0** times. Fixed in `dfedb1c`. |
| **Audit** (evaluator checklist item 11) | Fourteen evaluator reports record `PASS (AUTO)`. Of the ten that quote the detector, **six** quote `OK: 0 signatures verified` — a pass over an empty scan, indistinguishable in the record from a pass over checked attestations. |
| **The remediation itself** | The session that closed the incident announced "the first actual pi-agent dispatch in lab history" while dispatching `subagent_type=general-purpose` with the role named only in the description. The first typed `pi` dispatch came seven days later. **32 of 174** non-director receipts share that defect; one role was never dispatched by type at all. |

Underneath all four: of **195** episodic receipts, **0** are tracked in git — not gitignored,
simply never added. Of 99 program markdown files, 18 are tracked. `data/infra/session_logs/`
is gitignored. The load-bearing audit substrate was mutable, unversioned files inside the
Director's own write scope.

## The scripts

```
python3 audits/gate_audit.py              # did the gate ever run? 270 / 0 / 0
python3 audits/gate_audit.py --prove      # reproduces the shell precedence parse
python3 audits/role_identity_audit.py     # receipts vs actual subagent_type dispatches
python3 tools/verify_signatures.py --check-identity --verbose   # the same test, as a gate
```

Run them from a live lab tree. `gate_audit.py` and `role_identity_audit.py` expect the
lab at a path you set at the top of each file; `--check-identity` resolves from the repo
root. All three need `data/infra/session_logs/`, which `.gitignore` excludes — a fresh
clone has none, and `--check-identity` exits 2 ("cannot judge") rather than 0 in that case.

## Fixes shipped here

- **#38** — the signature format the incident actually used.
- **#39** — the three false-positive classes, plus `--as-of` for retrospective audits.
- **#40** — `--check-identity`: dispatch *identity*, not just dispatch existence, with 12
  regression tests including that absent logs exit "cannot judge" rather than "verified".
- `dfedb1c` — the gate's shell precedence, found and fixed by the operator independently
  of this audit and four days after the archive analysed here ends.

## The generalisable point

Each defect has one shape: a check bound to a proxy that the claim could satisfy without
possessing the property it named, over evidence the attester could author. The standard
conditions for trustworthy verification — external, mechanical and binding, pre-committed,
cost-inverting — are all properties of the *checker*, and this remedy arguably satisfied
all four while passing everything it was built to catch. What it lacked is a property of
the *evidence*: the record the check reads must be append-only and outside the attester's
write scope, and the check must bind to the property claimed rather than to a proxy.

Every one of the four was found the same way — by comparing a claim against a record
written by a different process than the one making the claim. In this lab the only such
records were git history and a shell script's logs.

Write-up: *Verifying the Verifier: Four Independent Defects in a Self-Graded Anti-Forgery
Remedy* (2026).
