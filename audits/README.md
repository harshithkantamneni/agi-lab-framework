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
python3 audits/gate_audit.py --csv OUT    # one row per session log, with no log text
python3 audits/role_identity_audit.py     # receipts vs actual subagent_type dispatches
python3 tools/verify_signatures.py --check-identity --legacy-logs --verbose   # the same test, as a command
```

Run them from a live lab tree. `gate_audit.py` and `role_identity_audit.py` expect the
lab at a path you set at the top of each file; `--check-identity` resolves from the repo
root. All three need `data/infra/session_logs/`, which `.gitignore` excludes — a fresh
clone has none, and `--check-identity` exits 2 ("cannot judge") rather than 0 in that case.

`--legacy-logs` and `--legacy-episodic` keep the original audit reproducible against the
archive. They read evidence agents can write, so the runner's gate never uses them.

## Per-session results

[`gate_audit_sessions.csv`](gate_audit_sessions.csv) is the `--csv` output for the archive
analysed here: 763 session logs, 270 reaching the gate's precondition, 0 containing the
detector's output, 0 blocks, and 6 where the detector was run by hand. The logs stay private.
Each row carries the log's size and SHA-256, so the table can be checked against the logs
if they're ever shared.

## Fixes shipped here

- **#38** — the signature format the incident actually used.
- **#39** — the three false-positive classes, plus `--as-of` for retrospective audits.
- **#40** — `--check-identity`: dispatch *identity*, not just dispatch existence, with 12
  regression tests including that absent logs exit "cannot judge" rather than "verified".
- `d5af181` — the gate's shell precedence, found and fixed by the operator independently
  of this audit and four days after the archive analysed here ends.
- `close-signoff-gaps` (November 2026): receipt-bound sign-offs and a fail-closed gate. See below.

## Gaps closed in November 2026

A code review in November 2026 listed gaps in these fixes, and the audit's root cause added one
more. Each has tests in `tests/test_signoff_gaps.py` that fail on the code before this change and
pass after it. The ledger and the sign-off format are in `tools/dispatch_ledger.py` and
`data/agents/_shared/signoff_contract.md`.

| gap | how it closed |
|---|---|
| The gate did not pass `--check-identity`, and passing it would have replaced the signature check. | The gate runs `--gate --check-identity`, which does both. Identity now reads the ledger. |
| Text in a director session could fake a `>>> AGENT pi` line. | Identity reads receipts built from structured `tool_use` blocks, never log lines. Assistant text, tool output and stderr create no receipt. |
| An empty scan exited 0. | It exits 3 and prints `EMPTY`, never `OK`. A scan that could not cover its corpus exits 2. The gate lets exit 3 through, recorded as `result=empty`, because a session with nothing to sign has to be able to end. The old default scope also read `data/state.md`, which the memory refactor archived, so it no longer reached `programs/`. The scope is now fixed. |
| A sign-off phrased outside the pattern list passed. | One token format is recognized. A malformed token fails. A wide net fails other lines that claim a role's approval. Closure memos and unanimous locks are unsigned without verified tokens, however their prose is worded. |
| A sign-off could cite any old receipt, and any recent file in the role's folder passed. | Episodic files are no longer evidence. A token needs a receipt with the same role and run, from a completed top-level dispatch, inside the window, whose own reply gave that verdict on that target. Receipts are HMAC-chained. |
| Nothing noticed if the gate failed to run. | The gate fails closed: only a detector that finishes and echoes a per-call nonce lets an exit through. It runs on every exit, not only when the Director claims a clean one. It no longer depends on `.venv`, the exec bit or the exit reason. Changed pinned files block. A block reaches the next Director, and three in a row halt for the operator. A run with a ledger and no verdict is reported at the next gate. |

To move a lab that already has sign-offs onto this, stop the runner and run
`make lab-evidence-init` once. That creates the key outside the repository, and accepts every
existing sign-off line and episodic file as a legacy baseline without verifying them. The runner
refuses to start on such a tree until this has been done.

## Threat model

The attester is the Director, or any role, running with shell access under `--dangerously-skip-permissions`.

**Held by the code and the tests, against an agent that does not have the key.** An agent can
print anything, write any document, and write a file shaped like a ledger. None of that produces
a passing sign-off: text creates no receipt, and a ledger signed without the key makes the check
exit 2. Editing or reordering a record, removing one from the middle, or adding one after the seal
is detected. Cutting records off the end, or deleting a run, is not reported as tampering, but it
only loses evidence: a sign-off citing a lost record fails, and the current run needs its seal.
The tests use a stubbed harness; see "Not verified" below.

**Conditional on the shipped configuration.** An agent running as the same OS user can read a key
file and rewrite any file that user owns, unless something stops it. Permission deny rules cover
the built-in file tools and commands the harness recognizes, not a Python script that opens a file
itself. The configuration that closes this is the OS sandbox. The repo ships it in
`config/claude_lab_settings.template.json`, and the runner passes it with `--settings` to the
Director and the consolidator:

- `sandbox.enabled`, `failIfUnavailable` and `allowUnsandboxedCommands: false`, so a command cannot
  fall back to running outside the sandbox.
- `filesystem.denyRead` on the key, and `denyWrite` on the evidence directory.
- `denyWrite` on everything the runner executes outside the sandbox: `run_agi_lab.sh`, `Makefile`,
  `config/`, `.venv/`, Python files directly in the repo root or `tools/`, shell scripts in `tools/`,
  `tools/__pycache__/` and `tools/retrieval/`. Without this, an agent could edit
  `tools/post_director.py` and have the runner read the key for it. A test fails if a new runner
  line executes something not on this list. The cost is that agents can no longer edit those files;
  new agent-written tools go in `tools/lab/`.
- The gate and recorder run with `python3 -I`, so a module planted next to them cannot shadow the
  standard library.

`LAB_SANDBOX=off` is an explicit opt-out for work the sandbox breaks. With it, this paragraph does
not hold: a shell command can read the key. Each run's ledger and each gate verdict record the
setting. A separate OS user or a VM for the agents would be a stronger boundary than a sandbox
under the same user. The repo cannot set that up on its own.

**Not closed.**

- A receipt shows that the named role ran and that its reply carried the line. It does not show
  the review was any good. A Director can steer a role through its prompt. Only an exact verdict in
  the prompt is rejected, and a paraphrase is not.
- A Director can edit a role's `procedural.md` or `agents.json` between sessions to weaken that
  role. Each run records the roster's digest, so a later audit can see the change. Nothing blocks it.
- A sign-off binds to a document's path, not its content. A document can change after it is signed.
- The net for free-text approvals is a heuristic, and prose is an open set. Wording it misses, or a
  sign-off written inside a code span or fence, passes outside the required-document list. That list
  covers closure memos and unanimous locks.
- `tools/post_director.py` runs before the gate, so a block does not undo that session's log,
  current or queue changes. The block persists and the next Director is told, but the changes stand.
- The identity check is per role and per window, not per file. A Director can add a role-filed
  episodic record during a window in which that role really ran. Sign-offs no longer rely on
  those records.
- A process that can write to the harness's stdout pipe could inject stream events. The tests show
  stderr is out of this path. A sandboxed command should not be able to open the pipe, but no test
  shows that.
- `--init-evidence` and `--rebaseline` accept existing lines without verifying them, as they say
  when run. Everything signed before cut-over stays exactly as trustworthy as it was.
- A file an agent writes and the operator later runs, such as a test or a script in `tools/lab/`,
  runs with the operator's rights.

**Not verified.** No live lab run was made for this change. The stream shape in the tests comes from
the harness documentation and a local transcript, not a live capture. That deny rules hold under
`--dangerously-skip-permissions`, and how sandbox rules resolve, comes from the harness
documentation. The sandbox itself was not exercised, and it needs a real macOS session to confirm:
that the key is unreadable from Bash, that Metal training started from Bash still works under
Seatbelt, that glob `denyWrite` entries behave as written, and that retrieval falls back cleanly if
its socket is blocked. If Metal jobs fail under the sandbox, that needs a decision. Adding them to
`excludedCommands` runs them outside the sandbox, and a script an agent can edit would then be a way
out.

## Known gaps

The items under "Not closed" above remain open. The first three matter most: a receipt is evidence
that a role ran and what it replied, not that it reviewed well, and neither the role's prompt nor the
signed document's content is bound to the sign-off.

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
