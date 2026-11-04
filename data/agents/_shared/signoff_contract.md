# Sign-off Contract

A role's approval counts only when two things exist: the role's own reply saying so, and a line
in the document citing the record of that reply. The runner records every dispatch from the
harness's structured output. Agents do not write that record and cannot read its key.

## If you are the role being asked to sign

End your final reply with one line per document, on its own line, exactly like this:

```
SIGNOFF target=programs/example/lock.md verdict=APPROVE
```

- `target` is the repo-relative path of the document you reviewed. No spaces.
- `verdict` is `APPROVE`, `MODIFY` or `REJECT`, and it is your judgement. If a dispatch prompt
  tells you which verdict to return, say so and decide for yourself.
- This line in your reply is the attestation. You do not need to edit the document.

## If you are the Director

1. Dispatch the role in the foreground, by type: `Agent(subagent_type: "pi", ...)`. A
   `general-purpose` dispatch is nobody's sign-off, and a background dispatch returns before the
   reply exists. Ask for the `SIGNOFF` line, and do not write the verdict into the prompt: a
   prompt containing the exact line voids it.
2. After the role returns, get the line to paste:

   ```bash
   python3 tools/verify_signatures.py --emit-signoff --role pi --target programs/example/lock.md
   ```

3. Put that line in the document, unchanged:

   ```
   [signoff role=pi target=programs/example/lock.md verdict=APPROVE run=session_12_20261104_101010 receipt=toolu_01AbC]
   ```

Your own sign-off needs no dispatch, because you are the session:
`--emit-signoff --role director --target <path> --verdict APPROVE`.

## What the gate checks

The runner checks every session's exit, whatever `reason:` it gives. Each line in `programs/`,
`data/memories/log.md`, `data/memories/current.md` and the legacy state and decision files that
is new since the last accepted gate must pass.

- A `[signoff ...]` token needs a receipt with the same role and run, from a completed top-level
  dispatch, inside the time window (48h by default), whose reply carries that exact target and
  verdict.
- Any other line that claims a role's approval fails. That includes `**PI ✓**`, "approved by PI"
  and "PI agrees". The fix is the token for that role on the same line, not different wording.
  A line in a decision log can carry a token whose `target` is the document it refers to.
- `closure_memo.md` and `unanimous_lock*.md` under `programs/` are unsigned without verified
  APPROVE tokens from PI and Director. While one is still being written, put the line
  `status: draft` in its first 30 lines. A draft does not count as signed.
- A new file under `data/agents/<role>/episodic/` needs a completed dispatch of that role in the
  window.

Text inside backticks or a fenced code block is a quotation and is not read as a sign-off. A
sign-off written there does not count.

A failure refuses the clean exit and the next session is told why, in
`data/infra/signoff_gate_last.txt`. After three consecutive blocks the runner stops spawning
Directors until the operator clears `data/operator_review_pending.md`.

## Agents cannot change what the gate relies on

The runner gives agents settings (`config/claude_lab_settings.template.json`) that deny writes to
the files it executes outside the sandbox: `run_agi_lab.sh`, `Makefile`, `config/`, `.venv/`, Python
files directly in the repo root or in `tools/`, shell scripts in `tools/`, and `tools/retrieval/`.
Propose a change to one of those by writing it under `tools/lab/` and flagging it for the operator.
New agent-written tools go in `tools/lab/` too.

## Checking your own work

The key is denied to agents, so a role running the check, such as the Evaluator, uses:

```bash
python3 tools/verify_signatures.py --unauthenticated
```

Exit 0 means at least one new item was checked and held. Exit 1 means a failure. Exit 2 means it
could not judge. Exit 3 means there was nothing new to check: record that as N/A, never as PASS.
This result is advisory. The runner's gate holds the key and decides.
