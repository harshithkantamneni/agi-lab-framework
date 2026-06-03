# Bug report: claude-plugins-official/remember — post-tool-hook silently fails on large session dirs (ARG_MAX)

**Plugin:** `remember@claude-plugins-official` v0.5.0
**File:** `scripts/post-tool-hook.sh` line 45
**Severity:** Plugin silently stops working for heavy users
**Discovered:** 2026-05-04

## Symptom

After a Claude Code user accumulates ~10,000+ session transcript files in
`~/.claude/projects/<sanitized-project-path>/`, the remember plugin stops
producing daily memory files. `today-YYYY-MM-DD.md` is no longer written.
`now.md` goes empty. `last-save.json` references an old session ID.

The plugin's hook IS firing (visible as entries in
`<project>/.remember/logs/memory-YYYY-MM-DD.log`), but post-tool-hook
exits early at the `[ -n "$LATEST_JSONL" ] || exit 0` check.

## Root cause

```bash
# scripts/post-tool-hook.sh line 45
LATEST_JSONL=$(ls -t "$SESSION_DIR"/*.jsonl 2>/dev/null | head -1)
```

When `$SESSION_DIR` contains enough `.jsonl` files that the expanded glob
exceeds the kernel's `ARG_MAX` limit, the shell silently fails the glob
expansion. `ls` runs with no arguments and lists `cwd` instead of the
session dir. Since `cwd` doesn't have `.jsonl` files (they're filtered by
the glob, which itself failed), `head -1` returns empty.

Concrete numbers from the discovered case:
- macOS `getconf ARG_MAX` = 1048576 bytes (1 MB)
- 14,204 jsonl files × ~95 bytes per path = ~1.35 MB > ARG_MAX
- glob expansion silently fails
- `LATEST_JSONL` stays empty → hook exits at the empty-check
- save-session.sh never fires
- daily memory files never written

## Reproduction

```bash
# Create a session dir with 15k empty jsonl files
mkdir -p ~/test-session-dir
cd ~/test-session-dir
for i in $(seq 1 15000); do : > "$(uuidgen).jsonl"; done

# Try the plugin's pattern
LATEST=$(ls -t ~/test-session-dir/*.jsonl 2>/dev/null | head -1)
echo "expected: a uuid.jsonl path"
echo "actual:   '$LATEST'"
# On macOS: actual is empty
```

## Fix

Replace the glob-expanding pattern with one that doesn't trigger shell
expansion of all matching paths:

```bash
# Old (broken on large dirs):
LATEST_JSONL=$(ls -t "$SESSION_DIR"/*.jsonl 2>/dev/null | head -1)

# New (works at any scale):
LATEST_JSONL=$(cd "$SESSION_DIR" 2>/dev/null && ls -t 2>/dev/null | grep -m1 '\.jsonl$')
[ -n "$LATEST_JSONL" ] && LATEST_JSONL="$SESSION_DIR/$LATEST_JSONL"
```

`cd` + `ls -t` (no glob argument) lists by mtime without expanding any
patterns. `grep -m1 '\.jsonl$'` filters the first match. No ARG_MAX hit.

Alternative using `find` (also works, slightly more robust if non-jsonl
files happen to be newest):
```bash
LATEST_JSONL=$(find "$SESSION_DIR" -maxdepth 1 -name "*.jsonl" -type f \
    -exec stat -f "%m %N" {} \; 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
```

## Local workaround applied

The user's local plugin install was patched directly. This will be
reverted when the plugin updates; the upstream fix is needed.

In addition, the user moved 8721 jsonl files older than 3 days into a
`<session_dir>/archive/` subdirectory to bring the count below ARG_MAX
limits as a temporary mitigation.

## 2026-05-05 update: same bug in save-session.sh + agent- session ID rejection

A second instance of the same glob bug was found in
`scripts/save-session.sh` line 105 (the same `ls -t .../*.jsonl` pattern
used as the SESSION_ID fallback when none is passed in). Patched locally
with the same `cd + ls -t + grep` replacement.

A separate but adjacent bug: line 110's session-ID validation
`[[ "$SESSION_ID" =~ ^[a-f0-9-]+$ ]]` rejects subagent-prefixed IDs
(`agent-XXXX...` — Claude Code uses these for `Task` tool subagents).
Every PostToolUse hook fired by an agent subagent generates a noisy
`[save] ERROR: invalid session ID: agent-XXXXXX` line. The fix is to
skip subagent sessions cleanly (exit 0) rather than failing them —
their context is short-lived and not worth Haiku consolidation; the
parent session captures the relevant state.

Patched locally:

```bash
# After SESSION_ID is resolved, before the UUID regex check:
if [[ "$SESSION_ID" == agent-* ]]; then
    [ "${REMEMBER_DEBUG:-1}" = "1" ] && log "save" "skipping subagent session: $SESSION_ID"
    exit 0
fi
```

Both patches should be folded into the upstream PR alongside the
post-tool-hook fix.

## Suggested PR

Title: `fix(remember): post-tool-hook silently fails when session dir exceeds ARG_MAX`

Files: `scripts/post-tool-hook.sh`

Test: add a test that constructs a synthetic session dir with N=10000
empty jsonl files and verifies `LATEST_JSONL` resolves correctly.
