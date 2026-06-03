#!/usr/bin/env python3
"""Stream formatter for Claude Code stream-json output.
Reads JSON lines from stdin, prints human-readable real-time output.
Shows: assistant text, tool calls, tool results, agent launches, thinking."""

import json
import sys
import os

# Unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# Rate limit tracking — writes resetsAt (Unix timestamp) to disk for the runner
RATE_LIMIT_RESETS_FILE = "data/infra/rate_limit_resets_at"
_last_resets_at = None
_last_rate_limit_status = None

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "blue": "\033[34m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
    "magenta": "\033[35m",
    "red": "\033[31m",
    "gray": "\033[90m",
}

def c(color, text):
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def format_tool_call(tool_name, params):
    """Format a tool call for display."""
    if tool_name == "Agent":
        agent_type = params.get("subagent_type", params.get("type", "general"))
        desc = params.get("description", "")
        return f"{c('magenta', '>>> AGENT')} {c('bold', agent_type)} — {desc}"
    elif tool_name == "Bash":
        cmd = params.get("command", "")
        if len(cmd) > 120:
            cmd = cmd[:120] + "..."
        return f"{c('yellow', '$ ')}{cmd}"
    elif tool_name in ("Read", "Write", "Edit"):
        path = params.get("file_path", "")
        short = path.replace(str(__import__("pathlib").Path(__file__).resolve().parent.parent) + "/", "")
        return f"{c('cyan', tool_name)} {short}"
    elif tool_name == "Glob":
        return f"{c('cyan', 'Glob')} {params.get('pattern', '')}"
    elif tool_name == "Grep":
        return f"{c('cyan', 'Grep')} {params.get('pattern', '')} in {params.get('path', '.')}"
    elif tool_name == "Skill":
        return f"{c('green', 'Skill')} {params.get('skill', '')}"
    else:
        return f"{c('cyan', tool_name)}"

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    try:
        msg = json.loads(line)
    except json.JSONDecodeError:
        # Not JSON — pass through raw
        print(line, flush=True)
        continue

    msg_type = msg.get("type", "")

    # Assistant text (streaming)
    if msg_type == "assistant":
        content = msg.get("message", {}).get("content", "")
        if isinstance(content, str) and content:
            print(content, end="", flush=True)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        print(block.get("text", ""), end="", flush=True)
                    elif block.get("type") == "tool_use":
                        tool = block.get("name", "")
                        params = block.get("input", {})
                        print(f"\n{c('dim', '┌')} {format_tool_call(tool, params)}", flush=True)

    # Content block delta (partial text streaming)
    elif msg_type == "content_block_delta":
        delta = msg.get("delta", {})
        if delta.get("type") == "text_delta":
            print(delta.get("text", ""), end="", flush=True)
        elif delta.get("type") == "thinking_delta":
            # Show thinking in gray
            print(c("gray", delta.get("thinking", "")), end="", flush=True)

    # Tool use
    elif msg_type == "tool_use":
        tool = msg.get("name", "")
        params = msg.get("input", {})
        print(f"\n{c('dim', '┌')} {format_tool_call(tool, params)}", flush=True)

    # Tool result
    elif msg_type == "tool_result":
        content = msg.get("content", "")
        if isinstance(content, str):
            lines = content.split("\n")
            if len(lines) > 5:
                for l in lines[:3]:
                    print(f"{c('dim', '│')} {l}", flush=True)
                print(f"{c('dim', '│')} ... ({len(lines)-3} more lines)", flush=True)
            else:
                for l in lines:
                    print(f"{c('dim', '│')} {l}", flush=True)
        print(f"{c('dim', '└')} done", flush=True)

    # Result (final message in --print mode)
    elif msg_type == "result":
        content = msg.get("result", "")
        if content:
            print(f"\n{c('green', '=== Session Output ===')}", flush=True)
            print(content, flush=True)
        cost = msg.get("cost_usd", 0)
        duration = msg.get("duration_ms", 0)
        if cost or duration:
            print(f"{c('dim', f'Cost: ${cost:.4f} | Duration: {duration/1000:.1f}s')}", flush=True)

    # System messages
    elif msg_type == "system":
        text = msg.get("message", "")
        if text:
            print(f"{c('blue', '[system]')} {text}", flush=True)

    # Error
    elif msg_type == "error":
        error = msg.get("error", {})
        print(f"{c('red', '[ERROR]')} {error.get('message', str(error))}", flush=True)

    # Subagent messages
    elif msg_type == "agent":
        action = msg.get("action", "")
        agent_name = msg.get("agent", "")
        if action == "launch":
            print(f"\n{c('magenta', '>>> Launching agent:')} {c('bold', agent_name)}", flush=True)
        elif action == "complete":
            print(f"{c('magenta', '<<< Agent complete:')} {c('bold', agent_name)}", flush=True)

    # Rate limit events — persist resetsAt for the runner's wait_for_rate_limit_reset()
    elif msg_type == "rate_limit_event":
        info = msg.get("rate_limit_info", {}) or {}
        resets_at = info.get("resetsAt")
        status = info.get("status")
        # Only an actual rejection gates the runner. An "allowed_warning" means the
        # request went through — persisting its resetsAt makes the pre-spawn gate sleep
        # for hours despite never being blocked (spurious-gating root cause).
        if status == "rejected" and resets_at and resets_at != _last_resets_at:
            try:
                os.makedirs(os.path.dirname(RATE_LIMIT_RESETS_FILE), exist_ok=True)
                with open(RATE_LIMIT_RESETS_FILE, "w") as f:
                    f.write(str(int(resets_at)))
                _last_resets_at = resets_at
            except Exception as e:
                print(c("red", f"[rate_limit_event] failed to write reset file: {e}"))
        # Only surface a user-visible message on rejection transitions (not allowed_warning spam)
        if status == "rejected" and status != _last_rate_limit_status:
            when = ""
            try:
                import datetime
                when = datetime.datetime.fromtimestamp(int(resets_at)).strftime("%H:%M:%S")
            except Exception:
                pass
            print(c("red", f"[RATE LIMIT] rejected; resets at {when} (Unix: {resets_at})"))
        _last_rate_limit_status = status
