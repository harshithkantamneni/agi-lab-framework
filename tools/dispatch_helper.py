"""Decision-table dispatcher: given (role, task), return (model, effort, reasoning).

Reads role's default_model from data/agents/agents.json. Applies task-keyword
overrides to upgrade or downgrade. Returns explicit dict so callers can log
the choice for telemetry.

Used by Director's procedural to pick the right Claude model for each
agent dispatch without overloading the Director's cognitive budget.

KEYWORD OVERRIDE RULES (in branch order — first match wins):
  1. Opus keywords (review, propose, verdict, approve, paper, close-out,
     sign-off, judge, critique, evaluate) → claude-opus-4-7, max effort.
     Reason: judgment-tier task signal regardless of role default.
  2. Haiku keywords (extract, archive, format, plot, fetch, index, list,
     tally, count, enumerate) → claude-haiku-4-5, low effort.
     Exception: opus-tier roles stay at opus but drop to low effort
     (preserves judgment depth on what looks mechanical).
  3. Sonnet keywords (implement, fix, wire, integrate, build, design,
     refactor, debug) → at least sonnet-4-6, high effort.
     Upgrades haiku-default roles; preserves opus-default roles.
  4. No match → role default from agents.json, high effort.

USAGE PATTERN (called from Director's procedural, post-Agent return):

    from tools.dispatch_helper import dispatch, log_outcome
    result = dispatch(role="findings_curator", task="archive last 30 D-N")
    # result = {"model": "claude-haiku-4-5", "effort": "low",
    #           "reasoning": "task is mechanical (downgraded from sonnet-4-6)"}
    # ... use result["model"] + result["effort"] in your Agent() dispatch ...
    log_outcome(role="findings_curator", model=result["model"],
                escalated=(agent_status == "BLOCKED"),
                verifier_pass=None, task_class="archive")

UNKNOWN ROLE FALLBACK: claude-sonnet-4-6 (safe middle ground).
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import TypedDict


_REPO = Path(__file__).resolve().parent.parent
_AGENTS_JSON = _REPO / "data/agents/agents.json"


class DispatchResult(TypedDict):
    model: str
    effort: str
    reasoning: str


# Keyword patterns that signal a specific tier of work
_OPUS_KEYWORDS = re.compile(
    r"\b(review|propose|verdict|approve|paper|close-?out|sign-?off|judge|critique|evaluate)\b",
    re.IGNORECASE,
)
_SONNET_KEYWORDS = re.compile(
    r"\b(implement|fix|wire|integrate|build|design|refactor|debug)\b",
    re.IGNORECASE,
)
_HAIKU_KEYWORDS = re.compile(
    r"\b(extract|archive|format|plot|fetch|index|list|tally|count|enumerate)\b",
    re.IGNORECASE,
)


def _load_role_default(role: str) -> str:
    """Read the default model for a role from agents.json. Falls back to
    sonnet-4-6 if the role is unknown or agents.json is missing."""
    if not _AGENTS_JSON.exists():
        return "claude-sonnet-4-6"
    try:
        agents = json.loads(_AGENTS_JSON.read_text())
    except json.JSONDecodeError:
        return "claude-sonnet-4-6"
    return agents.get(role, {}).get("model", "claude-sonnet-4-6")


def dispatch(role: str, task: str) -> DispatchResult:
    """Return the (model, effort, reasoning) for dispatching this task to this role.

    Algorithm:
    1. Load role's default model from agents.json.
    2. If task contains opus keywords (review/propose/verdict/etc): upgrade to opus
       (or keep, with reduced effort if the task is mechanical-but-judgment-tier).
    3. If task contains sonnet keywords (implement/fix/wire): keep at default tier
       or upgrade haiku → sonnet.
    4. If task contains haiku keywords (extract/archive/format): downgrade to haiku
       UNLESS role is opus-tier (then keep opus but reduce effort to "low").
    5. Otherwise: trust role default with default effort "high".
    """
    default = _load_role_default(role)

    if _OPUS_KEYWORDS.search(task):
        return DispatchResult(
            model="claude-opus-4-7",
            effort="max",
            reasoning=f"task signals judgment-tier (default was {default})",
        )

    if _HAIKU_KEYWORDS.search(task):
        if default == "claude-opus-4-7":
            return DispatchResult(
                model=default,
                effort="low",
                reasoning="task is mechanical but role is judgment-tier; effort low",
            )
        return DispatchResult(
            model="claude-haiku-4-5",
            effort="low",
            reasoning=f"task is mechanical (downgraded from {default})",
        )

    if _SONNET_KEYWORDS.search(task):
        if default == "claude-haiku-4-5":
            return DispatchResult(
                model="claude-sonnet-4-6",
                effort="high",
                reasoning=f"task signals integration (upgraded from {default})",
            )
        return DispatchResult(
            model=default,
            effort="high",
            reasoning=f"task signals integration; role default {default} retained",
        )

    return DispatchResult(
        model=default,
        effort="high",
        reasoning=f"no override signal; using role default {default}",
    )


def log_outcome(
    role: str,
    model: str,
    escalated: bool,
    verifier_pass: bool | None,
    task_class: str = "",
) -> None:
    """Append a single dispatch outcome record to data/infra/dispatch_telemetry.jsonl.

    Called by Director (or whoever dispatches an agent) AFTER the agent returns.
    Records: role, model used, whether the agent self-escalated, whether the
    verifier passed (None if no verifier loop), and an optional task class.

    Used by tools/dispatch_rollup.py to produce per-role rate summaries that
    drive default-tier adjustments.
    """
    import datetime
    rec = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "role": role,
        "model_dispatched": model,
        "task_class": task_class,
        "escalated": escalated,
        "verifier_pass": verifier_pass,
    }
    log_path = _REPO / "data/infra/dispatch_telemetry.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(json.dumps(rec) + "\n")
