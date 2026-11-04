#!/usr/bin/env python3
"""tools/dispatch_ledger.py: a record of real subagent dispatches that agents do not author.

The runner pipes the harness's stream-json output through tools/stream_formatter.py. This
module sits in that pipe and writes one receipt per Agent/Task `tool_use` block, plus one
completion record per matching `tool_result`. Three properties follow:

  * Role identity comes from the structured `subagent_type` field of a real tool call.
    Assistant text travels in `text` blocks and is never parsed for dispatches, so a
    Director that prints ">>> AGENT pi" creates no receipt.
  * What a role attested comes from the `tool_result` the harness returned for that exact
    dispatch: a line `SIGNOFF target=<path> verdict=<APPROVE|MODIFY|REJECT>` in the role's
    own reply. See data/agents/_shared/signoff_contract.md.
  * Records are HMAC-chained with a key the recorder and the gate read and agents are
    denied, so a record cannot be minted, edited or reordered without that key.

The evidence directory lives outside the repository. config/claude_lab_settings.template.json
is the configuration that keeps agents away from it. What that does and does not close is
stated in audits/README.md under "Threat model".
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import hmac
import json
import os
import re
import secrets
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

EVIDENCE_ENV = "LAB_EVIDENCE_DIR"
RUN_ENV = "LAB_RUN_ID"
SANDBOX_ENV = "LAB_SANDBOX_STATE"

KEY_NAME = "receipt.key"
LEDGER_SUBDIR = "ledger"
ACCEPTED_NAME = "accepted.json"
VERDICT_SUBDIR = "verdicts"
KEY_BYTES = 32

SETTINGS_TEMPLATE = Path("config") / "claude_lab_settings.template.json"

# The Agent tool was named Task in older harness versions; session logs show both.
DISPATCH_TOOLS = frozenset({"Agent", "Task"})
GENERIC_ROLE = "general-purpose"
VERDICTS = ("APPROVE", "MODIFY", "REJECT")
ATTEST_LINE = re.compile(
    r"^SIGNOFF target=(\S+) verdict=(APPROVE|MODIFY|REJECT)[ \t]*$", re.MULTILINE
)
_RUN_ID = re.compile(r"^[A-Za-z0-9_.\-]+$")


class EvidenceError(Exception):
    """The evidence is missing, unreadable or fails authentication. Never means 'clean'."""


def evidence_dir(repo_root: Path | None = None) -> Path:
    """Evidence directory: $LAB_EVIDENCE_DIR, else one directory per lab tree under ~/.agi-lab."""
    override = os.environ.get(EVIDENCE_ENV)
    if override:
        return Path(override).expanduser()
    root = Path(repo_root or REPO_ROOT).resolve()
    tag = hashlib.sha256(str(root).encode()).hexdigest()[:10]
    return Path.home() / ".agi-lab" / f"{root.name}-{tag}"


def key_path(directory: Path) -> Path:
    return directory / KEY_NAME


def ledger_path(directory: Path) -> Path:
    return directory / LEDGER_SUBDIR


def init_key(directory: Path) -> None:
    """Create the key once. Refuses to overwrite: a new key would orphan every old receipt."""
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    fd = os.open(key_path(directory), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(secrets.token_bytes(KEY_BYTES))


def load_key(directory: Path) -> bytes:
    path = key_path(directory)
    try:
        mode = path.stat().st_mode
        key = path.read_bytes()
    except OSError as e:
        raise EvidenceError(f"cannot read receipt key at {path}: {e}") from e
    if stat.S_IMODE(mode) & 0o077:
        raise EvidenceError(f"receipt key {path} is readable beyond its owner (mode "
                            f"{stat.S_IMODE(mode):o}); treating it as exposed")
    if len(key) < KEY_BYTES:
        raise EvidenceError(f"receipt key {path} is too short")
    return key


def canonical(record: dict) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sign(key: bytes, record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "mac"}
    return hmac.new(key, canonical(body), hashlib.sha256).hexdigest()


def _genesis(run: str) -> str:
    return hashlib.sha256(f"genesis:{run}".encode()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def normalize_target(raw: str) -> str | None:
    """A repo-relative POSIX path, or None. Absolute and parent-escaping targets are refused."""
    target = raw.strip()
    while target.startswith("./"):
        target = target[2:]
    if not target or target.startswith("/") or ".." in target.split("/"):
        return None
    return target


class LedgerWriter:
    """Appends HMAC-chained records to one run's file. One writer per run, by construction."""

    def __init__(self, directory: Path, key: bytes, run: str):
        if not _RUN_ID.match(run):
            raise EvidenceError(f"invalid run id: {run!r}")
        ledger_path(directory).mkdir(parents=True, exist_ok=True)
        # O_EXCL: a file planted at this path, or a reused run id, stops the recorder. The
        # run then has no seal and the gate fails closed.
        self._fd = os.open(ledger_path(directory) / f"{run}.jsonl",
                           os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
        self._key = key
        self.run = run
        self.seq = 0
        self.prev = _genesis(run)

    def append(self, kind: str, **fields) -> dict:
        record = {"v": 1, "run": self.run, "seq": self.seq, "prev": self.prev,
                  "ts": time.time(), "kind": kind, **fields}
        record["mac"] = _sign(self._key, record)
        os.write(self._fd, canonical(record) + b"\n")
        os.fsync(self._fd)
        self.seq += 1
        self.prev = record["mac"]
        return record


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type") == "text"
                         and isinstance(b.get("text"), str))
    return ""


def _parse_attestations(text: str) -> list[list[str]]:
    found = []
    for raw_target, verdict in ATTEST_LINE.findall(text):
        target = normalize_target(raw_target)
        if target is not None and [target, verdict] not in found:
            found.append([target, verdict])
    return found


class Recorder:
    """Turns harness stream events into receipts. Never looks at assistant text blocks."""

    def __init__(self, writer: LedgerWriter, sandbox: str = "unknown", roster_sha256: str | None = None):
        self.writer = writer
        self.pending: set[str] = set()
        self.seen: set[str] = set()
        self.completed = 0
        writer.append("open", sandbox=sandbox, roster_sha256=roster_sha256)

    @classmethod
    def from_env(cls, repo_root: Path | None = None) -> "Recorder | None":
        """Build from the runner's environment. None when not running under the runner."""
        run = os.environ.get(RUN_ENV)
        if not run:
            return None
        directory = evidence_dir(repo_root)
        roster = Path(repo_root or REPO_ROOT) / "data" / "agents" / "agents.json"
        roster_sha = hashlib.sha256(roster.read_bytes()).hexdigest() if roster.is_file() else None
        return cls(LedgerWriter(directory, load_key(directory), run),
                   sandbox=os.environ.get(SANDBOX_ENV, "unknown"), roster_sha256=roster_sha)

    def observe(self, msg: dict) -> None:
        kind = msg.get("type")
        message = msg.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return
        if kind == "assistant":
            for block in content:
                if (isinstance(block, dict) and block.get("type") == "tool_use"
                        and block.get("name") in DISPATCH_TOOLS):
                    self._dispatch(block, msg)
        elif kind == "user":
            for block in content:
                if (isinstance(block, dict) and block.get("type") == "tool_result"
                        and block.get("tool_use_id") in self.pending):
                    self._complete(block)

    def _dispatch(self, block: dict, msg: dict) -> None:
        receipt = block.get("id")
        if not isinstance(receipt, str) or not receipt or receipt in self.seen:
            return
        params = block.get("input") if isinstance(block.get("input"), dict) else {}
        prompt = params.get("prompt") if isinstance(params.get("prompt"), str) else ""
        description = params.get("description") if isinstance(params.get("description"), str) else ""
        self.writer.append(
            "dispatch",
            receipt=receipt,
            role=str(params.get("subagent_type") or GENERIC_ROLE),
            parent=msg.get("parent_tool_use_id"),
            session=msg.get("session_id"),
            description=description[:200],
            prompt_sha256=_sha256_text(prompt),
            dictated=_parse_attestations(prompt),
            background=bool(params.get("run_in_background")),
        )
        self.seen.add(receipt)
        self.pending.add(receipt)

    def _complete(self, block: dict) -> None:
        receipt = block["tool_use_id"]
        text = _result_text(block.get("content"))
        self.writer.append(
            "complete",
            receipt=receipt,
            is_error=bool(block.get("is_error")),
            attests=_parse_attestations(text),
            result_sha256=_sha256_text(text),
        )
        self.pending.discard(receipt)
        self.completed += 1

    def seal(self) -> None:
        """Written at end of stream. A run with no seal means the recorder did not finish."""
        self.writer.append("seal", dispatches=len(self.seen), completed=self.completed,
                           unresolved=sorted(self.pending))


@dataclass
class Receipt:
    receipt: str
    run: str
    role: str
    parent: str | None
    dispatched_at: float
    description: str = ""
    completed_at: float | None = None
    is_error: bool = False
    attests: list = field(default_factory=list)
    dictated: list = field(default_factory=list)
    background: bool = False


@dataclass
class RunInfo:
    sealed: bool = False
    sandbox: str = "unknown"
    opened_at: float = 0.0


@dataclass
class Ledger:
    receipts: dict = field(default_factory=dict)
    runs: dict = field(default_factory=dict)
    authenticated: bool = True

    def latest_run(self) -> str | None:
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: self.runs[r].opened_at)


def _read_run(path: Path, key: bytes | None, ledger: Ledger) -> None:
    run = path.stem
    info = RunInfo()
    prev = _genesis(run)
    expected_seq = 0
    dispatches = 0
    sealed_at_seq = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as e:
        raise EvidenceError(f"cannot read ledger {path}: {e}") from e
    for lineno, raw in enumerate(lines, start=1):
        where = f"{path.name}:{lineno}"
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as e:
            raise EvidenceError(f"{where}: not JSON: {e}") from e
        if not isinstance(record, dict):
            raise EvidenceError(f"{where}: not a record")
        if key is not None and not hmac.compare_digest(str(record.get("mac", "")), _sign(key, record)):
            raise EvidenceError(f"{where}: HMAC does not verify")
        if record.get("run") != run:
            raise EvidenceError(f"{where}: record names run {record.get('run')!r}")
        if record.get("seq") != expected_seq or record.get("prev") != prev:
            raise EvidenceError(f"{where}: chain broken (a record was removed, reordered or inserted)")
        if sealed_at_seq is not None:
            raise EvidenceError(f"{where}: record after the seal")
        expected_seq += 1
        prev = str(record.get("mac", ""))

        kind = record.get("kind")
        ts = float(record.get("ts", 0.0))
        if kind == "open":
            info.sandbox = str(record.get("sandbox", "unknown"))
            info.opened_at = ts
        elif kind == "dispatch":
            receipt = record.get("receipt")
            if receipt in ledger.receipts:
                raise EvidenceError(f"{where}: receipt {receipt} appears twice")
            dispatches += 1
            ledger.receipts[receipt] = Receipt(
                receipt=receipt, run=run, role=str(record.get("role")),
                parent=record.get("parent"), dispatched_at=ts,
                description=str(record.get("description", "")),
                dictated=[list(x) for x in record.get("dictated", [])],
                background=bool(record.get("background")),
            )
        elif kind == "complete":
            found = ledger.receipts.get(record.get("receipt"))
            if found is None or found.run != run or found.completed_at is not None:
                raise EvidenceError(f"{where}: completion with no open dispatch")
            found.completed_at = ts
            found.is_error = bool(record.get("is_error"))
            found.attests = [list(x) for x in record.get("attests", [])]
        elif kind == "seal":
            if record.get("dispatches") != dispatches:
                raise EvidenceError(f"{where}: seal counts {record.get('dispatches')} dispatches, "
                                    f"file holds {dispatches}")
            info.sealed = True
            sealed_at_seq = record.get("seq")
        else:
            raise EvidenceError(f"{where}: unknown record kind {kind!r}")
    ledger.runs[run] = info


def read_ledger(directory: Path, key: bytes | None) -> Ledger:
    """Read and check every run. `key=None` checks the chain only and marks the result unauthenticated."""
    ledger = Ledger(authenticated=key is not None)
    root = ledger_path(directory)
    if not root.exists():
        return ledger
    if not root.is_dir():
        raise EvidenceError(f"{root} is not a directory")
    for path in sorted(root.glob("*.jsonl")):
        try:
            _read_run(path, key, ledger)
        except (TypeError, ValueError, AttributeError) as e:
            raise EvidenceError(f"{path.name}: malformed record: {e}") from e
    return ledger


def line_key(rel: str, line: str) -> str:
    return hashlib.sha256(f"{rel}\0{line.strip()}".encode("utf-8", "replace")).hexdigest()


def load_accepted(directory: Path, key: bytes | None) -> dict:
    """The set of lines and files a passing gate (or the operator's baseline) has accepted."""
    path = directory / ACCEPTED_NAME
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise EvidenceError(f"cannot read accepted baseline {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise EvidenceError(f"accepted baseline {path} is not JSON: {e}") from e
    if key is not None and not hmac.compare_digest(str(record.get("mac", "")), _sign(key, record)):
        raise EvidenceError(f"accepted baseline {path}: HMAC does not verify")
    return {"lines": set(record.get("lines", [])), "files": set(record.get("files", []))}


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_accepted(directory: Path, key: bytes, lines: set, files: set) -> None:
    record = {"v": 1, "ts": time.time(), "lines": sorted(lines), "files": sorted(files)}
    record["mac"] = _sign(key, record)
    _atomic_write(directory / ACCEPTED_NAME, canonical(record) + b"\n")


def write_verdict(directory: Path, key: bytes, run: str, **fields) -> None:
    """The gate's own record of a run. A run with a ledger and no verdict was never gated."""
    path = directory / VERDICT_SUBDIR
    path.mkdir(parents=True, exist_ok=True)
    record = {"v": 1, "run": run, "ts": time.time(), **fields}
    record["mac"] = _sign(key, record)
    _atomic_write(path / f"{run}.json", canonical(record) + b"\n")


def verdict_exists(directory: Path, run: str) -> bool:
    return (directory / VERDICT_SUBDIR / f"{run}.json").is_file()


def render_settings(repo_root: Path, directory: Path, sandbox: bool = True) -> dict:
    """Fill the settings template with absolute paths.

    The sandbox and permission docs resolve relative paths differently depending on which
    settings file holds them, so everything is rendered absolute. Glob entries are kept and
    also expanded to the files that exist now, so an existing file stays covered even where
    a platform's sandbox treats a pattern more narrowly than macOS does.
    """
    repo_root = Path(repo_root).resolve()
    directory = directory.expanduser()
    text = (repo_root / SETTINGS_TEMPLATE).read_text(encoding="utf-8")
    text = text.replace("{{REPO}}", str(repo_root)).replace("{{EVIDENCE}}", str(directory))
    settings = json.loads(text)
    settings.pop("_comment", None)
    if not sandbox:
        settings.pop("sandbox", None)
        return settings
    filesystem = settings["sandbox"]["filesystem"]
    expanded = []
    for entry in filesystem.get("denyWrite", []):
        expanded.append(entry)
        if "*" in entry:
            expanded.extend(p for p in sorted(glob.glob(entry)) if p not in expanded)
    filesystem["denyWrite"] = expanded
    return settings


def covered_by_deny_write(settings: dict, path: str) -> bool:
    """True if an absolute path falls under a denyWrite entry. Used by the wiring tests.

    A `*` stops at a path separator, so tools/*.py covers tools/x.py and not tools/lab/x.py.
    """
    parts = path.split("/")
    for entry in settings["sandbox"]["filesystem"]["denyWrite"]:
        if "*" not in entry:
            if path == entry or path.startswith(entry.rstrip("/") + "/"):
                return True
            continue
        pattern = entry.split("/")
        if len(pattern) == len(parts) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(parts, pattern)):
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render-settings", help="print the harness settings JSON for --settings")
    render.add_argument("--sandbox", choices=("on", "off"), default="on")
    sub.add_parser("evidence-dir", help="print the evidence directory for this tree")
    args = parser.parse_args()
    if args.command == "evidence-dir":
        print(evidence_dir())
        return 0
    print(json.dumps(render_settings(REPO_ROOT, evidence_dir(), sandbox=args.sandbox == "on")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
