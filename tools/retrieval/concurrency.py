"""Subprocess-isolated retrieval worker.

Loads sentence-transformers model ONCE per worker subprocess, then serves
queries via stdin/stdout JSON-RPC. Cheap to spawn (~2s); cheap to query
(~50ms after first). Use when multiple in-process callers need to share
a model instance OR when concurrent CLI invocations would otherwise race
on the model cache.
"""
from __future__ import annotations
import json
import select
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Worker script — fully self-contained, no template substitution.
# The repo path arrives via stdin as the first JSON message (cmd="init"),
# BEFORE we emit the "ready" event. This eliminates the code-injection
# risk that string-substitution-into-Python-source has for paths
# containing quotes/backslashes/newlines.
_WORKER_SCRIPT = r"""
import json, sys
from pathlib import Path

# 1. First message on stdin must be {"cmd": "init", "repo": "<path>"}.
init_line = sys.stdin.readline()
if not init_line:
    print(json.dumps({"event": "error", "msg": "stdin closed before init"}), flush=True)
    sys.exit(1)
try:
    init = json.loads(init_line)
except json.JSONDecodeError as e:
    print(json.dumps({"event": "error", "msg": f"init not JSON: {e}"}), flush=True)
    sys.exit(1)
if init.get("cmd") != "init" or "repo" not in init:
    print(json.dumps({"event": "error", "msg": f"first message must be init: {init!r}"}), flush=True)
    sys.exit(1)

repo = Path(init["repo"]).resolve()
sys.path.insert(0, str(repo))
from tools.lab_memory import LabMemory

lm = LabMemory(str(repo / "tools/lab_memory.db"))
print(json.dumps({"event": "ready"}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        req = json.loads(line)
    except json.JSONDecodeError as e:
        print(json.dumps({"event": "error", "msg": str(e)}), flush=True)
        continue
    if req.get("cmd") == "shutdown":
        break
    if req.get("cmd") == "search":
        try:
            hits = lm.search(
                req["query"],
                program_id=req.get("program"),
                role=req.get("role"),
                phase=req.get("phase"),
                top_k=req.get("top_k", 10),
            )
            payload = [
                {"id": h.id, "program_id": h.program_id, "phase": h.phase,
                 "role": h.role, "source_path": h.source_path,
                 "chunk_text": h.chunk_text, "distance": h.distance}
                for h in hits
            ]
            print(json.dumps({"event": "result", "hits": payload}), flush=True)
        except Exception as e:
            print(json.dumps({"event": "error", "msg": str(e)}), flush=True)
"""


@dataclass
class RetrievalWorker:
    proc: subprocess.Popen
    repo: Path

    @classmethod
    def spawn(cls, repo: Path, spawn_timeout: float = 60.0,
              _python_override: str | None = None) -> "RetrievalWorker":
        # Use the running interpreter (portable across repos / no hardcoded venv path).
        venv_py = sys.executable
        if _python_override == "__sleep__":
            # test hook: a worker that never signals ready (exercises the timeout)
            cmd = [str(venv_py), "-c", "import time; time.sleep(3600)"]
        else:
            cmd = [str(venv_py), "-c", _WORKER_SCRIPT]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        def _die(reason: str):
            # Kill BEFORE reading stderr so a still-alive child can't block us.
            proc.kill()
            try:
                _out, err = proc.communicate(timeout=5)
            except Exception:
                err = ""
            raise RuntimeError(f"{reason}; stderr={err!r}")

        # Hand the repo path over via stdin as the first message. No string
        # substitution into Python source: a path containing quotes/backslashes/
        # newlines is safe because json handles escaping and the worker parses
        # it with json.loads().
        try:
            proc.stdin.write(json.dumps({"cmd": "init", "repo": str(repo)}) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, ValueError) as e:
            _die(f"worker init write failed: {e}")

        # Bound the ready handshake: a wedged or slow-loading worker (e.g. a cold
        # torch import stalling on iCloud materialization) must not hang spawn().
        ready, _, _ = select.select([proc.stdout], [], [], spawn_timeout)
        if not ready:
            _die(f"worker did not signal ready within {spawn_timeout}s")
        line = proc.stdout.readline()
        if not line:
            _die("worker closed stdout before signaling ready")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _die(f"worker startup not JSON: line={line!r} ({e})")
        if msg.get("event") != "ready":
            _die(f"worker did not signal ready: {msg!r}")
        return cls(proc=proc, repo=repo)

    def query(self, q: str, top_k: int = 10, timeout: float | None = None, **filters: Any) -> list[dict]:
        """Run one search through the worker.

        Args:
            q: query string
            top_k: max hits
            timeout: if not None, max seconds to wait for the worker to respond.
                Raises TimeoutError on expiry. Trade-off: a hung worker is
                detected promptly, but the worker subprocess is then in an
                undefined state (it may still emit the late response on the
                next call). Caller should typically discard the worker and
                spawn a fresh one if a timeout fires.
            **filters: forwarded to LabMemory.search (program, role, phase).
        """
        req = {"cmd": "search", "query": q, "top_k": top_k, **filters}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()

        if timeout is not None:
            ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
            if not ready:
                raise TimeoutError(
                    f"worker did not respond within {timeout}s (query={q!r})"
                )
        line = self.proc.stdout.readline()
        msg = json.loads(line)
        if msg.get("event") == "error":
            raise RuntimeError(f"worker error: {msg.get('msg')}")
        return msg.get("hits", [])

    def shutdown(self) -> None:
        try:
            self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Child hung — force-terminate and reap unconditionally so the
            # process descriptor doesn't leak.
            self.proc.kill()
            self.proc.wait()
