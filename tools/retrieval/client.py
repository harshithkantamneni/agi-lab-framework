"""Thin client for the persistent retrieval server (tools/retrieval/server.py).

`query_server` raises ServerUnavailable on ANY connection/timeout/protocol
failure; callers (tools/lab_memory.py) catch it and fall back to in-process
search() so the lab never hangs or errors when the server is down/slow.

Also exposes a small CLI used by run_agi_lab.sh:
    python -m tools.retrieval.client ping      # exit 0 iff a live server pongs
    python -m tools.retrieval.client reload     # tell the server to re-read indexes
    python -m tools.retrieval.client shutdown   # ask the server to exit
"""
from __future__ import annotations
import json
import os
import socket
import sys
from pathlib import Path


MAX_LINE = 1 << 20  # mirror server.MAX_LINE — bound the response read.


class ServerUnavailable(Exception):
    """Raised when the persistent server cannot satisfy a request (any reason)."""


def sock_path(repo: Path) -> Path:
    # Must agree with tools.retrieval.server.sock_path: honor RETRIEVAL_SOCK,
    # else repo/data/infra/retrieval.sock. (Kept local rather than importing the
    # server module so the lightweight client never pulls server deps.)
    override = os.environ.get("RETRIEVAL_SOCK")
    if override:
        return Path(override)
    return Path(repo) / "data/infra/retrieval.sock"


def _round_trip(repo: Path, req: dict, connect_timeout: float, read_timeout: float) -> dict:
    sp = sock_path(repo)
    if not sp.exists():
        raise ServerUnavailable(f"no socket at {sp}")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(connect_timeout)
    try:
        s.connect(str(sp))
    except (OSError, socket.timeout) as e:
        s.close()
        raise ServerUnavailable(f"connect failed: {e}") from e
    try:
        s.sendall((json.dumps(req) + "\n").encode())
        s.settimeout(read_timeout)
        line = s.makefile("rb").readline(MAX_LINE)
        if not line:
            raise ServerUnavailable("server closed connection without responding")
        return json.loads(line)
    except (OSError, socket.timeout, json.JSONDecodeError) as e:
        raise ServerUnavailable(f"request failed: {e}") from e
    finally:
        s.close()


def query_server(query, repo, top_k=5, program=None, role=None, phase=None,
                 use_graph=True, use_rerank=True,
                 connect_timeout=0.5, read_timeout=20.0):
    """Run one search through the warm server. Returns a list of hit dicts
    (same shape as tools.retrieval.search.search). Raises ServerUnavailable."""
    req = {"cmd": "search", "query": query, "top_k": top_k,
           "program": program, "role": role, "phase": phase,
           "use_graph": use_graph, "use_rerank": use_rerank}
    msg = _round_trip(Path(repo), req, connect_timeout, read_timeout)
    if msg.get("event") == "error":
        raise ServerUnavailable(f"server error: {msg.get('msg')}")
    if msg.get("event") != "result":
        raise ServerUnavailable(f"unexpected response: {msg!r}")
    return msg.get("hits", [])


def ping(repo, connect_timeout=0.5, read_timeout=5.0) -> dict:
    msg = _round_trip(Path(repo), {"cmd": "ping"}, connect_timeout, read_timeout)
    if msg.get("event") != "pong":
        raise ServerUnavailable(f"no pong: {msg!r}")
    return msg


def control(repo, cmd, connect_timeout=0.5, read_timeout=30.0) -> dict:
    return _round_trip(Path(repo), {"cmd": cmd}, connect_timeout, read_timeout)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Retrieval server client/CLI.")
    ap.add_argument("cmd", choices=["ping", "reload", "shutdown"])
    ap.add_argument("--repo", default=".")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "ping":
            info = ping(args.repo)
            print(f"pong corpus_chunks={info.get('corpus_chunks')}")
        else:
            msg = control(args.repo, args.cmd)
            print(msg.get("event", msg))
        return 0
    except ServerUnavailable as e:
        sys.stderr.write(f"retrieval server unavailable: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
