"""Persistent retrieval daemon (Unix-domain socket).

Holds ONE warm embedder (LabMemory) + ONE cross-encoder reranker + ONE parsed
BM25 index in memory and serves full 4-layer searches over a UDS socket using
newline-delimited JSON. Started once by run_agi_lab.sh; lab_memory.py's `search`
subcommand is a thin client (tools/retrieval/client.py) that falls back to
in-process search() on any failure. This amortizes the ~1 GB reranker and the
slow first-import of sentence_transformers across the whole lab run instead of
paying them per query subprocess.

Protocol (newline-delimited UTF-8 JSON, one request line -> one response line):
  {"cmd":"search","query":str,"top_k":int,"program":str|null,"role":str|null,
   "phase":str|null,"use_graph":bool,"use_rerank":bool,"req_id":str}
      -> {"event":"result","req_id":str,"hits":[...]}
       | {"event":"error","req_id":str,"msg":str}
  {"cmd":"ping"}     -> {"event":"pong","corpus_chunks":N}
  {"cmd":"reload"}   -> {"event":"reloaded"}
  {"cmd":"shutdown"} -> {"event":"bye"} then the server exits

No pickle anywhere. Single-threaded accept loop (one model copy in RAM; queries
serialize — warm rerank of ~30 pairs is ~50-100 ms so this is not a bottleneck).
"""
from __future__ import annotations
import json
import os
import signal
import socket
import sys
import threading
from pathlib import Path

# Network-free model loading on the warm path. Set before torch/transformers
# import so HF Hub is never contacted when weights are already cached.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

MAX_LINE = 1 << 20  # 1 MB request line cap — bound server-side buffering.


def sock_path(repo: Path) -> Path:
    # AF_UNIX paths have a ~104-char limit (macOS). Default lives under the repo
    # (well within the limit for normal checkouts); RETRIEVAL_SOCK overrides it
    # for deep checkout paths or tests.
    override = os.environ.get("RETRIEVAL_SOCK")
    if override:
        return Path(override)
    return Path(repo) / "data/infra/retrieval.sock"


def pid_path(repo: Path) -> Path:
    return Path(repo) / "data/infra/retrieval.pid"


class RetrievalServer:
    def __init__(self, repo: Path):
        self.repo = Path(repo).resolve()
        self.bm25_path = self.repo / "tools/lab_bm25.json"
        self.lab_memory_db = self.repo / "tools/lab_memory.db"
        self._lm = None
        self._reranker = None
        self._bm25 = None
        self._bm25_mtime = None

    # ---- singletons --------------------------------------------------------
    def _ensure_models(self) -> None:
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        from tools.lab_memory import LabMemory
        from tools.retrieval.rerank import CrossEncoderReranker
        if self._lm is None:
            self._lm = LabMemory(str(self.lab_memory_db))
            try:
                self._lm._embed(["warmup"])  # force the embedding model to load now
            except Exception as e:  # warm-up is best-effort; lazy load still works
                sys.stderr.write(f"server: embedder warmup skipped ({e}).\n")
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
            try:
                self._reranker._get_model()  # force the cross-encoder to load now
            except Exception as e:
                sys.stderr.write(f"server: reranker warmup skipped ({e}).\n")

    def _load_bm25(self) -> None:
        from tools.retrieval.bm25 import BM25Index
        if not self.bm25_path.exists():
            return
        try:
            self._bm25 = BM25Index.load(self.bm25_path)
            self._bm25_mtime = self.bm25_path.stat().st_mtime
        except Exception as e:  # truncated/corrupt: keep the previous good index
            sys.stderr.write(f"server: bm25 reload failed ({e}); keeping previous index.\n")

    def _maybe_reload_bm25(self) -> None:
        # mtime self-heal: pick up a runner rebuild even if a reload signal was missed.
        if not self.bm25_path.exists():
            return
        m = self.bm25_path.stat().st_mtime
        if self._bm25 is None or m != self._bm25_mtime:
            self._load_bm25()

    # ---- request handling --------------------------------------------------
    def handle(self, req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "ping":
            n = len(self._bm25.chunks) if self._bm25 is not None else 0
            return {"event": "pong", "corpus_chunks": n}
        if cmd == "reload":
            self._load_bm25()
            return {"event": "reloaded"}
        if cmd == "search":
            from tools.retrieval.search import search
            self._maybe_reload_bm25()
            hits = search(
                query=req["query"], repo_root=self.repo,
                top_k=req.get("top_k", 5),
                program=req.get("program"), role=req.get("role"), phase=req.get("phase"),
                use_graph=req.get("use_graph", True),
                use_rerank=req.get("use_rerank", True),
                reranker=self._reranker, lab_memory=self._lm, bm25_index=self._bm25,
            )
            return {"event": "result", "req_id": req.get("req_id"), "hits": hits}
        return {"event": "error", "req_id": req.get("req_id"), "msg": f"unknown cmd {cmd!r}"}

    # ---- serve loop --------------------------------------------------------
    def serve(self) -> None:
        sp, pp = sock_path(self.repo), pid_path(self.repo)
        sp.parent.mkdir(parents=True, exist_ok=True)
        pp.parent.mkdir(parents=True, exist_ok=True)  # may differ from sp.parent under RETRIEVAL_SOCK
        # Don't stomp a live server: if the socket already answers, another
        # daemon owns it — exit WITHOUT unlinking (else we'd orphan a ~1GB
        # process and overwrite its pidfile). Only unlink a truly-stale socket.
        if sp.exists():
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.5)
            try:
                probe.connect(str(sp))
                sys.stderr.write(f"server: a live server already owns {sp}; exiting.\n")
                probe.close()
                return
            except OSError:
                probe.close()
                try:
                    sp.unlink()  # stale socket from a crashed prior server
                except FileNotFoundError:
                    pass

        self._ensure_models()
        self._load_bm25()

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o077)  # owner-only socket (no group/other connect)
        try:
            srv.bind(str(sp))
        finally:
            os.umask(old_umask)
        srv.listen(16)
        pp.write_text(str(os.getpid()))
        sys.stderr.write(f"server: ready on {sp} (pid {os.getpid()})\n")
        sys.stderr.flush()

        def _cleanup() -> None:
            for p in (sp, pp):
                try:
                    p.unlink()
                except FileNotFoundError:
                    pass

        def _on_term(signum, frame):
            _cleanup()
            sys.exit(0)

        # Signal handlers can only be installed from the main thread; serve()
        # may run in a worker thread under test, so guard it.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, _on_term)
            signal.signal(signal.SIGINT, _on_term)

        conn_timeout = float(os.environ.get("RETRIEVAL_CONN_TIMEOUT", "30"))
        try:
            while True:
                conn, _ = srv.accept()
                # Bound every per-connection read/write: a half-open or silent
                # client must NOT wedge this single-threaded loop forever.
                conn.settimeout(conn_timeout)
                try:
                    with conn:
                        rfile = conn.makefile("rb")
                        line = rfile.readline(MAX_LINE)
                        if not line:
                            continue
                        try:
                            req = json.loads(line)
                        except json.JSONDecodeError as e:
                            conn.sendall((json.dumps({"event": "error", "msg": f"bad json: {e}"}) + "\n").encode())
                            continue
                        if req.get("cmd") == "shutdown":
                            conn.sendall(b'{"event":"bye"}\n')
                            break
                        try:
                            resp = self.handle(req)
                        except Exception as e:  # one bad request must not kill the daemon
                            resp = {"event": "error", "req_id": req.get("req_id"),
                                    "msg": f"{type(e).__name__}: {e}"}
                        conn.sendall((json.dumps(resp) + "\n").encode())
                except (BrokenPipeError, ConnectionResetError, socket.timeout, TimeoutError):
                    # Slow/half-open/dropped client — drop this connection, keep serving.
                    continue
        finally:
            srv.close()
            _cleanup()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Persistent retrieval daemon (UDS).")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args(argv)
    RetrievalServer(Path(args.repo).resolve()).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
