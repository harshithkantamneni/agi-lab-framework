"""Tests for the persistent retrieval server + thin client + CLI fallback.

The fast tests use a fake in-thread UDS echo server (no models). The single
integration test (marked slow via its name + real-model guard) spawns the real
daemon against a tiny throwaway repo to prove warm persistence.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------
# Fast unit tests — client behaviour, no models, no real server.
# --------------------------------------------------------------------------
def test_query_server_raises_fast_when_no_socket(tmp_path):
    from tools.retrieval.client import query_server, ServerUnavailable
    t0 = time.time()
    with pytest.raises(ServerUnavailable):
        query_server("q", repo=tmp_path, connect_timeout=0.5)
    assert time.time() - t0 < 2.0, "missing-socket must fail fast, not hang"


class _FakeServer:
    """A one-shot-per-connection UDS server that returns canned responses."""
    def __init__(self, sock_file: Path, responder):
        self.sock_file = sock_file
        self.responder = responder
        self._stop = False
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(str(sock_file))
        self.srv.listen(8)
        self.srv.settimeout(0.25)
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while not self._stop:
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            with conn:
                line = conn.makefile("rb").readline()
                if not line:
                    continue
                req = json.loads(line)
                resp = self.responder(req)
                if resp is not None:
                    conn.sendall((json.dumps(resp) + "\n").encode())

    def stop(self):
        self._stop = True
        self.t.join(timeout=2)
        self.srv.close()


def _mk_sockfile(tmp_path: Path, monkeypatch) -> Path:
    # pytest tmp_path is too deep for AF_UNIX's ~104-char limit on macOS, so put
    # the socket at a short path under /tmp and point both ends at it via the
    # RETRIEVAL_SOCK override (also the production knob for deep checkouts).
    sf = Path("/tmp") / f"retr_test_{os.getpid()}_{abs(hash(str(tmp_path))) % 100000}.sock"
    try:
        sf.unlink()
    except FileNotFoundError:
        pass
    monkeypatch.setenv("RETRIEVAL_SOCK", str(sf))
    return sf


def test_query_server_parses_result(tmp_path, monkeypatch):
    from tools.retrieval.client import query_server
    sf = _mk_sockfile(tmp_path, monkeypatch)
    hits = [{"id": 1, "source_path": "a.md", "chunk_text": "x", "rerank_score": 0.9}]
    fake = _FakeServer(sf, lambda req: {"event": "result", "req_id": req.get("req_id"), "hits": hits})
    try:
        out = query_server("q", repo=tmp_path, connect_timeout=1.0, read_timeout=2.0)
        assert out == hits
    finally:
        fake.stop()


def test_query_server_error_event_raises(tmp_path, monkeypatch):
    from tools.retrieval.client import query_server, ServerUnavailable
    sf = _mk_sockfile(tmp_path, monkeypatch)
    fake = _FakeServer(sf, lambda req: {"event": "error", "msg": "boom"})
    try:
        with pytest.raises(ServerUnavailable):
            query_server("q", repo=tmp_path, connect_timeout=1.0, read_timeout=2.0)
    finally:
        fake.stop()


def test_query_server_read_timeout_raises_not_hang(tmp_path, monkeypatch):
    from tools.retrieval.client import query_server, ServerUnavailable
    sf = _mk_sockfile(tmp_path, monkeypatch)
    fake = _FakeServer(sf, lambda req: None)  # accept, never respond
    try:
        t0 = time.time()
        with pytest.raises(ServerUnavailable):
            query_server("q", repo=tmp_path, connect_timeout=1.0, read_timeout=0.5)
        assert time.time() - t0 < 3.0, "wedged server must trip read timeout, not hang"
    finally:
        fake.stop()


def test_ping_pong(tmp_path, monkeypatch):
    from tools.retrieval.client import ping
    sf = _mk_sockfile(tmp_path, monkeypatch)
    fake = _FakeServer(sf, lambda req: {"event": "pong", "corpus_chunks": 42})
    try:
        info = ping(tmp_path, connect_timeout=1.0, read_timeout=2.0)
        assert info["corpus_chunks"] == 42
    finally:
        fake.stop()


def _search_args(tmp_path, **over):
    import argparse
    base = dict(query="router entropy", db=str(tmp_path / "tools" / "lab_memory.db"),
                top_k=3, program=None, role=None, phase=None, type=None,
                legacy=False, no_graph=False, no_rerank=False, no_server=False)
    base.update(over)
    return argparse.Namespace(**base)


def test_cli_falls_back_to_inprocess_when_server_down(tmp_path, monkeypatch, capsys):
    """lab_memory.py search must catch a down/unreachable server and transparently
    run in-process search() — never error out. (Model-free: search() is stubbed.)"""
    import tools.lab_memory as lm
    import tools.retrieval.client as rc
    import tools.retrieval.search as rs

    def _boom(**kw):
        raise rc.ServerUnavailable("server down")
    monkeypatch.setattr(rc, "query_server", _boom)
    called = {}

    def _fake_search(**kw):
        called["kw"] = kw
        return [{"source_path": "fallback.md", "chunk_text": "hi", "rrf_score": 0.5}]
    monkeypatch.setattr(rs, "search", _fake_search)

    lm._cmd_search(_search_args(tmp_path))
    out = capsys.readouterr().out
    assert "fallback.md" in out, f"fallback output missing: {out!r}"
    assert called.get("kw"), "in-process search() was not called on fallback"


def test_cli_uses_server_when_available(tmp_path, monkeypatch, capsys):
    """When the server answers, the CLI prints its hits and does NOT run the
    in-process path (no redundant cold model load)."""
    import tools.lab_memory as lm
    import tools.retrieval.client as rc
    import tools.retrieval.search as rs

    monkeypatch.setattr(rc, "query_server",
                        lambda **kw: [{"source_path": "warm.md", "chunk_text": "x", "rerank_score": 0.9}])

    def _should_not_run(**kw):
        raise AssertionError("in-process search() ran despite a healthy server")
    monkeypatch.setattr(rs, "search", _should_not_run)

    lm._cmd_search(_search_args(tmp_path))
    out = capsys.readouterr().out
    assert "warm.md" in out, f"warm-path output missing: {out!r}"


def test_serve_loop_survives_slow_client_and_serves_ping(tmp_path, monkeypatch):
    """The single-threaded serve() loop must NOT wedge on a half-open client
    that connects but never sends a full line — the per-connection timeout drops
    it and the loop keeps serving. Stubs out model loading (no torch)."""
    from tools.retrieval.server import RetrievalServer
    from tools.retrieval import client as rclient

    sock = Path("/tmp") / f"retr_serve_{os.getpid()}.sock"
    try:
        sock.unlink()
    except FileNotFoundError:
        pass
    monkeypatch.setenv("RETRIEVAL_SOCK", str(sock))
    monkeypatch.setenv("RETRIEVAL_CONN_TIMEOUT", "1")  # drop a silent client after 1s

    srv = RetrievalServer(tmp_path)
    srv._ensure_models = lambda: None   # no real models
    srv._load_bm25 = lambda: None
    t = threading.Thread(target=srv.serve, daemon=True)
    t.start()
    try:
        # wait for readiness
        deadline = time.time() + 10
        while time.time() < deadline and not sock.exists():
            time.sleep(0.05)
        assert rclient.ping(tmp_path, connect_timeout=2, read_timeout=3)["corpus_chunks"] == 0

        # half-open client: connect, send nothing, never read. Server must drop it
        # (1s timeout) rather than wedge.
        bad = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bad.connect(str(sock))  # no sendall, no read

        # The loop must still serve a subsequent ping (proves it didn't wedge).
        t0 = time.time()
        assert rclient.ping(tmp_path, connect_timeout=2, read_timeout=5)["corpus_chunks"] == 0
        assert time.time() - t0 < 5, "serve loop wedged on the slow client"
        bad.close()
    finally:
        try:
            rclient.control(tmp_path, "shutdown")
        except Exception:
            pass
        t.join(timeout=5)
        try:
            sock.unlink()
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------
# Integration — real daemon, real models. Slow; skipped if models unavailable.
# --------------------------------------------------------------------------
def _models_cached() -> bool:
    hub = Path.home() / ".cache/huggingface/hub"
    return (hub / "models--sentence-transformers--all-MiniLM-L6-v2").exists()


# Opt-in: loads real models. In this environment .venv is on an iCloud-synced
# path, so a cold `import torch` (2141 files) can take MINUTES when evicted —
# too slow/flaky for CI. Gate behind an env flag; the fast tests above cover the
# socket protocol + fallback deterministically. This test proves the CORE
# property the daemon exists for — ONE model held + reused across queries — by
# driving RetrievalServer in-process (single torch import; no subprocess, no
# socket, no warmup-timeout fragility).
@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_RETRIEVAL_TESTS") != "1" or not _models_cached(),
    reason="slow real-model test; set RUN_SLOW_RETRIEVAL_TESTS=1 (and cache models)",
)
def test_server_reuses_one_model_across_queries(tmp_path):
    """RetrievalServer must load the embedder + reranker ONCE and reuse the same
    instances across queries (the whole point of the daemon)."""
    repo = tmp_path
    (repo / "tools").mkdir()
    (repo / "data" / "infra").mkdir(parents=True)
    db = repo / "tools" / "lab_memory.db"

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.lab_memory import LabMemory
    from tools.retrieval.bm25 import BM25Index
    from tools.retrieval.server import RetrievalServer

    src = repo / "doc.md"
    src.write_text("Router entropy collapse at step 500.\n\nDense versus MoE at sub-100M.\n")
    lm = LabMemory(str(db))
    lm.init()
    lm.ingest(str(src))
    BM25Index.from_lab_memory_db(db).save(repo / "tools/lab_bm25.json")

    srv = RetrievalServer(repo)
    srv._ensure_models()      # one cold load
    srv._load_bm25()
    reranker_obj = srv._reranker
    model_obj = srv._reranker._get_model()  # cached underlying cross-encoder
    lm_obj = srv._lm

    r1 = srv.handle({"cmd": "search", "query": "router entropy", "top_k": 2})
    r2 = srv.handle({"cmd": "search", "query": "dense moe", "top_k": 2})

    assert r1["event"] == "result" and r1["hits"], f"no hits: {r1}"
    assert r2["event"] == "result" and r2["hits"], f"no hits: {r2}"
    # Same instances after both queries -> models held + reused, not reloaded.
    assert srv._reranker is reranker_obj
    assert srv._reranker._get_model() is model_obj, "reranker model reloaded between queries"
    assert srv._lm is lm_obj, "embedder reloaded between queries"
    # ping reflects the loaded corpus
    assert srv.handle({"cmd": "ping"})["corpus_chunks"] >= 1
