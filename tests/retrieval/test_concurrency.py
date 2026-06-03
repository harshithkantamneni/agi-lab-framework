"""Concurrency amortization for retrieval.

Background: when `tools/lab_memory.py search` is run concurrently as separate
subprocesses (PI did this with 3 parallel queries on 2026-05-19), each child
loaded the sentence-transformers model fresh and raced on the model cache,
hitting a 5+ minute timeout.

The PRODUCTION fix is the persistent daemon (tools/retrieval/server.py): one
warm process holds the models and serves all queries; concurrent CLI calls
become thin socket clients. That end-to-end amortization is proven by
tests/retrieval/test_server.py::test_server_persists_and_serves (same PID across
queries). `tools.retrieval.concurrency.RetrievalWorker` remains the underlying
spawn-once-serve-many building block; the tests below exercise it directly.

The real-model tests are gated behind RUN_SLOW_RETRIEVAL_TESTS=1 because in this
environment .venv is on an iCloud-synced path, so a cold `import torch` (2141
files) can take minutes when evicted — which would make these slow/flaky in CI.
"""
import os

import pytest

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent

_SLOW = pytest.mark.skipif(
    os.environ.get("RUN_SLOW_RETRIEVAL_TESTS") != "1",
    reason="loads real models; set RUN_SLOW_RETRIEVAL_TESTS=1 to run",
)


@_SLOW
def test_concurrency_worker_handles_n_queries():
    """RetrievalWorker is spawn-once-serve-many: assert the worker PID is
    identical before and after both queries — proves the same subprocess (and
    thus the same in-memory model) served both, rather than restarting."""
    from tools.retrieval.concurrency import RetrievalWorker
    w = RetrievalWorker.spawn(REPO)
    pid_before = w.proc.pid
    try:
        r1 = w.query("router entropy", top_k=2)
        r2 = w.query("dense vs MoE", top_k=2)
        assert isinstance(r1, list) and len(r1) > 0, f"r1 empty: {r1!r}"
        assert isinstance(r2, list) and len(r2) > 0, f"r2 empty: {r2!r}"
        assert pid_before == w.proc.pid, "worker subprocess restarted — model not reused"
        assert w.proc.poll() is None, "worker subprocess died between queries"
    finally:
        w.shutdown()


@_SLOW
def test_query_timeout_raises_TimeoutError():
    """query(timeout=...) bounds the stdout read so a hung worker can't deadlock
    the caller. An absurdly small timeout (1 ms) forces the select() to expire
    before the embed+similarity result arrives."""
    from tools.retrieval.concurrency import RetrievalWorker
    w = RetrievalWorker.spawn(REPO)
    try:
        with pytest.raises(TimeoutError):
            w.query("router entropy", top_k=2, timeout=0.001)
    finally:
        w.shutdown()


def test_spawn_handshake_times_out_on_slow_worker(tmp_path, monkeypatch):
    """RetrievalWorker.spawn must NOT block forever if the worker never signals
    ready (e.g. a wedged/slow model import). With a short spawn_timeout it raises
    rather than hanging. Uses a fake python that sleeps instead of loading models."""
    from tools.retrieval.concurrency import RetrievalWorker
    import time as _t
    t0 = _t.time()
    with pytest.raises((TimeoutError, RuntimeError)):
        RetrievalWorker.spawn(REPO, spawn_timeout=0.5, _python_override="__sleep__")
    assert _t.time() - t0 < 5.0, "spawn must fail fast on a non-responsive worker"
