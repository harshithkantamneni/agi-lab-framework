"""Pinned model revisions for reproducibility.

Reproducibility principle: an embedding/reranker model is a function f_theta(x)
parameterized by weights theta. Results are only comparable across runs/machines
if theta is FIXED. Loading "by tag" (e.g. 'BAAI/bge-reranker-v2-m3') resolves to
whatever revision the Hub currently serves — theta can change silently. Pinning
`revision=<commit-sha>` forces the exact weights.

This module loads the repo-root `model_revisions.json` lockfile (model id ->
commit SHA) and exposes `pinned_revision(model_id)`. Returns None if the lockfile
is missing or the model isn't pinned (callers then load 'latest' and should log
a warning).
"""
from __future__ import annotations
import json
from pathlib import Path

_CACHE: dict | None = None


def _load() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        here = Path(__file__).resolve()
        # lockfile lives at repo root (tools/retrieval/model_pins.py -> ../../)
        candidates = [
            here.parent.parent.parent / "model_revisions.json",  # repo root
            here.parent / "model_revisions.json",                # alongside, fallback
        ]
        for cand in candidates:
            if cand.exists():
                try:
                    data = json.loads(cand.read_text())
                    _CACHE = {k: v for k, v in data.items() if not k.startswith("_")}
                    break
                except (OSError, json.JSONDecodeError):
                    continue
    return _CACHE


def pinned_revision(model_id: str) -> str | None:
    """Return the pinned HF revision (commit SHA) for `model_id`, or None if it
    is not pinned / the lockfile is absent."""
    return _load().get(model_id)
