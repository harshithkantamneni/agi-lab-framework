"""L4: Cross-encoder reranker over hybrid retrieval top-K.

Uses sentence-transformers' CrossEncoder with `bge-reranker-v2-m3` by default
(SOTA listwise reranker; 568M params -> ~1.05 GB resident RSS in fp32, NOT
568 MB; first-load ~11s, then ~50-100ms for 30 pairs). Falls back to
`ms-marco-MiniLM-L-12-v2` (~120 MB; faster, English-only) if the default fails
to load. To halve the resident footprint, fp16 weights can be used on MPS/GPU
(`model_kwargs={"torch_dtype": torch.float16}`) — left fp32 here because CPU
half-precision ops are poorly supported; revisit when running on MPS.

The persistent server (tools/retrieval/server.py) holds ONE instance of this
reranker for the whole lab run, so the ~1 GB cost is paid once, not per query.
"""
from __future__ import annotations
import sys
from dataclasses import dataclass


DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"
FALLBACK_RERANKER = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@dataclass
class CrossEncoderReranker:
    model_name: str = DEFAULT_RERANKER
    _model: object | None = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            from tools.retrieval.model_pins import pinned_revision
            rev = pinned_revision(self.model_name)
            try:
                self._model = CrossEncoder(self.model_name, revision=rev)
            except Exception as e:
                # Explicit, LOGGED fallback — never a silent model swap (a silent
                # swap would change retrieval rankings with no trace).
                sys.stderr.write(
                    f"rerank: failed to load {self.model_name}@{rev or 'latest'} "
                    f"({type(e).__name__}: {e}); falling back to {FALLBACK_RERANKER}.\n"
                )
                try:
                    self._model = CrossEncoder(
                        FALLBACK_RERANKER, revision=pinned_revision(FALLBACK_RERANKER)
                    )
                except Exception as e2:
                    raise RuntimeError(
                        f"rerank: both primary ({self.model_name}) and fallback "
                        f"({FALLBACK_RERANKER}) rerankers failed to load: {e2}"
                    ) from e2
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
        text_key: str = "chunk_text",
    ) -> list[dict]:
        if not candidates:
            return []
        model = self._get_model()
        pairs = [(query, c.get(text_key, "")) for c in candidates]
        scores = model.predict(pairs)
        scored = [
            {**c, "rerank_score": float(s)}
            for c, s in zip(candidates, scores)
        ]
        scored.sort(key=lambda c: c["rerank_score"], reverse=True)
        return scored[:top_k]
