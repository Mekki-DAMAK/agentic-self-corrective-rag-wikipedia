from __future__ import annotations

import os
from typing import Any

from sentence_transformers import CrossEncoder

from self_rag_pro.utils.device import resolve_device


class CrossEncoderReranker:
    def __init__(self, model_name: str, device_preference: str = "auto", offline: bool = False, required: bool = False):
        self.model = None
        self.available = False
        self.error = None
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self.error = "Offline mode enabled; using retrieval-score reranking."
            if required:
                raise RuntimeError(
                    "The real Cross-Encoder reranker is required, but runtime.offline=true. "
                    "Set runtime.offline=false in configs/default.yaml."
                )
            return
        device = resolve_device(device_preference)
        try:
            self.model = CrossEncoder(model_name, device=device)
            self.available = True
            self.error = None
        except Exception as exc:
            self.model = None
            self.available = False
            self.error = str(exc)
            if required:
                raise RuntimeError(
                    f"Cannot load required Cross-Encoder reranker '{model_name}'. "
                    "Check your internet connection or pre-download the Hugging Face model. "
                    f"Original error: {exc}"
                ) from exc

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not chunks:
            return []
        if not self.available:
            return sorted(chunks, key=lambda c: c.get("retrieval_score", 0), reverse=True)[:top_k]
        pairs = [(query, c["text"][:1200]) for c in chunks]
        model = self.model
        if model is None:
            return sorted(chunks, key=lambda c: c.get("retrieval_score", 0), reverse=True)[:top_k]
        scores = model.predict(pairs)
        out = []
        raw_scores = [float(s) for s in scores]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        spread = max(max_score - min_score, 1e-9)
        for c, s in zip(chunks, scores, strict=False):
            cc = dict(c)
            raw = float(s)
            rr = (raw - min_score) / spread
            cc["rerank_raw_score"] = raw
            cc["rerank_score"] = rr
            final_score = 0.35 * cc.get("retrieval_score", 0.0) + 0.65 * rr
            cc["score"] = max(0.0, min(1.0, final_score))
            out.append(cc)
        out.sort(key=lambda c: c["score"], reverse=True)
        return out[:top_k]
