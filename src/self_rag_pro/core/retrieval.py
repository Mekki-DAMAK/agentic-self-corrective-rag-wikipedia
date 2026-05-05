from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from self_rag_pro.ingest.storage import read_jsonl


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())


def definition_bonus(text: str) -> float:
    s = text.lower()[:700]
    patterns = [" is a ", " is an ", " refers to ", " is the ", " are a ", " is defined as"]
    return 1.0 if any(p in s for p in patterns) else 0.0


def title_match_bonus(query: str, title: str) -> float:
    q = set(tokenize(query))
    t = set(tokenize(title))
    if not q or not t:
        return 0.0
    return len(q & t) / max(1, len(t))


class HybridRetriever:
    def __init__(self, chunks_path: str, faiss_index_path: str, embedding_model: Any, bm25_path: str | None = None):
        self.chunks = read_jsonl(chunks_path)
        self.embedding_model = embedding_model
        self.index = faiss.read_index(str(faiss_index_path))
        if bm25_path and Path(bm25_path).exists():
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = BM25Okapi([tokenize(c["text"] + " " + c["title"]) for c in self.chunks])

    def bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.argsort(scores)[::-1][:top_k]
        max_score = float(scores[idx[0]]) if len(idx) and scores[idx[0]] > 0 else 1.0
        return [(int(i), float(scores[i] / max_score)) for i in idx]

    def vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        emb = self.embedding_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(emb, top_k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=False) if i >= 0]

    def search_many(self, queries: list[str], cfg: dict[str, Any]) -> list[dict[str, Any]]:
        acc: dict[int, dict[str, float]] = {}
        for query in queries:
            for i, s in self.bm25_search(query, cfg["bm25_top_k"]):
                entry = acc.setdefault(i, {"bm25": 0.0, "vector": 0.0, "bonus": 0.0})
                entry["bm25"] = max(entry["bm25"], s)
            for i, s in self.vector_search(query, cfg["vector_top_k"]):
                entry = acc.setdefault(i, {"bm25": 0.0, "vector": 0.0, "bonus": 0.0})
                entry["vector"] = max(entry["vector"], s)

        main_query = queries[0]
        fused = []
        for i, parts in acc.items():
            c = dict(self.chunks[i])
            title_bonus_value = cfg.get("title_bonus", 0.12) * title_match_bonus(main_query, c["title"])
            definition_bonus_value = cfg.get("definition_bonus", 0.08) * definition_bonus(c["text"])
            bonus = title_bonus_value + definition_bonus_value
            score = 0.48 * parts["bm25"] + 0.52 * parts["vector"] + bonus
            c["bm25_score"] = float(parts["bm25"])
            c["vector_score"] = float(parts["vector"])
            c["title_bonus"] = float(title_bonus_value)
            c["definition_bonus"] = float(definition_bonus_value)
            c["bonus_score"] = float(bonus)
            c["retrieval_score"] = float(score)
            c["score"] = float(score)
            c["rerank_score"] = 0.0
            fused.append(c)
        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[: cfg["fused_top_k"]]
