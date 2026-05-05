from __future__ import annotations

import hashlib
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from self_rag_pro.utils.device import resolve_device


class HashingEmbeddingModel:
    """Small deterministic fallback used when Hugging Face models are unavailable."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts, batch_size: int = 32, normalize_embeddings: bool = True, show_progress_bar: bool = False):
        if isinstance(texts, str):
            texts = [texts]
        vectors: np.ndarray = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, bucket] += sign
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        return vectors


def load_embedding_model(model_name: str, device_preference: str = "auto", offline: bool = False):
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        return HashingEmbeddingModel()
    device = resolve_device(device_preference)
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception as exc:
        mode = "offline local cache" if offline else "online download"
        print(
            f"[WARN] Cannot load embedding model '{model_name}' in {mode}. "
            f"Using deterministic local hashing embeddings instead. Original error: {exc}"
        )
        return HashingEmbeddingModel()
