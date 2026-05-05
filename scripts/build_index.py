from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_rag_pro.core.chunking import chunk_documents  # noqa: E402
from self_rag_pro.core.embeddings import load_embedding_model  # noqa: E402
from self_rag_pro.core.retrieval import tokenize  # noqa: E402
from self_rag_pro.ingest.sample_corpus import sample_articles  # noqa: E402
from self_rag_pro.ingest.storage import read_jsonl, write_jsonl  # noqa: E402
from self_rag_pro.utils.config import ensure_dirs, load_config  # noqa: E402


def merge_by_id(docs: list[dict]) -> list[dict]:
    merged = {}
    for doc in docs:
        merged[doc["id"]] = doc
    return list(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    try:
        docs = read_jsonl(cfg["data"]["wikipedia_jsonl"])
    except FileNotFoundError:
        docs = []
    if not docs:
        docs = sample_articles()
        write_jsonl(cfg["data"]["wikipedia_jsonl"], docs)
        print("[WARN] Missing or empty Wikipedia JSONL. Using bundled sample corpus.")
    else:
        docs = merge_by_id(docs + sample_articles())
        write_jsonl(cfg["data"]["wikipedia_jsonl"], docs)
        print("[INFO] Ensured bundled AI/ML corpus is present.")
    chunks = chunk_documents(docs, **cfg["chunking"])
    if not chunks:
        raise RuntimeError("No chunks were created. Check the input documents and chunking settings.")
    write_jsonl(cfg["data"]["chunks_jsonl"], chunks)
    print(f"Saved {len(chunks)} chunks")
    model = load_embedding_model(cfg["models"]["embedding"], cfg["runtime"]["device"], cfg["runtime"].get("offline", False))
    texts = [c["title"] + "\n" + c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True).astype("float32")
    np.save(cfg["data"]["embeddings_npy"], emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, cfg["data"]["faiss_index"])
    bm25 = BM25Okapi([tokenize(c["text"] + " " + c["title"]) for c in chunks])
    with open(cfg["data"]["bm25_pickle"], "wb") as f:
        pickle.dump(bm25, f)
    print(f"FAISS index saved to {cfg['data']['faiss_index']}")


if __name__ == "__main__":
    main()
