#!/usr/bin/env sh
set -eu

CONFIG_PATH="${CONFIG_PATH:-configs/default.yaml}"

mkdir -p data/raw data/processed data/eval

if [ ! -s data/raw/wikipedia_subset.jsonl ]; then
  echo "[docker] Raw corpus not found. Downloading Wikipedia subset or using bundled fallback corpus..."
  python scripts/download_wikipedia_subset.py --config "$CONFIG_PATH"
fi

if [ ! -s data/processed/faiss.index ] || [ ! -s data/processed/bm25.pkl ] || [ ! -s data/processed/chunks.jsonl ]; then
  echo "[docker] Processed indexes not found. Building BM25, FAISS, embeddings, and chunks..."
  python scripts/build_index.py --config "$CONFIG_PATH"
fi

echo "[docker] Starting Streamlit application..."
exec streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}"
