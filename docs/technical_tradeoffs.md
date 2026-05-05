# Technical Trade-offs

The pipeline combines lexical search, dense retrieval, reranking, agentic validation, and evaluation. This document summarizes the main design choices.

## Hybrid Retrieval

BM25 is useful for exact terms, acronyms, model names, and technical vocabulary. Dense retrieval with FAISS improves semantic recall when the question uses words that do not exactly match the source text. The system fuses both scores before reranking.

## Multi-query Search

Multi-query expansion improves recall, especially for short or noisy questions. The cost is a larger candidate set and slightly higher retrieval latency.

## Cross-Encoder Reranking

The Cross-Encoder compares the query and each candidate chunk jointly, which is more accurate than using embedding similarity alone. The trade-off is latency because every query-document pair must be scored.

## Self-RAG Loop

The judge can reject weak evidence and trigger spelling correction or query rewriting. This improves answer grounding but may require multiple retrieval attempts.

## RAGAS and Fallback Metrics

RAGAS is integrated for faithfulness and answer relevancy. A deterministic fallback is kept for offline demos or environments where the local evaluator model is unavailable.
