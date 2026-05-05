from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from self_rag_pro.agent.multi_query import generate_multi_queries, normalize_query, reformulate_query
from self_rag_pro.agent.query_correction import build_domain_vocabulary, has_correction_opportunity, is_real_correction
from self_rag_pro.agent.query_rewriter import LLMQueryRewriter
from self_rag_pro.agent.self_corrective import judge_sources, keywords
from self_rag_pro.core.answering import AnswerGenerator, extractive_answer, verify_answer
from self_rag_pro.core.embeddings import load_embedding_model
from self_rag_pro.core.reranking import CrossEncoderReranker
from self_rag_pro.core.retrieval import HybridRetriever
from self_rag_pro.evaluation.ragas_eval import evaluate_with_ragas_or_fallback
from self_rag_pro.models.schemas import AttemptTrace, RAGResult
from self_rag_pro.utils.config import load_config
from self_rag_pro.utils.telemetry import TelemetryLogger

ProgressFn = Callable[[str], None]


def filter_relevant_chunks(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q_terms = set(keywords(question))
    if not q_terms:
        return chunks
    relevant = []
    for chunk in chunks:
        text = f"{chunk.get('title', '')} {chunk.get('text', '')}"
        c_terms = set(keywords(text))
        if q_terms & c_terms:
            relevant.append(chunk)
    return relevant or chunks[:1]


def annotate_debug_scores(question: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    q_terms = set(keywords(question))
    annotated = []
    for chunk in chunks:
        item = dict(chunk)
        text = f"{item.get('title', '')} {item.get('text', '')}"
        c_terms = set(keywords(text))
        hits = sorted(q_terms & c_terms)
        item["judge_lexical_hits"] = len(hits)
        item["judge_hit_terms"] = ", ".join(hits)
        item["judge_title_match"] = bool(set(keywords(item.get("title", ""))) & q_terms)
        item["judge_useful"] = bool(hits or item["judge_title_match"])
        annotated.append(item)
    return annotated


class SelfRAGPipeline:
    def __init__(self, config_path: str = "configs/default.yaml", progress: ProgressFn | None = None):
        self.cfg = load_config(config_path)
        self.progress = progress or (lambda msg: None)
        self.telemetry = TelemetryLogger({**self.cfg.get("tracking", {}), "config": self.cfg})
        runtime = self.cfg["runtime"]
        data = self.cfg["data"]
        models = self.cfg["models"]
        self.progress("Loading YAML configuration")
        for required in [data["chunks_jsonl"], data["faiss_index"]]:
            if not Path(required).exists():
                raise FileNotFoundError(f"Missing {required}. Run: python scripts/download_wikipedia_subset.py then python scripts/build_index.py")
        self.progress("Loading embedding model")
        self.embedding_model = load_embedding_model(models["embedding"], runtime["device"], runtime.get("offline", False))
        self.progress("Loading Wikipedia chunks + FAISS index + BM25")
        self.retriever = HybridRetriever(data["chunks_jsonl"], data["faiss_index"], self.embedding_model, data.get("bm25_pickle"))
        self.domain_vocabulary, self.domain_phrases = build_domain_vocabulary(self.retriever.chunks)
        self.progress("Loading Cross-Encoder reranker")
        self.reranker = CrossEncoderReranker(
            models["reranker"],
            runtime["device"],
            runtime.get("offline", False),
            self.cfg["retrieval"].get("require_reranker", False),
        )
        self.progress("Loading LLM query rewriter")
        rewrite_cfg = self.cfg.get("query_rewriting", {})
        self.query_rewriter = None
        if rewrite_cfg.get("use_llm", False):
            self.query_rewriter = LLMQueryRewriter(
                models["query_rewriter"],
                runtime["device"],
                runtime.get("offline", False),
                rewrite_cfg.get("require_llm", False),
                rewrite_cfg.get("max_new_tokens", 32),
                rewrite_cfg.get("temperature", 0.2),
            )
        if self.cfg["generation"].get("use_local_generator", True):
            self.progress("Loading local generator")
        else:
            self.progress("Local generator disabled: using extractive fallback")
        self.generator = AnswerGenerator(models["generator"], runtime["device"], runtime.get("offline", False), self.cfg["generation"].get("use_local_generator", True))

    def ask(self, question: str, with_ragas: bool = False, progress: ProgressFn | None = None) -> dict[str, Any]:
        previous_progress = self.progress
        if progress is not None:
            self.progress = progress
        try:
            return self._ask(question, with_ragas=with_ragas)
        finally:
            self.progress = previous_progress

    def _ask(self, question: str, with_ragas: bool = False) -> dict[str, Any]:
        retrieval_cfg = self.cfg["retrieval"]
        gen_cfg = self.cfg["generation"]
        effective_question = normalize_query(question)
        query = effective_question
        timeline: list[dict[str, Any]] = []
        accepted_chunks: list[dict[str, Any]] = []
        accepted = False
        final_query = query
        judge = None
        attempted_queries: list[str] = []
        for attempt in range(1, retrieval_cfg["max_attempts"] + 1):
            attempted_queries.append(query)
            self.progress(f"Attempt {attempt}: generating alternative queries")
            queries = generate_multi_queries(query)
            self.progress("BM25 + FAISS search and score fusion")
            retrieved = self.retriever.search_many(queries, retrieval_cfg)
            self.progress("Cross-Encoder reranking")
            reranked = self.reranker.rerank(query, retrieved, retrieval_cfg["rerank_top_k"])
            reranked = annotate_debug_scores(effective_question, reranked)
            self.progress("Self-RAG judge: source accept/reject")
            judge = judge_sources(effective_question, query, reranked, retrieval_cfg["acceptance_threshold"], retrieval_cfg["min_useful_sources"])
            next_query = None
            correction_query, correction_pairs = has_correction_opportunity(query, self.domain_vocabulary, self.domain_phrases)
            force_correction_rewrite = (
                bool(correction_pairs)
                and is_real_correction(query, correction_query)
                and correction_query.lower() not in {normalize_query(q).lower() for q in attempted_queries}
                and attempt < retrieval_cfg["max_attempts"]
            )
            if force_correction_rewrite:
                next_query = correction_query
                judge.status = "rejected"
                judge.reason = (
                    f"Query spelling corrected before accepting sources: "
                    f"{', '.join(f'{old}->{new}' for old, new in correction_pairs)}."
                )
            if judge.status != "accepted" and attempt < retrieval_cfg["max_attempts"]:
                if next_query is not None:
                    pass
                elif self.query_rewriter is not None:
                    next_query = self.query_rewriter.rewrite(
                        effective_question,
                        query,
                        judge.reason,
                        reranked,
                        attempted_queries,
                        self.domain_vocabulary,
                        self.domain_phrases,
                    )
                else:
                    next_query = reformulate_query(effective_question, query, attempt, attempted_queries)
            trace = AttemptTrace(
                attempt=attempt,
                query=query,
                queries=queries,
                status=judge.status,
                reason=judge.reason,
                thinking_trace=[
                    f"Multi-query generated {len(queries)} variants.",
                    f"Hybrid retrieval returned {len(retrieved)} fused chunks.",
                    f"Reranker kept {len(reranked)} chunks.",
                    f"Judge status={judge.status}, useful_ratio={judge.useful_ratio:.2f}.",
                ],
                suggested_query=next_query,
                top_chunks=reranked,
            )
            timeline.append(asdict(trace))
            if judge.status == "accepted":
                accepted = True
                accepted_chunks = filter_relevant_chunks(effective_question, reranked)
                final_query = query
                break
            if next_query is None:
                final_query = query
                continue
            query = next_query
            effective_question = next_query
            final_query = query
            self.progress(f"Automatic reformulation: {query}")

        if not accepted:
            accepted_chunks = []
        self.progress("Generating grounded answer")
        answer, generation_mode = self.generator.generate(effective_question, accepted_chunks, gen_cfg)
        self.progress("Verifying answer against sources")
        verification = verify_answer(answer, accepted_chunks)
        if verification["grounding_ratio"] < gen_cfg.get("min_grounding_ratio", 0.55):
            answer = extractive_answer(effective_question, accepted_chunks)
            verification = verify_answer(answer, accepted_chunks)
            generation_mode = "extractive_grounding_fallback"
        if timeline:
            timeline[-1]["status"] = timeline[-1]["status"] + " / " + verification["status"]
            timeline[-1]["thinking_trace"].append(f"Answer generated with mode={generation_mode}.")
            timeline[-1]["thinking_trace"].append(f"Grounding ratio={verification['grounding_ratio']:.2f}.")
        best_score = max([c.get("score", 0.0) for c in accepted_chunks], default=0.0)
        confidence = min(1.0, 0.45 * best_score + 0.35 * verification["grounding_ratio"] + (0.20 if accepted else 0.0))
        ragas_result = None
        if with_ragas:
            self.progress("RAGAS evaluation / local fallback")
            ragas_result = evaluate_with_ragas_or_fallback(final_query, answer, accepted_chunks, self.cfg)
            ragas_result["evaluated_query"] = final_query
        result = RAGResult(
            question=question,
            final_query=final_query,
            answer=answer,
            confidence=round(float(confidence), 4),
            accepted=accepted,
            attempts=len(timeline),
            sources=accepted_chunks,
            timeline=timeline,
            verification=verification,
            ragas=ragas_result,
        )
        result_dict = asdict(result)
        telemetry_payload = {
            "rag/accepted": int(accepted),
            "rag/attempts": len(timeline),
            "rag/confidence": round(float(confidence), 4),
            "rag/grounding_ratio": verification.get("grounding_ratio", 0.0),
            "rag/source_count": len(accepted_chunks),
        }
        if ragas_result:
            telemetry_payload.update(
                {
                    "ragas/faithfulness": ragas_result.get("faithfulness", 0.0),
                    "ragas/answer_relevancy": ragas_result.get("answer_relevancy", 0.0),
                }
            )
        self.telemetry.log(telemetry_payload)
        return result_dict
