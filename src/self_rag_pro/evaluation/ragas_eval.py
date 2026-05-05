from __future__ import annotations

from typing import Any

from self_rag_pro.agent.self_corrective import keywords


def _source_contexts(sources: list[dict[str, Any]]) -> list[str]:
    contexts = []
    for source in sources:
        text = str(source.get("text", "")).strip()
        title = str(source.get("title", "")).strip()
        if text:
            contexts.append(f"{title}\n{text}" if title else text)
    return contexts


def lightweight_scores(question: str, answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    source_text = " ".join(_source_contexts(sources)).lower()
    answer_terms = keywords(answer)
    question_terms = set(keywords(question))
    faithfulness = sum(1 for term in answer_terms if term in source_text) / max(1, len(answer_terms))
    answer_relevancy = len(question_terms & set(keywords(answer))) / max(1, len(question_terms))
    return {
        "faithfulness": round(float(faithfulness), 4),
        "answer_relevancy": round(float(answer_relevancy), 4),
        "mode": "lightweight_fallback",
    }


def _evaluate_with_real_ragas(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    contexts = _source_contexts(sources)
    if not contexts or not answer.strip():
        return lightweight_scores(question, answer, sources)

    dataset = Dataset.from_dict(
        {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
    )

    evaluation_cfg = cfg.get("evaluation", {})
    use_local_wrappers = bool(evaluation_cfg.get("use_local_ragas_wrappers", False))

    if use_local_wrappers:
        from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

        models_cfg = cfg.get("models", {})
        runtime_cfg = cfg.get("runtime", {})
        model_name = models_cfg.get("ragas_llm", "google/flan-t5-small")
        embedding_name = models_cfg.get("ragas_embeddings", "sentence-transformers/all-MiniLM-L6-v2")
        device = 0 if runtime_cfg.get("device") == "cuda" else -1
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        text2text = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=int(evaluation_cfg.get("ragas_max_new_tokens", 256)),
            device=device,
        )
        llm = LangchainLLMWrapper(HuggingFacePipeline(pipeline=text2text))
        embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=embedding_name))
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm, embeddings=embeddings)
    else:
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])

    scores = result.to_pandas().iloc[0].to_dict()
    return {
        "faithfulness": round(float(scores.get("faithfulness", 0.0)), 4),
        "answer_relevancy": round(float(scores.get("answer_relevancy", 0.0)), 4),
        "mode": "ragas",
    }


def evaluate_with_ragas_or_fallback(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    evaluation_cfg = cfg.get("evaluation", {})
    if not evaluation_cfg.get("use_ragas", True):
        return lightweight_scores(question, answer, sources)

    try:
        return _evaluate_with_real_ragas(question, answer, sources, cfg)
    except Exception as exc:
        if not evaluation_cfg.get("fallback_lightweight_metrics", True):
            raise
        scores = lightweight_scores(question, answer, sources)
        scores["ragas_error"] = str(exc)[:300]
        return scores


def evaluate_batch(results: list[dict[str, Any]], questions: list[str], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    per_question = []
    config = cfg or {"evaluation": {"use_ragas": False}}
    for result, question in zip(results, questions, strict=False):
        sources = result.get("accepted_sources") or result.get("sources") or []
        evaluated_query = result.get("final_query") or question
        scores = evaluate_with_ragas_or_fallback(evaluated_query, result.get("answer", ""), sources, config)
        score_global = round((scores["faithfulness"] + scores["answer_relevancy"]) / 2.0, 4)
        per_question.append(
            {
                "question": question,
                "evaluated_query": evaluated_query,
                "faithfulness": scores["faithfulness"],
                "answer_relevancy": scores["answer_relevancy"],
                "score_global": score_global,
                "mode": scores["mode"],
            }
        )

    mean_faithfulness = sum(item["faithfulness"] for item in per_question) / max(1, len(per_question))
    mean_answer_relevancy = sum(item["answer_relevancy"] for item in per_question) / max(1, len(per_question))
    score_global = sum(item["score_global"] for item in per_question) / max(1, len(per_question))
    modes = sorted({item["mode"] for item in per_question})
    return {
        "per_question": per_question,
        "mean_faithfulness": round(float(mean_faithfulness), 4),
        "mean_answer_relevancy": round(float(mean_answer_relevancy), 4),
        "score_global": round(float(score_global), 4),
        "mode": ",".join(modes),
    }
