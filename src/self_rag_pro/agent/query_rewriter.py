from __future__ import annotations

import os
import re
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from self_rag_pro.agent.multi_query import core_query_terms, normalize_query
from self_rag_pro.agent.query_correction import correct_query_spelling
from self_rag_pro.agent.self_corrective import keywords
from self_rag_pro.utils.device import resolve_device


def _clean_query(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^(query|rewritten query|new query)\s*:\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    text = text.strip().strip("\"'`")
    text = re.split(r"[\n\r]", text)[0]
    text = re.sub(r"\s+", " ", text)
    return text[:120].strip(" .")


def _source_summary(chunks: list[dict[str, Any]], limit: int = 5) -> str:
    rows = []
    for chunk in chunks[:limit]:
        title = chunk.get("title", "")
        score = float(chunk.get("score", 0.0))
        rows.append(f"- {title} (score={score:.3f})")
    return "\n".join(rows) if rows else "- no chunks retrieved"


def _missing_terms(question: str, chunks: list[dict[str, Any]]) -> list[str]:
    q_terms = set(keywords(question))
    corpus_terms = set()
    for chunk in chunks:
        corpus_terms.update(keywords(f"{chunk.get('title', '')} {chunk.get('text', '')}"))
    return sorted(q_terms - corpus_terms)


def _fallback_query(original_question: str, previous_queries: list[str]) -> str:
    base = core_query_terms(original_question)
    variants = [
        f"{base} technical concept",
        f"{base} definition",
        f"what is {base}",
        f"{base} explanation",
        f"{base} related AI ML topic",
        f"{base} source evidence",
    ]
    previous_norm = {normalize_query(q).lower() for q in previous_queries}
    for variant in variants:
        candidate = normalize_query(variant)
        if candidate.lower() not in previous_norm:
            return candidate
    return normalize_query(f"{base} retry {len(previous_queries) + 1}")


class LLMQueryRewriter:
    def __init__(
        self,
        model_name: str,
        device_preference: str = "auto",
        offline: bool = False,
        required: bool = False,
        max_new_tokens: int = 32,
        temperature: float = 0.2,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.available = False
        self.error = None
        self.tokenizer = None
        self.model = None
        self.is_seq2seq = False
        self.device_name = resolve_device(device_preference)

        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            model_config = AutoConfig.from_pretrained(model_name)
            self.is_seq2seq = bool(getattr(model_config, "is_encoder_decoder", False))
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            dtype = torch.float16 if self.device_name == "cuda" and torch.cuda.is_available() else torch.float32
            model_cls = AutoModelForSeq2SeqLM if self.is_seq2seq else AutoModelForCausalLM
            self.model = model_cls.from_pretrained(model_name, torch_dtype=dtype)
            self.model.to(self.device_name)
            self.model.eval()
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            if required:
                raise RuntimeError(
                    f"Cannot load required LLM query rewriter '{model_name}'. "
                    "Check your internet connection or pre-download the Hugging Face model. "
                    f"Original error: {exc}"
                ) from exc

    def rewrite(
        self,
        original_question: str,
        current_query: str,
        judge_reason: str,
        rejected_chunks: list[dict[str, Any]],
        previous_queries: list[str],
        vocabulary: set[str] | None = None,
        domain_phrases: list[str] | None = None,
    ) -> str:
        if vocabulary:
            corrected, corrections = correct_query_spelling(current_query, vocabulary, domain_phrases)
            previous_norm = {normalize_query(q).lower() for q in previous_queries}
            if corrections and corrected.lower() not in previous_norm:
                return corrected

        if not self.available or self.model is None or self.tokenizer is None:
            return normalize_query(f"{core_query_terms(original_question)} technical explanation")

        previous = "\n".join(f"- {q}" for q in previous_queries)
        missing = ", ".join(_missing_terms(original_question, rejected_chunks)) or "none"
        prompt = (
            "You are the query rewriting module of a Self-RAG search agent.\n"
            "Rewrite the user's question into ONE better search query for an AI/ML technical dataset.\n"
            "Use the original intent. Do not answer the question. Do not explain.\n"
            "Avoid repeating any previous query. Return only the rewritten query.\n"
            "Never copy source titles, scores, bullet markers, or source text into the query.\n\n"
            "Domain hints: vague descriptions should be mapped to precise AI/ML terminology when appropriate. "
            "Examples: image filters or visual feature extractors may mean convolutional neural network CNN; "
            "training good but test bad may mean overfitting; exact word search may mean BM25; "
            "semantic nearest neighbors may mean vector search; query-document scoring may mean cross-encoder; "
            "randomly disabling neurons may mean dropout; splitting documents may mean chunking; "
            "queries keys values may mean attention mechanism.\n\n"
            f"Original question: {original_question}\n"
            f"Current failed query: {current_query}\n"
            f"Judge rejection reason: {judge_reason}\n"
            f"Missing important terms: {missing}\n"
            f"Previous queries:\n{previous}\n\n"
            f"Retrieved but rejected chunks:\n{_source_summary(rejected_chunks)}\n\n"
            "Rewritten search query:"
        )
        messages = [
            {"role": "system", "content": "You rewrite search queries for retrieval augmented generation systems."},
            {"role": "user", "content": prompt},
        ]
        if self.is_seq2seq:
            encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        elif hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt").input_ids
        if hasattr(encoded, "input_ids"):
            inputs = {key: value.to(self.device_name) for key, value in encoded.items()}
        else:
            inputs = {"input_ids": encoded.to(self.device_name)}

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0] if self.is_seq2seq else output[0][inputs["input_ids"].shape[-1] :]
        rewritten = _clean_query(self.tokenizer.decode(generated, skip_special_tokens=True))

        previous_norm = {normalize_query(q).lower() for q in previous_queries}
        invalid_markers = ["title=", "score=", "text=", "retrieved", "|", "self-rag search agent", "rewriting function"]
        invalid = any(marker in rewritten.lower() for marker in invalid_markers)
        q_terms = set(keywords(original_question))
        rewritten_terms = set(keywords(rewritten))
        changed_intent = bool(q_terms) and not bool(q_terms & rewritten_terms)
        invalid = invalid or changed_intent
        if not rewritten or invalid or rewritten.lower() in previous_norm:
            rewritten = _fallback_query(original_question, previous_queries)
        return rewritten
