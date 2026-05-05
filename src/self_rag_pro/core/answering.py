from __future__ import annotations

import os
import re
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from self_rag_pro.agent.self_corrective import keywords
from self_rag_pro.utils.device import resolve_device

BAD_SECTIONS = ["references", "see also", "external links", "further reading"]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 30]


def extractive_answer(question: str, sources: list[dict[str, Any]], max_sentences: int = 5) -> str:
    if not sources:
        return "I could not find any relevant information about this topic in my dataset."
    q_terms = set(keywords(question))
    candidates = []
    fallback_candidates = []
    for c in sources:
        for sent in split_sentences(c["text"]):
            if any(b in sent.lower()[:40] for b in BAD_SECTIONS):
                continue
            if sent and sent[0].islower():
                continue
            hits = len(q_terms & set(keywords(sent)))
            definition = 1 if any(p in sent.lower() for p in [" is a ", " is an ", " refers to ", " is the "]) else 0
            scored = (hits * 4 + definition * 2 + c.get("score", 0), sent)
            fallback_candidates.append(scored)
            if not q_terms or hits > 0:
                candidates.append(scored)
    if not candidates:
        candidates = fallback_candidates
    candidates.sort(reverse=True, key=lambda x: x[0])
    picked = []
    seen = set()
    for _, sent in candidates:
        key = sent[:80].lower()
        if key not in seen:
            picked.append(sent)
            seen.add(key)
        if len(picked) >= max_sentences:
            break
    if not picked:
        return "I could not generate a sufficiently grounded answer from the accepted sources."
    return " ".join(picked)


class AnswerGenerator:
    def __init__(self, model_name: str, device_preference: str = "auto", offline: bool = False, use_local: bool = True):
        self.available = False
        self.error = None
        self.pipe = None
        if not use_local:
            return
        if offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            device_name = resolve_device(device_preference)
            device_id = 0 if device_name == "cuda" and torch.cuda.is_available() else -1
            tok = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.pipe = pipeline("text2text-generation", model=model, tokenizer=tok, device=device_id)
            self.available = True
        except Exception as exc:
            self.error = str(exc)

    def generate(self, question: str, sources: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[str, str]:
        context = "\n\n".join([f"Source {i+1} - {s['title']}: {s['text'][:1000]}" for i, s in enumerate(sources)])
        context = context[: cfg.get("max_context_chars", 4200)]
        if self.available and self.pipe is not None:
            prompt = (
                "Answer the question using only the context. If the context is insufficient, say so.\n\n"
                f"Question: {question}\n\nContext:\n{context}\n\nGrounded answer:"
            )
            try:
                out = self.pipe(prompt, max_new_tokens=cfg.get("max_new_tokens", 180), do_sample=False)[0]["generated_text"]
                if len(out.strip()) > 20:
                    return out.strip(), "local_generator"
            except Exception as exc:
                self.error = str(exc)
        return extractive_answer(question, sources), "extractive_fallback"


def verify_answer(answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    answer_terms = [t for t in keywords(answer) if len(t) > 3]
    if not answer_terms:
        return {"grounding_ratio": 0.0, "unsupported_terms": [], "status": "answer_warning"}
    source_text = " ".join(s["text"] for s in sources).lower()
    supported = [t for t in answer_terms if t in source_text]
    unsupported = sorted(set([t for t in answer_terms if t not in source_text]))[:20]
    ratio = len(supported) / max(1, len(answer_terms))
    return {"grounding_ratio": ratio, "unsupported_terms": unsupported, "status": "answer_verified" if ratio >= 0.55 else "answer_warning"}
