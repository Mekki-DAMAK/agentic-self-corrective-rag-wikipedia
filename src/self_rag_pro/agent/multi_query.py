from __future__ import annotations

import re


def normalize_query(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip())


def core_query_terms(question: str) -> str:
    q = normalize_query(question).lower().strip(" ?!.")
    q = re.sub(r"^(what is|what are|who is|who are|define|explain|how does|why is|difference between)\s+", "", q).strip()
    return q or normalize_query(question)


def generate_multi_queries(question: str, max_queries: int = 5) -> list[str]:
    q = normalize_query(question)
    cleaned = core_query_terms(q)
    variants = [q]
    if cleaned and cleaned.lower() != q.lower().rstrip("?"):
        variants.append(cleaned)
    variants.append(f"{cleaned} definition" if cleaned else q)
    variants.append(f"{cleaned} overview" if cleaned else q)
    variants.append(f"what is {cleaned}" if cleaned else q.rstrip("?") + "?")
    seen: list[str] = []
    for v in variants:
        v = normalize_query(v)
        if v and v.lower() not in [x.lower() for x in seen]:
            seen.append(v)
    return seen[:max_queries]


def reformulate_query(original_question: str, current_query: str, attempt: int, previous_queries: list[str] | None = None) -> str:
    base = core_query_terms(original_question)
    templates = [
        "{base} definition",
        "what is {base}",
        "{base} meaning and overview",
        "{base} background information",
        "{base} related topic",
        "{base} source evidence",
    ]
    previous = {normalize_query(q).lower() for q in (previous_queries or [])}
    previous.add(normalize_query(current_query).lower())
    start = max(0, attempt - 1)
    ordered = templates[start:] + templates[:start]
    for template in ordered:
        candidate = normalize_query(template.format(base=base))
        if candidate.lower() not in previous:
            return candidate
    return normalize_query(f"{base} search attempt {attempt + 1}")
