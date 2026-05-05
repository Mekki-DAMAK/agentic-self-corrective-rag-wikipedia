from __future__ import annotations

import difflib
import re
from collections import Counter
from typing import Any

from self_rag_pro.agent.multi_query import normalize_query
from self_rag_pro.agent.self_corrective import STOPWORDS, keywords


def build_domain_vocabulary(chunks: list[dict[str, Any]], min_count: int = 1) -> tuple[set[str], list[str]]:
    counter: Counter[str] = Counter()
    phrases: Counter[str] = Counter()
    for chunk in chunks:
        title = str(chunk.get("title", "")).strip()
        text = f"{title} {chunk.get('text', '')}"
        for phrase in _candidate_phrases(title):
            phrases[phrase] += 3
        for phrase in _candidate_phrases(str(chunk.get("text", ""))):
            phrases[phrase] += 1
        for token in keywords(text):
            if token not in STOPWORDS and len(token) >= 4:
                counter[token] += 1
    vocabulary = {word for word, count in counter.items() if count >= min_count}
    domain_phrases = [
        phrase
        for phrase, count in phrases.most_common()
        if count >= min_count and all(token in vocabulary for token in phrase.split())
    ]
    return vocabulary, domain_phrases


def correct_query_spelling(
    query: str,
    vocabulary: set[str],
    domain_phrases: list[str] | None = None,
    cutoff: float = 0.84,
) -> tuple[str, list[tuple[str, str]]]:
    corrections: list[tuple[str, str]] = []
    phrase_corrected = _correct_domain_phrase(query, domain_phrases or [])
    if _canonical(phrase_corrected) != _canonical(query):
        corrections.append((query, phrase_corrected))
        return normalize_query(phrase_corrected), corrections

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        lower = token.lower()
        if lower in vocabulary or lower in STOPWORDS or len(lower) < 5:
            return token
        candidates = difflib.get_close_matches(lower, vocabulary, n=1, cutoff=cutoff)
        if not candidates:
            return token
        replacement = candidates[0]
        if replacement != lower:
            corrections.append((token, replacement))
            return _preserve_case(token, replacement)
        return token

    corrected = re.sub(r"[A-Za-z][A-Za-z0-9_-]*", replace, query)
    return normalize_query(corrected), corrections


def has_correction_opportunity(
    query: str,
    vocabulary: set[str],
    domain_phrases: list[str] | None = None,
    cutoff: float = 0.84,
) -> tuple[str, list[tuple[str, str]]]:
    corrected, corrections = correct_query_spelling(query, vocabulary, domain_phrases, cutoff)
    return corrected, corrections


def is_real_correction(original: str, corrected: str) -> bool:
    return _canonical(original) != _canonical(corrected)


def _preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement.capitalize()
    return replacement


def _canonical(text: str) -> str:
    return " ".join(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _correct_domain_phrase(query: str, domain_phrases: list[str], cutoff: float = 0.78) -> str:
    query_tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", query)]
    if not query_tokens or not domain_phrases:
        return query
    best_phrase = None
    best_score = 0.0
    for phrase in domain_phrases:
        phrase_tokens = phrase.split()
        if len(phrase_tokens) > len(query_tokens):
            continue
        for start in range(0, len(query_tokens) - len(phrase_tokens) + 1):
            window = query_tokens[start : start + len(phrase_tokens)]
            scores = [
                difflib.SequenceMatcher(None, source, target).ratio()
                for source, target in zip(window, phrase_tokens, strict=False)
            ]
            score = sum(scores) / len(scores)
            if score > best_score:
                best_score = score
                best_phrase = (start, start + len(phrase_tokens), phrase)
    if best_phrase is None or best_score < cutoff:
        return query
    start, end, phrase = best_phrase
    rebuilt = query_tokens[:start] + phrase.split() + query_tokens[end:]
    return " ".join(rebuilt)


def _candidate_phrases(text: str) -> list[str]:
    tokens = [t for t in keywords(text) if len(t) >= 4 and t not in STOPWORDS]
    phrases = []
    for size in (2, 3, 4):
        for start in range(0, len(tokens) - size + 1):
            phrase = " ".join(tokens[start : start + size])
            phrases.append(phrase)
    return phrases
