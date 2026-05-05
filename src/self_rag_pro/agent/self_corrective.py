from __future__ import annotations

import re
from collections import Counter
from typing import Any

from self_rag_pro.models.schemas import JudgeResult

STOPWORDS = set("""
a an the and or but if while with without of in on for to from by as is are was were be been being this that these those it its into about between within using used use can may many more most such not also their there which who what when where how why than then through over under based only among other
definition overview explain difference
automatic automatically meaning background information related topic source evidence
model models data system systems method methods technique techniques task tasks example examples
good bad picture thing stuff technical concept concepts source sources
win wins won winner world cup match tournament
""".split())


def keywords(text: str) -> list[str]:
    toks = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) > 2]


def judge_sources(question: str, query: str, chunks: list[dict[str, Any]], threshold: float, min_useful_sources: int) -> JudgeResult:
    if not chunks:
        return JudgeResult("rejected", "No sources retrieved.", 0, 0, False, False, 0, 0, suggested_query=query + " definition")
    q_terms = set(keywords(question))
    useful = 0
    lexical_hits_total = 0
    title_match = False
    definition_match = False
    scores = []
    for c in chunks:
        txt = (c["title"] + " " + c["text"]).lower()
        c_terms = set(keywords(txt))
        hits = len(q_terms & c_terms)
        lexical_hits_total += hits
        chunk_title_match = bool(set(keywords(c["title"])) & q_terms)
        title_match = title_match or chunk_title_match
        definition_match = definition_match or any(p in c["text"].lower()[:900] for p in [" is a ", " is an ", " refers to ", " is the ", " are a "])
        score = float(c.get("score", 0.0))
        scores.append(score)
        has_evidence_overlap = hits > 0 or chunk_title_match
        if has_evidence_overlap and (score >= threshold * 0.5 or hits >= 1):
            useful += 1
    mean_score = sum(scores) / max(1, len(scores))
    useful_ratio = useful / max(1, len(chunks))
    has_query_evidence = lexical_hits_total > 0 or title_match
    accepted = has_query_evidence and useful >= min_useful_sources and (mean_score >= threshold or lexical_hits_total >= 2 or title_match)
    if accepted:
        reason = f"Sources accepted: useful_sources={useful}, mean_score={mean_score:.3f}, lexical_hits={lexical_hits_total}."
        return JudgeResult("accepted", reason, mean_score, lexical_hits_total, title_match, definition_match, useful, useful_ratio)
    top_terms = [t for t, _ in Counter(keywords(question)).most_common(5)]
    suggested = " ".join(top_terms + ["definition", "overview"])
    reason = f"Sources rejected: useful_sources={useful}, mean_score={mean_score:.3f}, lexical_hits={lexical_hits_total}."
    return JudgeResult("rejected", reason, mean_score, lexical_hits_total, title_match, definition_match, useful, useful_ratio, suggested_query=suggested)
