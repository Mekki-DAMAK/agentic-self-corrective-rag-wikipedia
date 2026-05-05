from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    id: str
    title: str
    text: str
    url: str


@dataclass
class Chunk:
    id: str
    document_id: str
    title: str
    text: str
    url: str
    rank: int = 0
    score: float = 0.0
    retrieval_score: float = 0.0
    rerank_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeResult:
    status: str
    reason: str
    mean_score: float
    lexical_hits: int
    title_match: bool
    definition_match: bool
    useful_sources: int
    useful_ratio: float
    suggested_query: str | None = None


@dataclass
class AttemptTrace:
    attempt: int
    query: str
    queries: list[str]
    status: str
    reason: str
    thinking_trace: list[str] = field(default_factory=list)
    suggested_query: str | None = None
    top_chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RAGResult:
    question: str
    final_query: str
    answer: str
    confidence: float
    accepted: bool
    attempts: int
    sources: list[dict[str, Any]]
    timeline: list[dict[str, Any]]
    verification: dict[str, Any]
    ragas: dict[str, Any] | None = None
