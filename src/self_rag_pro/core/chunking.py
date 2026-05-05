from __future__ import annotations

import re

from self_rag_pro.models.schemas import Chunk

WIKI_NOISE_HEADINGS = {
    "references", "see also", "external links", "further reading", "bibliography", "notes"
}


def clean_wikipedia_text(text: str) -> str:
    text = re.sub(r"\n={2,}\s*(References|See also|External links|Further reading|Bibliography|Notes)\s*={2,}.*", "", text, flags=re.I | re.S)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(doc: dict, chunk_size: int = 900, chunk_overlap: int = 160, min_chunk_chars: int = 250) -> list[Chunk]:
    text = clean_wikipedia_text(doc["text"])
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    rank = 0
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if len(current) >= min_chunk_chars:
                chunks.append(Chunk(id=f"{doc['id']}::chunk-{rank}", document_id=doc["id"], title=doc["title"], text=current, url=doc["url"], rank=rank))
                rank += 1
            overlap = current[-chunk_overlap:] if current else ""
            current = (overlap + "\n\n" + para).strip()
    if len(current) >= min_chunk_chars:
        chunks.append(Chunk(id=f"{doc['id']}::chunk-{rank}", document_id=doc["id"], title=doc["title"], text=current, url=doc["url"], rank=rank))
    return chunks


def chunk_documents(docs: list[dict], chunk_size: int, chunk_overlap: int, min_chunk_chars: int) -> list[dict]:
    out = []
    for doc in docs:
        out.extend([c.to_dict() for c in split_into_chunks(doc, chunk_size, chunk_overlap, min_chunk_chars)])
    return out
