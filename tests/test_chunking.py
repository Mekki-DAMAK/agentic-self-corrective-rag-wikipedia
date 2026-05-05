from self_rag_pro.core.chunking import split_into_chunks


def test_chunking_creates_chunks():
    doc = {"id": "x", "title": "Machine learning", "url": "u", "text": "Machine learning is a field.\n\n" * 100}
    chunks = split_into_chunks(doc, chunk_size=500, chunk_overlap=50, min_chunk_chars=100)
    assert len(chunks) > 0
    assert chunks[0].title == "Machine learning"
