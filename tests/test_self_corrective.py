from self_rag_pro.agent.self_corrective import judge_sources


def test_judge_rejects_unrelated_query_even_with_scores():
    chunks = [
        {
            "title": "Artificial intelligence",
            "text": "Artificial intelligence is a field of computer science.",
            "score": 0.95,
        },
        {
            "title": "Machine learning",
            "text": "Machine learning is a branch of artificial intelligence.",
            "score": 0.88,
        },
    ]

    result = judge_sources("messi?", "messi?", chunks, threshold=0.34, min_useful_sources=2)

    assert result.status == "rejected"
    assert result.lexical_hits == 0


def test_judge_rejects_out_of_domain_world_cup_query():
    chunks = [
        {
            "title": "Backpropagation",
            "text": "In 1993, Eric Wan won an international pattern recognition contest through backpropagation.",
            "score": 0.95,
        },
        {
            "title": "Computer vision",
            "text": "Computer vision extracts information from the real world to understand digital images.",
            "score": 0.91,
        },
        {
            "title": "Neural network",
            "text": "Neural networks are machine learning models used for pattern recognition.",
            "score": 0.86,
        },
    ]

    result = judge_sources("Who won the 2022 FIFA World Cup?", "Who won the 2022 FIFA World Cup?", chunks, threshold=0.34, min_useful_sources=1)

    assert result.status == "rejected"
    assert result.lexical_hits == 0


def test_judge_accepts_related_query_with_evidence_overlap():
    chunks = [
        {
            "title": "Machine learning",
            "text": "Machine learning is a branch of artificial intelligence.",
            "score": 0.80,
        },
        {
            "title": "Artificial intelligence",
            "text": "Artificial intelligence includes machine learning systems.",
            "score": 0.70,
        },
    ]

    result = judge_sources("What is machine learning?", "What is machine learning?", chunks, threshold=0.34, min_useful_sources=2)

    assert result.status == "accepted"
    assert result.lexical_hits > 0
