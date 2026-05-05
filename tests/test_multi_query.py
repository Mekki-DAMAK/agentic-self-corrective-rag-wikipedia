from self_rag_pro.agent.multi_query import generate_multi_queries, reformulate_query


def test_multi_query_non_empty():
    queries = generate_multi_queries("What is machine learning?")
    assert len(queries) >= 3
    assert queries[0] == "What is machine learning?"


def test_reformulations_are_different_across_attempts():
    previous = ["messi ?"]
    reformulations = []
    current = previous[0]
    for attempt in range(1, 6):
        current = reformulate_query("messi ?", current, attempt, previous)
        reformulations.append(current)
        previous.append(current)

    assert len(reformulations) == len(set(reformulations))
    assert reformulations[0] == "messi definition"
