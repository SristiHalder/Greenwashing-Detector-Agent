# agent/utils.py
# Shared utilities used by report_generator.py and explain_agent.py


def select_key_chunks(chunks, limit=8, query=None):
    """
    Select and compress the most forensically relevant chunks.

    Ranking criteria:
      - external_critique verdict: +3
      - each red flag: +1
      - vague specificity: +1
      - optional keyword match against user query: +0.5 per matching word

    Args:
        chunks:  list of ai_chunks records (dicts with 'ai_analysis' key)
        limit:   max number of chunks to return
        query:   optional user question string for query-aware re-ranking

    Returns:
        list of compressed chunk dicts (verdict, red_flags, key_sentences, source_url)
    """

    scored = []

    for c in chunks:
        ai = c["ai_analysis"]
        weight = 0

        if ai.get("verdict") == "external_critique":
            weight += 3

        if ai.get("red_flags"):
            weight += len(ai["red_flags"])

        if ai.get("specificity") == "vague":
            weight += 1

        # Query-aware boost: re-rank chunks relevant to the user's question
        if query:
            chunk_text = " ".join(ai.get("key_sentences", [])).lower()
            for word in query.lower().split():
                if len(word) > 3 and word in chunk_text:  # skip short stop-words
                    weight += 0.5

        scored.append((weight, c))

    scored.sort(reverse=True, key=lambda x: x[0])

    top = [c for _, c in scored[:limit]]

    compressed = []
    for c in top:
        compressed.append({
            "verdict": c["ai_analysis"].get("verdict"),
            "specificity": c["ai_analysis"].get("specificity"),
            "red_flags": c["ai_analysis"].get("red_flags"),
            "key_sentences": c["ai_analysis"].get("key_sentences"),
            "source_url": c.get("source_url")
        })

    return compressed