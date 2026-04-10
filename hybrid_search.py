from rank_bm25 import BM25Okapi


def build_bm25(chunks):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def bm25_search(bm25, query, chunks, top_k=3):

    scores = bm25.get_scores(query.split())

    ranked = sorted(range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True)

    return ranked[:top_k]

def reciprocal_rank_fusion(rank_lists, k=60):
    fused_scores = {}

    for rank_list in rank_lists:
        for rank, doc_id in enumerate(rank_list, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return [doc_id for doc_id, _ in ranked]


def hybrid_search(vector_results, bm25_results, k=60):

    return reciprocal_rank_fusion([vector_results, bm25_results], k=k)

def numeric_boost_search(query, chunks):

    results = []

    for i, chunk in enumerate(chunks):

        if any(word in chunk.lower() for word in query.lower().split()):
            results.append(i)

    return results[:3]
