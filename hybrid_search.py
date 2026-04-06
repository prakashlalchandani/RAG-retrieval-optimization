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

def hybrid_search(vector_results, bm25_results):

    combined = []

    for i in vector_results:
        combined.append(i)

    for i in bm25_results:
        if i not in combined:
            combined.append(i)

    return combined[:5]

def numeric_boost_search(query, chunks):

    results = []

    for i, chunk in enumerate(chunks):

        if any(word in chunk.lower() for word in query.lower().split()):
            results.append(i)

    return results[:3]