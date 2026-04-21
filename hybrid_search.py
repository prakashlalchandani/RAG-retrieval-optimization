from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import re


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

import re

def numeric_boost_search(query, chunks):
    results = []
    
    # 1. Find all numbers in the user's query (e.g., "8" years, "2022")
    numbers_in_query = re.findall(r'\d+', query)
    
    # 2. If the user didn't type any numbers, skip this search!
    if not numbers_in_query:
        return []

    # 3. If they did type a number, find chunks that contain that number
    for i, chunk in enumerate(chunks):
        if any(num in chunk for num in numbers_in_query):
            results.append(i)

    return results[:3]
# Load the re-ranker model (this is standard for production RAG)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_results(query: str, retrieved_chunks: list, top_k=3):
    # Create pairs of [Query, Chunk] to score
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    
    # The cross-encoder predicts a highly accurate relevance score
    scores = reranker.predict(pairs)
    
    # Sort the chunks by their new score
    ranked_pairs = sorted(zip(scores, retrieved_chunks), reverse=True)
    
    # Return the exact text of the top chunks
    return [chunk for score, chunk in ranked_pairs][:top_k]