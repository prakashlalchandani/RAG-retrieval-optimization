import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Load the re-ranker model 
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def build_bm25(chunks):
    tokenized_chunks = [chunk.split() for chunk in chunks]
    return BM25Okapi(tokenized_chunks)

def bm25_search(bm25, query, chunks, top_k=3):
    scores = bm25.get_scores(query.split())
    ranked = sorted(range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True)
    return ranked[:top_k]

def numeric_boost_search(query, chunks):
    results = []
    
    numbers_in_query = re.findall(r'\d+', query)
    
    if not numbers_in_query:
        return []

    for i, chunk in enumerate(chunks):
        if any(num in chunk for num in numbers_in_query):
            results.append(i)

    return results[:3]

def hybrid_search(vector_results, bm25_results, top_k=5):
    combined = []
    
    # Interleave to give equal priority to Vector and Keyword algorithms
    max_len = max(len(vector_results), len(bm25_results))
    
    for i in range(max_len):
        # Grab the top vector match
        if i < len(vector_results) and vector_results[i] not in combined:
            combined.append(vector_results[i])
            
        # Immediately grab the top keyword match
        if i < len(bm25_results) and bm25_results[i] not in combined:
            combined.append(bm25_results[i])
            
        # Stop early if we hit our maximum allowed chunks
        if len(combined) >= top_k:
            break
            
    return combined[:top_k]

def rerank_results(query: str, retrieved_chunks: list, top_k=3):
    pairs = [[query, chunk] for chunk in retrieved_chunks]
    
    # If there are no chunks to score, safely return an empty list
    if not pairs:
        return []
        
    scores = reranker.predict(pairs)
    ranked_pairs = sorted(zip(scores, retrieved_chunks), reverse=True)
    
    return [chunk for score, chunk in ranked_pairs][:top_k]