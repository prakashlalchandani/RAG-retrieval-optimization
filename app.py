from fastapi import FastAPI, UploadFile, File
import shutil
import re
import uuid
import os
from collections import defaultdict
from chunking import create_chunks
from embeddings import create_embeddings, model
from vector_store import build_faiss_index, search
from hybrid_search import (
    build_bm25,
    bm25_search,
    reciprocal_rank_fusion,
    numeric_boost_search,
)
from evaluation import check_retrieval
from query_transforms import HYDE_CONFIG, generate_query_variants, text_for_retrieval


app = FastAPI(title="RAG Retrieval Optimization API")


# Global pipeline state
chunks = None
embeddings = None
index = None
bm25 = None


def _build_query_set(query: str, variant_count: int = 4):
    query_set = []
    for candidate in [query] + generate_query_variants(query, n=variant_count):
        normalized = " ".join(candidate.strip().split())
        if normalized and normalized not in query_set:
            query_set.append(normalized)
    return query_set


def _retrieve_hits(query: str, top_k: int = 5, rrf_k: int = 60):
    rank_lists = []
    used_hyde = False
    hyde_error = None

    query_set = _build_query_set(query.lower())

    # query metadata for debug/analysis
    query_debug = []

    # per-doc fusion bookkeeping
    fused_scores = defaultdict(float)
    contribution_map = defaultdict(list)

    for idx, query_variant in enumerate(query_set):
        retrieval_text, variant_used_hyde, variant_hyde_error = text_for_retrieval(query_variant)
        query_embedding = model.encode([retrieval_text])

        vector_results = list(search(index, query_embedding, top_k=max(top_k * 2, 8))[0])
        bm25_results = bm25_search(bm25, query_variant, chunks, top_k=max(top_k * 2, 8))

        query_kind = "raw" if idx == 0 else "variant"
        if variant_used_hyde:
            query_kind = "hyde"

        query_debug.append(
            {
                "query": query_variant,
                "query_kind": query_kind,
                "retrieval_text": retrieval_text,
                "used_hyde": variant_used_hyde,
            }
        )

        rank_lists.extend([vector_results, bm25_results])
        used_hyde = used_hyde or variant_used_hyde
        if variant_hyde_error and hyde_error is None:
            hyde_error = variant_hyde_error

        for rank, doc_id in enumerate(vector_results, start=1):
            contribution = 1.0 / (rrf_k + rank)
            fused_scores[doc_id] += contribution
            contribution_map[doc_id].append(
                {
                    "query": query_variant,
                    "query_kind": query_kind,
                    "retriever": "vector",
                    "rank": rank,
                    "score_contribution": contribution,
                }
            )

        for rank, doc_id in enumerate(bm25_results, start=1):
            contribution = 1.0 / (rrf_k + rank)
            fused_scores[doc_id] += contribution
            contribution_map[doc_id].append(
                {
                    "query": query_variant,
                    "query_kind": query_kind,
                    "retriever": "bm25",
                    "rank": rank,
                    "score_contribution": contribution,
                }
            )

    numeric_results = numeric_boost_search(query, chunks)
    rank_lists.append(numeric_results)

    for rank, doc_id in enumerate(numeric_results, start=1):
        contribution = 1.0 / (rrf_k + rank)
        fused_scores[doc_id] += contribution
        contribution_map[doc_id].append(
            {
                "query": query,
                "query_kind": "raw",
                "retriever": "numeric",
                "rank": rank,
                "score_contribution": contribution,
            }
        )

    # Keep ordering aligned with existing fusion behavior.
    fused_ranked_ids = reciprocal_rank_fusion(rank_lists, k=rrf_k)

    top_hits = []
    for doc_id in fused_ranked_ids[:top_k]:
        hit_contributions = contribution_map.get(doc_id, [])
        top_hits.append(
            {
                "chunk_id": doc_id,
                "fused_score": fused_scores.get(doc_id, 0.0),
                "contributing_queries": sorted({item["query"] for item in hit_contributions}),
                "source_retrievers": sorted({item["retriever"] for item in hit_contributions}),
                "fusion_details": hit_contributions,
            }
        )

    return top_hits, query_set, used_hyde, hyde_error, query_debug, numeric_results


@app.get("/")
def health():
    return {"status": "API is running"}


# -----------------------------
# Upload and index PDF
# -----------------------------
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global chunks, embeddings, index, bm25

    # 1. Get the file extension (e.g., ".pdf")
    file_extension = os.path.splitext(file.filename)[1]

    # 2. Generate a random UUID and combine it with the extension
    unique_filename = f"{uuid.uuid4()}{file_extension}"

    # 3. Create the new, guaranteed-unique file path
    file_location = f"sample_data/{unique_filename}"

    # Now, even if the user uploads the same file twice,
    # it saves as two separate files (e.g., 550e8400-e29b-41d4-a716-446655440000.pdf)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    chunks = create_chunks(file_location)

    embeddings = create_embeddings(chunks)

    index = build_faiss_index(embeddings)

    bm25 = build_bm25(chunks)

    return {
        "message": "Document uploaded and indexed successfully",
        "chunks_created": len(chunks),
    }


# -----------------------------
# Search endpoint
# -----------------------------
@app.get("/search")
def search_query(query: str, debug: bool = False):

    if index is None:
        return {"error": "Upload a PDF first."}

    top_hits, query_set, used_hyde, hyde_error, query_debug, numeric_results = _retrieve_hits(query, top_k=3)
    retrieved_chunks = [chunks[hit["chunk_id"]] for hit in top_hits]

    response = {
        "query": query,
        "results": retrieved_chunks,
    }

    if debug:
        response["debug"] = {
            "hyde_enabled": HYDE_CONFIG.enabled,
            "hyde_used": used_hyde,
            "hyde_fallback_reason": hyde_error,
            "query_set": query_set,
            "generated_hyde_text": [
                {
                    "query": item["query"],
                    "hyde_text": item["retrieval_text"],
                }
                for item in query_debug
                if item["used_hyde"]
            ],
            "generated_query_variants": query_set[1:],
            "numeric_candidates": numeric_results,
            "top_candidates": top_hits,
        }

    return response


# -----------------------------
# Dynamic expected-value extractor
# -----------------------------
def extract_expected_values(chunks):

    expected = {}

    patterns = {
        "What is EMI amount?": r"EMI.*?(\d[\d,]*)",
        "What is interest rate?": r"Interest.*?(\d+\.?\d*)",
        "Loan amount sanctioned?": r"Loan\s*Amount.*?(\d[\d,]*)",
        "Number of installments?": r"Installments?.*?(\d+)",
    }

    for question, pattern in patterns.items():

        for chunk in chunks:

            match = re.search(pattern, chunk, re.IGNORECASE)

            if match:
                expected[question] = match.group(1)
                break

    return expected


# -----------------------------
# Evaluation endpoint
# -----------------------------
@app.get("/evaluate")
def evaluate_system():

    global chunks

    if chunks is None:
        return {"error": "Upload a PDF first."}

    expected_answers = extract_expected_values(chunks)

    correct = 0
    total = len(expected_answers)

    results_summary = []

    for query, expected_answer in expected_answers.items():
        top_hits, query_set, used_hyde, hyde_error, _, _ = _retrieve_hits(query, top_k=5)
        retrieved_chunks = [chunks[hit["chunk_id"]] for hit in top_hits]

        match = check_retrieval(expected_answer, retrieved_chunks)

        results_summary.append(
            {
                "query": query,
                "expected": expected_answer,
                "match_found": match,
                "query_set": query_set,
                "hyde_used": used_hyde,
                "hyde_fallback_reason": hyde_error,
                "top_hits": top_hits,
            }
        )

        if match:
            correct += 1

    accuracy = correct / total if total else 0

    return {
        "accuracy": accuracy,
        "details": results_summary,
        "hyde_enabled": HYDE_CONFIG.enabled,
    }
