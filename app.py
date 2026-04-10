from fastapi import FastAPI, UploadFile, File
import shutil
import re
import uuid
import os
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


def _retrieve_chunk_ids(query: str, top_k: int = 5):
    rank_lists = []
    used_hyde = False
    hyde_error = None

    query_set = _build_query_set(query.lower())
    for query_variant in query_set:
        retrieval_text, variant_used_hyde, variant_hyde_error = text_for_retrieval(query_variant)
        query_embedding = model.encode([retrieval_text])

        vector_results = list(search(index, query_embedding, top_k=max(top_k * 2, 8))[0])
        bm25_results = bm25_search(bm25, query_variant, chunks, top_k=max(top_k * 2, 8))

        rank_lists.extend([vector_results, bm25_results])
        used_hyde = used_hyde or variant_used_hyde
        if variant_hyde_error and hyde_error is None:
            hyde_error = variant_hyde_error

    rank_lists.append(numeric_boost_search(query, chunks))
    fused_results = reciprocal_rank_fusion(rank_lists)
    return fused_results[:top_k], query_set, used_hyde, hyde_error


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
def search_query(query: str):

    if index is None:
        return {"error": "Upload a PDF first."}

    final_chunk_ids, query_set, used_hyde, hyde_error = _retrieve_chunk_ids(query, top_k=3)
    retrieved_chunks = [chunks[i] for i in final_chunk_ids]

    return {
        "query": query,
        "query_set": query_set,
        "chunk_ids": final_chunk_ids,
        "results": retrieved_chunks,
        "hyde_enabled": HYDE_CONFIG.enabled,
        "hyde_used": used_hyde,
        "hyde_fallback_reason": hyde_error,
    }


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
        final_results, query_set, used_hyde, hyde_error = _retrieve_chunk_ids(query, top_k=5)
        retrieved_chunks = [chunks[i] for i in final_results]

        match = check_retrieval(expected_answer, retrieved_chunks)

        results_summary.append(
            {
                "query": query,
                "expected": expected_answer,
                "match_found": match,
                "query_set": query_set,
                "hyde_used": used_hyde,
                "hyde_fallback_reason": hyde_error,
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
