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
    hybrid_search,
    numeric_boost_search,
)
from evaluation import check_retrieval
from query_transforms import HYDE_CONFIG, text_for_retrieval


app = FastAPI(title="RAG Retrieval Optimization API")


# Global pipeline state
chunks = None
embeddings = None
index = None
bm25 = None


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

    retrieval_text, used_hyde, hyde_error = text_for_retrieval(query.lower())
    query_embedding = model.encode([retrieval_text])

    vector_results = list(search(index, query_embedding)[0])

    bm25_results = bm25_search(bm25, query.lower(), chunks)

    numeric_results = numeric_boost_search(query, chunks)

    final_results = hybrid_search(
        vector_results + numeric_results,
        bm25_results,
    )

    retrieved_chunks = [chunks[i] for i in final_results[:3]]

    return {
        "query": query,
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

        retrieval_text, used_hyde, hyde_error = text_for_retrieval(query.lower())
        query_embedding = model.encode([retrieval_text])

        vector_results = list(search(index, query_embedding)[0])

        bm25_results = bm25_search(bm25, query.lower(), chunks)

        numeric_results = numeric_boost_search(query, chunks)

        final_results = hybrid_search(
            vector_results + numeric_results,
            bm25_results,
        )

        retrieved_chunks = [chunks[i] for i in final_results]

        match = check_retrieval(expected_answer, retrieved_chunks)

        results_summary.append(
            {
                "query": query,
                "expected": expected_answer,
                "match_found": match,
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
