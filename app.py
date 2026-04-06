from fastapi import FastAPI, UploadFile, File
import shutil
import re
from chunking import extract_text, create_chunks
from embeddings import create_embeddings, model
from vector_store import build_faiss_index, search
from hybrid_search import (
    build_bm25,
    bm25_search,
    hybrid_search,
    numeric_boost_search,
)
from evaluation import check_retrieval


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

    file_location = f"sample_data/{file.filename}"

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    text = extract_text(file_location)

    chunks = create_chunks(text)

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

    query_embedding = model.encode([query.lower()])

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

        query_embedding = model.encode([query.lower()])

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
            }
        )

        if match:
            correct += 1

    accuracy = correct / total if total else 0

    return {
        "accuracy": accuracy,
        "details": results_summary,
    }