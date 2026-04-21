from fastapi import FastAPI, UploadFile, File
import shutil
import re
import uuid
import os
from chunking import create_chunks
from query_transform import generate_hyde_document, generate_multi_queries
from embeddings import create_embeddings, model
from vector_store import build_qdrant_index, search_qdrant
from hybrid_search import (
    build_bm25,
    bm25_search,
    hybrid_search,
    numeric_boost_search,
    rerank_results,  # <-- NEW: Imported the re-ranker
)
from evaluation import check_retrieval
from generation import generate_final_answer  # <-- NEW: Imported the LLM generator


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

    build_qdrant_index(embeddings, chunks)

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

    # --- 1. HyDE for Vector Search (FAISS) ---
    # Generate the fake document
    hyde_doc = generate_hyde_document(query)
    # Embed the FAKE document, not the user's short query
    hyde_embedding = model.encode([hyde_doc.lower()])
    vector_results = search_qdrant(hyde_embedding[0])

    # --- 2. Multi-Query for Keyword Search (BM25) ---
    # Generate 3 variations of the query
    expanded_queries = generate_multi_queries(query)
    expanded_queries.append(query) # Always include the original!
    
    bm25_results = []
    # Search BM25 for every variation and combine them
    for q in expanded_queries:
        results = bm25_search(bm25, q.lower(), chunks)
        bm25_results.extend(results)
    
    # Remove duplicates from BM25 results while preserving order
    bm25_results = list(dict.fromkeys(bm25_results))

    # --- 3. Hybrid Merge ---
    numeric_results = numeric_boost_search(query, chunks)
    
    final_results = hybrid_search(
        vector_results + numeric_results,
        bm25_results,
    )

    # Get the top candidate chunks from the hybrid search
    retrieved_chunks = [chunks[i] for i in final_results]

    # --- 4. Cross-Encoder Re-Ranking (NEW) ---
    # Re-rank the candidates to get the absolute best 3
    best_chunks = rerank_results(query, retrieved_chunks, top_k=3)
    
    # --- 5. LLM Synthesis (NEW) ---
    # Generate the final, concise answer using only the best chunks
    final_answer = generate_final_answer(query, best_chunks)

    return {
        "query": query,
        "answer": final_answer,
        "sources_used": best_chunks,
        "hyde_document_used": hyde_doc,
        "expanded_queries_used": expanded_queries
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

        # --- THE UPDATE ---
        # Call the persistent Qdrant database instead of FAISS
        vector_results = search_qdrant(query_embedding[0])
        # ------------------

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