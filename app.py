from fastapi import FastAPI, UploadFile, File
import shutil
import re
import uuid
import os
from chunking import create_chunks
from query_transform import generate_hyde_document, generate_multi_queries
from embeddings import create_embeddings
# THE FIX: Imported our new get_all_chunks function
from vector_store import build_qdrant_index, search_qdrant, get_all_chunks
from hybrid_search import (
    build_bm25,
    bm25_search,
    hybrid_search,
    numeric_boost_search,
    rerank_results,  
)
from evaluation import check_retrieval
from generation import generate_answer

app = FastAPI(title="RAG Retrieval Optimization API")

# Global pipeline state (Cleaned up)
chunks = None
embeddings = None
bm25 = None

@app.get("/")
def health():
    return {"status": "API is running"}

# -----------------------------
# Upload and index PDF
# -----------------------------
@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    global chunks, embeddings, bm25

    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_location = f"sample_data/{unique_filename}"

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
    global chunks, bm25

    # --- THE STATELESS RECOVERY TRIGGER ---
    if chunks is None:
        print("RAM is empty! Recovering state directly from Qdrant database...")
        chunks = get_all_chunks()
        if not chunks:
            return {"error": "Database is completely empty. Upload a PDF first."}
        # Instantly rebuild the keyword search algorithm in memory
        bm25 = build_bm25(chunks)
    # --------------------------------------

    # 1. HyDE Vector Search 
    hyde_doc = generate_hyde_document(query)
    hyde_embedding = create_embeddings([hyde_doc.lower()])
    vector_results = search_qdrant(hyde_embedding[0])

    # 2. Multi-Query BM25 Search
    expanded_queries = generate_multi_queries(query)
    expanded_queries.append(query) 
    
    bm25_results = []
    for q in expanded_queries:
        results = bm25_search(bm25, q.lower(), chunks)
        bm25_results.extend(results)
    
    bm25_results = list(dict.fromkeys(bm25_results))

    # 3. Hybrid Merge 
    numeric_results = numeric_boost_search(query, chunks)
    final_results = hybrid_search(vector_results + numeric_results, bm25_results)
    retrieved_chunks = [chunks[i] for i in final_results if i < len(chunks)]

    # 4. Cross-Encoder Re-Ranking 
    best_chunks = rerank_results(query, retrieved_chunks, top_k=3)
    
    # 5. LLM Synthesis 
    final_answer = generate_answer(query, best_chunks)

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
    global chunks, bm25

    # --- THE STATELESS RECOVERY TRIGGER ---
    if chunks is None:
        chunks = get_all_chunks()
        if not chunks:
            return {"error": "Upload a PDF first."}
        bm25 = build_bm25(chunks)
    # --------------------------------------

    expected_answers = extract_expected_values(chunks)

    correct = 0
    total = len(expected_answers)
    results_summary = []

    for query, expected_answer in expected_answers.items():
        query_embedding = create_embeddings([query.lower()])

        vector_results = search_qdrant(query_embedding[0])
        bm25_results = bm25_search(bm25, query.lower(), chunks)
        numeric_results = numeric_boost_search(query, chunks)

        final_results = hybrid_search(vector_results + numeric_results, bm25_results)

        retrieved_chunks = [chunks[i] for i in final_results if i < len(chunks)]
        match = check_retrieval(expected_answer, retrieved_chunks)

        results_summary.append({
            "query": query,
            "expected": expected_answer,
            "match_found": match,
        })

        if match:
            correct += 1

    accuracy = correct / total if total else 0

    return {
        "accuracy": accuracy,
        "details": results_summary,
    }