Retrieval Optimization in RAG for Financial Documents
Overview

This project implements a Retrieval-Augmented Generation (RAG) retrieval optimization pipeline for structured financial documents such as loan agreements. The system improves retrieval accuracy using semantic embeddings, BM25 keyword search, and hybrid ranking strategies, and exposes functionality through a FastAPI backend service.

The objective is to demonstrate how retrieval quality improves when combining vector similarity search with keyword-aware retrieval techniques.

🎯 Problem Statement

Design and optimize a Retrieval-Augmented Generation (RAG) pipeline to improve relevance and accuracy of retrieved document sections by:

experimenting with embedding models
implementing hybrid retrieval (vector + keyword search)
improving ranking logic
evaluating performance using structured queries
🧠 Approach

The pipeline processes uploaded PDFs and retrieves relevant document sections using a hybrid retrieval strategy:

PDF Upload
↓
Text Extraction
↓
Semantic Chunking
↓
Embedding Generation
↓
Vector Index (FAISS)
+
Keyword Index (BM25)
↓
Hybrid Retrieval
↓
Numeric-aware Boosting
↓
Evaluation Metrics
↓
FastAPI Endpoints
⚙️ Features

✅ Semantic chunking for structured financial clauses
✅ Embedding-based similarity search
✅ BM25 keyword retrieval
✅ Hybrid ranking strategy
✅ Numeric-aware retrieval boost for structured fields
✅ Retrieval accuracy evaluation pipeline
✅ Dynamic PDF upload support
✅ FastAPI backend deployment
✅ Swagger UI testing interface

📁 Project Structure
rag_retrieval_optimization/
│
├── app.py
├── main.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── hybrid_search.py
├── evaluation.py
├── sample_data/
│
└── README.md
📦 Libraries Used
Library	Purpose
pypdf	Extract text from PDFs
sentence-transformers	Generate semantic embeddings
faiss-cpu	Vector similarity search
rank-bm25	Keyword-based retrieval
fastapi	Backend API service
uvicorn	ASGI server
🔍 Retrieval Pipeline Components
1. Document Chunking

Instead of fixed-length splitting, documents are divided using line-aware semantic chunking to isolate structured clauses such as:

Loan Amount Sanctioned
Interest Rate
EMI Amount
Installments

This significantly improves retrieval precision.

2. Embedding Generation

Chunks are converted into dense vectors using:

all-MiniLM-L6-v2

This enables semantic similarity matching between queries and document sections.

Example:

"What is EMI amount?"
≈
"Equated Monthly Installment"
3. Vector Retrieval (FAISS)

FAISS indexes embeddings for fast nearest-neighbor search:

query → embedding → vector similarity search → candidate chunks
4. Keyword Retrieval (BM25)

BM25 improves retrieval for structured financial fields where exact matches matter:

interest rate
loan amount
installments
5. Hybrid Retrieval Strategy

Final ranking combines:

Vector similarity
+
BM25 keyword scoring
+
Numeric-aware boosting

This improves retrieval reliability in financial documents containing structured attributes.

6. Numeric-Aware Boosting

Structured financial attributes often contain numbers:

Loan Amount : 200000
Interest Rate : 13.65%
Installments : 96

Numeric-aware boosting ensures these fields rank higher during retrieval.

📊 Evaluation Strategy

Retrieval performance is measured using document-specific benchmark queries:

Example:

What is EMI amount?
What is interest rate?
Loan amount sanctioned?
Number of installments?

Accuracy is calculated by checking whether expected answers appear in retrieved chunks.

Example:

Baseline vector search accuracy: 0.50
Hybrid retrieval accuracy: 1.00
Improvement: +50%
🚀 API Endpoints

The pipeline is deployed using FastAPI.

Health Check
GET /

Returns API status.

Upload Document
POST /upload

Uploads and indexes a new PDF dynamically.

Example response:

Document uploaded and indexed successfully
Search Document
GET /search?query=What is EMI amount?

Returns relevant document sections.

Example:

EMI Amount : Rs 2546
Evaluate Retrieval Accuracy
GET /evaluate

Returns benchmark accuracy results.

Example:

accuracy: 1.0
🧪 Running the Project
Step 1: Install Dependencies
pip install -r requirements.txt
Step 2: Start API Server
uvicorn app:app --reload

Open:

http://127.0.0.1:8000/docs

to access Swagger UI.

Step 3: Upload Document

Use:

POST /upload
Step 4: Run Query

Use:

GET /search

Example:

What is interest rate?
Step 5: Evaluate Retrieval Accuracy

Use:

GET /evaluate
📈 Results Summary

The hybrid retrieval pipeline improves structured financial field extraction accuracy by combining:

semantic similarity search
keyword ranking
numeric-aware boosting

Compared to baseline vector retrieval alone.

🏗️ Design Decisions

Key architectural improvements implemented:

paragraph-aware chunking
hybrid vector + BM25 retrieval
numeric-field ranking boost
evaluation normalization logic
dynamic document indexing
FastAPI deployment interface

These changes significantly improved retrieval precision for structured loan agreements.

🔮 Future Improvements

Potential enhancements:

reranking using cross-encoder models
metadata-aware chunk indexing
multi-document indexing support
caching embeddings
graph-based retrieval reasoning