import re
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from qdrant_client.http import models
from config.settings import settings
from config.clients import qdrant_client
from config.logger import logger
from services.embeddings import create_embeddings, get_query_embedding
from agentlens_sdk.instrumentation import trace_span
from agentlens_sdk.models.model import SpanType

logger.info("Loading CrossEncoder Re-ranker into memory...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

def tokenize_multilingual(text: str) -> list:
    """
    Extracts words using Unicode boundaries. 
    This prevents Gujarati/Hindi/Marathi characters from being fused with punctuation.
    """
    return re.findall(r'\w+', text.lower(), flags=re.UNICODE)

class RetrievalService:
    # Nested Dictionary structure: { session_id: { filename: [chunks] } }
    _chunks = {}
    # Dictionary for BM25 objects per session/document context
    _bm25 = {}

    def __init__(self):
        self.qdrant = qdrant_client

    # ==========================================
    # MEMORY MANAGEMENT
    # ==========================================
    @classmethod
    async def sync_memory_state(cls, session_id: str):
        """Stateless recovery: Pulls ALL documents for this session from Qdrant if server restarts."""
        logger.warning(f"RAM is empty for {session_id}! Recovering all documents from Qdrant...")
        if not await qdrant_client.collection_exists(collection_name=settings.COLLECTION_NAME):
            return False
            
        records, _ = await qdrant_client.scroll(
            collection_name=settings.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
            ),
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        if not records:
            return False

        # Initialize session structures
        cls._chunks[session_id] = {}
        
        # Group chunks by their document_name natively
        for record in records:
            doc_name = record.payload.get("document_name", "unknown_document")
            chunk_text = record.payload["text"]
            orig_index = record.payload["original_index"]
            
            if doc_name not in cls._chunks[session_id]:
                cls._chunks[session_id][doc_name] = []
            
            cls._chunks[session_id][doc_name].append((orig_index, chunk_text))

        # Sort chunks back into original order and finalize the structure
        for doc_name in cls._chunks[session_id]:
            cls._chunks[session_id][doc_name].sort(key=lambda x: x[0])
            cls._chunks[session_id][doc_name] = [text for _, text in cls._chunks[session_id][doc_name]]
            
            # Cache individual BM25 for this document using multilingual tokenizer
            tokenized = [tokenize_multilingual(chunk) for chunk in cls._chunks[session_id][doc_name]]
            if session_id not in cls._bm25:
                cls._bm25[session_id] = {}
            cls._bm25[session_id][doc_name] = BM25Okapi(tokenized)

        logger.info(f"Successfully recovered {len(cls._chunks[session_id])} documents for session {session_id}.")
        return True

    @classmethod
    def update_index(cls, session_id: str, filename: str, new_chunks: list):
        """Appends a new document into the session memory instead of overwriting."""
        if session_id not in cls._chunks:
            cls._chunks[session_id] = {}
        if session_id not in cls._bm25:
            cls._bm25[session_id] = {}

        # Save chunks under the specific filename
        cls._chunks[session_id][filename] = new_chunks
        
        # Build BM25 for this specific document using multilingual tokenizer
        tokenized = [tokenize_multilingual(chunk) for chunk in new_chunks]
        cls._bm25[session_id][filename] = BM25Okapi(tokenized)
        
        logger.info(f"Memory index appended with '{filename}' ({len(new_chunks)} chunks) for {session_id}.")

    # ==========================================
    # SEARCH PIPELINE
    # ==========================================
    @trace_span(name="qdrant_hybrid_search", span_type=SpanType.RETRIEVER)
    async def execute_hybrid_search(self, original_query: str, enriched_query: str, expanded_queries: list, session_id: str, document_selector: str = "all"):
                """Runs hybrid retrieval filtering by session_id and conditionally by document_name."""
                
                # 1. Recovery Safety Check (FIXED: changed cls to self.__class__)
                if session_id not in self.__class__._chunks:
                    success = await self.sync_memory_state(session_id)
                    if not success:
                        raise ValueError("No documents found for this session. Please upload a document first.")

                # 2. Compile Context (Specific Document vs All Documents)
                active_chunks = []
                
                if document_selector == "all":
                    # Flatten all documents chunks into a single list for global search
                    for doc_name, chunks in self.__class__._chunks[session_id].items():
                        active_chunks.extend(chunks)
                    
                    # Dynamically compile BM25 for the global selection
                    tokenized_all = [tokenize_multilingual(chunk) for chunk in active_chunks]
                    bm25_engine = BM25Okapi(tokenized_all)
                    
                    # Setup Qdrant filter for just the session
                    qdrant_filter = models.Filter(
                        must=[models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))]
                    )
                else:
                    if document_selector not in self.__class__._chunks[session_id]:
                        raise ValueError(f"Document '{document_selector}' not found in active session.")
                        
                    active_chunks = self.__class__._chunks[session_id][document_selector]
                    bm25_engine = self.__class__._bm25[session_id][document_selector]
                    
                    # Setup Qdrant filter for session AND specific document
                    qdrant_filter = models.Filter(
                        must=[
                            models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id)),
                            models.FieldCondition(key="document_name", match=models.MatchValue(value=document_selector))
                        ]
                    )

                if not active_chunks:
                    return []

                # 3. Vector Search (Qdrant)
                query_embedding = await get_query_embedding(enriched_query.lower())
                response = await self.qdrant.query_points(
                    collection_name=settings.COLLECTION_NAME,
                    query=query_embedding.tolist(),
                    query_filter=qdrant_filter,
                    limit=settings.TOP_K_RESULTS
                )
                
                # If searching "all", indices map differently, so we extract raw text directly from Qdrant payloads
                vector_chunks = [hit.payload["text"] for hit in response.points]

                # 4. Keyword Search (BM25)
                bm25_chunks = []
                all_queries = expanded_queries + [original_query, enriched_query]
                for q in all_queries:
                    # Score against the multilingually tokenized query
                    scores = bm25_engine.get_scores(tokenize_multilingual(q))
                    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    for idx in ranked_indices[:3]:
                        if idx < len(active_chunks):
                            bm25_chunks.append(active_chunks[idx])
                bm25_chunks = list(dict.fromkeys(bm25_chunks))

                # 5. Numeric Boost Search
                numeric_chunks = []
                numbers_in_query = re.findall(r'\d+', original_query)
                if numbers_in_query:
                    for chunk in active_chunks:
                        if any(num in chunk for num in numbers_in_query):
                            numeric_chunks.append(chunk)
                numeric_chunks = numeric_chunks[:3]

                # 6. Hybrid Merge (Combining results smoothly)
                combined_chunks = []
                max_len = max(len(vector_chunks + numeric_chunks), len(bm25_chunks))
                for i in range(max_len):
                    if i < len(vector_chunks) and vector_chunks[i] not in combined_chunks:
                        combined_chunks.append(vector_chunks[i])
                    if i < len(bm25_chunks) and bm25_chunks[i] not in combined_chunks:
                        combined_chunks.append(bm25_chunks[i])
                    if i < len(numeric_chunks) and numeric_chunks[i] not in combined_chunks:
                        combined_chunks.append(numeric_chunks[i])
                    if len(combined_chunks) >= settings.TOP_K_RESULTS * 2:
                        break

                # 7. Cross-Encoder Re-Ranking
                if not combined_chunks:
                    return []
                    
                pairs = [[enriched_query, chunk] for chunk in combined_chunks]
                scores = reranker.predict(pairs)
                ranked_pairs = sorted(zip(scores, combined_chunks), reverse=True)
                
                return [chunk for score, chunk in ranked_pairs][:settings.TOP_K_RESULTS]