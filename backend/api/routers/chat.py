import uuid
import asyncio
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

# --- AgentLens SDK Imports ---
from agentlens_sdk.context import set_current_trace_id, set_current_span_id
from agentlens_sdk.instrumentation import span_buffer, trace_span
from agentlens_sdk.models.model import Trace, StatusCode, SpanType
from agentlens_sdk.client import send_trace_to_agentlens

from config.database import get_db
from config.logger import logger
from services.generation_service import GenerationService
from services.retrieval_service import RetrievalService
from services.evaluation_service import run_ragas_evaluation
import models.model as models
from config.auth import get_current_user

router = APIRouter()

# --- 1. Agent Wrapper (Yeh saare internal steps ko ek block me trace karega) ---
@trace_span(name="FinAI_Core_Agent", span_type=SpanType.AGENT)
async def run_finai_pipeline(query, document_selector, session_id, gen_service, retrieval_service, db):
    # PHASE 1: ROUTING
    routing_decision = await gen_service.route_query(query)
    if routing_decision["route"] == "chat":
        await gen_service.save_message_to_db(db, session_id, "user", query)
        await gen_service.save_message_to_db(db, session_id, "ai", routing_decision["chat_response"])
        return routing_decision["chat_response"], [], "Fast Chat Agent", None
    
    # PHASE 2: EXPANSION
    legal_synonyms = await gen_service.expand_query_with_synonyms(query)
    enriched_search_query = f"{query} {legal_synonyms}"
    expanded_queries = await gen_service.generate_multi_queries(query) 

    # PHASE 3: RETRIEVAL (Trace Span will capture this)
    best_chunks = await retrieval_service.execute_hybrid_search(
        original_query=query, enriched_query=enriched_search_query,
        expanded_queries=expanded_queries, session_id=session_id, document_selector=document_selector
    )

    # PHASE 4: SYNTHESIS (Trace Span will capture this)
    final_answer = await gen_service.generate_answer(query, best_chunks, session_id, db)
    
    return final_answer, best_chunks, "RAG Pipeline", legal_synonyms

# --- 2. Main API Endpoint ---
@router.get("/search")
async def search_query(
    query: str,
    background_tasks: BackgroundTasks,  # <--- Added BackgroundTasks
    document_selector: str = "all",
    session_id: str = "default_user",
    gen_service: GenerationService = Depends(),
    retrieval_service: RetrievalService = Depends(),
    db: AsyncSession = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    if not session_id.startswith(f"{current_user.id}_"):
        raise HTTPException(status_code=403, detail="Invalid session_id for current user")
    
    # --- AgentLens Trace Setup ---
    trace_id = str(uuid.uuid4())
    set_current_trace_id(trace_id)
    set_current_span_id(None)
    span_buffer.clear() # When request comes make sure to clear old data
    start_time = datetime.now(timezone.utc)
    status = StatusCode.SUCCESS
    
    try:
        final_answer, best_chunks, routed_via, synonyms = await run_finai_pipeline(
            query, document_selector, session_id, gen_service, retrieval_service, db
        )
        
        # <--- CHANGE 2: Queue the Ragas evaluation asynchronously
        if best_chunks:
            background_tasks.add_task(
                run_ragas_evaluation,
                query=query,
                answer=final_answer,
                contexts=best_chunks
            )
        # --------------------------------------------------------
        
        response_data = {
            "query": query, "answer": final_answer,
            "sources_used": best_chunks, "routed_via": routed_via,
            "synonyms_generated": synonyms
        }
        
    except Exception as e:
        logger.error(f"❌ [QUERY FAILED]: {str(e)}", exc_info=True)
        status = StatusCode.ERROR
        response_data = {"error": str(e)}
    
    # --- Send Data to Cloud ---
    trace = Trace(
        trace_id=trace_id,
        name="FinAI_Search_Query",
        status=status,
        start_time=start_time,
        end_time=datetime.now(timezone.utc),
        spans=span_buffer.copy() 
    )
    
    # Non-blocking upload (User ko API response ka wait nahi karna padega)
    asyncio.create_task(send_trace_to_agentlens(trace))
    
    if status == StatusCode.ERROR:
        raise HTTPException(status_code=500, detail=response_data["error"])
        
    return response_data