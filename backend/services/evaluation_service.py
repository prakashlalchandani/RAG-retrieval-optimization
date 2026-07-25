import sys
import types
import asyncio
from typing import Dict, Any

# Bypassing the broken VertexAI dependency
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy 
from datasets import Dataset
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.outputs import ChatResult

from config.settings import settings
from config.logger import logger


# --- PERMANENT FIX 1: Concurrent 'n' Simulation for Groq ---
class SafeChatGroq(ChatGroq):
    """
    Groq API strictly forbids the 'n' parameter. Ragas requires 'n' for answer_relevancy.
    This wrapper intercepts 'n', runs concurrent requests to Groq, and perfectly 
    simulates the expected behavior by merging the ChatResults.
    """
    def _extract_n(self, kwargs: Dict[str, Any]) -> int:
        n = kwargs.pop("n", 1)
        # Langchain sometimes nests 'n' inside model_kwargs during bind()
        if "model_kwargs" in kwargs and "n" in kwargs["model_kwargs"]:
            n = kwargs["model_kwargs"].pop("n")
        return n

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        n = self._extract_n(kwargs)
        final_result = super()._generate(messages, stop, run_manager, **kwargs)
        
        # Sequentially generate remaining if n > 1 (Sync mode)
        for _ in range(n - 1):
            extra_result = super()._generate(messages, stop, run_manager, **kwargs)
            final_result.generations.extend(extra_result.generations)
        return final_result
        
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        n = self._extract_n(kwargs)
        
        # Concurrently generate 'n' requests (Async mode)
        tasks = [super()._agenerate(messages, stop, run_manager, **kwargs) for _ in range(n)]
        results = await asyncio.gather(*tasks)
        
        # Merge all generations into the first ChatResult
        final_result = results[0]
        for res in results[1:]:
            final_result.generations.extend(res.generations)
            
        return final_result
# -------------------------------------------------------------


# Initialize models using our new wrapper
eval_llm = SafeChatGroq(
    model=settings.ROUTER_MODEL, 
    temperature=0,
    api_key=settings.GROQ_API_KEY
)

eval_embeddings = GoogleGenerativeAIEmbeddings(
    model=settings.EMBEDDING_MODEL,
    google_api_key=settings.GEMINI_API_KEY
)

def run_ragas_evaluation(query: str, answer: str, contexts: list):
    """
    Runs reference-free Ragas evaluation synchronously. 
    FastAPI BackgroundTasks executes this in a separate thread.
    """
    data = {
        "question": [query],
        "answer": [answer],
        "contexts": [contexts]
    }
    dataset = Dataset.from_dict(data)
    
    try:
        logger.info(f"Initiating RAGAS evaluation for query: '{query}'...")
        
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False 
        )
        
        # --- PERMANENT FIX 2: Safe Result Extraction ---
        # Convert EvaluationResult directly to a dict to bypass the 
        # class's broken __contains__ iteration that causes KeyError: 0
        scores = dict(result)
        faith_score = scores.get("faithfulness", 0.0)
        rel_score = scores.get("answer_relevancy", 0.0)
        
        logger.info("\n" + "="*60)
        logger.info("📊 RAGAS LIVE EVALUATION SCORECARD 📊")
        logger.info(f"Query: {query}")
        logger.info("-" * 60)
        logger.info(f"Faithfulness       (0-1): {faith_score:.4f}")
        logger.info(f"Answer Relevancy   (0-1): {rel_score:.4f}")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Ragas Evaluation Failed: {e}", exc_info=True)