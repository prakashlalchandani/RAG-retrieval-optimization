import sys
import types

# Standard patch for Langchain/Ragas VertexAI dependency bug
dummy_chat = types.ModuleType("langchain_community.chat_models.vertexai")
dummy_chat.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = dummy_chat

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy 
from datasets import Dataset

# Swapping the evaluator to Google to bypass Groq's 'n=1' limitation
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config.settings import settings
from config.logger import logger

# Initialize models specifically for Ragas evaluation
eval_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=settings.GEMINI_API_KEY
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
        "contexts": [contexts] # This provides Ragas access to the document chunks
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
        
        # Safely extract scores handling potential list returns from Ragas
        try:
            faith_val = result["faithfulness"]
            # If Ragas returns a list [0.95], extract the first item. Otherwise, use the float directly.
            faith_score = float(faith_val[0]) if isinstance(faith_val, list) else float(faith_val)
        except (KeyError, TypeError, IndexError, ValueError):
            faith_score = 0.0
            
        try:
            rel_val = result["answer_relevancy"]
            rel_score = float(rel_val[0]) if isinstance(rel_val, list) else float(rel_val)
        except (KeyError, TypeError, IndexError, ValueError):
            rel_score = 0.0
        
        logger.info("\n" + "="*60)
        logger.info("📊 RAGAS LIVE EVALUATION SCORECARD 📊")
        logger.info(f"Query: {query}")
        logger.info("-" * 60)
        logger.info(f"Faithfulness       (0-1): {faith_score:.4f}")
        logger.info(f"Answer Relevancy   (0-1): {rel_score:.4f}")
        logger.info("="*60 + "\n")
    except Exception as e:
        logger.error(f"Error during Ragas evaluation: {e}")