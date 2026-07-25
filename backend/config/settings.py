from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # ==========================================
    # 1. API Keys & Security
    # ==========================================
    GROQ_API_KEY: str  # No default value; forces it to be in the .env file
    GEMINI_API_KEY: str

    # ==========================================
    # 2. Database (Qdrant Cloud)
    # ==========================================
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    FRONTEND_URL: str = "http://localhost:5173"
    COLLECTION_NAME: str = "finaudit_agreements"
    DATABASE_URL: str

    # ==========================================
    # 3. AI Models (Groq)
    # ==========================================
    SYNTHESIS_MODEL: str = "llama-3.3-70b-versatile"
    ROUTER_MODEL: str = "llama-3.1-8b-instant"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Authentication ---
    SECRET_KEY: str 
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 7 days (60 * 24 * 7)
    
    # ==========================================
    # 4. RAG Retrieval Variables
    # ==========================================
    TOP_K_RESULTS: int = 3

    # ==========================================
    # Pydantic Configuration
    # ==========================================
    # This tells Pydantic to automatically look for a .env file 
    # and ignore any extra variables it finds in the system environment.
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

# Instantiate the settings so we can import this single object everywhere
settings = Settings()