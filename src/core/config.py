from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API & Authentication
    HF_TOKEN: str = Field(..., description="Hugging Face API Token")
    
    # Model Endpoints (Using free inference endpoints)
    HF_EMBEDDING_URL: str = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
    HF_LLM_URL: str = "https://router.huggingface.co/v1/chat/completions"
    
    # RAG Hyperparameters
    CHUNK_SIZE: int = Field(default=500, description="Number of words per document chunk")
    CHUNK_OVERLAP: int = Field(default=50, description="Word overlap between chunks")
    TOP_K_RETRIEVAL: int = Field(default=3, description="Number of documents to retrieve")
    
    # Storage Paths
    INDEX_PATH: str = "data/storage/vector_store.index"
    METADATA_PATH: str = "data/storage/metadata.pkl"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Instantiate a global settings object to be imported across the app
settings = Settings() # type: ignore