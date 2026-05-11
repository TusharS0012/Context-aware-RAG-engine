from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseVectorStore(ABC):
    """
    Abstract interface for any Vector Database (FAISS, Pinecone, Vertex, etc.)
    """
    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict]) -> None:
        """Adds normalized vectors and their corresponding metadata to the store."""
        pass

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict]:
        """Searches the store for the top_k most similar vectors."""
        pass

class BaseEmbedder(ABC):
    """
    Abstract interface for Text Embedding Models.
    """
    @abstractmethod
    async def get_embeddings_async(self, texts: List[str]) -> List[Any]:
        """Returns embedding objects for a list of strings."""
        pass

class BaseLLM(ABC):
    """
    Abstract interface for Generative Language Models (used in Strategy B / HyDE).
    """
    @abstractmethod
    async def generate_content_async(self, prompt: str) -> Any:
        """Returns a generation response object for a given prompt."""
        pass