import logging
from typing import List
from src.adapters.hf_client import HFClient

logger = logging.getLogger(__name__)

# --- Mocking the SDK Response Objects ---
class MockTextEmbedding:
    def __init__(self, values: List[float]):
        self.values = values

class MockGenerationResponse:
    def __init__(self, text: str):
        self.text = text

# --- Mocking the SDK Models ---
class TextEmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = HFClient()
        logger.info(f"Initialized Mock TextEmbeddingModel targeting {model_name} (Routed to HF)")

    @classmethod
    def from_pretrained(cls, model_name: str):
        return cls(model_name)

    async def get_embeddings_async(self, texts: List[str]) -> List[MockTextEmbedding]:
        """Mimics the async embedding call of Vertex AI."""
        try:
            # Under the hood, this calls our resilient Hugging Face adapter
            hf_embeddings = await self.client.get_embeddings(texts)
            return [MockTextEmbedding(values=emb) for emb in hf_embeddings]
        except Exception as e:
            logger.error(f"Failed to get embeddings: {e}")
            raise

class GenerativeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = HFClient()
        logger.info(f"Initialized Mock GenerativeModel targeting {model_name} (Routed to HF)")

    async def generate_content_async(self, prompt: str) -> MockGenerationResponse:
        """Mimics the Vertex AI LLM generation."""
        try:
            generated_text = await self.client.generate_text(prompt)
            return MockGenerationResponse(text=generated_text)
        except Exception as e:
            logger.error(f"Failed to generate content: {e}")
            raise