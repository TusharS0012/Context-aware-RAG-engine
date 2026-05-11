import httpx
from typing import List, Any
from src.core.config import settings
from src.utils.resilience import retry_on_api_limit

class HFClient:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"}
        self.timeout = httpx.Timeout(30.0)

    @retry_on_api_limit()
    async def post_request(self, url: str, payload: Any) -> Any:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url, 
                headers=self.headers, 
                json=payload
            )
            # This triggers the retry logic if status is 4xx or 5xx
            response.raise_for_status()
            return response.json()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Calls the embedding model API."""
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        return await self.post_request(settings.HF_EMBEDDING_URL, payload)

    async def generate_text(self, prompt: str) -> str:
        """Calls the LLM for query expansion (Strategy B) using the modern Chat API."""
        payload = {
            # Using Zephyr as it is an un-gated, highly capable model for RAG
            "model": "HuggingFaceH4/zephyr-7b-beta", 
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        result = await self.post_request(settings.HF_LLM_URL, payload)
        
        # Parse the OpenAI-compatible response structure
        return result['choices'][0]['message']['content']