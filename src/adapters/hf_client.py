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
        """
        Generic POST handler with automatic resilience.
        The @retry_on_api_limit decorator handles backoff for us.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url, 
                headers=self.headers, 
                json=payload
            )
            # Triggers retry logic if the API returns 429 or other errors
            response.raise_for_status()
            return response.json()

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Calls the embedding model API.
        Uses the 'feature-extraction' pipeline to get raw vectors for FAISS.
        """
        payload = {"inputs": texts, "options": {"wait_for_model": True}}
        return await self.post_request(settings.HF_EMBEDDING_URL, payload)

    async def generate_text(self, prompt: str) -> str:
        """
        Calls the LLM for query expansion (Strategy B).
        Uses the modern v1/chat/completions (OpenAI-compatible) endpoint.
        """
        payload = {
            # Qwen 2.5 is un-gated and highly responsive on the HF router
            "model": "Qwen/Qwen2.5-7B-Instruct", 
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "stream": False
        }
        
        result = await self.post_request(settings.HF_LLM_URL, payload)
        
        # Parses the OpenAI-standard response structure
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        
        # Fallback for unexpected response structures
        return str(result)