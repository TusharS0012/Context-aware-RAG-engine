import logging
from typing import List, Dict
from src.api_mocks.vertex_ai import TextEmbeddingModel, GenerativeModel
from src.adapters.faiss_manager import FAISSManager

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        # We instantiate our Mocks as if we are writing production GCP code
        self.embedder = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        self.llm = GenerativeModel("gemini-1.5-pro-preview-0409")
        
        # Load our local Vector Store
        self.vector_store = FAISSManager()

    async def strategy_a_direct_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        STRATEGY A: Raw Vector Search. 
        Embeds the user's exact query and searches the index.
        """
        logger.info(f"Executing Strategy A for query: '{query}'")
        
        # 1. Embed Query
        embedding_response = await self.embedder.get_embeddings_async([query])
        query_vector = embedding_response[0].values
        
        # 2. Search FAISS
        results = self.vector_store.search(query_vector, top_k=top_k)
        return results

    async def strategy_b_hyde_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        STRATEGY B: AI-Enhanced Retrieval (HyDE).
        Uses an LLM to generate a hypothetical ideal document, then embeds THAT document.
        """
        logger.info(f"Executing Strategy B for query: '{query}'")
        
        # 1. Generate Hypothetical Document (Query Expansion)
        hyde_prompt = f"""
        Please write a brief, highly technical paragraph that directly answers the following question. 
        Write it as if it were a snippet from an official engineering documentation page.
        Question: {query}
        """
        try:
            llm_response = await self.llm.generate_content_async(hyde_prompt)
            hypothetical_doc = llm_response.text
            logger.info(f"HyDE Expansion generated: {hypothetical_doc[:100]}...")
            
            # 2. Embed the Hypothetical Document (NOT the raw query)
            embedding_response = await self.embedder.get_embeddings_async([hypothetical_doc])
            query_vector = embedding_response[0].values
            
            # 3. Search FAISS
            results = self.vector_store.search(query_vector, top_k=top_k)
            return results
            
        except Exception as e:
            logger.warning(f"Strategy B failed ({e}). Falling back to Strategy A.")
            # Senior Move: Never crash the pipeline if the LLM is down. Degrade gracefully.
            return await self.strategy_a_direct_search(query, top_k)