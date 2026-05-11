import asyncio
import json
import logging
from tqdm import tqdm
from typing import List, Dict
from src.core.engine import RAGEngine
from src.utils.text_processor import RecursiveTextSplitter

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self, batch_size: int = 20, max_concurrency: int = 5):
        self.engine = RAGEngine()
        self.splitter = RecursiveTextSplitter()
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_batch(self, batch_texts: List[str]):
        """Processes a single batch: embeds and adds to FAISS."""
        async with self.semaphore:
            # 1. Get embeddings via our Mocked/HF-routed embedder
            embeddings_objs = await self.engine.embedder.get_embeddings_async(batch_texts)
            vectors = [emb.values for emb in embeddings_objs]
            
            # 2. Create metadata for each chunk
            metadata = [{"content": text, "source": "corpus_batch"} for text in batch_texts]
            
            # 3. Add to FAISS index
            self.engine.vector_store.add_vectors(vectors, metadata)

    async def run_ingestion(self, raw_data: List[str]):
        """
        Orchestrates the full ingestion of 1,000 paragraphs.
        """
        # Check if we already have data (Avoid redundant API calls/costs)
        if self.engine.vector_store.index.ntotal > 0:
            logger.info(f"Index already contains {self.engine.vector_store.index.ntotal} docs. Skipping ingestion.")
            return

        logger.info(f"Starting ingestion of {len(raw_data)} paragraphs...")
        
        # Prepare batches
        tasks = []
        for i in range(0, len(raw_data), self.batch_size):
            batch = raw_data[i : i + self.batch_size]
            tasks.append(self._process_batch(batch))

        # Run with a progress bar
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Ingesting Batches"):
            await f

        logger.info("✅ Ingestion complete. FAISS Index persisted.")