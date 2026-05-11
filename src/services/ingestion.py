import logging
from src.core.engine import RAGEngine
from src.utils.pdf_loader import PDFLoader
from tqdm import asyncio as tqdm_asyncio
import asyncio

logger = logging.getLogger(__name__)

class IngestionService:
    def __init__(self, engine: RAGEngine):
        self.engine = engine

    async def run_ingestion(self, pdf_path: str):
        """Processes a PDF, chunks it, and embeds it into FAISS."""
        if self.engine.vector_store.index.ntotal > 0:
            logger.info("Index already contains data. Skipping ingestion.")
            return

        logger.info(f"📄 Loading PDF: {pdf_path}")
        raw_text = PDFLoader.extract_text(pdf_path)
        chunks = PDFLoader.get_chunks(raw_text)
        
        logger.info(f"Created {len(chunks)} chunks. Starting ingestion...")
        
        # Process in batches of 25 for API efficiency
        batch_size = 25
        tasks = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            tasks.append(self._process_batch(batch))

        for f in tqdm_asyncio.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Ingesting Batches"):
            await f

        self.engine.vector_store.save()
        logger.info("✅ Ingestion complete. PDF content is now searchable.")

    async def _process_batch(self, batch_texts: list[str]):
        # 1. Get the list of MockTextEmbedding objects
        embeddings_objs = await self.engine.embedder.get_embeddings_async(batch_texts)
        
        # 2. Extract the raw numerical vectors (.values) from the objects
        # This fixes the "Argument of type List[MockTextEmbedding]" error
        vectors = [emb.values for emb in embeddings_objs]
        
        # 3. Create metadata and add to store
        metadata = [{"text": t} for t in batch_texts]
        self.engine.vector_store.add_vectors(vectors, metadata)