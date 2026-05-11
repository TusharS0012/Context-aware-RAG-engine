import os
import faiss
import pickle
import numpy as np
import logging
from typing import List, Dict
from src.core.interfaces import BaseVectorStore
from src.core.config import settings

logger = logging.getLogger(__name__)

class FAISSManager(BaseVectorStore):
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize the FAISS HNSW Index.
        embedding_dim: 384 is the default for all-MiniLM-L6-v2.
        """
        self.index_path = settings.INDEX_PATH
        self.metadata_path = settings.METADATA_PATH
        self.embedding_dim = embedding_dim
        
        # Metadata storage: Maps FAISS integer IDs to document dictionaries
        self.metadata: Dict[int, Dict] = {}
        
        # Load existing index if it exists (Cold Start Optimization)
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self._load_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        """Creates a new HNSW index optimized for Cosine Similarity."""
        logger.info("Creating new FAISS HNSW index...")
        # M=32 is the number of bi-directional links created for every new element.
        # It's a great balance between memory consumption and search speed.
        self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        
        # Ensure the storage directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

    def _load_index(self):
        """Loads the index and metadata from disk."""
        logger.info(f"Loading existing FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        logger.info(f"Loaded {self.index.ntotal} vectors from disk.")

    def _normalize(self, vectors: List[List[float]]) -> np.ndarray:
        """Applies L^2 normalization to vectors to enable Cosine Similarity."""
        np_vectors = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(np_vectors)
        return np_vectors

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict]):
        """Normalizes and adds vectors to the index, then persists to disk."""
        if not vectors:
            return

        if len(vectors[0]) != self.embedding_dim:
            raise ValueError(f"Expected vector dimension {self.embedding_dim}, got {len(vectors[0])}")

        np_vectors = self._normalize(vectors)
        
        # Track the starting ID for this batch
        start_id = self.index.ntotal
        self.index.add(np_vectors) # type: ignore
        
        # Store metadata mapping
        for i, meta in enumerate(metadata):
            self.metadata[start_id + i] = meta
            
        self._save_index()
        logger.info(f"Added {len(vectors)} vectors. Total vectors: {self.index.ntotal}")

    def _save_index(self):
        """Persists the index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)

    def search(self, query_vector: List[float], top_k: int = 3) -> List[Dict]:
        """Searches the index for the most similar vectors."""
        if self.index.ntotal == 0:
            logger.warning("Search attempted on an empty index.")
            return []

        # Query must also be L^2 normalized
        np_query = self._normalize([query_vector])
        
        # Distances represent the cosine similarity score (closer to 1.0 is better)
        distances, indices = self.index.search(np_query, top_k) # type: ignore
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 means FAISS couldn't find enough neighbors
                result_meta = self.metadata[idx].copy()
                result_meta['score'] = float(distances[0][i])
                results.append(result_meta)
                
        return results