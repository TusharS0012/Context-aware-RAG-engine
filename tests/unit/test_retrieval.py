import pytest
import asyncio
from src.core.engine import RAGEngine
from src.api_mocks.vertex_ai import TextEmbeddingModel, GenerativeModel

# --- Fixtures ---

@pytest.fixture
def engine():
    """Provides an instance of the RAGEngine for testing."""
    return RAGEngine()

@pytest.fixture(autouse=True)
def setup_test_index(engine):
    """
    Automatically injects a dummy document into the FAISS index 
    before each test to ensure search functions have data to retrieve.
    """
    dummy_vector = [0.1] * 384  # 384 matches the MiniLM-L6-v2 dimension
    dummy_metadata = [{"content": "This is a test document about system scaling.", "source": "test"}]
    engine.vector_store.add_vectors([dummy_vector], dummy_metadata)
    yield

# --- Tests for the SDK Mocking Requirement ---

@pytest.mark.asyncio
async def test_mock_vertex_embedding():
    """Verifies that our SDK mock correctly routes to Hugging Face and formats the response."""
    model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
    
    embeddings = await model.get_embeddings_async(["How do we scale databases?"])
    
    # Check SDK structure compliance
    assert len(embeddings) == 1
    assert hasattr(embeddings[0], 'values')
    assert len(embeddings[0].values) == 384

@pytest.mark.asyncio
async def test_mock_vertex_generation():
    """Verifies the GenerativeModel mock returns the expected object structure."""
    model = GenerativeModel("gemini-1.5-pro-preview-0409")
    
    response = await model.generate_content_async("What is sharding?")
    
    # Check SDK structure compliance
    assert hasattr(response, 'text')
    assert isinstance(response.text, str)
    assert len(response.text) > 0

# --- Tests for the Retrieval Logic ---

@pytest.mark.asyncio
async def test_strategy_a_direct_search(engine):
    """Verifies the baseline vector search executes without errors."""
    query = "Explain system scaling."
    results = await engine.strategy_a_direct_search(query, top_k=1)
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert "score" in results[0]
    assert "content" in results[0]

@pytest.mark.asyncio
async def test_strategy_b_hyde_search(engine):
    """Verifies the AI-Enhanced retrieval path completes successfully."""
    query = "Explain system scaling."
    results = await engine.strategy_b_hyde_search(query, top_k=1)
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert "score" in results[0]
    assert "content" in results[0]