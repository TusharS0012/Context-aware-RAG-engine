# Context-Aware RAG Engine: Strategy Benchmarking
A high-performance, asynchronous Retrieval-Augmented Generation (RAG) engine designed to benchmark traditional vector similarity against Hypothetical Document Embeddings (HyDE). This system is engineered for dense technical corpora, utilizing an HNSW-indexed vector store for low-latency, high-precision retrieval.

## 🚀 Key Features
**Dual-Strategy Retrieval:** Real-time comparison between baseline vector search (Strategy A) and AI-enhanced HyDE expansion (Strategy B).

**HyDE Implementation:** Utilizes a Generative LLM to recast asymmetric queries into symmetric document-to-document retrieval tasks.

**Asynchronous Pipeline:** Built with asyncio and httpx for non-blocking API interactions and batch embedding.

**Persistent Vector Store:** FAISS-based HNSW index with local metadata serialization for optimized cold starts.

**Automated Benchmarking:** Generates a detailed retrieval_benchmark.md report analyzing similarity scores and latency ROI.

## 🛠️ Tech Stack
Engine: Python 3.11+

Vector Store: FAISS (Facebook AI Similarity Search)

Embeddings: sentence-transformers/all-MiniLM-L6-v2 (via Hugging Face)

LLM Expansion: Qwen 2.5 (OpenAI-compatible)

PDF Parsing: PyPDF

Schema Validation: Pydantic V2

## 📂 Project Structure
```Plaintext
├── src/
│   ├── adapters/       # FAISS & API Mocks
│   ├── core/           # RAG Logic & Engine
│   ├── services/       # Ingestion & Evaluation
│   └── utils/          # PDF Loading & Cleaning
├── data/
│   └── storage/        # Persistent FAISS Index
├── tests/              # Pytest Unit Tests
├── main.py             # Interactive Entry Point
└── .env                # API Credentials
```

## 📦 Installation & Setup
**1. Clone and Navigate**
```PowerShell
cd Context-aware-RAG-engine
```
**2. Environment Setup**

Create a virtual environment and install dependencies:

```PowerShell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
**3. Configuration**

Create a .env file in the root directory and add your Hugging Face API token:

```Code snippet
HF_TOKEN=your_hugging_face_token_here
HF_EMBEDDING_URL=https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction
HF_LLM_URL=https://router.huggingface.co/v1/chat/completions
```
## 🚦 Usage
**Running the Engine**

The engine operates in an interactive mode. It will automatically detect any .pdf file in the root directory for ingestion.

```PowerShell
python main.py
```
Ingestion: On the first run, the system will chunk the PDF and generate embeddings.

Querying: Enter your questions in the live prompt.

Exit: Type exit or q to terminate the session and generate the final benchmark report.

Running Tests
To verify the mocking architecture and retrieval logic:

```PowerShell
python -m pytest tests/unit/test_retrieval.py -v
```
## 📊 Benchmarking Strategy
The engine compares two distinct approaches to retrieval:

**Strategy A:** Traditional: Directly encodes the user query and searches the FAISS index. Best for keyword-heavy, technical lookups.

**Strategy B: HyDE (AI-Enhanced):** Passes the query through a generative LLM to create a "hypothetical" answer, which is then used as the search vector. This bridges the semantic gap for conceptual or metaphorical queries.

Detailed results are saved to retrieval_benchmark.md after each session.

# Final Analysis: The Benchmarking Verdict 
These results provide a textbook demonstration of the trade-offs between Keyword-Dense Retrieval and Semantic Expansion. By comparing these two queries, we can see exactly where Strategy B (HyDE) justifies its "latency tax" and where Strategy A (Traditional) remains the efficiency king.
### 1. The "Keyword Overlap" Effect (Query 1)
In the first query regarding encoder-decoder architecture, Strategy A actually outperformed Strategy B in similarity score (0.54 vs. 0.52) while being nearly 5x faster.
**Why?**
 The terminology used in the query was highly technical and likely appeared verbatim in the PDF.The Lesson: When a user knows exactly what they are looking for and uses the "correct" technical jargon, the LLM expansion in Strategy B can actually introduce "semantic noise," slightly diluting the vector's precision.
### 2. The "Conceptual Bridge" Triumph (Query 2)
The second query regarding HyDE as a 'filter' is where Strategy B showed its true power, crushing Strategy A with a score of 0.4270 vs. 0.1778.
**Why?** 
The user used the word "filter," which is a conceptual metaphor. The paper itself uses terms like "lossy compressor" or "dense bottleneck." Strategy A failed because it couldn't find a literal keyword match for "filter" in a meaningful context. Strategy B's LLM expansion understood the concept of filtering and generated a hypothetical document that used the paper's actual vocabulary, successfully bridging the semantic gap.

## 💡 Executive Summary of Findings
| Metric | Strategy A (Traditional) | Strategy B (HyDE) |
|:--------|:------:|:------:|
| Best for | Fact-finding with specific keywords. | Abstract concepts or "Human-speak" queries. |
| Speed | High Performance (<800ms) | Resource Heavy (>2500ms) |
| Realiability | Consistent but "literal-minded." | High-quality retrieval but prone to expansion drift. |
