# Context-Aware RAG Engine: Strategy Benchmarking

A high-performance, asynchronous Retrieval-Augmented Generation (RAG) engine designed to benchmark traditional vector similarity against **Hypothetical Document Embeddings (HyDE)**. This system is engineered for dense technical corpora, utilizing an HNSW-indexed vector store for low-latency, high-precision retrieval.

## 🚀 Key Features

- **Dual-Strategy Retrieval**: Real-time comparison between baseline vector search (Strategy A) and AI-enhanced HyDE expansion (Strategy B).
- **HyDE Implementation**: Utilizes a Generative LLM to recast asymmetric queries into symmetric document-to-document retrieval tasks.
- **Asynchronous Pipeline**: Built with `asyncio` and `httpx` for non-blocking API interactions and batch embedding.
- **Persistent Vector Store**: FAISS-based HNSW index with local metadata serialization for optimized cold starts.
- **Automated Benchmarking**: Generates a detailed `retrieval_benchmark.md` report analyzing similarity scores and latency ROI.

## 🛠️ Tech Stack

- **Engine**: Python 3.11+
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
- **LLM Expansion**: Qwen 2.5 (OpenAI-compatible)
- **PDF Parsing**: PyPDF
- **Schema Validation**: Pydantic V2

---

## 📦 Installation & Setup

### 1. Clone and Navigate

```powershell
cd Context-aware-RAG-engine
2. Environment Setup
Create a virtual environment and install dependencies:

PowerShell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
3. Configuration
Create a .env file in the root directory and add your Hugging Face API token:

Code snippet
HF_TOKEN=your_hugging_face_token_here
🚦 Usage
Running the Engine
The engine operates in an interactive mode. It will automatically detect any .pdf file in the root directory for ingestion.

PowerShell
python main.py
Ingestion: On the first run, the system will chunk the PDF and generate embeddings.

Querying: Enter your questions in the live prompt.

Exit: Type exit or q to terminate the session and generate the final benchmark report.

Running Tests
To verify the mocking architecture and retrieval logic:

PowerShell
python -m pytest tests/unit/test_retrieval.py -v
📊 Benchmarking Strategy
The engine compares two distinct approaches to retrieval:

Strategy A: Traditional
Directly encodes the user query and searches the FAISS index. Best for keyword-heavy, technical lookups.

Strategy B: HyDE (AI-Enhanced)
Passes the query through a generative LLM to create a "hypothetical" answer, which is then used as the search vector. This bridges the semantic gap for conceptual or metaphorical queries.

Detailed results are saved to retrieval_benchmark.md after each session.

📂 Project Structure
Plaintext
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

<img width="564" height="569" alt="WhatsApp Image 2026-05-11 at 1 26 05 PM" src="https://github.com/user-attachments/assets/b566e3de-1ede-431b-8c53-624c24771f38" />


Developed by Tushar Sharma
```
