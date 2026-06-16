# GitHub Codebase RAG 🔍

An advanced, local **codebase Q&A assistant** that allows developers to ask questions about any GitHub repository using a stateful agent pipeline. 

The application utilizes **Hybrid Search (Vector + Lexical)** with **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, and **LangGraph** to coordinate agents. Chat history is preserved persistently in **PostgreSQL**.

---

## 🏗️ Architecture & Pipeline Flow

The application is structured into a stateful workflow managed by **LangGraph**:

```
           [User Chat Query]
                   │
                   ▼
         [Intent Classification] (Intent Agent)
                   │
                   ▼
     [Hybrid Search (Vector + BM25)] (Retrieval Agent)
                   │
                   ▼
        [Cross-Encoder Rerank] (Reranker)
                   │
                   ▼
       [Context Mapping & AST] (Analysis Agent)
                   │
                   ▼
       [Response Generation] (Answer Agent via Ollama)
                   │
                   ▼
  [Postgres Message Storage & Response]
```

1. **Ingestion**: Clones a repository, scans it for 20+ programming languages, chunks code with **tree-sitter AST nodes** (or regex fallbacks), creates vector embeddings in **ChromaDB**, and builds a **BM25 lexical index** for keyword search.
2. **Retrieval**: Leverages the classified intent to run RRF-based hybrid retrieval combining ChromaDB cosine similarity and BM25 matches.
3. **Rerank**: Prioritizes candidates using a local sentence-transformers Cross-Encoder.
4. **Analysis & Generation**: Maps imports and dependencies across retrieved chunks, builds a context brief, and triggers Ollama (running `qwen2.5:3b` locally) to generate a response.
5. **History**: Stores chat message history in **PostgreSQL** to inject past conversational turns into subsequent prompts.

---

## ⚡ Features

* **Local Inference**: Completely private—runs locally using Ollama (`qwen2.5:3b` and `nomic-embed-text`) and local Sentence-Transformers.
* **Persistent Session Memory**: Sidebar session picker to reload past chats or create new ones backed by PostgreSQL.
* **Type Safety**: Fully annotated types satisfying strict compiler limits.
* **LangSmith Tracing**: Full visual tracing integration to inspect step latencies, database inputs, and model parameters.

---

## 🛠️ Prerequisites

* **Python**: `>=3.11`
* **Docker Desktop**: For running the PostgreSQL instance.
* **Ollama**: Installed and running locally.
  * Download the required models in your terminal:
    ```bash
    ollama pull qwen2.5:3b
    ollama pull nomic-embed-text
    ```

---

## 🚀 Setup & Installation

### 1. Clone the repository & Create Environment
```bash
git clone <this-repo-url>
cd github-codebase-rag
python -m venv .venv
# Activate virtualenv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r pyproject.toml
# Or if using uv:
uv pip install -r pyproject.toml
```

### 3. Set Environment Variables
Create a `.env` file in the project root:
```ini
# LangSmith Tracing (Optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT='https://api.smith.langchain.com'
LANGCHAIN_API_KEY='your-langsmith-key'
LANGCHAIN_PROJECT='github-codebase-rag'

# Database
DATABASE_URL=postgresql+asyncpg://rag_user:rag_password@localhost:5435/rag_chat

# Models & Paths
CHROMA_DB_PATH=./chroma_db
REPOS_PATH=./repos
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=qwen2.5:3b

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-base
RERANKER_BACKEND=cross_encoder
RERANKER_TOP_K=5
BM25_INDEX_PATH=./repos/bm25_index.pkl
```

---

## 🏃 Running the Application

### Step 1: Start PostgreSQL (Docker)
Ensure Docker Desktop is running, then start the database container:
```bash
docker compose up -d
```
*(Postgres will map container port 5432 to host port **`5435`** to prevent collisions with any native PostgreSQL installs).*

### Step 2: Start the FastAPI Backend
Start the backend server on port `8080`:
```bash
python main.py
```
*(On startup, this will automatically initialize the Postgres schemas `chat_sessions` and `chat_messages` if they do not exist).*

### Step 3: Start the Streamlit Frontend
In a new terminal window, activate the virtual environment and start the frontend on port `8888`:
```bash
python -m streamlit run frontend/streamlit_app.py --server.port 8888
```

Open your browser to `http://localhost:8888` to begin chat sessions and index code repositories!

---

## 🧪 CLI Testing
To quickly test the indexing, retrieval, and reranking modules directly from the command line without launching the APIs, run:
```bash
python test_pipeline.py
```
