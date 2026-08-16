# GitHub Codebase RAG 🔍

An advanced, local **codebase Q&A assistant** that allows developers to ask questions about any GitHub repository using a stateful agent pipeline. 

The application utilizes **Hybrid Search (Vector + Lexical)** with **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, and **LangGraph** to coordinate agents. Chat history is preserved persistently in **PostgreSQL**.

---

## 🏗️ Architecture & Pipeline Flow

The application is structured into a stateful workflow managed by **LangGraph** with advanced verification steps:

```
                                 [User Chat Query]
                                         │
                                         ▼
                               [Intent Classifier]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ (Codebase Context?)                           │ (No)
                 ▼                                               ▼
         [Retrieval Agent]                              [Direct LLM Answer]
                 │                                               │
                 ▼                                               │
             [Reranker]                                          │
                 │                                               │
                 ▼                                               │
        [Document Grader] (CRAG)                                 │
                 │                                               │
                 ├──────────────────────────────┐ (Irrelevant)   │
                 ▼ (Relevant Chunks Found)      ▼                │
         [Analysis Agent]               [Query Rewrite]          │
                 │                              │                │
                 ▼                              └─► (Retry Once) │
          [Answer Agent] ◄──────────────┐                        │
                 │                      │ (Regenerate)           │
                 ▼                      │                        │
         [Self-RAG Critique] ───────────┘                        │
                 │                                               │
                 └──────────────────────┬────────────────────────┘
                                        ▼
                              [Response & Postgres DB]
```

1. **Ingestion**: Clones a repository, scans it for 20+ programming languages, chunks code with **tree-sitter AST nodes** (or regex fallbacks), creates vector embeddings in **ChromaDB**, and builds a **BM25 lexical index** for keyword search.
2. **Retrieve Gate (Self-RAG)**: Checks if the query actually needs codebase context. If it's a general coding question or greeting, it bypasses retrieval and routes directly to the direct answer node.
3. **Retrieval & Rerank**: Leverages the classified intent to run RRF-based hybrid retrieval combining ChromaDB and BM25 matches, then prioritizes them using a Cross-Encoder.
4. **Document Grader (CRAG)**: Evaluates each retrieved chunk's relevance to the question. If all chunks are irrelevant, it rewrites the query and retries retrieval once. Otherwise, it filters out irrelevant chunks and proceeds.
5. **Answer Agent & Critique (Self-RAG)**: Generates a response which is graded for grounding (hallucination detection) and utility (question relevance). If it fails grounding/utility, it loops back to regenerate with a stricter factual-enforcement prompt (up to 1 retry).
6. **History**: Stores chat message history in **PostgreSQL** to inject past conversational turns into subsequent prompts.

---

## ⚡ Features

* **Local Inference**: Completely private—runs locally using Ollama (`qwen2.5:3b` and `nomic-embed-text`) and local Sentence-Transformers.
* **Corrective RAG (CRAG)**: Intelligent document grading that auto-corrects retrieval failures via query expansion/rewrites.
* **Self-RAG Reflection**: Dual LLM critic loops that check generated answers for hallucinations and relevance to prevent incorrect responses.
* **Strict Mode Toggle**: Exposes a `strict_mode` parameter (default `true`) in chat endpoints. Setting `strict_mode: false` bypasses the critique and grader LLM round-trips for high-speed local queries.
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

---

## 📊 Evaluation Harness
To measure the retrieval and generation quality of the system quantitatively, we provide an evaluation suite.

### Running Evaluations
1. Make sure you have ingested a repository first (e.g. via `test_pipeline.py` or the web UI).
2. Run the evaluation runner:
```bash
python -m app.eval.eval_runner
```

This will run a golden test dataset of **24 complex queries** covering all 5 intents (`code_search`, `explain`, `trace_flow`, `architecture`, `debug`) targeting this codebase, and output:
* **Retrieval Metrics**: Precision@1/3/5, Recall@1/3/5, and Mean Reciprocal Rank (MRR) by comparing the file paths of retrieved code chunks with the expected ground-truth files.
* **Generation Metrics**: Faithfulness (groundedness/hallucination checks) and Utility (question relevance) using local LLM judges.
* **Detailed Logs**: Saves query-by-query latency and score details in `repos/eval_results.csv` for inspection.

