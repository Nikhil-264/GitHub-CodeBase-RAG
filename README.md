# GitHub Codebase RAG 🔍

An advanced, local **codebase Q&A assistant** that allows developers to ask questions about any GitHub repository using a stateful agent pipeline. 

The application utilizes **Repository-Scoped Hybrid Search (Vector + Lexical)** with **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Reranking**, **Corrective RAG (CRAG)**, **Self-RAG Reflection**, and **LangGraph** to coordinate agents. Chat history and repository bindings are preserved persistently in **PostgreSQL**.

---

## 🏗️ Architecture & Pipeline Flow

The application is structured into a stateful workflow managed by **LangGraph** with strict repository scoping and advanced verification steps:

```
                                [User Chat Query]
                                        │
                                        ▼
                              [Intent Classifier]
                                        │
                ┌───────────────────────┴───────────────────────┐
                │ (Codebase Context?)                           │ (No)
                ▼                                               ▼
     [Scoped Retrieval Agent]                          [Direct LLM Answer]
  (ChromaDB Filter + BM25 Repo)                                 │
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

1. **Ingestion & Scoping**: Clones a repository, scans it for 20+ programming languages, chunks code with **tree-sitter AST nodes** (or regex fallbacks), embeds chunks with repo metadata (`repo: name`) in **ChromaDB**, and builds a per-repo **BM25 lexical index** (`repos/bm25_{repo_name}.pkl`).
2. **Session-Repo Binding**: Binds chat sessions in **PostgreSQL** (`chat_sessions.repo_url`) so that querying in a session strictly filters vector search matches (`where={"repo": repo_name}`) and loads the target repository's BM25 index.
3. **Retrieve Gate (Self-RAG)**: Checks if the query actually needs codebase context. General programming questions or greetings bypass retrieval and route directly to the direct answer node.
4. **Retrieval & Rerank**: Executes intent-driven RRF hybrid search combining repo-filtered ChromaDB and BM25 matches, then prioritizes them using a Cross-Encoder reranker.
5. **Document Grader (CRAG)**: Evaluates each retrieved chunk's relevance. If all chunks are irrelevant, it rewrites the query and retries retrieval once. Otherwise, it filters out irrelevant chunks and proceeds.
6. **Answer Agent & Critique (Self-RAG)**: Generates a response which is graded for grounding (hallucination detection) and utility (question relevance). If it fails grounding/utility, it loops back to regenerate with a stricter factual-enforcement prompt (up to 1 retry).
7. **Persistent Memory**: Stores turn-by-turn chat history in **PostgreSQL** to inject conversational context into subsequent turns.

---

## ⚡ Features

* **Strict Repository Scoping**: Isolate chats to target repositories so vector search and BM25 queries never cross-contaminate chunks between different codebase ingestions.
* **Local Inference**: Completely private—runs locally using Ollama (`qwen2.5:3b` and `nomic-embed-text`) or Google Gemini (`gemini-2.5-flash`), with local Sentence-Transformers.
* **Corrective RAG (CRAG)**: Intelligent document grading that auto-corrects retrieval failures via query expansion/rewrites.
* **Self-RAG Reflection**: Dual LLM critic loops that check generated answers for hallucinations and relevance.
* **Strict Mode Toggle**: Exposes a `strict_mode` parameter (default `true`). Setting `strict_mode: false` bypasses critique and grader LLM round-trips for high-speed local responses.
* **Persistent Session Memory**: Sidebar session picker to switch past chats or create new repository-bound conversations in PostgreSQL.
* **Data Reset Utilities**: Clear all chats, ChromaDB vector collections, and cached repository indices via `python reset.py`, `POST /reset`, or the Streamlit sidebar reset button.
* **3-Step Evaluation Suite**: Quantitative evaluation harness measuring component performance, RAG Triad quality metrics via DeepEval, and end-to-end application latency.
* **LangSmith Tracing**: Full visual tracing integration to inspect step latencies, database inputs, and model parameters.

---

## 🛠️ Prerequisites

* **Python**: `>=3.11`
* **Docker Desktop**: For running the PostgreSQL database.
* **Ollama** or **Gemini API Key**: Installed locally or API configured in `.env`.
  * If using Ollama locally:
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
CHROMA_PATH=./chroma_db
REPOS_PATH=./repos
OLLAMA_BASE_URL=http://localhost:11434
EMBED_MODEL=nomic-embed-text
LLM_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your-gemini-key

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
*(Postgres maps container port 5432 to host port **`5435`** to prevent collisions with native PostgreSQL installs).*

### Step 2: Start the FastAPI Backend
Start the backend server on port `8080`:
```bash
python main.py
```
*(On startup, this will automatically initialize the Postgres tables `chat_sessions` and `chat_messages`).*

### Step 3: Start the Streamlit Frontend
In a new terminal window, activate the virtual environment and start the frontend on port `8888`:
```bash
python -m streamlit run frontend/streamlit_app.py --server.port 8888
```

Open your browser to `http://localhost:8888` to begin chat sessions and index code repositories!

---

## 🧹 Resetting & Clearing Data
To wipe all stored chat sessions, ChromaDB vector embeddings, and cached repository indices for a clean slate:

* **Via Terminal**:
  ```bash
  python reset.py
  ```
* **Via Web UI**: Click the **"🗑️ Clear All Data (Reset)"** button in the Streamlit sidebar.
* **Via API**: Send a `POST` request to `http://localhost:8080/reset`.

---

## 🧪 CLI Testing
To test repo cloning, AST chunking, vector embedding, and hybrid search directly from the command line:
```bash
python test_pipeline.py
```

---

## 📊 3-Step Unified Evaluation Suite

We provide a comprehensive 3-level evaluation harness to measure retrieval precision, pipeline correctness, generation quality, and latency performance.

### Running Evaluations

Ensure you have ingested a repository first (e.g. via `test_pipeline.py` or the web UI), then run:

```bash
# Run all evaluation levels:
python -m app.eval.eval_runner --level all

# Run specific evaluation levels:
python -m app.eval.eval_runner --level component
python -m app.eval.eval_runner --level pipeline
python -m app.eval.eval_runner --level app

# Run with a smaller sample size (for quick testing):
python -m app.eval.eval_runner --level all --sample 5
```

### Evaluation Hierarchy & Metrics

#### Level 1: Component-Level Evaluation (`component_eval.py`)
* **Intent Classifier**: Evaluates rule-based and LLM classification accuracy across query types (`code_search`, `explain`, `trace_flow`, `architecture`, `debug`).
* **Isolated Retrieval Comparison**: Compares standalone Vector search vs. standalone BM25 search across Precision@k, Recall@k, and MRR.
* **Cross-Encoder Reranker Impact**: Measures Precision@1 and MRR gain before vs. after reranking.
* **Document Grader (CRAG)**: Evaluates CRAG relevance classification accuracy against ground-truth code files.

#### Level 2: Pipeline-Level & RAG Triad Evaluation (`pipeline_eval.py` & `deepeval_judge.py`)
* **Retrieve Gate Accuracy (Self-RAG)**: Verifies whether the system correctly decides when codebase context is needed vs. when to direct-answer.
* **Self-RAG Critique Accuracy**: Validates grounding and utility critique node decisions.
* **RAG Triad Metrics (DeepEval / LLM Judge)**:
  * **Faithfulness**: Measures hallucination rate by verifying every claim in the answer against retrieved chunks.
  * **Answer Relevancy / Utility**: Measures whether the answer directly addresses the user question.
  * **Contextual Relevancy**: Evaluates the signal-to-noise ratio of retrieved code chunks.

#### Level 3: Application-Level E2E Evaluation (`app_eval.py`)
* **Strict Mode vs. Fast Mode Comparison**: Compares end-to-end latency, retrieval accuracy, and response quality when CRAG and Self-RAG loops are active (`strict_mode=True`) vs. bypassed (`strict_mode=False`).
* **G-Eval Assessment**: Evaluates answer quality on a 1-5 scale using custom LLM judge prompts.
* **CSV Logging**: Logs query-by-query latencies, retrieved files, intent classifications, and scores to [`repos/eval_results.csv`](file:///c:/Users/HP/Documents/Coding%20journeys/CV%20projects/GitHub%20Codebase%20RAG/repos/eval_results.csv).
