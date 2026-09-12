# Self-Correcting RAG & Evaluation Platform

A production-grade, configurable Retrieval-Augmented Generation (RAG) learning and evaluation platform. It enables hands-on experimentation with document ingestion, chunking strategies, dense vector search, neural reranking, high-throughput LLM synthesis via Groq (Llama 3.3 70B), and multi-metric factual and linguistic evaluation (NLI Faithfulness, Cosine Similarity, BERTScore, and Fluency). It also features persistent session logging in PostgreSQL and a dynamic code exporter that generates standalone Python pipelines.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
3. [API Keys & Environment Variables](#api-keys--environment-variables)
   - [Where to Put Real API Keys](#where-to-put-real-api-keys)
   - [Environment Variables Reference](#environment-variables-reference)
   - [Obtaining a Free Groq API Key](#obtaining-a-free-groq-api-key)
4. [How to Run the Application](#how-to-run-the-application)
   - [Prerequisites](#prerequisites)
   - [Step-by-Step Local Setup](#step-by-step-local-setup)
   - [Running with npm / Dev Server](#running-with-npm--dev-server)
5. [Complete User Guide](#complete-user-guide)
   - [Panel 1: Document Upload](#panel-1-document-upload)
   - [Panel 2: Interactive Pipeline Configuration](#panel-2-interactive-pipeline-configuration)
   - [Panel 3: Querying the Pipeline](#panel-3-querying-the-pipeline)
   - [Panel 4: Results & Multi-Metric Evaluation](#panel-4-results--multi-metric-evaluation)
   - [Panel 5: Session History](#panel-5-session-history)
   - [Panel 6: Dynamic Pipeline Code Exporter](#panel-6-dynamic-pipeline-code-exporter)
6. [Evaluation Metrics Explained](#evaluation-metrics-explained)
7. [REST API Endpoints](#rest-api-endpoints)
8. [Running Automated Tests](#running-automated-tests)
9. [Troubleshooting & FAQs](#troubleshooting--faqs)

---

## Key Features

- **PDF Ingestion & Page-Aware Extraction**: Upload any PDF document with extraction of page numbers and document metadata.
- **Configurable Text Chunking**: Adjust chunk size (100–2,000 characters) and overlap (0–500 characters) with strict validation.
- **Dense Vector Search**: Powered by FAISS (CPU) and Sentence-Transformers (`all-MiniLM-L6-v2` or `all-mpnet-base-v2`) with L2-normalized cosine similarity.
- **Cross-Encoder Neural Reranking**: Optional second-stage reranking via `ms-marco-MiniLM-L-6-v2` to compute query-chunk cross-attentive relevance.
- **Llama 3.3 70B Generation**: High-speed LLM inference via Groq API (`llama-3.3-70b-versatile`) conditioned strictly on retrieved context.
- **Multi-Metric Evaluation Suite**:
  - **NLI Faithfulness**: Context-entailment check to verify factual consistency and detect hallucinations.
  - **Cosine Semantic Similarity**: Semantic overlap between context and generated answer (with explicit caveat that similarity ≠ factual accuracy).
  - **BERTScore F1**: Deep contextual token-level matching.
  - **Linguistic Fluency**: Coherence, length, and sentence structure evaluation.
- **Latency Breakdown**: Granular millisecond timing for Retrieval, LLM Inference, and Evaluation stages.
- **Persistent Query History**: PostgreSQL storage via SQLAlchemy recording all configurations, questions, answers, evaluation scores, and latencies.
- **Dynamic Code Exporter**: Download customized, standalone Python scripts or an entire runnable ZIP project implementing the exact active pipeline.

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────┐
│                      Client Browser                         │
│        HTMX Reactive UI + Tailwind CSS + Lucide Icons       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP / JSON / HTML Partials
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Web Application                  │
│                     (Python 3.10+, Uvicorn)                 │
├─────────────────────────────────────────────────────────────┤
│ 1. PDF Parser (pypdf) -> Text & Page Boundaries             │
│ 2. Chunker (Sliding window with overlap)                    │
│ 3. Embedder (Sentence-Transformers: all-MiniLM-L6-v2)       │
│ 4. Vector Store (FAISS FlatIP Index)                        │
│ 5. Neural Reranker (Cross-Encoder ms-marco-MiniLM-L-6-v2)   │
│ 6. LLM Generator (Groq SDK: Llama 3.3 70B Versatile)        │
│ 7. Multi-Metric Evaluator (NLI, Cosine, BERTScore, Fluency) │
│ 8. Dynamic Code Generator (Jinja2 Code Templates)           │
└──────────────────────────────┬──────────────────────────────┘
                               │ SQLAlchemy ORM
┌──────────────────────────────▼──────────────────────────────┐
│                   PostgreSQL Database (rag_db)              │
│       Sessions, Pipeline Configs, Query Logs, Eval Scores   │
└─────────────────────────────────────────────────────────────┘
```

---

## API Keys & Environment Variables

### Where to Put Real API Keys

All secrets and environment configurations must be placed in a **`.env`** file in the root directory of the project (at the same level as `package.json` and `requirements.txt`).

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Open `.env` in any text editor and configure your keys:
   ```env
   # PostgreSQL Connection String
   DATABASE_URL="postgresql://postgres@localhost:5432/rag_db"

   # Groq API Key for Llama 3.3 70B Generation
   GROQ_API_KEY="gsk_your_actual_groq_api_key_here"

   # Optional: Hugging Face Token for Higher Download Rate Limits
   HF_TOKEN=""

   # Hosted App URL (Optional)
   APP_URL="http://localhost:3000"
   ```

### Environment Variables Reference

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `GROQ_API_KEY` | **Recommended** | `""` | Groq API key for live Llama 3.3 70B synthesis. If omitted, the pipeline still retrieves chunks, computes scores, and displays a graceful notice with retrieval results. |
| `DATABASE_URL` | **Required** | `postgresql://postgres@localhost:5432/rag_db` | PostgreSQL connection URL used to persist sessions, pipeline configurations, and query logs. |
| `HF_TOKEN` | Optional | `""` | Hugging Face user access token. Useful to avoid rate limits when downloading embedding and reranking models on initial boot. |
| `APP_URL` | Optional | `http://localhost:3000` | Base URL where the applet is hosted. |

### Obtaining a Free Groq API Key

1. Go to [Groq Console](https://console.groq.com/).
2. Sign in or register for an account.
3. In the sidebar, navigate to **API Keys**.
4. Click **Create API Key**, copy the key (starts with `gsk_`), and paste it into `.env`:
   ```env
   GROQ_API_KEY="gsk_..."
   ```
5. Restart the server or app to apply the key.

---

## How to Run the Application

### Prerequisites

- **Python 3.10+**
- **pip** and **virtualenv**
- **PostgreSQL 14+** (running locally or accessible via network)
- *(Optional)* **Node.js 18+** if managing via `npm` scripts

### Step-by-Step Local Setup

#### 1. Clone or Open the Project
```bash
cd /path/to/project
```

#### 2. Create and Activate a Python Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure PostgreSQL Database
Ensure your PostgreSQL server is active, and create the database `rag_db`:
```bash
# Using psql
psql -U postgres -c "CREATE DATABASE rag_db;"
```

If your PostgreSQL user has a password or runs on a custom port, update `DATABASE_URL` in `.env`:
```env
DATABASE_URL="postgresql://username:password@localhost:5432/rag_db"
```

The database tables (`sessions`, `configs`, `query_logs`, `evaluation_scores`) are automatically created on startup via SQLAlchemy ORM.

#### 5. Configure Your Real API Keys
Create your `.env` file as described in [Where to Put Real API Keys](#where-to-put-real-api-keys):
```bash
cp .env.example .env
# Edit .env with your favorite editor
```

#### 6. Start the Server
Run Uvicorn directly:
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 3000 --reload
```

Or via `npm`:
```bash
npm run dev
```

#### 7. Open the App in Your Browser
Visit [http://localhost:3000](http://localhost:3000) to access the interactive web interface.

---

## Complete User Guide

The web UI is organized into 6 intuitive panels designed for end-to-end RAG experimentation:

### Panel 1: Document Upload
- **How to use**: Drag and drop a PDF file into the upload zone or click to select a file.
- **What happens**: The document is uploaded, text is extracted page by page using `pypdf`, chunked according to your active settings, embedded into vector space, and indexed in an in-memory FAISS store.
- **Feedback**: Displays document name, page count, total chunk count, and processing status.

### Panel 2: Interactive Pipeline Configuration
Allows real-time tuning of RAG retrieval parameters:
- **Chunk Size**: Target character length per chunk (100 to 2,000).
- **Chunk Overlap**: Character overlap between adjacent chunks (0 to 500). *Validation enforces overlap < chunk size.*
- **Embedding Model**: Choose between `all-MiniLM-L6-v2` (fast, lightweight 384-d) and `all-mpnet-base-v2` (high-accuracy 768-d).
- **Top-K Retrieval**: Number of context chunks retrieved for generation (1 to 10).
- **Cross-Encoder Reranking**: Toggle two-stage retrieval. When enabled, retrieved candidates are re-scored using `ms-marco-MiniLM-L-6-v2`.
- Click **"Save Configuration"** to update your session and re-index the active document if chunk parameters changed.

### Panel 3: Querying the Pipeline
- Type a domain-specific question into the query input box.
- Press **Enter** or click **"Submit Query"**.
- The query triggers dense retrieval, optional CrossEncoder reranking, Groq LLM synthesis, and full multi-metric evaluation.

### Panel 4: Results & Multi-Metric Evaluation
Once a query is processed, Panel 4 displays:
1. **Generated Answer**: The contextual synthesis from Groq's Llama 3.3 70B model.
2. **Multi-Metric Raw Evaluation**:
   - **NLI Faithfulness**: Context entailment score (0.00 to 1.00).
   - **Cosine Similarity**: Semantic overlap score (with caution banner).
   - **BERTScore F1**: Token-level contextual match score.
   - **Fluency**: Linguistic coherence score.
3. **Latency Breakdown**: Visual breakdown showing exact execution times:
   - *Retrieval & Rerank Time*
   - *LLM Generation Time*
   - *Evaluation Time*
   - *Total End-to-End Latency*
4. **Retrieved Sources**: Each chunk used in the context prompt, showing:
   - Source page number
   - Chunk ID
   - Reranking indicator (if applicable)
   - Calculated relevance score
   - Full chunk text

### Panel 5: Session History
- Lists previously executed queries for your current session.
- Displays question, truncated answer, individual evaluation scores, and total latency.
- Stored durably in PostgreSQL for review and comparison across parameter changes.

### Panel 6: Dynamic Pipeline Code Exporter
- Select the pipeline components you wish to export:
  - `Document Parser (PDF & Chunker)`
  - `Dense Vector Retrieval (FAISS)`
  - `Groq Llama 3.3 Generation`
  - `Multi-Metric Evaluator`
- Choose export format:
  - **Single Script (`pipeline.py`)**: A clean, standalone Python script ready to run anywhere.
  - **Complete Project (`.zip`)**: A zip archive containing modular files (`parser.py`, `retriever.py`, `generator.py`, `evaluator.py`, `requirements.txt`, and `README.md`).
- Click **"Download Code"** to export.

---

## Evaluation Metrics Explained

| Metric | Range | Underlying Method | Meaning & Purpose |
| :--- | :---: | :--- | :--- |
| **NLI Faithfulness** | `0.0 – 1.0` | Natural Language Inference Entailment Probability | Measures whether the statements in the generated answer are strictly entailed by the retrieved context. High score indicates absence of hallucinations. |
| **Cosine Similarity** | `0.0 – 1.0` | Vector Dot Product of Embeddings | Measures semantic alignment between retrieved chunks and generated text. <br>⚠️ *Caution: High cosine similarity does NOT guarantee factual accuracy!* |
| **BERTScore F1** | `0.0 – 1.0` | Pairwise Token Cosine Similarities | Computes precision and recall of tokens in context versus answer using contextual embeddings, capturing semantic equivalence even with different phrasing. |
| **Fluency** | `0.0 – 1.0` | Heuristic / Perplexity Scoring | Evaluates grammatical correctness, sentence structure variation, and formatting quality. |

---

## REST API Endpoints

The backend exposes a full JSON REST API alongside the HTMX web interface:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Check service health, PostgreSQL connection, and Groq status |
| `POST` | `/api/upload` | Upload a PDF file via multipart form-data |
| `POST` | `/api/config` | Update active session RAG pipeline parameters |
| `POST` | `/api/query` | Execute retrieval, generation, and multi-metric evaluation |
| `GET` | `/api/history` | Retrieve query log history for the current session |
| `POST` | `/api/export` | Generate standalone Python code for the pipeline |

---

## Running Automated Tests

The platform includes comprehensive test suites covering unit tests, API tests, vector operations, chunking validation, code generation, and full end-to-end flows.

### Run the Complete Test Suite
```bash
python3 -m pytest -v
```

### Run Individual Test Modules
```bash
# Test API endpoints
python3 -m pytest -v tests/test_api.py

# Test text chunker & overlap constraints
python3 -m pytest -v tests/test_chunker.py

# Test FAISS vector store & indexing
python3 -m pytest -v tests/test_vector_store.py

# Test multi-metric evaluator
python3 -m pytest -v tests/test_evaluator.py

# Test dynamic code generator and zip exporter
python3 -m pytest -v tests/test_code_generator.py

# Test complete end-to-end user journey
python3 -m pytest -v tests/test_e2e.py
```

---

## Troubleshooting & FAQs

### 1. `[Groq API Key Required for Live Llama 3.3 70B Generation]`
- **Cause**: The `GROQ_API_KEY` environment variable is either unset or empty in `.env`.
- **Solution**: Obtain a free API key at [console.groq.com](https://console.groq.com/keys) and add it to `.env`:
  ```env
  GROQ_API_KEY="gsk_your_key_here"
  ```
  Restart the server.

### 2. Database Connection Error / Status Degraded
- **Cause**: PostgreSQL is not running or credentials in `DATABASE_URL` are incorrect.
- **Solution**:
  - Verify PostgreSQL is running: `sudo systemctl status postgresql`
  - Verify database exists: `psql -U postgres -c "\l"`
  - Verify `DATABASE_URL` in `.env` matches your local setup.

### 3. Rate Limits or Slow Hugging Face Downloads
- **Cause**: Downloading models (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`) anonymously.
- **Solution**: Generate a free Hugging Face read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and set `HF_TOKEN="hf_..."` in `.env`.

### 4. Overlap Must Be Less Than Chunk Size Error
- **Cause**: Configuring a chunk overlap value greater than or equal to chunk size.
- **Solution**: In Panel 2, ensure `chunk_overlap < chunk_size` (e.g., chunk size 500, overlap 100). The UI validates this automatically.

---

## License

This project is open-source and distributed under the MIT License.
