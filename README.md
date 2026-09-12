# 🧠 RagLab — Configurable RAG & Multi-Metric Evaluation Platform

> An end-to-end Retrieval-Augmented Generation (RAG) platform with configurable retrieval, optional reranking, LLM generation, multi-metric evaluation, and cloud deployment.

<p align="center">

🔗 **[GitHub Repository](https://github.com/sharathop/RagLab)**
🚀 **[Live Deployment](http://16.171.208.210:8000)**

</p>

---

## 🚀 Overview

**RagLab** is an end-to-end RAG platform designed to demonstrate the complete lifecycle of a document-grounded LLM application.

Users can upload PDF documents, configure the RAG pipeline, ask questions against the uploaded content, inspect retrieved sources, and evaluate generated answers using multiple independent metrics.

The project focuses on making the RAG pipeline **configurable, observable, and evaluatable** rather than treating the LLM as a black box.

---

## ✨ Key Features

* 📄 PDF document ingestion
* ✂️ Configurable chunk size and overlap
* 🧠 Multiple embedding models
* 🔎 FAISS vector similarity search
* 🔢 Configurable Top-K retrieval
* 🎯 Optional CrossEncoder reranking
* 🤖 OpenAI `gpt-oss-120b` LLM integration
* 🛡️ NLI-based faithfulness evaluation
* 📐 Cosine similarity evaluation
* 🧬 BERTScore evaluation
* ⏱️ Latency breakdown
* 📚 Retrieved source inspection
* 🗄️ PostgreSQL persistence
* 📜 Query/history tracking
* ⚙️ Runtime RAG configuration
* 📦 RAG pipeline export
* 🐳 Dockerized deployment
* ☁️ AWS EC2 deployment

---

# 🔄 RAG Pipeline

```text
📄 PDF Upload
      ↓
📝 Text Extraction
      ↓
✂️ Chunking
      ↓
🧠 Embedding Generation
      ↓
🗂️ FAISS Vector Index
      ↓
❓ User Question
      ↓
🔢 Question Embedding
      ↓
🔎 Similarity Search
      ↓
📚 Top-K Relevant Chunks
      ↓
🎯 Optional CrossEncoder Reranking
      ↓
🧾 Context Construction
      ↓
🤖 OpenAI GPT-OSS 120B
      ↓
💬 Generated Answer
      ↓
🧪 Multi-Metric Evaluation
      ↓
📊 Results + Sources + Latency
```

---

# 🏗️ System Architecture

```text
                         👤 User
                           │
                           ▼
                  ┌─────────────────┐
                  │    Frontend     │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     FastAPI     │
                  │     Backend     │
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
       📄 PDF Flow    ⚙️ Config      🗄️ PostgreSQL
             │             │             │
             ▼             │             │
        Text Extraction    │             │
             │             │             │
             ▼             │             │
          Chunking         │             │
             │             │             │
             ▼             │             │
        Embeddings         │             │
             │             │             │
             ▼             │             │
          FAISS ◄──────────┘             │
             │                            │
             ▼                            │
       Top-K Retrieval                    │
             │                            │
             ▼                            │
     🎯 Optional Reranking                │
             │                            │
             ▼                            │
     🤖 GPT-OSS 120B                      │
             │                            │
             ▼                            │
      🧪 Evaluation ◄─────────────────────┘
             │
             ▼
      📊 Results & Sources
```

---

# 📄 1. PDF Document Ingestion

Users can upload PDF documents through the application.

The document processing pipeline performs:

1. PDF upload
2. Text extraction
3. Configurable chunking
4. Embedding generation
5. FAISS index creation

```text
PDF
 ↓
Text
 ↓
Chunks
 ↓
Embeddings
 ↓
FAISS Index
```

The resulting vector index is then used for semantic retrieval.

---

# ✂️ 2. Configurable Chunking

RagLab allows the user to configure how documents are split into chunks.

### Supported parameters

| Parameter     |    Range |
| ------------- | -------: |
| Chunk Size    | 100–2000 |
| Chunk Overlap |    0–500 |

Example:

```text
Chunk Size    = 500
Chunk Overlap = 100
```

### Why configurable chunking?

Different documents benefit from different chunk sizes.

Small chunks can provide:

* 🎯 More precise retrieval
* 📌 Less irrelevant context

Large chunks can provide:

* 📚 More surrounding context
* 🔗 Better preservation of related information

RagLab allows these parameters to be changed without modifying the source code.

---

# 🧠 3. Embedding Models

RagLab currently supports:

```text
all-MiniLM-L6-v2
all-mpnet-base-v2
```

Embedding models convert text into numerical vectors.

For example:

```text
"What is machine learning?"
            ↓
[0.21, -0.43, 0.72, ...]
```

These vectors allow semantically similar questions and document chunks to be compared.

---

# 🔎 4. FAISS Vector Retrieval

RagLab uses **FAISS** for vector similarity search.

The retrieval process is:

```text
User Question
      ↓
Question Embedding
      ↓
FAISS Similarity Search
      ↓
Top-K Chunks
```

The number of retrieved chunks is configurable.

```text
Top-K = 1 → 20
```

The retrieved chunks are then passed to the optional reranking stage or directly to the LLM.

---

# 🎯 5. Optional CrossEncoder Reranking

RagLab supports an optional second-stage reranking process.

The CrossEncoder model used is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Without reranking

```text
Question
   ↓
FAISS
   ↓
Top-K Chunks
   ↓
LLM
```

### With reranking

```text
Question
   ↓
FAISS
   ↓
Candidate Chunks
   ↓
CrossEncoder
   ↓
Re-ranked Chunks
   ↓
LLM
```

This allows retrieval performance to be compared with and without reranking.

---

# 🤖 6. LLM Generation

The retrieved context is passed to:

```text
openai/gpt-oss-120b
```

The model generates an answer using the retrieved document context.

```text
Question
    +
Retrieved Context
    ↓
GPT-OSS 120B
    ↓
Generated Answer
```

The RAG pipeline is designed to ground responses in the retrieved document content.

---

# 🧪 7. Multi-Metric Evaluation

A major feature of RagLab is its **multi-metric evaluation layer**.

Instead of relying on one score, generated answers are evaluated using multiple independent signals.

---

## 🛡️ NLI Faithfulness

Natural Language Inference is used to evaluate whether the generated answer is supported by the retrieved context.

Conceptually:

```text
Retrieved Context
       +
Generated Answer
       ↓
NLI Model
       ↓
Faithfulness Assessment
```

This helps identify unsupported or potentially hallucinated claims.

---

## 📐 Cosine Similarity

Cosine similarity measures semantic similarity between text representations.

It is used as an additional evaluation signal.

> ⚠️ High semantic similarity does not necessarily mean that an answer is factually correct.

---

## 🧬 BERTScore

BERTScore evaluates semantic similarity using contextual representations.

It provides another signal for comparing the generated answer against the retrieved context.

---

## 📊 Evaluation Example

```text
┌──────────────────────────────┐
│     Multi-Metric Evaluation  │
├──────────────────────────────┤
│ NLI Faithfulness     0.9475  │
│ Cosine Similarity    0.6775  │
│ BERTScore F1         0.8459  │
└──────────────────────────────┘
```

The individual metrics are exposed separately to make evaluation more transparent.

---

# ⏱️ 8. Latency Breakdown

RagLab measures latency across different stages of the pipeline.

Example:

```text
Retrieval          → 32.5 ms
Reranking           → 0.0 ms
LLM Generation   → 1115.4 ms
Evaluation       → 10961.6 ms
──────────────────────────────
Total Latency    → 12112.2 ms
```

This helps identify which part of the RAG system is contributing most to the overall response time.

---

# 📚 9. Retrieved Source Inspection

RagLab exposes information about the retrieved chunks.

Users can inspect:

* 📄 Page number
* 🔢 Chunk ID
* 📊 Relevance score
* 📝 Retrieved content

This makes it possible to understand **why a particular answer was generated** and whether the retrieved context was relevant.

---

# 🗄️ 10. PostgreSQL

PostgreSQL is used for persistent application data.

The project uses:

* PostgreSQL
* SQLAlchemy
* Alembic

Data such as configuration, query history, and evaluation information can be persisted rather than relying entirely on application memory.

---

# ⚙️ 11. Runtime Configuration

The RAG pipeline can be configured through the API.

Example:

```json
{
  "chunk_size": 500,
  "chunk_overlap": 100,
  "embedding_model": "all-MiniLM-L6-v2",
  "top_k": 5,
  "reranking_enabled": false
}
```

This allows experiments with different RAG configurations without changing the application code.

---

# 📡 API Endpoints

| Method | Endpoint      | Description              |
| ------ | ------------- | ------------------------ |
| `GET`  | `/`           | Application interface    |
| `POST` | `/upload`     | Upload and process PDF   |
| `POST` | `/config`     | Configure RAG pipeline   |
| `POST` | `/query`      | Execute RAG query        |
| `GET`  | `/history`    | Retrieve query history   |
| `POST` | `/export`     | Export RAG pipeline      |
| `POST` | `/reset`      | Reset active session     |
| `GET`  | `/api/health` | Application health check |

---

# ❤️ Health Check

```http
GET /api/health
```

The endpoint checks application state and database connectivity.

Example:

```json
{
  "status": "ok",
  "database": "connected",
  "active_sessions": 0,
  "embedding_models_available": [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
  ],
  "groq_api_configured": true
}
```

---

# 🐳 Docker Deployment

RagLab is containerized using Docker.

The application is packaged together with its runtime environment and Python dependencies.

```text
Dockerfile
     ↓
docker build
     ↓
Docker Image
     ↓
docker run
     ↓
Docker Container
     ↓
FastAPI / Uvicorn
```

### Build

```bash
docker build -t raglab .
```

### Run

```bash
docker run -d \
  --name raglab \
  -p 8000:8000 \
  --env-file .env \
  raglab:latest
```

---

# ☁️ AWS EC2 Deployment

RagLab is deployed on **AWS EC2** and runs inside a Docker container.

The deployed application is accessible through the EC2 instance on port `8000`.

### Deployment Architecture

```text
🌍 Internet
      │
      ▼
☁️ AWS EC2
      │
      ▼
🐳 Docker Container
      │
      ▼
⚡ FastAPI / Uvicorn
      │
      ├── 🧠 Embedding Models
      ├── 🗂️ FAISS
      ├── 🗄️ PostgreSQL / Neon
      └── 🤖 LLM API
```

Docker provides a consistent runtime environment between local development and the EC2 deployment.

### 🚀 Live Deployment

**http://16.171.208.210:8000**

### 📚 API Documentation

**http://16.171.208.210:8000/docs**

---

# 🔐 Environment Variables

Sensitive credentials are stored using environment variables.

Example:

```env
DATABASE_URL=your_database_url
OPENAI_API_KEY=your_api_key
```

For security, `.env` should not be committed to GitHub.

Use `.env.example` to document the required variables without exposing secrets.

---

# 🐳 Docker Architecture

```text
                 RagLab Docker Image
                         │
                         ▼
                ┌─────────────────┐
                │ Docker Container│
                │                 │
                │  FastAPI        │
                │  Uvicorn        │
                │  RAG Pipeline   │
                │  FAISS          │
                │  Embeddings     │
                └────────┬────────┘
                         │
                         │ Port 8000
                         ▼
                    AWS EC2
```

---

# 📂 Project Structure

```text
RagLab/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── main.py
│   └── ...
│
├── alembic/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .env.example
└── README.md
```

---

# ⚙️ Local Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/sharathop/RagLab.git
cd RagLab
```

## 2️⃣ Create Virtual Environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file:

```env
DATABASE_URL=your_postgresql_connection_string
OPENAI_API_KEY=your_api_key
```

---

## 5️⃣ Run Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Application:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

# 🧪 Example Workflow

### Step 1 — Upload

```text
📄 Upload PDF
      ↓
📝 Extract Text
      ↓
✂️ Create Chunks
```

### Step 2 — Index

```text
Chunks
   ↓
🧠 Embeddings
   ↓
🗂️ FAISS
```

### Step 3 — Query

```text
❓ User Question
       ↓
🧠 Question Embedding
       ↓
🔎 FAISS Retrieval
       ↓
📚 Top-K Chunks
```

### Step 4 — Rerank

```text
Top-K Chunks
      ↓
🎯 CrossEncoder
      ↓
Best Context
```

### Step 5 — Generate

```text
Question + Context
        ↓
🤖 GPT-OSS 120B
        ↓
💬 Answer
```

### Step 6 — Evaluate

```text
Answer
  │
  ├── 🛡️ NLI Faithfulness
  ├── 📐 Cosine Similarity
  ├── 🧬 BERTScore
  └── ⏱️ Latency
```

---

# 🧰 Technology Stack

| Category                | Technology                        |
| ----------------------- | --------------------------------- |
| 🐍 Programming Language | Python 3.11                       |
| ⚡ Backend               | FastAPI                           |
| 🚀 Server               | Uvicorn                           |
| 🤖 LLM                  | OpenAI GPT-OSS 120B               |
| 🧠 Embeddings           | SentenceTransformers              |
| 🗂️ Vector Search       | FAISS                             |
| 🎯 Reranking            | CrossEncoder                      |
| 🧪 Evaluation           | NLI, Cosine Similarity, BERTScore |
| 🗄️ Database            | PostgreSQL                        |
| 🔗 ORM                  | SQLAlchemy                        |
| 🔄 Migrations           | Alembic                           |
| 🐳 Containerization     | Docker                            |
| ☁️ Cloud                | AWS EC2                           |
| 🖥️ Frontend            | Vite                              |

---

# 🎯 Engineering Goals

RagLab was built to demonstrate practical understanding of an end-to-end RAG system.

The project focuses on:

* 🔍 Retrieval quality
* 🧠 Embedding selection
* 🎯 Reranking
* 🤖 LLM integration
* 🛡️ Faithfulness evaluation
* 🧪 Multi-metric evaluation
* ⏱️ Latency analysis
* ⚙️ Configurable pipelines
* 🗄️ Persistent application data
* 🐳 Containerization
* ☁️ Cloud deployment

Rather than implementing only:

```text
PDF → Vector DB → LLM
```

RagLab exposes and evaluates the individual stages of the RAG pipeline.

---

# 🚧 Future Improvements

* ⚡ Redis caching
* 🔎 Hybrid BM25 + dense retrieval
* 🗂️ Persistent vector database
* 🔐 Authentication and authorization
* 📊 Production monitoring
* 🔄 Background document processing
* 📚 Additional document formats
* 🤖 Multiple LLM providers
* 📈 Automated RAG evaluation benchmarks
* 🌐 Production HTTPS/domain setup

---

# 👨‍💻 Author

## Sharath M

**AI / ML / GenAI Engineer**

🎓 MCA
☁️ AWS Certified Cloud Practitioner

### 🔗 Links

* 💻 **GitHub:** https://github.com/sharathop
* 🔗 **RagLab Repository:** https://github.com/sharathop/RagLab
* 🚀 **Live Deployment:** http://16.171.208.210:8000

---

<p align="center">

⭐ **If you find this project useful, consider giving the repository a star!**

</p>
