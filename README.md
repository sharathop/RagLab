<div align="center">

# 🧪 RAGLABB

**A configurable, self-evaluating RAG pipeline — deployed live, not just a notebook demo.**

[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen?style=for-the-badge&logo=googlechrome&logoColor=white)](http://16.171.208.210:8000)
[![GitHub Repo](https://img.shields.io/badge/repo-GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/sharathop/RagLab)

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS_EC2-FF9900?style=flat-square&logo=amazonaws&logoColor=white)
![Postgres](https://img.shields.io/badge/PostgreSQL-Neon-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Meta-0467DF?style=flat-square)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=flat-square)
![HTMX](https://img.shields.io/badge/HTMX-3D72D7?style=flat-square&logo=htmx&logoColor=white)

</div>

---

## 📖 What this is

Upload a PDF, control every stage of the RAG pipeline yourself — chunking strategy,
chunk size/overlap, embedding model, reranking — and ask questions against it. Instead
of a single black-box "correct/incorrect" verdict, you see **raw evaluation scores**
(NLI, cosine similarity, BERTScore) for every answer, with a warning when a low score
is explained by a known metric bias rather than an actual bad answer.

Originally extended from a standalone LLM hallucination-detection framework, rebuilt
from a Streamlit prototype into a production FastAPI service — deployed end-to-end on
AWS with a managed Postgres backend.

> 🔗 **[Live demo](http://16.171.208.210:8000)** · 💻 **[Source](https://github.com/sharathop/RagLab)** · 📓 **[Eval framework docs](https://sha6th-llm-eval-ap.hf.space/docs)**

---

## 🏗️ Architecture

```
User (browser)
     │
     ▼
🌐 FastAPI (HTMX server-rendered UI)
     │
     ├── 📄 Upload PDF ──► Parser ──► Chunker ──► Embeddings ──► 🔍 FAISS (in-memory, per-session)
     │
     ├── ⚙️  Change config (chunk size/overlap/model/rerank) ──► RE-CHUNKS + RE-EMBEDS + REBUILDS
     │                                                            the index automatically, no Save button
     │
     └── ❓ Ask a question
              │
              ▼
        🔎 Retriever (FAISS search, optional CrossEncoder reranking)
              │
              ▼
        🤖 Generator (Groq — Llama 3.3 70B)
              │
              ▼
        📊 Evaluator ──► HTTP ──► external hallucination-detection framework
              │                   (separate project, hosted on 🤗 Hugging Face Spaces)
              ▼
        Raw NLI / Cosine / BERTScore scores — NO fused verdict
        ⚠️  cosine gets a bias warning when a low score is explained by
        answer length/list-format rather than an actual faithfulness issue
              │
              ▼
        🗄️  Persisted to PostgreSQL: query history, evaluation scores, config versions
```

**Deployment:** 🐳 Docker image built and run directly on an AWS EC2 (free tier)
instance, connected to Neon (managed Postgres, free tier) and the Groq API (free
tier). No nginx/TLS or systemd in front yet — see [Known limitations](#-known-limitations).

---

## 🧠 Design decisions (and why)

| Decision | Why |
|---|---|
| **Raw scores, no fused verdict** | Fusing NLI/cosine/BERTScore into one verdict can mislead — a low cosine score alone can look damning even when it's fully explained by answer length/format, not a real faithfulness issue. All three scores are shown, with a contextual warning. |
| **Cosine compares QUESTION ↔ answer**, not context ↔ answer | Known real bias: a short question paired with a long, correct, list-formatted answer naturally drifts in embedding space regardless of accuracy. Flagged automatically, not hidden. |
| **Evaluation is delegated, not reimplemented** | NLI/cosine/BERTScore come from a separate hallucination-detection framework over HTTP. No local fallback — if it's unreachable, the request fails loudly instead of silently substituting a weaker heuristic. |
| **No silent fallbacks anywhere** | Embedding/reranking models raise a clear error if they fail to load, instead of substituting fake vectors or a naive word-overlap score. |
| **Config changes rebuild, never append** | Changing chunk size/overlap/embedding model re-chunks the original PDF from scratch and rebuilds FAISS fully. A failed re-index rolls back to the last known-good config. |
| **Auto-applies on change** | No Save button — any control change submits automatically; sliders fire on release, not mid-drag, so an expensive re-embed doesn't run continuously. |
| **One PDF per session, FAISS only** | Deliberately narrow scope — this demonstrates RAG design and honest evaluation, not infrastructure breadth. |

---

## 🧰 Stack

- **Backend:** FastAPI, server-rendered HTMX (no separate JS framework)
- **PDF parsing:** PyPDF, page-by-page
- **Chunking:** configurable size/overlap, re-chunkable on demand from stored raw pages
- **Embeddings:** `sentence-transformers` — `all-MiniLM-L6-v2` or `all-mpnet-base-v2`
- **Vector store:** FAISS, in-memory, rebuilt fresh per config change
- **Reranking:** CrossEncoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), toggle on/off
- **Generation:** Groq API — Llama 3.3 70B
- **Evaluation:** external hallucination-detection framework (NLI, cosine, BERTScore) over HTTP
- **Database:** PostgreSQL (Neon), SQLAlchemy + Alembic
- **Deployment:** Docker on AWS EC2

---

## 🚀 Running it locally

```bash
# 1. Install dependencies
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# fill in DATABASE_URL, GROQ_API_KEY, EVAL_FRAMEWORK_URL

# 3. Run the migration
alembic upgrade head

# 4. Start the app
uvicorn app.main:app --reload
```

Open `http://localhost:8000`.

### 🐳 Or via Docker

```bash
docker build -t raglabb .
docker run --rm --env-file .env raglabb alembic upgrade head
docker run -d --name raglabb-app --env-file .env -p 8000:8000 --restart unless-stopped raglabb
```

---

## 🖱️ Using it

1. **📄 Upload a PDF** (non-scanned, text-based)
2. **⚙️ Adjust the pipeline config** — chunk size, overlap, embedding model, top-K, reranking. Applies automatically.
3. **❓ Ask a question** — answered from retrieved context only
4. **📊 Read the three raw scores** — NLI, cosine, BERTScore — plus a bias warning if applicable
5. **⬇️ Download a stripped-down version** of the pipeline (parser, chunking, FAISS, CrossEncoder) as a single Python file

---

## ⚠️ Known limitations

- **No nginx/TLS** — served over plain HTTP, no reverse-proxy rate limiting (deliberate scope cut)
- **No systemd** — relies on Docker's `--restart unless-stopped` rather than a supervised process with structured logging
- **Single-process session state** — in-memory, not Postgres/Redis-backed; fine for one worker, not multi-worker
- **One PDF per session** — replaced entirely on next upload (with a confirmation prompt)
- **No retrieval self-eval/retry loop** — considered, deliberately scoped out due to added latency cost for a single-user demo
- **No multi-VDB, per-user API keys, horizontal scaling, or CI/CD** — out of scope by design
- **Depends on a free-tier Hugging Face Space for evaluation** — can cold-start (30-90s) after inactivity; the app's timeout accounts for this, but it's a real dependency, not hidden

---
