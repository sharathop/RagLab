import os
import time
import tempfile
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.database import get_db, check_db_connection
from app.db.repository import Repository
from app.models.config import PipelineConfig, ALLOWED_EMBEDDING_MODELS, ConfigResponse
from app.models.schemas import (
    UploadResponse,
    AskRequest,
    AskResponse,
    HealthResponse,
    HistoryItem,
    DownloadRequest
)
from app.services.session_manager import SessionManager
from app.services.pdf_parser import PDFParserService, PDFParserError
from app.services.chunker import ChunkerService
from app.services.embeddings import EmbeddingService, EmbeddingError
from app.services.retriever import RetrieverService
from app.services.reranker import RerankerError
from app.services.generator import GeneratorService, GeneratorError
from app.services.evaluator import EvaluatorService, EvaluatorError
from app.services.code_generator import CodeGeneratorService

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthResponse)
def health_check():
    db_ok = check_db_connection()
    groq_ok = bool(os.getenv("GROQ_API_KEY", "").strip())
    active_count = SessionManager.get_active_count()

    return HealthResponse(
        status="ok" if db_ok else "degraded",
        database="connected" if db_ok else "disconnected",
        active_sessions=active_count,
        embedding_models_available=ALLOWED_EMBEDDING_MODELS,
        groq_api_configured=groq_ok
    )


@router.post("/upload", response_model=UploadResponse)
async def api_upload_pdf(
    session_id: Optional[str] = None,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    sid, session_data = SessionManager.get_or_create(session_id, db=db)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

        temp_file.write(content)
        temp_file.flush()
        temp_file.close()

        pages_data, total_pages = PDFParserService.parse_pdf(temp_file.name)
        cfg = session_data.current_config
        chunks = ChunkerService.chunk_pages(pages_data, cfg.chunk_size, cfg.chunk_overlap)

        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text chunks found in document.")

        chunk_texts = [c.text for c in chunks]
        embeddings = EmbeddingService.embed_texts(chunk_texts, model_name=cfg.embedding_model)

        session_data.init_vector_store(cfg.embedding_model)
        session_data.vector_store.add_chunks(chunks, embeddings)
        session_data.chunks = chunks
        session_data.pages_data = pages_data
        session_data.filename = file.filename
        session_data.total_pages = total_pages

        return UploadResponse(
            session_id=sid,
            filename=file.filename,
            total_pages=total_pages,
            message=f"Successfully indexed {len(chunks)} chunks using {cfg.embedding_model}."
        )
    except PDFParserError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmbeddingError as e:
        raise HTTPException(status_code=502, detail=str(e))
    finally:
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass


@router.post("/config", response_model=ConfigResponse)
def api_update_config(
    config: PipelineConfig,
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    sid, session_data = SessionManager.get_or_create(session_id, db=db)
    old_config = session_data.current_config
    session_data.current_config = config

    saved_cfg = Repository.save_config(
        db=db,
        session_id=sid,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedding_model=config.embedding_model,
        top_k=config.top_k,
        reranking_enabled=config.reranking_enabled
    )
    if saved_cfg:
        session_data.config_id = saved_cfg.id
        session_data.config_version += 1

    needs_rechunk = (
        old_config.chunk_size != config.chunk_size
        or old_config.chunk_overlap != config.chunk_overlap
    )
    needs_reembed_only = old_config.embedding_model != config.embedding_model

    # Re-chunk AND re-embed if chunk_size/overlap changed (old chunks are
    # stale — different boundaries entirely); re-embed only if just the
    # model changed. On failure, roll back rather than leaving the config
    # claiming a state the vector store doesn't actually reflect — that
    # mismatch would silently corrupt every retrieval/score afterward.
    if (needs_rechunk or needs_reembed_only) and session_data.pages_data:
        try:
            if needs_rechunk:
                new_chunks = ChunkerService.chunk_pages(
                    pages_data=session_data.pages_data,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap
                )
                if not new_chunks:
                    raise ValueError(
                        f"Re-chunking with size={config.chunk_size}, "
                        f"overlap={config.chunk_overlap} produced no chunks."
                    )
            else:
                new_chunks = session_data.chunks

            chunk_texts = [c.text for c in new_chunks]
            embeddings = EmbeddingService.embed_texts(chunk_texts, model_name=config.embedding_model)
            session_data.init_vector_store(config.embedding_model)
            session_data.vector_store.add_chunks(new_chunks, embeddings)
            session_data.chunks = new_chunks
        except (EmbeddingError, ValueError) as e:
            session_data.current_config = old_config
            raise HTTPException(
                status_code=502,
                detail=f"Could not apply new config (chunk_size={config.chunk_size}, "
                       f"overlap={config.chunk_overlap}, model={config.embedding_model}): {e}. "
                       f"Kept previous settings so the index stays consistent."
            )

    return ConfigResponse(
        id=saved_cfg.id if saved_cfg else 1,
        version=session_data.config_version,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        embedding_model=config.embedding_model,
        top_k=config.top_k,
        reranking_enabled=config.reranking_enabled,
        created_at=str(saved_cfg.created_at if saved_cfg else "")
    )


@router.post("/query", response_model=AskResponse)
def api_query(
    body: AskRequest,
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required.")

    session_data = SessionManager.get(session_id, db=db)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found or has expired.")

    if not session_data.vector_store or session_data.vector_store.total_chunks == 0:
        raise HTTPException(status_code=400, detail="No active document in session. Upload a PDF first.")

    t0 = time.perf_counter()
    cfg = session_data.current_config

    try:
        sources, retrieval_time_ms, reranking_time_ms, retrieval_warning = RetrieverService.retrieve(
            query=body.question,
            vector_store=session_data.vector_store,
            embedding_model=cfg.embedding_model,
            top_k=cfg.top_k,
            reranking_enabled=cfg.reranking_enabled
        )
    except EmbeddingError as e:
        raise HTTPException(status_code=502, detail=f"Could not embed your question: {e}")

    try:
        answer, llm_time_ms = GeneratorService.generate_answer(body.question, sources)
    except GeneratorError as ge:
        if "GROQ_API_KEY is not configured" in str(ge):
            answer = (
                "[Groq API Key Required for Live Llama 3.3 70B Generation] "
                f"Retrieved {len(sources)} source chunks from document."
            )
            llm_time_ms = 0.0
        else:
            raise HTTPException(status_code=502, detail=str(ge))

    try:
        evaluation, eval_time_ms = EvaluatorService.evaluate(
            question=body.question,
            answer=answer,
            sources=sources,
            embedding_model=cfg.embedding_model
        )
    except EvaluatorError as e:
        raise HTTPException(status_code=502, detail=str(e))

    total_time_ms = (time.perf_counter() - t0) * 1000.0

    # Save to PostgreSQL
    log_obj = Repository.save_query_log(
        db=db,
        session_id=session_id,
        question=body.question,
        answer=answer,
        config_id=session_data.config_id,
        retrieval_time_ms=retrieval_time_ms,
        llm_time_ms=llm_time_ms,
        evaluation_time_ms=eval_time_ms,
        total_time_ms=total_time_ms
    )
    if log_obj:
        Repository.save_evaluation_scores(
            db=db,
            query_id=log_obj.id,
            nli_score=evaluation.nli_score,
            cosine_score=evaluation.cosine_score,
            bertscore=evaluation.bertscore
        )

    default_note = (
        "Cosine similarity measures semantic similarity. It does not prove that the answer is "
        "factually supported by the context."
    )
    combined_warning = f"{default_note} {retrieval_warning}" if retrieval_warning else default_note

    return AskResponse(
        question=body.question,
        answer=answer,
        sources=sources,
        evaluation=evaluation,
        performance={
            "retrieval_time_ms": retrieval_time_ms,
            "reranking_time_ms": reranking_time_ms,
            "llm_time_ms": llm_time_ms,
            "evaluation_time_ms": eval_time_ms,
            "total_time_ms": total_time_ms
        },
        warning=combined_warning
    )


@router.get("/history", response_model=List[HistoryItem])
def api_history(
    session_id: str,
    db: Session = Depends(get_db)
):
    history_logs = Repository.get_session_history(db, session_id)
    items = []
    for log in history_logs:
        scores = log.evaluation_scores
        items.append(
            HistoryItem(
                id=log.id,
                question=log.question,
                answer=log.answer,
                config_id=log.config_id,
                retrieval_time_ms=log.retrieval_time_ms,
                llm_time_ms=log.llm_time_ms,
                evaluation_time_ms=log.evaluation_time_ms,
                total_time_ms=log.total_time_ms,
                created_at=str(log.created_at),
                nli_score=scores.nli_score if scores else None,
                cosine_score=scores.cosine_score if scores else None,
                bertscore=scores.bertscore if scores else None
            )
        )
    return items


@router.post("/export")
def api_export(
    request: DownloadRequest,
    session_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    sid, session_data = SessionManager.get_or_create(session_id, db=db)
    resolved, _ = CodeGeneratorService.resolve_dependencies(request.components)
    cfg = session_data.current_config

    if request.format == "file":
        code_str = CodeGeneratorService.generate_single_file(resolved, cfg)
        return Response(
            content=code_str,
            media_type="text/x-python",
            headers={"Content-Disposition": 'attachment; filename="self_correcting_rag.py"'}
        )
    else:
        zip_io = CodeGeneratorService.generate_project_zip(resolved, cfg)
        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="self-correcting-rag.zip"'}
        )