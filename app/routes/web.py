import os
import time
import tempfile
from typing import List, Optional
from fastapi import APIRouter, Request, Response, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.db.database import get_db, check_db_connection
from app.db.repository import Repository
from app.models.config import PipelineConfig, ALLOWED_EMBEDDING_MODELS
from app.services.session_manager import SessionManager, SessionData
from app.services.pdf_parser import PDFParserService, PDFParserError
from app.services.chunker import ChunkerService
from app.services.embeddings import EmbeddingService, EmbeddingError
from app.services.retriever import RetrieverService
from app.services.generator import GeneratorService, GeneratorError
from app.services.evaluator import EvaluatorService
from app.services.code_generator import CodeGeneratorService

router = APIRouter()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))
SESSION_COOKIE_NAME = "rag_session_id"


def get_current_session(request: Request, db: Session) -> tuple[str, SessionData, bool]:
    """Retrieve existing session or create a new one. Returns (session_id, session_data, is_new)."""
    raw_cookie = request.cookies.get(SESSION_COOKIE_NAME)
    session_data = None
    is_new = False

    if raw_cookie:
        session_data = SessionManager.get(raw_cookie, db=db)

    if not session_data:
        session_id, session_data = SessionManager.get_or_create(None, db=db)
        is_new = True
    else:
        session_id = raw_cookie

    return session_id, session_data, is_new


def attach_session_cookie(response: Response, session_id: str):
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=1800,  # 30 minutes TTL
        httponly=True,
        samesite="lax"
    )


@router.get("/", response_class=HTMLResponse)
async def index_page(request: Request, db: Session = Depends(get_db)):
    session_id, session_data, is_new = get_current_session(request, db)

    db_connected = check_db_connection()
    groq_configured = bool(os.getenv("GROQ_API_KEY", "").strip())
    history = Repository.get_session_history(db, session_id)

    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "session_id": session_id,
            "session_data": session_data,
            "config": session_data.current_config,
            "db_connected": db_connected,
            "groq_configured": groq_configured,
            "history": history
        }
    )
    if is_new:
        attach_session_cookie(response, session_id)
    return response


@router.post("/upload", response_class=HTMLResponse)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    session_id, session_data, is_new = get_current_session(request, db)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        html = templates.TemplateResponse(
            request=request,
            name="partials/error_alert.html",
            context={
                "error_title": "Invalid File Format",
                "error_message": "Only PDF files (.pdf) are supported. Please select a valid document."
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp

    # Write to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    t0 = time.perf_counter()
    try:
        content = await file.read()
        if len(content) == 0:
            raise PDFParserError("The uploaded file is empty (0 bytes).")
        if len(content) > 10 * 1024 * 1024:
            raise PDFParserError("File exceeds 10 MB limit.")

        temp_file.write(content)
        temp_file.flush()
        temp_file.close()

        # Step 1: Parse PDF
        pages_data, total_pages = PDFParserService.parse_pdf(temp_file.name)

        # Step 2: Chunk Document
        cfg = session_data.current_config
        chunks = ChunkerService.chunk_pages(
            pages_data=pages_data,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap
        )

        if not chunks:
            raise PDFParserError("Could not extract any chunks from the document text.")

        # Step 3: Embed Chunks
        chunk_texts = [c.text for c in chunks]
        embeddings = EmbeddingService.embed_texts(chunk_texts, model_name=cfg.embedding_model)

        # Step 4: Index into in-memory FAISS
        session_data.init_vector_store(cfg.embedding_model)
        session_data.vector_store.add_chunks(chunks, embeddings)
        session_data.chunks = chunks
        session_data.pages_data = pages_data
        session_data.filename = file.filename
        session_data.total_pages = total_pages

        ingest_time_ms = (time.perf_counter() - t0) * 1000.0

        html = templates.TemplateResponse(
            request=request,
            name="partials/upload_status.html",
            context={
                "filename": file.filename,
                "total_pages": total_pages,
                "total_chunks": len(chunks),
                "embedding_model": cfg.embedding_model,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "ingest_time_ms": ingest_time_ms
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp

    except PDFParserError as pe:
        html = templates.TemplateResponse(
            request=request,
            name="partials/error_alert.html",
            context={
                "error_title": "PDF Ingestion Error",
                "error_message": str(pe)
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp

    except Exception as ex:
        html = templates.TemplateResponse(
            request=request,
            name="partials/error_alert.html",
            context={
                "error_title": "System Error During Parsing",
                "error_message": f"An unexpected error occurred: {str(ex)}"
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp

    finally:
        # Crucial requirement: Delete temporary PDF immediately after parsing
        if os.path.exists(temp_file.name):
            try:
                os.remove(temp_file.name)
            except Exception:
                pass


@router.post("/config", response_class=HTMLResponse)
async def update_pipeline_config(
    request: Request,
    chunk_size: int = Form(...),
    chunk_overlap: int = Form(...),
    embedding_model: str = Form(...),
    top_k: int = Form(...),
    reranking_enabled: bool = Form(default=False),
    db: Session = Depends(get_db)
):
    session_id, session_data, is_new = get_current_session(request, db)

    try:
        new_config = PipelineConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            top_k=top_k,
            reranking_enabled=reranking_enabled
        )
    except Exception as val_err:
        html = templates.TemplateResponse(
            request=request,
            name="partials/error_alert.html",
            context={
                "error_title": "Configuration Validation Error",
                "error_message": str(val_err)
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp

    old_config = session_data.current_config
    session_data.current_config = new_config

    # Persist in PostgreSQL
    saved_cfg = Repository.save_config(
        db=db,
        session_id=session_id,
        chunk_size=new_config.chunk_size,
        chunk_overlap=new_config.chunk_overlap,
        embedding_model=new_config.embedding_model,
        top_k=new_config.top_k,
        reranking_enabled=new_config.reranking_enabled
    )
    if saved_cfg:
        session_data.config_id = saved_cfg.id
        session_data.config_version += 1

    msg = "Configuration saved successfully."

    needs_rechunk = (
        old_config.chunk_size != new_config.chunk_size
        or old_config.chunk_overlap != new_config.chunk_overlap
    )
    needs_reembed_only = old_config.embedding_model != new_config.embedding_model

    # Re-chunk AND re-embed if chunk_size/overlap changed (the actual chunk
    # boundaries are different, so the old chunks are stale); re-embed only
    # if just the embedding model changed. Either way, if it fails, roll
    # back so current_config never claims a state the vector store doesn't
    # actually reflect — that mismatch would silently corrupt every
    # retrieval/score computed afterward, which is worse than the config
    # update simply not taking effect.
    if (needs_rechunk or needs_reembed_only) and session_data.pages_data:
        try:
            if needs_rechunk:
                new_chunks = ChunkerService.chunk_pages(
                    pages_data=session_data.pages_data,
                    chunk_size=new_config.chunk_size,
                    chunk_overlap=new_config.chunk_overlap
                )
                if not new_chunks:
                    raise ValueError(
                        f"Re-chunking with size={new_config.chunk_size}, "
                        f"overlap={new_config.chunk_overlap} produced no chunks."
                    )
            else:
                new_chunks = session_data.chunks

            chunk_texts = [c.text for c in new_chunks]
            embeddings = EmbeddingService.embed_texts(chunk_texts, model_name=new_config.embedding_model)
            session_data.init_vector_store(new_config.embedding_model)
            session_data.vector_store.add_chunks(new_chunks, embeddings)
            session_data.chunks = new_chunks

            if needs_rechunk:
                msg += f" Re-chunked into {len(new_chunks)} chunks (size={new_config.chunk_size}, overlap={new_config.chunk_overlap}) and re-indexed."
            else:
                msg += f" Re-indexed {len(new_chunks)} chunks with {new_config.embedding_model}."
        except (EmbeddingError, ValueError) as e:
            session_data.current_config = old_config
            new_config = old_config
            msg = (
                f"Configuration NOT applied: re-chunking/re-indexing failed ({e}). "
                f"Kept previous settings so the index stays consistent with what's actually indexed."
            )

    html = templates.TemplateResponse(
        request=request,
        name="partials/config_status.html",
        context={
            "config": new_config,
            "config_version": session_data.config_version,
            "message": msg
        }
    ).body
    resp = HTMLResponse(content=html, status_code=200)
    if is_new:
        attach_session_cookie(resp, session_id)
    return resp


@router.post("/query", response_class=HTMLResponse)
async def execute_query(
    request: Request,
    question: str = Form(...),
    db: Session = Depends(get_db)
):
    session_id, session_data, is_new = get_current_session(request, db)

    clean_question = question.strip()
    if not clean_question:
        html = templates.TemplateResponse(
            request=request,
            name="partials/error_alert.html",
            context={
                "error_title": "Empty Question",
                "error_message": "Please enter a question to ask the document."
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp

    if not session_data.vector_store or session_data.vector_store.total_chunks == 0:
        html = templates.TemplateResponse(
            request=request,
            name="partials/error_alert.html",
            context={
                "error_title": "No Document Indexed",
                "error_message": "Please upload a PDF document before asking questions."
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp

    t_total_start = time.perf_counter()
    cfg = session_data.current_config

    try:
        # Step 1: Retrieval & Optional Reranking
        sources, retrieval_time_ms, reranking_time_ms, retrieval_warning = RetrieverService.retrieve(
            query=clean_question,
            vector_store=session_data.vector_store,
            embedding_model=cfg.embedding_model,
            top_k=cfg.top_k,
            reranking_enabled=cfg.reranking_enabled
        )

        # Step 2: Generation via Groq Llama 3.3 70B
        try:
            answer, llm_time_ms = GeneratorService.generate_answer(clean_question, sources)
        except GeneratorError as ge:
            # If Groq key not set, provide helpful demonstration answer with the actual retrieved context
            if "GROQ_API_KEY is not configured" in str(ge):
                answer = (
                    "**[Groq API Key Required for Live Llama 3.3 70B Generation]**\n\n"
                    "The RAG retrieval pipeline successfully found and ranked the relevant document chunks shown below. "
                    "To generate a contextual synthesis answer with Groq Llama 3.3 70B, set your `GROQ_API_KEY` in `.env`. "
                    f"Showing retrieved context from {len(sources)} source chunks."
                )
                llm_time_ms = 0.0
            else:
                raise

        # Step 3: Independent Multi-Metric Evaluation
        evaluation, eval_time_ms = EvaluatorService.evaluate(
            question=clean_question,
            answer=answer,
            sources=sources,
            embedding_model=cfg.embedding_model
        )

        total_time_ms = (time.perf_counter() - t_total_start) * 1000.0

        performance = {
            "retrieval_time_ms": retrieval_time_ms,
            "reranking_time_ms": reranking_time_ms,
            "llm_time_ms": llm_time_ms,
            "evaluation_time_ms": eval_time_ms,
            "total_time_ms": total_time_ms
        }

        # Step 4: Persist in PostgreSQL (query_logs + evaluation_scores)
        query_log = Repository.save_query_log(
            db=db,
            session_id=session_id,
            question=clean_question,
            answer=answer,
            config_id=session_data.config_id,
            retrieval_time_ms=retrieval_time_ms,
            llm_time_ms=llm_time_ms,
            evaluation_time_ms=eval_time_ms,
            total_time_ms=total_time_ms
        )

        if query_log:
            Repository.save_evaluation_scores(
                db=db,
                query_id=query_log.id,
                nli_score=evaluation.nli_score,
                cosine_score=evaluation.cosine_score,
                bertscore=evaluation.bertscore
            )

        html = templates.TemplateResponse(
            request=request,
            name="partials/query_result.html",
            context={
                "question": clean_question,
                "answer": answer,
                "sources": sources,
                "evaluation": evaluation,
                "performance": performance,
                "retrieval_warning": retrieval_warning
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp

    except Exception as ex:
        html = templates.TemplateResponse(
            request=request,
            name="partials/error_alert.html",
            context={
                "error_title": "Query Processing Error",
                "error_message": f"Failed to execute RAG query pipeline: {str(ex)}"
            }
        ).body
        resp = HTMLResponse(content=html, status_code=200)
        if is_new:
            attach_session_cookie(resp, session_id)
        return resp


@router.get("/history", response_class=HTMLResponse)
async def query_history(request: Request, db: Session = Depends(get_db)):
    session_id, _, is_new = get_current_session(request, db)
    history = Repository.get_session_history(db, session_id)
    html = templates.TemplateResponse(
        request=request,
        name="partials/history_list.html",
        context={
            "history": history
        }
    ).body
    resp = HTMLResponse(content=html, status_code=200)
    if is_new:
        attach_session_cookie(resp, session_id)
    return resp


@router.post("/export/resolve", response_class=HTMLResponse)
async def export_resolve_dependencies(request: Request):
    form_data = await request.form()
    components = form_data.getlist("components")
    resolved, notifications = CodeGeneratorService.resolve_dependencies(components)

    return templates.TemplateResponse(
        request=request,
        name="partials/export_status.html",
        context={
            "resolved": sorted(list(resolved)),
            "notifications": notifications
        }
    )


@router.post("/export/download")
async def export_download(
    request: Request,
    format: str = Form(default="zip"),
    db: Session = Depends(get_db)
):
    session_id, session_data, _ = get_current_session(request, db)
    form_data = await request.form()
    selected_components = form_data.getlist("components")

    if not selected_components:
        selected_components = ["parser", "chunking", "embeddings", "retrieval", "generation", "evaluation"]

    resolved, _ = CodeGeneratorService.resolve_dependencies(selected_components)
    config = session_data.current_config

    if format == "file":
        code_str = CodeGeneratorService.generate_single_file(resolved, config)
        return Response(
            content=code_str,
            media_type="text/x-python",
            headers={"Content-Disposition": 'attachment; filename="self_correcting_rag.py"'}
        )
    else:
        zip_io = CodeGeneratorService.generate_project_zip(resolved, config)
        return StreamingResponse(
            zip_io,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="self-correcting-rag.zip"'}
        )


@router.post("/reset", response_class=HTMLResponse)
async def reset_session(request: Request, db: Session = Depends(get_db)):
    raw_cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if raw_cookie:
        SessionManager.expire(raw_cookie, db=db)

    # Issue a brand new session
    new_id, new_session = SessionManager.get_or_create(None, db=db)
    db_connected = check_db_connection()
    groq_configured = bool(os.getenv("GROQ_API_KEY", "").strip())
    history = Repository.get_session_history(db, new_id)

    response = templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "session_id": new_id,
            "session_data": new_session,
            "config": new_session.current_config,
            "db_connected": db_connected,
            "groq_configured": groq_configured,
            "history": history
        }
    )
    attach_session_cookie(response, new_id)
    return response