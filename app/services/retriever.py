import time
from typing import List, Optional, Tuple
from app.models.schemas import RetrievedSource
from app.services.vector_store import FAISSVectorStore
from app.services.embeddings import EmbeddingService
from app.services.reranker import RerankerService, RerankerError


class RetrieverService:
    @staticmethod
    def retrieve(
        query: str,
        vector_store: FAISSVectorStore,
        embedding_model: str,
        top_k: int = 5,
        reranking_enabled: bool = False
    ) -> Tuple[List[RetrievedSource], float, float, Optional[str]]:
        """
        Execute RAG retrieval pipeline with optional CrossEncoder reranking.
        Returns:
            Tuple of (sources, retrieval_time_ms, reranking_time_ms, warning)
            warning is None unless reranking was requested but could not run, in which
            case it explains that results fell back to unreranked vector-search order —
            we never silently relabel unreranked results as "reranked".
        """
        if vector_store.total_chunks == 0:
            return [], 0.0, 0.0, None

        # Step 1: Embed query and retrieve from FAISS
        t0 = time.perf_counter()
        query_embedding = EmbeddingService.embed_query(query, model_name=embedding_model)

        # If reranking enabled, retrieve top_n candidates (e.g. 2x top_k, min 10)
        top_n = max(top_k * 2, 10) if reranking_enabled else top_k
        raw_results = vector_store.search(query_embedding, top_k=top_n)
        retrieval_time_ms = (time.perf_counter() - t0) * 1000.0

        # Step 2: Optional Reranking
        reranking_time_ms = 0.0
        warning: Optional[str] = None
        reranked_ok = False

        if reranking_enabled and raw_results:
            t_rerank = time.perf_counter()
            try:
                reranked_results = RerankerService.rerank(query, raw_results, top_k=top_k)
                reranking_time_ms = (time.perf_counter() - t_rerank) * 1000.0
                reranked_ok = True
            except RerankerError as e:
                # Degrade honestly: fall back to vector-search order, tell the user why,
                # and do NOT mark these sources as reranked=True.
                reranking_time_ms = (time.perf_counter() - t_rerank) * 1000.0
                warning = f"Reranking was enabled but unavailable ({e}); showing unreranked vector-search results instead."

        if reranked_ok:
            sources = [
                RetrievedSource(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    score=round(score, 4),
                    text=chunk.text,
                    reranked=True
                )
                for chunk, score in reranked_results
            ]
        else:
            sources = [
                RetrievedSource(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    score=round(score, 4),
                    text=chunk.text,
                    reranked=False
                )
                for chunk, score in raw_results[:top_k]
            ]

        return sources, retrieval_time_ms, reranking_time_ms, warning
