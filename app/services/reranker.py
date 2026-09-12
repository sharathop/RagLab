from typing import List, Tuple
import math
from app.models.session import ChunkMetadata


class RerankerError(Exception):
    """Raised when the CrossEncoder reranker model can't be loaded or run.

    No silent fallback to a word-overlap scorer here either — a fake
    reranker that looks plausible would quietly change which chunks the
    user thinks were "best matched," with no indication that reranking
    wasn't actually performed by the real model.
    """
    pass


class RerankerService:
    """
    Lazy-loading CrossEncoder reranking service.
    Loads cross-encoder/ms-marco-MiniLM-L-6-v2 only when reranking is requested.
    """
    _model = None

    @classmethod
    def load_model(cls):
        if cls._model is None:
            try:
                from sentence_transformers import CrossEncoder
                cls._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception as e:
                raise RerankerError(
                    f"Failed to load reranker model 'cross-encoder/ms-marco-MiniLM-L-6-v2': {e}. "
                    "Check network access / that the model is cached locally. "
                    "Refusing to fall back to a fake word-overlap reranker, since that would "
                    "silently change result ordering without telling the user."
                ) from e
        return cls._model

    @classmethod
    def rerank(
        cls,
        query: str,
        candidates: List[Tuple[ChunkMetadata, float]],
        top_k: int
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Rerank retrieved candidates using CrossEncoder.
        Args:
            query: User's question
            candidates: List of (ChunkMetadata, initial_score)
            top_k: Final number of chunks to return
        Returns:
            List of (ChunkMetadata, reranked_score) sorted descending

        Raises:
            RerankerError: if the real model can't be loaded or fails to
            score the candidates. Callers should surface this to the user
            (e.g. a clear "reranking unavailable" message) rather than
            silently returning unreranked results, so the user knows the
            "reranking enabled" toggle didn't actually do anything this time.
        """
        if not candidates:
            return []

        pairs = [(query, chunk.text) for chunk, _ in candidates]
        model = cls.load_model()
        try:
            scores = model.predict(pairs)
        except Exception as e:
            raise RerankerError(f"Reranker failed to score candidates: {e}") from e

        reranked = []
        for (chunk, _), score in zip(candidates, scores):
            norm_score = float(score)
            if norm_score < 0:
                # Sigmoid approximation if negative logits
                norm_score = 1.0 / (1.0 + math.exp(-norm_score))
            norm_score = max(0.0, min(1.0, norm_score))
            reranked.append((chunk, norm_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
