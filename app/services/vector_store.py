import faiss
import numpy as np
from typing import List, Tuple, Optional
from app.models.session import ChunkMetadata


class FAISSVectorStore:
    """
    In-memory FAISS Index wrapper for a single session.
    Uses IndexFlatIP with L2-normalized vectors for exact cosine similarity search.
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: List[ChunkMetadata] = []

    def add_chunks(self, chunks: List[ChunkMetadata], embeddings: np.ndarray) -> None:
        """
        Add chunks and their normalized embeddings to the index.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index {self.dimension}")

        # Ensure float32 and L2 normalization for cosine similarity
        vectors = embeddings.astype(np.float32)
        faiss.normalize_L2(vectors)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[ChunkMetadata, float]]:
        """
        Search for top_k most similar chunks.
        Returns:
            List of (ChunkMetadata, score) sorted descending by score.
        """
        if self.index.ntotal == 0:
            return []

        # Prepare query vector
        query_vec = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, actual_k)

        results: List[Tuple[ChunkMetadata, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                # Cosine similarity is in [-1, 1], normalized to [0, 1] for display if desired
                norm_score = max(0.0, min(1.0, float(score)))
                results.append((self.chunks[idx], norm_score))

        return results

    def clear(self) -> None:
        """Reset index and chunk storage."""
        self.index.reset()
        self.chunks.clear()

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal
