import numpy as np
from typing import List, Dict
from app.models.config import ALLOWED_EMBEDDING_MODELS


class EmbeddingError(Exception):
    """Raised whenever real embeddings cannot be produced.

    IMPORTANT: this service intentionally has NO silent fallback to a fake
    (random/hash-based) embedder. A fallback that returns plausible-looking
    vectors would make every cosine/BERTScore number downstream meaningless
    without telling the user — that directly undermines the "show raw,
    honest scores" design of this project. If the real model can't load,
    the request must fail loudly so the user knows the scores they're
    about to see (or already trusted) aren't real.
    """
    pass


class EmbeddingService:
    """
    Lazy-loading SentenceTransformer embedding service.
    Models are loaded into memory only when first invoked.
    """
    _models: Dict[str, any] = {}

    # Cache dimensions only after a real model has successfully loaded once,
    # so we never report a dimension for a model we haven't actually verified.
    _known_dimensions: Dict[str, int] = {}

    @classmethod
    def get_dimension(cls, model_name: str) -> int:
        if model_name not in ALLOWED_EMBEDDING_MODELS:
            raise EmbeddingError(
                f"Model '{model_name}' is not in the allowed models: {ALLOWED_EMBEDDING_MODELS}"
            )
        if model_name in cls._known_dimensions:
            return cls._known_dimensions[model_name]

        # Not loaded yet — load it now so the dimension we report is real,
        # not an assumed constant that could silently drift from the actual
        # model output.
        model = cls.load_model(model_name)
        dim = int(model.get_sentence_embedding_dimension())
        cls._known_dimensions[model_name] = dim
        return dim

    @classmethod
    def load_model(cls, model_name: str):
        if model_name not in ALLOWED_EMBEDDING_MODELS:
            raise EmbeddingError(f"Model '{model_name}' is not allowed. Choose from {ALLOWED_EMBEDDING_MODELS}")

        if model_name not in cls._models:
            try:
                from sentence_transformers import SentenceTransformer
                cls._models[model_name] = SentenceTransformer(model_name)
            except Exception as e:
                # No fallback. Surface the real problem (offline, model not
                # cached, disk/network issue, bad model name) so it can
                # actually be fixed, instead of quietly serving fake vectors.
                raise EmbeddingError(
                    f"Failed to load embedding model '{model_name}': {e}. "
                    "Check network access / that the model is cached locally. "
                    "Refusing to fall back to a fake embedder, since that would "
                    "silently invalidate every similarity score computed with it."
                ) from e
        return cls._models[model_name]

    @classmethod
    def embed_texts(cls, texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        if not texts:
            dim = cls.get_dimension(model_name)
            return np.empty((0, dim), dtype=np.float32)

        model = cls.load_model(model_name)
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e
        return np.array(embeddings, dtype=np.float32)

    @classmethod
    def embed_query(cls, query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
        res = cls.embed_texts([query], model_name=model_name)
        return res[0]
