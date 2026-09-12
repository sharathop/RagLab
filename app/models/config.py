from pydantic import BaseModel, Field, model_validator
from typing import Literal

ALLOWED_EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
]

EmbeddingModelType = Literal["all-MiniLM-L6-v2", "all-mpnet-base-v2"]


class PipelineConfig(BaseModel):
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Chunk size in characters or tokens (100-2000)")
    chunk_overlap: int = Field(default=100, ge=0, description="Chunk overlap (0 to chunk_size - 1)")
    embedding_model: EmbeddingModelType = Field(default="all-MiniLM-L6-v2", description="Allowed embedding model")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of retrieved chunks (1-20)")
    reranking_enabled: bool = Field(default=False, description="Enable CrossEncoder reranking")

    @model_validator(mode="after")
    def validate_overlap(self) -> "PipelineConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})")
        return self


class ConfigResponse(BaseModel):
    id: int
    version: int
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    top_k: int
    reranking_enabled: bool
    created_at: str
