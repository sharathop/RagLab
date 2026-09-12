from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    total_pages: int
    message: str


class IngestResponse(BaseModel):
    session_id: str
    total_chunks: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    ingest_time_ms: float
    message: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")


class RetrievedSource(BaseModel):
    chunk_id: str
    page: int
    score: float
    text: str
    reranked: bool = False


class EvaluationMetrics(BaseModel):
    nli_score: float = Field(description="Natural Language Inference faithfulness score (0-1), from the external hallucination-detection framework")
    nli_label: Optional[str] = Field(default=None, description="Framework's own single-metric verdict for NLI, e.g. 'Faithful' / 'Hallucinated' / 'Unverifiable'")
    cosine_score: float = Field(description="Cosine similarity between the QUESTION and the answer (0-1) — not context vs answer. See cosine_warning.")
    cosine_label: Optional[str] = Field(default=None, description="Framework's own single-metric verdict for cosine, e.g. 'Relevant' / 'Partially Relevant' / 'Irrelevant'")
    bertscore: float = Field(description="Token-level contextual BERTScore F1 between context and answer (0-1)")
    bertscore_label: Optional[str] = Field(default=None, description="Framework's own single-metric verdict for BERTScore")
    cosine_warning: Optional[str] = Field(
        default=None,
        description=(
            "Set only when the answer is much longer/more structured (e.g. a numbered list) than "
            "the question. Cosine similarity naturally drops for long, well-formed answers to short "
            "questions regardless of correctness — this flags that pattern instead of letting a low "
            "score be read as a sign of a bad answer on its own. Cross-check NLI/BERTScore first."
        )
    )


class PerformanceMetrics(BaseModel):
    parsing_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    reranking_time_ms: float = 0.0
    llm_time_ms: float = 0.0
    evaluation_time_ms: float = 0.0
    total_time_ms: float = 0.0


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[RetrievedSource]
    evaluation: EvaluationMetrics
    performance: PerformanceMetrics
    warning: Optional[str] = "Cosine similarity measures semantic similarity. It does not prove that the answer is factually supported by the context."


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    config_id: Optional[int]
    retrieval_time_ms: float
    llm_time_ms: float
    evaluation_time_ms: float
    total_time_ms: float
    created_at: str
    nli_score: Optional[float] = None
    cosine_score: Optional[float] = None
    bertscore: Optional[float] = None


class DownloadRequest(BaseModel):
    format: str = Field(default="zip", description="'file' for single script or 'zip' for project")
    components: List[str] = Field(
        default=[
            "parser",
            "chunking",
            "embeddings",
            "retrieval",
            "reranking",
            "generation",
            "evaluation",
            "fastapi",
            "web_ui"
        ]
    )


class HealthResponse(BaseModel):
    status: str
    database: str
    active_sessions: int
    embedding_models_available: List[str]
    groq_api_configured: bool
