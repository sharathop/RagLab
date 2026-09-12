from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChunkMetadata(BaseModel):
    chunk_id: str = Field(description="Unique ID of chunk, e.g., chunk_0")
    page: int = Field(description="1-based page number where chunk originated")
    text: str = Field(description="Extracted chunk text")
    token_count: Optional[int] = Field(default=None, description="Approximate token or character count")


class SessionState(BaseModel):
    session_id: str
    status: str = "active"  # active, expired
    created_at: datetime
    last_accessed: datetime
    expires_at: datetime
    filename: Optional[str] = None
    total_pages: int = 0
    total_chunks: int = 0
    embedding_model: str = "all-MiniLM-L6-v2"
