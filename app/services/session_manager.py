import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List
from sqlalchemy.orm import Session
from app.models.config import PipelineConfig
from app.models.session import ChunkMetadata
from app.services.vector_store import FAISSVectorStore
from app.services.embeddings import EmbeddingService
from app.db.repository import Repository

DEFAULT_TTL_MINUTES = 30


def utc_now():
    return datetime.now(timezone.utc)


class SessionData:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.vector_store: Optional[FAISSVectorStore] = None
        self.chunks: List[ChunkMetadata] = []
        # Raw parsed pages from the uploaded PDF, kept so the document can be
        # RE-chunked from scratch whenever chunk_size/chunk_overlap change —
        # not just re-embedded. Without this, changing chunk settings after
        # upload would silently update the displayed config while the actual
        # FAISS index kept serving chunks built with the OLD size/overlap.
        self.pages_data: Optional[List] = None
        self.current_config = PipelineConfig()
        self.config_id: Optional[int] = None
        self.config_version: int = 1
        self.last_accessed = utc_now()
        self.created_at = utc_now()
        self.filename: Optional[str] = None
        self.total_pages: int = 0
        self.status = "active"

    def touch(self):
        self.last_accessed = utc_now()

    def is_expired(self, ttl_minutes: int = DEFAULT_TTL_MINUTES) -> bool:
        if self.status == "expired":
            return True
        return utc_now() - self.last_accessed > timedelta(minutes=ttl_minutes)

    def init_vector_store(self, embedding_model: str):
        dim = EmbeddingService.get_dimension(embedding_model)
        self.vector_store = FAISSVectorStore(dimension=dim)

    def clear(self):
        if self.vector_store:
            self.vector_store.clear()
            self.vector_store = None
        self.chunks.clear()
        self.pages_data = None
        self.status = "expired"


class SessionManager:
    _sessions: Dict[str, SessionData] = {}

    @classmethod
    def get_or_create(cls, session_id: Optional[str], db: Optional[Session] = None) -> Tuple[str, SessionData]:
        cls.cleanup_expired(db)

        if session_id and session_id in cls._sessions:
            session_data = cls._sessions[session_id]
            if not session_data.is_expired():
                session_data.touch()
                if db:
                    Repository.touch_session(db, session_id)
                return session_id, session_data
            else:
                # Expired in memory
                cls.expire(session_id, db)

        # Generate new session
        new_id = str(uuid.uuid4())
        session_data = SessionData(session_id=new_id)
        session_data.init_vector_store(session_data.current_config.embedding_model)
        cls._sessions[new_id] = session_data

        if db:
            Repository.create_session(
                db=db,
                session_id=new_id,
                embedding_model=session_data.current_config.embedding_model,
                ttl_minutes=DEFAULT_TTL_MINUTES
            )
            cfg = Repository.save_config(
                db=db,
                session_id=new_id,
                chunk_size=session_data.current_config.chunk_size,
                chunk_overlap=session_data.current_config.chunk_overlap,
                embedding_model=session_data.current_config.embedding_model,
                top_k=session_data.current_config.top_k,
                reranking_enabled=session_data.current_config.reranking_enabled
            )
            if cfg:
                session_data.config_id = cfg.id
                session_data.config_version = 1

        return new_id, session_data

    @classmethod
    def get(cls, session_id: str, db: Optional[Session] = None) -> Optional[SessionData]:
        cls.cleanup_expired(db)
        session_data = cls._sessions.get(session_id)
        if not session_data:
            return None
        if session_data.is_expired():
            cls.expire(session_id, db)
            return None
        session_data.touch()
        if db:
            Repository.touch_session(db, session_id)
        return session_data

    @classmethod
    def expire(cls, session_id: str, db: Optional[Session] = None) -> None:
        if session_id in cls._sessions:
            session_data = cls._sessions.pop(session_id)
            session_data.clear()
        if db:
            Repository.expire_session(db, session_id)

    @classmethod
    def cleanup_expired(cls, db: Optional[Session] = None) -> int:
        """
        Purge expired sessions to prevent unbounded memory growth.
        """
        expired_ids = [
            sid for sid, data in cls._sessions.items()
            if data.is_expired()
        ]
        for sid in expired_ids:
            cls.expire(sid, db)
        return len(expired_ids)

    @classmethod
    def get_active_count(cls) -> int:
        return len([s for s in cls._sessions.values() if not s.is_expired()])