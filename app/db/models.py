import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Boolean,
    Text,
    ForeignKey
)
from sqlalchemy.orm import relationship
from app.db.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    last_accessed = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    status = Column(String(20), default="active", nullable=False)  # active, expired
    embedding_model = Column(String(100), nullable=False, default="all-MiniLM-L6-v2")

    # Relationships
    configs = relationship("ConfigModel", back_populates="session", cascade="all, delete-orphan", order_by="desc(ConfigModel.created_at)")
    query_logs = relationship("QueryLogModel", back_populates="session", cascade="all, delete-orphan", order_by="desc(QueryLogModel.created_at)")


class ConfigModel(Base):
    __tablename__ = "configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    chunk_size = Column(Integer, nullable=False)
    chunk_overlap = Column(Integer, nullable=False)
    embedding_model = Column(String(100), nullable=False)
    top_k = Column(Integer, nullable=False)
    reranking_enabled = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    session = relationship("SessionModel", back_populates="configs")
    query_logs = relationship("QueryLogModel", back_populates="config")


class QueryLogModel(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    config_id = Column(Integer, ForeignKey("configs.id", ondelete="SET NULL"), nullable=True)
    retrieval_time_ms = Column(Float, nullable=False, default=0.0)
    llm_time_ms = Column(Float, nullable=False, default=0.0)
    evaluation_time_ms = Column(Float, nullable=False, default=0.0)
    total_time_ms = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    # Relationships
    session = relationship("SessionModel", back_populates="query_logs")
    config = relationship("ConfigModel", back_populates="query_logs")
    evaluation_scores = relationship("EvaluationScoreModel", back_populates="query_log", uselist=False, cascade="all, delete-orphan")

    @property
    def nli_score(self):
        return self.evaluation_scores.nli_score if self.evaluation_scores else None

    @property
    def cosine_score(self):
        return self.evaluation_scores.cosine_score if self.evaluation_scores else None

    @property
    def bertscore(self):
        return self.evaluation_scores.bertscore if self.evaluation_scores else None

    @property
    def fluency_score(self):
        return self.evaluation_scores.fluency_score if self.evaluation_scores else None


class EvaluationScoreModel(Base):
    __tablename__ = "evaluation_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_id = Column(Integer, ForeignKey("query_logs.id", ondelete="CASCADE"), nullable=False)
    nli_score = Column(Float, nullable=False, default=0.0)
    cosine_score = Column(Float, nullable=False, default=0.0)
    bertscore = Column(Float, nullable=False, default=0.0)
    fluency_score = Column(Float, nullable=False, default=0.0)

    # Relationship
    query_log = relationship("QueryLogModel", back_populates="evaluation_scores")
