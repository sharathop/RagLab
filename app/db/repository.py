from datetime import datetime, timedelta, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.db.models import SessionModel, ConfigModel, QueryLogModel, EvaluationScoreModel


def utc_now():
    return datetime.now(timezone.utc)


class Repository:
    @staticmethod
    def create_session(
        db: Session,
        session_id: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        ttl_minutes: int = 30
    ) -> Optional[SessionModel]:
        try:
            now = utc_now()
            expires_at = now + timedelta(minutes=ttl_minutes)
            session_obj = SessionModel(
                id=session_id,
                created_at=now,
                last_accessed=now,
                expires_at=expires_at,
                status="active",
                embedding_model=embedding_model
            )
            db.add(session_obj)
            db.commit()
            db.refresh(session_obj)
            return session_obj
        except SQLAlchemyError:
            db.rollback()
            return None

    @staticmethod
    def get_session(db: Session, session_id: str) -> Optional[SessionModel]:
        try:
            return db.query(SessionModel).filter(SessionModel.id == session_id).first()
        except SQLAlchemyError:
            return None

    @staticmethod
    def touch_session(db: Session, session_id: str, ttl_minutes: int = 30) -> bool:
        try:
            session_obj = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            if session_obj:
                now = utc_now()
                session_obj.last_accessed = now
                session_obj.expires_at = now + timedelta(minutes=ttl_minutes)
                db.commit()
                return True
            return False
        except SQLAlchemyError:
            db.rollback()
            return False

    @staticmethod
    def expire_session(db: Session, session_id: str) -> bool:
        try:
            session_obj = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            if session_obj:
                session_obj.status = "expired"
                db.commit()
                return True
            return False
        except SQLAlchemyError:
            db.rollback()
            return False

    @staticmethod
    def save_config(
        db: Session,
        session_id: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_model: str,
        top_k: int,
        reranking_enabled: bool
    ) -> Optional[ConfigModel]:
        try:
            config = ConfigModel(
                session_id=session_id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                embedding_model=embedding_model,
                top_k=top_k,
                reranking_enabled=reranking_enabled,
                created_at=utc_now()
            )
            db.add(config)
            db.commit()
            db.refresh(config)
            return config
        except SQLAlchemyError:
            db.rollback()
            return None

    @staticmethod
    def get_latest_config(db: Session, session_id: str) -> Optional[ConfigModel]:
        try:
            return (
                db.query(ConfigModel)
                .filter(ConfigModel.session_id == session_id)
                .order_by(ConfigModel.created_at.desc())
                .first()
            )
        except SQLAlchemyError:
            return None

    @staticmethod
    def get_all_configs(db: Session, session_id: str) -> List[ConfigModel]:
        try:
            return (
                db.query(ConfigModel)
                .filter(ConfigModel.session_id == session_id)
                .order_by(ConfigModel.created_at.asc())
                .all()
            )
        except SQLAlchemyError:
            return []

    @staticmethod
    def save_query_log(
        db: Session,
        session_id: str,
        question: str,
        answer: str,
        config_id: Optional[int],
        retrieval_time_ms: float,
        llm_time_ms: float,
        evaluation_time_ms: float,
        total_time_ms: float
    ) -> Optional[QueryLogModel]:
        try:
            log = QueryLogModel(
                session_id=session_id,
                question=question,
                answer=answer,
                config_id=config_id,
                retrieval_time_ms=retrieval_time_ms,
                llm_time_ms=llm_time_ms,
                evaluation_time_ms=evaluation_time_ms,
                total_time_ms=total_time_ms,
                created_at=utc_now()
            )
            db.add(log)
            db.commit()
            db.refresh(log)
            return log
        except SQLAlchemyError:
            db.rollback()
            return None

    @staticmethod
    def save_evaluation_scores(
        db: Session,
        query_id: int,
        nli_score: float,
        cosine_score: float,
        bertscore: float,
        fluency_score: float = 0.0  # fluency no longer computed for this project;
                                     # column kept as 0.0 rather than a new migration
                                     # to drop it. See EvaluatorService docstring.
    ) -> Optional[EvaluationScoreModel]:
        try:
            eval_score = EvaluationScoreModel(
                query_id=query_id,
                nli_score=nli_score,
                cosine_score=cosine_score,
                bertscore=bertscore,
                fluency_score=fluency_score
            )
            db.add(eval_score)
            db.commit()
            db.refresh(eval_score)
            return eval_score
        except SQLAlchemyError:
            db.rollback()
            return None

    @staticmethod
    def get_session_history(db: Session, session_id: str) -> List[QueryLogModel]:
        try:
            return (
                db.query(QueryLogModel)
                .filter(QueryLogModel.session_id == session_id)
                .order_by(QueryLogModel.created_at.desc())
                .all()
            )
        except SQLAlchemyError:
            return []
