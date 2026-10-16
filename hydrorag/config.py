from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Small, environment-driven configuration object."""

    root_dir: Path
    documents_dir: Path
    chroma_dir: Path
    memory_db: Path
    jobs_db: Path
    audit_log: Path
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    top_k: int
    chunk_chars: int
    chunk_overlap: int
    embedding_dim: int
    embedding_backend: str
    embedding_model: str
    reranker_enabled: bool
    reranker_model: str
    retrieval_candidates: int
    reranker_candidates: int
    bm25_weight: float
    vector_weight: float
    metadata_weight: float
    temperature: float
    retrieval_distance_threshold: float
    short_term_turns: int
    ocr_language: str
    tesseract_cmd: str
    tessdata_dir: str
    max_upload_mb: int
    api_token: str
    require_api_token: bool
    rate_limit_per_minute: int

    @classmethod
    def from_env(cls, root_dir: Path | None = None) -> "Settings":
        root = (root_dir or Path(__file__).resolve().parents[1]).resolve()
        load_dotenv(root / ".env")
        data_dir = root / "data"
        return cls(
            root_dir=root,
            documents_dir=data_dir / "documents",
            chroma_dir=data_dir / "chroma",
            memory_db=data_dir / "memory.sqlite3",
            jobs_db=data_dir / "jobs.sqlite3",
            audit_log=data_dir / "logs" / "audit.log",
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=os.getenv(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
            ).strip(),
            deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
            top_k=_positive_int("HYDRORAG_TOP_K", 6),
            chunk_chars=_positive_int("HYDRORAG_CHUNK_CHARS", 1200),
            chunk_overlap=_positive_int("HYDRORAG_CHUNK_OVERLAP", 180),
            embedding_dim=_positive_int("HYDRORAG_EMBEDDING_DIM", 512),
            embedding_backend=os.getenv("HYDRORAG_EMBEDDING_BACKEND", "fastembed").strip(),
            embedding_model=os.getenv(
                "HYDRORAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
            ).strip(),
            reranker_enabled=_bool_env("HYDRORAG_RERANKER_ENABLED", True),
            reranker_model=os.getenv(
                "HYDRORAG_RERANKER_MODEL", "BAAI/bge-reranker-base"
            ).strip(),
            retrieval_candidates=_positive_int("HYDRORAG_RETRIEVAL_CANDIDATES", 40),
            reranker_candidates=_positive_int("HYDRORAG_RERANKER_CANDIDATES", 24),
            bm25_weight=_float_between("HYDRORAG_BM25_WEIGHT", 0.35, 0.0, 1.0),
            vector_weight=_float_between("HYDRORAG_VECTOR_WEIGHT", 0.50, 0.0, 1.0),
            metadata_weight=_float_between("HYDRORAG_METADATA_WEIGHT", 0.15, 0.0, 1.0),
            temperature=float(os.getenv("HYDRORAG_TEMPERATURE", "0.1")),
            retrieval_distance_threshold=_float_between(
                "HYDRORAG_DISTANCE_THRESHOLD", 1.15, 0.0, 2.0
            ),
            short_term_turns=_positive_int("HYDRORAG_SHORT_TERM_TURNS", 8),
            ocr_language=os.getenv("HYDRORAG_OCR_LANGUAGE", "chi_sim+eng").strip(),
            tesseract_cmd=os.getenv("HYDRORAG_TESSERACT_CMD", "").strip(),
            tessdata_dir=os.getenv("HYDRORAG_TESSDATA_DIR", "").strip(),
            max_upload_mb=_positive_int("HYDRORAG_MAX_UPLOAD_MB", 25),
            api_token=os.getenv("HYDRORAG_API_TOKEN", "").strip(),
            require_api_token=_bool_env("HYDRORAG_REQUIRE_API_TOKEN", False),
            rate_limit_per_minute=_positive_int("HYDRORAG_RATE_LIMIT_PER_MINUTE", 60),
        )

    def validate_auth(self) -> None:
        if self.require_api_token and not self.api_token:
            raise ValueError(
                "HYDRORAG_REQUIRE_API_TOKEN=true requires HYDRORAG_API_TOKEN"
            )

    def ensure_dirs(self) -> None:
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log.parent.mkdir(parents=True, exist_ok=True)


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _float_between(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
