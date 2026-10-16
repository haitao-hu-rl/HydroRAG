from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from pydantic import BaseModel, Field

from .agent import HydroRAGAgent
from .audit import configure_audit_logger
from .config import Settings
from .documents import SUPPORTED_EXTENSIONS, extract_document, ocr_health
from .jobs import IndexJobStore
from .metrics import Metrics
from .rate_limit import RateLimiter
from .vectorstore import LocalChromaStore


settings = Settings.from_env()
settings.validate_auth()
settings.ensure_dirs()
audit_logger = configure_audit_logger(settings.audit_log)
job_store = IndexJobStore(settings.jobs_db)
metrics = Metrics()
rate_limiter = RateLimiter()
index_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hydrorag-index")
index_lock = threading.Lock()
store = LocalChromaStore(
    settings.chroma_dir,
    settings.embedding_dim,
    embedding_backend=settings.embedding_backend,
    embedding_model=settings.embedding_model,
    reranker_enabled=settings.reranker_enabled,
    reranker_model=settings.reranker_model,
    retrieval_candidates=settings.retrieval_candidates,
    reranker_candidates=settings.reranker_candidates,
    vector_weight=settings.vector_weight,
    bm25_weight=settings.bm25_weight,
    metadata_weight=settings.metadata_weight,
)
agent = HydroRAGAgent(store, settings)
static_dir = Path(__file__).resolve().parent / "static"


class ChatInput(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    document_ids: list[str] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=20)
    thread_id: str = Field(default="web", min_length=1, max_length=100)
    user_id: str = Field(default="local", min_length=1, max_length=100)


def invoke_agent(payload: ChatInput | dict[str, Any]) -> dict[str, Any]:
    request = payload if isinstance(payload, ChatInput) else ChatInput.model_validate(payload)
    started = time.perf_counter()
    result = agent.run(
        query=request.query,
        document_ids=request.document_ids,
        top_k=request.top_k,
        thread_id=request.thread_id,
        user_id=request.user_id,
    )
    audit_logger.info(
        "chat_completed",
        extra={
            "query_sha256": hashlib.sha256(request.query.encode("utf-8")).hexdigest(),
            "user_id_sha256": hashlib.sha256(request.user_id.encode("utf-8")).hexdigest(),
            "thread_id_sha256": hashlib.sha256(request.thread_id.encode("utf-8")).hexdigest(),
            "grounded": result.get("grounded"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )
    return result


app = FastAPI(
    title="HydroRAG",
    description="面向企业工程资料的检索增强问答智能体",
    version="0.1.0",
)


@app.middleware("http")
async def api_authentication(request, call_next):
    started = time.perf_counter()
    request_id = secrets.token_hex(8)
    protected = request.url.path.startswith("/api/") and request.url.path != "/api/health"
    if settings.api_token and protected:
        if request.headers.get("authorization", "") != f"Bearer {settings.api_token}":
            metrics.record_http(401, (time.perf_counter() - started) * 1000)
            return JSONResponse(status_code=401, content={"detail": "需要有效的 Bearer Token"})
    identity = request.headers.get("authorization", "")
    if not identity and request.client:
        identity = request.client.host
    rate_limited = protected and not (
        request.method == "GET" and request.url.path.startswith("/api/jobs/")
    )
    if rate_limited:
        rate_key = hashlib.sha256((identity or "local").encode("utf-8")).hexdigest()
        allowed, retry_after = rate_limiter.allow(rate_key, settings.rate_limit_per_minute)
        if not allowed:
            metrics.increment("rate_limit_rejections_total")
            metrics.record_http(429, (time.perf_counter() - started) * 1000)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={"detail": "请求过于频繁，请稍后重试"},
            )
    try:
        response = await call_next(request)
    except Exception as exc:
        metrics.increment("http_exceptions_total")
        audit_logger.error(
            "http_exception",
            extra={"method": request.method, "path": request.url.path, "error_type": type(exc).__name__},
        )
        raise
    latency_ms = (time.perf_counter() - started) * 1000
    metrics.record_http(response.status_code, latency_ms)
    audit_path = re.sub(r"/api/(jobs|conversations|memories)/[^/]+", r"/api/\1/{id}", request.url.path)
    audit_logger.info(
        "http_completed",
        extra={
            "method": request.method,
            "path": audit_path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 1),
            "request_id": request_id,
        },
    )
    return response


chat_runnable = RunnableLambda(invoke_agent).with_types(input_type=ChatInput)
add_routes(app, chat_runnable, path="/api/chat", enabled_endpoints=["invoke", "batch", "stream"])


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "model": settings.deepseek_model,
        "deepseek_configured": bool(settings.deepseek_api_key),
        "api_auth_enabled": bool(settings.api_token),
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "reranker_enabled": settings.reranker_enabled,
        "reranker_model": settings.reranker_model,
        "documents": len(store.documents()),
        "chunks": store.count,
        "index_jobs": job_store.counts(),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "ocr": ocr_health(
            settings.ocr_language,
            settings.tesseract_cmd,
            settings.tessdata_dir,
        ),
    }


@app.get("/api/documents")
def documents() -> dict[str, Any]:
    return {"documents": store.documents(), "chunks": store.count}


@app.get("/api/metrics")
def metrics_endpoint() -> dict[str, Any]:
    return metrics.snapshot({"index_jobs": job_store.counts()})


@app.get("/api/jobs/{job_id}")
def index_job(job_id: str) -> dict[str, Any]:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="索引任务不存在")
    return {"job": job}


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="支持 PDF、图片、DOCX、XLSX 和 CSV 文件",
        )
    safe_name = re.sub(r"[^\w.()\-\u4e00-\u9fff ]+", "_", filename).strip() or f"document{suffix}"
    target = settings.documents_dir / safe_name
    if job_store.active_for_path(str(target.resolve())):
        raise HTTPException(status_code=409, detail="同一文件已有索引任务在执行")
    content = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"文件不能超过 {settings.max_upload_mb} MB")
    target.write_bytes(content)
    job_id = job_store.create(str(target.resolve()), safe_name)
    index_executor.submit(_run_index_job, job_id, target)
    metrics.increment("index_jobs_submitted_total")
    audit_logger.info("index_job_submitted", extra={"job_id": job_id, "file_name": safe_name})
    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "pending", "file_name": safe_name},
    )


def _run_index_job(job_id: str, target: Path) -> None:
    job_store.mark_running(job_id)
    try:
        started = time.perf_counter()
        with index_lock:
            document, chunks = extract_document(target, settings)
            store.delete_document(document["document_id"])
            store.upsert(chunks)
        job_store.complete(job_id, document["document_id"], len(chunks))
        metrics.increment("index_jobs_succeeded_total")
        metrics.observe("index_latency_ms", (time.perf_counter() - started) * 1000)
        audit_logger.info(
            "index_job_completed",
            extra={"job_id": job_id, "file_name": target.name, "document_id": document["document_id"], "chunk_count": len(chunks)},
        )
    except Exception as exc:
        job_store.fail(job_id, type(exc).__name__)
        metrics.increment("index_jobs_failed_total")
        audit_logger.error(
            "index_job_failed",
            extra={"job_id": job_id, "file_name": target.name, "error_type": type(exc).__name__},
        )


for _pending_job in job_store.recover_after_restart():
    _pending_path = Path(_pending_job["source_path"])
    if _pending_path.is_file():
        index_executor.submit(_run_index_job, _pending_job["job_id"], _pending_path)
    else:
        job_store.fail(_pending_job["job_id"], "uploaded file missing after restart")


@app.get("/api/memories")
def memories(user_id: str = "local") -> dict[str, Any]:
    return {"memories": agent.memory.list_long_term(user_id)}


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: int, user_id: str = "local") -> dict[str, Any]:
    deleted = agent.memory.delete_long_term(user_id, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="长期记忆不存在")
    return {"deleted": True}


@app.delete("/api/conversations/{thread_id}")
def clear_conversation(thread_id: str, user_id: str = "local") -> dict[str, Any]:
    agent.memory.clear_conversation(user_id, thread_id)
    return {"cleared": True}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


app.mount("/static", StaticFiles(directory=static_dir), name="static")
