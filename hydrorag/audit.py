from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


class _JsonFormatter(logging.Formatter):
    _fields = (
        "request_id",
        "method",
        "path",
        "status_code",
        "latency_ms",
        "query_sha256",
        "user_id_sha256",
        "thread_id_sha256",
        "user_id_sha256",
        "job_id",
        "file_name",
        "document_id",
        "chunk_count",
        "grounded",
        "error_type",
    )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for field in self._fields:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_audit_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("hydrorag.audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    resolved = str(Path(path).resolve())
    if not any(getattr(handler, "baseFilename", "") == resolved for handler in logger.handlers):
        handler = RotatingFileHandler(
            resolved,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
    return logger
