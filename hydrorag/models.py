from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    title: str
    page: int
    section: str
    chunk_type: str
    text: str
    source_path: str
    source_type: str = "pdf"
    location: str = ""
    file_name: str = ""
    sheet_name: str = ""
    headers: str = ""

    def document(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    title: str
    page: int
    section: str
    chunk_type: str
    text: str
    distance: float | None = None
    source_type: str = "pdf"
    location: str = ""
    file_name: str = ""
    sheet_name: str = ""
    headers: str = ""
    score: float | None = None

    def document(self) -> dict[str, Any]:
        return asdict(self)
