from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import pymupdf as fitz

from .config import Settings
from .models import DocumentChunk


def document_id_for(path: Path) -> str:
    """Return a stable ID for a file path so re-ingestion replaces old chunks."""

    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", path.stem).strip("-").lower()
    return f"{slug or 'document'}-{digest}"


def extract_pdf(path: Path, settings: Settings) -> tuple[dict[str, str], list[DocumentChunk]]:
    """Extract page-aware text and best-effort table chunks from one PDF."""

    path = path.resolve()
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {path}")

    document_id = document_id_for(path)
    chunks: list[DocumentChunk] = []
    with fitz.open(path) as document:
        metadata = document.metadata or {}
        title = (metadata.get("title") or "").strip() or path.stem
        author = (metadata.get("author") or "").strip()

        for page_index, page in enumerate(document, start=1):
            text = _clean_text(page.get_text("text"))
            if not text:
                text = _ocr_page(page, settings)
            section = _guess_section(text, title)
            for index, piece in enumerate(
                chunk_text(text, settings.chunk_chars, settings.chunk_overlap)
            ):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}-p{page_index}-text-{index}",
                        document_id=document_id,
                        title=title,
                        page=page_index,
                        section=section,
                        chunk_type="text",
                        text=piece,
                        source_path=str(path),
                        source_type="pdf",
                        location=f"p.{page_index}",
                        file_name=path.name,
                    )
                )

            for index, table in enumerate(_extract_tables(page)):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}-p{page_index}-table-{index}",
                        document_id=document_id,
                        title=title,
                        page=page_index,
                        section=section,
                        chunk_type="table",
                        text=table,
                        source_path=str(path),
                        source_type="pdf",
                        location=f"p.{page_index}",
                        file_name=path.name,
                    )
                )

    return {
        "document_id": document_id,
        "title": title,
        "author": author,
        "source_path": str(path),
        "source_type": "pdf",
    }, chunks


def _ocr_page(page: fitz.Page, settings: Settings) -> str:
    """OCR image-only PDF pages while keeping the existing PDF path unchanged."""

    try:
        import os

        import pytesseract
        from PIL import Image, ImageOps

        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        if settings.tessdata_dir:
            os.environ["TESSDATA_PREFIX"] = settings.tessdata_dir
        pixmap = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), alpha=False)
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        image = ImageOps.autocontrast(ImageOps.grayscale(image))
        try:
            text = pytesseract.image_to_string(
                image, lang=settings.ocr_language, config="--psm 3"
            )
        except TypeError:
            text = pytesseract.image_to_string(image, lang=settings.ocr_language)
        return _clean_text(text)
    except Exception:
        return ""


def chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text without losing page-local provenance."""

    text = _clean_text(text)
    if not text:
        return []
    max_chars = max(200, max_chars)
    overlap = max(0, min(overlap, max_chars // 2))
    paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs or [text]:
        if len(paragraph) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            sentences = [
                item.strip()
                for item in re.split(r"(?<=[。！？；.!?;])\s+", paragraph)
                if item.strip()
            ]
            if len(sentences) > 1:
                pieces.extend(_pack_sentences(sentences, max_chars, overlap))
            else:
                pieces.extend(_window(paragraph, max_chars, overlap))
            continue
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{paragraph}".strip()
    if current:
        pieces.append(current)
    return pieces


def _pack_sentences(sentences: list[str], max_chars: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_window(sentence, max_chars, overlap))
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            pieces.append(current)
            tail = current[-overlap:] if overlap else ""
            current = f"{tail} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _window(text: str, size: int, overlap: int) -> Iterable[str]:
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            yield piece
        if start + size >= len(text):
            break


def _clean_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_section(text: str, title: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line or line == title:
            continue
        if len(line) <= 90 and (re.match(r"^(\d+\.?\s+|[A-Z][A-Z\s]{3,}$)", line)):
            return line
    return "正文"


def _extract_tables(page: fitz.Page) -> list[str]:
    """Use PyMuPDF's layout-aware table API when available."""

    find_tables = getattr(page, "find_tables", None)
    if find_tables is None:
        return []
    try:
        result = find_tables()
        tables = getattr(result, "tables", [])
    except Exception:
        return []

    outputs: list[str] = []
    for table in tables:
        try:
            rows = table.extract()
        except Exception:
            continue
        lines = []
        for row in rows or []:
            cells = [re.sub(r"\s+", " ", str(cell or "").strip()) for cell in row]
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")
        if lines:
            outputs.append("抽取表格：\n" + "\n".join(lines))
    return outputs
