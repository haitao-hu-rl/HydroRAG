from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .models import DocumentChunk, SearchHit


_GENERIC_METADATA_TERMS = {
    "中线",
    "南水北调",
    "工程",
    "资料",
    "数据",
    "研究",
    "模型",
    "方案",
    "报告",
}
_SUMMARY_QUERY_TERMS = (
    "多少",
    "数量",
    "总数",
    "覆盖",
    "范围",
    "记录数",
    "统计",
    "表头",
    "字段",
    "单位",
)
_METADATA_PHRASE_STOPS = {"中线", "工程", "资料", "数据", "水量", "调度", "研究"}


class LocalHashEmbeddingFunction:
    """Deterministic fallback used by isolated tests and offline migrations."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def __call__(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in input]

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(input, list):
            return [self._embed(text) for text in input]
        return self._embed(input)

    def name(self) -> str:
        return f"hydrorag-hash-{self.dimension}"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())
        for index, token in enumerate(tokens):
            self._add(vector, token, 1.0)
            if index + 1 < len(tokens):
                self._add(vector, f"{token}|{tokens[index + 1]}", 0.35)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def _add(self, vector: list[float], token: str, weight: float) -> None:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest, "little") % self.dimension
        vector[index] += weight


class LocalFastEmbedFunction:
    """ONNX Runtime embedding wrapper for the production local retrieval path."""

    def __init__(self, model_name: str, cache_dir: Path):
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(input)]

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        texts = [input] if isinstance(input, str) else input
        vectors = self.embed_documents(texts)
        return vectors[0] if isinstance(input, str) else vectors

    def name(self) -> str:
        return f"hydrorag-fastembed-{self.model_name}"


class LocalChromaStore:
    def __init__(
        self,
        path: Path,
        embedding_dim: int = 384,
        embedding_backend: str = "hash",
        embedding_model: str = "",
        reranker_enabled: bool = False,
        reranker_model: str = "",
        retrieval_candidates: int = 40,
        reranker_candidates: int = 24,
        vector_weight: float = 0.50,
        bm25_weight: float = 0.35,
        metadata_weight: float = 0.15,
    ):
        import chromadb

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.embedding_backend = embedding_backend or "hash"
        self.embedding_model = embedding_model
        self.reranker_enabled = reranker_enabled
        self.reranker_model = reranker_model
        self.retrieval_candidates = max(1, retrieval_candidates)
        self.reranker_candidates = max(1, reranker_candidates)
        self.vector_weight = max(0.0, vector_weight)
        self.bm25_weight = max(0.0, bm25_weight)
        self.metadata_weight = max(0.0, metadata_weight)
        model_cache = self.path.parent / "models"
        model_cache.mkdir(parents=True, exist_ok=True)
        if self.embedding_backend == "fastembed":
            self._embedding = LocalFastEmbedFunction(embedding_model, model_cache)
        elif self.embedding_backend == "hash":
            self._embedding = LocalHashEmbeddingFunction(embedding_dim)
        else:
            raise ValueError(f"Unsupported embedding backend: {embedding_backend}")
        signature = f"{self.embedding_backend}:{embedding_model or embedding_dim}"
        suffix = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:10]
        self.collection_name = f"hydrorag_chunks_{suffix}"
        self._client = chromadb.PersistentClient(path=str(self.path))
        self._collection = self._create_collection()
        self._lexical_dirty = True
        self._lexical_records: list[dict[str, Any]] = []
        self._lexical_tokens: list[list[str]] = []
        self._lexical_document_frequency: Counter[str] = Counter()
        self._reranker: Any | None = None

    def _create_collection(self):
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_backend": self.embedding_backend,
                "embedding_model": self.embedding_model,
            },
            embedding_function=self._embedding,
        )

    @property
    def count(self) -> int:
        return int(self._collection.count())

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=self._embedding.embed_documents(
                [_chunk_search_text(chunk) for chunk in chunks]
            ),
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "title": chunk.title,
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_type": chunk.chunk_type,
                    "source_path": chunk.source_path,
                    "source_type": chunk.source_type,
                    "location": chunk.location,
                    "file_name": chunk.file_name,
                    "sheet_name": chunk.sheet_name,
                    "headers": chunk.headers,
                }
                for chunk in chunks
            ],
        )
        self._lexical_dirty = True

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={"document_id": document_id})
        self._lexical_dirty = True

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._create_collection()
        self._lexical_dirty = True

    def query(
        self,
        query: str,
        top_k: int = 6,
        document_ids: list[str] | None = None,
    ) -> list[SearchHit]:
        if self.count == 0:
            return []
        self._ensure_lexical_index()
        candidate_limit = min(max(top_k, self.retrieval_candidates), self.count)
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": candidate_limit,
            "include": ["documents", "metadatas", "distances"],
        }
        if document_ids:
            kwargs["where"] = (
                {"document_id": document_ids[0]}
                if len(document_ids) == 1
                else {"document_id": {"$in": document_ids}}
            )
        vector_hits = self._query_hits(self._collection.query(**kwargs))
        vector_by_id = {hit.chunk_id: hit for hit in vector_hits}
        vector_scores = {
            hit.chunk_id: 1.0 / (1.0 + max(hit.distance or 0.0, 0.0))
            for hit in vector_hits
        }

        lexical_scores = self._bm25_scores(query, set(document_ids or []))
        lexical_ranked = sorted(lexical_scores, key=lexical_scores.get, reverse=True)[
            :candidate_limit
        ]
        lexical_by_id = {item: lexical_scores[item] for item in lexical_ranked}
        records_by_id = {record["chunk_id"]: record for record in self._lexical_records}
        metadata_ids = {
            record["chunk_id"]
            for record in self._lexical_records
            if (not document_ids or record["document_id"] in document_ids)
            and _has_metadata_match(query, record)
        }
        candidate_ids = (
            set(vector_by_id) | set(lexical_by_id) | metadata_ids
        ) & set(records_by_id)
        if not candidate_ids:
            return []

        vector_norm = _normalise({item: vector_scores.get(item, 0.0) for item in candidate_ids})
        lexical_norm = _normalise({item: lexical_by_id.get(item, 0.0) for item in candidate_ids})
        metadata_norm = {
            item: _metadata_score(query, records_by_id[item]) for item in candidate_ids
        }
        summary_query = any(term in query for term in _SUMMARY_QUERY_TERMS)
        scores = {
            item: (
                self.vector_weight * vector_norm[item]
                + self.bm25_weight * lexical_norm[item]
                + self.metadata_weight * metadata_norm[item]
                + 1.00 * int(_has_metadata_match(query, records_by_id[item]))
                + 0.75 * _exact_metadata_score(query, records_by_id[item])
                + 1.00
                * int(summary_query and records_by_id[item]["chunk_type"] == "table_summary")
                + 0.75 * _metadata_phrase_score(query, records_by_id[item])
            )
            for item in candidate_ids
        }
        ranked_ids = sorted(candidate_ids, key=scores.get, reverse=True)
        hits = [
            _record_to_hit(
                records_by_id[item],
                distance=vector_by_id.get(item).distance if item in vector_by_id else None,
                score=scores[item],
            )
            for item in ranked_ids
        ]
        if self.reranker_enabled and hits:
            rerank_count = min(self.reranker_candidates, len(hits))
            exact_hits = [
                hit
                for hit in hits
                if _exact_metadata_score(
                    query,
                    {
                        "title": hit.title,
                        "file_name": hit.file_name,
                        "sheet_name": hit.sheet_name,
                    },
                )
                > 0
            ]
            exact_hits.sort(
                key=lambda hit: _metadata_phrase_score(
                    query,
                    {
                        "title": hit.title,
                        "file_name": hit.file_name,
                        "sheet_name": hit.sheet_name,
                    },
                ),
                reverse=True,
            )
            rerank_pool = _diverse_prefix(exact_hits + hits, rerank_count)
            reranked_ids = {hit.chunk_id for hit in rerank_pool}
            remaining = [hit for hit in hits if hit.chunk_id not in reranked_ids]
            hits = self._rerank(query, rerank_pool) + remaining
        return hits[:top_k]

    def _query_hits(self, result: dict[str, Any]) -> list[SearchHit]:
        documents = (result.get("documents") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        hits: list[SearchHit] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] or {}
            hits.append(
                SearchHit(
                    chunk_id=str(ids[index]) if index < len(ids) else "",
                    document_id=str(metadata.get("document_id", "")),
                    title=str(metadata.get("title", "")),
                    page=int(metadata.get("page", 0)),
                    section=str(metadata.get("section", "")),
                    chunk_type=str(metadata.get("chunk_type", "text")),
                    text=str(text),
                    distance=float(distances[index]) if index < len(distances) else None,
                    source_type=str(metadata.get("source_type", "pdf")),
                    location=str(metadata.get("location", "")),
                    file_name=str(metadata.get("file_name", "")),
                    sheet_name=str(metadata.get("sheet_name", "")),
                    headers=str(metadata.get("headers", "")),
                )
            )
        return hits

    def _ensure_lexical_index(self) -> None:
        if not self._lexical_dirty:
            return
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        self._lexical_records = []
        self._lexical_tokens = []
        self._lexical_document_frequency = Counter()
        for index, text in enumerate(documents):
            metadata = metadatas[index] or {}
            record = {
                "chunk_id": str(ids[index]) if index < len(ids) else f"row-{index}",
                "text": str(text),
                "document_id": str(metadata.get("document_id", "")),
                "title": str(metadata.get("title", "")),
                "page": int(metadata.get("page", 0)),
                "section": str(metadata.get("section", "")),
                "chunk_type": str(metadata.get("chunk_type", "text")),
                "source_type": str(metadata.get("source_type", "pdf")),
                "location": str(metadata.get("location", "")),
                "file_name": str(metadata.get("file_name", "")),
                "sheet_name": str(metadata.get("sheet_name", "")),
                "headers": str(metadata.get("headers", "")),
            }
            tokens = _tokenize(_record_lexical_text(record))
            self._lexical_records.append(record)
            self._lexical_tokens.append(tokens)
            self._lexical_document_frequency.update(set(tokens))
        self._lexical_dirty = False

    def _bm25_scores(self, query: str, document_ids: set[str]) -> dict[str, float]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self._lexical_records:
            return {}
        total_docs = len(self._lexical_records)
        average_length = sum(len(tokens) for tokens in self._lexical_tokens) / total_docs
        query_terms = set(query_tokens)
        scores: dict[str, float] = {}
        for record, tokens in zip(self._lexical_records, self._lexical_tokens):
            if document_ids and record["document_id"] not in document_ids:
                continue
            term_counts = Counter(tokens)
            length = len(tokens) or 1
            score = 0.0
            for term in query_terms:
                document_frequency = self._lexical_document_frequency.get(term, 0)
                if not document_frequency:
                    continue
                idf = math.log(
                    1.0 + (total_docs - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                frequency = term_counts.get(term, 0)
                denominator = frequency + 1.5 * (0.25 + 0.75 * length / max(average_length, 1))
                score += idf * frequency * 2.5 / denominator if frequency else 0.0
            if score > 0:
                scores[record["chunk_id"]] = score
        return scores

    def _rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        try:
            if self._reranker is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                self._reranker = TextCrossEncoder(
                    model_name=self.reranker_model,
                    cache_dir=str(self.path.parent / "models"),
                )
            documents = [_hit_search_text(hit) for hit in hits]
            scores = list(self._reranker.rerank(query, documents))
        except Exception:
            return hits
        rerank_norm = _normalise(
            {str(index): float(score) for index, score in enumerate(scores)}
        )
        final_scores = {}
        for index, hit in enumerate(hits):
            metadata = {
                "title": hit.title,
                "file_name": hit.file_name,
                "sheet_name": hit.sheet_name,
                "section": hit.section,
                "headers": hit.headers,
            }
            final_scores[index] = (
                0.65 * rerank_norm.get(str(index), 0.0)
                + 0.20 * (hit.score or 0.0)
                + 0.15 * _metadata_score(query, metadata)
                + 1.00 * int(_has_metadata_match(query, metadata))
                + 0.75 * _exact_metadata_score(query, metadata)
                + 0.75
                * int(
                    any(term in query for term in _SUMMARY_QUERY_TERMS)
                    and hit.chunk_type == "table_summary"
                )
                + 0.75 * _metadata_phrase_score(query, metadata)
            )
        return [
            hit
            for index, hit in sorted(
                enumerate(hits), key=lambda pair: final_scores[pair[0]], reverse=True
            )
        ]

    def table_chunks(self, document_ids: list[str] | None = None) -> list[SearchHit]:
        return self._typed_chunks(document_ids, "table")

    def summary_chunks(self, document_ids: list[str] | None = None) -> list[SearchHit]:
        return self._typed_chunks(document_ids, "table_summary")

    def _typed_chunks(
        self,
        document_ids: list[str] | None,
        chunk_type: str,
    ) -> list[SearchHit]:
        if self.count == 0:
            return []
        result = self._collection.get(
            where={"chunk_type": chunk_type},
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        hits: list[SearchHit] = []
        for index, (text, metadata) in enumerate(
            zip(result.get("documents") or [], result.get("metadatas") or [])
        ):
            metadata = metadata or {}
            document_id = str(metadata.get("document_id", ""))
            if document_ids and document_id not in document_ids:
                continue
            hits.append(
                SearchHit(
                    chunk_id=str(ids[index]) if index < len(ids) else "",
                    document_id=document_id,
                    title=str(metadata.get("title", "")),
                    page=int(metadata.get("page", 0)),
                    section=str(metadata.get("section", "")),
                    chunk_type=chunk_type,
                    text=str(text),
                    source_type=str(metadata.get("source_type", "pdf")),
                    location=str(metadata.get("location", "")),
                    file_name=str(metadata.get("file_name", "")),
                    sheet_name=str(metadata.get("sheet_name", "")),
                    headers=str(metadata.get("headers", "")),
                )
            )
        return hits

    def documents(self) -> list[dict[str, str]]:
        if self.count == 0:
            return []
        result = self._collection.get(include=["metadatas"])
        seen: dict[str, dict[str, str]] = {}
        for metadata in result.get("metadatas") or []:
            metadata = metadata or {}
            document_id = str(metadata.get("document_id", ""))
            if document_id and document_id not in seen:
                seen[document_id] = {
                    "document_id": document_id,
                    "title": str(metadata.get("title", "")),
                    "source_path": str(metadata.get("source_path", "")),
                    "source_type": str(metadata.get("source_type", "pdf")),
                }
        return sorted(seen.values(), key=lambda item: item["title"].lower())


def _chunk_search_text(chunk: DocumentChunk) -> str:
    return _labelled_text(
        chunk.title,
        chunk.file_name,
        chunk.sheet_name,
        chunk.section,
        chunk.headers,
        chunk.text,
    )


def _record_search_text(record: dict[str, Any]) -> str:
    return _labelled_text(
        record["title"],
        record["file_name"],
        record["sheet_name"],
        record["section"],
        record["headers"],
        record["text"],
    )


def _record_lexical_text(record: dict[str, Any]) -> str:
    # Repeat short structural fields so BM25 can distinguish a matching
    # workbook title from a long report that merely shares common characters.
    return " ".join(
        [
            record["title"] * 4,
            record["file_name"] * 3,
            record["sheet_name"] * 2,
            record["section"],
            record["headers"],
            record["text"],
        ]
    )


def _hit_search_text(hit: SearchHit) -> str:
    return _labelled_text(
        hit.title,
        hit.file_name,
        hit.sheet_name,
        hit.section,
        hit.headers,
        hit.text,
    )


def _labelled_text(
    title: str,
    file_name: str,
    sheet_name: str,
    section: str,
    headers: str,
    text: str,
) -> str:
    parts = [
        f"文件标题：{title}" if title else "",
        f"文件名：{file_name}" if file_name else "",
        f"工作表：{sheet_name}" if sheet_name else "",
        f"章节：{section}" if section else "",
        f"表头：{headers}" if headers else "",
        f"正文：{text}",
    ]
    return "\n".join(item for item in parts if item)


def _record_to_hit(
    record: dict[str, Any],
    distance: float | None,
    score: float | None,
) -> SearchHit:
    return SearchHit(
        chunk_id=record["chunk_id"],
        document_id=record["document_id"],
        title=record["title"],
        page=record["page"],
        section=record["section"],
        chunk_type=record["chunk_type"],
        text=record["text"],
        distance=distance,
        source_type=record["source_type"],
        location=record["location"],
        file_name=record["file_name"],
        sheet_name=record["sheet_name"],
        headers=record["headers"],
        score=score,
    )


def _diverse_prefix(hits: list[SearchHit], limit: int) -> list[SearchHit]:
    selected: list[SearchHit] = []
    seen_documents: set[str] = set()
    for hit in hits:
        if hit.document_id in seen_documents:
            continue
        selected.append(hit)
        seen_documents.add(hit.document_id)
        if len(selected) == limit:
            return selected
    for hit in hits:
        if hit not in selected:
            selected.append(hit)
            if len(selected) == limit:
                break
    return selected


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for part in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[index : index + 2] for index in range(len(part) - 1))
        else:
            tokens.append(part)
    return tokens


def _normalise(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    minimum = min(values.values())
    maximum = max(values.values())
    if maximum == minimum:
        return {key: 1.0 if maximum > 0 else 0.0 for key in values}
    return {key: (value - minimum) / (maximum - minimum) for key, value in values.items()}


def _metadata_score(query: str, record: dict[str, Any]) -> float:
    normalized_query = re.sub(r"\s+", "", query).lower()
    exact_score = 0.0
    for value, weight in (
        (record["title"], 1.0),
        (record["file_name"], 0.8),
        (record["sheet_name"], 0.6),
    ):
        normalized_value = re.sub(r"\s+", "", value).lower()
        if (
            normalized_value
            and normalized_value not in _GENERIC_METADATA_TERMS
            and len(normalized_value) >= 2
            and normalized_value in normalized_query
        ):
            exact_score = max(exact_score, weight)
        exact_score = max(
            exact_score,
            _longest_metadata_phrase(normalized_query, normalized_value),
        )
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return exact_score
    fields = (
        (record["title"], 3.0),
        (record["file_name"], 2.0),
        (record["sheet_name"], 1.5),
        (record["section"], 1.0),
        (record["headers"], 1.0),
    )
    total = sum(weight for _, weight in fields)
    score = 0.0
    for value, weight in fields:
        field_tokens = query_tokens & set(_tokenize(value))
        score += weight * len(field_tokens) / len(query_tokens)
    return min(1.0, max(exact_score, score / total))


def _has_metadata_match(query: str, record: dict[str, Any]) -> bool:
    normalized_query = re.sub(r"\s+", "", query).lower()
    for value in (record["title"], record["file_name"], record["sheet_name"]):
        normalized_value = re.sub(r"\s+", "", value).lower()
        if (
            normalized_value
            and normalized_value not in _GENERIC_METADATA_TERMS
            and len(normalized_value) >= 2
            and normalized_value in normalized_query
        ):
            return True
        if _longest_metadata_phrase(normalized_query, normalized_value) >= 0.5:
            return True
    return False


def _longest_metadata_phrase(query: str, value: str) -> float:
    if not query or not value:
        return 0.0
    max_length = min(len(query), 16)
    for length in range(max_length, 2, -1):
        for start in range(len(query) - length + 1):
            phrase = query[start : start + length]
            if (
                phrase in value
                and phrase not in _GENERIC_METADATA_TERMS
                and phrase not in _METADATA_PHRASE_STOPS
                and "南水北" not in phrase
                and "北调" not in phrase
                and "工程" not in phrase
            ):
                return min(1.0, length / 8)
    return 0.0


def _metadata_phrase_score(query: str, record: dict[str, Any]) -> float:
    normalized_query = re.sub(r"\s+", "", query).lower()
    return max(
        (
            _longest_metadata_phrase(
                normalized_query,
                re.sub(r"\s+", "", value).lower(),
            )
            for value in (
                record.get("title", ""),
                record.get("file_name", ""),
                record.get("sheet_name", ""),
            )
        ),
        default=0.0,
    )


def _exact_metadata_score(query: str, record: dict[str, Any]) -> float:
    normalized_query = re.sub(r"\s+", "", query).lower()
    best = 0.0
    for value, weight in (
        (record.get("title", ""), 1.0),
        (record.get("file_name", ""), 0.8),
        (record.get("sheet_name", ""), 0.6),
    ):
        normalized_value = re.sub(r"\s+", "", value).lower()
        if (
            normalized_value
            and normalized_value not in _GENERIC_METADATA_TERMS
            and len(normalized_value) >= 2
            and normalized_value in normalized_query
        ):
            best = max(best, weight)
    return best
