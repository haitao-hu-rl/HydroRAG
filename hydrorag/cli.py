from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import HydroRAGAgent
from .backup import create_backup, restore_backup
from .config import Settings
from .documents import SUPPORTED_EXTENSIONS, extract_document, ocr_health
from .vectorstore import LocalChromaStore


def main() -> None:
    _configure_console()
    parser = argparse.ArgumentParser(description="HydroRAG 面向企业工程资料的检索增强问答智能体")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="解析并索引文档")
    ingest.add_argument("paths", nargs="+", help="受支持的文档或目录")

    ask = sub.add_parser("ask", help="向问答智能体提问")
    ask.add_argument("query")
    ask.add_argument("--document-id", action="append", default=[], help="限定资料 ID，可重复传入")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--thread-id", default="cli")
    ask.add_argument("--user-id", default="local")

    sub.add_parser("list", help="列出已入库资料")
    sub.add_parser("health", help="检查配置与依赖")
    sub.add_parser("demo", help="显示离线演示命令")
    sub.add_parser("reset", help="删除本地向量索引")
    backup = sub.add_parser("backup", help="创建本地资料、索引和记忆备份")
    backup.add_argument("destination", nargs="?", help="可选的备份 ZIP 路径")
    restore = sub.add_parser("restore", help="从备份恢复本地资料和索引")
    restore.add_argument("archive", help="HydroRAG 备份 ZIP")
    restore.add_argument("--confirm", action="store_true", help="确认覆盖当前数据")
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()

    if args.command == "demo":
        _demo(settings)
        return
    if args.command == "health":
        _health(settings)
        return
    if args.command == "backup":
        path = create_backup(settings, Path(args.destination) if args.destination else None)
        print(f"备份已创建：{path}")
        return
    if args.command == "restore":
        try:
            safety_backup = restore_backup(settings, Path(args.archive), confirm=args.confirm)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(f"恢复完成。覆盖前的安全备份：{safety_backup}")
        print("请重新启动服务；如恢复期间有服务运行，请重建 BM25 内存索引后再提供查询。")
        return

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
    if args.command == "ingest":
        _ingest(args.paths, settings, store)
    elif args.command == "list":
        print(json.dumps(store.documents(), ensure_ascii=False, indent=2))
        print(f"chunks={store.count}")
    elif args.command == "ask":
        result = HydroRAGAgent(store, settings).run(
            args.query,
            document_ids=args.document_id,
            top_k=args.top_k,
            thread_id=args.thread_id,
            user_id=args.user_id,
        )
        print(f"[task] {result.get('task', 'unknown')}")
        print(result.get("output", ""))
        citations = result.get("citations", [])
        if citations:
            print("\n[回溯来源]")
            for citation in citations:
                print(
                    f"- {citation.get('title', '')} | p.{citation.get('page', '?')}"
                    f" | {citation.get('document_id', '')}"
                )
    elif args.command == "reset":
        store.reset()
        print(f"已清空本地索引：{settings.chroma_dir}")


def _configure_console() -> None:
    """Avoid Windows GBK failures when a PDF contains mathematical glyphs."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _ingest(raw_paths: list[str], settings: Settings, store: LocalChromaStore) -> None:
    files: list[Path] = []
    for raw in raw_paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(
                file
                for file in sorted(path.rglob("*"))
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif path.suffix.lower() in SUPPORTED_EXTENSIONS and path.exists():
            files.append(path)
        else:
            print(f"跳过不存在或不支持的路径：{path}")
    if not files:
        raise SystemExit("没有找到受支持的文档。")
    for path in files:
        document, chunks = extract_document(path, settings)
        store.delete_document(document["document_id"])
        store.upsert(chunks)
        print(
            f"已索引：{document['title']} | id={document['document_id']} | chunks={len(chunks)}"
        )


def _health(settings: Settings) -> None:
    modules = {}
    for name in (
        "pymupdf",
        "chromadb",
        "langgraph",
        "langchain_core",
        "docx",
        "openpyxl",
        "PIL",
        "pytesseract",
    ):
        try:
            __import__(name)
            modules[name] = "ok"
        except Exception as exc:
            modules[name] = f"missing: {exc}"
    print(json.dumps({"settings": {
        "root": str(settings.root_dir),
        "deepseek_key_configured": bool(settings.deepseek_api_key),
        "model": settings.deepseek_model,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
        "reranker_enabled": settings.reranker_enabled,
        "reranker_model": settings.reranker_model,
    }, "modules": modules, "ocr": ocr_health(
        settings.ocr_language,
        settings.tesseract_cmd,
        settings.tessdata_dir,
    )}, ensure_ascii=False, indent=2))


def _demo(settings: Settings) -> None:
    print("HydroRAG 离线演示")
    print("1. 将工程资料放到：", settings.documents_dir)
    print("2. 执行：python -m hydrorag ingest data/documents")
    print("3. 执行：python -m hydrorag ask \"总结这份资料的方法和主要结论\"")
    print("未配置 DEEPSEEK_API_KEY 时仍可进行文档解析和证据检索。")
