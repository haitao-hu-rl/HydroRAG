from __future__ import annotations

import ast
import json
import operator
from typing import Any

from langchain_core.tools import StructuredTool

from .vectorstore import LocalChromaStore


class AgentTools:
    """Tools exposed to the graph nodes and, later, to model tool-calling."""

    def __init__(self, store: LocalChromaStore):
        self.store = store

    def search_documents(
        self,
        query: str,
        top_k: int = 6,
        document_ids: str = "",
    ) -> str:
        ids = _split_ids(document_ids)
        hits = self.store.query(query, top_k=top_k, document_ids=ids or None)
        return json.dumps([hit.document() for hit in hits], ensure_ascii=False)

    def extract_tables(self, document_id: str = "", page: int = 0) -> str:
        ids = [document_id] if document_id else None
        hits = self.store.table_chunks(ids)
        if page:
            hits = [hit for hit in hits if hit.page == page]
        return json.dumps([hit.document() for hit in hits], ensure_ascii=False)

    def calculate(self, expression: str) -> str:
        try:
            value = safe_calculate(expression)
        except ValueError as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        return json.dumps({"ok": True, "expression": expression, "value": value})

    def as_tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.search_documents,
                name="search_documents",
                description="在已入库资料中检索与问题相关的证据片段，返回原文位置和来源元数据。",
            ),
            StructuredTool.from_function(
                func=self.extract_tables,
                name="extract_tables",
                description="从已入库资料中读取表格抽取结果，可按资料 ID 和页码过滤。",
            ),
            StructuredTool.from_function(
                func=self.calculate,
                name="calculate",
                description="计算安全的四则运算表达式，适合复核资料中的简单数值关系。",
            ),
        ]


def _split_ids(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calculate(expression: str) -> float | int:
    expression = expression.strip().replace("^", "**")
    if not expression or len(expression) > 120:
        raise ValueError("表达式为空或过长")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("表达式格式不正确") from exc
    value = _eval_node(tree.body)
    if isinstance(value, complex) or abs(float(value)) > 1e100:
        raise ValueError("结果超出安全范围")
    return value


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        try:
            return _BINOPS[type(node.op)](left, right)
        except (ZeroDivisionError, OverflowError) as exc:
            raise ValueError("计算失败") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval_node(node.operand))
    raise ValueError("只允许数字和 + - * / % ^ 运算")
