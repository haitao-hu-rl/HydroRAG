from __future__ import annotations

import json
import re
from typing import Any, Iterable, TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Settings
from .llm import build_chat_model, message_text
from .memory import SQLiteMemoryStore
from .tools import AgentTools
from .validation import audit_numeric_claims
from .vectorstore import LocalChromaStore


TASKS = {"meta", "qa", "compare", "table", "summary", "calculate"}


class AgentState(TypedDict, total=False):
    query: str
    task: str
    document_ids: list[str]
    top_k: int
    hits: list[dict[str, Any]]
    output: str
    citations: list[dict[str, Any]]
    tool_result: str
    history: list[dict[str, Any]]
    conversation_summary: str
    long_term_memories: list[dict[str, Any]]
    retrieval_query: str
    grounded: bool
    evidence_conflict: bool


class HydroRAGAgent:
    """LangGraph workflow for an internal-document research agent."""

    def __init__(self, store: LocalChromaStore, settings: Settings):
        self.store = store
        self.settings = settings
        self.model = build_chat_model(settings)
        self.memory = SQLiteMemoryStore(settings.memory_db)
        self.agent_tools = AgentTools(store)
        tools = self.agent_tools.as_tools()
        self.search_tool = next(tool for tool in tools if tool.name == "search_documents")
        self.table_tool = next(tool for tool in tools if tool.name == "extract_tables")
        self.calculate_tool = next(tool for tool in tools if tool.name == "calculate")
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("route", self._route)
        builder.add_node("meta", self._meta)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_node("table", self._table)
        builder.add_node("calculate", self._calculate)
        builder.add_edge(START, "route")
        builder.add_conditional_edges(
            "route",
            lambda state: state["task"],
            {
                "meta": "meta",
                "qa": "retrieve",
                "summary": "retrieve",
                "compare": "retrieve",
                "table": "table",
                "calculate": "calculate",
            },
        )
        builder.add_edge("retrieve", "generate")
        builder.add_edge("meta", END)
        builder.add_edge("generate", END)
        builder.add_edge("table", END)
        builder.add_edge("calculate", END)
        return builder.compile()

    def run(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        thread_id: str = "default",
        user_id: str = "local",
    ) -> dict[str, Any]:
        memory_result = self._handle_memory_command(query, user_id)
        if memory_result is not None:
            self.memory.append_turn(user_id, thread_id, query, memory_result["output"])
            return memory_result
        state = self._build_state(query, document_ids, top_k, thread_id, user_id)
        result = self.graph.invoke(state)
        self._persist_turn(user_id, thread_id, query, result)
        return result

    def run_stream(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
        thread_id: str = "default",
        user_id: str = "local",
    ):
        """Yield Server-Sent-Event frames for one query.

        Verification still runs to completion before any token is released, so a
        grounded answer streams verbatim while a rejected one streams the refusal
        text instead. Streaming never bypasses the citation, numeric or
        cross-document citation checks.
        """

        yield _sse({"type": "start"})
        result = self.run(
            query=query,
            document_ids=document_ids,
            top_k=top_k,
            thread_id=thread_id,
            user_id=user_id,
        )
        output = str(result.get("output", ""))
        citations = result.get("citations", []) or []

        if citations:
            yield _sse({"type": "citations", "citations": citations})
        for chunk in _stream_chunks(output):
            yield _sse({"type": "delta", "text": chunk})
        yield _sse(
            {
                "type": "end",
                "grounded": bool(result.get("grounded", False)),
                "task": result.get("task", ""),
                "citations": citations,
            }
        )

    def _build_state(
        self,
        query: str,
        document_ids: list[str] | None,
        top_k: int | None,
        thread_id: str,
        user_id: str,
    ) -> AgentState:
        summary, history = self.memory.recent_history(
            user_id,
            thread_id,
            self.settings.short_term_turns,
        )
        return {
            "query": query,
            "document_ids": document_ids or [],
            "top_k": top_k or self.settings.top_k,
            "history": history,
            "conversation_summary": summary,
            "long_term_memories": self.memory.list_long_term(user_id),
        }

    def _persist_turn(
        self,
        user_id: str,
        thread_id: str,
        query: str,
        result: dict[str, Any],
    ) -> None:
        self.memory.append_turn(
            user_id,
            thread_id,
            query,
            result.get("output", ""),
            result.get("citations", []),
        )
        try:
            self.memory.compact_history(
                user_id,
                thread_id,
                self.settings.short_term_turns,
                self._summarize_history if self.model else None,
            )
        except Exception:
            pass

    def _route(self, state: AgentState) -> dict[str, str]:
        task = heuristic_task(state["query"])
        if task == "meta":
            return {"task": task}
        if self.model:
            prompt = f"""你是任务路由器。根据用户请求选择唯一任务类型，只输出 JSON：
{{"task":"meta|qa|compare|table|summary|calculate"}}

任务含义：
- meta：回答 HydroRAG 的身份、能力或使用方式，不检索资料
- qa：根据资料证据回答问题
- compare：比较两份或多份资料
- table：提取或整理资料表格
- summary：总结资料、章节或方法
- calculate：计算明确的数学表达式

用户请求：{state['query']}"""
            try:
                candidate = _parse_task(message_text(self.model.invoke(prompt)))
                if candidate:
                    task = candidate
            except Exception:
                pass
        return {"task": task}

    def _meta(self, state: AgentState) -> dict[str, Any]:
        return {
            "output": (
                "我是 HydroRAG，一个面向企业工程资料的检索增强问答智能体。"
                "我可以检索资料内容、总结关键信息、比较多份文档、提取表格，并进行基础计算。"
            ),
            "citations": [],
            # Identity answers carry no retrieved evidence, so they are grounded
            # by construction: nothing outside the system prompt is claimed.
            "grounded": True,
        }

    def _retrieve(self, state: AgentState) -> dict[str, Any]:
        top_k = state.get("top_k", self.settings.top_k)
        contextual_query = self._contextualize_query(state)
        retrieval_query = _expand_query(contextual_query, state.get("task", "qa"))
        raw = self.search_tool.invoke(
            {
                "query": retrieval_query,
                "top_k": max(top_k * 3, 12),
                "document_ids": ",".join(state.get("document_ids", [])),
            }
        )
        hits = json.loads(raw)
        hits = [
            hit
            for hit in hits
            if hit.get("distance") is None
            or float(hit["distance"]) <= self.settings.retrieval_distance_threshold
        ]
        if not _asks_for_references(state["query"]):
            filtered = [
                hit for hit in hits if str(hit.get("section", "")).upper() != "REFERENCES"
            ]
            if filtered:
                hits = filtered
        hits = hits[:top_k]
        if _asks_for_summary(state["query"]) and hits:
            document_ids_for_summary = list(
                dict.fromkeys(hit.get("document_id", "") for hit in hits if hit.get("document_id"))
            )[:3]
            summaries = self.store.summary_chunks(document_ids_for_summary)
            summary_by_document = {
                summary.document_id: summary.document() for summary in summaries
            }
            enriched: list[dict[str, Any]] = []
            seen: set[str] = set()
            for hit in hits:
                summary = summary_by_document.get(str(hit.get("document_id", "")))
                for item in (summary, hit):
                    if item and str(item.get("chunk_id", "")) not in seen:
                        enriched.append(item)
                        seen.add(str(item.get("chunk_id", "")))
            hits = enriched[:top_k]
        return {
            "retrieval_query": contextual_query,
            "hits": hits,
            "evidence_conflict": _has_evidence_conflict(hits),
            "citations": [
                {
                    "evidence_id": f"E{index}",
                    "title": hit.get("title", ""),
                    "page": hit.get("page", 0),
                    "document_id": hit.get("document_id", ""),
                    "location": hit.get("location", ""),
                    "source_type": hit.get("source_type", "pdf"),
                    "evidence_text": str(hit.get("text", ""))[:1800],
                }
                for index, hit in enumerate(hits, start=1)
            ],
        }

    def _generate(self, state: AgentState) -> dict[str, Any]:
        hits = state.get("hits", [])
        if not hits:
            return {
                "output": "现有资料中没有检索到达到相关性阈值的证据，无法可靠回答。请调整问题、检索范围或补充资料。",
                "citations": [],
                "grounded": False,
            }
        context = _format_context(hits)
        if not self.model:
            return {
                "output": (
                    "当前未配置 DeepSeek API，处于离线检索模式。以下是召回的证据片段：\n\n"
                    + context
                ),
                "grounded": True,
            }
        task = state.get("task", "qa")
        instruction = {
            "qa": "回答用户问题，只使用证据；关键结论后标注 [E编号]。",
            "summary": "按研究问题、方法、数据/实验和结论总结，避免补写证据中没有的内容。",
            "compare": "以 Markdown 表格比较不同资料的主题、方法、数据/指标、结果和局限；每份参与比较的资料至少引用一条直接支撑结论的对应证据。",
        }.get(task, "根据证据回答问题，并标注引用编号。")
        prompt = f"""你是严谨的公司内部资料问答智能体。
{instruction}
只允许使用下方证据回答。每一项可核验事实后必须标注对应的 [E编号]。
不得创造证据中不存在的数值、日期、文件名或工程结论。
回答中的数值必须保留证据原文单位；年份、百分比、米、立方米每秒等单位不得省略。
如果证据不足或相互冲突，请明确说明并拒绝推断。
{('证据中存在文件名范围与实际记录范围不一致，请同时说明两者。' if state.get('evidence_conflict') else '')}

对话摘要：
{state.get('conversation_summary') or '无'}

最近对话：
{_format_history(state.get('history', []))}

用户已确认的长期偏好：
{_format_memories(state.get('long_term_memories', []))}

用户问题：{state['query']}

证据：
{context}
"""
        output = message_text(self.model.invoke(prompt))
        valid_ids = _valid_evidence_ids(output, len(hits))
        if not valid_ids:
            repair_prompt = f"""请修复下面的回答。只能使用提供的证据，所有事实必须标注有效引用 [E1] 至 [E{len(hits)}]；如果无法得到有依据的回答，只输出“证据不足，无法可靠回答”。

用户问题：{state['query']}

原回答：{output}

证据：
{context}
"""
            output = message_text(self.model.invoke(repair_prompt))
            valid_ids = _valid_evidence_ids(output, len(hits))
        uncited_documents = (
            _uncited_document_ids(hits, valid_ids) if task == "compare" else set()
        )
        if valid_ids and uncited_documents:
            source_titles = sorted(
                {
                    str(hit.get("title", ""))
                    for hit in hits
                    if str(hit.get("document_id", "")) in uncited_documents
                }
            )
            repair_prompt = f"""请修复下面的跨文档比较回答。必须为每份资料至少引用一条直接支持结论的证据；缺少证据时应明确说明，不得猜测。
需要补充引用的资料：{"、".join(source_titles)}
只能使用证据 [E1] 至 [E{len(hits)}]。

用户问题：{state['query']}
原回答：{output}
证据：
{context}
"""
            output = message_text(self.model.invoke(repair_prompt))
            valid_ids = _valid_evidence_ids(output, len(hits))
            uncited_documents = _uncited_document_ids(hits, valid_ids)
        if not valid_ids:
            return {
                "output": "回答未通过引用校验，系统已阻止输出。现有证据不足以形成可验证回答。",
                "citations": [],
                "grounded": False,
            }
        numeric_audit = audit_numeric_claims(output, context)
        if not numeric_audit["supported"]:
            repair_prompt = f"""请修复下面的回答，只保留证据中明确出现的数值和单位。
不得新增、推断或改写任何数值；无法核实的数值请删除，并保留有效引用 [E1] 至 [E{len(hits)}]。

用户问题：{state['query']}
原回答：{output}
未通过校验的数值：{numeric_audit['unsupported']}
证据：
{context}
"""
            output = message_text(self.model.invoke(repair_prompt))
            valid_ids = _valid_evidence_ids(output, len(hits))
            numeric_audit = audit_numeric_claims(output, context)
            uncited_documents = (
                _uncited_document_ids(hits, valid_ids) if task == "compare" else set()
            )
        if not valid_ids or not numeric_audit["supported"] or uncited_documents:
            return {
                "output": "回答未通过数值或引用校验，系统已阻止输出。现有证据不足以形成可验证回答。",
                "citations": [],
                "grounded": False,
            }
        citations = [
            citation
            for citation in state.get("citations", [])
            if citation.get("evidence_id") in valid_ids
        ]
        return {"output": output, "citations": citations, "grounded": True}

    def _table(self, state: AgentState) -> dict[str, Any]:
        document_ids = state.get("document_ids", [])
        scoped_ids = list(document_ids)
        if not scoped_ids:
            search_raw = self.search_tool.invoke(
                {"query": state["query"], "top_k": 12, "document_ids": ""}
            )
            try:
                search_hits = json.loads(search_raw)
            except json.JSONDecodeError:
                search_hits = []
            scoped_ids = list(
                dict.fromkeys(
                    str(hit.get("document_id", ""))
                    for hit in search_hits
                    if hit.get("document_id")
                )
            )[:1]
        raw = self.table_tool.invoke(
            {"document_id": scoped_ids[0] if len(scoped_ids) == 1 else ""}
        )
        try:
            tables = json.loads(raw)
        except json.JSONDecodeError:
            tables = []
        if not tables:
            return {"output": "没有发现可用的表格抽取结果。扫描版 PDF 可能需要先接入 OCR。", "tool_result": raw}
        context_tables = tables
        if _asks_for_summary(state["query"]):
            summaries = self.store.summary_chunks(scoped_ids or None)
            if summaries:
                context_tables = [summary.document() for summary in summaries]
        if not self.model:
            return {
                "output": "当前为离线模式，表格抽取结果：\n" + _format_context(context_tables),
                "hits": context_tables,
                "tool_result": raw,
                "grounded": True,
            }
        if len(context_tables) > 24:
            scoped_ids = document_ids or list(
                dict.fromkeys(str(item.get("document_id", "")) for item in tables)
            )
            relevant = self.store.query(
                state["query"], top_k=24, document_ids=scoped_ids or None
            )
            relevant_ids = {hit.chunk_id for hit in relevant}
            context_tables = [
                item for item in tables if str(item.get("chunk_id", "")) in relevant_ids
            ][:24]
            if not context_tables:
                context_tables = tables[:24]
        prompt = f"""请根据以下表格抽取结果回答用户请求，保留表头、数值和来源位置，不要臆造缺失字段。每项事实后标注 [E编号]。
用户请求：{state['query']}
表格（仅展示与问题最相关的分块）：
{_format_context(context_tables)}"""
        output = message_text(self.model.invoke(prompt))
        valid_ids = _valid_evidence_ids(output, len(context_tables))
        citations = _citations_for_items(context_tables, valid_ids)
        if not valid_ids:
            return {
                "output": "表格回答未通过引用校验，系统已阻止输出。请直接查看抽取结果或缩小资料范围。",
                "citations": [],
                "hits": context_tables,
                "tool_result": raw,
                "grounded": False,
            }
        return {
            "output": output,
            "citations": citations,
            "hits": context_tables,
            "tool_result": raw,
            "grounded": True,
        }

    def _calculate(self, state: AgentState) -> dict[str, Any]:
        expression = _find_expression(state["query"])
        if not expression:
            return {
                "output": "未识别出可计算的数字表达式。示例：计算 (12.5 - 3.2) / 2。",
                "citations": [],
                "grounded": False,
            }
        raw = self.calculate_tool.invoke({"expression": expression})
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"output": raw, "tool_result": raw, "citations": [], "grounded": False}
        if data.get("ok"):
            # Arithmetic is evaluated deterministically, so the result is exact
            # rather than model-generated and needs no citation check.
            return {
                "output": f"计算结果：{data['expression']} = {data['value']}",
                "citations": [],
                "tool_result": raw,
                "grounded": True,
            }
        return {
            "output": f"计算失败：{data.get('error', '未知错误')}",
            "citations": [],
            "tool_result": raw,
            "grounded": False,
        }

    def _contextualize_query(self, state: AgentState) -> str:
        history = state.get("history", [])
        query = state["query"]
        if not history or not _needs_conversation_context(query):
            return query
        if not self.model:
            previous_user = next(
                (item["content"] for item in reversed(history) if item.get("role") == "user"),
                "",
            )
            return f"{previous_user}；后续问题：{query}" if previous_user else query
        prompt = f"""结合最近对话，将当前问题改写为可独立检索的问题。只输出改写后的问题，不要回答。
最近对话：
{_format_history(history)}
当前问题：{query}"""
        try:
            return message_text(self.model.invoke(prompt)).strip() or query
        except Exception:
            return query

    def _handle_memory_command(self, query: str, user_id: str) -> dict[str, Any] | None:
        text = query.strip()
        if text in {"你记得什么", "查看长期记忆", "列出长期记忆"}:
            memories = self.memory.list_long_term(user_id)
            if not memories:
                return {"output": "当前没有已确认的长期记忆。", "citations": [], "grounded": True}
            lines = [f"- #{item['id']} {item['content']}" for item in memories]
            return {"output": "已确认的长期记忆：\n" + "\n".join(lines), "citations": [], "grounded": True}
        remember_match = re.match(r"^(?:请)?记住[：:\s]*(.+)$", text)
        if remember_match:
            content = remember_match.group(1).strip()
            memory_id = self.memory.add_long_term(user_id, content)
            return {
                "output": f"已记住这项偏好（#{memory_id}）：{content}",
                "citations": [],
                "grounded": True,
            }
        forget_match = re.match(r"^(?:请)?忘记[：:\s]*(.*)$", text)
        if forget_match:
            target = forget_match.group(1).strip()
            count = (
                self.memory.clear_long_term(user_id)
                if target in {"", "全部", "所有长期记忆"}
                else self.memory.forget_matching(user_id, target)
            )
            return {
                "output": f"已删除 {count} 条匹配的长期记忆。",
                "citations": [],
                "grounded": True,
            }
        return None

    def _summarize_history(self, previous_summary: str, transcript: str) -> str:
        prompt = f"""将以下较早的对话压缩成不超过500字的事实摘要，仅保留用户目标、已确认实体、文档范围、未解决问题和回答偏好。不得加入新事实。
既有摘要：{previous_summary or '无'}
待压缩对话：
{transcript}"""
        return message_text(self.model.invoke(prompt))


def _sse(payload: dict[str, Any]) -> str:
    """Format one Server-Sent-Event data frame."""

    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_chunks(text: str, size: int = 12) -> Iterable[str]:
    """Split a verified answer into small pieces so the UI can type it out."""

    for start in range(0, len(text), size):
        yield text[start : start + size]


def heuristic_task(query: str) -> str:
    normalized = query.strip().lower().strip("？?！!。,.，")
    if normalized in {
        "你是谁",
        "你是什么",
        "你叫什么",
        "你能做什么",
        "你可以做什么",
        "你的功能是什么",
        "你是什么模型",
        "你用的什么模型",
        "介绍一下你",
        "介绍一下自己",
        "如何使用你",
        "怎么使用你",
        "你好",
        "您好",
        "嗨",
        "hello",
        "hi",
    }:
        return "meta"
    lower = query.lower()
    if any(word in query for word in ("比较", "对比", "异同", "差异")) or "compare" in lower:
        return "compare"
    table_request = any(
        word in query
        for word in ("提取", "导出", "完整表格", "表头", "行列", "原始表", "table")
    )
    if table_request and any(word in query for word in ("表格", "表", "table", "指标列")):
        return "table"
    if any(word in query for word in ("总结", "概括", "摘要", "梳理")) or "summary" in lower:
        return "summary"
    if "计算" in query or re.search(r"\d[\d\s+\-*/().%^]+\d", query):
        return "calculate"
    return "qa"


def _parse_task(text: str) -> str | None:
    try:
        data = json.loads(text)
        task = data.get("task")
        return task if task in TASKS else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*?\}", text, flags=re.S)
        if not match:
            return None
        try:
            task = json.loads(match.group(0)).get("task")
            return task if task in TASKS else None
        except json.JSONDecodeError:
            return None


def _format_context(items: list[dict[str, Any]]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        title = item.get("title", "未命名资料")
        page = item.get("page", 0)
        location = item.get("location") or (f"p.{page}" if page else "位置未知")
        section = item.get("section", "正文")
        text = str(item.get("text", "")).strip()
        lines.append(f"[E{index}] {title} | {location} | {section}\n{text[:1800]}")
    return "\n\n".join(lines)


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "无"
    labels = {"user": "用户", "assistant": "助手"}
    return "\n".join(
        f"{labels.get(item.get('role', ''), '未知')}：{str(item.get('content', ''))[:1200]}"
        for item in history
    )


def _format_memories(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "无"
    return "\n".join(f"- {item.get('content', '')}" for item in memories)


def _valid_evidence_ids(output: str, item_count: int) -> set[str]:
    valid = {f"E{index}" for index in range(1, item_count + 1)}
    cited = {f"E{value}" for value in re.findall(r"\[E(\d+)\]", output)}
    return cited if cited and cited <= valid else set()


def _uncited_document_ids(
    hits: list[dict[str, Any]], evidence_ids: set[str]
) -> set[str]:
    cited_documents = {
        str(hit.get("document_id", ""))
        for index, hit in enumerate(hits, start=1)
        if f"E{index}" in evidence_ids and hit.get("document_id")
    }
    return {
        str(hit["document_id"])
        for hit in hits
        if hit.get("document_id") and str(hit["document_id"]) not in cited_documents
    }


def _citations_for_items(
    items: list[dict[str, Any]],
    evidence_ids: set[str],
) -> list[dict[str, Any]]:
    citations = []
    for index, item in enumerate(items, start=1):
        evidence_id = f"E{index}"
        if evidence_id not in evidence_ids:
            continue
        citations.append(
            {
                "evidence_id": evidence_id,
                "title": item.get("title", ""),
                "page": item.get("page", 0),
                "document_id": item.get("document_id", ""),
                "location": item.get("location", ""),
                "source_type": item.get("source_type", ""),
                "evidence_text": str(item.get("text", ""))[:1800],
            }
        )
    return citations


def _needs_conversation_context(query: str) -> bool:
    if len(query.strip()) <= 12:
        return True
    return any(
        marker in query
        for marker in ("它", "其", "这个", "该文档", "上一份", "刚才", "前面", "上述", "继续")
    )


def _find_expression(query: str) -> str | None:
    tail = query.split("计算", 1)[-1] if "计算" in query else query
    match = re.search(r"([0-9(][0-9\s+\-*/().%^]*)", tail)
    return match.group(1).strip() if match else None


def _expand_query(query: str, task: str) -> str:
    task_terms = {
        "qa": "",
        "summary": "abstract contribution method experiment conclusion result",
        "compare": "research problem method dataset metric experiment result limitation",
    }.get(task, "")
    return f"{query} {task_terms}".strip()


def _asks_for_references(query: str) -> bool:
    lower = query.lower()
    return any(word in query for word in ("参考文献", "引用文献")) or "reference" in lower


def _asks_for_summary(query: str) -> bool:
    return any(
        term in query
        for term in ("多少", "数量", "总数", "覆盖", "范围", "记录数", "统计", "表头", "字段", "单位")
    )


def _has_evidence_conflict(hits: list[dict[str, Any]]) -> bool:
    title_range = re.compile(r"(?<!\d)(\d{4})[-~至](\d{4})(?!\d)")
    actual_range = re.compile(r"时间范围\s*(\d{4})[-/]\d{1,2}[-/]\d{1,2}\s*至\s*(\d{4})[-/]\d{1,2}[-/]\d{1,2}")
    for hit in hits:
        title_match = title_range.search(str(hit.get("title", "")))
        actual_match = actual_range.search(str(hit.get("text", "")))
        if title_match and actual_match and title_match.group(2) != actual_match.group(2):
            return True
    return False
