from hydrorag.agent import (
    HydroRAGAgent,
    _has_evidence_conflict,
    _needs_conversation_context,
    _uncited_document_ids,
    _valid_evidence_ids,
)
from hydrorag.memory import SQLiteMemoryStore


def test_short_and_long_term_memory_are_scoped(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    store.append_turn("user-a", "thread-1", "G61 是什么？", "它是节制闸。")

    summary, history = store.recent_history("user-a", "thread-1", turns=4)
    assert summary == ""
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert store.recent_history("user-b", "thread-1", turns=4)[1] == []

    memory_id = store.add_long_term("user-a", "回答时附带来源位置")
    assert store.list_long_term("user-a")[0]["content"] == "回答时附带来源位置"
    assert store.list_long_term("user-b") == []
    assert store.delete_long_term("user-a", memory_id)


def test_history_compaction_and_citation_validation(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    for index in range(4):
        store.append_turn("user", "thread", f"问题{index}", f"回答{index}")
    store.compact_history("user", "thread", keep_turns=2)
    summary, history = store.recent_history("user", "thread", turns=2)
    assert "问题0" in summary
    assert len(history) == 4

    assert not _valid_evidence_ids("结论 [E1]，另见不存在的 [E3]。", 2)
    assert _valid_evidence_ids("结论 [E1]，另见 [E2]。", 2) == {"E1", "E2"}
    assert not _valid_evidence_ids("没有引用", 2)
    assert _needs_conversation_context("它有什么限制？")
    assert _has_evidence_conflict(
        [{"title": "数据 1969-2014", "text": "时间范围 1969-01-01 至 2013-12-31"}]
    )
    assert not _has_evidence_conflict(
        [{"title": "数据 1950-2023", "text": "时间范围 1950-01-01 至 2023-12-31"}]
    )


def test_generation_keeps_only_cited_evidence():
    class FakeMessage:
        content = "G61 的控制约束见证据。[E1]"

    class FakeModel:
        def invoke(self, prompt):
            return FakeMessage()

    agent = HydroRAGAgent.__new__(HydroRAGAgent)
    agent.model = FakeModel()
    state = {
        "query": "G61 的控制约束是什么？",
        "hits": [
            {
                "title": "调度报告",
                "page": 3,
                "location": "p.3",
                "section": "运行约束",
                "text": "G61 的控制约束为……",
            },
            {
                "title": "无关报告",
                "page": 8,
                "location": "p.8",
                "section": "正文",
                "text": "其他内容",
            },
        ],
        "citations": [
            {"evidence_id": "E1", "title": "调度报告", "page": 3},
            {"evidence_id": "E2", "title": "无关报告", "page": 8},
        ],
    }
    result = agent._generate(state)
    assert result["grounded"] is True
    assert [item["evidence_id"] for item in result["citations"]] == ["E1"]


def test_compare_prompt_requires_citations_for_each_source():
    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeModel:
        prompt = ""
        calls = 0
        prompts = []

        def invoke(self, prompt):
            self.prompt = prompt
            self.prompts.append(prompt)
            self.calls += 1
            answer = "资料一指出水量仿真。[E1]" if self.calls == 1 else "资料一指出水量仿真，资料二研究优化调度。[E1][E2]"
            return FakeMessage(answer)

    agent = HydroRAGAgent.__new__(HydroRAGAgent)
    agent.model = FakeModel()
    state = {
        "query": "比较两份研究",
        "task": "compare",
        "hits": [
            {"title": "仿真调度研究", "text": "水量仿真调度模型", "document_id": "a"},
            {"title": "优化调度研究", "text": "水量优化调度方法", "document_id": "b"},
        ],
        "citations": [
            {"evidence_id": "E1", "title": "仿真调度研究", "document_id": "a"},
            {"evidence_id": "E2", "title": "优化调度研究", "document_id": "b"},
        ],
    }

    result = agent._generate(state)

    assert agent.model.calls == 2
    assert "每份参与比较的资料至少引用一条" in agent.model.prompts[0]
    assert "需要补充引用的资料：优化调度研究" in agent.model.prompt
    assert [item["evidence_id"] for item in result["citations"]] == ["E1", "E2"]
    assert _uncited_document_ids(state["hits"], {"E1"}) == {"b"}
    assert _uncited_document_ids(state["hits"], {"E1", "E2"}) == set()
