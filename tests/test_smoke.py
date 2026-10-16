from hydrorag.agent import heuristic_task
from hydrorag.pdfs import chunk_text
from hydrorag.tools import safe_calculate


def test_task_router():
    assert heuristic_task("你是谁") == "meta"
    assert heuristic_task("比较两篇论文的方法") == "compare"
    assert heuristic_task("提取实验表格") == "table"
    assert heuristic_task("测站信息表包含多少个测站？") == "qa"
    assert heuristic_task("节制闸水位和流量表主要记录哪些信息？") == "qa"
    assert heuristic_task("总结这篇论文") == "summary"
    assert heuristic_task("计算 (12 - 2) / 2") == "calculate"


def test_chunk_text_preserves_content():
    pieces = chunk_text("第一段\n\n第二段\n\n第三段", max_chars=20, overlap=4)
    assert pieces
    assert "第一段" in "\n".join(pieces)
    assert "第三段" in "\n".join(pieces)


def test_safe_calculate():
    assert safe_calculate("(12 - 2) / 2") == 5
