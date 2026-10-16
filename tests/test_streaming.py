"""Regression tests for the SSE streaming endpoint and answer grounding flags.

Streaming must never bypass the hallucination guards: the answer is generated and
verified in full before the first delta is released, so a rejected answer streams
its refusal notice instead of unverified text.
"""

import json

import pytest
from fastapi.testclient import TestClient

from hydrorag.agent import HydroRAGAgent, _sse, _stream_chunks


def test_sse_frame_is_valid_wire_format():
    frame = _sse({"type": "delta", "text": "节制闸 350 m³/s"})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    # ensure_ascii=False keeps CJK readable for SSE clients
    assert "\\u" not in frame
    assert json.loads(frame[6:].strip())["text"] == "节制闸 350 m³/s"


@pytest.mark.parametrize("size", [1, 3, 12, 64, 1000])
@pytest.mark.parametrize(
    "text",
    ["", "短", "节制闸设计流量 350 m³/s [E1]。", "a" * 500, "混合 mixed 内容" * 40],
)
def test_stream_chunks_are_lossless(size, text):
    pieces = list(_stream_chunks(text, size))
    assert "".join(pieces) == text
    assert all(len(piece) <= size for piece in pieces)


class _StubAgent(HydroRAGAgent):
    """Bypass retrieval/LLM and return a fixed verifier result."""

    def __init__(self, result):
        self._fixed = result

    def run(self, query, **kwargs):
        return dict(self._fixed)


def _frames(result):
    return [
        json.loads(frame[6:].strip())
        for frame in _StubAgent(result).run_stream("任意问题")
    ]


def test_run_stream_orders_frames_and_reassembles_answer():
    answer = "节制闸设计流量 350 m³/s [E1]。"
    frames = _frames(
        {
            "output": answer,
            "citations": [{"evidence_id": "E1", "title": "渠道控制计算模型"}],
            "grounded": True,
            "task": "qa",
        }
    )
    kinds = [frame["type"] for frame in frames]
    assert kinds[0] == "start"
    assert kinds[-1] == "end"
    # citations are advertised before any delta so the UI can render sources early
    assert kinds.index("citations") < kinds.index("delta")
    assert "".join(f["text"] for f in frames if f["type"] == "delta") == answer
    assert frames[-1]["grounded"] is True


def test_run_stream_preserves_blocked_answer_and_grounded_false():
    blocked = "回答未通过引用校验，系统已阻止输出。现有证据不足以形成可验证回答。"
    frames = _frames(
        {"output": blocked, "citations": [], "grounded": False, "task": "qa"}
    )
    assert "".join(f["text"] for f in frames if f["type"] == "delta") == blocked
    assert frames[-1]["grounded"] is False
    assert not [f for f in frames if f["type"] == "citations"]


def test_run_stream_handles_empty_output():
    frames = _frames({"output": "", "citations": [], "grounded": True, "task": "meta"})
    assert frames[0]["type"] == "start"
    assert frames[-1]["type"] == "end"
    assert not [f for f in frames if f["type"] == "delta"]


def test_stream_endpoint_is_registered_once_and_streams():
    from hydrorag import server

    stream_routes = [
        route for route in server.app.routes if getattr(route, "path", "") == "/api/chat/stream"
    ]
    assert len(stream_routes) == 1, "LangServe must not shadow the custom SSE endpoint"

    client = TestClient(server.app)
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"query": "你是谁", "user_id": "pytest", "thread_id": "pytest-stream"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    frames = [line for line in body.split("\n") if line.startswith("data: ")]
    assert frames, "expected at least one SSE frame"
    kinds = [json.loads(frame[6:])["type"] for frame in frames]
    assert kinds[0] == "start" and kinds[-1] == "end"
    assert kinds.count("delta") > 1, "meta answer should stream in several pieces"


@pytest.mark.parametrize(
    ("query", "task", "grounded"),
    [
        ("你是谁", "meta", True),
        ("你好", "meta", True),
        ("计算 (12.5 - 3.2) / 2", "calculate", True),
        ("计算 未识别的表达式", "calculate", False),
    ],
)
def test_grounded_flag_is_set_for_every_task(query, task, grounded):
    """meta/calculate previously left `grounded` unset, which read as False."""

    from hydrorag import server

    client = TestClient(server.app)
    response = client.post(
        "/api/chat/invoke",
        json={"input": {"query": query, "user_id": "pytest", "thread_id": "pytest-grounded"}},
    )
    assert response.status_code == 200
    payload = response.json()
    result = payload.get("output", payload)
    assert result.get("task") == task
    assert result.get("grounded") is grounded
