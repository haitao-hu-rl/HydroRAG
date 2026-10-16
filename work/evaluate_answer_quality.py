from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydrorag.agent import HydroRAGAgent
from hydrorag.config import Settings
from hydrorag.vectorstore import LocalChromaStore


REFUSAL_MARKERS = (
    "证据不足",
    "无法可靠回答",
    "无法确认",
    "无法明确",
    "资料不足",
    "无法回答",
    "不能回答",
    "不能确定",
    "未给出",
    "没有给出",
    "没有足够",
    "未检索到",
)
UNIT_ALIASES = {
    "%": "%",
    "％": "%",
    "米": "m",
    "m": "m",
    "米水位": "m",
    "年": "年",
    "号": "号",
    "万元": "万元",
    "万立方米": "万m3",
    "万m³": "万m3",
    "m³": "m3",
    "m3": "m3",
    "m²/s": "m2/s",
    "m2/s": "m2/s",
    "m³/s": "m3/s",
    "m3/s": "m3/s",
    "万m³": "万m3",
    "万m3": "万m3",
}
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,])[-+]?\d+(?:[,.]\d+)?(?:[eE][-+]?\d+)?\s*"
    r"(m²/s|m2/s|m³/s|m3/s|万m³|万m3|%|％|米水位|米|m³|m3|万立方米|万元|号|m|年)?"
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not case.get("id") or not case.get("question"):
            raise ValueError(f"Case requires id and question at {path}:{line_number}")
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found: {path}")
    return cases


def resolve_document_ids(store: LocalChromaStore, case: dict[str, Any]) -> set[str]:
    ids = {str(item) for item in case.get("relevant_doc_ids", [])}
    names = {str(item).lower() for item in case.get("relevant_sources", [])}
    if not names:
        return ids
    for document in store.documents():
        source_name = Path(document.get("source_path", "")).name.lower()
        if source_name in names:
            ids.add(str(document.get("document_id", "")))
    return ids


def resolve_source_ids(store: LocalChromaStore, sources: list[Any]) -> set[str]:
    names = {str(item).lower() for item in sources}
    if not names:
        return set()
    return {
        str(document.get("document_id", ""))
        for document in store.documents()
        if Path(document.get("source_path", "")).name.lower() in names
    }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).lower()


def matches_pattern(text: str, pattern: str) -> bool:
    if pattern.startswith("re:"):
        return re.search(pattern[3:], text, flags=re.IGNORECASE | re.DOTALL) is not None
    return normalize_text(pattern) in normalize_text(text)


def evaluate_fact_points(output: str, points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {"scored": False, "score": None, "matched": [], "total": 0}
    matched = []
    weighted_score = 0.0
    total_weight = 0.0
    for point in points:
        weight = float(point.get("weight", 1.0))
        patterns = [str(item) for item in point.get("patterns", [])]
        ok = bool(patterns) and all(matches_pattern(output, pattern) for pattern in patterns)
        matched.append({"id": point.get("id", ""), "matched": ok, "patterns": patterns})
        total_weight += weight
        if ok:
            weighted_score += weight
    return {
        "scored": True,
        "score": weighted_score / total_weight if total_weight else 0.0,
        "matched": matched,
        "total": len(points),
    }


def extract_numbers(text: str) -> list[tuple[float, str]]:
    values = []
    for match in NUMBER_PATTERN.finditer(text):
        raw = match.group(0).replace(",", "").strip()
        unit = UNIT_ALIASES.get((match.group(1) or "").strip(), "")
        number = raw[: len(raw) - len(match.group(1) or "")].strip()
        try:
            values.append((float(number), unit))
        except ValueError:
            continue
    return values


def evaluate_numeric_facts(output: str, facts: list[dict[str, Any]]) -> dict[str, Any]:
    if not facts:
        return {
            "scored": False,
            "value_accuracy": None,
            "unit_accuracy": None,
            "value_unit_accuracy": None,
            "matched": [],
        }
    actual = extract_numbers(output)
    matched = []
    value_hits = unit_hits = exact_hits = 0
    for fact in facts:
        expected = float(fact["value"])
        expected_unit = UNIT_ALIASES.get(str(fact.get("unit", "")), str(fact.get("unit", "")))
        tolerance = float(fact.get("tolerance", max(abs(expected) * 0.001, 1e-6)))
        candidates = [
            (value, unit)
            for value, unit in actual
            if abs(value - expected) <= tolerance
        ]
        value_ok = bool(candidates)
        unit_ok = bool(candidates) and any(not expected_unit or unit == expected_unit for _, unit in candidates)
        exact_ok = value_ok and unit_ok
        value_hits += int(value_ok)
        unit_hits += int(unit_ok)
        exact_hits += int(exact_ok)
        matched.append(
            {
                "value": expected,
                "unit": expected_unit,
                "value_match": value_ok,
                "unit_match": unit_ok,
            }
        )
    total = len(facts)
    return {
        "scored": True,
        "value_accuracy": value_hits / total,
        "unit_accuracy": unit_hits / total,
        "value_unit_accuracy": exact_hits / total,
        "matched": matched,
    }


def is_refusal(output: str, grounded: bool | None) -> bool:
    if grounded is False:
        return True
    normalized = normalize_text(output)
    return any(normalize_text(marker) in normalized for marker in REFUSAL_MARKERS)


def evaluate_citations(
    result: dict[str, Any],
    relevant_ids: set[str],
    answerable: bool,
) -> dict[str, Any]:
    hits = result.get("hits", []) or []
    citations = result.get("citations", []) or []
    valid_ids = {f"E{index}" for index in range(1, len(hits) + 1)}
    citation_ids = {str(item.get("evidence_id", "")) for item in citations}
    valid_citations = citation_ids & valid_ids
    cited_documents = {
        str(item.get("document_id", ""))
        for item in citations
        if str(item.get("evidence_id", "")) in valid_citations
    }
    if not answerable:
        return {
            "scored": False,
            "validity": None,
            "precision": None,
            "recall": None,
            "cited_documents": sorted(cited_documents),
        }
    required = relevant_ids
    supported = cited_documents & required
    return {
        "scored": True,
        "validity": len(valid_citations) / len(citation_ids) if citation_ids else 0.0,
        "precision": len(supported) / len(cited_documents) if cited_documents else 0.0,
        "recall": len(supported) / len(required) if required else 0.0,
        "cited_documents": sorted(cited_documents),
    }


def evaluate_case(
    agent: HydroRAGAgent,
    store: LocalChromaStore,
    case: dict[str, Any],
) -> dict[str, Any]:
    relevant_ids = resolve_document_ids(store, case)
    citation_ids = resolve_source_ids(
        store, case.get("required_citation_documents", [])
    ) or relevant_ids
    result = agent.run(
        case["question"],
        document_ids=[str(item) for item in case.get("document_ids", [])],
        top_k=int(case.get("top_k", 6)),
        thread_id=f"eval-{case['id']}",
        user_id="evaluation",
    )
    hits = result.get("hits", []) or []
    ranked_documents = []
    for hit in hits:
        document_id = str(hit.get("document_id", ""))
        if document_id and document_id not in ranked_documents:
            ranked_documents.append(document_id)
    top1 = bool(ranked_documents and ranked_documents[0] in relevant_ids)
    output = str(result.get("output", ""))
    answerable = bool(case.get("answerable", True))
    fact_points = evaluate_fact_points(output, case.get("required_points", []))
    numeric = evaluate_numeric_facts(output, case.get("numeric_facts", []))
    refusal = is_refusal(output, result.get("grounded"))
    if answerable and (
        (fact_points.get("score") or 0.0) > 0.0
        or (numeric.get("value_accuracy") or 0.0) > 0.0
    ):
        refusal = False
    citations = evaluate_citations(result, citation_ids, answerable)
    return {
        "id": case["id"],
        "question": case["question"],
        "answerable": answerable,
        "relevant_doc_ids": sorted(relevant_ids),
        "required_citation_doc_ids": sorted(citation_ids),
        "top1_document_accuracy": int(top1) if relevant_ids else None,
        "ranked_documents": ranked_documents,
        "output": output,
        "grounded": result.get("grounded"),
        "refusal": refusal,
        "refusal_correct": int(refusal == (not answerable)),
        "fact_points": fact_points,
        "numeric": numeric,
        "citations": citations,
    }


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in results if item["answerable"]]
    unanswerable = [item for item in results if not item["answerable"]]
    fact_scores = [
        item["fact_points"]["score"]
        for item in answerable
        if item["fact_points"]["scored"]
    ]
    value_scores = [
        item["numeric"]["value_accuracy"]
        for item in answerable
        if item["numeric"]["scored"]
    ]
    unit_scores = [
        item["numeric"]["unit_accuracy"]
        for item in answerable
        if item["numeric"]["scored"]
    ]
    exact_scores = [
        item["numeric"]["value_unit_accuracy"]
        for item in answerable
        if item["numeric"]["scored"]
    ]
    citation_precision = [item["citations"]["precision"] for item in answerable]
    citation_recall = [item["citations"]["recall"] for item in answerable]
    return {
        "cases": len(results),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "top1_document_accuracy": mean(
            [item["top1_document_accuracy"] for item in answerable if item["top1_document_accuracy"] is not None]
        ),
        "final_answer_fact_accuracy": mean(fact_scores),
        "citation_precision": mean(citation_precision),
        "citation_recall": mean(citation_recall),
        "numeric_value_accuracy": mean(value_scores),
        "numeric_unit_accuracy": mean(unit_scores),
        "numeric_value_unit_accuracy": mean(exact_scores),
        "refusal_recall": mean([item["refusal"] for item in unanswerable]),
        "false_refusal_rate": mean([item["refusal"] for item in answerable]),
        "safe_answer_rate": mean([item["refusal_correct"] for item in results]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HydroRAG 答案级质量评测")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("answer_eval_cases.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("answer_eval_results.json"))
    args = parser.parse_args()

    settings = Settings.from_env()
    settings.ensure_dirs()
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
    agent = HydroRAGAgent(store, settings)
    results = [evaluate_case(agent, store, case) for case in load_cases(args.cases)]
    payload = {
        "evaluation_unit": "answer-level deterministic audit with manually annotated fact points",
        "config": {
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
            "documents": len(store.documents()),
            "chunks": store.count,
        },
        "metrics": summarize(results),
        "details": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
