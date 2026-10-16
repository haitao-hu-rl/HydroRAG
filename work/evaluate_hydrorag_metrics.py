from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydrorag.agent import HydroRAGAgent
from hydrorag.config import Settings
from hydrorag.tools import AgentTools
from hydrorag.vectorstore import LocalChromaStore


SOURCE_DIRS = [
    Path(r"C:\Users\HaiTao_Hu\Desktop\南水北调中线\南水北调中线"),
    Path(r"C:\Users\HaiTao_Hu\Desktop\中线数据\中线数据"),
]


def ndcg(ranked: list[str], relevant: set[str], k: int) -> float:
    values = [int(item in relevant) for item in ranked[:k]]
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(values))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0


def main() -> None:
    settings = Settings.from_env()
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
    documents = {
        Path(item["source_path"]).name: item
        for item in store.documents()
        if any(str(item["source_path"]).startswith(str(directory)) for directory in SOURCE_DIRS)
    }

    file_keys = {
        "plan2023": "20231023水利部关于印发南水北调中线一期工程2023-2024年度水量调度计划的通知（水南调函2023-107号）.pdf",
        "plan2014": "中线2014-2015年度水量调度计划的批文.pdf",
        "water_scheme": "南水北调中线一期工程输水调度方案研究.pdf",
        "sim_model": "南水北调中线工程水量仿真调度模型研究.pdf",
        "opt_model": "南水北调中线工程水量优化调度研究.pdf",
        "sim_thesis": "南水北调中线工程水量调度仿真研究.pdf",
        "gate_discuss": "南水北调中线干渠闸门调度运行方式探讨.pdf",
        "outlet_group": "南水北调中线总干渠分水口群运用方式研究.pdf",
        "dispatch_system": "南水北调中线总干渠水量调度模型研究及系统开发.pdf",
        "gate_scan": "南水北调中线总干渠节制闸控制运行方式分析.pdf",
        "gate_thesis": "南水北调中线总干渠节制闸控制运行方式研究.pdf",
        "channel_model": "南水北调中线渠道控制计算模型.pdf",
        "control_improve": "南水北调中线输水调度控制模型改进研究.pdf",
        "hanjiang": "汉江水量分配（汇总170416）.pdf",
        "outlet_grade": "输水渠道大流量分水口门的分级分水方式.pdf",
        "huanghuaihai": "黄淮海水资源配置.pdf",
        "supplement": "（水科院）南水北调中线工程补充研究.pdf",
        "reference": "参考.pdf",
        "daily_outflow": "1969-2014日均出库.xlsx",
        "ankang_runoff": "安康站日平均径流1950-2023.xlsx",
        "danjiangkou_flood": "丹江口水库防洪限制库容-水位.xlsx",
        "groundwater": "地下水.xlsx",
        "diversion_segment": "diversion_and_segment.xlsx",
        "gate_flow": "节制闸水位、流量表.xlsx",
        "receiving_area": "受水区.xlsx",
        "receiving_reservoir": "受水区水库防洪控制库容（总干渠以西）.xlsx",
        "rainfall": "雨量.xlsx",
        "evaporation": "蒸散发.xlsx",
        "channel_data": "channel.xlsx",
        "city_area": "city_and_area.xlsx",
        "precipitation_surface": "降水和地表水资源.xlsx",
        "station_info": "station_info_698.xlsx",
    }
    gold = {key: documents[name]["document_id"] for key, name in file_keys.items()}
    queries = [
        ("2023-2024年度南水北调中线一期工程水量调度计划的主要要求", ["plan2023"]),
        ("2014-2015年度水量调度计划的实施依据和运行安全要求", ["plan2014"]),
        ("南水北调中线一期工程输水调度方案中的充水过程和目标控制水位", ["water_scheme"]),
        ("南水北调中线工程水量仿真调度模型的系统网络结构和计算原则", ["sim_model"]),
        ("南水北调中线工程水量优化调度的多目标和联合调度方法", ["opt_model"]),
        ("南水北调中线工程水量调度仿真研究的仿真模型和调度结果", ["sim_thesis"]),
        ("南水北调中线干渠闸门同步控制与时序控制的比较", ["gate_discuss"]),
        ("南水北调中线总干渠分水口群顺序同步逆序运用方式的水力控制指标", ["outlet_group"]),
        ("南水北调中线总干渠水量调度模型研究及系统开发的模型架构", ["dispatch_system"]),
        ("南水北调中线总干渠节制闸控制运行方式分析", ["gate_scan"]),
        ("南水北调中线总干渠节制闸控制运行方式研究的控制策略", ["gate_thesis"]),
        ("南水北调中线渠道控制计算模型的闸前常水位方式", ["channel_model"]),
        ("南水北调中线输水调度控制模型的前馈反馈和三点控制改进", ["control_improve"]),
        ("汉江水量分配涉及的湖北省地级行政区和分配原则", ["hanjiang"]),
        ("输水渠道大流量分水口门的分级分水方式和水力响应", ["outlet_grade"]),
        ("黄淮海流域水资源配置原则和南水北调工程风险分析", ["huanghuaihai"]),
        ("南水北调中线工程对汉江中下游水环境影响与对策", ["supplement"]),
        ("国科办厅字1998年82号项目的水资源配置内容", ["reference"]),
        ("1969到2014年丹江口水库日均出库数据", ["daily_outflow"]),
        ("安康站1950到2023年日平均径流资料", ["ankang_runoff"]),
        ("丹江口水库防洪限制库容与水位关系", ["danjiangkou_flood"]),
        ("南水北调中线地下水数据", ["groundwater"]),
        ("南水北调中线分水口和渠段数据", ["diversion_segment"]),
        ("节制闸水位和流量表", ["gate_flow"]),
        ("南水北调中线受水区及受水区水库防洪控制库容", ["receiving_area", "receiving_reservoir"]),
        ("南水北调中线雨量与蒸散发数据", ["rainfall", "evaporation"]),
        ("降水和地表水资源数据", ["precipitation_surface"]),
        ("698个测站信息", ["station_info"]),
        ("南水北调中线城市和区域面积数据", ["city_area"]),
        ("南水北调中线渠道基础数据", ["channel_data"]),
        ("南水北调中线工程节制闸控制运行方式和渠道水力控制", ["gate_scan", "gate_thesis", "gate_discuss", "channel_model"]),
        ("南水北调中线工程水量调度模型、仿真与优化调度研究", ["sim_model", "opt_model", "sim_thesis", "dispatch_system"]),
        ("2023-2024年度水量调度计划的调度原则和供水安排", ["plan2023"]),
        ("中线2014-2015年度调度计划中的水量配置与安全要求", ["plan2014"]),
        ("输水调度方案的充水期控制目标和水位控制方法", ["water_scheme"]),
        ("水量仿真调度模型的网络结构、边界条件和计算流程", ["sim_model"]),
        ("水量优化调度研究中的多目标优化和联合调度策略", ["opt_model"]),
        ("水量调度仿真研究采用的模型、场景和验证结果", ["sim_thesis"]),
        ("干渠闸门同步控制与时序控制各自适用的运行场景", ["gate_discuss"]),
        ("分水口群不同运用方式对应的水力控制指标", ["outlet_group"]),
        ("总干渠水量调度系统的模型组成和软件实现", ["dispatch_system"]),
        ("总干渠节制闸控制运行方式分析中的控制模式", ["gate_scan"]),
        ("节制闸控制运行方式研究中的控制策略和适用条件", ["gate_thesis"]),
        ("渠道控制计算模型的闸前常水位控制方式", ["channel_model"]),
        ("输水调度控制模型如何结合前馈、反馈和三点控制", ["control_improve"]),
        ("汉江水量分配的行政区范围和分配原则", ["hanjiang"]),
        ("大流量分水口门的分级分水运行方式", ["outlet_grade"]),
        ("黄淮海水资源配置中的工程风险与配置原则", ["huanghuaihai"]),
        ("南水北调中线工程对汉江中下游水环境影响的主要对策", ["supplement"]),
        ("国科办厅字1998年82号项目的水资源配置研究内容", ["reference"]),
        ("日均出库数据覆盖的年份和数据字段", ["daily_outflow"]),
        ("安康站日平均径流资料的时间范围", ["ankang_runoff"]),
        ("丹江口水库防洪限制库容与对应水位的关系", ["danjiangkou_flood"]),
        ("地下水数据表包含哪些南水北调中线数据", ["groundwater"]),
        ("分水口和渠段数据表的主要字段", ["diversion_segment"]),
        ("节制闸水位流量表中的监测字段", ["gate_flow"]),
        ("受水区水库防洪控制库容表的数据范围", ["receiving_reservoir"]),
        ("雨量数据和蒸散发数据分别记录什么内容", ["rainfall", "evaporation"]),
        ("降水和地表水资源数据的主要指标", ["precipitation_surface"]),
        ("698个测站信息表中的测站字段", ["station_info"]),
        ("城市和区域面积数据的统计对象", ["city_area"]),
        ("渠道基础数据中的渠道和断面信息", ["channel_data"]),
        ("哪些资料讨论了节制闸控制运行方式", ["gate_scan", "gate_thesis", "gate_discuss"]),
        ("哪些资料研究了水量调度模型及其仿真验证", ["sim_model", "sim_thesis", "dispatch_system"]),
    ]

    agent = HydroRAGAgent.__new__(HydroRAGAgent)
    agent.settings = settings
    agent.search_tool = AgentTools(store).as_tools()[0]
    allowed_ids = [item["document_id"] for item in documents.values()]
    metric_values = {k: {name: [] for name in ("precision", "recall", "ndcg", "hit")} for k in (1, 3, 5, 10)}
    reciprocal_ranks: list[float] = []
    details = []

    for query, keys in queries:
        relevant = {gold[key] for key in keys}
        result = agent._retrieve(
            {"query": query, "task": "qa", "top_k": 30, "document_ids": allowed_ids}
        )
        ranked: list[str] = []
        titles: dict[str, str] = {}
        for hit in result.get("hits", []):
            document_id = hit.get("document_id", "")
            if document_id not in ranked:
                ranked.append(document_id)
                titles[document_id] = hit.get("title", "")
        first = next((index + 1 for index, item in enumerate(ranked) if item in relevant), None)
        reciprocal_ranks.append(1 / first if first else 0.0)
        for k in metric_values:
            top = ranked[:k]
            relevant_count = sum(item in relevant for item in top)
            metric_values[k]["precision"].append(relevant_count / k)
            metric_values[k]["recall"].append(relevant_count / len(relevant))
            metric_values[k]["ndcg"].append(ndcg(ranked, relevant, k))
            metric_values[k]["hit"].append(int(relevant_count > 0))
        details.append(
            {
                "query": query,
                "gold": keys,
                "first_relevant_rank": first,
                "top10": [titles.get(item, item) for item in ranked[:10]],
            }
        )

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4)

    output = {
        "source_files_indexed": len(documents),
        "project_documents_total": len(store.documents()),
        "project_chunks_total": store.count,
        "queries": len(queries),
        "evaluation_unit": "unique document ranking; binary document relevance",
        "metrics": {
            str(k): {name: mean(values) for name, values in metrics.items()}
            for k, metrics in metric_values.items()
        },
        "mrr": mean(reciprocal_ranks),
        "details": details,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
