# HydroRAG
面向南水北调工程资料的水利文档检索与问答智能体。支持 PDF/图片/Word/Excel/CSV 多格式解析并保留页码与表格来源；基于 BGE + BM25 混合检索与 Cross-Encoder 重排，LangGraph 编排意图路由与工具调用，采用引用校验与证据不足拒答治理幻觉。开发集 Recall@5 94.01%，FastAPI 服务化。
