# 施工记录索引

本目录记录 Minimal Hybrid RAG Demo 各施工阶段的目标、实现、验证证据和讲解要点。
每个阶段只有在测试、真实环境验证和对应施工记录全部完成后才视为完成。

| 阶段 | 记录 | 状态 |
|---|---|---|
| 0 | [工程基线](phase-00-engineering-baseline.md) | 已完成 |
| 1A | [数据库基础](phase-01a-database-foundation.md) | 已完成 |
| 1B | [Markdown 解析与 Chunk](phase-01b-markdown-chunking.md) | 已完成 |
| 1C | [Embedding 与摄取](phase-01c-embedding-ingestion.md) | 已完成 |
| 1D | [Dense 精确检索与 HNSW](phase-01d-dense-retrieval.md) | 已完成 |
| 2 | [BM25 与 RRF 混合召回](phase-02-bm25-rrf.md) | 已完成 |
| 3A | [Reranker 与最终搜索服务](phase-03a-reranker-search-service.md) | 已完成 |
| 3B | [应用生命周期与 Typer CLI](phase-03b-application-cli.md) | 已完成 |
| 3C | [FastAPI REST 与接口一致性](phase-03c-fastapi-rest.md) | 已完成 |
| 4A | [FastAPI 语料工具与五路评测](phase-04a-corpus-evaluation.md) | 已完成 |
| 4B | [真实语料验收、README 与最终演示](phase-04b-real-acceptance.md) | 已完成 |
| 5A | [Front Matter 与首个 H1 结构修正](phase-05a-frontmatter-heading-preservation.md) | 已完成 |
| 5B | [摄取幂等键自动生成](phase-05b-automatic-idempotency-key.md) | 已完成 |
| 6 | [确定性 Query Rewrite 与效果对比](phase-06-deterministic-query-rewrite.md) | 已完成 |
| 7 | [同步与异步 Python SDK](phase-07-python-sdk.md) | 已完成 |
| 8 | [低可信检索结果提醒](phase-08-low-confidence-warning.md) | 已完成 |
