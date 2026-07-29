# 阶段 3A：Reranker 与最终搜索服务

## 阶段目标

调用本地 llama.cpp Cross-Encoder 对 RRF 候选重新评分，形成最终 Top-K，同时保留 Dense、
BM25、RRF 和 Reranker 每一阶段的排名、原始分数、耗时和候选数量。

## 完成内容

### Reranker Client

新增 `src/rag_demo/reranker_client.py`，使用共享的 `httpx.AsyncClient` 调用：

```http
POST /reranking
Authorization: Bearer <configured-key>
Content-Type: application/json
```

请求体：

```json
{
  "query": "用户问题",
  "documents": [
    "候选 retrieval_text"
  ]
}
```

客户端行为：

- 一次批量提交 RRF 候选；
- 按响应中的 `index` 将 score 恢复到输入 Document 顺序；
- 校验结果数量、索引类型、索引范围、重复索引和缺失索引；
- 拒绝 NaN 和 Infinity；
- 将 HTTP/JSON 错误包装为不暴露 API Key 的领域异常；
- 空 Document 列表不发起 HTTP 请求；
- 使用独立的连接、写入、连接池和可配置读取超时。

`relevance_score` 只被称为相关性分数，不解释为概率。

### 核心 SearchRequest

在 `src/rag_demo/search_service.py` 增加不可变的 `SearchRequest`，统一承载：

- Query；
- Dense/BM25 Top-K；
- RRF rank constant；
- Rerank Top-K；
- Final Top-K；
- exact/HNSW 模式；
- 是否启用 Reranker；
- Debug 标记。

对象在进入 Service 前校验非空 Query、所有正整数限制，以及
`final_top_k <= rerank_top_k`。后续 CLI 和 REST 将构造同一个核心请求对象。

### 最终搜索服务

新增 `HybridSearchService`，固定执行：

```text
Dense + BM25 并行召回
→ RRF union
→ RRF Top rerank_top_k
→ Cross-Encoder 批量打分
→ relevance_score 降序
→ Final Top final_top_k
```

关闭 Reranker 时：

```text
RRF order
→ Final Top final_top_k
```

不重新查询 Chunk，也不覆盖之前的排名和分数。

### 完整候选证据

每个 `SearchCandidate` 保存：

- Chunk/Document ID；
- 标题、仓库、commit、相对路径和语言；
- Heading 路径；
- 原文、`retrieval_text` 和 metadata；
- Dense rank/score；
- BM25 rank/score；
- RRF rank/score；
- Reranker score；
- Final rank。

`SearchResponse` 另外保留：

- 原始 `HybridRecallResponse`；
- 全部 Reranker 排序候选；
- 最终候选；
- 是否实际启用 Reranker；
- 各阶段耗时；
- 各分支和并集候选数量。

### 可观测性

`SearchTimings`：

```text
query_embedding_ms
dense_search_ms
bm25_search_ms
rrf_ms
rerank_ms
total_ms
```

`SearchCandidateCounts`：

```text
dense_candidate_count
bm25_candidate_count
union_candidate_count
```

Dense 和 BM25 并行执行，因此总耗时不能简单理解为各阶段耗时之和。

## TDD 过程

先通过 Mock HTTP 和固定候选编写行为测试：

- Reranker 鉴权和请求格式；
- 响应乱序时恢复输入顺序；
- 空候选不发 HTTP；
- 错误数量、重复索引、非法索引和非有限分数；
- HTTP 错误不泄漏密钥；
- 只将 RRF Top-K 提交给 Reranker；
- Reranker 改变最终顺序；
- Dense/BM25/RRF rank 与 score 不被覆盖；
- `--no-rerank` 语义保持 RRF 顺序；
- 最终排名连续；
- 不一致 Top-K 参数被拒绝。

模块和接口不存在时测试按预期在收集阶段失败；实现后全部转绿。

## 真实服务契约

在不打印 API Key 的前提下探测当前 llama.cpp 服务，确认响应形状：

```json
{
  "model": "bge-reranker-v2-m3",
  "object": "list",
  "results": [
    {
      "index": 0,
      "relevance_score": -6.56
    }
  ]
}
```

分数可以是负数，因此实现只要求数值有限，不错误地要求 `[0, 1]`。

真实测试 Query：

```text
FastAPI 如何接收 JSON 请求体？
```

候选：

```text
1. 使用 Pydantic 模型声明 JSON 请求体。
2. 这是一段无关的数据库迁移说明。
```

真实 bge-reranker-v2-m3 对候选 1 的分数高于候选 2。

## 验证证据

完整自动检查：

```text
76 passed
Total coverage: 88.58%
Ruff: passed
Ruff format: passed
Mypy strict: passed
```

固定 RRF 顺序：

```text
20, 30, 10, 40
```

Reranker 对前三个候选返回：

```text
20 → 0.1
30 → 0.9
10 → 0.2
```

最终 Top-2：

```text
30, 10
```

测试同时确认 30 仍保留自己的 Dense rank 3、BM25 rank 2 和 RRF rank 2。

## 面试讲解要点

- Bi-Encoder Dense 适合高效召回，Cross-Encoder 同时观察 Query 与 Passage，适合小候选集重排。
- Reranker 只处理 RRF Top 20，不对整个语料运行，控制推理成本。
- llama.cpp Reranker 不是 OpenAI 标准接口，因此客户端保持很薄，不设计多后端框架。
- 响应顺序不能被默认信任，必须依赖 index 恢复输入映射。
- Reranker score 可能是负数，只用于相对排序。
- 每个阶段的 rank/score 都保留，因此最终顺序变化可以被完整解释。
- `--no-rerank` 不走另一套业务逻辑，只跳过同一 Service 中的重排步骤。

## 当前限制与下一阶段

当前核心 Python Service 已完成，但还没有统一的资源生命周期和用户入口。阶段 3B 将：

- 在一个应用容器中创建/关闭 Database Pool 与 HTTP Client；
- 组装 Embedding、Dense、BM25、RRF、Reranker 和摄取对象；
- 增加 `doctor`、`ingest`、`bm25 rebuild`、`search`、`compare` 等 Typer 命令；
- 用真实 CLI 验证核心 Service 可演示。
