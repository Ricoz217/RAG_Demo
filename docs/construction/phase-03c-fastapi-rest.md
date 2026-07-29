# 阶段 3C：FastAPI REST 与接口一致性

## 阶段目标

使用与 CLI 相同的 `RAGApplication` 和核心 Service 提供本地 FastAPI REST，完成健康检查、
统计、搜索、摄取和 BM25 重建，并验证 CLI 与 REST 对相同 Query 返回完全相同的最终
Chunk 顺序。

## 完成内容

### FastAPI Lifespan

新增 `src/rag_demo/api.py` 和 `create_app()`：

- 启动时创建一个 `RAGApplication`；
- 打开 PostgreSQL Pool 与共享 HTTP Client；
- 尝试加载 BM25 `CURRENT`；
- BM25 尚未建立时允许进程启动，但 readiness 为失败；
- 关闭时统一释放 HTTP Client 和数据库池。

REST 没有建立第二套依赖组装和业务对象，CLI 与 FastAPI 生命周期共享相同应用容器。

### 健康检查

新增：

```text
GET /health/live
GET /health/ready
```

`live` 只表示 HTTP 进程存活。

`ready` 要求：

- PostgreSQL 状态可读取；
- HNSW 索引存在；
- 数据库至少有一个 Chunk；
- BM25 generation 已加载；
- 最终 Search Service 已完成组装。

未摄取语料或未构建 BM25 时返回：

```http
503 Service Unavailable
```

而不是把“进程活着”错误地当作“系统可搜索”。

### Stats

新增：

```text
GET /v1/stats
```

返回：

- Document 数；
- Chunk 数；
- 当前 Embedding 模型和维度；
- HNSW 状态；
- BM25 ready；
- BM25 generation；
- BM25 Chunk 数。

数据库密码、DSN 和 API Key 不进入响应。

### Search

新增：

```text
POST /v1/search
```

请求支持：

- Query；
- Dense/BM25/RRF/Rerank/Final 参数覆盖；
- exact/HNSW；
- Reranker 开关；
- Debug。

未提供的参数使用同一份 `Settings`。Endpoint 构造核心 `SearchRequest` 并调用
`RAGApplication.search()`。

普通响应包含：

- Final results；
- 全部来源与阶段 rank/score；
- timings；
- candidate counts；
- Reranker 状态。

`debug=true` 额外包含：

- Dense；
- BM25；
- RRF；
- Reranker。

### Ingest

新增：

```text
POST /v1/ingest
Idempotency-Key: <stable-key>
```

请求接受本地 Markdown 路径、repo、commit、language 和可选 source root。未给 source
root 时优先向上寻找 `.git`，否则使用 Markdown 目录。

HTTP 映射：

- 相同 Key + 相同请求：200，返回首次结果；
- 相同 Key + 不同请求：409；
- 相同 Key 正在处理：409；
- 来源不存在：404；
- 非法路径或参数：422。

持久化幂等仍由阶段 1C 的数据库记录和事务实现，Endpoint 不维护进程内缓存。

### BM25 Rebuild

新增：

```text
POST /v1/bm25/rebuild
Idempotency-Key: <stable-key>
```

BM25 rebuild 使用数据库持久化请求状态，内部 Key 增加：

```text
bm25-rebuild:
```

命名空间，因此不会与摄取 Key 冲突。请求指纹包含操作名、Embedding 模型、维度和索引路径。

行为：

- 相同 Key 重复请求返回首次 generation 结果；
- 相同 Key 并发请求只有一个实际构建，另一个返回 409；
- 失败记录允许同 Key 重试；
- 服务重启后仍保留完成语义；
- 构建成功后原子切换 `CURRENT` 并刷新应用 Search Service。

沿用现有 `ingestion_requests` 作为本 Demo 的持久化操作请求账本，通过 Key 命名空间区分
操作类型，没有为单个额外操作增加新表。

### Serve CLI

新增：

```powershell
python -m rag_demo serve --host 127.0.0.1 --port 8000
```

默认仅监听回环地址。命令使用 Uvicorn factory 模式创建 FastAPI 应用。

## TDD 过程

先编写完整 REST 工作流测试，`rag_demo.api` 不存在时按预期在收集阶段失败；实现后测试执行：

1. live 为 200；
2. 空语料 ready 为 503；
3. REST 摄取一份真实 Markdown；
4. 重复摄取返回相同结果；
5. 同 Key 改变 commit 返回 409；
6. REST 重建 BM25；
7. 重复重建返回同一 generation 结果；
8. ready 变为 200；
9. stats 显示 1 Document、2 Chunk 和 BM25 2 Chunk；
10. Debug 搜索返回四层中间结果；
11. CLI 对同一 Query 搜索；
12. 比较 REST 与 CLI 的最终 Chunk ID 顺序；
13. 清理数据库和临时索引。

额外通过延迟注入放大 BM25 构建窗口，同时发送两个相同 Key 的请求，稳定验证返回码为：

```text
200
409
```

## 接口一致性证据

测试对同一个 Query、同一配置、同一 BM25 generation 分别执行：

```text
POST /v1/search
python -m rag_demo search
```

捕获 CLI 实际收到的核心 `SearchResponse`，与 REST JSON 中的最终 `chunk_id` 列表比较，
顺序完全一致。这证明一致性来自共用核心 Service，而不是只比较两套相似实现的表面输出。

## 验证证据

完整自动检查：

```text
79 passed
Total coverage: 88.05%
Ruff: passed
Ruff format: passed
Mypy strict: passed
```

环境清理结果：

```text
Documents: 0
Chunks: 0
Formal data/indexes/bm25/CURRENT: absent
```

测试使用 pytest 临时 BM25 路径，没有覆盖用户正式索引。

## 面试讲解要点

- live 与 ready 语义不同；没有语料或 BM25 时进程可存活但不可搜索。
- FastAPI lifespan 是资源所有者，Endpoint 不创建临时 Pool 或 HTTP Client。
- CLI 与 REST 只负责请求/响应适配，排名逻辑只有一份。
- REST Debug 直接序列化核心层保留的四阶段证据。
- 摄取和 BM25 rebuild 的幂等状态都持久化在 PostgreSQL。
- BM25 的数据库幂等解决重复执行，generation 原子发布解决文件系统失败安全，两者职责不同。
- HTTP 409 明确表达幂等冲突或处理中，不把它们伪装成 500。
- 异步 API 测试使用 ASGI transport 和真实 lifespan，没有启动外部测试端口。

## 当前限制与下一阶段

Milestone 3 已完成：核心异步 API、CLI、REST、Reranker、幂等与全阶段 Debug 均可用。

阶段 4 将完成：

- FastAPI 官方文档浅克隆命令；
- Git commit 自动记录；
- 固定 10 条 evaluation query；
- Recall@5、Recall@10、MRR@10、平均和 P95 延迟；
- Dense exact/HNSW、BM25、Hybrid、Reranker 对照；
- 最终 README 从零部署与现场演示脚本。
