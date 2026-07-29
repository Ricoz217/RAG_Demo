# 阶段 1C：Embedding 与摄取

## 阶段目标

把阶段 1B 产生的 Chunk 送入本地 llama.cpp Embedding 服务，并以可重试、可并发、
可增量更新的方式写入 PostgreSQL。该阶段完成从 Markdown 文件到 pgvector 的首条真实
端到端链路。

## 完成内容

### Embedding Client

新增 `src/rag_demo/embedding_client.py`：

- 复用外部传入的 `httpx.AsyncClient`，避免每批请求重复建立连接；
- 使用 OpenAI 兼容的 `/v1/embeddings` 接口；
- 按配置进行批量请求，并携带 Bearer API Key；
- 不依赖服务端响应顺序，而是根据 `index` 重排结果；
- 校验返回条数、索引连续性、维度和所有数值的有限性；
- 统一输出 NumPy `float32` 数组，直接满足 pgvector-python 的类型契约；
- 分别设置连接、写入、连接池和可配置读取超时。

Embedding 服务契约为：

```text
model: bge-m3
dimensions: 1024
endpoint: /v1/embeddings
```

服务地址和密钥只从 `.env` 加载，测试和施工记录均不保存实际密钥。

### Document 摄取事务

新增 `src/rag_demo/ingest_service.py`，实现 `DocumentIngestor`。单个 Markdown 文件的
摄取流程为：

1. 异步读取文件并解析 Markdown；
2. 生成结构化 Chunk 和统一的 `retrieval_text`；
3. 获取以来源 revision 为键的 PostgreSQL transaction advisory lock；
4. upsert Document；
5. 识别可以复用 Embedding 的 Chunk；
6. 只为新增或已变化的 Chunk 调用 Embedding；
7. upsert 当前 Chunk，并删除该 Document 已不存在的旧 Chunk；
8. 在同一事务中提交全部修改。

同一来源的并发摄取由数据库锁串行化，不依赖单进程内的 Python 锁。不同进程或未来多个
API worker 也遵循相同互斥规则。

### 增量 Embedding 判定

只有以下条件全部一致时才复用已有向量：

- `content_hash`
- Chunker 版本
- Embedding 模型
- Embedding 维度

`content_hash` 改为基于 `retrieval_text`，而不是只基于 `content_raw`。这是因为 Document
标题和 Heading breadcrumb 同样会进入 Embedding 输入；即使正文不变，只要检索上下文变化，
旧向量就必须失效。

### 持久化幂等语义

`ingestion_requests` 保存 `Idempotency-Key`、请求指纹、状态、结果或错误：

- 同一个 Key 和同一个请求重复提交，直接返回第一次的完成结果；
- 同一个 Key 对应不同请求时明确报冲突；
- 正在处理的请求不会被第二次执行；
- 失败请求会留下失败状态，并允许使用同一个 Key 重试；
- 完成结果以 JSONB 保存，进程重启后仍然可复用。

请求指纹包含来源路径、仓库、commit、语言、Chunker 版本、Embedding 模型和维度，防止环境
或算法配置变化后错误复用旧请求。

### 摄取统计

每次成功结果记录：

- 扫描和写入的 Document 数；
- 总 Chunk 数；
- 新生成与跳过的 Embedding 数；
- 删除的过期 Chunk 数；
- Embedding 请求批次数；
- 总耗时。

这些数据后续可以直接用于 CLI/API 返回和 Demo 讲解。

## TDD 过程

先使用 `httpx.MockTransport` 编写 Embedding Client 行为测试，再实现最小客户端；随后使用
真实 PostgreSQL 编写摄取集成测试，覆盖：

- 返回顺序重排、分批、鉴权和异常响应；
- 非法数量、索引、维度、NaN/Infinity 拒绝；
- 一个真实 Markdown 文件完成解析、Embedding 和 pgvector 写入；
- 重复幂等请求、Key 冲突、处理中状态和失败重试；
- 内容不变时跳过 Embedding；
- Heading 改变使检索文本 hash 失效；
- Embedding 模型切换时重新计算向量；
- 文件缩短后删除过期 Chunk；
- 同一来源使用不同 Key 并发摄取时保持串行和最终一致。

集成过程中发现并修复两个实际问题：

1. PostgreSQL `text` 不允许 NUL 字符，advisory lock 的稳定键改为规范化 JSON 文本；
2. 同正文不同 Heading 的向量输入不同，因此 Chunk hash 必须覆盖完整 `retrieval_text`。

## 验证证据

完整自动检查结果：

```text
50 passed
Total coverage: 89.43%
Ruff: passed
Ruff format: passed
Mypy strict: passed
```

真实基础设施验证包括：

- 调用当前运行中的 llama.cpp Embedding REST 服务；
- 返回两条顺序正确的 1024 维 `float32` 向量；
- 将真实 Markdown 解析、分块、生成向量并写入 PostgreSQL `vector(1024)`；
- 从数据库读回向量并验证维度和非空数据；
- 所有集成测试完成后清理数据。

最终数据库状态：

```text
PostgreSQL       18.4
pgvector         0.8.5
Documents        0
Chunks           0
```

## 面试讲解要点

- 幂等不等于“内容相同就跳过”：它同时需要请求级幂等和 Chunk 级增量计算。
- 请求级幂等持久化在 PostgreSQL 中，因此不受 API 进程重启影响。
- 同来源互斥使用 PostgreSQL advisory lock，可覆盖多进程并发场景。
- Document、Chunk、过期数据删除和请求完成状态在事务边界内保持一致。
- Embedding 输入是带标题与章节上下文的 `retrieval_text`，hash 也必须覆盖相同输入。
- NumPy `float32` 是客户端与 pgvector 之间的明确向量类型契约。
- 文件读取通过工作线程执行，避免同步磁盘 I/O 阻塞异步事件循环。

## 当前限制与下一阶段

当前已能可靠地把 Markdown 写入向量库，但尚未提供 Dense 查询。阶段 1D 将实现：

- pgvector 余弦精确检索；
- HNSW 近似检索；
- 查询向量生成与结果模型；
- exact/ANN 对照测试及真实端到端验证。
