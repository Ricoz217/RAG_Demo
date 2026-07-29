# 阶段 1D：Dense 精确检索与 HNSW

## 阶段目标

在同一份 pgvector 数据上提供可对照的 `exact` 和 `hnsw` 两种余弦检索模式，并通过
`EXPLAIN (ANALYZE, BUFFERS)` 证明两种模式实际使用了不同的 PostgreSQL 执行计划。

## 完成内容

### Dense Retriever

新增 `src/rag_demo/dense_retriever.py`，核心对象为：

- `DenseRetriever`
- `DenseQueryService`
- `DenseSearchMode`
- `DenseSearchResult`
- `DenseRetrievalResponse`
- `DenseQueryResponse`
- `DenseQueryPlan`

`DenseRetriever` 接收已经生成的查询向量，返回：

- Dense rank；
- Chunk ID 和 Document ID；
- Document 标题、仓库、commit、相对路径和语言；
- Heading 路径；
- 原文与检索文本；
- metadata；
- cosine distance 和 cosine similarity；
- 查询模式、候选数量和数据库检索耗时。

### Exact 模式

`exact` 模式在当前事务中设置：

```sql
SET LOCAL enable_indexscan = off;
SET LOCAL enable_bitmapscan = off;
```

随后执行 pgvector `<=>` 余弦距离排序。这样即使 HNSW 索引已经存在，exact 仍通过
Sequential Scan 计算所有符合向量空间条件的 Chunk，作为 ANN 的真值基线。

### HNSW 模式

`hnsw` 模式在当前事务中设置：

```sql
SET LOCAL enable_seqscan = off;
SELECT set_config('hnsw.ef_search', configured_value, true);
```

查询使用参数化向量、模型、维度和 Top-K，不拼接用户输入。`ef_search` 只在当前事务生效，
不会污染连接池中该连接的后续请求。

### 先向量 Top-K，再关联来源

初版 SQL 直接 JOIN `documents` 后再按向量距离排序。在测试数据量很小时，PostgreSQL
规划器选择：

```text
documents_pkey
→ chunks_document_id_idx
→ Sort
```

这条计划虽然结果正确，却没有实际使用 HNSW。最终 SQL 先在 `chunks` 子查询内完成向量
Top-K，再 JOIN `documents` 补齐来源信息。这样 HNSW 模式的计划明确出现：

```text
Index Scan using chunks_embedding_hnsw on chunks
```

这个调整同时避免了为了展示索引而删除必要的 Document 来源字段。

### 向量空间隔离

所有 Dense 查询都要求：

```sql
embedding_model = configured_model
embedding_dimensions = configured_dimensions
```

`DenseQueryService` 在构造时也校验查询 Embedding Client 与 Retriever 的模型名称和维度
一致。相同维度但不同模型的向量不会被混合比较。

### 查询向量与校验

查询向量必须：

- 是一维向量；
- 维度与数据库配置一致；
- 全部为有限数值；
- 不是全零向量。

`DenseQueryService` 使用 Embedding Client 为一条非空 Query 生成一个向量，并分别记录：

- `query_embedding_ms`
- `dense_search_ms`
- `total_ms`
- `candidate_count`

## TDD 过程

先编写真实 PostgreSQL 集成测试，模块不存在时测试收集按预期失败；随后实现最小检索层。
固定向量数据覆盖：

- exact 的真实 cosine 排序；
- rank 从 1 连续编号；
- Document/Chunk 来源元数据映射；
- 其他 Embedding 模型的 Chunk 被排除；
- 非法维度和 NaN 向量被拒绝；
- Query 只调用一次 Embedding；
- 查询各阶段耗时可见。

第一次 HNSW 计划测试失败，暴露了 JOIN 顺序让规划器绕过 HNSW 的问题。调整为“向量 Top-K
子查询后再 JOIN”后，exact 和 HNSW 的计划证据同时转绿。

## 验证证据

完整自动检查：

```text
55 passed
Total coverage: 89.12%
Ruff: passed
Ruff format: passed
Mypy strict: passed
```

固定向量验证：

```text
query = [1, 0, ...]
nearest similarity =  1.0
middle  similarity =  0.0
farthest similarity = -1.0
```

执行计划验证：

```text
exact: Seq Scan
hnsw: Index Scan using chunks_embedding_hnsw
hnsw.ef_search: 100
```

真实链路验证：

```text
Markdown
→ Chunk
→ llama.cpp bge-m3 Embedding
→ PostgreSQL vector(1024)
→ exact Top-1
→ HNSW Top-1
```

使用目标 Chunk 的完整 `retrieval_text` 作为 Query，exact 与 HNSW 均返回该 Chunk，
cosine similarity 约为 1。

## 面试讲解要点

- exact 与 HNSW 不是两个不同的业务检索器，而是同一 pgvector 数据和距离算子的两种执行方式。
- exact 用作召回基线；HNSW 用于展示近似检索和延迟收益。
- 不能只证明“建过 HNSW 索引”，必须用 `EXPLAIN ANALYZE` 证明查询实际命中。
- SQL 的 JOIN 与 ORDER BY 结构会影响规划器是否能使用向量索引。
- `SET LOCAL` 将 planner 开关和 `ef_search` 限制在单个事务内，适合连接池。
- 模型名与维度共同定义向量空间；只有维度相同并不足以安全混用。
- cosine distance 越小越近，Demo 同时输出 `1 - distance` 作为更直观的 similarity。

## 当前限制与下一阶段

Milestone 1 的数据库、Markdown、Embedding、摄取、exact 和 HNSW 已形成完整闭环。当前只有
Dense 分支，尚不能利用代码标识符或精确术语的词法匹配。

阶段 2 将使用 `bm25s + jieba`：

- 从 PostgreSQL Chunk 全量重建本地 BM25 派生索引；
- 实现中文和代码标识符分词；
- 通过 `chunk_id` 对齐数据库事实源；
- 使用 RRF 融合 Dense 和 BM25 排名。
