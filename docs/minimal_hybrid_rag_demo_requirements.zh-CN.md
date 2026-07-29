# 最小可演示 Hybrid RAG 系统需求书

## 1. 项目定位

实现一个可以在本机从零部署、可重复运行、可观察各阶段结果的最小 Hybrid RAG 检索 Demo。项目只用于证明能够正确搭建和调用一条标准 RAG 检索链路，不用于研究算法实现，也不作为后续产品或长期维护项目的开发基底。

本项目只覆盖检索链路，不接入 Agent，不调用生成式 LLM，不生成最终自然语言答案：

```text
公开文档
  → 解析与 Chunk
  → llama.cpp Embedding API
  → PostgreSQL + pgvector
  → Dense ANN / HNSW
  ┐
  ├→ RRF 融合 → llama.cpp Rerank API → 输出最终证据结果
  ┘
  → BM25
```

项目目标不是构建生产级知识库，而是用一个足够小、足够透明的实现证明以下能力：

1. 能解释并搭建从文档摄取到检索结果的完整链路。
2. 能展示 Dense、BM25、Hybrid、Reranker 各自对结果的影响。
3. 能展示 pgvector 的 `vector(1024)`、余弦距离和 HNSW ANN 索引。
4. 能通过 CLI 和 REST 运行查询并观察排名、分数、来源和耗时。
5. Embedding 使用标准 API 协议，与模型部署职责分离。
6. 能调用成熟库的功能不自行复现；可读、可运行、可演示优先于算法实现细节。

## 2. 是否足以作为面试 Demo

本范围足够，而且比加入 Agent 或回答生成更适合作为检索方向的面试 Demo。

最终演示必须能清楚回答：

- 文档如何切分，为什么这样切。
- Embedding 在哪里生成，维度是多少。
- pgvector 中保存了什么。
- HNSW 如何建立和命中。
- Dense 和 BM25 分别找到了什么。
- RRF 如何融合两个排名。
- Cross-Encoder 为什么改变最终顺序。
- HNSW 相比精确向量扫描的结果和耗时有什么差异。
- 某条结果来自哪个文件、章节和 Chunk。

如果只能展示一次最终查询结果，而不能显示中间排名和对照实验，则 Demo 的说服力不足。

## 3. 明确不做的内容

第一版不得主动扩展以下功能：

- Agent、Tool Calling、LangGraph 工作流。
- 生成式回答、引用生成、对话历史。
- Query Rewrite、HyDE、Multi-Query。
- Graph RAG、知识图谱、实体关系抽取。
- 用户系统、权限系统、前端页面。
- Redis、消息队列、分布式任务。
- Elasticsearch、OpenSearch、专用向量数据库。
- 文档 OCR、PDF 表格重建、多模态解析。
- 微服务拆分和 Kubernetes。
- 手写 BM25、手写向量索引或自行实现 Cross-Encoder。
- 为未知的后续需求预留复杂扩展层。

## 4. 本机环境审计结果

审计日期：2026-07-29。

### 4.1 硬件

| 项目 | 本机结果 | 结论 |
|---|---|---|
| 操作系统 | Windows 11 专业工作站版，64-bit | 满足 |
| CPU | Intel Core i9-13900K，24 核、32 线程 | 充足 |
| 内存 | 63.75 GB | 充足 |
| GPU | NVIDIA GeForce RTX 4090 | 充足 |
| 显存 | 24564 MiB，约 24 GB | 充足 |
| NVIDIA 驱动 | 581.42 | 满足 |
| 驱动报告 CUDA | 13.0 | llama.cpp 模型服务可使用该 GPU；具体参数由用户负责 |

当前 GPU 已被桌面程序和 NVIDIA Broadcast 占用约 2.9 GiB，剩余显存仍足够运行本 Demo 的 Embedding 与 Reranker。

### 4.2 软件环境

| 项目 | 当前状态 |
|---|---|
| Python 3.12 | 已安装：`C:\Users\Rico\AppData\Local\Programs\Python\Python312\python.exe` |
| 默认 `python` | 当前指向另一个 Python 3.9 虚拟环境，不得使用 |
| `uv` | 未进入 PATH |
| Docker | 未进入 PATH |
| `psql` | 未进入 PATH |
| PostgreSQL / pgvector | 由用户安装并提供连接参数 |
| llama.cpp 模型服务 | 由用户独立部署并提供 Embedding/Reranker URL |

项目实现必须使用：

```powershell
py -3.12
```

创建独立的项目级 `.venv`，不得复用当前默认 `python` 指向的旧虚拟环境。RAG 应用本身不安装 PyTorch、Transformers 或模型权重；GPU 和模型运行状态由外部 llama.cpp 服务负责。

## 5. 技术选型

### 5.1 编程语言与工程方式

- Python 3.12。
- 使用对象化、异步的摄取和检索服务。
- 不使用 LangChain 和 LangGraph。
- 同时提供异步 Python API、CLI 和 FastAPI REST。
- CLI、REST 必须调用同一套核心 Service，不得复制业务逻辑。
- 数据库使用 Psycopg 3 异步连接池，HTTP 使用 `httpx.AsyncClient`。

不使用 LangChain/LangGraph 的理由：

- 当前管线是线性的，没有需要状态机表达的复杂分支。
- 直接调用 pgvector、BM25 库和模型 API 更容易观察完整链路。
- 这不代表需要自行实现 BM25、ANN 或模型推理。

### 5.2 Embedding 服务

模型由用户通过 llama.cpp 独立部署，RAG 应用只依赖 OpenAI 兼容的 Embedding API：

```http
POST {EMBEDDING_BASE_URL}/v1/embeddings
Authorization: Bearer {EMBEDDING_API_KEY}
Content-Type: application/json
```

```json
{
  "model": "configured-model-name",
  "input": [
    "first text",
    "second text"
  ],
  "encoding_format": "float"
}
```

要求：

- 使用异步 HTTP 客户端。
- 支持单条 Query 和批量文档输入。
- 按响应中的 `index` 恢复批量结果顺序。
- 校验向量数量、有限数值和维度。
- Demo 默认数据库维度为 1024。
- 模型名称、URL、API Key、超时和批量大小全部配置化。
- 摄取和查询必须使用相同的模型标识和维度。
- 数据库记录当前语料使用的 `embedding_model` 和 `embedding_dimensions`。
- 切换模型不修改业务代码，但必须重新 Embedding 全部 Chunk；即使新旧模型维度相同，两个向量空间也不能混用。
- 如果新模型维度不是 1024，需要重建向量列和 HNSW 索引。

llama.cpp Server 参考：

- <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>

### 5.3 Reranker 服务

Reranker同样由用户通过独立 llama.cpp 服务部署。RAG 应用调用配置的 Rerank URL：

```http
POST {RERANKER_BASE_URL}/reranking
Authorization: Bearer {RERANKER_API_KEY}
Content-Type: application/json
```

请求语义：

```json
{
  "query": "user query",
  "documents": [
    "candidate passage 1",
    "candidate passage 2"
  ]
}
```

- 使用异步 HTTP 客户端。
- 对 RRF 后的候选重新打分。
- 批量提交候选文本。
- 校验返回结果能映射回原始候选。
- Reranker 分数只用于相对排序，不在输出中称为“正确概率”。
- 允许通过配置或 CLI 关闭 Reranker。
- llama.cpp 的 Rerank 接口不是 OpenAI 标准接口，因此封装在一个很薄的 `RerankerClient` 中，不为其他未来后端设计复杂抽象。

### 5.4 Dense 存储和 ANN

- PostgreSQL。
- pgvector 扩展。
- `vector(1024)`。
- Cosine distance。
- HNSW 索引。
- Python 连接使用 Psycopg 3。
- Python 向量适配使用 `pgvector-python`。

参考：

- <https://github.com/pgvector/pgvector>
- <https://pypi.org/project/pgvector/>

### 5.5 BM25

第一版直接使用成熟 BM25 库，不自行实现公式、倒排索引或增量索引。默认使用 `bm25s + jieba`；如果实施者发现等价库在 Windows/Python 3.12 下更容易安装，可以替换，但必须保留相同的输入输出行为。

使用：

```text
bm25s + jieba
```

要求：

- Chunk 正文和 metadata 仍以 PostgreSQL 为事实源。
- BM25 索引是可以从 PostgreSQL Chunk 全量重建的本地派生文件。
- 中文使用 `jieba.lcut_for_search` 或等价搜索分词。
- 拉丁字母统一小写。
- 保留代码标识符、下划线名称、HTTP 状态码和常见技术词。
- BM25 结果通过 `chunk_id` 与 PostgreSQL 中的 Chunk 对齐。
- 不要求讲解或测试 BM25 内部公式，只验证该检索分支能正确返回候选。

BM25 索引建议保存到：

```text
data/indexes/bm25/
```

参考：

- <https://github.com/xhluca/bm25s>

### 5.6 排名顺序

正式查询链路固定为：

```text
Dense HNSW Top 30
        ┐
        ├→ RRF → Top 20 → Cross-Encoder → Final Top 5
        ┘
BM25 Top 30
```

默认参数：

```text
dense_top_k = 30
bm25_top_k = 30
rrf_rank_constant = 60
rerank_top_k = 20
final_top_k = 5
```

所有参数必须可通过配置文件或 CLI 覆盖。

## 6. 演示语料

默认使用 FastAPI 官方仓库中的中文文档：

```text
https://github.com/fastapi/fastapi
```

摄取范围优先：

```text
docs/zh/docs/**/*.md
```

如果某些中文页面缺失，可以同时摄取：

```text
docs/en/docs/**/*.md
```

要求：

1. 提供下载语料的脚本或 CLI 命令。
2. 使用浅克隆即可。
3. 摄取时记录 Git commit SHA。
4. 不把整个 FastAPI 仓库提交进 Demo 仓库。
5. 本地语料目录加入 `.gitignore`。
6. 每个 Chunk 保存仓库 URL、commit SHA 和源文件相对路径。

FastAPI 项目及文档采用 MIT License：

- <https://github.com/fastapi/fastapi>
- <https://github.com/fastapi/fastapi/blob/master/LICENSE>

## 7. 文档解析与 Chunk

### 7.1 支持格式

第一版只支持：

```text
.md
```

不得主动加入 PDF、DOCX、HTML 解析。

### 7.2 Markdown 处理

需要识别：

- 文档标题。
- Heading 层级。
- 普通段落。
- 列表。
- 代码块。

每个 Chunk 必须保留完整章节路径：

```json
{
  "heading_path": [
    "教程",
    "请求体",
    "多个参数"
  ]
}
```

### 7.3 Chunk 策略

Chunk 不依赖模型进程内 tokenizer。按 Markdown 结构和字符长度切分，避免为了取得精确 token 数再次引入模型依赖。

默认值：

```text
target_chars = 1400
max_chars = 2200
overlap_chars = 200
```

规则：

1. 优先按 Heading 和段落边界切分。
2. 在不超过 `max_chars` 的前提下保持代码块完整。
3. 超大代码块允许单独成为 Chunk 或二次切分。
4. 不得只按固定字符数无脑切分。
5. overlap 不应跨越不相关的一级章节。

### 7.4 原文与检索文本

分别保存：

```text
content_raw
retrieval_text
```

`retrieval_text` 由以下内容组成：

```text
文档标题
Heading Breadcrumb
Chunk 原文
```

例如：

```text
[文档：FastAPI 教程]
[章节：安全 / OAuth2 / JWT]
原始 Chunk 内容……
```

Embedding、BM25 和 Reranker默认使用 `retrieval_text`。

### 7.5 幂等性

重复执行摄取不得生成重复 Chunk。

至少保存：

- `source_path`
- `source_commit`
- `chunk_index`
- `content_hash`
- `chunker_version`
- `embedding_model`
- `embedding_dimensions`

可使用：

```text
(source_path, source_commit, chunk_index, chunker_version)
```

作为逻辑唯一键。

幂等性要求：

1. 同一来源、同一 revision、同一模型重复摄取，不得新增重复 Document 或 Chunk。
2. Document 与 Chunk 依靠数据库唯一约束兜底，不能只在应用层先查后插。
3. `content_hash`、`chunker_version`、`embedding_model` 和维度都未变化时跳过重复 Embedding。
4. 单个来源的 Chunk 使用 `(document_id, chunk_index)` Upsert，并在同一事务中删除新版本不再存在的旧 Chunk；失败不得留下半份可查询数据。
5. 并发摄取同一来源时使用 PostgreSQL advisory lock 或等价数据库锁串行化。
6. 同一来源切换 `embedding_model` 时，更新现有 Chunk 的向量和模型标识，不复制 Document/Chunk。
7. REST 摄取要求使用 `Idempotency-Key` 请求头。相同 Key 和相同请求体重复调用返回第一次结果；相同 Key 携带不同请求体返回 `409 Conflict`。
8. 同一 Key 正在处理时返回明确的处理中状态；失败记录允许使用同一 Key 重试。
9. BM25 索引重建先写临时目录，完成后原子替换正式索引，失败时继续使用旧索引。
10. Search 接口无副作用，天然幂等。

## 8. PostgreSQL Schema

最低要求：

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    source_repo text NOT NULL,
    source_commit text NOT NULL,
    source_path text NOT NULL,
    language text NOT NULL,
    title text,
    content_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),

    UNIQUE (source_repo, source_commit, source_path)
);

CREATE TABLE chunks (
    id bigserial PRIMARY KEY,
    document_id bigint NOT NULL
        REFERENCES documents(id) ON DELETE CASCADE,

    chunk_index integer NOT NULL,
    chunker_version text NOT NULL,
    heading_path text[] NOT NULL DEFAULT '{}',
    content_raw text NOT NULL,
    retrieval_text text NOT NULL,
    char_count integer NOT NULL,
    content_hash text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    embedding_model text NOT NULL,
    embedding_dimensions integer NOT NULL,
    embedding vector(1024) NOT NULL,

    UNIQUE (document_id, chunk_index)
);

CREATE TABLE ingestion_requests (
    idempotency_key text PRIMARY KEY,
    request_hash text NOT NULL,
    status text NOT NULL,
    result jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
```

HNSW：

```sql
CREATE INDEX chunks_embedding_hnsw
ON chunks
USING hnsw (
    embedding vector_cosine_ops
)
WITH (
    m = 16,
    ef_construction = 64
);
```

建议附加索引：

```sql
CREATE INDEX chunks_document_id_idx
ON chunks (document_id);

CREATE INDEX chunks_metadata_gin
ON chunks
USING gin (metadata);
```

查询时允许设置：

```sql
SET LOCAL hnsw.ef_search = 100;
```

## 9. Dense 查询

使用参数化 SQL：

```sql
SELECT
    c.id,
    c.document_id,
    c.content_raw,
    c.retrieval_text,
    c.heading_path,
    c.metadata,
    c.embedding <=> $1 AS cosine_distance,
    1 - (c.embedding <=> $1) AS cosine_similarity
FROM chunks c
ORDER BY c.embedding <=> $1
LIMIT $2;
```

必须支持两种模式：

```text
exact
hnsw
```

用途：

- `exact`：作为召回基线。
- `hnsw`：展示 ANN。

测试中应能通过 `EXPLAIN (ANALYZE, BUFFERS)` 验证 HNSW 索引是否实际使用。

## 10. RRF

RRF 可以使用成熟库，也可以用一个很短的本地函数实现；不把“手写算法”作为项目目标。无论具体实现方式，行为应符合：

```text
score(chunk) =
    Σ 1 / (rank_constant + rank_i(chunk))
```

要求：

- 输入为 Dense 和 BM25 的有序 `chunk_id` 列表。
- 结果按 `chunk_id` 取并集。
- 某 Chunk 不在某一路中时，该路贡献为 0。
- 输出保存各路 rank 和原始 score。
- RRF score 只表示融合排序，不作为概率。
- 使用小型固定输入测试融合顺序、缺失候选和重复候选行为。

建议候选对象：

```json
{
  "chunk_id": 123,
  "dense_rank": 3,
  "dense_score": 0.82,
  "bm25_rank": 1,
  "bm25_score": 11.4,
  "rrf_score": 0.0322,
  "rerank_score": null
}
```

## 11. Cross-Encoder Rerank

RRF Top 20 进入 Reranker：

```text
(query, retrieval_text)
```

要求：

- 通过异步 HTTP 批量请求 llama.cpp Reranker 服务。
- 按 Reranker score 降序排列。
- 最终返回 Top 5。
- 保留 RRF 之前的所有排名信息，不能覆盖。
- 记录 Rerank 阶段耗时。
- 可通过 CLI `--no-rerank` 关闭，便于对照。

## 12. CLI

建议命令：

```powershell
py -3.12 -m rag_demo doctor
py -3.12 -m rag_demo db init
py -3.12 -m rag_demo corpus download-fastapi
py -3.12 -m rag_demo ingest --source data/corpus/fastapi/docs/zh/docs
py -3.12 -m rag_demo bm25 rebuild
py -3.12 -m rag_demo search "FastAPI 如何声明请求体？"
py -3.12 -m rag_demo search "OAuth2PasswordBearer 有什么作用？" --debug
py -3.12 -m rag_demo compare "依赖项在一次请求中会执行几次？"
py -3.12 -m rag_demo benchmark
py -3.12 -m rag_demo serve --host 127.0.0.1 --port 8000
```

### 12.1 `doctor`

必须检查：

- Python 版本。
- llama.cpp Embedding API 是否可访问。
- 使用测试文本调用 Embedding API，并确认返回 1024 维有限数值。
- llama.cpp Reranker API 是否可访问并能对两个测试文本返回排序结果。
- PostgreSQL 是否可连接。
- pgvector 扩展是否存在。
- BM25 索引是否存在。
- 数据库 Chunk 数量。
- 当前配置的 Embedding 模型是否与已摄取语料记录一致。

### 12.2 `search`

普通输出：

```text
Query
Final Top 5
每条结果的标题、章节、源文件、文本预览、最终分数
各阶段耗时
```

`--debug` 输出：

```text
Dense Top-K
BM25 Top-K
RRF Top-K
Reranker Top-K
每条结果在各阶段的 rank 和 score
```

### 12.3 `compare`

同一个 Query 输出四组结果：

```text
BM25 only
Dense only
Hybrid RRF
Hybrid RRF + Reranker
```

这是本项目最重要的面试演示命令。

## 13. 异步 Python API 与 FastAPI REST

### 13.1 核心对象接口

核心对象全部提供异步接口：

```python
class DocumentIngestor:
    async def ingest_path(
        self,
        source: Path,
        *,
        idempotency_key: str,
    ) -> IngestResult:
        ...


class HybridSearchService:
    async def search(
        self,
        request: SearchRequest,
    ) -> SearchResponse:
        ...
```

要求：

- 使用 `psycopg_pool.AsyncConnectionPool`。
- 使用一个可复用的 `httpx.AsyncClient`，不得每次请求重新创建客户端。
- Dense 和 BM25 两个检索分支使用 `asyncio.TaskGroup` 或 `asyncio.gather` 并行执行。
- `bm25s` 是同步 CPU 库，查询和重建通过 `asyncio.to_thread()` 包装，避免阻塞 FastAPI 事件循环。
- CLI 入口使用 `asyncio.run()` 调用同一核心 Service。
- 不为玩具项目增加依赖注入框架；应用启动时手动组装对象即可。

### 13.2 REST 接口

FastAPI 至少提供：

```text
GET  /health/live
GET  /health/ready
GET  /v1/stats
POST /v1/search
POST /v1/ingest
POST /v1/bm25/rebuild
```

`POST /v1/search` 请求示例：

```json
{
  "query": "FastAPI 的依赖项会不会重复执行？",
  "dense_top_k": 30,
  "bm25_top_k": 30,
  "rerank_top_k": 20,
  "final_top_k": 5,
  "debug": true
}
```

`POST /v1/ingest` 只服务本机 Demo，可接受本地 Markdown 目录：

```json
{
  "source": "data/corpus/fastapi/docs/zh/docs",
  "source_repo": "https://github.com/fastapi/fastapi",
  "source_commit": "git-commit-sha"
}
```

要求请求头：

```http
Idempotency-Key: caller-generated-stable-key
```

本 Demo 的摄取接口可以等待摄取完成后直接返回，不引入任务队列和 Job Worker。服务默认只监听 `127.0.0.1`。

`POST /v1/bm25/rebuild` 同样要求 `Idempotency-Key`，重复请求不得并发覆盖索引。

### 13.3 应用生命周期

FastAPI lifespan 中创建并复用：

```text
PostgreSQL AsyncConnectionPool
httpx.AsyncClient
EmbeddingClient
RerankerClient
BM25 Index
HybridSearchService
DocumentIngestor
```

关闭应用时正确关闭 HTTP Client 和数据库连接池。

## 14. 测试查询

至少准备 10 条固定测试查询，覆盖：

### 14.1 精确术语

```text
OAuth2PasswordBearer 有什么作用？
BackgroundTasks 应该怎么使用？
422 状态码通常在什么情况下出现？
```

预期 BM25 对精确名称和代码标识符表现较好。

### 14.2 语义表达

```text
怎样让接口接收一个 JSON 对象？
如何让多个接口共享同一段参数检查逻辑？
请求结束以后怎样执行不阻塞响应的任务？
```

预期 Dense 能找到未使用完全相同词语的相关章节。

### 14.3 混合检索

```text
FastAPI 的依赖项在同一次请求里会不会被反复执行？
如何使用 OAuth2 和 JWT 保护需要登录的接口？
上传文件和普通表单字段能否一起接收？
如何给 API 定义自定义异常处理器？
```

预期 Hybrid 和 Reranker综合表现更稳定。

每条查询应保存：

- 预期源文件或路径关键字。
- 预期 Heading 关键字。
- 是否属于 exact-term 或 semantic-query。

## 15. 最小评测

不得只凭肉眼说“效果不错”。

建立一个小型：

```text
evaluation_queries.json
```

格式示例：

```json
[
  {
    "query": "怎样让接口接收一个 JSON 对象？",
    "relevant_path_contains": [
      "tutorial/body"
    ],
    "relevant_heading_contains": [
      "请求体"
    ]
  }
]
```

至少计算：

- Recall@5
- Recall@10
- MRR@10
- 平均查询耗时
- P95 查询耗时

分别对比：

```text
BM25
Dense exact
Dense HNSW
Hybrid RRF
Hybrid RRF + Reranker
```

数据集很小，指标不具有学术结论意义，但可以证明系统可观测、可比较。

## 16. 性能与可观测性

每次查询记录：

```text
query_embedding_ms
dense_search_ms
bm25_search_ms
rrf_ms
rerank_ms
total_ms
dense_candidate_count
bm25_candidate_count
union_candidate_count
```

摄取记录：

```text
document_count
chunk_count
parse_ms
embedding_ms
database_insert_ms
bm25_build_ms
embedding_batch_size
embedding_http_request_count
```

日志中不得打印数据库密码或完整 DSN。

## 17. 配置

通过 `.env` 或配置文件提供：

```text
DATABASE_URL
EMBEDDING_BASE_URL
EMBEDDING_API_KEY
EMBEDDING_MODEL
EMBEDDING_DIMENSIONS
EMBEDDING_TIMEOUT_SECONDS
EMBEDDING_BATCH_SIZE
RERANKER_BASE_URL
RERANKER_API_KEY
RERANKER_MODEL
RERANKER_TIMEOUT_SECONDS
CHUNK_TARGET_CHARS
CHUNK_MAX_CHARS
CHUNK_OVERLAP_CHARS
DENSE_TOP_K
BM25_TOP_K
RRF_RANK_CONSTANT
RERANK_TOP_K
FINAL_TOP_K
HNSW_EF_SEARCH
```

仓库只提交：

```text
.env.example
```

不得提交真实 `.env`。

## 18. 建议依赖

基础依赖：

```text
fastapi
uvicorn
httpx
psycopg[binary]
psycopg-pool
pgvector
bm25s
jieba
numpy
pydantic
pydantic-settings
typer
rich
markdown-it-py
```

开发依赖：

```text
pytest
pytest-cov
ruff
mypy
```

版本应在实施时锁定，并生成可复现的 lock 文件。

## 19. 建议目录

该 Demo 建议新建独立目录，不混入当前已有的日志、简历和演示项目代码：

```text
projects/minimal_hybrid_rag_demo/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── migrations/
│   ├── 001_enable_vector.sql
│   └── 002_create_tables.sql
├── src/
│   └── rag_demo/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── api.py
│       ├── application.py
│       ├── config.py
│       ├── db.py
│       ├── models.py
│       ├── corpus.py
│       ├── markdown_parser.py
│       ├── chunker.py
│       ├── embedding_client.py
│       ├── dense_retriever.py
│       ├── bm25_retriever.py
│       ├── rrf.py
│       ├── reranker_client.py
│       ├── search_service.py
│       ├── ingest_service.py
│       └── evaluation.py
├── tests/
│   ├── unit/
│   └── integration/
└── data/
    ├── corpus/
    ├── indexes/
    └── evaluation_queries.json
```

## 20. 测试要求

最低单元测试：

- Markdown Heading 路径解析。
- Chunk 字符上限和 overlap。
- content hash 稳定性。
- 重复摄取和 `Idempotency-Key` 幂等性。
- 相同 `Idempotency-Key` 携带不同请求体时返回冲突。
- 中文 BM25 tokenizer。
- RRF 融合顺序。
- RRF 缺失列表项。
- 候选合并和去重。
- Reranker 排序。
- 配置加载和错误提示。
- 使用 Mock HTTP 验证 Embedding 批量结果按 `index` 还原。
- 使用 Mock HTTP 验证 Reranker 结果映射。

最低集成测试：

- PostgreSQL + pgvector 连接。
- 建表和 HNSW 索引。
- 插入 1024 维向量。
- exact 查询返回正确最近邻。
- HNSW 查询可执行。
- 一份 Markdown 从摄取到最终搜索的端到端测试。
- FastAPI `/v1/search`、`/v1/ingest` 和健康检查。
- CLI 与 REST 对同一查询返回相同最终 Chunk 顺序。

测试不得依赖真实 GPU 或真实 llama.cpp 服务；HTTP 模型服务使用 Mock。另提供手工端到端命令连接用户部署的真实服务。

## 21. 验收标准

满足以下全部条件才算完成：

1. 使用 Python 3.12 独立虚拟环境。
2. `doctor` 能通过 HTTP 调用用户部署的 llama.cpp Embedding 和 Reranker 服务。
3. PostgreSQL 已启用 `vector` 扩展。
4. `chunks.embedding` 为 `vector(1024)`。
5. HNSW 索引创建成功。
6. FastAPI 中文文档能够一条命令摄取。
7. 重复摄取不会重复写入 Chunk。
8. BM25 索引可以从 PostgreSQL 重建。
9. Dense、BM25、RRF、Reranker 均有可见中间结果。
10. `compare` 能展示四套检索结果。
11. `benchmark` 能输出 Recall、MRR 和延迟。
12. 最终结果包含源文件、Heading 路径和文本预览。
13. 至少一个测试 Query 能展示 Dense 找到 BM25 漏掉的结果。
14. 至少一个测试 Query 能展示 BM25 对精确术语的优势。
15. 至少一个测试 Query 能展示 Reranker 调整候选顺序。
16. README 包含从环境创建到现场演示的完整命令。
17. 提供可直接 `await` 的 Python 摄取和查询接口。
18. FastAPI REST 查询可用。
19. CLI 和 REST 复用同一核心 Service。
20. 相同语料重复摄取不会增加 Document/Chunk 数量。
21. 并发重复摄取不会产生重复行或半完成数据。

## 22. 实施顺序

严格按以下顺序实现：

### Milestone 1：数据库和 Dense

- 项目初始化。
- PostgreSQL 连接。
- pgvector migration。
- Markdown 解析和 Chunk。
- 异步 llama.cpp Embedding Client。
- 入库。
- exact Dense 查询。
- HNSW 查询。

### Milestone 2：BM25 和 RRF

- 中文 tokenizer。
- 从 PostgreSQL 重建 BM25 索引。
- BM25 查询。
- 接入 RRF。
- Debug 输出。

### Milestone 3：异步接口与 Reranker

- 异步 llama.cpp Reranker Client。
- 异步 Python Service。
- Typer CLI。
- FastAPI REST。
- 摄取幂等性和并发锁。
- 输出全部阶段排名。

### Milestone 4：语料与评测

- FastAPI 文档下载命令。
- 固定评测 Query。
- Dense/BM25/Hybrid/Reranker 对照。
- 延迟和指标。
- README 演示脚本。

每个 Milestone 完成后应先运行测试和真实 CLI，再进入下一阶段。

## 23. 最终现场演示脚本

```powershell
# 1. 查看环境
py -3.12 -m rag_demo doctor

# 2. 查看数据库和语料状态
py -3.12 -m rag_demo db status

# 3. 对比精确术语
py -3.12 -m rag_demo compare "OAuth2PasswordBearer 有什么作用？"

# 4. 对比语义问题
py -3.12 -m rag_demo compare "怎样让接口接收一个 JSON 对象？"

# 5. 展示完整调试链路
py -3.12 -m rag_demo search `
  "FastAPI 的依赖项在同一次请求里会不会被反复执行？" `
  --debug

# 6. 输出小型评测
py -3.12 -m rag_demo benchmark

# 7. 启动 REST API
py -3.12 -m rag_demo serve --host 127.0.0.1 --port 8000
```

面试讲解顺序：

```text
先展示最终结果
→ 展示 Dense 和 BM25 的差异
→ 展示 RRF 融合
→ 展示 Reranker 调整
→ 展示 HNSW SQL 和 EXPLAIN
→ 展示摄取和 Chunk metadata
→ 最后展示小型评测
```
