# 阶段 3B：应用生命周期与 Typer CLI

## 阶段目标

把已经完成的数据库、模型 Client、摄取、Dense、BM25、RRF 和 Reranker 组装为一套可复用
应用资源，并通过 Typer CLI 提供从环境诊断到四路结果对比的真实演示入口。

## 完成内容

### 应用资源容器

新增 `src/rag_demo/application.py` 中的 `RAGApplication`。它统一创建和关闭：

```text
PostgreSQL AsyncConnectionPool
httpx.AsyncClient
EmbeddingClient
RerankerClient
MarkdownChunker
DenseRetriever
DenseQueryService
BM25IndexManager / BM25Retriever
HybridRecallService
HybridSearchService
```

生命周期规则：

- `open()` 只创建一套 Database Pool 和 HTTP Client；
- Embedding 与 Reranker 共用该 HTTP Client；
- BM25 在首次搜索时惰性加载；
- BM25 重建成功后立即加载新 generation 并刷新 Search Service；
- `close()` 先丢弃派生 Service，再关闭 HTTP Client 和数据库池；
- 支持 `async with RAGApplication(settings)`；
- 生命周期外访问资源会抛出明确的 `ApplicationNotStartedError`。

Document 摄取的 repo、commit、root 和语言属于单次来源参数，因此通过
`create_ingestor()` 创建轻量来源对象，共享底层 Pool、HTTP Client、Embedding Client 和
Chunker。

### BM25 路径配置

配置新增：

```text
BM25_INDEX_PATH=data/indexes/bm25
```

它有安全默认值，也可以由测试、CLI 现场环境或未来 REST 部署覆盖。`.env.example` 已同步，
真实 `.env` 仍未进入 Git。

### Doctor

新增：

```powershell
python -m rag_demo doctor
```

实际执行：

- Python 必须为 3.12；
- PostgreSQL、pgvector、`vector(1024)` 和 HNSW；
- 使用测试文本真实调用 Embedding API；
- 校验返回维度；
- 使用 Query 与两个 Document 真实调用 Reranker API；
- 加载 BM25 `CURRENT`；
- 显示数据库 Chunk 数量；
- 校验数据库中所有 Chunk 的模型名和维度与当前配置完全一致。

输出每项 `PASS/FAIL` 和不含密钥的细节；任何一项失败时 CLI 返回非零退出码。

### Ingest CLI

新增：

```powershell
python -m rag_demo ingest <source> \
  --source-repo <repo> \
  --source-commit <sha> \
  --source-root <root> \
  --language zh \
  --idempotency-key <stable-key>
```

命令只负责构造来源特定的 `DocumentIngestor`，核心解析、Embedding、事务和幂等逻辑没有
复制到 CLI。输出 Document、Chunk、Embedding、跳过、删除、HTTP 请求和总耗时。

### BM25 CLI

新增：

```powershell
python -m rag_demo bm25 rebuild
```

输出新 generation、Chunk 数和构建耗时。命令返回前已完成 `CURRENT` 原子切换并加载验证。

### Search CLI

新增：

```powershell
python -m rag_demo search "query"
python -m rag_demo search "query" --debug
python -m rag_demo search "query" --no-rerank
python -m rag_demo search "query" --dense-mode exact
```

支持覆盖：

- Dense Top-K；
- BM25 Top-K；
- Rerank Top-K；
- Final Top-K；
- exact/HNSW；
- Reranker 开关；
- Debug 输出。

未覆盖的数值来自 `.env` Settings。所有命令构造同一个核心 `SearchRequest`。

普通输出：

- Query；
- Final results；
- 标题与 Heading；
- 源路径；
- 最终分数；
- 正文预览；
- 各阶段耗时。

Debug 额外输出：

- Dense Top-K；
- BM25 Top-K；
- RRF Top-K；
- Reranker Top-K。

### Compare CLI

新增：

```powershell
python -m rag_demo compare "query"
```

一次执行完整 Search Service 后，从同一个 `SearchResponse` 展示：

```text
BM25 only
Dense only
Hybrid RRF
Hybrid RRF + Reranker
```

它不会为了四张表重复运行四次模型和数据库查询，因此四组结果具有同一 Query 和同一时刻的
候选基础。

## TDD 过程

先编写应用生命周期和真实 CLI 工作流测试：

- 未启动时禁止读取数据库资源；
- CLI 对一份真实 Markdown 执行摄取；
- 发布 BM25 generation；
- `search --no-rerank --debug` 输出全部召回阶段；
- `compare` 输出四组固定标签；
- `doctor` 检查 Embedding、Reranker 和 BM25 并全部 PASS；
- 最终清理 Document、Chunk 和 Idempotency 记录。

应用模块不存在时测试按预期在收集阶段失败；完成资源容器和 CLI 后转绿。

## 真实 CLI 验证

测试语料：

```text
FastAPI 教程
├── 请求体：Pydantic + JSON
└── 安全：OAuth2PasswordBearer + Bearer Token
```

现场自动执行：

```text
ingest
→ bm25 rebuild
→ search "FastAPI 如何接收 JSON 请求体？" --no-rerank --debug
→ compare "OAuth2PasswordBearer 有什么作用？"
→ doctor
```

验证结果：

- 1 个 Markdown Document；
- 2 个结构化 Chunk；
- Dense、BM25、RRF 和 Final 表均包含 `demo.md` 来源；
- Compare 四组输出齐全；
- Doctor 的 PostgreSQL、Embedding、Reranker、BM25 和向量空间全部 PASS；
- CLI 返回码均为 0。

测试 BM25 使用 pytest 临时路径，没有覆盖正式 `data/indexes/bm25/`。测试结束后数据库
Document/Chunk 为 0。

## 验证证据

完整自动检查：

```text
78 passed
Total coverage: 89.10%
Ruff: passed
Ruff format: passed
Mypy strict: passed
```

## 面试讲解要点

- CLI 只是适配层，摄取和搜索业务都在可直接 `await` 的核心对象中。
- Database Pool 与 HTTP Client 在一个应用生命周期中只创建一次。
- 搜索惰性加载 BM25，摄取和数据库命令不会因为 BM25 尚未构建而失败。
- BM25 rebuild 后刷新同一应用中的 Search Service，后续请求使用新 generation。
- Doctor 不是只做端口探测，而是对两个模型各执行一次真实推理请求。
- Compare 从一次完整响应派生四组视图，避免重复查询造成不可比。
- Rich 只负责可读输出，不参与排名逻辑。

## 当前限制与下一阶段

CLI 已能完整演示核心检索链路。阶段 3C 将用同一个 `RAGApplication` 和核心 Service 提供：

- FastAPI lifespan；
- live/ready 健康检查；
- stats；
- search；
- ingest；
- BM25 rebuild；
- HTTP 幂等错误映射；
- CLI 与 REST 最终 Chunk 顺序一致性测试。
