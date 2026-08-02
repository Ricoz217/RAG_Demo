# Minimal Hybrid RAG Demo

一个可在 Windows 本机完整运行、可观察每个排名阶段的最小 Hybrid RAG 检索 Demo。

本项目只实现检索链路，不接 Agent，不调用生成式 LLM，也不生成自然语言答案：

```text
FastAPI Markdown
→ 结构化 Chunk
→ llama.cpp bge-m3 Embedding
→ PostgreSQL + pgvector
   ├─ Dense exact / HNSW
   └─ bm25s + jieba
→ RRF
→ llama.cpp bge-reranker-v2-m3
→ 带来源与全阶段分数的 Final Top-K
```

CLI、FastAPI REST 和可直接 `await` 的 Python Service 使用同一套核心对象。

## 已验证环境

最后验收日期：2026-07-29。

| 组件 | 已验证版本/配置 |
|---|---|
| Windows | Windows 11 x64 |
| Python | 3.12.9 |
| PostgreSQL | 18.4 |
| pgvector | 0.8.5 |
| 向量列 | `vector(1024)` |
| ANN | HNSW + cosine |
| Embedding | llama.cpp / bge-m3 |
| Reranker | llama.cpp / bge-reranker-v2-m3 |
| GPU | NVIDIA GeForce RTX 4090 |

依赖的精确版本保存在 `uv.lock`。

## 1. 创建项目环境

必须使用独立 Python 3.12 环境，不要复用系统中其他项目的 Python：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install uv
.\.venv\Scripts\uv.exe sync --all-groups
Copy-Item .env.example .env
```

真实 `.env` 已被 Git 忽略。不要把数据库密码或模型 API Key 写入代码、README 或 commit。

## 2. 准备 PostgreSQL + pgvector

以下 SQL 中的密码只是占位符，请替换：

```sql
CREATE ROLE rag_app WITH LOGIN PASSWORD 'replace-with-a-strong-password';
CREATE DATABASE rag_demo OWNER rag_app ENCODING 'UTF8';
\c rag_demo
CREATE EXTENSION vector;
```

在 `.env` 设置：

```dotenv
DATABASE_URL=postgresql://rag_app:replace-with-your-database-password@127.0.0.1:5432/rag_demo
```

应用账号不需要超级用户权限；`vector` 扩展由管理员预先创建。

初始化并检查 Schema：

```powershell
.\.venv\Scripts\python.exe -m rag_demo db init
.\.venv\Scripts\python.exe -m rag_demo db status
```

Migration 使用 SHA-256 checksum、数据库 advisory lock 和事务，可安全重复执行。已经应用的
SQL migration 不应修改。

## 3. 启动 llama.cpp 模型服务

模型服务由项目外的 llama.cpp 独立运行。本应用不安装 PyTorch、Transformers 或模型权重。

Embedding 服务需要兼容：

```text
POST http://127.0.0.1:8081/v1/embeddings
model: bge-m3
dimensions: 1024
```

Reranker 服务需要兼容：

```text
POST http://127.0.0.1:8082/reranking
model: bge-reranker-v2-m3
```

在 `.env` 设置实际 URL、Key、模型名、超时和批量大小。示例见 `.env.example`。

`EMBEDDING_MODEL` 或维度变化后必须重新生成所有 Chunk 向量；同维度的不同模型也不能混用。
非 1024 维模型还需要重建向量列和 HNSW 索引。

## 4. 下载、摄取并建立 BM25

浅克隆 FastAPI 官方中英文文档：

```powershell
.\.venv\Scripts\python.exe -m rag_demo corpus download-fastapi
```

摄取中文文档：

```powershell
.\.venv\Scripts\python.exe -m rag_demo corpus ingest-fastapi --language zh
```

命令自动记录 Git origin、commit SHA 和仓库相对路径，并为当前 revision/配置生成稳定
Idempotency Key。

建立 BM25 派生索引：

```powershell
.\.venv\Scripts\python.exe -m rag_demo bm25 rebuild
```

BM25 索引位于 `data/indexes/bm25/`，使用 immutable generation 和原子 `CURRENT` 指针。
它可以随时从 PostgreSQL Chunk 全量重建，不是第二份事实源。

最后运行：

```powershell
.\.venv\Scripts\python.exe -m rag_demo doctor
```

Doctor 会真实调用两个模型，而不只是检查端口。

## 5. 搜索与对照演示

普通搜索：

```powershell
.\.venv\Scripts\python.exe -m rag_demo search `
  "FastAPI 的依赖项在同一次请求里会不会被反复执行？"
```

显示 Dense、BM25、RRF、Reranker 全部中间排名：

```powershell
.\.venv\Scripts\python.exe -m rag_demo search `
  "FastAPI 的依赖项在同一次请求里会不会被反复执行？" `
  --debug
```

关闭 Reranker：

```powershell
.\.venv\Scripts\python.exe -m rag_demo search `
  "怎样让接口接收一个 JSON 对象？" `
  --no-rerank
```

最终结果携带 `confidence`。默认使用 Top-1 Reranker 原始分数和
`RERANKER_LOW_CONFIDENCE_THRESHOLD=-4.0` 做提醒：低于告警线时保留全部 Top-K，但 CLI 会提示
结果可能不可信；关闭 Reranker 时状态为 `unassessed`。这个分数不是概率，告警线只适用于当前
`bge-reranker-v2-m3` 与已验收语料，替换模型或语料后必须重新校准。

切换为 exact 向量扫描：

```powershell
.\.venv\Scripts\python.exe -m rag_demo search `
  "怎样让接口接收一个 JSON 对象？" `
  --dense-mode exact
```

一次展示四组结果：

```powershell
.\.venv\Scripts\python.exe -m rag_demo compare `
  "OAuth2PasswordBearer 有什么作用？"
```

Query Rewrite 默认关闭。需要时可启用确定性的 NFKC、空白规范化和别名扩展：

```powershell
.\.venv\Scripts\python.exe -m rag_demo search `
  "如何声明请求体？" `
  --rewrite `
  --debug
```

别名字典保存在 `data/query_aliases.json`，扩展时同时保留规范词和原始别名，例如
`请求体` 会成为 `Request Body (请求体)`。它不调用 LLM，也不保证检索结果更好。

真实执行 Rewrite 关闭、开启两次检索，并中性展示 Final Top-K 的成员与排名差异：

```powershell
.\.venv\Scripts\python.exe -m rag_demo compare-rewrite `
  "如何声明请求体？"
```

结果相同、变化或丢失原有候选都属于有效实验结论。该命令显示的是一次顺序执行的原始耗时，受冷启动与
缓存影响，不应直接当作严谨的性能基准。

输出小型五路评测：

```powershell
.\.venv\Scripts\python.exe -m rag_demo benchmark
.\.venv\Scripts\python.exe -m rag_demo benchmark --debug
```

## 6. Python SDK

普通 Python 脚本使用同步门面；一个 `with` 块内的所有调用共享同一个数据库连接池、HTTP Client、
BM25 generation 和私有事件循环：

```python
from rag_demo import RAG

rag = RAG()
applied = rag.init_database()  # 新数据库执行 migration；重复调用返回空元组

with rag:
    response = rag.search(
        "FastAPI 如何接收 JSON 请求体？",
        final_top_k=5,
    )

    for candidate in response.results:
        print(candidate.final_rank, candidate.source_path)
        print(candidate.content_raw)

    print(response.confidence.status, response.confidence.warning)
```

可直接运行完整示例：

```powershell
.\.venv\Scripts\python.exe examples\python_sdk_demo.py
```

FastAPI、异步 Agent 和已有事件循环使用 `AsyncRAG`：

```python
from rag_demo import AsyncRAG

rag = AsyncRAG()
applied = await rag.init_database()

async with rag:
    response = await rag.search("FastAPI 的依赖缓存如何工作？")
```

`init_database()` 使用 `DATABASE_URL` 中的应用账号执行 migration。目标数据库必须先由
PostgreSQL 管理员执行 `CREATE EXTENSION vector;`；普通数据库 OWNER 不能安装当前默认配置下的
pgvector 扩展，但可以继续创建本项目的表和索引。

`search()` 返回现有强类型 `SearchResponse`。同步门面需要观察 Query Rewrite 时使用：

```python
with RAG() as rag:
    execution = rag.search_query("如何声明请求体？", rewrite=True)
    print(execution.rewrite.effective_query)
    print(execution.response.results)
```

SDK 还提供 `init_database()`、`ingest()`、`rebuild_bm25()`、`reload_bm25()`、`doctor()` 和
`compare_rewrite()`。同步 `RAG` 不能在已运行的事件循环中使用，此时应选择 `AsyncRAG`。不要为每次
查询创建一个引擎；常驻进程应在启动时创建一次，在关闭时释放。

## 7. FastAPI REST

启动：

```powershell
.\.venv\Scripts\python.exe -m rag_demo serve `
  --host 127.0.0.1 `
  --port 8000
```

端点：

```text
GET  /health/live
GET  /health/ready
GET  /v1/stats
POST /v1/search
POST /v1/ingest
POST /v1/bm25/rebuild
```

PowerShell 搜索示例：

```powershell
$body = @{
  query = "怎样让接口接收一个 JSON 对象？"
  dense_top_k = 30
  bm25_top_k = 30
  rerank_top_k = 20
  final_top_k = 5
  rewrite = $false
  debug = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/v1/search" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

`POST /v1/search` 的 `rewrite` 默认是 `false`。响应同时返回 `query`、`effective_query`、
`rewrite_enabled`、`rewrite` 和 `confidence` 详情，便于调用方保存实验条件，并在低可信时向最终用户
展示警告，而不是把相对排名误当成可靠答案。

摄取和 BM25 rebuild 必须携带：

```http
Idempotency-Key: caller-generated-stable-key
```

相同 Key + 相同请求返回首次结果；冲突或正在处理返回 HTTP 409。幂等状态保存在
PostgreSQL，不依赖进程内缓存。

## 8. 真实验收结果

FastAPI corpus：

```text
commit: 628663f4f899c465da423bce681c7adf9a218948
Chinese Markdown: 124
Documents: 124
Chunks: 1145
Embedding HTTP requests: 126
Initial ingestion: 14.77 s
BM25 build: 663.53 ms
```

相同语料使用新的 Idempotency Key 再摄取：

```text
Documents: 124
Chunks: 1145
Embedded: 0
Embedding skipped: 1145
Embedding HTTP requests: 0
```

10 条固定 Query 的一次实测：

| Method | Recall@5 | Recall@10 | MRR@10 | Avg ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.900 | 0.900 | 0.717 | 36.32 | 194.18 |
| Dense exact | 0.800 | 0.900 | 0.767 | 23.35 | 32.67 |
| Dense HNSW | 0.800 | 0.900 | 0.767 | 13.59 | 18.15 |
| Hybrid RRF | 0.900 | 0.900 | 0.775 | 13.23 | 15.74 |
| Hybrid RRF + Reranker | 0.900 | 1.000 | 0.867 | 215.96 | 278.57 |

数据集很小，数值不具有学术结论意义。它们用于证明系统能一致地计算、展示并比较检索质量和
延迟。

验收中观察到：

- “怎样让接口接收一个 JSON 对象？”：BM25 Top-10 漏掉，Dense 排第 6，Reranker 提到第 1。
- “如何让多个接口共享同一段参数检查逻辑？”：BM25 排第 1，Dense Top-10 漏掉。
- HNSW 与 exact 在该小数据集上的 Recall/MRR 相同，平均 Dense 阶段耗时更低。
- Reranker 将 Recall@10 从 0.9 提升到 1.0，并改变最终候选顺序。

## 9. 工程检查

```powershell
.\.venv\Scripts\pytest.exe
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\ruff.exe format src tests --check
.\.venv\Scripts\mypy.exe src tests
```

最终施工阶段运行结果：

```text
117 tests passed
coverage: 85.50%
SDK 相关文件 Ruff passed
Mypy strict passed
```

测试包含真实 PostgreSQL、pgvector、真实本地模型 smoke test、Mock HTTP 契约测试、HNSW
`EXPLAIN (ANALYZE, BUFFERS)`、摄取并发、REST 幂等和 CLI/REST 顺序一致性。

## 10. 项目结构

```text
migrations/                 PostgreSQL + pgvector migration
src/rag_demo/
  models/                   按领域集中管理公共数据契约
    document.py             Markdown、Block 与 Chunk
    retrieval.py            Dense、BM25、RRF 与最终搜索结果
    ingestion.py            摄取结果
    rewrite.py              Query Rewrite 与效果对比结果
    operations.py           数据库、Doctor 与语料操作结果
    evaluation.py           Benchmark 查询、命中与指标
  application.py            共享资源生命周期与对象组装
  markdown_parser.py        Markdown 结构解析
  chunker.py                Heading-aware Chunk
  embedding_client.py       llama.cpp Embedding Client
  ingest_service.py         事务、增量与幂等摄取
  dense_retriever.py        exact/HNSW pgvector 检索
  bm25_retriever.py         jieba + bm25s generation
  rrf.py                    Reciprocal Rank Fusion
  reranker_client.py        llama.cpp Reranker Client
  query_rewriter.py         确定性 Query Rewrite 与中性 A/B 差异
  search_service.py         并行召回与最终排名
  sdk.py                    同步/异步 Python SDK 门面
  api.py                    FastAPI REST
  cli.py                    Typer CLI
  corpus.py                 FastAPI sparse clone
  evaluation.py             五路质量与延迟评测
tests/                      unit + integration
examples/python_sdk_demo.py 可直接运行的同步 SDK 演示
data/evaluation_queries.json
data/query_aliases.json
docs/construction/          分阶段施工与验证记录
```

## 11. 常见问题

`doctor` 显示 BM25 未就绪：

```powershell
.\.venv\Scripts\python.exe -m rag_demo bm25 rebuild
```

`ready` 返回 503：

- 确认数据库有 Chunk；
- 确认 HNSW 存在；
- 确认 BM25 `CURRENT` 已建立。

Windows 下 Psycopg 异步报 Proactor 错误：

- 从项目 CLI 启动；项目入口会使用 Windows Selector event loop。

切换模型后搜索不到旧语料：

- 重新摄取全部语料；
- 重建 BM25；
- 用 `doctor` 确认 corpus vector space 与配置一致。

## 施工记录

每个阶段的目标、取舍、失败案例、测试证据和面试讲解要点都保存在
[`docs/construction/`](docs/construction/README.md)。
