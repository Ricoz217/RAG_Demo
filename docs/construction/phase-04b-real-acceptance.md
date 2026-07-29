# 阶段 4B：真实语料验收、README 与最终演示

## 阶段目标

在用户当前真实 PostgreSQL、llama.cpp Embedding 和 Reranker 服务上完成 FastAPI 中文文档
下载、摄取、索引、搜索、五路评测和全部验收，并把可复现命令与实测证据整理进 README。

## 真实语料

通过项目 CLI 执行 sparse shallow clone：

```text
Repository: https://github.com/fastapi/fastapi.git
Commit: 628663f4f899c465da423bce681c7adf9a218948
Chinese Markdown: 124
English Markdown: 155
```

真实下载首次暴露了一个 Windows Git cone-mode 约束：

```text
fatal: 'LICENSE' is not a directory
```

初版 sparse path 同时包含两个目录和单个 LICENSE 文件，cone mode 默认拒绝文件路径。
修复为只 sparse checkout 中英文文档目录。下载器同时增加半完成 checkout 恢复逻辑：

- 已有 `.git`；
- 目标 sparse path 尚未出现；
- worktree 无本地修改；
- 重新执行 `sparse-checkout set`；
- 不删除、不重新克隆。

已有本地修改时明确拒绝改变 sparse path。

## 真实摄取

执行：

```powershell
python -m rag_demo corpus ingest-fastapi --language zh
```

结果：

```text
Documents: 124
Chunks: 1145
Embedded: 1145
Embedding skipped: 0
Embedding HTTP requests: 126
Total: 14771.00 ms
```

Embedding 请求数不是简单的 `ceil(1145 / batch_size)`，因为当前事务边界按 Document，
每个 Document 独立批处理。这样单文档失败不会留下半份 Chunk。

## 双层重复摄取验收

第一层：使用自动生成的相同 Idempotency Key 重复命令，约 1 秒返回首次持久化结果，
数据库行数不变。

第二层：使用一个新的 Key 摄取相同 repo/commit/config，强制进入 Chunk 增量判定：

```text
Documents: 124
Chunks: 1145
Embedded: 0
Embedding skipped: 1145
Embedding HTTP requests: 0
Total: 374.34 ms
```

这分别证明：

- 请求级幂等；
- Chunk 级增量 Embedding 跳过；
- Document/Chunk 唯一约束没有产生重复行。

## BM25 与 Doctor

正式 BM25：

```text
Generation: generation-acf162c79c1e437989fc64d0b7d54cf0
Chunks: 1145
Build: 663.53 ms
```

Doctor 全部 PASS：

```text
Python 3.12.9
PostgreSQL 18.4
pgvector 0.8.5
chunks=1145
bge-m3 / 1024
bge-reranker-v2-m3
BM25 chunks=1145
Corpus vector space=bge-m3/1024
```

## 五路评测

执行：

```powershell
python -m rag_demo benchmark --debug
```

一次真实运行：

| Method | Recall@5 | Recall@10 | MRR@10 | Avg ms | P95 ms |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.900 | 0.900 | 0.717 | 36.32 | 194.18 |
| Dense exact | 0.800 | 0.900 | 0.767 | 23.35 | 32.67 |
| Dense HNSW | 0.800 | 0.900 | 0.767 | 13.59 | 18.15 |
| Hybrid RRF | 0.900 | 0.900 | 0.775 | 13.23 | 15.74 |
| Hybrid RRF + Reranker | 0.900 | 1.000 | 0.867 | 215.96 | 278.57 |

小型评测只用于 Demo 可观察性，不用于声称算法优劣的普遍结论。

### Dense 找到 BM25 漏掉

Query：

```text
怎样让接口接收一个 JSON 对象？
```

首个相关结果：

```text
BM25 Top-10: missing
Dense exact: rank 6
Dense HNSW: rank 6
Hybrid RRF Top-10: missing
Hybrid + Reranker: rank 1
```

### BM25 精确术语优势

Query：

```text
如何让多个接口共享同一段参数检查逻辑？
```

首个相关结果：

```text
BM25: rank 1
Dense exact Top-10: missing
Dense HNSW Top-10: missing
Hybrid RRF: rank 4
Hybrid + Reranker: rank 6
```

### HNSW 与 exact

本次 1145 Chunk 数据上：

- Recall@5、Recall@10 和 MRR@10 完全相同；
- exact 平均 23.35 ms；
- HNSW 平均 13.59 ms；
- HNSW 平均耗时约低 41.8%。

数据规模小，主要用于证明执行模式、计划和指标可对照。

### Reranker 调整

语义 JSON Query 中，RRF Top-10 没有相关 Chunk，但 RRF Top-20 候选经 Cross-Encoder
重排后，正确的请求体 Chunk 成为最终 rank 1。

整体 Recall@10：

```text
Hybrid RRF: 0.9
Hybrid RRF + Reranker: 1.0
```

MRR@10：

```text
Hybrid RRF: 0.775
Hybrid RRF + Reranker: 0.867
```

## README

README 已扩展为从零部署手册，包含：

- Python 3.12 与 uv；
- PostgreSQL/pgvector；
- llama.cpp 两个服务契约；
- `.env` 安全；
- 下载、摄取、BM25、Doctor；
- search/debug/exact/no-rerank/compare/benchmark；
- FastAPI REST 与 Idempotency-Key；
- 真实指标；
- 项目结构；
- 常见问题；
- 完整工程检查命令。

示例中没有真实密码和 API Key。

## 最终验收映射

- Python 3.12 独立 `.venv`：完成。
- Doctor 调用两个真实 llama.cpp 服务：完成。
- pgvector `vector(1024)`：完成。
- HNSW 建立并由 EXPLAIN 命中：完成。
- FastAPI 中文文档一条命令摄取：完成。
- 重复/并发摄取不重复：完成。
- BM25 从 PostgreSQL 原子重建：完成。
- Dense/BM25/RRF/Reranker 中间结果：完成。
- Compare 四组结果：完成。
- 五路 Recall/MRR/延迟：完成。
- 最终结果来源、Heading、预览：完成。
- Dense 对语义 Query 的优势案例：完成。
- BM25 对精确/词法匹配的优势案例：完成。
- Reranker 调整顺序案例：完成。
- 可直接 await 的摄取和搜索 API：完成。
- FastAPI REST：完成。
- CLI/REST 共用核心 Service 且顺序一致：完成。
- 数据库幂等、advisory lock 和文件 generation：完成。

## 最终工程检查

在保留 124 个正式 FastAPI Document 的数据库状态下重新运行全部测试。REST 测试使用
“全局基线 + 测试自身数据”断言，因此既支持空库 CI，也支持本机保留演示语料：

```text
83 passed
Total coverage: 84.12%
Ruff: passed
Ruff format: passed
Mypy strict: passed
```

## 保留的本机演示状态

最终没有清理真实 FastAPI 数据，以便用户可以直接演示：

```text
PostgreSQL Documents: 124
PostgreSQL Chunks: 1145
data/corpus/fastapi: present, Git-ignored
data/indexes/bm25/CURRENT: present, Git-ignored
```

这些运行数据不进入项目 Git。
