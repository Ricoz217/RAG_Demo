# 阶段 4A：FastAPI 语料工具与五路评测

## 阶段目标

提供可重复的 FastAPI 官方文档下载与摄取入口，并用 10 条固定、带相关性标签的 Query
计算五种检索方法的 Recall、MRR 和延迟，避免只凭肉眼评价检索效果。

## 完成内容

### 语料下载模块

新增 `src/rag_demo/corpus.py`：

- 默认仓库为 FastAPI 官方 GitHub 仓库；
- `git clone --depth 1` 浅克隆；
- `--filter=blob:none` 延迟获取无关 blob；
- sparse checkout 只选择中文文档、英文文档和 LICENSE；
- 读取 origin URL；
- 读取当前 commit SHA；
- 统计中英文 Markdown 文件数；
- 已存在的 Git checkout 直接检查并复用；
- 已存在的非 Git 非空目录明确拒绝，不覆盖用户文件；
- Git 子进程通过 `asyncio.to_thread()` 执行，避免阻塞事件循环；
- Git 调用使用参数列表，不经过 Shell 拼接。

`CorpusInfo` 保存路径、仓库 URL、commit 和 Markdown 数量，并能返回 zh/en 语料目录。

### Corpus CLI

新增：

```powershell
python -m rag_demo corpus download-fastapi
python -m rag_demo corpus ingest-fastapi --language zh
```

下载命令显示：

- 本地路径；
- origin；
- commit；
- 中英文 Markdown 数。

摄取命令自动使用：

- Git origin 作为 `source_repo`；
- HEAD SHA 作为 `source_commit`；
- checkout 根目录作为 `source_root`；
- zh/en 作为 language。

如果调用者没有提供 `Idempotency-Key`，CLI 根据 repo、commit、language、Embedding 模型、
维度和 Chunk 参数生成稳定 Key。相同 revision 和配置重复执行会返回首次摄取结果；配置变化
会产生新 Key。

### 固定评测 Query

新增 `data/evaluation_queries.json`，包含 10 条需求书规定的 Query：

- 3 条 exact-term；
- 3 条 semantic-query；
- 4 条 hybrid。

每条保存：

- Query；
- 预期源路径关键字；
- 预期 Heading 关键字；
- 分类。

JSON 属于评测定义，进入 Git；下载语料和索引仍由 `.gitignore` 排除。

### 相关性判定

`EvaluationHit` 只使用最终结果已有的：

```text
source_path
heading_path
```

如果来源路径命中任一期望路径关键字，或 Heading breadcrumb 命中任一期望 Heading 关键字，
则该结果视为相关。比较统一转小写。

这种规则适合可讲解的小型 Demo，但不声称替代人工大规模相关性标注。

### 指标

新增 `src/rag_demo/evaluation.py`，计算：

- Recall@5；
- Recall@10；
- MRR@10；
- 平均查询耗时；
- P95 查询耗时；
- Query 数。

Recall 是“该 Query 的 Top-K 是否出现至少一个相关结果”的二元平均；MRR 使用 Top-10
第一个相关结果的位置。

### 五路评测

`BenchmarkService` 对同一组 Query 顺序执行：

```text
BM25
Dense exact
Dense HNSW
Hybrid RRF
Hybrid RRF + Reranker
```

Dense exact/HNSW 分别调用同一 `DenseQueryService` 的不同模式；Hybrid 使用 HNSW。为计算
Recall@10，评测阶段把 Final Top-K 设为 10，不改变普通搜索的默认 Top-5。

`RAGApplication` 新增只读 `dense_service` 访问点，评测仍复用应用生命周期内的共享
Embedding Client 和数据库池。

### Benchmark CLI

新增：

```powershell
python -m rag_demo benchmark
```

输出五行对照表：

```text
Method
Recall@5
Recall@10
MRR@10
Avg ms
P95 ms
```

默认读取已提交的 `data/evaluation_queries.json`，可通过参数覆盖。

## TDD 过程

先编写：

- 临时本地 Git 仓库浅克隆/稀疏检出测试；
- commit 与中英文 Markdown 计数测试；
- 重复下载复用测试；
- 非 Git 目录拒绝覆盖测试；
- evaluation JSON 解析测试；
- 固定两个 Query 的 Recall/MRR/平均/P95 算例。

模块不存在时测试按预期收集失败；实现后转绿。

指标固定算例：

```text
Query 1: relevant rank = 1, latency = 10 ms
Query 2: relevant rank = 6, latency = 30 ms
```

期望：

```text
Recall@5  = 0.5
Recall@10 = 1.0
MRR@10    = (1 + 1/6) / 2
Avg       = 20 ms
P95       = 29 ms
```

测试结果完全吻合。

## 验证证据

完整自动检查：

```text
83 passed
Total coverage: 84.95%
Ruff: passed
Ruff format: passed
Mypy strict: passed
```

## 面试讲解要点

- 语料 revision 必须保存 Git SHA，否则评测结果无法复现。
- sparse clone 降低下载量，整个 FastAPI 仓库不进入 Demo Git。
- 相关性定义来自可审阅 JSON，不在评测代码中硬编码。
- Recall 回答“有没有召回”，MRR 进一步回答“第一个相关结果排多靠前”。
- exact/HNSW 共用标签集，能够对照 ANN 的召回差异。
- P95 在 10 条 Query 上只用于展示可观测性，不具有生产性能统计意义。
- 五种方法通过现有核心 Service 运行，不维护专用评测检索实现。

## 当前限制与下一阶段

代码和固定标签已经就绪，但尚未对真实 FastAPI checkout 执行下载、摄取、BM25 和 benchmark。

阶段 4B 将：

- 实际浅克隆 FastAPI；
- 核对当前文档路径与评测标签；
- 摄取中文文档并保留数据库与正式 BM25 索引供演示；
- 运行五路 benchmark；
- 记录真实指标、commit 和数据量；
- 完成从零部署 README 与最终验收清单。
