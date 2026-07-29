# 阶段 2：BM25 与 RRF 混合召回

## 阶段目标

在 Dense 之外增加对中文、代码标识符和精确术语敏感的 BM25 分支，并使用 RRF 对两个有序
结果集进行可解释的融合。BM25 只保存可从 PostgreSQL 重建的派生索引，Chunk 正文和 metadata
仍以数据库为事实源。

## 完成内容

### 中文与技术文本分词

新增 `tokenize_bm25()`，版本标识为：

```text
jieba-search-code-v1
```

分词规则：

- 中文片段使用 `jieba.lcut_for_search`；
- 拉丁字母统一转为小写；
- `OAuth2PasswordBearer` 等驼峰技术名称保留为完整 token；
- `request_body` 等下划线标识符保持完整；
- `foo.bar` 等点分标识符保持完整；
- `HTTP` 和 `422` 状态码可被独立召回；
- 空白和标点不进入索引。

索引与 Query 使用同一个分词函数，避免训练时和查询时的 token 规则漂移。

### BM25 generation

新增 `src/rag_demo/bm25_retriever.py`：

- `BM25IndexManager` 从 PostgreSQL 读取当前模型和维度对应的全部 Chunk；
- 使用 `retrieval_text` 建立 `bm25s` Lucene 风格索引；
- `chunk_id` 顺序写入 manifest，与 BM25 文档位置一一对齐；
- `BM25Retriever` 在工作线程执行同步分词和检索；
- 得到候选 ID 后回 PostgreSQL读取正文、Heading 和来源 metadata；
- 只返回正分候选，未知词不会用零分文档填满 Top-K；
- 返回 BM25 rank、原始 score、候选数量和检索耗时。

manifest 保存：

```text
format_version
tokenizer_version
generation
chunk_ids
embedding_model
embedding_dimensions
```

加载时校验格式、分词器版本、模型和维度，防止过期或不兼容索引被静默使用。

### 原子发布

索引目录采用 immutable generation：

```text
data/indexes/bm25/
├── CURRENT
└── generations/
    └── generation-<uuid>/
```

重建流程：

1. 在 `generations/.build-<uuid>` 写入完整新索引；
2. 构建成功后将临时目录重命名为正式 generation；
3. 写入临时指针文件并 `fsync`；
4. 使用 `os.replace` 原子替换 `CURRENT`。

搜索端只根据 `CURRENT` 加载完整且不可变的 generation。如果任意构建步骤失败，旧
`CURRENT` 不变，旧索引仍然可以查询。这种“原子切换小文件指针”的方式也避开了 Windows
不能直接原子覆盖非空目录的问题。

### RRF

新增 `src/rag_demo/rrf.py`。输入为 Dense 和 BM25 两个有序 `chunk_id + score` 列表，
对每个候选计算：

```text
Σ 1 / (rank_constant + rank)
```

输出保留：

- `dense_rank`
- `dense_score`
- `bm25_rank`
- `bm25_score`
- `rrf_score`

某个 Chunk 只在一路出现时，另一路的 rank/score 为 `None`，贡献为 0。同一路重复的
Chunk 只采用第一次出现的位置和分数，不重复累加。

### 并行混合召回

新增 `src/rag_demo/search_service.py` 中的 `HybridRecallService`：

- 使用 `asyncio.TaskGroup` 并行启动 Dense 与 BM25；
- 两路完成后执行轻量 RRF；
- `HybridRecallResponse` 保留 Dense、BM25 和 RRF 三套完整中间结果；
- 记录 RRF 耗时、总耗时和候选并集数量。

这些中间对象将在下一阶段直接驱动 CLI `--debug`、REST Debug 响应和 Reranker，不需要
重新查询或复制排名逻辑。

## TDD 过程

先定义并运行以下行为测试，模块不存在时按预期在收集阶段失败：

- 中文搜索分词；
- 技术标识符、小写化和状态码保留；
- RRF 融合顺序与缺失分支；
- 同一路重复候选去重；
- 非法 rank constant；
- PostgreSQL 三个固定 Chunk 的 BM25 重建、加载和检索；
- 未知词返回空候选；
- 模拟新 generation 构建失败后继续加载旧 generation；
- 缺失索引时给出明确错误；
- Dense/BM25 两路确实并发启动；
- RRF 响应保留各路 Debug 证据。

实现后上述测试全部转绿。

## 验证证据

完整自动检查：

```text
65 passed
Total coverage: 88.76%
Ruff: passed
Ruff format: passed
Mypy strict: passed
```

真实 PostgreSQL 固定语料验证：

```text
OAuth2PasswordBearer → Heading: OAuth2
接口接收 JSON 对象   → Heading: 请求体
BackgroundTasks      → Heading: 后台任务
不存在的词条          → 0 candidate
```

故障注入验证：

```text
首次 generation 构建成功
→ 第二次构建在写入阶段抛出模拟异常
→ CURRENT 仍指向首次 generation
→ 旧索引继续返回 BackgroundTasks 结果
```

测试完成后 PostgreSQL 中的 Document 和 Chunk 均清理为 0；测试 BM25 generation 位于
pytest 临时目录，没有写入正式 `data/indexes/bm25/`。

## 面试讲解要点

- BM25 索引不是第二份事实源；它只保存稀疏检索结构和稳定的 `chunk_id` 映射。
- 搜索命中后回 PostgreSQL hydration，最终来源、正文和 metadata 始终来自数据库。
- jieba 解决中文切分，额外的片段规则保护代码标识符和 HTTP 状态码。
- BM25 原始 score 与 cosine similarity 不在同一数值尺度，不能直接相加。
- RRF 只使用排名位置融合，因此无需校准两路分数。
- RRF score 是融合排序分数，不是概率。
- immutable generation + 原子指针兼顾失败安全和 Windows 目录替换限制。
- `TaskGroup` 让同步 BM25 的线程任务与异步 Embedding/Dense 查询并行运行。

## 当前限制与下一阶段

目前核心层已经能输出 Dense、BM25 和 RRF 的完整排名，但还没有 Cross-Encoder
Reranker，也尚未组装最终 CLI 和 REST 生命周期。

阶段 3 将实现：

- llama.cpp Reranker Client；
- RRF Top-K 到最终 Top-K 的重排；
- 应用对象生命周期；
- Typer 摄取、BM25、搜索和对比命令；
- FastAPI 健康检查、统计、搜索、摄取和 BM25 重建接口；
- CLI/REST 复用相同核心 Service 的一致性测试。
