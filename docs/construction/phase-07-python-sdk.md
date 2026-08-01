# 阶段 7：同步与异步 Python SDK

## 阶段目标

在现有 CLI 和 FastAPI REST 之外提供稳定、对象式的 Python 入口，使普通演示脚本和常驻 Agent
进程不需要导入内部 `application.py`、手工构造 `SearchRequest`，也不需要通过子进程调用 CLI。

## 用户旅程

1. 普通 Python 脚本使用 `with RAG() as rag`，连续执行多次检索后自动释放资源。
2. FastAPI、异步 Agent 或 Notebook 使用 `async with AsyncRAG() as rag`，复用调用方事件循环。
3. 调用方直接获得现有 `SearchResponse`、`QuerySearchResponse`、`IngestResult` 等强类型对象。
4. Python API 与 CLI、REST 委托同一个 `RAGApplication`，不复制解析、检索和排序逻辑。

## 设计决策

### 双门面、单核心

`AsyncRAG` 是异步主门面，直接管理 `RAGApplication`。`RAG` 是脚本友好的同步适配器，内部持有
Python 3.12 `asyncio.Runner` 和一个 `AsyncRAG`。同一同步引擎的启动、查询和关闭始终运行在同一个
私有事件循环，避免为每次调用执行 `asyncio.run()` 后让 Psycopg 连接池或 `httpx.AsyncClient`
跨事件循环复用。

同步门面不允许在已有事件循环中运行；这种误用会给出明确异常并引导调用方改用 `AsyncRAG`。

### 显式生命周期

导入 `rag_demo` 或构造 SDK 对象不会连接数据库，也不会调用模型服务。推荐使用同步或异步上下文
管理器；也可以显式 `open()`/`close()`。未启动时调用业务方法会抛出 `RAGNotStartedError`，已经永久
关闭的同步引擎不能重新打开。

### 配置与返回类型

省略检索参数时使用 `.env` 经 `Settings` 校验后的默认值，而不是重新维护一套 SDK 默认值。
`dense_mode` 同时接受 `DenseSearchMode` 和 `"exact"`/`"hnsw"` 字符串。SDK 返回核心层已有的
不可变 dataclass，没有增加 JSON 风格的第二套响应模型。

### 公开包边界

包根目录导出 SDK 门面、设置、检索模式、主要响应类型和生命周期异常，并通过 `py.typed` 声明
PEP 561 类型信息。内部服务仍可用于测试和研究，但不属于首选 SDK 入口。

## 对外能力

- `search()`：完整 Dense + BM25 + RRF + 可选 Reranker 检索；
- `search_query()`：可选 Query Rewrite 并保留改写证据；
- `compare_rewrite()`：执行关闭/开启 Rewrite 的中性对照；
- `ingest()`：摄取一个 Markdown 文件或目录；
- `rebuild_bm25()`：从 PostgreSQL 重建并加载 BM25；
- `reload_bm25()`：长驻进程加载其他进程发布的 CURRENT；
- `doctor()`：执行基础设施检查。

## TDD 过程

先增加 SDK 契约测试，并确认测试收集阶段因为包根目录不存在 `AsyncRAG` 导出而失败。随后实现最小
门面使测试转绿。单元测试使用 Fake Application 验证可观察行为，不测试内部字段：

- `.env`/`Settings` 默认检索参数是否传递；
- 字符串 Dense 模式是否转换为枚举；
- 同步与异步上下文是否打开、关闭资源；
- 同步多次调用是否复用同一个事件循环；
- Rewrite、对照、入库、BM25 和 Doctor 是否委托核心应用；
- 未启动调用和异步上下文误用是否给出明确错误。

另增加真实基础设施集成测试，通过 `AsyncRAG` 执行一次 Hybrid Search，验证公开入口可以贯通
PostgreSQL、BM25 和本地 Embedding 服务。

第一次直接运行同步示例时，Fake 测试和 pytest 内部测试均已通过，但独立 Windows 进程实际使用了
默认 `ProactorEventLoop`。Psycopg 拒绝该事件循环，连接池等待 30 秒后抛出 `PoolTimeout`。随后增加
独立子进程回归测试，确认其稳定复现，再将 CLI 已验证的 Selector loop factory 提升为
`create_compatible_event_loop()`，同时供 `run_async()` 和同步 SDK 的长期 `asyncio.Runner` 使用。修复后
同一子进程用例约 2 秒完成，避免 pytest 的全局 Selector 策略掩盖真实脚本错误。

## 修改模块

- `src/rag_demo/sdk.py`
- `src/rag_demo/__init__.py`
- `src/rag_demo/py.typed` 的打包声明
- `src/rag_demo/asyncio_compat.py`
- `examples/python_sdk_demo.py`
- `README.md`
- `pyproject.toml`
- SDK 单元测试与真实集成测试

## 验证证据

- 全套单元与真实集成测试：`117 passed`；
- 总覆盖率：`85.50%`，通过项目 `80%` 门槛；
- SDK 定向测试：`9 passed`；
- 独立 Windows 同步 SDK 子进程：通过；
- `examples/python_sdk_demo.py`：真实 Dense、BM25、RRF、Reranker 链路通过并返回 5 条结果；
- Mypy strict：`51 source files` 无错误；
- 本阶段修改文件的 Ruff lint 与 format：通过；
- Wheel 与 source distribution：构建成功，Wheel 包含 `rag_demo/py.typed`；
- 全仓 Ruff 仍被代码审阅期间留在 `bm25_retriever.py`、`chunker.py`、`dense_retriever.py`、
  `search_service.py` 的 7 条超长中文注释阻塞；全仓 format 还会改动若干既有审阅文件。本阶段没有
  擅自重写这些用户审阅内容。

## 讲解要点

- SDK、CLI 和 REST 是三个适配入口，核心应用服务只有一套。
- 单例应该是“一个应用进程一个有生命周期的引擎”，不是 import 时创建的隐藏全局对象。
- 同步异步资源不能通过反复 `asyncio.run()` 随意跨事件循环复用。
- `RAG` 适合普通脚本；已有事件循环和高并发服务应使用 `AsyncRAG`。
