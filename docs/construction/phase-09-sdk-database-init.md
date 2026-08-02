# 阶段 9：Python SDK 数据库初始化接口

## 阶段目标

让只使用 Python SDK 的调用方无需切换到 CLI，也能对新数据库显式执行现有 migration。CLI、SDK继续共用 `migrations.py::apply_migrations()`，不复制建表SQL，也不把自动迁移塞进应用启动过程。

## 用户旅程

同步脚本：

```python
from rag_demo import RAG

rag = RAG()
applied = rag.init_database()
with rag:
    response = rag.search("query")
```

异步应用：

```python
from rag_demo import AsyncRAG

rag = AsyncRAG()
applied = await rag.init_database()
async with rag:
    response = await rag.search("query")
```

## 设计决策

### 显式迁移，不自动迁移

`open()`仍只打开数据库连接池和HTTP Client。`init_database()`必须由调用方显式执行，避免普通服务启动隐式修改Schema。

### 管理员基础设施与应用迁移分层

`init_database()`沿用 `DATABASE_URL` 中的应用账号，不提升数据库权限。pgvector在当前安装配置下属于
非trusted扩展，因此新数据库需要先由PostgreSQL管理员执行 `CREATE EXTENSION vector;`。完成该前置步骤后，
应用账号会幂等经过 `001_enable_vector.sql`，再执行建表与索引migration。

这里区分两层职责：管理员负责数据库级扩展，SDK/CLI负责应用Schema。`IF NOT EXISTS`只处理扩展已经存在的
情况，不会让数据库OWNER获得安装非trusted扩展的权限。

### 初始化不要求应用已打开

Migration通过自己的短连接完成，不依赖 `RAGApplication` 的数据库池、Embedding Client或Reranker Client。因此 `AsyncRAG.init_database()`可以在 `open()`之前调用；同步门面通过已有的私有Selector事件循环运行同一异步实现。

### 返回底层可观察结果

返回值沿用 `apply_migrations()` 的 `tuple[str, ...]`：新数据库返回本次应用的migration名称，Schema已是最新时返回空元组。Migration异常不包装，保留原始数据库错误和 `MigrationError` 诊断信息。

### 生命周期约束不放宽

同步 `RAG.init_database()`仍拒绝在活动异步事件循环中调用，且永久关闭的同步对象不能重新执行初始化。它不触发 `RAGApplication.open()`，也不会启动模型客户端。

## TDD过程

先增加同步与异步契约测试，并确认它们因 `rag_demo.sdk.apply_migrations` 和 `init_database()`尚不存在而失败；随后实现最小委托使测试转绿。测试验证：

- Async SDK在未打开应用时使用 `.env`中的数据库URL执行migration；
- 返回值原样传递给调用方；
- 初始化不会打开 `RAGApplication`；
- Sync SDK复用其长期Windows兼容事件循环；
- 真实PostgreSQL连续初始化两次时，第二次返回空元组。
- 集成测试从当前DSN读取库名和用户名，不再绑定某个演示数据库；SDK搜索验证也不依赖数据库预先残留语料。

## 修改模块

- `src/rag_demo/sdk.py`
- `tests/unit/test_sdk.py`
- `tests/integration/test_python_sdk.py`
- `tests/integration/test_database.py`
- `README.md`

## 验证结果

- 在新库 `rag_demo_02` 预装pgvector后，真实SDK初始化测试通过，连续调用第二次返回空元组；
- 全量测试：`125 passed`；
- 分支覆盖率：`85.46%`，高于项目要求的80%；
- 全项目mypy严格类型检查通过；
- 本阶段修改文件的ruff检查通过。

## 讲解要点

- “提供初始化接口”和“启动时自动初始化”是两件事；本阶段只补前者。
- CLI与SDK是不同适配入口，真正的migration实现仍只有一套。
- 数据库Migration不需要持有RAG运行时资源，因此适合放在引擎生命周期之前。
