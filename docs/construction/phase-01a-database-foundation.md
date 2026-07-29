# 阶段 1A：数据库基础

## 阶段目标

使用应用账号建立可重复执行的 PostgreSQL migration、Psycopg 3 异步连接池和
pgvector Schema，并用真实数据库证明 `vector(1024)`、余弦操作符与 HNSW 基础设施可用。

## 完成内容

### Migration

新增：

- `migrations/001_enable_vector.sql`
- `migrations/002_create_tables.sql`
- `src/rag_demo/migrations.py`

Migration runner 会：

- 按 `NNN_name.sql` 发现并排序 migration；
- 对 SQL 内容计算 SHA-256 checksum；
- 在 `schema_migrations` 保存版本、名称、checksum 和应用时间；
- 使用 PostgreSQL transaction advisory lock 串行化并发初始化；
- 在同一事务中应用全部待执行 migration；
- 重复执行时跳过已应用且 checksum 相同的版本；
- 已应用 migration 被改名或修改内容时拒绝继续。

### Schema

创建需求书规定的：

- `documents`
- `chunks`
- `ingestion_requests`

并创建：

- `chunks.embedding vector(1024)`
- `chunks_embedding_hnsw`，使用 `vector_cosine_ops`
- HNSW 参数 `m=16`、`ef_construction=64`
- `chunks_document_id_idx`
- `chunks_metadata_gin`

数据库约束包含：

- Document 来源 revision 唯一键；
- `(document_id, chunk_index)` 唯一键；
- Document 删除时级联删除 Chunk；
- `embedding_dimensions = 1024` 检查约束。

### 异步数据库模块

新增 `src/rag_demo/db.py`：

- 使用 `psycopg_pool.AsyncConnectionPool`；
- Pool 显式 `open=False`，由异步生命周期负责打开和关闭；
- 每个连接通过 `pgvector.psycopg.register_vector_async` 注册向量适配器；
- 提供异步 context manager 借用连接；
- 提供 `DatabaseStatus`，集中读取数据库、扩展、向量列、索引、migration 和行数。

### Windows asyncio 兼容

新增 `src/rag_demo/asyncio_compat.py`。Psycopg 异步模式在 Windows 上不支持默认的
Proactor event loop，因此 CLI 使用 Selector loop factory。测试会话也切换为 Selector
policy；核心数据库 API 仍保持可直接 `await`。

### CLI

新增：

```text
python -m rag_demo db init
python -m rag_demo db status
```

CLI 仅负责加载配置、运行异步入口和展示结果，migration 与状态逻辑都在数据库模块中。

## TDD 过程

1. 先写 migration 文件发现、排序、checksum、重复版本和非法文件名测试。
2. 先写真实数据库 migration 幂等、Schema 状态和向量写入测试。
3. 首次集成运行发现 Psycopg 无法使用 Windows Proactor loop，据官方异常提示改用
   Selector loop。
4. 第二轮发现普通 `list[float]` 会被 Psycopg 识别为 `double precision[]`，不能参与
   pgvector `<=>`。测试改为 NumPy `float32` 向量，明确了后续 Embedding 到数据库的类型契约。

## 验证证据

自动检查结果：

- 18 个测试全部通过；
- 总覆盖率 88.81%；
- Ruff 全部通过；
- Mypy strict 全部通过。

真实 CLI 状态：

```text
User             rag_app
Database         rag_demo
PostgreSQL       18.4
pgvector         0.8.5
Embedding column vector(1024)
HNSW             HNSW ready
Documents        0
Chunks           0
Migrations       001_enable_vector, 002_create_tables
```

直接查询 PostgreSQL 系统目录确认：

```text
USING hnsw (embedding vector_cosine_ops)
m=16
ef_construction=64
```

集成测试在强制回滚事务内插入一个 Document 和一个 1024 维 Chunk，验证向量维度为
1024、相同非零向量余弦相似度为 1。退出事务后 Document/Chunk 数量仍为 0。

## 面试讲解要点

- HNSW 不只是写在 SQL 文件中；CLI 和系统目录检查证明索引真实存在且 operator class 正确。
- Migration 依靠数据库 advisory lock 和唯一版本记录，而不是进程内锁。
- pgvector-python 负责 Python/NumPy 与 PostgreSQL vector 的类型适配。
- 全零向量没有定义良好的余弦距离，因此校验使用非零向量。
- Windows 异步事件循环是 Psycopg 的实际部署约束，CLI 在统一入口显式处理。

## 当前限制与下一阶段

当前数据库为空，尚未解析文档、生成 Embedding 或执行 Dense 查询。阶段 1B 将实现纯
Markdown 结构解析与 Chunk；该阶段不写数据库，以便对 Chunk 行为进行快速、确定的单元测试。
