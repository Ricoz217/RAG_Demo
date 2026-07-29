# 阶段 0：工程基线

## 阶段目标

建立一个可以重复安装、统一读取配置并执行质量检查的 Python 3.12 `src` 布局，
为后续数据库、摄取和检索模块提供稳定入口。

## 完成内容

### 依赖与环境

- 保留项目级 `.venv`，实际解释器为 Python 3.12.9。
- 使用 `uv` 解析并同步项目依赖。
- 生成 `uv.lock`，锁定生产和开发依赖的精确版本。
- 生产依赖包含 FastAPI、HTTPX、Psycopg 3、pgvector、bm25s、jieba、
  Pydantic Settings、Typer、Rich 和 markdown-it-py。
- 开发依赖包含 pytest、pytest-asyncio、pytest-cov、Ruff 和 Mypy。

### 配置模块

新增 `src/rag_demo/config.py`：

- 从 `.env` 加载 PostgreSQL、Embedding、Reranker、Chunk 和检索参数。
- 校验 PostgreSQL DSN、HTTP URL、正数维度、批量大小和超时。
- 校验 Chunk target/max/overlap 之间的关系。
- 校验 `final_top_k` 不大于 `rerank_top_k`。
- 生成 `/v1/embeddings` 和 `/reranking` 完整端点。
- 数据库密码和两个 API Key 使用秘密类型，避免出现在对象表示和日志中。

`.env` 保存本机真实连接信息并由 Git 忽略；`.env.example` 只保存可提交的占位配置。

### CLI 骨架

新增：

- `src/rag_demo/cli.py`
- `src/rag_demo/__main__.py`
- `src/rag_demo/py.typed`

当前 CLI 提供统一应用入口、帮助信息和版本输出。后续阶段只扩展命令，不创建第二套业务逻辑。

### 工程质量

`pyproject.toml` 统一配置：

- pytest 严格配置和 marker 检查；
- pytest-cov 分支覆盖率，最低门槛 80%；
- Ruff 的 Python 3.12、导入、异步和常见错误规则；
- Mypy strict 模式和 Pydantic 插件。

## TDD 过程

先创建配置和 CLI 行为测试，首次运行因 `rag_demo.config` 和 `rag_demo.cli` 尚不存在而
在收集阶段失败。最小实现后，测试发现 `PostgresDsn` 会在 `Settings.__repr__` 中暴露
密码；随后改为经过 PostgreSQL DSN 校验的 `SecretStr`，修复了该安全问题。

## 验证证据

阶段完成时执行：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\mypy.exe src tests
.\.venv\Scripts\python.exe -m rag_demo --version
.\.venv\Scripts\python.exe -m rag_demo --help
```

结果：

- 10 个单元测试全部通过。
- 总覆盖率 91.67%，高于 80% 门槛。
- Ruff 全部通过。
- Mypy strict 全部通过。
- CLI 输出版本 `0.1.0`，帮助页正常。
- 使用真实 `.env` 创建 `Settings` 成功，Embedding 维度为 1024。

## 面试讲解要点

- CLI、REST 和 Python API 将共享同一个强类型配置与核心服务。
- 密钥只存在于未跟踪的 `.env`，并且配置对象自身也不会打印明文。
- `uv.lock` 与 Python 3.12 约束保证现场可以重建相同依赖环境。
- 测试不是实现后的装饰；第一轮测试实际发现并阻止了 DSN 密码泄漏。

## 当前限制与下一阶段

本阶段尚未创建数据库连接池、业务表或 migration。阶段 1A 将实现 PostgreSQL
异步连接、pgvector Schema、HNSW 索引以及 `db init/status` 命令。
