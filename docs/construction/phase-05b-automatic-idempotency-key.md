# 阶段 5B：摄取幂等键自动生成

## 阶段目标

让 Demo 的通用 CLI、FastAPI REST 和核心摄取服务在调用方未提供 `idempotency_key`
时自动生成稳定键，同时保留显式键作为覆盖接口。生成规则必须包含会改变摄取结果的流水线版本，避免
Markdown Parser/Chunker 升级后错误命中旧的已完成请求。

本阶段不增加幂等记录 TTL、租约、后台清理任务或 `processing` 崩溃恢复机制。

## 设计决定

自动键统一由 `DocumentIngestor` 生成，CLI 和 REST 不各自维护一套哈希算法：

```text
ingest-<SHA-256 canonical request hash>
```

规范化请求包含：

- 相对于 `source_root` 的来源路径；
- `source_repo` 和 `source_commit`；
- 文档语言；
- Chunker 版本；
- `target_chars`、`max_chars`、`overlap_chars` 三项切分参数；
- Embedding 模型名和向量维度。

绝对的本机 checkout 路径和 `source_root` 不进入哈希。因此，同一仓库、同一提交、同一相对来源和
同一流水线配置复制到另一台机器后仍会得到相同的键。调用方显式传入非空键时仍原样使用；显式空字符串
不会被当成“未提供”，仍由原有校验拒绝。

文件内容本身不重复扫描进幂等哈希，Git/外部来源的内容版本由 `source_commit` 表达。调用方若在不改变
`source_commit` 的情况下修改工作区内容，相当于破坏来源版本契约。

## 接口行为

- `DocumentIngestor.ingest_path(..., idempotency_key=None)` 自动生成键。
- `DocumentIngestor.idempotency_key_for(source)` 可在执行前查看默认键。
- `rag-demo ingest` 的 `--idempotency-key` 改为可选，并在输出中显示最终使用的键。
- `rag-demo corpus ingest-fastapi` 复用同一核心算法，并继续显示最终键。
- `POST /v1/ingest` 的 `Idempotency-Key` 请求头改为可选；成功响应通过同名响应头返回最终键。
- 显式键的冲突、处理中和失败重试语义不变。

## TDD 过程

先增加失败用例，确认旧实现缺少以下行为：

- 核心摄取器不存在默认键生成接口；
- CLI 仍要求必填 `--idempotency-key`；
- REST OpenAPI 仍把 `Idempotency-Key` 标记为必填；
- 相同逻辑来源位于不同 checkout 目录时，旧请求哈希不稳定；
- 相同 Chunker 版本但切分参数不同时，旧请求哈希不会变化。

实现后增加真实 PostgreSQL 集成测试：不传键连续摄取两次，第二次直接读取已完成结果，Embedding Provider
只调用一次。

## 修改模块

- `src/rag_demo/ingest_service.py`
- `src/rag_demo/chunker.py`
- `src/rag_demo/cli.py`
- `src/rag_demo/api.py`
- `tests/unit/test_ingest_idempotency.py`
- `tests/unit/test_cli.py`
- `tests/integration/test_ingest_service.py`

## 验证证据

- 自动键与 Chunker 定向测试：`17 passed`
- 真实 PostgreSQL 自动键集成测试：`1 passed`
- 全套测试：`93 passed`
- 总覆盖率：`84.37%`，通过项目 `80%` 门槛
- Mypy strict：通过
- 本阶段独立修改文件的 Ruff 检查：通过
- 全仓 Ruff：工作区中用户审阅期间加入 `chunker.py` 的两行长注释触发既有 `E501`，本阶段未改写该审阅内容

第一次全套测试时 8081/8082 没有监听，六项真实模型测试收到 HTTP 502。使用项目既有
`D:\Llama_cpp\runs` 启动脚本恢复两个 llama.cpp 服务后，Embedding 与 Reranker 定向测试通过，随后全套
93 项测试通过。一次全套复跑中，并发状态测试因固定 50ms 调度窗口出现一次时序抖动；该用例随后连续
两次单独运行均通过，没有修改并发实现来迁就偶发调度。

## 讲解要点

- 幂等键是一次逻辑操作的外部身份，请求哈希是服务端验证该身份是否仍映射到同一请求的指纹；本 Demo
  在未提供外部身份时，用请求哈希派生默认身份。
- 相对来源路径保证跨 checkout 稳定，仓库和提交保证不同版本不会混淆。
- 流水线版本与切分参数都必须进入指纹：代码规则升级和运行配置变化都会改变 Chunk 边界或检索文本。
- 自动生成降低 Demo 调用门槛，显式覆盖仍用于上游已有请求 ID、作业 ID 或消息 ID 的场景。
