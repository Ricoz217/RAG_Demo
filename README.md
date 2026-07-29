# Minimal Hybrid RAG Demo

一个面向检索链路演示的最小 Hybrid RAG 项目。

项目采用 Python 3.12 和 `src` 布局，只覆盖文档摄取、Dense/BM25 检索、RRF
融合和 Cross-Encoder Rerank，不生成最终自然语言答案。

## 开发环境

在 PowerShell 中执行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install uv
.\.venv\Scripts\uv.exe sync --all-groups
Copy-Item .env.example .env
```

然后根据本机 PostgreSQL、Embedding 和 Reranker 服务修改 `.env`。真实 `.env`
包含密钥且已被 Git 忽略。

## 工程检查

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\mypy.exe src tests
.\.venv\Scripts\python.exe -m rag_demo --help
```

依赖的精确版本保存在 `uv.lock`。施工过程和各阶段验收证据保存在
`docs/construction/`。

