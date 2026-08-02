# 阶段 5A：Front Matter 与首个 H1 结构修正

## 阶段目标

修正第一版 Markdown Parser 对“首个 H1 必然是文档标题”的假设，使规范的 `SKILL.md`
和没有统一标题规范的 Markdown 都能保留真实章节结构；同时解析 YAML Front Matter，
但不把来源路径、文件名、Preamble 或任意元数据自动传播到检索文本。

本阶段只调整 Markdown 解析和相邻的 Chunk/摄取类型衔接，不扩展数据库文档版本、对象
存储或证据文件管理。

## 设计决定

- 使用 `python-frontmatter`，不自行实现 Front Matter 边界和 YAML 解析。
- Front Matter 先从正文中分离，剩余正文再交给 `markdown-it-py`。
- `frontmatter.title` 优先作为显式语义标题；缺失时使用 `frontmatter.name`。
- 没有显式 Front Matter 标题时，`title` 为 `None`，不再从文件名或 H1 猜测。
- 首个 H1 与后续 H1 采用完全相同的章节规则，不再被 Parser 消耗。
- 单独记录 `first_h1`，但它不自动成为文档标题。
- H1 之前的 Preamble 保留为 `heading_path=()` 的普通正文块，不复制到其他 Chunk。
- `source_path` 继续用于来源追踪，但不加入 `retrieval_text`。
- Front Matter 的 `description` 等字段只保存在解析结果中，本阶段不自动注入所有 Chunk。

## 数据结构变化

`ParsedMarkdownDocument` 新增：

```text
first_h1
frontmatter
```

`title` 从必填字符串改为可选字符串：

```text
frontmatter.title
→ frontmatter.name
→ None
```

完整原始 Markdown（包含 Front Matter）仍用于 Document SHA-256；MarkdownIt 只解析
剥离 Front Matter 后的正文。

## Heading 行为

以下文档：

```markdown
文档前言

# 任务

任务正文

# 规则与说明

规则正文
```

解析为：

```text
文档前言      heading_path=()
任务正文      heading_path=("任务",)
规则正文      heading_path=("规则与说明",)
```

标准 `SKILL.md` 的 Front Matter 不再被 CommonMark 误判或静默丢弃，首个 H1 也会进入
完整 breadcrumb：

```text
("Python Development Patterns", "When to Activate")
```

## 检索文本变化

存在显式 Front Matter 标题时：

```text
[文档：python-patterns]
[章节：Python Development Patterns / When to Activate]
Chunk 原文
```

没有显式标题时：

```text
[章节：任务]
Chunk 原文
```

不会自动添加文件名、路径或 Preamble。由于 Heading 路径和检索文本语义发生变化，
Chunker 版本从 `markdown-structure-v1` 提升为 `markdown-structure-v2`，避免错误复用
旧向量。

## TDD 过程

先增加以下失败用例：

- YAML Front Matter 完整解析并从 Markdown 正文中移除；
- `title` 和 `name` 的显式语义标题优先级；
- 首个 H1 保留在 `heading_path`；
- 多个 H1 正确重置章节路径；
- 无 Front Matter 时不从文件名或 H1 猜测标题；
- Preamble 只保留在自己的正文块；
- 来源路径与 Preamble 不传播到其他 Chunk；
- Front Matter `description` 不自动传播到检索文本。

首次运行结果为 `10 failed, 9 passed`，证明新增用例能够捕获旧行为；完成最小实现后，
Parser/Chunker 定向测试为 `19 passed`。

## 修改模块

- `src/rag_demo/markdown_parser.py`
- `src/rag_demo/chunker.py`
- `src/rag_demo/ingest_service.py`
- `tests/unit/test_markdown_parser.py`
- `tests/unit/test_chunker.py`
- `pyproject.toml`
- `uv.lock`

新增运行依赖：

```text
python-frontmatter 1.3.0
PyYAML 6.0.3
```

`python-frontmatter` 默认使用 PyYAML 的 SafeLoader。

## 验证证据

- Parser/Chunker 定向测试：`19 passed`
- 全部单元测试：`63 passed`
- 本次改动模块覆盖率：Parser `95%`、Chunker `96%`
- 全套测试：`86 passed`
- 总覆盖率：`84.23%`，通过项目 `80%` 门槛
- Ruff：通过
- Mypy strict：通过
- 真实文件 smoke test：
  - 标准 `python-patterns/SKILL.md`：Front Matter、首个 H1 和 41 个 Block 解析正确；
  - `group_chat_system_0710.md`：Preamble、首个 H1 和 50 个 Block 解析正确。

首次运行集成测试时 PostgreSQL 尚未启动；数据库恢复后重新执行全部 86 项测试，数据库、
Embedding、Reranker、摄取、Dense、BM25、CLI 和 REST 集成链路全部通过。

## 讲解要点

- 文档身份、显式语义标题和 Markdown 章节是三种不同概念。
- H1 只负责正文结构，不再因出现顺序而获得特殊语义。
- Front Matter 是作者显式提供的元数据，优先级高于启发式猜测。
- Preamble 采用保守策略：原样索引，但不假设它是全局摘要。
- Chunker 版本必须覆盖所有会改变 `retrieval_text` 的规则，否则可能复用错误的
  Embedding。
