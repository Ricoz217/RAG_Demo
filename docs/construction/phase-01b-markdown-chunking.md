# 阶段 1B：Markdown 解析与 Chunk

## 阶段目标

只依赖 Markdown 结构和字符长度，把 `.md` 原文转换成可追溯、可重复、适合 Embedding/
BM25/Reranker 共用的 Chunk，不引入模型 tokenizer。

## 完成内容

### Markdown 解析

新增 `src/rag_demo/markdown_parser.py`，使用 `markdown-it-py` 的 block token 和源代码
行号映射识别：

- 文档标题；
- Heading 层级；
- 普通段落；
- 有序和无序列表；
- fenced code block 和 indented code block；
- HTML/blockquote 等可保留的正文容器。

解析器保留原始 Markdown 文本，而不是只保存渲染后的纯文本。首个 H1 被识别为文档标题，
不会在章节 breadcrumb 中重复；后续 Heading 根据层级形成完整路径。同级或更高层 Heading
会正确清除旧的子路径。

### 结构模型

解析阶段输出不可变对象：

- `ParsedMarkdownDocument`
- `ParsedBlock`
- `BlockKind`

每个 Block 保存：

- 类型；
- 原始文本；
- `heading_path`；
- `section_root`。

`section_root` 用于阻止 overlap 跨越新的一级章节。

### Chunker

新增 `src/rag_demo/chunker.py`：

- 默认版本：`markdown-structure-v1`；
- 只合并具有相同 Heading 路径的连续 Block；
- `target_chars` 是软目标；
- `max_chars` 是不可超过的硬上限；
- 合法尺寸的代码块保持完整；
- 超大代码块允许按字符窗口拆分；
- overlap 只发生在同一 Heading 路径内；
- Chunk 在整个 Document 内使用连续 `chunk_index`；
- `char_count` 统计 `content_raw`；
- `content_hash` 使用 SHA-256。

### 原文与检索文本

每个 Chunk 分别保存：

```text
content_raw
retrieval_text
```

检索文本格式：

```text
[文档：文档标题]
[章节：父章节 / 子章节]
Chunk 原文
```

后续 Embedding、BM25 和 Reranker 将统一使用该字段。

### 跨平台稳定性

解析前统一把 CRLF 和旧式 CR 换行规范化为 LF，因此同一 Markdown 在 Windows/Linux 上会
产生相同 Document hash、Block 和 Chunk hash。

## TDD 过程

先编写以下外部行为测试，再创建解析器与 Chunker：

- 标题、Heading 路径、列表和代码块解析；
- Heading 嵌套、同级替换和新 H1 重置；
- 无标题文档使用文件名；
- CRLF/LF hash 稳定性；
- 合法代码块保持完整；
- 不合并不同章节；
- overlap 保留且不跨章节；
- 超大代码块拆分后不超过上限；
- Chunk metadata 与 retrieval context；
- 无 overlap 模式和非法参数边界。

首次运行因模块尚不存在而在测试收集阶段失败；完成最小实现后所有行为测试通过。

## 验证证据

全套检查结果：

- 33 个测试全部通过；
- Ruff 全部通过；
- Mypy strict 全部通过；
- 总覆盖率高于 80% 门槛。

使用当前项目的中文需求书进行真实 smoke test：

```text
Markdown blocks: 192
Code blocks: 54
Chunks: 47
Maximum content_raw chars: 1533
Configured maximum chars: 2200
```

说明解析器能够处理包含大量标题、列表、表格文本和代码块的真实中文技术文档，并且所有
Chunk 均符合字符硬上限。

## 面试讲解要点

- Chunker 没有固定每 N 个字符截断，而是先通过 Markdown 结构取得语义边界。
- Document title 与 Heading breadcrumb 分开保存，避免检索前缀重复。
- 不同 Heading 不进入同一个 Chunk，使来源章节保持确定。
- overlap 是召回辅助信息，不允许跨越新章节污染语义。
- 原文和 retrieval text 分离，展示时可使用原文，检索时可利用标题与 breadcrumb。
- hash 先规范化换行，避免同一 Git 文档因操作系统换行差异被误判为内容变化。

## 当前限制与下一阶段

- 第一版只支持 Markdown，不解析 PDF、DOCX 或 HTML 文件。
- 超大代码块允许拆分，拆分片段不保证各自仍是语法完整的 fenced Markdown。
- 尚未把 Chunk 写入 PostgreSQL。

阶段 1C 将实现 llama.cpp Embedding Client、Document/Chunk 摄取事务、数据库 advisory
lock、重复 Embedding 跳过规则和 REST `Idempotency-Key` 的底层持久化语义。
