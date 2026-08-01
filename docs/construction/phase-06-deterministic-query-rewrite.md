# 阶段 6：确定性 Query Rewrite 与效果对比

## 阶段目标

补全 RAG 检索前常见的 Query Rewrite 教学环节，但不引入额外 LLM 链路，也不预设 Rewrite 会提高
检索质量。本阶段把 Rewrite 当作一个可开关、可观察、可重复的实验变量：结果不变、排名变化、候选进入
或离开 Final Top-K 都是应当如实记录的结果。

## 设计边界

`HybridSearchService`、Dense、BM25、RRF 和 Reranker 保持不感知 Rewrite。新模块位于完整检索链路之前：

```text
original query
  -> optional QueryRewriter
  -> effective query
  -> unchanged Hybrid Search pipeline
```

语义改写、对话指代消解、HyDE、多查询扩展、翻译、拼写模型和 LLM 调用均不在本阶段范围内。Agent
仍应负责需要对话历史和语义理解的改写；RAG 侧这里只演示确定性的检索输入治理。

## 实现内容

### 1. 独立、版本化的别名字典

`data/query_aliases.json` 是提交进 Git 的 JSON 词典，格式为：

```json
{
  "version": 1,
  "aliases": {
    "PG": "PostgreSQL",
    "请求体": "Request Body"
  }
}
```

路径由 `QUERY_ALIASES_PATH` 配置，默认值是 `data/query_aliases.json`。词典加载会拒绝不支持的版本、
空词典、空规范词、大小写不敏感后重复的别名和损坏的 JSON。Rewrite 关闭时不会读取词典，因此词典故障
不会破坏默认检索链路。

### 2. 确定性规则

`QueryRewriter` 按以下顺序执行：

1. Unicode NFKC 规范化；
2. 去除首尾空白并折叠连续空白；
3. 大小写不敏感地匹配别名；
4. 将别名扩展为 `规范词 (原始别名)`，不删除原有检索词。

英文数字别名使用 ASCII 字母数字边界，所以 `API接口` 能命中 `API`，而 `FastAPI` 不会被误拆。
替换只扫描规范化后的原始查询一次，不会让新插入的规范词继续触发其他规则；对自身输出再次 Rewrite
也是幂等的。

NFKC 本身也会产生可观察影响，例如中文全角问号 `？` 会成为 `?`。这不是别名规则，却仍记录为
`unicode_nfkc`。

### 3. 可观察结果

每次可选 Rewrite 都保留：

- `original_query`；
- `normalized_query`；
- `effective_query`；
- `applied_rules`；
- `changed`；
- `rewrite_ms`。

CLI `search` 增加默认关闭的 `--rewrite/--no-rewrite`。REST `POST /v1/search` 增加默认值为
`false` 的 `rewrite`，并在响应中同时返回原始查询、有效查询和 Rewrite 明细。

### 4. 中性的效果对比

`rag-demo compare-rewrite <query>` 顺序执行两次真实检索：第一次关闭 Rewrite，第二次开启 Rewrite。
即使 Rewrite 后字符串完全相同，也不会复用或短路第二次运行。

报告包含：

- 两侧 Final Top-K 完整结果；
- 相同 Chunk 的前后排名和 `rank_delta = without_rank - with_rank`；
- 只出现在关闭侧或开启侧的 Chunk；
- 最终顺序是否完全一致；
- Rewrite 本身和两次搜索的原始耗时。

字段只陈述变化，不使用“提升”“退化”等质量判断。没有相关性标注时，排名向前不等于答案更好。
两次查询按固定顺序运行，单次耗时会受模型冷启动、数据库缓存和 BM25 页缓存影响，不能直接用来得出
性能结论；严谨延迟比较应进行预热、交替顺序和多轮统计。

## TDD 过程

先写失败测试并确认旧代码缺少 `rag_demo.query_rewriter`、CLI 开关、REST 字段和配置项。随后补齐：

- NFKC 与空白规范化；
- 保留原词的别名扩展；
- 大小写不敏感、非级联和英文 token 边界；
- 词典错误路径；
- Rewrite 输出幂等性；
- 无变化时仍真实执行两次检索；
- 相同结果、成员变化和排名变化的中性差异模型；
- CLI、REST 和真实本地模型链路。

第一次真实链路测试揭示 NFKC 会把 `？` 转成 `?`，测试改为验证该真实行为。随后新增回归测试发现
全角括号输出经 NFKC 后会破坏二次 Rewrite 幂等性，最终改用稳定的 ASCII 括号格式。

## 修改模块

- `src/rag_demo/query_rewriter.py`
- `src/rag_demo/application.py`
- `src/rag_demo/cli.py`
- `src/rag_demo/api.py`
- `src/rag_demo/config.py`
- `data/query_aliases.json`
- `.env.example`
- `README.md`
- Query Rewrite 单元测试及 CLI/REST 集成测试

## 验证证据

- Query Rewrite 与应用边界定向测试：通过
- CLI + REST 真实端到端测试：`2 passed`
- 全套测试：`109 passed`
- 总覆盖率：`85.37%`，通过项目 `80%` 门槛
- Mypy strict：通过
- 本阶段修改文件的 Ruff lint 与 format 检查：通过
- 全仓 Ruff lint 仅被用户代码审阅期间留在 `chunker.py` 和 `dense_retriever.py` 的三条长注释拦截；
  本阶段保留这些审阅内容，没有擅自改写

在既有 FastAPI 中文语料上，以关闭 Reranker 的 `如何声明请求体？` 做一次观察：Rewrite 前后的
Final Top-5 只有两个共同 Chunk，成员和顺序都发生变化。这只能证明 Rewrite 确实会改变检索结果，不能在
缺少相关性标签的情况下证明变化是正面或负面。

## 讲解要点

- Query Rewrite 是检索前处理，不属于 Dense Retriever 内部职责。
- LLM 不是 Rewrite 的必要条件；格式规范化和领域别名映射可以完全确定性地完成。
- 保留原始查询与有效查询，才能复现一次检索并解释排名为何变化。
- 对比实验的职责是暴露影响，不是保证“开启功能后指标必然上升”。
- 有标注评测集时再谈 Recall/MRR 是否改善；没有标签时只能描述候选与排名变化。
