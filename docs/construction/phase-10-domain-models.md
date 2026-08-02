# 阶段 10：公共数据模型领域化整理

## 阶段目标

解决公共数据结构散落在解析器、Retriever、Service、Application等实现模块中的问题。调用方可以从
`rag_demo.models`统一查找数据契约，阅读业务实现时也不再需要反复跳转寻找dataclass定义。

本阶段是纯重构：不修改数据库Schema、REST JSON、SDK返回类型的字段或检索行为。

## 问题盘点

重构前约有39个公共dataclass与枚举分散在十余个模块。典型依赖包括：

- `Chunk`定义在 `chunker.py`，但由摄取服务消费；
- Dense、BM25、RRF和最终搜索各自定义一套结果结构；
- SDK为了声明返回类型，需要从多个业务实现模块导入；
- Doctor、数据库状态和评测报告与其执行逻辑混在一起。

直接建立单个 `model.py` 会形成数百行的万能模块，因此改用按领域拆分的 `models/` 包。

## 最终布局

```text
src/rag_demo/models/
  __init__.py       统一公共导出
  document.py       Markdown、Block、Chunk
  retrieval.py      Dense、BM25、RRF、Search
  ingestion.py      IngestResult
  rewrite.py        Rewrite结果与A/B排名差异
  operations.py     DatabaseStatus、Doctor、Corpus
  evaluation.py     Benchmark查询、命中与指标
```

依赖方向固定为：

```text
models <- parser / retriever / service / application <- SDK / API / CLI
```

除 `models.rewrite` 引用同包的 `models.retrieval.SearchResponse` 外，模型包不引用任何业务实现，保持为
稳定的依赖叶节点。

## 保留在原模块的内容

没有把所有“看起来像类型”的对象都迁入模型包：

- `_IndexManifest`、`_ScoredChunk`、`_DocumentResult`、`_MutableFusedCandidate`等私有实现结构；
- 各Service消费的 `Protocol`；
- REST专用的Pydantic输入模型 `SearchBody`、`IngestBody`；
- `Settings`、`Migration`及具体异常子类；
- Retriever、Service、Client等有行为的对象。

`BM25IndexError`基类与 `BM25BuildResult.from_json()`存在直接异常契约，因此一起放在
`models.retrieval`，原 `bm25_retriever.py`继续重新导出它；具体索引异常子类仍留在实现模块。

## 向后兼容

旧路径没有立即删除，而是重新导出同一个类对象。例如：

```python
from rag_demo.models import SearchResponse
from rag_demo.search_service import SearchResponse as LegacySearchResponse

assert SearchResponse is LegacySearchResponse
```

这样已有测试、演示脚本和外部调用不会因为本次整理中断；新代码统一推荐从 `rag_demo.models`或顶层
`rag_demo`公共SDK入口导入。

## TDD过程

先增加模型组织契约测试，验证：

- 新公共类型的 `__module__` 必须属于 `rag_demo.models.*`，排除只做聚合导出的假整理；
- 旧模块导入与新模块导入必须是同一个类，避免产生两套运行时类型。

测试最初因 `rag_demo.models`不存在而在收集阶段失败；完成领域模型迁移和兼容导出后转绿。

## 验证结果

- 关键模型及关联业务单元测试：`51 passed`；
- 全量测试：`127 passed`；
- 分支覆盖率：`85.61%`；
- 全项目ruff检查通过；
- 全项目mypy严格类型检查通过，共检查59个源文件；
- 真实PostgreSQL、Embedding、Dense/BM25、SDK、REST和CLI集成测试未出现行为回归。

## 修改模块

- 新增 `src/rag_demo/models/` 六个领域模型文件及统一导出；
- 原模型定义所在模块改为消费并兼容导出新类型；
- SDK、REST、CLI和应用组装层改为从模型包读取公共契约；
- 新增 `tests/unit/test_models.py`；
- 更新README项目结构。

## 讲解要点

- 集中管理不等于集中到一个文件；按领域聚合能同时改善导航和模块内聚。
- 模型包只描述跨边界传递的数据，Protocol和私有状态仍与其消费者或实现放在一起。
- 兼容导出允许大型项目分阶段迁移调用方，不必在一次重构中破坏全部旧导入。
