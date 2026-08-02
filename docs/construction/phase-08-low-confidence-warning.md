# 阶段 8：低可信检索结果提醒

## 阶段目标

补齐检索系统面对乱码、域外问题和无厘头 Query 时的可信度表达。当前阶段不增加硬过滤，不改变
Dense、BM25、RRF、Reranker 或 Final Top-K 的成员和顺序；它只在最终证据偏弱时返回结构化警告，
避免调用方把“所有候选里相对最高”误解成“已经足够相关”。

## 用户旅程

1. 正常可回答 Query 继续返回原有 Top-K，不显示额外警告。
2. 乱码或域外 Query 即使仍有最近邻结果，也会收到“结果可能不可信”的提示。
3. CLI、REST 和 Python SDK 使用同一份核心可信度对象，不各自推导阈值。
4. 关闭 Reranker 时明确返回 `unassessed`，不使用 RRF 或 BM25 原始分数冒充最终可信度。

## 设计决策

### 只告警，不过滤

本阶段保留 Demo 的完整可观察链路。低于告警线时仍返回 `final_top_k`，便于演示 Dense 强制最近邻、
BM25 词面召回、RRF 相对排名和 Reranker 绝对分数之间的区别。未来增加拒答时，可以在同一个
`SearchConfidence` 结果之上决定是否清空证据，但不会把过滤策略藏进 Retriever。

### 以 Top-1 Reranker 分数作为最终证据

Dense cosine、BM25 score 和 RRF score 的量纲不同，且 RRF 只表达相对排名。Reranker 直接联合观察
Query 与候选全文，所以本阶段使用最终第 1 名的 Reranker 原始分数作为告警依据。

`SearchConfidence` 有三种状态：

- `not_flagged`：Top-1 分数大于或等于告警线；
- `low`：Top-1 分数低于告警线，结果保留但附带警告；
- `unassessed`：Reranker 被关闭，或没有最终结果，无法执行该项评估。

`not_flagged` 只表示没有触发当前启发式规则，不表示结果经过概率校准，也不承诺答案正确。

### 可配置、有限数值阈值

`.env.example` 增加：

```dotenv
RERANKER_LOW_CONFIDENCE_THRESHOLD=-4.0
```

配置拒绝 `NaN` 和无穷值。等于阈值时不告警，只有严格低于阈值才进入 `low`。替换 Reranker 模型、
模型量化、服务端打分变换或语料后，必须重新采样校准，不能机械复用 `-4.0`。

## 真实校准

在 1145 个 FastAPI 中文文档 Chunk、`bge-m3` 和 `bge-reranker-v2-m3` 的当前环境中采样：

- 10 个已提交评测 Query 的 Top-1 Reranker 分数范围：`-2.881` 到 `6.658`；
- 3 个域外 Query 与 3 个乱码 Query 的范围：`-8.653` 到 `-5.206`；
- 两组样本在当前小规模验收中没有重叠，因此选择中间的 `-4.0` 作为 Demo 默认告警线。

这个校准集很小，只足以支撑“提示”功能，不足以支撑生产级拒答、置信概率或统计泛化结论。

## 对外行为

核心 `SearchResponse` 新增：

```text
confidence.status
confidence.score
confidence.threshold
confidence.warning
```

REST `/v1/search` 始终序列化该对象。Python SDK 直接返回相同的强类型对象，并从包根目录导出
`SearchConfidence` 和 `SearchConfidenceStatus`。CLI 只在存在 warning 时打印黄色提醒；低分时同时展示
Top-1 Reranker 分数和告警线，便于现场讲解。

## TDD 过程

先增加核心、配置、CLI 和 REST 契约测试，确认旧代码因缺少 `SearchConfidenceStatus` 和 CLI 输出函数
在测试收集阶段失败。随后实现最小功能并验证：

- 低于告警线只提示，不删除或重排结果；
- 等于告警线不误报；
- 关闭 Reranker 返回 `unassessed`；
- 默认值、环境变量覆盖和非有限配置校验；
- CLI 打印分数证据；
- REST 返回结构化可信度对象。

定向测试的 22 个断言全部通过。定向运行因项目配置要求全仓 80% 覆盖率而返回非零退出码；这是只执行
三个测试文件时全仓覆盖率为 38% 所致，不是断言失败。随后运行全套测试确认覆盖率门槛通过。

## 修改模块

- `src/rag_demo/search_service.py`
- `src/rag_demo/application.py`
- `src/rag_demo/config.py`
- `src/rag_demo/cli.py`
- `src/rag_demo/api.py`
- `src/rag_demo/__init__.py`
- `.env.example`
- `README.md`
- 核心、配置、CLI 与 REST 测试

## 验证证据

- 基础设施 Doctor：全部通过；
- 正常真实 Query `OAuth2PasswordBearer`：Top-1 `4.635`，不告警；
- 真实乱码 Query `asdfghjklqwertyuiopzxcvbnm`：Top-1 `-6.458`，保留 5 条结果并显示警告；
- 真实 REST 乱码请求：HTTP 200、5 条结果、`confidence.status=low`，分数与阈值完整返回；
- 全套单元与真实集成测试：`122 passed`；
- 总覆盖率：`85.47%`，通过项目 `80%` 门槛；
- Mypy strict：`51 source files` 无错误；
- 本阶段修改文件 Ruff format：通过；
- Ruff lint 除既有代码审阅长注释的 `E501` 外通过，本阶段没有擅自改写审阅注释。

## 讲解要点

- Top-K 是数量约束，不是相关性保证；最近邻一定能找到“最近的”，但最近的不一定可信。
- RRF 排名分数不能直接充当绝对相关性阈值。
- Reranker 原始分数可以做当前环境的启发式告警，但不是概率。
- `not_flagged` 不等于“正确”，`low` 也不删除证据；调用方仍能观察完整链路。
- 生产级拒答需要更大的域内/域外标注集、模型版本化、阈值校准和持续监控。
