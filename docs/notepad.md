# 笔记

## INGEST 幂等性

| 当前记录 | 本次请求 | 处理结果 |
|---|---|---|
| 不存在 | 任意合法请求 | 插入 `processing`，开始执行 |
| `completed` | 相同 Key、相同 Hash | 直接返回以前保存的结果 |
| `processing` | 相同 Key、相同 Hash | 报告正在处理，避免并发重复执行 |
| `failed` | 相同 Key、相同 Hash | 改回 `processing`，允许重试 |
| 任意状态 | 相同 Key、不同 Hash | 幂等键冲突，拒绝执行 |


## 生产级 RAG 系统的文档来源 ID

- 内部 Document ID
- 外部来源 ID: 分布式id/数据库id/url
- 具体版本 ID: 版本标识  


## RRF 公式

`fused.rrf_score += 1 / (rank_constant + rank)`
取 `rank_constant = 60`


## 去重逻辑

- TIYA 已经在使用的 hashsim
- 限制每个 chunk 的结果数量
- 多版本去重
- 来源去重