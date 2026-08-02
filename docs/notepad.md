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


## 生产级 RAG 链路

```text
数据连接器
Git / S3 / SharePoint / Notion / 本地上传
        ↓
原始文件存储
Object Storage + Content Hash + Version
        ↓
MIME 检测和格式路由
        ↓
格式专用 Parser
PDF / Word / PPT / Excel / HTML / OCR
        ↓
统一 Document IR
节点、层级、顺序、坐标、表格、来源
        ↓
质量检查
乱码、空页、OCR 置信度、表格完整性
        ↓
内容增强
语言、实体、别名、摘要、权限、PII
        ↓
格式感知 Chunk
短片段合并、超长拆分、Parent/Child
        ↓
派生检索表达
retrieval_text / table_html / captions
        ↓
多路索引
Dense / BM25 / Metadata / Entity / Summary
        ↓
Query 处理
规范化、权限、路由、可选 Rewrite
        ↓
多路召回
        ↓
融合和候选保护
        ↓
Reranker
        ↓
Parent/Neighbor 扩展
        ↓
去重与 Context Packing
        ↓
LLM 生成和引用
        ↓
反馈、评测和持续重建
```
