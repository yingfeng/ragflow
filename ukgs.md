# Unified Knowledge Graph System (UKGS) 设计文档

> 版本: v2.0 | 日期: 2026-05-07
>
> **核心统一**: RAGFlow 的 "chunk" ≡ UKGS 的 "CONTEXT"。二者是同一个东西。

---

## 1. 系统目标与设计原则

### 1.1 核心目标

统一 BookRAG、GraphRAG、LinearRAG、MGranRAG 四套系统的图谱能力，设计为单一 schema、统一存储、可切换的索引/查询策略。**所有索引管线与 RAGFlow 现有 chunk 流水线无缝集成**。

### 1.2 设计原则

1. **chunk = CONTEXT**：RAGFlow 的 chunk 概念与 UKGS 的 CONTEXT（passage/paragraph）完全统一。chunk 就是图的"上下文"节点，不再另建索引。
2. **一个索引，一个增量写入器**：存储层只有 CONTEXT（chunk 表）和 CONCEPT（新表）两类节点。写入时根据 `index_mode` 选择 LLM 或 spaCy 策略，所有写入通过增量管线在 Tokenizer 之后自动触发。
3. **一个索引，三种查询入口**：查询时根据 `query_mode` 选择入口（结构优先/语义优先/纯图谱），底层共享同一局部子图 PPR 引擎。
4. **Lineage SET 精确更新**：实体/关系按文档切片存储，增量索引只替换属于该文档的切片。通过 chunk 表的 `doc_id` 字段天然支持。
5. **不加载全图到内存**：查询时通过 ES/Infinity 检索驱动，在内存中组装局部小子图（<500 节点）。
6. **不丢失的向后兼容**：旧版 GraphRAG 数据无需重建即可共存，逐步迁移。

### 1.3 四个系统的本质保留与舍弃

| 系统 | 保留的本质 | 舍弃的负担 |
|------|-----------|-----------|
| **GraphRAG** | 实体-关系语义图 + 社区摘要 | 全局图 JSON blob；entity/relation 分两张表 |
| **LinearRAG** | PPR 传播 + sentence embedding 相似度加权 | GPU 稀疏矩阵（局部子图 CPU 即可）；三层 node 类型 |
| **MGranRAG** | Query Decomposition + 迭代重排序 + POS 短语抽取 | Phrase 作为独立 node 类型；Doc 作为独立 node 类型 |
| **BookRAG** | 文档层次结构（TOC）导航 + 结构过滤 | tree2kg 映射表；TreeNode 独立存储 |

---

## 2. 系统架构

### 2.1 核心统一：chunk = CONTEXT

```
┌─────────────────────────────────────────────────────────────────────┐
│  Unified Knowledge Graph System (UKGS)                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  用户配置层:                                                          │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  index_mode: "llm" | "spacy"                                │    │
│  │  query_mode: "tree_guided" | "graph_guided" | "graph_only"  │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌───────────────── 现有 RAGFlow 索引管线 ─────────────────────────┐│
│  │                                                                  ││
│  │  Pipeline DSL:                                                   ││
│  │    File → Parser → TokenChunker → Tokenizer                      ││
│  │                                                                  ││
│  │  Parser._invoke()  →  json_result (结构化块: text/img/table)      ││
│  │  TokenChunker._invoke() → chunks[] (每个 chunk 是 dict)          ││
│  │  Tokenizer._invoke()  → chunks[] (追加 tokenization + embedding) ││
│  │                                                                  ││
│  │  输出: chunks = CONTEXT 节点（已含 embedding）                    ││
│  └────────────────────────────┬─────────────────────────────────────┘│
│                               │                                      │
│               ┌───────────────┴───────────────┐                      │
│               ▼                               ▼                      │
│  ┌────────────────────────┐     ┌───────────────────────────┐        │
│  │ insert_chunks()         │     │ GraphIndexer (新 Component) │        │
│  │ 写入 chunk 表            │     │ 在 Tokenizer 后自动运行      │        │
│  │ (CONTEXT 持久化)         │     │                            │        │
│  └────────────────────────┘     │ 1. 读取已完成的 chunks       │        │
│               │                 │ 2. 根据 index_mode 抽取概念   │        │
│               │                 │ 3. 写入 CONCEPT 表            │        │
│               │                 │ 4. 回填 chunk.mentions       │        │
│               │                 └──────────────┬──────────────┘        │
│               │                                │                       │
│               ▼                                ▼                       │
│  ┌─────────────────────────────────────────────────────────────┐      │
│  │  存储层 (ES/Infinity)                                        │      │
│  │                                                             │      │
│  │  ┌──────────────────────┐  ┌────────────────────────┐      │      │
│  │  │  chunk 表 (= CONTEXT) │  │  concept_node 表 (新)  │      │      │
│  │  │                      │  │                        │      │      │
│  │  │ 同一张表存所有 chunk   │  │ CONCEPT 节点           │      │      │
│  │  │ schema 扩展了:        │  │ id, name, concept_type  │      │      │
│  │  │  • source_lineage    │  │ neighbors, rank, q_vec │      │      │
│  │  │  • mentions          │  │ source_lineage          │      │      │
│  │  │  • mention_details   │  │ description_parts       │      │      │
│  │  │  • granularity       │  │ context_details         │      │      │
│  │  │  • depth/path/parent │  │                        │      │      │
│  │  │                      │  │ COMMUNITY 节点复用      │      │      │
│  │  │                      │  │ knowledge_graph_kwd=    │      │      │
│  │  │                      │  │ "community_report"      │      │      │
│  │  └──────────────────────┘  └────────────────────────┘      │      │
│  └─────────────────────────────────────────────────────────────┘      │
│                                                                     │
│  查询层:                                                             │
│  ┌──────────────────────┐  ┌──────────────────────┐                 │
│  │ TreeGuidedQuery      │  │ GraphGuidedQuery     │                 │
│  │ 1. 结构过滤 chunk     │  │ 1. 向量搜索 CONCEPT   │                 │
│  │ 2. 通过 mentions      │  │ 2. 读取邻居/上下文     │                 │
│  │    获取关联 CONCEPT   │  │ 3. 局部子图 + PPR      │                 │
│  │ 3. 局部子图 + PPR     │  │ 4. 聚合到 chunk        │                 │
│  │ 4. 排序返回 chunk      │  │ 5. 排序返回 chunk      │                 │
│  └──────────┬──────────┘   └──────────┬───────────┘                 │
│             │                          │                            │
│             └──────────┬───────────────┘                            │
│                        ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Shared Propagation Engine                                   │    │
│  │  - 局部子图 PPR（< 500 节点）                                   │    │
│  │  - Query Embedding vs Chunk Embedding 相似度加权              │    │
│  │  - search_epochs 迭代重排序（可选）                             │    │
│  │  - 完全复用 MGranRAG 的 PPR 算法逻辑                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  可视化/交互层:                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  GraphViewer & MultiHopQA                                   │   │
│  │  - 展示 CONCEPT 子图（实体+关系）                               │   │
│  │  - 用户点击实体展开邻居                                         │   │
│  │  - 多跳提问（"A→B→C 之间的关系？"）                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 与 RAGFlow 现有管线的集成

UKGS 不创建新的独立管线。它通过两个点与现有管线协同：

**集成点 1：索引管线（chunk 产出即 CONTEXT）**

```
现有管线（无修改）:
  File → Parser → TokenChunker → Tokenizer → insert_chunks

UKGS 集成后（GraphIndexer 自动运行）:
  File → Parser → TokenChunker → Tokenizer ──→ insert_chunks ✅ （写 chunk 表）
                                               │
                                               └──→ GraphIndexer（新步骤）
                                                      ├─ 从 chunks 抽取 CONCEPT
                                                      ├─ 写入 concept_node 表
                                                      └─ 回填 chunk.mentions
```

**调用链集成**：在 `do_handle_task()` 中，标准流程为 `build_chunks → insert_chunks`。在其后追加 UKGS 调用：

```python
# rag/svr/task_executor.py（修改点）

async def do_handle_task(task):
    ...
    chunks = await build_chunks(task, ...)   # Pipeline: File→Parser→Chunker→Tokenizer
    chunk_ids = await insert_chunks(...)      # 写入 chunk 表（不变）
    
    # === UKGS 新增：在 insert_chunks 之后增量抽取 ===
    if parser_config.get("ukgs", {}).get("enabled", False):
        await handle_ukgs_incremental(
            task=task,
            tenant_id=task_tenant_id,
            kb_id=task_dataset_id,
            doc_id=task_doc_id,
            chunks=chunks,      # ← 就是 CONTEXT 节点列表
        )
```

**集成点 2：查询管线（chunk 表检索 + 图谱增强）**

```
现有检索:
  retriever.search(query) → 只检索 chunk 表 → 返回 chunks

UKGS 增强检索:
  retriever.search(query) → 检索 chunk 表 + UKGS 图谱增强 → 返回 chunks
                                                             ↑ 新字段 mentions + source_lineage
```

**chunk = CONTEXT 带来的优势**：

| 维度 | 旧设计（独立的 context_node 表） | 新设计（chunk = CONTEXT） |
|------|-------------------------------|-------------------------|
| 存储 | 两张相似的表：chunk + context_node | 一张表：chunk（扩展字段） |
| 写入 | 需要两个流水线：parser 产 chunk + 独立预处理做 context | 零额外写入：chunk 即是 CONTEXT |
| 维护 | 两个 ES index 需要分别维护 mapping、shard、alias | 一个 index，schema 统一 |
| 删除 | 删除 document 需要级联清理 context_node | 按 doc_id 删除 chunk 即可，天然级联 |
| 向前兼容 | 旧数据只有 chunk，需要迁移脚本 | 旧 chunk 自动是 CONTEXT（字段缺失=空） |
| 学习成本 | 用户需要理解"chunk"和"context_node"两个概念 | 只有一个概念：chunk |

### 2.3 不会产生的成本

```
"chunk = CONTEXT" 是否带来性能损失？
  - 否。chunk 原有的全文索引/向量索引/位置索引完全不变。
  - 新增字段 source_lineage、mentions 只在 UKGS 查询时使用。
  - UKGS 未启用时，这些字段不存在/为空，无性能开销。

"chunk 的 granularity 字段是否与现有 chunk 概念冲突？"
  - 否。现有 chunk 已经是 passage/paragraph 级别。
  - granularity 字段在 UKGS 启用后才被 TreeGuidedQuery 用于结构过滤。
```

---

## 3. 统一存储 Schema

### 3.1 核心设计理念

> **世界上只有两种数据——"chunk"（文本块，即 CONTEXT）和"概念"（语义单元）。所有系统的图结构都统一为一个带属性的二分图，chunk 就是其中的"上下文节点"。**

chunk 是图的第一类节点。边不单独建表，以内联数组的形式存储在 chunk 和概念节点中。

### 3.2 chunk 表（= CONTEXT 节点）

chunk 是 RAGFlow 已有的核心数据模型。UKGS 不创建新表，而是在现有 chunk schema 上**增量追加字段**。

```
索引: chunk（RAGFlow 原有索引，名称不变）
主键: id (string)

现有字段（完全保留，查询/写入逻辑不变）:
┌──────────────────────┬──────────┬────────────────────────────────┐
│ 字段                  │ 类型      │ 说明                           │
├──────────────────────┼──────────┼────────────────────────────────┤
│ id (主键)            │ keyword  │ chunk_id                       │
│ kb_id                │ keyword  │ 知识库 ID                      │
│ doc_id               │ keyword  │ 所属文档 ID                    │
│ content_with_weight  │ text     │ chunk 文本内容（CONTEXT 内容）   │
│ docnm_kwd            │ keyword  │ 文档名称                        │
│ title_tks            │ text     │ 标题 token                     │
│ content_ltks         │ text     │ 分词后的内容（关键词检索）        │
│ content_sm_ltks      │ text     │ 细粒度 token                   │
│ q_*_vec              │ float[]  │ 向量嵌入（dim=configurable）     │
│ chunk_order_int      │ integer  │ 块序号                         │
│ page_num_int         │ integer  │ 页码                           │
│ position_int         │ integer  │ 位置                           │
│ create_timestamp_flt │ float    │ 创建时间                        │
│ knowledge_graph_kwd  │ keyword  │ 标记旧版 GraphRAG 类型          │
└──────────────────────┴──────────┴────────────────────────────────┘

新增字段（UKGS 启用后追加，使 chunk = CONTEXT）:
┌──────────────────────┬──────────┬────────────────────────────────┐
│ 字段                  │ 类型      │ 说明                           │
├──────────────────────┼──────────┼────────────────────────────────┤
│ === 节点类型 ===      │          │                                │
│ node_type            │ keyword  │ 固定值: "context"（标识 UKGS）  │
│                      │          │                                │
│ === 层次位置 ===      │          │                                │
│ granularity          │ keyword  │ passage/paragraph/sentence     │
│                      │          │ /table/image                   │
│ depth                │ integer  │ 结构树深度（0=doc, 1=section)  │
│ parent_id            │ keyword  │ 父 chunk ID（可选）             │
│ path                 │ keyword[]│ 标题路径                        │
│                      │          │ ["Chapter 1", "背景介绍"]      │
│                      │          │                                │
│ === 内联边 ==========  │          │                                │
│ mentions             │ keyword[]│ 本 chunk 中提及的概念名称        │
│                      │          │ ["Nike", "NBA"]                │
│ mention_details      │ nested   │ 详细提及信息                    │
│  ├.concept_id        │ keyword  │ CONCEPT 节点 ID                │
│  ├.name              │ keyword  │ 提及文本                        │
│  └.pos_start/_end    │ integer  │ 在 content 中的字符位置         │
│                      │          │                                │
│ === Lineage SET ===  │          │                                │
│ source_lineage       │ nested   │ 文档级溯源                     │
│  ├.document_id       │ keyword  │ 来源文档 ID                    │
│  ├.parse_version     │ keyword  │ 解析版本号                     │
│  └.parent_block_ids  │ keyword[]│ 上游区块 ID                    │
└──────────────────────┴──────────┴────────────────────────────────┘
```

**示例 chunk（UKGS 启用后）**：

```json
{
  "id": "chunk_s10042",
  "kb_id": "kb_123",
  "doc_id": "doc_007",
  "docnm_kwd": "annual_report_2025.pdf",
  "content_with_weight": "Nike sponsors NBA athletes worldwide.",
  "chunk_order_int": 42,
  "page_num_int": 3,
  "q_1024_vec": [0.1, 0.2, 0.3, ...],

  "node_type": "context",
  "granularity": "sentence",
  "depth": 3,
  "parent_id": "chunk_p10041",
  "path": ["Chapter 1", "背景介绍", "1.1 公司概况"],
  "mentions": ["Nike", "NBA"],
  "mention_details": [
    {"concept_id": "con_Nike", "name": "Nike", "pos_start": 0, "pos_end": 4},
    {"concept_id": "con_NBA", "name": "NBA", "pos_start": 13, "pos_end": 16}
  ],
  "source_lineage": [
    {"document_id": "doc_007", "parse_version": "v1", "parent_block_ids": []}
  ]
}
```

**索引策略**（新增字段的索引，原有字段索引不变）：

| 字段 | 索引类型 | 用途 |
|------|---------|------|
| `node_type` | keyword | 区分 UKGS 启用的 chunk 与旧 chunk |
| `granularity` | keyword | 粒度过滤（如只查 section 级 chunk） |
| `depth` | integer range | 深度范围过滤 |
| `parent_id` | keyword | 子树检索（TOC 导航） |
| `path` | keyword (multi) | 标题路径匹配 |
| `mentions` | keyword (multi) | chunk↔concept 双向搜索 |
| `source_lineage.document_id` | keyword | 文档级精确更新 |

### 3.3 CONCEPT 节点（新建索引）

CONCEPT 节点是语义单元。一个概念可以是命名实体（Nike）、短语（"market share"）、术语（"GDPR"）、或主题（"data augmentation"）。

```
索引: concept_node（新索引，与 chunk 索引并列）
主键: id (string)

字段:
┌──────────────────────┬──────────┬────────────────────────────────┐
│ 字段                  │ 类型     │ 说明                            │
├──────────────────────┼──────────┼────────────────────────────────┤
│ id (主键)             │ keyword  │ 唯一 ID: con_{hashed_name}     │
│ kb_id                │ keyword  │ 知识库 ID                      │
│ node_type            │ keyword  │ 固定值: "concept"              │
│                      │          │                                │
│ === 名称与别名 ===     │          │                                │
│ name                 │ keyword  │ 概念名称（规范化）               │
│ canonical            │ keyword  │ 去重后的标准名（可选）           │
│                      │          │ "NIKE", "Nike Inc." → "Nike"  │
│ aliases              │ keyword[]│ 所有变体名称                   │
│                      │          │                                │
│ === 语义类型 ===      │          │                                │
│ concept_type         │ keyword  │ LLM: ORG/PERSON/LOC/PRODUCT    │
│                      │          │ spaCy: PER/ORG/GPE/PHRASE      │
│                      │          │                                │
│ === 描述 ===          │          │                               │
│ description          │ text     │ LLM 压缩的统一描述              │
│ description_parts    │ nested   │ 按文档切片的原始描述            │
│  ├.document_id       │ keyword  │ 来源文档 ID                    │
│  ├.parse_version     │ keyword  │ 解析版本                       │
│  └.text              │ text     │ 该文档中涉及本概念的原始文本     │
│                      │          │                                │
│ === 关联的 chunk ===  │          │                                │
│ source_chunks        │ keyword[]│ 关联的 chunk ID 列表            │
│ chunk_details        │ nested   │ 详细上下文信息                 │
│  ├.chunk_id          │ keyword  │ chunk ID                       │
│  ├.text              │ text     │ chunk 文本片段                 │
│  └.weight            │ float    │ 重要性权重                     │
│                      │          │                                │
│ === 概念间关系边 ===  │          │                                │
│ neighbors            │ keyword[]│ 邻接 CONCEPT 节点 ID 列表      │
│ neighbor_details     │ nested   │ 详细边信息                     │
│  ├.neighbor          │ keyword  │ 邻接概念节点 ID                │
│  ├.type              │ keyword  │ "semantic" | "co_occur"       │
│  ├.name              │ keyword  │ LLM: "sponsors"; spaCy: ""    │
│  └.weight            │ float    │ 边权重                         │
│                      │          │                                │
│ === 图属性 ===       │          │                                │
│ rank                 │ float    │ PageRank/重要性分数             │
│ degree               │ integer  │ 度                             │
│                      │          │                                │
│ === 社区（可选）===   │          │                                │
│ community_id         │ keyword  │ 所属社区 ID（仅 LLM 模式）      │
│                      │          │                                │
│ === Lineage SET ===  │          │                                │
│ source_lineage       │ nested   │ 文档级溯源                      │
│  ├.document_id       │ keyword  │ 来源文档 ID                    │
│  ├.parse_version     │ keyword  │ 解析版本                       │
│  └.chunk_ids         │ keyword[]│ 该概念在其中出现的 chunk ID     │
│                      │          │                                │
│ === Embedding ===    │          │                                │
│ q_vec                │ float[]  │ 概念向量嵌入                   │
└──────────────────────┴──────────┴────────────────────────────────┘
```

**示例文档（LLM 模式）**：

```json
{
  "id": "con_Nike",
  "kb_id": "kb_123",
  "node_type": "concept",
  "name": "Nike",
  "canonical": "Nike Inc.",
  "aliases": ["NIKE", "Nike, Inc.", "Nike Inc."],
  "concept_type": "ORG",
  "description": "Nike is an American sportswear corporation headquartered in Beaverton, Oregon.",
  "description_parts": [
    {"document_id": "doc_007", "parse_version": "v1", "text": "Nike sponsors NBA."},
    {"document_id": "doc_012", "parse_version": "v1", "text": "Nike Inc. is the largest shoe supplier."}
  ],
  "source_chunks": ["chunk_s10042", "chunk_s20015"],
  "chunk_details": [
    {"chunk_id": "chunk_s10042", "text": "Nike sponsors NBA.", "weight": 0.9},
    {"chunk_id": "chunk_s20015", "text": "Nike Inc. is the largest...", "weight": 0.7}
  ],
  "neighbors": ["con_NBA", "con_Adidas", "con_LeBron_James"],
  "neighbor_details": [
    {"neighbor": "con_NBA", "type": "semantic", "name": "sponsors", "weight": 0.9},
    {"neighbor": "con_Adidas", "type": "semantic", "name": "competes_with", "weight": 0.6},
    {"neighbor": "con_LeBron_James", "type": "semantic", "name": "endorses", "weight": 0.8}
  ],
  "rank": 12.5,
  "degree": 42,
  "community_id": "comm_001",
  "source_lineage": [
    {"document_id": "doc_007", "parse_version": "v1", "chunk_ids": ["chunk_s10042"]}
  ],
  "q_vec": [0.1, 0.2, 0.3, ...]
}
```

**示例文档（spaCy 模式）**：

```json
{
  "id": "con_Nike",
  "kb_id": "kb_123",
  "node_type": "concept",
  "name": "Nike",
  "canonical": "",
  "aliases": [],
  "concept_type": "ORG",
  "description": "",
  "description_parts": [],
  "source_chunks": ["chunk_s10042", "chunk_s20015"],
  "chunk_details": [
    {"chunk_id": "chunk_s10042", "text": "Nike sponsors NBA.", "weight": 1.0},
    {"chunk_id": "chunk_s20015", "text": "Nike Inc. is the largest...", "weight": 1.0}
  ],
  "neighbors": ["con_NBA", "con_Adidas"],
  "neighbor_details": [
    {"neighbor": "con_NBA", "type": "co_occur", "name": "", "weight": 0.5},
    {"neighbor": "con_Adidas", "type": "co_occur", "name": "", "weight": 0.3}
  ],
  "rank": 5.0,
  "degree": 2,
  "community_id": "",
  "source_lineage": [
    {"document_id": "doc_007", "parse_version": "v1", "chunk_ids": ["chunk_s10042"]}
  ],
  "q_vec": [0.1, 0.2, 0.3, ...]
}
```

**两种模式的字段差异**：

| 字段 | LLM 模式 | spaCy 模式 |
|------|---------|-----------|
| `canonical` / `aliases` | ✅ 去重后填入 | ❌ 留空 |
| `description` / `description_parts` | ✅ LLM 生成的描述 | ❌ 空数组 |
| `neighbor_details[].type` | `"semantic"` + `"co_occur"` | 只有 `"co_occur"` |
| `neighbor_details[].name` | `"sponsors"`（语义标签） | 空字符串 |
| `community_id` | ✅ 可选 | ❌ 空字符串 |
| `mention_details[].pos_start/pos_end` | ✅ 可输出 | ✅ 可输出 |

**索引策略**：

| 字段 | 索引类型 | 用途 |
|------|---------|------|
| `kb_id` | keyword | 知识库隔离 |
| `name` | keyword | 精确匹配 |
| `canonical` | keyword | 去重查询 |
| `aliases` | keyword (multi) | 别名搜索 |
| `concept_type` | keyword | 类型过滤 |
| `source_chunks` | keyword (multi) | chunk↔concept 双向检索 |
| `neighbors` | keyword (multi) | 邻居查询 |
| `community_id` | keyword | 社区过滤 |
| `source_lineage.document_id` | keyword | 文档级精确更新 |
| `q_vec` | dense_vector | 向量检索 |

### 3.4 COMMUNITY 节点（可选，仅 LLM 模式）

复用 RAGFlow 现有模式——以 chunk 形式存储，`knowledge_graph_kwd: "community_report"` 区分。

```
存储方式: chunk 表（非新索引）
knowledge_graph_kwd: "community_report"

字段:
┌──────────────────────┬──────────┬────────────────────────────────┐
│ 字段                  │ 类型      │ 说明                           │
├──────────────────────┼──────────┼────────────────────────────────┤
│ id                   │ keyword  │ 唯一 ID: comm_{uuid}           │
│ kb_id                │ keyword  │ 知识库 ID                      │
│ knowledge_graph_kwd  │ keyword  │ 固定值: "community_report"     │
│ title_tks            │ text     │ 社区标题                        │
│                      │          │ "Sports Apparel & Endorsements"│
│ content_with_weight  │ text     │ LLM 生成的社区摘要              │
│ content_ltks         │ text     │ 证据文本                       │
│ source_id            │ keyword[]│ 成员 CONCEPT 节点 ID 列表        │
│ weight_flt           │ float    │ 社区紧密程度                    │
└──────────────────────┴──────────┴────────────────────────────────┘
```

### 3.5 两张存储表的关系图

```
                             chunk 表 (= CONTEXT)
┌──────────────────────────────────────────────────────────────────┐
│  id, kb_id, doc_id, content_with_weight, q_1024_vec, ...         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ 新增字段:                                              │         │
│  │  - granularity, depth, path, parent_id（结构导航）      │         │
│  │  - mentions, mention_details（概念提及列表）             │         │
│  │  - source_lineage（文档级溯源）                          │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ COMMUNITY 复用 chunk 表:                              │         │
│  │ knowledge_graph_kwd="community_report"               │         │
│  │ title_tks=社区标题, content_with_weight=社区摘要       │         │
│  └──────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
                         │ mentions
                         │ (chunk.mentions ↔ concerpt.source_chunks)
                         ▼
           concept_node 表（新索引）
┌──────────────────────────────────────────────────────────────────┐
│  id, kb_id, name, concept_type, rank, q_vec                      │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ chunk_details (Concept → Chunk)                       │         │
│  │ chunk_id, text, weight                                │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ neighbor_details (Concept → Concept)                  │         │
│  │ type: semantic | co_occur, name, weight               │         │
│  └──────────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

**核心数据流**：

```
写入：
  Document → Parser + TokenChunker + Tokenizer → chunks (CONTEXT，含 embedding)
    → GraphIndexer: 从 chunks 中抽取概念 → CONCEPT nodes（含 neighbor_details）
    → 回填 chunk.mentions

查询：
  Query → ES 检索 chunk 或 concept → 读取内联边 →
  内存组装局部子图 → PPR → 排序返回 chunk
```

---

## 4. 向后兼容：从 RAGFlow GraphRAG 迁移

### 4.1 旧版 RAGFlow GraphRAG 存储格式

当前 RAGFlow 的图谱数据以 chunk 形式存在 ES/Infinity 中，通过 `knowledge_graph_kwd` 字段标记类型：

```python
# Entity chunk
{
    "id": "ent_xxx",
    "kb_id": "xxx",
    "knowledge_graph_kwd": "entity",
    "entity_kwd": "NIKE",
    "entity_type_kwd": "ORG",
    "source_id": ["chunk_key_001", "chunk_key_002"],
    "content_with_weight": json.dumps({
        "description": "Nike is an American...",
        "entity_type": "ORG",
        "source_id": ["chunk_key_001"]
    }),
    "q_1024_vec": [0.1, ...]
}

# Relation chunk
{
    "id": "rel_xxx",
    "kb_id": "xxx",
    "knowledge_graph_kwd": "relation",
    "from_entity_kwd": "NIKE",
    "to_entity_kwd": "NBA",
    "weight_int": 1,
    "source_id": ["chunk_key_001"],
    "content_with_weight": json.dumps({
        "description": "Nike sponsors NBA.",
        "source_id": ["chunk_key_001"]
    }),
    "q_1024_vec": [0.3, ...]
}

# Global graph chunk（可选）
{
    "id": "graph_xxx",
    "kb_id": "xxx",
    "knowledge_graph_kwd": "graph",
    "content_with_weight": json.dumps(nx.node_link_data(G))  # 整个图序列化
}
```

### 4.2 迁移原则

1. **不强制重建**：旧 KB 继续工作，新写入代码自动识别版本
2. **查询层透明**：`EntityChunkAdapter` 封装新旧格式读取差异
3. **懒迁移**：首次触发重建/增量时自动转换
4. **schema_version 标记**：`kb_meta.graph_schema_version = "1" | "2"`

### 4.3 版本标记机制

在知识库元数据中增加字段：

```python
# api/db/services/kb_service.py

class KBMeta:
    graph_schema_version: str  # "1" = 旧版 chunk 格式, "2" = Lineage SET
    # 首次创建 KB 时:
    #   新 KB → "2" (默认)
    #   存量 KB → 检测到 knowledge_graph_kwd="graph" 无 graph_schema_version → "1"
```

检测逻辑：

```python
async def detect_schema_version(tenant_id, kb_id):
    """
    检测存量 KB 的图谱 schema 版本
    """
    res = await es_search(
        index="chunk",
        query={
            "bool": {
                "must": [
                    {"term": {"kb_id": kb_id}},
                    {"term": {"knowledge_graph_kwd": "graph"}}
                ]
            }
        },
        size=1
    )

    if res.hits.total.value == 0:
        return "2"

    hit = res.hits.hits[0]
    if "_source" in hit and hit["_source"].get("graph_schema_version") == "2":
        return "2"

    return "1"
```

### 4.4 Schema 映射：旧 → 新

```
旧版 (v1)                             新版 (v2)
─────────────────                     ─────────────────

Entity chunk                          CONCEPT node (新索引 concept_node)
  entity_kwd        ─→  name
  entity_type_kwd   ─→  concept_type
  content_with_weight.description ─→  description
  source_id          ─→  source_chunks (chunk ID 列表)
  q_1024_vec         ─→  q_vec

  【无】              ─→  chunk_details（从 source_id 关联）
                         source_lineage（从 source_id 关联）

Relation chunk                        CONCEPT node.neighbor_details
  from_entity_kwd + to_entity_kwd ─→  neighbor_details[].neighbor
  content_with_weight.description ─→  neighbor_details[].name
  weight_int         ─→  neighbor_details[].weight

  【无】              ─→  neighbor_details[].type = "semantic"

Global graph chunk                   【降级为缓存，不再视为真相源】
  content_with_weight (nx JSON)  ─→  重建后删除，或标记 graph_role="cache"
```

### 4.5 迁移代码

#### 读取层的兼容封装

```python
# rag/ukgs/compat.py

class EntityChunkAdapter:
    """
    统一读取 v1 (旧 GraphRAG chunk) 和 v2 (chunk + concept_node)
    """

    @staticmethod
    def concept_from_old_chunk(es_hit: dict) -> dict:
        """将 v1 entity chunk 映射为 v2 concept 格式（只读视图）"""
        raw = json.loads(es_hit.get("content_with_weight", "{}"))

        return {
            "id": es_hit["id"],
            "kb_id": es_hit.get("kb_id"),
            "node_type": "concept",
            "name": es_hit.get("entity_kwd", ""),
            "canonical": "",
            "aliases": [],
            "concept_type": es_hit.get("entity_type_kwd", raw.get("entity_type", "UNKNOWN")),
            "description": raw.get("description", ""),
            "description_parts": [],
            "source_chunks": es_hit.get("source_id", []),
            "chunk_details": [],
            "neighbors": [],
            "neighbor_details": [],
            "rank": es_hit.get("rank_flt", 0.0),
            "degree": 0,
            "community_id": "",
            "source_lineage": [
                {"document_id": sid, "parse_version": "v1", "chunk_ids": []}
                for sid in es_hit.get("source_id", [])
            ],
            "q_vec": es_hit.get("q_1024_vec", [])
        }

    @staticmethod
    def read_concept(es_hit: dict) -> dict:
        """自动识别版本读取"""
        if es_hit.get("node_type") == "concept":
            return es_hit  # v2，直接返回
        if es_hit.get("knowledge_graph_kwd") == "entity":
            return EntityChunkAdapter.concept_from_old_chunk(es_hit)
        return es_hit
```

#### 写层的版本路由

```python
# rag/ukgs/index.py

async def merge_subgraph(tenant_id, kb_id, doc_id, subgraph, embedding_model, callback):
    """
    统一写入口：自动路由到 v1 或 v2 写入逻辑
    """
    schema_version = await detect_schema_version(tenant_id, kb_id)

    if schema_version == "2":
        return await merge_subgraph_v2(...)
    else:
        return await merge_subgraph_v1(...)
```

### 4.6 存量 KB 的懒迁移

```python
async def lazy_migrate_kb_to_v2(tenant_id, kb_id):
    """
    懒迁移：遍历所有旧 entity/relation chunk，改写为 v2 格式
    不删除旧数据，只新增概念节点
    """
    await set_migration_status(tenant_id, kb_id, "migrating")

    # 迁移所有 entity chunk → concept_node 表
    old_entities = await es_search(
        index="chunk",
        query={"bool": {"must": [
            {"term": {"kb_id": kb_id}},
            {"term": {"knowledge_graph_kwd": "entity"}}
        ]}},
        size=10000
    )

    for hit in old_entities:
        concept = EntityChunkAdapter.concept_from_old_chunk(hit["_source"])
        await write_concept_node(tenant_id, kb_id, concept)

    # 迁移所有 relation chunk → 更新 CONCEPT 的 neighbor_details
    old_relations = await es_search(
        index="chunk",
        query={"bool": {"must": [
            {"term": {"kb_id": kb_id}},
            {"term": {"knowledge_graph_kwd": "relation"}}
        ]}},
        size=10000
    )

    for hit in old_relations:
        rel = hit["_source"]
        await update_concept_neighbor(...)

    # 标记旧 graph chunk 为缓存
    await mark_old_graph_as_cache(tenant_id, kb_id)

    # 更新 KB 版本
    await set_schema_version(tenant_id, kb_id, "2")
```

### 4.7 迁移状态机

```
                ┌──────────┐
                │  无图谱   │  (新 KB，直接 v2)
                └────┬─────┘
                     │ create_kb
                     ▼
            ┌─────────────────┐
            │ schema_version  │
            │   = "2"         │  (新 KB 默认 v2)
            └────────┬────────┘
                     │
                     ▼
          ┌─────────────────────┐
          │    正常写入/查询      │
          └─────────────────────┘

存量 KB:
  ┌─────────────────┐
  │  schema_version  │  (检测到旧 chunk)
  │   = "1" (隐式)  │
  └────────┬────────┘
           │
     ┌─────┴──────┐
     │            │
     ▼            ▼
  ┌────────┐  ┌──────────┐
  │ 继续旧  │  │懒迁移→v2 │
  │ 逻辑    │  │          │
  └────────┘  └────┬─────┘
                   │
                   ▼
          ┌─────────────────┐
          │ schema_version  │
          │   = "2"         │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────────┐
          │  正常 v2 写入/查询   │
          └─────────────────────┘
```

---

## 5. 索引管线

### 5.1 公共前处理（现有 RAGFlow Pipeline）

UKGS 不重复实现文档解析/chunk 生成/embedding 计算。这些全部由 RAGFlow 现有流水线完成：

```
RAGFlow 流水线 (Pipeline DSL 定义):
  File → Parser → TokenChunker → Tokenizer

输出: chunks[] （列表，每个元素是 dict）

chunk 数据结构（已包含）:
  id, kb_id, doc_id,
  content_with_weight (文本),
  q_1024_vec (embedding),
  chunk_order_int (序号),
  page_num_int (页码),
  ...
```

**UKGS 对前置步骤的唯一要求**：chunk 中需要包含所在文档的 `doc_id`，以便后续增量索引识别归属。这是现有 pipeline 已经满足的。

### 5.2 GraphIndexer（新 Component）

GraphIndexer 是一个新的 Pipeline Component，注册在 Tokenizer 之后、不改变现有组件的输出。它在 `do_handle_task()` 中的 `insert_chunks()` 之后调用，读取已经 embedding 完毕的 chunks，抽取概念并写入 CONCEPT 表。

```python
# rag/ukgs/indexer.py

class GraphIndexer:
    """
    在 Parser + TokenChunker + Tokenizer 完成后运行。
    接收已完成 embedding 的 chunks 列表，输出 concepts。

    chunk = CONTEXT：chunk 的 id/content/embedding 天然就是 CONTEXT 的属性。
    """

    def __init__(self, index_mode: str, llm_bundle=None, embedder=None, spacy_model=None):
        self.mode = index_mode
        if index_mode == "llm":
            self.indexer = LLMIndexer(llm_bundle, embedder)
        elif index_mode == "spacy":
            self.indexer = SpacyIndexer(spacy_model or "en_core_web_trf", embedder)
        else:
            raise ValueError(f"Unknown index_mode: {index_mode}")

    async def extract_from_chunks(self, kb_id: str, doc_id: str,
                                   chunks: List[dict]) -> List[dict]:
        """
        从已完成的 chunks 中抽取概念。

        参数:
        - chunks: Tokenizer 输出的 chunks（已有 id, content, embedding）

        返回:
        - concepts: 抽取的 CONCEPT 节点列表
        """
        return await self.indexer.extract_concepts(kb_id, chunks)
```

### 5.3 LLM Indexer

```python
class LLMIndexer:
    """
    使用 LLM 抽取高精度知识图谱

    chunk = CONTEXT：直接使用 chunk.content_with_weight 作为 LLM 输入。
    """

    async def extract_concepts(self, kb_id: str, chunks: List[dict]):
        """
        从 chunk 集合中抽取 concept

        Pipeline:
        1. 对每个 chunk 调用 LLM 抽取实体和关系
        2. 聚合去重同实体
        3. 构建 semantic 类型的邻居边
        4. 计算 embedding
        5. 社区检测（可选）
        6. 写入 concept_node 表
        7. 回填 chunk.mentions
        """

        # === Step 1: LLM 批量抽取 ===
        entity_rels_batch = []
        for batch in self.batch_chunks(chunks, batch_size=10):
            # 每个元素是列表: [chunk.content_with_weight, ...]
            texts = [c["content_with_weight"] for c in batch]
            result = await self.llm.extract_entity_relation(
                texts,
                # 输出:
                # {
                #   "entities": [{"name": "...", "type": "ORG", "description": "..."}],
                #   "relations": [{"src": "...", "tgt": "...", "name": "sponsors"}]
                # }
            )
            entity_rels_batch.append(result)

        # === Step 2: 实体去重 ===
        if self.config.enable_resolution:
            merged_entities = self.resolve_entities(entity_rels_batch)
        else:
            merged_entities = self.aggregate_entities(entity_rels_batch)

        # === Step 3: 构建 CONCEPT 节点 ===
        concepts = []
        for ent in merged_entities:
            concept = ConceptNode(
                name=ent["name"],
                concept_type=ent["entity_type"],
                description=ent["description"],
                chunk_details=self.build_chunk_details(ent, chunks),
                neighbor_details=self.build_neighbor_details(ent, entity_rels_batch),
                source_lineage=[{
                    "document_id": doc_id,
                    "parse_version": "v1",
                    "chunk_ids": [c["id"] for c in chunks if ent["name"] in c.get("content_with_weight", "")]
                }]
            )
            concepts.append(concept)

        # === Step 4: 概念 embedding ===
        for concept in concepts:
            concept.q_vec = await self.embedder.encode(
                f"{concept.name}: {concept.description}" if concept.description else concept.name
            )

        # === Step 5: 社区检测（可选） ===
        if self.config.enable_community:
            communities = self.detect_communities(concepts)
            for concept in concepts:
                concept.community_id = communities.get(concept.id, "")

        # === Step 6: 写入 concept_node 表 ===
        await self.write_concepts(concepts)

        # === Step 7: 回填 chunk.mentions ===
        await self.backfill_chunk_mentions(kb_id, concepts, chunks)
```

### 5.4 spaCy Indexer

```python
class SpacyIndexer:
    """
    使用 spaCy 构建轻量化知识图谱

    chunk = CONTEXT：直接在 chunk.content_with_weight 上运行 spaCy NER + POS 短语抽取。
    """

    def __init__(self, spacy_model="en_core_web_trf", embedder=None):
        self.nlp = spacy.load(spacy_model)
        self.embedder = embedder

    async def extract_concepts(self, kb_id: str, chunks: List[dict]):
        """
        Pipeline:
        1. 对每个 chunk.content_with_weight 做 spaCy NER + POS 短语
        2. 聚合去重命名实体和短语
        3. 构建共现边
        4. 计算 embedding
        5. 写入 concept_node 表
        6. 回填 chunk.mentions
        """

        named_entities = defaultdict(list)
        pos_phrases = defaultdict(list)

        for chunk in chunks:
            text = chunk["content_with_weight"]
            doc = self.nlp(text)
            chunk_id = chunk["id"]

            # 1a. NER
            for ent in doc.ents:
                key = (ent.text.strip(), ent.label_)
                named_entities[key].append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "pos_start": ent.start_char,
                    "pos_end": ent.end_char,
                    "weight": 1.0
                })

            # 1b. POS 短语抽取（MGranRAG 风格栈式合并）
            phrases = self.extract_phrases(doc)
            for phrase in phrases:
                pos_phrases[phrase.text].append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "weight": 1.0
                })

        # === Step 2-6：与 LLM Indexer 类似，见下文 ===
        # ...

    def extract_phrases(self, doc):
        """
        MGranRAG 风格的 POS 短语抽取（栈式合并连续的 PROPN/NOUN/NUM）
        """
        phrases = []
        stack = []

        for token in doc:
            if token.pos_ in {"PROPN", "NOUN", "NUM"}:
                stack.append(token)
            else:
                if len(stack) >= 1:
                    phrase_text = " ".join(t.text for t in stack)
                    phrases.append(phrase_text)
                    stack = []

        if len(stack) >= 1:
            phrase_text = " ".join(t.text for t in stack)
            phrases.append(phrase_text)

        return phrases

    def build_cooccurrence_edges(self, concepts, chunks):
        """
        构建共现边：同一 chunk 中的概念两两连边，weight 随共现次数递增
        """
        concept_chunks = defaultdict(set)
        for c in concepts:
            concept_chunks[c.id] = {cd["chunk_id"] for cd in c.chunk_details}

        edge_weights = defaultdict(int)

        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                shared = concept_chunks[concepts[i].id] & concept_chunks[concepts[j].id]
                if shared:
                    edge_weights[(concepts[i].id, concepts[j].id)] = len(shared)

        for (cid1, cid2), weight in edge_weights.items():
            conc1 = next(c for c in concepts if c.id == cid1)
            conc2 = next(c for c in concepts if c.id == cid2)

            conc1.add_neighbor(
                neighbor_id=cid2,
                type="co_occur",
                name="",
                weight=min(weight / 10.0, 1.0)
            )
            conc2.add_neighbor(
                neighbor_id=cid1,
                type="co_occur",
                name="",
                weight=min(weight / 10.0, 1.0)
            )
```

### 5.5 两种 Indexer 对比总结

| 维度 | LLM Indexer | spaCy Indexer |
|------|------------|--------------|
| **成本** | 高（token 消耗） | 几乎为零 |
| **速度** | 慢（受限于 LLM API） | 快（纯本地 NLP） |
| **命名实体** | 准确、覆盖全面 | 可接受（spaCy transformer） |
| **短语抽取** | 无 | ✅ MGranRAG 风格栈式合并 |
| **关系边** | ✅ 语义标签（"sponsors"） | ⚠️ 仅有 co_occur |
| **实体描述** | ✅ LLM 生成 | ❌ 无 |
| **社区检测** | ✅ 可选 | ❌ |
| **CONCEPT 数量** | 少（去重后） | 多（原始提及 + 短语） |
| **适用场景** | 专业知识库、高精度问答 | 快速原型、资源受限 |

---

### 5.6 增量索引机制

#### 5.6.1 动机

非增量方案下，每次新增/更新/删除文档都需要**全量重跑**整个知识库的图谱：

```
读取 KB 所有 chunk → 全部重跑 NER/LLM → 删除旧 CONCEPT → 写入新 CONCEPT
```

当知识库增长到百万级 chunk 时，这种方式完全不可行。增量索引的核心目标：

> **一次文档变更，只影响与该文档相关的 chunk 和 CONCEPT 节点，不触碰其他文档的数据。**

因为 chunk = CONTEXT，chunk 表天然支持按 `doc_id` 删除和查询。增量索引在 chunk 层级的操作与现有 RAGFlow 的文档级删除/更新完全一致。

#### 5.6.2 三种文档事件的处理策略

```
事件类型:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  ADD doc    │  │ UPDATE doc  │  │ DELETE doc  │
└──────┬──────┘  └──────┬──────┘  └──────┬──────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────┐
│            Delta Calculator                  │
│                                             │
│  ADD:     Δ_remove = ∅,      Δ_add = doc    │
│  DELETE:  Δ_remove = doc,    Δ_add = ∅      │
│  UPDATE:  Δ_remove = old,    Δ_add = new    │
└─────────────────────┬───────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
  ┌──────────────┐       ┌──────────────┐
  │ Delete Phase │       │  Add Phase   │
  │ 精确删除     │       │ 精确添加     │
  │ doc_id 的    │       │ doc_id 的    │
  │ 所有贡献     │       │ 所有贡献     │
  └──────────────┘       └──────────────┘
```

#### 5.6.3 Delete Phase：精确删除

chunk 天然按 `doc_id` 分组，删除文档的 chunk 就是删除 CONTEXT。

**步骤 A：删除该文档的 chunk（已有 RAGFlow 逻辑，无需新增）**

```python
# 已有方法: chunk_service.delete_chunks_by_doc_id(doc_id)
# 执行 ES/Infinity delete_by_query on doc_id
```

**步骤 B：从 CONCEPT 节点中移除该文档的贡献**

```python
async def remove_doc_from_concepts(tenant_id, kb_id, doc_id):
    """
    从所有 CONCEPT 节点中移除指定 doc_id 的贡献

    利用 source_lineage.document_id 精确移除
    """
    # 找到该 doc 的所有 chunk_id
    doc_chunk_ids = await get_chunk_ids_by_doc(kb_id, doc_id)

    # Painless Script 原地更新
    script = """
    ctx._source.source_lineage.removeIf(l -> l.document_id == params.doc_id);
    if (ctx._source.description_parts != null) {
        ctx._source.description_parts.removeIf(p -> p.document_id == params.doc_id);
    }
    if (ctx._source.chunk_details != null) {
        ctx._source.chunk_details.removeIf(
            d -> params.doc_chunk_ids.contains(d.chunk_id)
        );
    }
    """

    await es_update_by_query(
        index="concept_node",
        query={
            "bool": {
                "must": [
                    {"term": {"kb_id": kb_id}},
                    {"nested": {
                        "path": "source_lineage",
                        "query": {"term": {"source_lineage.document_id": doc_id}}
                    }}
                ]
            }
        },
        script=script,
        params={"doc_id": doc_id, "doc_chunk_ids": doc_chunk_ids}
    )

    # 清理孤儿 CONCEPT
    await cleanup_orphan_concepts(kb_id)
```

**步骤 C：从 CONCEPT 的 neighbor_details 中移除失效边**

```python
async def recalc_neighbor_edges(tenant_id, kb_id, affected_concept_ids):
    """
    对受影响的 CONCEPT 重新计算 neighbor_details
    """
    concepts = await batch_get_concepts(kb_id, affected_concept_ids)
    concept_to_chunks = {
        c.id: {cd["chunk_id"] for cd in c.chunk_details}
        for c in concepts
    }

    for c in concepts:
        new_neighbors = []
        for nb in c.neighbor_details:
            if nb["type"] == "semantic":
                new_neighbors.append(nb)
            elif nb["type"] == "co_occur":
                nb_chunks = concept_to_chunks.get(nb["neighbor"], set())
                shared = concept_to_chunks[c.id] & nb_chunks
                if shared:
                    nb["weight"] = min(len(shared) / 10.0, 1.0)
                    new_neighbors.append(nb)

        c.neighbor_details = new_neighbors
        c.neighbors = [nb["neighbor"] for nb in new_neighbors]

    await batch_write_concepts(concepts)
```

#### 5.6.4 Add Phase：精确添加

```python
async def add_doc_to_graph(tenant_id, kb_id, doc_id, chunks, indexer):
    """
    增量添加一个文档的图谱贡献

    参数:
    - chunks: 该文档的已完成 embedding 的 chunk 列表（= CONTEXT 节点）
    - indexer: LLMIndexer 或 SpacyIndexer 实例
    """

    # === Step 1: chunk 已由 insert_chunks 写入，无需重复操作 ===

    # === Step 2: 从新增 chunk 中抽取概念 ===
    new_concepts = await indexer.extract_concepts_from_chunks(kb_id, chunks)

    # 2b. 对每个新概念，判断是"创建"还是"追加"（Lineage SET）
    for nc in new_concepts:
        existing = await get_concept_by_name(kb_id, nc.name)

        if existing is None:
            await write_concept_node(nc)
        else:
            await append_to_concept(
                concept=existing,
                new_parts=nc.description_parts,
                new_lineage=nc.source_lineage,
                new_chunk_details=nc.chunk_details,
            )

    # === Step 3: 更新 chunk.mentions ===
    await indexer.backfill_chunk_mentions(kb_id, new_concepts, chunks)

    # === Step 4: 更新概念间共现边 ===
    await build_incremental_edges(kb_id, new_concepts, chunks)
```

**`append_to_concept` 的精确实现（Lineage SET 核心）**：

```python
async def append_to_concept(concept, new_parts, new_lineage, new_chunk_details):
    """
    Lineage SET 精确追加
    用 document_id 做 UPSERT，避免重复写入
    """
    script = """
    // 追加 description_parts（按 document_id UPSERT）
    for (part in params.new_parts) {
        int idx = ctx._source.description_parts.findIndex(
            p -> p.document_id == part.document_id
        );
        if (idx >= 0) {
            ctx._source.description_parts[idx] = part;
        } else {
            ctx._source.description_parts.add(part);
        }
    }

    // 追加 source_lineage
    for (nl in params.new_lineage) {
        int idx = ctx._source.source_lineage.findIndex(
            l -> l.document_id == nl.document_id
        );
        if (idx >= 0) {
            ctx._source.source_lineage[idx] = nl;
        } else {
            ctx._source.source_lineage.add(nl);
        }
    }

    // 追加 chunk_details（仅新增不重复的 chunk_id）
    for (cd in params.new_chunk_details) {
        if (!ctx._source.chunk_details.stream()
                .anyMatch(d -> d.chunk_id == cd.chunk_id)) {
            ctx._source.chunk_details.add(cd);
        }
    }
    """

    await es_update(
        index="concept_node",
        id=concept.id,
        script=script,
        params={
            "new_parts": new_parts,
            "new_lineage": new_lineage,
            "new_chunk_details": new_chunk_details
        }
    )
```

#### 5.6.5 完整增量管线编排

```python
class IncrementalIndexPipeline:
    """
    增量索引管线编排

    Update = DELETE(old) + ADD(new)，以 doc_id 为原子单位
    """

    async def process_document_change(
        self,
        tenant_id: str,
        kb_id: str,
        doc_id: str,
        change_type: str,                # "add" | "update" | "delete"
        new_chunks: Optional[List],      # update/add 时的新 chunk 列表
        indexer: Union[LLMIndexer, SpacyIndexer],
    ):
        # === Phase 1: Delete Phase ===
        if change_type in ("delete", "update"):
            affected = await remove_doc_from_concepts(tenant_id, kb_id, doc_id)
            if affected:
                await recalc_neighbor_edges(tenant_id, kb_id, affected)

            if change_type == "delete":
                await self.post_delete_cleanup(kb_id, doc_id)
                return

        # === Phase 2: Add Phase ===
        if change_type in ("add", "update") and new_chunks:
            await add_doc_to_graph(tenant_id, kb_id, doc_id, new_chunks, indexer)
```

#### 5.6.6 次要属性的增量维护

| 属性 | 增量策略 | 说明 |
|------|---------|------|
| **`rank` (概念重要性)** | **Stale-mark + 后台异步重算** | 每次增删文档后标记 rank 为 stale；后台定时任务全量重算。查询时降级使用 rank=1.0 |
| **`chunk_details[].weight`** | **立即计算** | 基于该 chunk 在该概念的全部 chunk 中的 TF 比例 |
| **`community_id`** | **Stale-mark + 定时全量重建** | Leiden 等社区检测算法不适合增量 |
| **co_occur 边 weight** | **立即增量** | weight = min(new_shared_count / 10, 1.0) |
| **embedding** | **立即计算** | 概念 q_vec 每次新增/更新时用 name+description 重新编码 |
| **chunk.mentions** | **立即回填** | 新增 chunk 时写入 mentions，删除时整 chunk 删除 |

#### 5.6.7 孤儿清理

```python
async def cleanup_orphan_concepts(kb_id):
    """
    清理 source_lineage 和 chunk_details 都为空的概念
    """
    await es_delete_by_query(
        index="concept_node",
        query={
            "bool": {
                "must": [
                    {"term": {"kb_id": kb_id}},
                    {"bool": {"must_not": {"nested": {
                        "path": "source_lineage",
                        "query": {"exists": {"field": "source_lineage.document_id"}}
                    }}}},
                    {"bool": {"must_not": {"nested": {
                        "path": "chunk_details",
                        "query": {"exists": {"field": "chunk_details.chunk_id"}}
                    }}}}
                ]
            }
        }
    )
```

#### 5.6.8 幂等性保证

增量索引必须是幂等的。由于 `append_to_concept` 使用 `document_id` 做 UPSERT，重复调用 delete+add 是安全的：

- 重复 DELETE：`removeIf` 对不存在的 `document_id` 不做任何变更
- 重复 ADD：`document_id` 匹配时执行替换（不是追加），不会产生重复数据

#### 5.6.9 增量 vs 全量重建的决策矩阵

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 首次建库 | **全量** | 没有存量数据需要维护 |
| 新增 1 篇文档 | **增量** | 影响范围限单文档 |
| 更新 1 篇文档 | **增量** | DELETE(old chunk) + ADD(new) |
| 批量导入 1000 篇文档 | **增量（合并 batch）** | 合并为一次 chunk 写入 + 一次性 edge 重算 |
| 用户手动触发"更新社区" | **全量社区** | 社区检测不支持增量 |
| 数据不一致 | **全量重建** | 增量依赖"当前状态正确" |
| 长期运行后 rank 偏移 | **全量 rank（后台定时）** | 增量 rank 难保证全局归一化 |
| 删除 1 篇文档 | **增量** | 精确删除，不影响其他文档 |

---

## 6. 查询管线

### 6.1 入口路由

```python
class UKGSQueryRouter:
    """
    根据 query_mode 分派到不同的查询引擎
    """

    async def query(self, request):
        mode = request.get("query_mode", "auto")

        if mode == "auto":
            mode = await self.task_planner.classify(request["query"])

        if mode == "tree_guided":
            engine = TreeGuidedQuery(embedder=self.embedder, index_mode=self.index_mode)
        elif mode == "graph_guided":
            engine = GraphGuidedQuery(embedder=self.embedder, index_mode=self.index_mode)
        elif mode == "graph_only":
            engine = GraphOnlyQuery(embedder=self.embedder)
        else:
            raise ValueError(f"Unknown query_mode: {mode}")

        return await engine.query(request)
```

### 6.2 TreeGuidedQuery

```
适用: query_mode = "tree_guided"
来源: BookRAG 风格
核心思想: 先定位文档结构位置（利用 chunk 的 granularity/path），再搜索概念

流程:
  Query
   │
   ├─ Step 1: 结构过滤 chunk 表
   │    filters = [section="Methodology"]
   │    → ES term query on chunk.path + chunk.granularity
   │    → candidate_chunks (top-200)
   │
   ├─ Step 2: 从 query 提取种子概念
   │    index_mode="spacy" → spaCy NER + POS
   │    index_mode="llm"   → LLM 抽取
   │    → query_concept_names
   │
   ├─ Step 3: 匹配种子 CONCEPT
   │    在 candidate_chunks.mentions 范围内搜索
   │    ES query: name IN query_concept_names AND source_chunks IN candidate_chunk_ids
   │    → seed_concepts (top-10)
   │
   ├─ Step 4: 扩展邻居（读取 concept_node.neighbor_details）
   │    → all_concepts = seed + neighbors (top-50)
   │
   ├─ Step 5: 获取相关 chunk
   │    从 all_concepts.chunk_details 收集 chunk_id
   │    → related_chunks (top-50)
   │
   ├─ Step 6: 构建局部子图（chunk + concept 二分图）
   │    → local_subgraph
   │
   ├─ Step 7: PPR 传播
   │    personalization = query_embedding × chunk_embedding 相似度
   │    seed = seed_concepts
   │    alpha = 0.85
   │    → 所有节点的 PPR score
   │
   ├─ Step 8: 聚合到 chunk
   │    chunk_score = sum(concept_ppr × edge_weight)
   │
   └─ Step 9: 排序返回 top-k chunk
```

```python
class TreeGuidedQuery:
    def __init__(self, embedder, index_mode):
        self.embedder = embedder
        self.index_mode = index_mode

    async def query(self, request):
        kb_id = request["kb_id"]
        query_text = request["query"]
        filters = request.get("structure_filters", [])
        top_k = request.get("top_k", 10)
        ppr_alpha = request.get("ppr_alpha", 0.85)

        # Step 1: 结构过滤（在 chunk 表上）
        candidate_chunks = await self.structural_filter(
            kb_id=kb_id,
            filters=filters,
            top_n=200
        )
        if not candidate_chunks:
            return {"chunks": [], "graph": {}}

        # Step 2: 提取查询概念
        query_concept_names = await self.extract_query_concepts(query_text)
        if not query_concept_names:
            return await self.vector_fallback(query_text, kb_id, top_k)

        # Step 3: 匹配种子 CONCEPT
        seed_concepts = await self.match_concepts_in_scope(
            kb_id=kb_id,
            concept_names=query_concept_names,
            scope_chunk_ids=[c["id"] for c in candidate_chunks],
            top_k=10
        )
        if not seed_concepts:
            return await self.vector_fallback(query_text, kb_id, top_k)

        # Step 4: 扩展邻居
        all_concept_ids = set(c.id for c in seed_concepts)
        for c in seed_concepts:
            for nb in c.neighbor_details[:5]:
                all_concept_ids.add(nb["neighbor"])
        all_concepts = await self.batch_get_concepts(kb_id, list(all_concept_ids))

        # Step 5: 获取相关 chunk
        chunk_ids = set()
        for c in all_concepts:
            for cd in c.chunk_details[:5]:
                chunk_ids.add(cd["chunk_id"])
        candidate_ids = set(c["id"] for c in candidate_chunks)
        chunk_ids = chunk_ids & candidate_ids
        related_chunks = await self.batch_get_chunks(kb_id, list(chunk_ids))

        # Step 6-9: 组装子图 + PPR + 排序（复用 PropagationEngine）
        subgraph = self.assemble_subgraph(seed_concepts, all_concepts, related_chunks)
        ppr_scores = await PropagationEngine().ppr_rerank(
            subgraph=subgraph,
            seed_concept_ids=[c.id for c in seed_concepts],
            query_emb=await self.embedder.encode(query_text),
            chunk_embeddings={c["id"]: c["q_1024_vec"] for c in related_chunks},
            alpha=ppr_alpha
        )
        ranked = self.aggregate_to_chunks(ppr_scores, subgraph)
        return {"chunks": ranked[:top_k], "graph": subgraph}

    async def structural_filter(self, kb_id, filters, top_n):
        """在 chunk 表上做结构过滤"""
        must = [{"term": {"kb_id": kb_id}}]

        for f in filters:
            if f["type"] == "section":
                must.append({"wildcard": {"path": f"*{f['value']}*"}})
            elif f["type"] == "chapter":
                must.append({"term": {"depth": 1}})
                if "value" in f:
                    must.append({"match": {"content_with_weight": f["value"]}})
            elif f["type"] == "page":
                must.append({"range": {"page_num_int": {"lte": f["value"]}}})
            elif f["type"] == "type":
                must.append({"term": {"granularity": f["value"]}})

        results = await es_search(
            index="chunk",  # ← 直接在 chunk 表上查
            query={"bool": {"must": must}},
            size=top_n
        )
        return [hit["_source"] for hit in results]
```

### 6.3 GraphGuidedQuery

```
适用: query_mode = "graph_guided"
来源: GraphRAG / MGranRAG 风格
核心思想: 直接向量搜索概念图，然后扩展并传播

流程:
  Query
   │
   ├─ Step 1: 向量搜索 CONCEPT（concept_node.q_vec）
   │    → seed_concepts (top-10)
   │
   ├─ Step 2: 扩展邻居（concept_node.neighbor_details）
   │    → all_concepts (top-50)
   │
   ├─ Step 3: 获取相关 chunk
   │    从 all_concepts.chunk_details 收集 chunk_id
   │    → related_chunks (top-50)
   │
   ├─ Step 4-7:（与 TreeGuidedQuery 相同）
   │    局部子图 + PPR + 聚合 + 排序
   └─
```

```python
class GraphGuidedQuery:
    """
    语义优先的查询引擎，直接向量搜索 concept_node 表
    """

    async def query(self, request):
        kb_id = request["kb_id"]
        query_text = request["query"]
        top_k = request.get("top_k", 10)
        ppr_alpha = request.get("ppr_alpha", 0.85)

        # Step 1: 向量搜索 CONCEPT
        query_emb = await self.embedder.encode(query_text)
        seed_concepts = await self.vector_search_concepts(
            kb_id=kb_id,
            query_emb=query_emb,
            top_k=10,
            sim_threshold=0.3
        )
        if not seed_concepts:
            return await self.vector_fallback(query_text, kb_id, top_k)

        # Step 2-7: 与 TreeGuidedQuery 的逻辑相同
        # ...
```

### 6.4 GraphOnlyQuery

```
适用: query_mode = "graph_only"
用途: 用户浏览图谱、多跳提问
核心: 不检索 chunk，只在 CONCEPT 图上推理

流程:
  Query
   │
   ├─ Step 1: 向量搜索 CONCEPT（或名称匹配）
   │    → seed_concepts
   │
   ├─ Step 2: BFS 子图展开
   │    从种子概念开始，读取 neighbor_details
   │    每步取 top-k 个邻居
   │    → subgraph_concepts
   │
   ├─ Step 3: 序列化为文本
   │    concept names + descriptions + relations → graph_context
   │
   └─ Step 4: LLM 推理
        question + graph_context → LLM → answer
```

### 6.5 Shared Propagation Engine（PPR）

```python
class PropagationEngine:
    """
    PPR 传播引擎

    同时被 TreeGuidedQuery 和 GraphGuidedQuery 复用。
    基于 MGranRAG 的 local_ppr 逻辑，两处改进:
    1. 支持 query embedding 相似度加权 chunk 节点
    2. 支持 search_epochs 迭代重排序
    """

    def ppr_rerank(self, subgraph, seed_concept_ids, query_emb, chunk_embeddings,
                   alpha=0.85, max_iter=10, tol=1e-6):
        """
        在局部子图上跑 Personalized PageRank

        subgraph: networkx.Graph（chunk + concept 二分图）
        seed_concept_ids: 种子概念 ID 列表
        chunk_embeddings: {chunk_id: embedding_vector}
        """
        n = subgraph.number_of_nodes()
        if n == 0:
            return {}

        # personalization vector
        p = defaultdict(float)
        for cid in seed_concept_ids:
            if cid in subgraph:
                p[cid] = 1.0 / len(seed_concept_ids)

        # chunk 节点按与 query 的相似度加权
        for chunk_id, emb in chunk_embeddings.items():
            if chunk_id in subgraph:
                sim = cosine_similarity(query_emb, emb)
                p[chunk_id] = max(p.get(chunk_id, 0), sim * 0.5)

        total = sum(p.values())
        if total > 0:
            for k in p:
                p[k] /= total

        # 构建转移矩阵
        transition = defaultdict(dict)
        for u, v, data in subgraph.edges(data=True):
            weight = data.get("weight", 1.0)
            transition[u][v] = transition[u].get(v, 0) + weight

        for u in transition:
            total_weight = sum(transition[u].values())
            if total_weight > 0:
                for v in transition[u]:
                    transition[u][v] /= total_weight

        # PPR 迭代
        x = {node: 1.0 / n for node in subgraph.nodes()}
        x.update(p)

        for _ in range(max_iter):
            x_new = defaultdict(float)
            for node in subgraph.nodes():
                x_new[node] = (1 - alpha) * p.get(node, 0)
                for neighbor, weight in self._incoming_edges(node, transition):
                    x_new[node] += alpha * weight * x.get(neighbor, 0)

            diff = sum(abs(x_new.get(n, 0) - x.get(n, 0)) for n in subgraph.nodes())
            x = x_new
            if diff < tol:
                break

        return dict(x)

    def search_epochs(self, subgraph, seed_concept_ids, query_emb, chunk_embeddings,
                      ppr_alpha=0.85, epochs=3, top_k_per_epoch=5):
        """MGranRAG 风格的迭代重排序"""
        current_ppr = self.ppr_rerank(subgraph, seed_concept_ids, query_emb,
                                       chunk_embeddings, alpha=ppr_alpha)

        for epoch in range(1, epochs):
            top_chunks = sorted(
                [(n, s) for n, s in current_ppr.items()
                 if subgraph.nodes[n].get("type") == "chunk"],
                key=lambda x: x[1], reverse=True
            )[:top_k_per_epoch]

            new_concept_ids = set()
            for ctx_id, score in top_chunks:
                for edge in subgraph.edges(ctx_id):
                    other = edge[1] if edge[0] == ctx_id else edge[0]
                    if subgraph.nodes[other].get("type") == "concept":
                        new_concept_ids.add(other)

            if not new_concept_ids:
                break

            expanded_subgraph = self.expand_subgraph(subgraph, list(new_concept_ids))
            current_ppr = self.ppr_rerank(
                expanded_subgraph, list(new_concept_ids),
                query_emb, chunk_embeddings, alpha=ppr_alpha
            )

        return current_ppr

    def _incoming_edges(self, node, transition):
        for src, neighbors in transition.items():
            if node in neighbors:
                yield src, neighbors[node]
```

---

## 7. 用户配置 API

### 7.1 知识库级别配置（创建时设定）

```python
# POST /api/v1/knowledge_base
{
    "name": "公司知识库",
    "parser_config": {
        "chunk_method": "naive",        # 现有 + ukgs 新增
        "chunk_token_count": 512,
        "ukgs": {                        # 新增 UKGS 配置块
            "enabled": False,            # 默认关闭，向后兼容
            "index_mode": "llm",         # "llm" | "spacy"
            "llm_config": {
                "extraction_model": "gpt-4o",
                "enable_entity_resolution": True,
                "enable_community_detection": True,
                "relation_extraction": True
            },
            "spacy_config": {
                "spacy_model": "en_core_web_trf",
                "enable_pos_phrase": True,
                "enable_named_entity": True,
                "enable_phrase_dedup": True
            },
            "common_config": {
                "index_granularity": "sentence",
                "enable_hierarchy": True
            }
        }
    }
}
```

### 7.2 查询级别配置（每次请求指定）

```python
# POST /api/v1/query
{
    "query": "Nike 的市场份额在 Methodology 章节中如何描述？",
    "kb_id": "kb_123",
    "query_config": {
        "mode": "tree_guided",
        "structure_filters": [
            {"type": "section", "value": "Methodology"}
        ],
        "graph_config": {
            "top_k_context": 10,
            "ppr_alpha": 0.85,
            "search_epochs": 3,
            "enable_query_decomposition": True
        }
    }
}
```

### 7.3 图谱浏览 API

```python
# 获取子图供前端展示
# GET /api/v1/knowledge_base/{kb_id}/graph?seed=Nike&hops=2
{
    "nodes": [
        {"id": "con_Nike", "name": "Nike", "type": "ORG", "description": "..."},
        {"id": "con_NBA", "name": "NBA", "type": "ORG"},
        {"id": "con_LeBron_James", "name": "LeBron James", "type": "PERSON"}
    ],
    "edges": [
        {"source": "con_Nike", "target": "con_NBA", "type": "semantic",
         "label": "sponsors", "weight": 0.9},
        {"source": "con_NBA", "target": "con_LeBron_James", "type": "semantic",
         "label": "employs", "weight": 0.8}
    ]
}

# 多跳推理问答
# POST /api/v1/knowledge_base/{kb_id}/graph/qa
{
    "query": "Nike 的 CEO 是否也是耐克品牌的创始人？",
    "hops": 3
}
```

---

## 8. 系统矩阵

### 8.1 索引模式 × 查询模式

```
                        query_mode
               tree_guided          graph_guided         graph_only
           ┌─────────────────┬─────────────────────┬──────────────────┐
           │ ✅ 推荐组合       │ ✅ 可行组合           │ ✅ 可行组合        │
index   llm │ 文档问答 + 图谱   │ 语义搜索 + 多跳推理   │ 多跳图谱 QA        │
_mode      │ BookRAG 场景     │ GraphRAG 场景         │ 需 LLM 推理        │
           │ 最精确的检索      │ 最丰富的语义推理      │                    │
           ├─────────────────┼─────────────────────┼──────────────────┤
           │ ✅ 推荐组合       │ ⚠️ 可行但有限制       │ ⚠️ 有限制           │
spacy      │ 低成本文档问答    │ 图谱浏览（无语义标签） │ 共现拓扑推理        │
           │ 轻量 BookRAG     │ MGranRAG 风格        │ 仅拓扑 + 问题      │
           │ NER + POS 短语   │ 只有 co_occur 边     │ embedding 匹配     │
           │ + 结构过滤        │ + PPR                │ 无语义关系          │
           │ + 共现 PPR       │                     │                    │
           └─────────────────┴─────────────────────┴──────────────────┘
```

### 8.2 四种系统的能力保留映射

| 原系统能力 | UKGS 中的实现 | 依赖的表/字段 |
|-----------|--------------|--------------|
| BookRAG 按 section/page/type 过滤 | chunk.granularity + chunk.path + chunk.page_num_int ES query | chunk 表 |
| BookRAG tree2kg 映射 | chunk.mentions ↔ concept.chunk_details | chunk + concept_node |
| GraphRAG 实体检索 | concept_node.q_vec vector search | concept_node |
| GraphRAG 关系检索 | concept_node.neighbor_details[] | concept_node |
| GraphRAG 社区报告 | knowledge_graph_kwd="community_report" | chunk 表（复用） |
| LinearRAG entity↔sentence↔passage 传播 | PPR on chunk↔concept↔chunk 子图 | chunk + concept_node |
| LinearRAG cosine 加权 PPR | PropagationEngine 的 chunk_embeddings 相似度 | chunk.q_vec |
| MGranRAG 短语抽取 | SpacyIndexer.extract_phrases() | 索引阶段 |
| MGranRAG Query Decomposition | TaskPlanner.classify() | 查询路由 |
| MGranRAG search_epochs | PropagationEngine.search_epochs() | 查询阶段 |

---

## 9. 实现计划

### Phase 1：Schema 扩展 + 写入层（2 周）

- 定义 chunk 表的新增字段 mapping（`node_type`, `granularity`, `depth`, `path`, `parent_id`, `mentions`, `mention_details`, `source_lineage`）
- 创建 `concept_node` 新 ES/Infinity 索引
- 实现 `SpacyIndexer`（NER + POS 短语 + 共现边 + 回填 chunk.mentions）
- 实现 `LLMIndexer` 骨架（调用 LLM 抽取 + 写入 concept_node）
- 实现 `EntityChunkAdapter`（兼容旧版 GraphRAG chunk 读取）
- 在 `do_handle_task()` 的 `insert_chunks` 之后插入 UKGS 调用点

### Phase 2：查询层（2 周）

- 实现 `SharedPropagationEngine`（局部子图 PPR）
- 实现 `TreeGuidedQuery`（结构过滤 chunk 表 + 概念搜索 + PPR）
- 实现 `GraphGuidedQuery`（向量搜索 concept_node + 扩展 + PPR）
- 实现 `GraphOnlyQuery`（子图展示 + 多跳 QA）

### Phase 3：旧版迁移 + 集成（1 周）

- 实现 `lazy_migrate_kb_to_v2()`
- 修改 `rag/graphrag/general/index.py` 的 `merge_subgraph` 加入版本路由
- 在知识库配置页面增加 UKGS 开关；默认关闭

### Phase 4：前端图谱浏览器（2 周）

- 基于 `GraphOnlyQuery.view_subgraph()` 的图谱可视化界面
- 节点展开/收起、关系标签展示
- 多跳提问输入框

---

## 10. 附录：与四个原始系统的本质关系

```
Original systems
│
├─ GraphRAG ───→ CONCEPT node (entity + relation) + COMMUNITY node (chunk 表复用)
│                ↑ 保留了核心语义，去掉了全局图 blob 和分离的 relation 表
│
├─ LinearRAG ──→ chunk (= CONTEXT) + PPR 算法
│                ↑ 保留了多层传播机制，去掉了 3 种 node 类型和 GPU 稀疏矩阵
│
├─ MGranRAG ───→ chunk (= sentence-level CONTEXT) + CONCEPT node (phrase) +
│                query decomposition + search_epochs
│                ↑ 保留了细粒度传播和迭代重排序，去掉了 igraph 全量图
│
└─ BookRAG ────→ chunk hierarchy (depth/path/parent_id) +
                 chunk↔concept 反向映射 (mentions ↔ chunk_details)
                 ↑ 保留了文档结构导航，去掉了独立 TreeNode 和 tree2kg 表
```

### 关键术语对照表

| RAGFlow 概念 | UKGS 概念 | 说明 |
|-------------|-----------|------|
| chunk (文本块) | CONTEXT | **完全统一**。chunk 就是图的下文节点 |
| chunk.content_with_weight | CONTEXT.content | chunk 的正文就是 CONTEXT 的内容 |
| chunk.id | CONTEXT.id | chunk 的主键就是 CONTEXT 的主键 |
| chunk.q_*_vec | CONTEXT.q_vec | chunk 的 embedding 就是 CONTEXT 的 embedding |
| chunk.doc_id | CONTEXT.source_lineage[].document_id | chunk 的所属文档就是 CONTEXT 的文档溯源 |
| chunk.kb_id | CONTEXT.kb_id | 知识库 ID，完全一致 |
| chunk 表 (ES/Infinity) | 无新索引 | chunk 表同时扮演 CONTEXT 表 |



---

## 11. 容错分析（基于当前 GraphRAG 实现的实践观察）

本章分析当前 `rag/graphrag/general/` 实现在**小规模验证场景**下的容错问题。即使只处理 1-2 个文档，也常出现卡住数十分钟乃至数小时的故障。以下按根因链梳理。

### 11.1 根因一：无有效超时机制

代码中所有外部超时依赖 `ENABLE_TIMEOUT_ASSERTION` 环境变量（默认未设置）：

```python
# general/index.py:121 — 文档级超时
timeout_sec = max(120, len(chunks) * 60 * 10) if enable_timeout_assertion else 10000000000
#                                              ↑ 关闭 → ≈317年
```

```python
# utils.py:92 — 存储写入超时
timeout_s = 3 if enable_timeout_assertion else 30000000
#                                     ↑ 关闭 → ≈347天
```

唯一生效的超时是 `_async_chat` 中的硬编码 20 分钟：

```python
# general/extractor.py:93
response = await asyncio.wait_for(
    self._llm.async_chat(...),
    timeout=60 * 20,  # 硬编码 20 分钟，不区分模型/场景
)
```

**问题链**：`ENABLE_TIMEOUT_ASSERTION` 未设 → 外部超时天文数字 → `_async_chat` 20 分钟硬编码成为唯一超时 → 每次 LLM 调用卡满 20 分钟才 timeout。

### 11.2 根因二：每 chunk 串行多次 LLM 调用且不检查取消状态

单 chunk 的调用链包含**初始抽取 + `max_gleanings`(默认 2) 次 gleaning = 3 次串行 LLM 调用**：

```python
# general/graph_extractor.py
async def _process_single_content(self, chunk_key_dp, ...):
    response = await self._async_chat(hint_prompt, ...)  # 调用 #1
    for i in range(self._max_gleanings):                 # 默认 2
        response = await self._async_chat("", history, ...)  # 调用 #2
        # ← 缺失 has_canceled() 检查！
        continuation = await self._async_chat("", ..., LOOP_PROMPT)  # 调用 #3
```

50 个 chunk × 3 次串行调用 = 150 次 LLM 调用。如果每次卡满 20 分钟，理论最差耗时 = **50 × 3 × 20min = 50 小时**。

**gleaning 循环及各阶段间在任何 LLM 调用后都不执行 `has_canceled()` 检查**，用户取消任务后仍需等待当前 chunk 的所有 gleaning 完成。

### 11.3 根因三：全局信号量导致所有阶段互相阻塞

```python
chat_limiter = asyncio.Semaphore(10)  # rag/graphrag/utils.py，全局共享
```

所有阶段（chunk 抽取、entity resolution、community report、embedding）共享同一个 `chat_limiter`。如果前 10 个 LLM 调用都卡在 20 分钟 timeout 上 → 占满所有信号量槽 → 后续所有操作（包括已完成 chunk 的后处理）在等信号量释放 → 全局死等。

### 11.4 根因四：分布式锁无最大等待时间

```python
# rag/utils/redis_conn.py:555-560
async def spin_acquire(self):
    while True:
        if self.lock.acquire(token=self.lock_value):
            break
        await asyncio.sleep(10)  # 每 10 秒重试一次
    # timeout=1200s（20分钟自动释放）
```

在 `run_graphrag` 路径中，merge → resolution → community **三阶段各 `spin_acquire()` 一次**：

```python
# general/index.py:148-183
await graphrag_task_lock.spin_acquire()   # 锁 #1：merge 前
await graphrag_task_lock.spin_acquire()   # 锁 #2：resolution 前
await graphrag_task_lock.spin_acquire()   # 锁 #3：community 前
```

如果同一 KB 的并发任务持有锁，且前一个任务在锁内 hang 住，`spin_acquire` 会**无限自旋**直到锁 TTL(20 min) 自动过期——即使只有 1 个文档。

### 11.5 根因五：`_async_chat` 的 TimeoutError 不重试

```python
# general/extractor.py:100-105
except asyncio.TimeoutError:
    logging.warning("_async_chat timed out after 20 minutes")
    raise  # timeout is not a transient error; do not retry
```

其他异常（如 HTTP 5xx、连接重置）有 3 次重试指数退避，但 `TimeoutError` **直接 raise、不重试**。网络抖动或 LLM 后端瞬时过载引起的瞬态超时本应重试，当前行为导致一次超时就使对应 chunk 失败、计入 `GRAPHRAG_MAX_ERRORS`（默认 3 次）。

### 11.6 根因六：`merge_subgraph` 超时与 `set_graph` 执行时间不匹配

```python
# general/index.py:148
@timeout(60 * 3)  # 3 分钟超时
async def merge_subgraph(...):
    await set_graph(...)
```

`set_graph` 内部包含：
1. 遍历所有 nodes → `graph_node_to_chunk`（含 embedding 调用）
2. 遍历所有 edges → `graph_edge_to_chunk`（含 embedding 调用）
3. 删除旧的 graph/subgraph
4. 删除旧的 entity/relation chunks
5. 插入新的 chunks

如果 embedding model 慢，阶段 1-2 可能超过 3 分钟 → `merge_subgraph` 被强制超时 → 但 `set_graph` 内部的 `asyncio.gather` 仍在后台运行 → **部分写入不一致**。

### 11.7 改进方向

| 优先级 | 改进项 | 影响范围 | 修改位置 |
|--------|--------|---------|---------|
| P0 | `_async_chat` 用合理 per-call 超时(30-60s)替代硬编码 20min | 所有 LLM 调用 | `extractor.py` |
| P0 | gleaning 循环加入 `has_canceled()` 检查 | chunk 抽取 | `graph_extractor.py` |
| P0 | `spin_acquire()` 增加 `max_wait_seconds` 参数 | 分布式锁 | `redis_conn.py` |
| P1 | 全局 `chat_limiter` 拆分为阶段级信号量 | 信号量竞争 | `utils.py` |
| P1 | `_async_chat` 的 TimeoutError 也纳入重试逻辑 | LLM 重试 | `extractor.py` |
| P1 | `merge_subgraph` 取消 `@timeout(60*3)` 或在外部包装 | 部分写入不一致 | `index.py` |
| P1 | `ENABLE_TIMEOUT_ASSERTION` 拆分成分项可配置的环境变量 | 整体超时体系 | `utils.py`, `index.py` |
| P2 | `message_fit_in` 截空后的无用 LLM 调用保护 | 无效 LLM 调用 | `extractor.py` |
| P2 | `_process_single_content` 与 `_resolve_candidate` 错误处理对齐 | 统一语义 | `graph_extractor.py`, `entity_resolution.py` |

### UKGS 设计中的容错内建

UKGS 通过架构级改变从根本避免以上问题：

1. **增量索引取代全量图**：每个文档的索引独立写入，不存在 `merge_subgraph` 全局锁竞争。并发写的只读查询不受影响。
2. **无全局 LLM 信号量**：每个文档的索引管线有独立的 `asyncio.Semaphore`，文档间不互相阻塞。
3. **无分布式锁依赖**：增量写入天然隔离，不需要 `spin_acquire`。唯一的锁 (KB 级 schema 迁移) 设计为可重入 + 超时。
4. **无全局图驻留**：`chunk` 和 `concept_node` 都在 ES/Infinity 中，索引管线写完即释放，无全量图 OOM 风险。

---

> **版本历史**
> - v2.0 (2026-05-07): 核心统一 chunk = CONTEXT，重写存储架构和索引管线集成；新增 §11 容错分析
> - v1.0 (2026-04-30): 初始设计
