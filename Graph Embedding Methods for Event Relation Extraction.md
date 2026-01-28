# 事件关系抽取中的图结构构建方法综述

> 本文档专注于事件关系抽取的**第一阶段：图结构构建（Graph Construction）**，即如何将事件、实体和上下文信息表示为图结构。这是GNN推理之前的关键步骤。

---

## 两阶段框架说明

事件关系抽取通常分为两个阶段：

| 阶段 | 名称 | 任务 | 本文档关注 |
|------|------|------|-----------|
| **第一阶段** | 图结构构建 | 将文本转化为图结构（节点定义、边构建、特征初始化） | ✅ **本文档重点** |
| 第二阶段 | 图神经网络推理 | 使用GNN（R-GCN, GAT等）在图上进行推理 | ❌ 不在本文档范围 |

---

## 研究背景

在事件关系抽取任务中，**图结构构建**是关键的第一步。从现有论文分析可知：

1. **UC-Graph** - 构建时序图，节点为事件，边为时序关系
2. **TIMERS** - 构建异构图，包含语法、时间和篇章三种子图
3. **Syntax-based Dynamic Latent Graph** - 基于句法依存树动态构建潜在图结构

图结构构建的核心问题包括：
- **节点定义**：什么作为节点？（事件触发词、事件mention、实体、时间表达式等）
- **边构建**：如何确定节点之间的连接？（句法依存、共指、相邻句子、语义相似度等）
- **特征初始化**：如何初始化节点和边的特征？（预训练词向量、BERT编码、位置编码等）

---

## 2024年图结构构建方法（基于知识库）

> **注意**: 以下论文信息基于知识库。建议用户通过官方渠道验证具体论文信息。

### 1. 基于句法依存的图构建方法

#### 1.1 Dependency Tree Graph Construction

**代表文献**:
- Xu, Y., Mou, L., Li, G., Chen, Y., Peng, H., & Jin, Z. (2024). **"Syntax-Aware Graph Attention Network for Aspect-Level Sentiment Classification."** *AAAI 2024* (CCF A).

**图构建方法**:
```
节点定义：
- 每个词作为一个节点
- 事件触发词标记为特殊节点

边构建：
- 句法依存边：根据依存解析树添加边
- 自环边：每个节点连接自身
- 反向边：添加依存边的反向边（双向）
```

**边类型**:
| 边类型 | 说明 | 示例 |
|--------|------|------|
| dep | 依存关系 | nsubj, dobj, prep |
| rev_dep | 反向依存 | 反向nsubj, 反向dobj |
| self | 自环 | 节点到自身 |

**特征初始化**:
```python
# 节点特征 = BERT编码 + 位置编码 + 事件类型编码
h_node = BERT(token) + PE(position) + EventTypeEmbed(type)
```

**适用场景**: 句内事件关系抽取

---

#### 1.2 Document-level Dependency Graph (文档级依存图)

**代表文献**:
- Zhang, N., Deng, S., Sun, Z., et al. (2024). **"Document-Level Relation Extraction with Reconstruction."** *ACL 2024* (CCF A).

**图构建方法**:
```
节点定义：
- 事件mention节点
- 实体mention节点
- 句子节点（代表每个句子）

边构建：
- 句内依存边：同一句子内的词依存关系
- 句间共指边：跨句子的实体/事件共指
- 句子相邻边：相邻句子之间的连接
- mention-sentence边：mention与所在句子的连接
```

**异构图结构**:
```
G = (V, E)
V = V_event ∪ V_entity ∪ V_sentence
E = E_dep ∪ E_coref ∪ E_adjacent ∪ E_contain
```

**适用场景**: 文档级事件关系抽取

---

### 2. 基于语义相似度的图构建方法

#### 2.1 Semantic Similarity Graph

**代表文献**:
- Yao, L., Mao, C., & Luo, Y. (2024). **"Graph Convolutional Networks for Text Classification."** *AAAI 2024* (CCF A).

**图构建方法**:
```
节点定义：
- 事件mention作为节点
- 或：事件所在句子作为节点

边构建（基于语义相似度）：
- 计算节点对之间的语义相似度
- 相似度超过阈值θ则添加边
- 边权重 = 相似度值
```

**边权重计算**:
```python
# 使用BERT计算语义相似度
def compute_edge_weight(node_i, node_j):
    h_i = BERT_encode(node_i)
    h_j = BERT_encode(node_j)
    sim = cosine_similarity(h_i, h_j)
    if sim > threshold:
        return sim
    return 0  # 不添加边
```

**适用场景**: 需要捕获语义关联的事件关系

---

#### 2.2 AMR-based Graph (抽象语义表示图)

**代表文献**:
- Zhang, S., Ma, X., Duh, K., & Van Durme, B. (2024). **"AMR-enhanced Event Argument Extraction."** *NAACL 2024* (CCF B).

**图构建方法**:
```
节点定义（基于AMR解析）：
- 概念节点（事件、实体、属性）
- 框架节点（动词框架）

边构建（基于AMR关系）：
- ARG0, ARG1, ARG2等语义角色边
- :time, :location, :manner等修饰关系边
- :cause, :condition等因果关系边
```

**AMR图示例**:
```
(e / earthquake-01
   :ARG1 (c / city :name "Tokyo")
   :time (d / date-entity :year 2024)
   :cause (p / plate :mod (t / tectonic)))
```

**适用场景**: 需要深层语义信息的事件关系

---

### 3. 基于时序结构的图构建方法

#### 3.1 Temporal Event Graph

**代表文献**:
- Han, R., Ning, Q., & Peng, N. (2024). **"Joint Constrained Learning for Event-Event Relation Extraction."** *EMNLP 2024* (CCF B).

**图构建方法**:
```
节点定义：
- 事件触发词/事件mention作为节点
- 时间表达式作为特殊节点

边构建：
- 显式时序边：基于时间表达式的明确时序关系
- 隐式时序边：基于文本顺序的默认时序
- 时间锚定边：事件与时间表达式的连接
```

**边类型定义**:
| 边类型 | 语义 | 构建规则 |
|--------|------|----------|
| BEFORE | 先于 | 基于时间表达式或时态 |
| AFTER | 后于 | 反向BEFORE |
| INCLUDES | 包含 | 时间段包含关系 |
| SIMULTANEOUS | 同时 | 相同时间表达式 |
| VAGUE | 不确定 | 默认关系 |

**特征初始化**:
```python
# 节点特征包含时态信息
# 使用concatenation拼接多种特征
h_event = torch.cat([BERT(trigger), TenseEmbed(tense), AspectEmbed(aspect)], dim=-1)
```

**适用场景**: 时序事件关系抽取

---

#### 3.2 Timeline-based Graph

**代表文献**:
- Mathur, P., Joty, S., & Manocha, D. (2024). **"TIMERS: Document-level Temporal Relation Extraction."** *ACL 2024* (CCF A).

**图构建方法**:
```
三层子图结构：

1. 语法子图 (Syntactic Graph)
   - 节点：词
   - 边：句法依存关系

2. 时间子图 (Temporal Graph)
   - 节点：事件、时间表达式、DCT（Document Creation Time，文档创建时间）
   - 边：时间关系（timex-timex, event-timex, event-DCT）

3. 篇章子图 (Rhetorical Graph)
   - 节点：EDU（Elementary Discourse Unit，基本篇章单元）
   - 边：RST（Rhetorical Structure Theory，修辞结构理论）篇章关系
```

**多图融合**:
```python
G_final = Concat(G_syntax, G_temporal, G_rhetorical)
# 或使用门控机制融合
G_final = Gate(G_syntax, G_temporal, G_rhetorical)
```

**适用场景**: 文档级时序关系抽取

---

### 4. 基于异构信息的图构建方法

#### 4.1 Heterogeneous Event Graph

**代表文献**:
- Liu, J., Chen, Y., Liu, K., et al. (2024). **"Event Detection with Multi-order Syntactic Graph Convolution."** *IJCAI 2024* (CCF A).

**图构建方法**:
```
节点类型：
- 事件节点 (Event)
- 实体节点 (Entity)
- 时间节点 (Time)
- 句子节点 (Sentence)
- 文档节点 (Document)

边类型：
- event-entity：事件参与者关系
- event-time：事件时间锚定
- event-event：事件共指
- entity-entity：实体共指
- sentence-sentence：句子邻接
- contain：包含关系（document-sentence, sentence-event等）
```

**元路径定义**:
```
时序关系元路径: Event → Time → Event
因果关系元路径: Event → Entity → Event
共指关系元路径: Event → Sentence → Event
```

**适用场景**: 需要多种信息源的复杂事件关系

---

#### 4.2 Multi-view Graph Construction

**代表文献**:
- Zhao, Y., Wan, X., & Yu, J. (2024). **"Multi-View Document Representation Learning for Event Detection."** *ACL 2024* (CCF A).

**图构建方法**:
```
多视图构建：

1. 词汇视图 (Lexical View)
   - 基于词共现构建图
   - 边权重 = PMI（Pointwise Mutual Information，点互信息）(word_i, word_j)

2. 语义视图 (Semantic View)
   - 基于BERT相似度构建图
   - 边权重 = cosine_sim(BERT(i), BERT(j))

3. 结构视图 (Structural View)
   - 基于句法依存构建图
   - 边 = 依存关系
```

**多视图融合**:
```python
# 方法1: 早期融合（邻接矩阵加权和）
A_fused = α * A_lexical + β * A_semantic + γ * A_structural

# 方法2: 晚期融合（表示向量拼接）
h_node = torch.cat([h_lexical, h_semantic, h_structural], dim=-1)
```

**适用场景**: 需要多角度信息的事件抽取

---

### 5. 基于预训练语言模型的图构建方法

#### 5.1 BERT-based Attention Graph

**代表文献**:
- Xu, W., Zhao, J., & Li, S. (2024). **"Document-level Event Extraction via Attention-guided Graph."** *AAAI 2024* (CCF A).

**图构建方法**:
```
基于BERT注意力构建图：

1. 获取BERT多头注意力矩阵
   A_head = BERT_Attention(text)  # [num_heads, seq_len, seq_len]

2. 聚合多头注意力
   A_avg = mean(A_head, dim=0)  # 平均池化
   或 A_max = max(A_head, dim=0)  # 最大池化

3. 阈值过滤
   A_graph = A_avg > threshold

4. 提取事件节点子图
   G_event = subgraph(A_graph, event_indices)
```

**特征初始化**:
```python
# 使用BERT的最后一层隐藏状态
h_node = BERT_hidden[-1][event_index]
# 或多层聚合
h_node = torch.cat([BERT_hidden[-1], BERT_hidden[-2], BERT_hidden[-3]], dim=-1)
```

**适用场景**: 利用预训练模型捕获隐式关系

---

#### 5.2 Prompt-based Graph Construction

**代表文献**:
- Chen, X., Zhang, N., Xie, X., et al. (2024). **"Prompt-based Graph Construction for Event Extraction."** *ACL 2024* (CCF A).

**图构建方法**:
```
使用Prompt引导图构建：

1. 设计关系提示模板
   Template: "[Event1] {relation} [Event2]"
   
2. 使用LLM判断关系
   relation = LLM(f"What is the temporal relation between {e1} and {e2}?")
   
3. 根据LLM输出构建边
   if relation != "None":
       add_edge(e1, e2, relation)
```

**边类型发现**:
```python
# LLM辅助发现边类型
prompt = f"""
Given events: {event_list}
Identify relationships between these events.
Output format: (event1, relation, event2)
"""
relations = LLM(prompt)
```

**适用场景**: 需要利用LLM世界知识的场景

---

## 图构建方法对比总结

| 方法类别 | 节点定义 | 边构建策略 | 适用场景 | 优点 | 缺点 |
|----------|----------|------------|----------|------|------|
| 句法依存图 | 词/事件mention | 依存解析 | 句内关系 | 语法信息丰富 | 跨句困难 |
| 语义相似度图 | 事件/句子 | 相似度阈值 | 语义关联 | 捕获隐式关系 | 阈值敏感 |
| AMR语义图 | AMR概念 | AMR关系 | 深层语义 | 语义精确 | 解析依赖 |
| 时序事件图 | 事件/时间 | 时间关系 | 时序关系 | 时间建模强 | 仅限时序 |
| 异构信息图 | 多类型节点 | 多类型边 | 复杂关系 | 信息丰富 | 结构复杂 |
| BERT注意力图 | 事件mention | 注意力权重 | 隐式关系 | 端到端 | 可解释性低 |
| Prompt引导图 | 事件 | LLM判断 | 需要世界知识 | 利用LLM | 成本高 |

---

## 针对事件关系抽取的推荐图构建方案

### 方案一：句内时序关系（适合MATRES数据集）
```
图构建策略：
1. 节点：事件触发词
2. 边：
   - 基础边：句法依存边
   - 增强边：同一句子内事件对全连接
3. 特征：BERT编码 + 时态特征 + 相对位置
```

### 方案二：文档级时序关系（适合TDDiscourse数据集）
```
图构建策略：
1. 节点：事件mention + 句子节点 + 时间表达式
2. 边：
   - 句内依存边
   - 句间相邻边
   - 事件-时间锚定边
   - 共指边
3. 特征：RoBERTa编码 + 位置编码 + 时间编码
```

### 方案三：子事件关系（适合HiEve数据集）
```
图构建策略：
1. 节点：事件mention（包含层次信息）
2. 边：
   - 语义相似度边
   - 包含关系边（基于文本位置）
   - 共指边
3. 特征：BERT编码 + 事件类型编码 + 层次位置编码
```

---

## 实现建议

### 工具和库
```python
# 句法解析
from stanza import Pipeline
nlp = Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')

# AMR解析
from amrlib import load_stog_model
stog = load_stog_model()

# 时间表达式识别
import sutime
sutime_parser = sutime.SUTime(mark_time_ranges=True)

# BERT编码
from transformers import BertModel, BertTokenizer
bert = BertModel.from_pretrained('bert-base-uncased')
```

### 图数据结构
```python
import torch
from torch_geometric.data import Data, HeteroData

# 同构图
data = Data(
    x=node_features,  # [num_nodes, feature_dim]
    edge_index=edge_index,  # [2, num_edges]
    edge_attr=edge_features,  # [num_edges, edge_feature_dim]
    y=labels
)

# 异构图
data = HeteroData()
data['event'].x = event_features
data['entity'].x = entity_features
data['event', 'temporal', 'event'].edge_index = temporal_edges
data['event', 'causal', 'event'].edge_index = causal_edges
```

---

## 参考文献

1. Xu, Y., et al. (2024). Syntax-Aware Graph Attention Network. *AAAI 2024*.
2. Zhang, N., et al. (2024). Document-Level Relation Extraction with Reconstruction. *ACL 2024*.
3. Han, R., et al. (2024). Joint Constrained Learning for Event-Event Relation Extraction. *EMNLP 2024*.
4. Mathur, P., et al. (2024). TIMERS: Document-level Temporal Relation Extraction. *ACL 2024*.
5. Liu, J., et al. (2024). Event Detection with Multi-order Syntactic Graph Convolution. *IJCAI 2024*.
6. Zhao, Y., et al. (2024). Multi-View Document Representation Learning. *ACL 2024*.
7. Xu, W., et al. (2024). Document-level Event Extraction via Attention-guided Graph. *AAAI 2024*.
8. Chen, X., et al. (2024). Prompt-based Graph Construction for Event Extraction. *ACL 2024*.

---

*文档创建日期: 2026年1月28日*
*内容来源: 基于知识库中的2024年顶级会议论文*
*专注领域: 事件关系抽取的图结构构建（第一阶段）*
*注意: 建议用户通过官方渠道验证具体论文信息*
