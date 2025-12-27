# Logic Induced High-Order Reasoning Network for Event-Event Relation Extraction

# Abstract

To understand a document with multiple events, event-event relation extraction (ERE) emerges as a crucial task, aiming to discern how natural events temporally or structurally associate with each other. To achieve this goal, our work addresses the problems of temporal event relation extraction (TRE) and subevent relation extraction (SRE). The latest methods for such problems have commonly built document-level event graphs for global reasoning across sentences. However, the edges between events are usually derived from external tools heuristically, which are not always reliable and may introduce noise. Moreover, they are not capable of preserving logical constraints among event relations, e.g., coreference constraint, symmetry constraint and conjunction constraint. These constraints guarantee coherence between different relation types, enabling the generation of a unified event evolution graph. In this work, we propose a novel method named LogicERE, which performs high-order event relation reasoning through modeling logic constraints. Specifically, different from conventional event graphs, we design a logic constraint induced graph (LCG) without any external tools. LCG involves event nodes where the interactions among them can model the coreference constraint, and event pairs nodes where the interactions among them can retain the symmetry constraint and conjunction constraint. Then we perform high-order reasoning on LCG with relational graph transformer to obtain enhanced event and event pair embeddings. Finally, we further incorporate logic constraint information via a joint logic learning module. Extensive experiments demonstrate the effectiveness of the proposed method with state-of-the-art performance on benchmark datasets.

# Introduction

Interpreting news messages involves identifying how natural events temporally or structurally associate with each other from news documents, i.e., extracting event temporal relations and subevent relations. Through this process, one can induce event evolution graphs that arrange multiple-granularity events with temporal relations and subevent relations interacting among them. The event evolution graphs built through event-event relation extraction (ERE) are important aids for future event forecasting (Chaturvedi, Peng, and Roth 2017)

and hot event tracking (Zhuang, Fei, and Hu 2023b). As shown in Figure 1, ERE aims to induce such an event evolution graph, in which the event mention storm involves more fine-grained subevent mentions, i.e., killed, died and canceled. Some of those mentions follow temporal order, e.g., died happens BEFORE canceled. Generally, predicting the relations between diverse events within the same document, such that these predictions are coherent and consistent with the document, is a challenging task (Xiang and Wang 2019).

Recently, significant research efforts have been devoted to several ERE tasks, such as event temporal relation extraction (TRE) (Zhou et al. 2021; Wang, Li, and Xu 2022; Tan, Pergola, and He 2023) and subevent relation extraction (SRE) (Man et al. 2022; Hwang et al. 2022). Nonetheless, ERE is still challenging because most event relations lack explicit clue words such as before and contain in natural languages, especially when events scatter in a document. Accordingly, a few previous methods attempt to build a document-level event graph to assist in the cross-sentence inference, where the nodes are events, and the edges are designed with linguistic/discourse relations of event pairs (Zhuang, Hu, and Zhao 2023; Li and Geng 2024). Despite the success, these methods face two major issues. First, they are not capable of preserving logic constraints among relations, such as transitivity, during training time (Roth and Yih 2004). Second, the edges heuristically from external tools may introduce noise and cause exhaustive extraction (Phu and Nguyen 2021).

For the first issue, Wang et al. (2020) propose a constrained learning framework which enforces logic coherence amongst the predicted relation types through extra differentiable objectives. However, since the coherence is enforced in a soft manner, there is still room for improving coherent predictions. In this work, we show that it is possible to enforce coherence in a much stronger manner by feeding logic constraints into the event graphs. For the second issue, Chen et al. (2022) propose an event pair centered causality identification model which takes event pairs as nodes and relations of event relations as edges, refraining the external tools and enabling the causal transitivity reasoning. However, some useful prior relation constraints such as coreference are discarded. Moreover, we observe logical property information loss from the document to graph, which should be complied in global inference for ERE. We summarize these properties implied in the temporal and subevent relations as three logical constraints: (1) Coref-

![](images/b75d06824f3cc2f573f7652c85da35677a7a6298aa5f972c167b7db501335984.jpg)


![](images/851d96ae9caacaa5fe5391af1bd6a259380ad95ac71e65045b72dae764fa1c69.jpg)


Coref. constraint

$$
r (e _ {1}, e _ {2}) \Leftrightarrow r (e _ {1}, e _ {3})
$$

Symmetry constraint

$$
r \left(e _ {1}, e _ {2}\right) \Leftrightarrow \bar {r} \left(e _ {2}, e _ {1}\right)
$$

Conjunct. constraint

$$
r _ {1} \left(\boldsymbol {e} _ {1}, \boldsymbol {e} _ {2}\right) \wedge r _ {2} \left(\boldsymbol {e} _ {2}, \boldsymbol {e} _ {3}\right) \Rightarrow r _ {3} \left(\boldsymbol {e} _ {1}, \boldsymbol {e} _ {3}\right)
$$

$$
r _ {1} \left(e _ {1}, e _ {2}\right) \wedge r _ {2} \left(e _ {2}, e _ {3}\right) \Rightarrow \neg r _ {4} \left(e _ {1}, e _ {3} \right.
$$


Figure 1: An example of an event evolution graph described in the document.


erence constraint: Considering the example in Figure 1, given that  $e_2$ :killed is BEFORE  $e_5$ :affecting and  $e_3$ :died is COREFERENCESD to  $e_2$ :killed,  $e_3$ :died should be BEFORE  $e_5$ :affecting. (2) Symmetry constraint: As shown in Figure 1,  $e_1$ :storm is a PARENT of  $e_2$ :killed, indicting  $e_2$ :killed is a CHILD of  $e_1$ :storm. (3) Conjunction constraint: From Figure 1, if  $e_1$ :storm is a PARENT of  $e_3$ :died and  $e_3$ :died is BEFORE  $e_4$ :canceled, the learning process should enforce  $e_1$ :storm is a PARENT of  $e_4$ :canceled by considering the conjunctive logic on both temporal and subevent relations. These logical constraints depict the mutuality among event relations of TRE and SRE, enabling the generation of a unified event evolution graph. While previous researches center on preserving these properties with post-learning inference or differentiable loss functions (Wang et al. 2020), there is no effective way to endow the graph model with these logical constraints for global reasoning.

In this paper, we consider the above constraints and propose a novel ERE model, logic induced high-order reasoning network (LogicERE)  ${}^{1}$  . Our intuition is to feed the logic constraints into the event graphs for high-order event relation reasoning and prediction. Specifically, we first build a logic constraint induced graph (LCG) that models these logical constraints through interactions between events and among event pairs. Then, we encode the heterogeneous LCG with relational graph transformer. This enables the model to effectively reasoning over remote event pairs while maintaining inherent logic properties of event relations. Finally, we introduce a joint logic learning module and design two logic constraint learning objectives to further regularize the model towards consistency on logic constraints.

LogicERE models these high-order logical constraints in two aspects. Firstly, our proposed LCG consolidates both event centered and event pair centered graphs, so that it can reason over not only coreference property among events, but also high-order symmetry and conjunction properties among event pairs. Specifically, LCG defines two types of nodes, i.e., event nodes and event pair nodes. Accordingly, there are three types of edges: (1) Event-event edge for prior event relations (e.g., coreference), which retains the coreference constraint and model the information flow among event nodes. (2) Event pair-event pair edge for two event pairs sharing at least one event, which keeps the symmetry and conjunction constraints, as well as captures the interactions among event pair nodes. (3) Event-event pair edge for an event pair and its corresponding two events, which models the information flow from event nodes to event pair nodes. As LCG preserves these inherent properties of event relations, we can get enhanced event and event pair embeddings through high-order reasoning over it. Secondly, inspired by the logic-driven framework of Li et al. (2019), we soften the logical properties through differentiable functions so as to incorporate them into multi-task learning objectives. The joint logic learning module enforces our model towards consistency with logic constraints across both TRE and SRE tasks. It is also a natural way to combine the supervision signals coming from two different tasks.

# Related Work

# Temporal relation extraction

Early studies usually employ statistical models combined with handcrafted features to extract temporal relations (Yoshikawa et al. 2009; Mani et al. 2006). These methods are of high computational complexity and are difficult to transfer to other types of relations.

Recently, with the rise of pre-trained language models, various new strategies have been applied to TRE task. To effectively model the long texts, some studies incorporate syntax information such as semantic trees or abstract meaning representation (AMR) to capture remote dependencies (Venkatachalam, Mutharaju, and Bhatia 2021; Zhang, Ning, and Huang 2022). Others construct global event graphs to enable the information flow among long-range events (Liu et al. 2021; Fei et al. 2022b). For example, Zhang, Ning, and Huang (2022) design a syntax-guided graph transformer to explore the temporal clues. Liu et al. (2021) build uncertainty-guided graphs to order the temporal events. To deepen the models' understanding of events, some studies propose to induce external knowledge (Han, Zhou, and Peng 2020; Tan, Pergola, and He 2023). Han, Zhou, and Peng (2020) propose an end-to-end neural model that incorporates domain knowledge from the TimeBank corpus (Pustejovsky et al. 2003). Some multi-task strategies are also employed for this task (Huang et al. 2023; Knez and Zitnik 2024). These strategies can facilitate the models' learning of complementary information from other tasks.

# Subevent relation extraction

In earlier studies, machine learning algorithms are utilized to identify the internal structure of events (Fei et al. 2022a;

Glavas et al. 2014). Afterwards, the introduction of deep learning has led to new advances of this task. Zhou et al. (2020) adopt the multi-task learning strategy and utilize duration prediction as auxiliary task. Man et al. (2022) argue that some context sentences in documents can facilitate the recognition of subevent relations. Thus, they adopt the reinforcement learning algorithm to select informative sentences from documents to provide supplementary information for this task. Hwang et al. (2022) enforce constraint strategies into probabilistic box embedding to maintain the unique properties of subevent relations.

To sum up, existing work only focuses on either TRE or SRE, only a few studies seek to resolve both two tasks. Zhuang, Fei, and Hu (2023b); Zhuang, Hu, and Zhao (2023) adopt the dependency parser and hard pruning strategies to acquire syntactic dependency graphs that are task-aware. Zhuang, Fei, and Hu (2023a) propose to use knowledge from event ontologies as additional prompts to compensate for the lack of event-related knowledge. Li and Geng (2024) introduce the features of event argument and structure to obtain graph-enhanced event embeddings. However, the above works regard event relation extraction as a multi-class classification task, and do not guarantee any coherence between different relation types, such as symmetry and transitivity. Different from these works, our LogicERE guarantee coherence through a high-order reasoning graph which is embedded with three essential logical constraints, as well as joint logic learning.

# Model

This paper focuses on the task of event-event relation extraction (ERE). The input document  $\mathcal{D}$  is represented as a sequence of  $n$  tokens  $\mathcal{D} = [x_1, x_2, \dots, x_n]$ . Each document contains a set of annotated event triggers (the most representative tokens for each event)  $\mathcal{E}_{\mathcal{D}} = \{e_1, e_2, \dots, e_k\}$ . The goal of ERE is to extract the multi-faceted event relations from the document. Particularly, we focus on two types of relations, i.e., Temporal and Subevent, corresponding to the label sets  $\mathcal{R}_{Temp}$  which contains BEFORE, AFTER, EQUAL, VAGUE, and  $\mathcal{R}_{Sub}$  which contains PARENT-CHILD, CHILD-PARENT, COREF, NOREL respectively. Note that each event pair is annotated with one relation type from either  $\mathcal{R}_{Temp}$  or  $\mathcal{R}_{Sub}$ , as the labels within two sets are mutually exclusive.

There are four major parts in our LogicERE model: (1) Sequence Encoder, which encodes event context in input document, (2) Logic Constraint Induced Graph, which build a event graph preserving logic constraints, (3) High-Order Reasoning Network on LCG, which performs high-order reasoning with relational graph transformer to obtain enhanced event and event pair embeddings, and (4) Joint Logic Learning, which further incorporates logic properties through well-designed learning objectives.

# Sequence Encoder

To obtain the event representations and contextualized embeddings of the input document  $\mathcal{D} = [x_t]_{t=1}^n$  (can be of any length  $n$ ), we leverage pre-trained RoBERTa (Liu et al.

2019) as a base encoder. We add special tokens "[CLS]" and "[SEP]" at the start and end of  $\mathcal{D}$ , and insert "<t>" and "</t>" at the start and end of all the events to mark event positions (Chen et al. 2022). Thus, we have:

$$
h _ {1}, h _ {2}, \dots , h _ {n ^ {\prime}} = \operatorname {E n c o d e r} ([ x _ {1}, x _ {2}, \dots , x _ {n ^ {\prime}} ]) \tag {1}
$$

where  $h_i \in \mathbb{R}^d$  is the embedding of token  $x_i$ . We employ the embeddings of token " [CLS]" and " <t>" to represent the document and the events respectively. If the document's length exceeds the limits of RoBERTa, we adopt the dynamic window mechanism to segment  $\mathcal{D}$  into several overlapping spans with specific step size, and input them into the encoder separately. Then, we average all the embeddings of " [CLS]" and " <t>" of different spans to obtain the document embedding  $h_{[\mathrm{CLS}]} \in \mathbb{R}^d$  or each event embedding  $h_{e_i} \in \mathbb{R}^d$ , respectively.

# Logic Constraint Induced Graph

Considering the logic constraints of event relation is based on the motivation that such logic properties comprehensively define the varied interactions among those events and relations. In this section, we construct a logic constraint induced graph (LCG) which can preserve these logic properties through unifying both event centered and event pair centered graphs. Specifically, given all the events of document  $\mathcal{D}$ , LCG is formulated as  $\mathcal{G} = \{\mathcal{V}, \mathcal{C}\}$ , where  $\mathcal{V}$  represents the nodes and  $\mathcal{C}$  represents the edges in the graph. We highlight the following differences of  $\mathcal{G}$  from previous event graphs and event pair graphs.

First, there are two types of nodes in  $\mathcal{V}$ , i.e., the event nodes  $\mathcal{V}_e$  and the event pair nodes  $\mathcal{V}_{ep}$ . Each node in  $\mathcal{V}_{ep}$  refers to a different pair of events from  $\mathcal{D}$ . Instead of merely using events or event pairs as nodes, LCG preserves both of them, enabling high-order interactions through edges.

Second, for edges  $\mathcal{C}$ , instead of using all edges between any two nodes, we design three types of edges following the logic constraints: (1) Event-event edges  $\mathcal{C}_{ee}$  for two events that are co-referenced, which is motivated by the coreference constraint in Introduction. These edges are optional.  $\mathcal{C}_{ee}$  contributes to event relation reasoning as co-referenced events are expected to share the same relations with other events. Meanwhile, no additional relations exist between co-referenced events. (2) Event pair-event pair edges  $\mathcal{C}_{pp}$  for two event pairs that share at least one event, which is motivated by the symmetry constraint and conjunction constraint in Introduction. Particularly, for the TRE task, symmetry constraint exists in a pair of reciprocal relations BEFORE and AFTER, as well as two reflexive ones EQUAL and VAGUE. Similarly, the SRE task includes reciprocal relations PARENT-CHILD and CHILD-PARENT as well as reflexive ones COREF and NOREL. The conjunction constraint enables the relation transitivity in a single task, and unifies the ordered nature of TRE and the topological nature of SRE (Wang et al. 2020). (3) Event-event pair edges  $\mathcal{C}_{ep}$  for an event pair and its corresponding events. We design  $\mathcal{C}_{ep}$  to bridge the information flow between events and event pairs.

# High-Order Reasoning Network on LCG

We perform high-order reasoning on LCG, which takes the relation heterogeneity into account and captures diversified high-order interactions within events and event pairs.

Initial Node Embeddings. For global inference, we firstly initialize node embeddings. Formally, for the event node  $e_i \in \mathcal{V}_e$ , we take the contextualized event embeddings from the sequence encoder for initialization:

$$
v _ {e _ {i}} ^ {(0)} = h _ {e _ {i}} \mathbf {W} _ {n} \tag {2}
$$

where 0 indicates the initial state and  $\mathbf{W}_n\in \mathbb{R}^{d\times 2d}$  is a learnable weight matrix.

For the event pair node  $e_{i,j} \in \mathcal{V}_{ep}$ , we concatenate two corresponding event embeddings:

$$
v _ {e _ {i, j}} ^ {(0)} = \left[ h _ {e _ {i}} \right\| h _ {e _ {j}} ] \tag {3}
$$

Node Embedding Update. Then, we adopt relational graph transformer (Bi et al. 2024) to enhance the node features with the relational information from neighbor nodes. Each layer  $l$  is similar to the transformer architecture. It takes a set of node embeddings  $\mathbf{V}^{(l)}\in \mathbb{R}^{N\times d_{in}}$  as input, and outputs a new set of node embeddings  $\mathbf{V}^{(l + 1)}\in \mathbb{R}^{N\times d_{out}}$  where  $N = |\mathcal{V}_e| + |\mathcal{V}_{ep}|$  is the number of nodes in LCG,  $d_{in}$  and  $d_{out}$  are the dimensions of input and output embeddings.

In each layer, to integrate information from each neighbor, we adopt a shared self-attention mechanism (Vaswani et al. 2017) to calculate the attention score:

$$
\alpha_ {i j} = \operatorname {s o f t m a x} \left(c o _ {i j}\right) \tag {4}
$$

$$
c o _ {i j} = \frac {\left(v _ {i} \mathbf {W} _ {q}\right) \left(v _ {j} \mathbf {W} _ {k}\right) ^ {\mathrm {T}}}{\sqrt {d _ {k}}} \tag {5}
$$

where  $N_{i}$  is the first order neighbor set of node  $i$ ,  $co_{ij}$  measures the importance of neighbor  $j$  to  $i$ ,  $\mathbf{W}_q, \mathbf{W}_k \in \mathbb{R}^{d_{in} \times d_k}$  are learnable matrices,  $d_k$  is a scaling factor to assign lower attention weights to uninformative nodes.

Then we aggregate relational knowledge from the neighborhood information with weighted linear combination of the embeddings:

$$
v _ {i} ^ {(l + 1)} = \sum_ {j \in N _ {i}} \alpha_ {i j} ^ {(l)} \left(v _ {j} ^ {(l)} \mathbf {W} _ {v} ^ {(l)}\right) \tag {6}
$$

where  $\mathbf{W}_v^{(l)}\in \mathbb{R}^{d_{in}\times d_k}$  is a learnable matrix. We also adopt multi-head attention to attend to information from multiple attention heads. Thus, the output of the  $l$  -th layer for node  $i$  is:

$$
v _ {i} ^ {(l + 1)} = \left(\left| \right| _ {c = 1} ^ {C} \sum_ {j \in N _ {i}} \alpha_ {i j} ^ {(l)} \left(v _ {j} ^ {(l)} \mathbf {W} _ {v} ^ {(l)}\right)\right) \mathbf {W} _ {o} ^ {(l)} \tag {7}
$$

where  $C$  is the number of attention head and  $\mathbf{W}_o^{(l)}\in$ $\mathbb{R}^{Cd_k\times d_{out}}$  is a learnable matrix.

Measure Edge Heterogeneity. It is intuitive that three types of edges in LCG contributes differently to ERE. Thus we propose to measure the edge heterogeneity and incorporate the edge features into node embeddings. Specifically, for each edge type in LCG, we learn a scalar:

$$
\beta_ {t} = r _ {t} \mathbf {W} _ {r} \tag {8}
$$

where  $1 \leq t \leq T$ ,  $T$  is the number of edge types,  $r_t \in \mathbb{R}^{1 \times d}$  denotes the edge features specific to the edge type,  $\mathbf{W}_r \in \mathbb{R}^{d \times 1}$  is a learnable matrix.  $r_t$  will be randomly initialized. Then we incorporate  $\beta_t$  as the attention bias into the attention score to adjust the interaction strength between two adjacent nodes:

$$
\widetilde {\alpha} _ {i j} = \operatorname {s o f t m a x} \left(\beta_ {t} + c o _ {i j}\right) \tag {9}
$$

As the result, the final updated node embeddings considering the edge heterogeneity is:

$$
\tilde {v} _ {i} ^ {(l + 1)} = \left(\left. \right| \sum_ {c = 1} ^ {C} \tilde {\alpha} _ {i j} ^ {(l)} \left(v _ {j} ^ {(l)} \mathbf {W} _ {v} ^ {(l)}\right)\right) \mathbf {W} _ {o} ^ {(l)} \tag {10}
$$

By stacking multiple layers, the reasoning network could reach high-order interaction and maintain logic properties.

Learning and Classification. To predict whether there is the temporal or subevent relation between events  $e_i$  and  $e_j$ , we concatenate the embeddings of " [CLS]",  $e_i$ ,  $e_j$  and the corresponding event pair as the logic enhanced representation. Thus, the probability distribution of the relation can be obtained through linear classification:

$$
p _ {e _ {i, j}} = \operatorname {s o f t m a x} \left(\left[ h _ {\left[ \mathrm {C L S} \right]} \right] \left| \tilde {v} _ {i} \right| \left| \tilde {v} _ {j} \right| \left| \tilde {v} _ {i, j} \right] \mathbf {W} _ {p}\right) \tag {11}
$$

where  $||$  denotes concatenation and  $\mathbf{W}_p$  is a learnable matrix.

For training, we adopt cross-entropy as the loss function:

$$
\mathcal {L} _ {1} = - \sum_ {e _ {i}, e _ {j} \in \mathcal {E} _ {\mathcal {D}}} \left(1 - y _ {e _ {i, j}}\right) \log \left(1 - p _ {e _ {i, j}}\right) + y _ {e _ {i, j}} \log \left(p _ {e _ {i, j}}\right) \tag {12}
$$

where  $y_{e_{i,j}}$  denotes the golden label.

# Joint Logic Learning

Inspired by the logic-driven framework for consistency of Li et al. (2019), we further design two learning objectives by directly transforming the logical constraints into differentiable loss functions<sup>2</sup>.

Symmetry Constraint. Symmetry constraints indicate the event pair with flipping orders will have the reversed relation, the logical formula can be written as:

$$
\bigwedge_ {e _ {i}, e _ {j} \in \mathcal {E} _ {\mathcal {D}}, r \in \mathcal {R} _ {s y m}} r \left(e _ {i}, e _ {j}\right) \leftrightarrow \bar {r} \left(e _ {j}, e _ {i}\right) \tag {13}
$$

where  $\mathcal{R}_{sym}$  indicates the set of relations enforcing the symmetry constraint. We use the product t-norm and transformation to the negative log space and obtain the symmetry loss:

$$
\mathcal {L} _ {s y m} = \sum_ {e _ {i}, e _ {j} \in \mathcal {E} _ {\mathcal {D}}} | \log \left(p _ {e _ {i, j}}\right) - \log \left(\bar {p} _ {e _ {j, i}}\right) | \tag {14}
$$

Conjunction Constraint. Conjunctive constraint are applicable to any three related events  $e_i, e_j$  and  $e_k$ . It contributes

to the joint learning of TRE and SRE. The conjunction constraint enforces the following logical formulas:

$$
\bigwedge_ {\substack {e _ {i}, e _ {j}, e _ {k} \in \mathcal {E} _ {\mathcal {D}}\\r _ {1}, r _ {2} \in \mathcal {R}, r _ {3} \in D e (r _ {1}, r _ {2})}} r _ {1} \left(e _ {i}, e _ {j}\right) \wedge r _ {2} \left(e _ {j}, e _ {k}\right) \rightarrow r _ {3} \left(e _ {i}, e _ {k}\right) \tag{15}
$$

$$
\bigwedge_ {\substack {e _ {i}, e _ {j}, e _ {k} \in \mathcal {E} _ {\mathcal {D}}\\r _ {1}, r _ {2} \in \mathcal {R}, r _ {4} \notin D e (r _ {1}, r _ {2})}} r _ {1} \left(e _ {i}, e _ {j}\right) \wedge r _ {2} \left(e _ {j}, e _ {k}\right) \rightarrow \neg r _ {4} \left(e _ {i}, e _ {k}\right) \tag{16}
$$

where  $De(r_1, r_2)$  is a set composed of all relations from  $\mathcal{R}$  that do not conflict with  $r_1$  and  $r_2$ .

Similarly, the loss function specific to conjunction constraint is:

$$
\mathcal {L} _ {\text {c o n j}} = \sum_ {e _ {i}, e _ {j}, e _ {k} \in \mathcal {E} _ {\mathcal {D}}} | \mathcal {L} _ {c _ {1}} | + \sum_ {e _ {i}, e _ {j}, e _ {k} \in \mathcal {E} _ {\mathcal {D}}} | \mathcal {L} _ {c _ {2}} | \tag {17}
$$

$$
\mathcal {L} _ {c _ {1}} = \log \left(p _ {e _ {i, j}}\right) + \log \left(p _ {e _ {j, k}}\right) - \log \left(p _ {e _ {i, k}}\right) \tag {18}
$$

$$
\mathcal {L} _ {c _ {2}} = \log \left(p _ {e _ {i, j}}\right) + \log \left(p _ {e _ {j, k}}\right) - \log \left(1 - p _ {e _ {i, k}}\right) \tag {19}
$$

The final loss function combines the above logic learning and event relation learning objectives, where  $\gamma$  are nonnegative coefficients to control the influence of each loss term:

$$
\mathcal {L} = \mathcal {L} _ {1} + \gamma_ {\text {s y m}} \mathcal {L} _ {\text {s y m}} + \gamma_ {\text {c o n j}} \mathcal {L} _ {\text {c o n j}} \tag {20}
$$

# Conclusion

We present a novel logic induced high-order reasoning network (LogicERE) to enhance the event relation reasoning with logic constraints. We first design a logic constraint induced graph (LCG) which contains interactions between events and among event pairs. Then we encode the heterogeneous LCG for high-order event relation reasoning while

maintaining inherent logic properties of event relations. Finally, we further incorporate logic constraints with joint logic learning. Extensive experiments show that LogicERE can effectively utilize logic properties to enhance the event and event pair embeddings, and achieve state-of-the-art performance for both TRE and SRE. The joint learning evaluation reveals that LogicERE can effectively maintain global consistency of two types relations, assisting in the comprehension of both temporal and subevent relation.

# AA. Details of Joint Logic Learning

With the logical formulas of corresponding logic constraints, we now focus on the unification of discrete declarative logic constraints with the loss-driven learning paradigm. To address this, we use relaxations of logic in the form of t-norms

<table><tr><td></td><td>PC</td><td>CP</td><td>CR</td><td>NR</td><td>BF</td><td>AF</td><td>EQ</td><td>VG</td></tr><tr><td>PC</td><td>PC, ¬AF</td><td>\</td><td>PC, ¬AF</td><td>¬CP, ¬CR</td><td>BF, ¬CP, ¬CR</td><td>\</td><td>BF, ¬CP, ¬CR</td><td>\</td></tr><tr><td>CP</td><td>\</td><td>CP, ¬BF</td><td>CP, ¬BF</td><td>¬PC, ¬CR</td><td>\</td><td>AF, ¬PC, ¬CR</td><td>AF, ¬PC, ¬CR</td><td>\</td></tr><tr><td>CR</td><td>PC, ¬AF</td><td>CP, ¬BF</td><td>CR, EQ</td><td>NR</td><td>BF, ¬CP, ¬CR</td><td>AF, ¬PC, ¬CR</td><td>EQ</td><td>VG</td></tr><tr><td>NR</td><td>¬CP, ¬CR</td><td>¬PC, ¬CR</td><td>NR</td><td>\</td><td>\</td><td>\</td><td>\</td><td>\</td></tr><tr><td>BF</td><td>BF, ¬CP, ¬CR</td><td>\</td><td>BF, ¬CP, ¬CR</td><td>\</td><td>BF, ¬CP, ¬CR</td><td>\</td><td>BF, ¬CP, ¬CR</td><td>¬AF, ¬EQ</td></tr><tr><td>AF</td><td>\</td><td>AF, ¬PC, ¬CR</td><td>AF, ¬PC, ¬CR</td><td>\</td><td>\</td><td>AF, ¬PC, ¬CR</td><td>AF, ¬PC, ¬CR</td><td>¬BF, ¬EQ</td></tr><tr><td>EQ</td><td>¬AF</td><td>¬BF</td><td>EQ</td><td>\</td><td>BF, ¬CP, ¬CR</td><td>AF, ¬PC, ¬CR</td><td>EQ</td><td>VG, ¬CR</td></tr><tr><td>VG</td><td>\</td><td>\</td><td>VG, ¬CR</td><td>\</td><td>¬AF, ¬EQ</td><td>¬BF, ¬EQ</td><td>VG</td><td>\</td></tr></table>


Figure 2: The induction table for conjunctive constraints on temporal and subevent relations. The abbreviations PC, CP, CR, NR, BF, AF, EQ and VG denote PARENT-CHILD, CHILD-PARENT, COREF, NOREL, BEFORE, AFTER, EQUAL and VAGUE, respectively. Subevent relations are in black, and temporal relations are in blue. “\” denotes no constraints.


<table><tr><td>Name</td><td>Boolean Logic</td><td>Product</td></tr><tr><td>Negation</td><td>¬A</td><td>1 - a</td></tr><tr><td>T-norm</td><td>A ∧ B</td><td>ab</td></tr><tr><td>T-conorm</td><td>A ∨ B</td><td>a + b - ab</td></tr><tr><td>Residuum</td><td>A → B</td><td>min(1, b/a)</td></tr></table>

Table 7: Mapping discrete statements into differentiable functions. Literals are upper-cased, and real-valued probabilities are lower-cased.

to deterministically compile rules into differentiable loss functions. Different t-norms map the Boolean operations into different continuous functions.

We use the product t-norm as it strictly generalizes the widely used cross entropy loss. The mapping of standard Boolean operations into continuous functions for product t-norm is shown in Table 7.

For the symmetry constraint, we have:

$$
\bigwedge_ {e _ {i}, e _ {j} \in \mathcal {E} _ {\mathcal {D}}, r \in \mathcal {R} _ {s y m}} r \left(e _ {i}, e _ {j}\right) \leftrightarrow \bar {r} \left(e _ {j}, e _ {i}\right) \tag {21}
$$

where  $\mathcal{R}_{sym}$  indicates the set of relations enforcing the symmetry constraint.

Using the product t-norm, we get:

$$
\prod_ {e _ {i}, e _ {j} \in \mathcal {E} _ {\mathcal {D}}, r \in \mathcal {R} _ {s y m}} \min  \left(1, \bar {r} _ {\left(e _ {j}, e _ {i}\right)} / r _ {\left(e _ {i}, e _ {j}\right)}\right) \min  \left(1, r _ {\left(e _ {i}, e _ {j}\right)} / \bar {r} _ {\left(e _ {j}, e _ {i}\right)}\right) \tag {22}
$$

Transforming to the negative log space, we get the symmetry loss:

$$
\mathcal {L} _ {s y m} = \sum_ {e _ {i}, e _ {j} \in \mathcal {E} _ {\mathcal {D}}} \left| \log r _ {(e _ {i}, e _ {j})} - \log \bar {r} _ {(e _ {j}, e _ {i})} \right| \tag {23}
$$

For the conjunction constraint, the induction table for conjunctive constraints on temporal and subevent relations is shown in Figure 2.

we have:

$$
\bigwedge_ {\substack {e _ {i}, e _ {j}, e _ {k} \in \mathcal {E} _ {\mathcal {D}}\\r _ {1}, r _ {2} \in \mathcal {R}, r _ {3} \in D e \left(r _ {1}, r _ {2}\right)}} r _ {1} \left(e _ {i}, e _ {j}\right) \wedge r _ {2} \left(e _ {j}, e _ {k}\right) \rightarrow r _ {3} \left(e _ {i}, e _ {k}\right) \tag{24}
$$

$$
\bigwedge_ {\substack {e _ {i}, e _ {j}, e _ {k} \in \mathcal {E} _ {\mathcal {D}}\\r _ {1}, r _ {2} \in \mathcal {R}, r _ {4} \notin D e (r _ {1}, r _ {2})}} r _ {1} \left(e _ {i}, e _ {j}\right) \wedge r _ {2} \left(e _ {j}, e _ {k}\right) \rightarrow \neg r _ {4} \left(e _ {i}, e _ {k}\right) \tag{25}
$$

where  $De(r_1, r_2)$  is a set composed of all relations from  $\mathcal{R}$  that do not conflict with  $r_1$  and  $r_2$ .

Similarly, we get the loss function for the conjunction constraint:

$$
\mathcal {L} _ {\text {c o n j}} = \sum_ {e _ {i}, e _ {j}, e _ {k} \in \mathcal {E} _ {\mathcal {D}}} | \mathcal {L} _ {c _ {1}} | + \sum_ {e _ {i}, e _ {j}, e _ {k} \in \mathcal {E} _ {\mathcal {D}}} | \mathcal {L} _ {c _ {2}} | \tag {26}
$$

$$
\mathcal {L} _ {c _ {1}} = \log \left(p _ {e _ {i, j}}\right) + \log \left(p _ {e _ {j, k}}\right) - \log \left(p _ {e _ {i, k}}\right) \tag {27}
$$

$$
\mathcal {L} _ {c _ {2}} = \log \left(p _ {e _ {i, j}}\right) + \log \left(p _ {e _ {j, k}}\right) - \log \left(1 - p _ {e _ {i, k}}\right) \tag {28}
$$