# TIMERS: Document-level Temporal Relation Extraction

# Abstract

We present TIMERS - a TIME, Rhetorical and Syntactic-aware model for document-level temporal relation classification. Our proposed method leverages rhetorical discourse features and temporal arguments from semantic role labels, in addition to traditional local syntactic features, trained through a Gated Relational-GCN. Extensive experiments show that the proposed model outperforms previous methods by  $5 - 18\%$  on the TDDiscourse, TimeBank-Dense, and MATRES datasets due to our discourse-level modeling.

# 1 Introduction

Temporal relation extraction (TempRel) is a challenging task that involves determining the temporal order between two events in a text (Pustejovsky et al., 2003). Understanding the temporal ordering of events in a document plays a key role in downstream tasks such as timeline creation (Leeuwenberg and Moens, 2018), time-aware summarization (Noh et al., 2020), temporal question-answering (Ning et al., 2020), and temporal information extraction (Leeuwenberg and Moens, 2019).

Prior work focuses on extracting temporal relations between event pairs (a.k.a., TLINKS) present in the same sentence (Intra-sentence TLINKS) or adjacent sentences (Inter-sentence TLINKS), mostly ignoring document-level pairs (Cross-document TLINKS) (Reimers et al., 2016). Past works have used RNN (Cheng and Miyao, 2017; Meng et al., 2017; Goyal and Durrett, 2019; Ning et al., 2019; Han et al., 2019a,c,b, 2020b) and Transformer networks (Ballesteros et al., 2020; Zhao et al., 2020b) for encoding a few sentences or a short paragraph but do not capture long-range dependencies and multi-hop reasoning at the document-level. This shortcoming is shown in the TDDiscourse dataset (Naik et al., 2019), which was

designed to highlight global discourse-level challenges, e.g., multi-hop chain reasoning, future or hypothetical events, and reasoning requiring world knowledge.

We propose TIMERS - a TIME, Rhetorical, and Syntactic-aware model for document-level temporal relation extraction. TIMERS uses discourse features in the form of connections from Rhetorical Structure Theory (RST) parsers (Bhatia et al., 2015) to leverage long-range inter-sentential relationships. It also extends existing contextual embeddings with structural and syntactic dependency parse connections. Lastly, it uses timextimex relations,  $dct$  (document creation date)-timex relations, and temporal arguments obtained via sentence-level semantic role labeling. These rhetorical, syntactic, and temporal features are learned through a modified version of Relational Graph Convolutional Networks (R-GCN) with a gating mechanism (GR-GCN) (Schlichtkrull et al., 2018), which learns highly relational data relationships in densely-connected graph networks.

Our main contribution is a document-level model that incorporates these three features to improve temporal relationship extraction. We obtain state-of-the-art performance across three datasets with  $5 - 18\%$  relative improvement, showing improvement for events that require chain reasoning, causal prerequisite links, and future events.

# 2 Methodology

Let document  $D$  be defined as a sequence of  $n$  tokens  $w_{i}\in W = \{w_{1},\dots ,w_{n}\}$  . The entire document is a list of  $m$  sentences  $V = [v_{1},\dots ,v_{m}]$  Each document has a set of  $p$  events  $E = \{e_1,\dots ,e_p\}$  and  $q$  timexes  $T = \{t_1,\dots ,t_q\}$  where  $p,q\leq n$  . The creation date of the document is represented by timestamp  $t_{DCT}$  . We denote the source and target events by  $e_s$  and  $e_t$  , respec

![](images/e39cfb7d9d6fd821c1fc792aff1bffe35b6ed573217b4b40c7fbe4dd6accae38.jpg)



Figure 1: Three graphs are created from the input document. Time-aware Graph  $(G_{TG})$ : DCT-Timex associations, Timex-Timex associations, and Temporal Argument connections from semantic role labels; Syntactic-aware Graph  $(G_{SG})$ : structural and syntactic connections; and Rhetoric-aware Graph  $(G_{DG})$ : rhetorical relations between EDU's  $(h_i)$ .


tively. The task is to identify the temporal relation  $y \in R$  between the source and target event in a multi-class classification setup, where  $R$  is the set of all possible temporal links (TLINKs).

To solve this task, our model (Fig.1) builds the TIMERS-graph, which consists of a Syntactic Graph (Sec.2.1), a Time Graph (Sec. 2.2), and a Rhetorical Graph (Sec.2.3). Each graph is learned through GR-GCN to extract the embeddings used for temporal relation extraction (Fig.2, Sec.2.4).

# 2.1 Syntactic-Aware Graph

The syntactic graph captures the document structure and word dependency. Our syntactic-aware graph  $(\mathcal{G}_{SG})$  is made of separate nodes to represent the document  $D$ , each of its inherent sentences  $v_{i} \in V$ , and all the constituent words  $w_{i} \in W$  of each sentence. The edges of the Syntactic Graph encode five relations: (1) Document-Sentence Affiliation and (2) Sentence-Word Affiliation model the hierarchical structure of the document through a directed edge from the document node to each sentence node and from a sentence node to each word in the sentence. (3) Sentence-Sentence Adjacency and (4) Word-Word Adjacency to preserve sequential ordering for consecutive sentence and word nodes. (5) Word-Word Dependency

![](images/2595990f4c1fffeae188126c7370023aaf9fd5c86c441afdab67879e57dc7d33.jpg)



Figure 2: TIMERS learns rhetorical, syntactic, and temporal features through a Gated Relational-Graph Convolutional Networks (GR-GCN). The output of  $G_{SG}$  forms the input of  $G_{TG}$ . The output corresponding to the source and target nodes learned by  $G_{TG}(O_T)$  and  $G_{DG}(O_{EDU})$  are concatenated with the output of the BERT based context encoder  $(O_{CE})$ , which forms the final output  $h_G$  that passes through the Softmax layer to predict the temporal relation.


encodes the syntactical nature of the word-level relationships by adding an undirected edge between two word nodes if they share a parent-child relationship in the sentence-level dependency tree.

We use BERT to encode each  $w_{i}$  and obtain sentence embeddings  $v_{i}^{\prime}$  by averaging the second-to-last hidden layer of BERT for each token. The document vector embedding  $D_{i}^{\prime}$  was calculated as the average of all sentence embedding  $(D_{i}^{\prime} = \sum_{i = 0}^{m}v_{i}^{\prime})$

# 2.2 Time-Aware Graph

When events are anchored to a specific time, it becomes easier to infer event relationships from their associated date and time. The time-aware graph  $(\mathcal{G}_{TG})$  exploits this intuition and propagates relational information among events, timexes, and the Document Creation Time (DCT). The document node  $D$  is the node corresponding to the document creation date while the timexes  $t_i$  and events  $e_i$  are characterized by their corresponding word nodes in the Syntactic Graph. We design three types of edge connections: (1) DCT-Timex Association: exploit the ordering of timexes with respect to the document creation time through directed weighted edges from DCT to timexes. (2) Timex-Timex Association: capture inherent non-local timeline ordering between timex pairs by a

directed weighted edge. (3) Predicate-Temporal Argument: anchor local temporal relations at the sentence level by connecting each event verb predicate to its temporal argument with a directed edge. The connections formed between temporal entities help navigate information from the source event to the target event while exploring interactions with other events, timexes,  $dct$ , and temporal arguments.

We calculate timestamps for timexes and the  $DCT$  from the annotated TimeML format of input documents. The weight of the  $DCT$ -timex and timex-timex edges is determined based on the temporal order of the entities  $\{After, Before, Simultaneous, None\}$ . We added None as a relation when one of the timestamps cannot be anchored in time.

# 2.3 Rhetorical-Aware Graph

We use discourse features based on Rhetorical Structure Theory (RST) (Mann and Thompson, 1988) to leverage long-range inter-dependencies through a discourse tree. The rhetorical discourse tree of a document contains nodes of phrases, where each phrase (a.k.a, Elementary Discourse Unit or EDU) is contiguous, adjacent and nonoverlapping. The interdependencies among EDUs are represented by conventional rhetorical relations (Mann, 1987), e.g. Elaboration, Span, Condition, Attribution. Prior work showed discourse features in the form of RST connections help leverage long-range document-level interactions between phrase units (Bhatia et al., 2015) and identify background foreground events (Aldawsari et al., 2020).

Elementary Discourse Unit (EDU), a subsentence phrase unit, is the minimal selection unit for discourse segmentation of a document. We generate the document vector representations at EDU-level  $h_i \in H = \{h_1, \dots, h_d\}$  via the Self-Attentive Span Extractor (SpanExt) from Lee et al. (2017) over the BERT token embeddings. We use the converted dependency version of the tree to build the Rhetorical-aware graph  $(\mathcal{G}_{DG})$  by treating every discourse dependency from the  $i$ -th EDU to the  $j$ -th EDU as a directed edge weighted by the type of the rhetorical relation.

# 2.4 Temporal Relation Extraction

Each graph is instantiated as a gated variant of Relational Graph Convolutional Networks (R-GCN) (Schlichtkrull et al., 2018), which we term as Gated Relational Graph Convolution Network (GR-GCN). GR-GCN propagates messages among the nodes to

<table><tr><td>Dataset</td><td>Train</td><td>Validation</td><td>Test</td><td>Labels</td></tr><tr><td>TDDMan (Naik et al., 2019)</td><td>4000</td><td>650</td><td>1500</td><td>a, b, s, i, ii</td></tr><tr><td>TDDAuto (Naik et al., 2019)</td><td>32609</td><td>1435</td><td>4258</td><td>a, b, s, i, ii</td></tr><tr><td>MATRES (Ning et al., 2018a) ##</td><td>231</td><td>25</td><td>20</td><td>e,a,b,v</td></tr><tr><td>TimeBank-Dense (Cassidy et al., 2014)</td><td>4032</td><td>629</td><td>1427</td><td>a, b, s, i, ii, v</td></tr></table>


Table 1: Train/Val/Test data distribution for TDDMan, TDDAuto, MATRES, and TimeBank-Dense; a: After, b: Before, s: Simultaneous, i: Includes, ii: Is included, v: Vague, e: Equal. (# Ning et al. (2019) use TimeBank and Aquaint for training, Platinum for test;  $20\%$  of train as validation)


<table><tr><td>Corpus</td><td>Model</td><td>F1</td></tr><tr><td rowspan="5">TB-Dense</td><td>Vashishta et al. (2019)</td><td>56.6</td></tr><tr><td>EventPlus (Ma et al., 2021)</td><td>64.5</td></tr><tr><td>CTRL-PG (Zhou et al., 2020)</td><td>65.2</td></tr><tr><td>DEER (Han et al., 2020a)</td><td>66.8</td></tr><tr><td>TIMERS (ours)</td><td>67.8</td></tr><tr><td rowspan="9">MATRES</td><td>CogCompTime (Ning et al., 2018b)</td><td>66.6</td></tr><tr><td>Goyal and Durrett (2019)</td><td>68.61</td></tr><tr><td>BiLSTM+MAP (Han et al., 2019c)</td><td>75.5</td></tr><tr><td>EventPlus (Ma et al., 2021)</td><td>75.5</td></tr><tr><td>Wang et al. (2020)</td><td>78.8</td></tr><tr><td>DEER (Han et al., 2020a)</td><td>79.3</td></tr><tr><td>Zhao et al. (2020a)</td><td>79.6</td></tr><tr><td>SMTL (Ballesteros et al., 2020)</td><td>81.6</td></tr><tr><td>TIMERS (ours)</td><td>82.3</td></tr></table>

Table 2: Comparison of TIMERS with recent state-of-the-art models on TimeBank-Dense and MATRES dataset. TIMERS outperforms all recent top-performing systems.

obtain a learned node representation and is inspired by (Zhang et al., 2020). Fig. 2 shows how the learned representations obtained from the syntactic-aware graph forms the input to the time-aware graph. For the time-aware graphs, the learned representations of nodes corresponding to the source event  $e_{s}$  and target event  $e_{t}$  are extracted  $(O_{T})$ . In the case of the rhetorical graphs, the span representations of the EDU span nodes corresponding to the source event  $(h_{e})$  and target event  $(h_{s})$  are extracted  $(O_{EDU})$ .

The output corresponding to the source and target nodes learnt by  $G_{TG}(O_T)$  and  $G_{DG}(O_{EDU})$  are concatenated with output of BERT based context encoder  $(O_{CE})$  (similar to BERT encoding in (Zhao et al., 2020a)):  $z_G = \mathrm{ReLU}(W[O_T; O_{EDU}; O_{CE}] + b)$ . This is followed by a Softmax layer to predict temporal relations.

# 4 Conclusion

This work presents a neural architecture that utilizes local syntactic features, rhetorical discourse features, and temporal arguments in semantic role labels through a Gated Relational-GCN for document-level temporal relation extraction on TDDiscourse, MATRES, and TimeBank-Dense datasets. Experiments show that TIMERS shows substantial improvement for events that require chain reasoning and causal prerequisite links. Future work will focus on exploring real-world scenarios in which the temporal extraction task suffers from absent or erroneous event and timex annotations. We believe our proposed methods can also be adapted for other languages as well by overcoming possible limitations such as dependency parsing, semantic parsing, Timex normalization for the non-English corpora.

# A Experiment Settings

# A.1 Node Connections

We detail the node connections present in each graph of our proposed model along with edge attributes in Table 4.

# A.2 Edge Relations

Table 6 lists rhetorical relations used in Rhetoric-aware graph  $G_{DG}$  in the TIMERS model, along with the definitions as provided by Mann (1987). The weights of the Rhetoric graph  $G_{DG}$  are determined based on the RST relations described in this table. Table 7 details the type of relations between timex-timex and DCT-timex nodes of the Time-aware graph  $G_{TG}$ .

# A.3 Training Setup

Hyperparameter: Hyper-parameters for our model were tuned on the respective validation set to find the best configurations for different datasets. We summarize the range of our model's hyper parameters such as: number of hidden layers in GR-GCN  $\{1,2,3\}$ , size of hidden layers in GR-GCN  $\{64,128,256,512\}$ , BERT embedding size, dropout  $\delta \in \{0.2,0.3,0.4,0.5,0.6\}$ , learning rate  $\lambda \in \{1e - 5,1e - 4,1e - 3,1e - 2,1e - 1\}$ , weight decay  $\omega \in \{1e - 6,1e - 5,1e - 4,1e - 3\}$ , batch size  $b\in \{16,32,64\}$  and epochs  $(\leq 100)$ .

Contextual Encoder: We used BERT-base-uncased for generating token embedding of size 1x 768. As BERT-base Transformer provides a stronger baseline as compared to RoBERTa, we utilized BERT Transformer for Contextual Encoder in TIMERS architecture. We use the default dropout rate (0.1) on BERT's self attention layers but do not use additional dropout at the top linear layer The output from the Contextual Encoder is a 1-D vector of size 768.

Loss Function and Inference: TIMERS is trained end to end using Binary Cross Entropy loss with Adam optimizer. Across all four datasets, we found the best results correspond with the use of Adam optimiser set with default values  $\beta_{1} = 0.9$ ,  $\beta_{2} = 0.999$ ,  $\epsilon = 1e - 8$ , weight-decay of  $5e - 4$  and an initial learning rate of 0.001. We evaluate the performance of temporal relation extraction systems in terms of F1, precision and recall score.

Computing Infrastructure: TIMERS is written in PyTorch library and was trained on Nvidia GeForce RTX 2080 GPU. Average Runtime: The model takes a maximum of approximately 6,500

seconds to train on either of the four datasets.

Dataset Access Links to download TD-Discourse (Naik et al., 2019) dataset: https://github.com/aakanksha19/TDDiscourse Link to download MATRES (Ning et al., 2018a) dataset: https://github.com/qiangning/MATRES Link to download TimeBank-Dense (Cassidy et al., 2014) dataset: https://github.com/muk343/TimeBank-dense

# A.4 Reproducibility

Table 5 lists the range ad best values of the hyperparameters used in TIMERS model for different data settings. We used grid search to choose the best set of training configurations across each dataset. We run 5 rounds of hyper-parameter search trials and report average of observed results.

# B Additional Results

We observe from Figure 4 a similar trend to TDDMan, although with a stronger support for SS, CR, TI and FE. This is partly due to the fact that TDDAuto was generated automatically (Naik et al., 2019) using weakly annotated time relations. Moreover,  $90\%$  of samples in TDDAuto require SS. Hence, TIMERS trained exclusively on TDDAuto performs worse on challenging phenomenon like HN and CP. Consistent with results on TDDMan, TIMERS and its ablations trained on TDDAuto struggle on EC and WK.

<table><tr><td>Edge</td><td>Graph</td><td>Source</td><td>Target</td><td>Directed</td><td>Weighted</td></tr><tr><td>Document-Sentence Affiliation</td><td>Syntactic</td><td>Doc Node</td><td>Sent Nod</td><td>✓</td><td>✗</td></tr><tr><td>Sentence-Word Affiliation</td><td>Syntactic</td><td>Sent Nod</td><td>Word Node</td><td>✓</td><td>✗</td></tr><tr><td>Sentence-Sentence Adjacency</td><td>Syntactic</td><td>Sent Nod</td><td>Sent Nod</td><td>✓</td><td>✗</td></tr><tr><td>Word-Word Adjacency</td><td>Syntactic</td><td>Word Node</td><td>Word Node</td><td>✓</td><td>✗</td></tr><tr><td>Word-Word Dependency</td><td>Syntactic</td><td>Word Node</td><td>Word Node</td><td>✗</td><td>✗</td></tr><tr><td>DCT-Timex Association</td><td>Time</td><td>Doc Node</td><td>Timex</td><td>✓</td><td>✓</td></tr><tr><td>Timex-Timex Association</td><td>Time</td><td>Timex</td><td>Timex</td><td>✓</td><td>✓</td></tr><tr><td>Predicate-Temporal Argument</td><td>Time</td><td>Word Node</td><td>Timex</td><td>✗</td><td>✗</td></tr><tr><td>RST Discourse</td><td>Discourse</td><td>EDU</td><td>EDU</td><td>✓</td><td>✓</td></tr></table>


Table 4: List of node connections in TIMERS.


<table><tr><td></td><td colspan="4">Dataset</td></tr><tr><td>Hyperparameters</td><td>TDDMan</td><td>TDDAuto</td><td>MATRES</td><td>TB-Dense</td></tr><tr><td>Dropout Ratio</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Adam</td><td>Adam</td><td>Adam</td></tr><tr><td>Input Dimension (Context Encoder)</td><td>(n,768)</td><td>(n,768)</td><td>(n,768)</td><td>(n,768)</td></tr><tr><td>Input Dimension (Syntactic Graph)</td><td>(n,768)</td><td>(n,768)</td><td>(n,768)</td><td>(n,768)</td></tr><tr><td>Input Dimension (Time Graph)</td><td>(n,256)</td><td>(n,256)</td><td>(n,64)</td><td>(n,64)</td></tr><tr><td>Input Dimension (Rhetoric Graph)</td><td>(n,768)</td><td>(n,768)</td><td>(n,768)</td><td>(n,768)</td></tr><tr><td>Hidden Dimension (GR-GCN)</td><td>256</td><td>256</td><td>64</td><td>64</td></tr><tr><td>Number of hidden layers (GR-GCN)</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Hidden Dimension of SpanExt</td><td>{256, 64}</td><td>{256, 64}</td><td>{128, 64}</td><td>{128, 64}</td></tr><tr><td>Epochs</td><td>20</td><td>20</td><td>20</td><td>20</td></tr><tr><td>Batch Size</td><td>8</td><td>8</td><td>16</td><td>16</td></tr><tr><td>Activation Function of Linear layers</td><td>ReLU</td><td>ReLU</td><td>ReLU</td><td>ReLU</td></tr><tr><td>Dimension of final FCN</td><td>[(1792 x r)]</td><td>[(1792 x r)]</td><td>[(1024 x r)]</td><td>[(1024 x r)]</td></tr><tr><td>Output Classes</td><td>5</td><td>5</td><td>4</td><td>5</td></tr></table>


Table 5: Hyperparameters Details: Training hyperparameters of TIMERS for TDDMan, TDDAuto, MATRES and TB-Dense datasets. n refers to the number of input samples; r refers to the number of total relation classes


![](images/d131da7965fc8542c7a35764aa92aa6feec474268429db6b925ec0173688c17a.jpg)



Figure 4: Error analysis on manually annotated discourse-level phenomenon in test set of TDDAuto. SS: SingleSent, CR: Chain Reasoning, TI: Tense Indicator, FE: Future Events, HN: Hypothetical/Negated, EC: Event Coreference, CP: Causal/Prerequisite, WK: World Knowledge. We observe a stronger support for SS, CR, TI and and FE as compared to TDDMan. TIMERS trained exclusively on TDDAuto performs worse on challenging phenomenon like HN and CP. Consistent with results on TDDMan, TIMERS and its ablations trained on TDDAuto struggle on EC and WK.


<table><tr><td>Relation Label</td><td>Definition</td></tr><tr><td>Temporal</td><td>Relating to time</td></tr><tr><td>Summary</td><td>Shorter restatement</td></tr><tr><td>Same-unit</td><td>Part of the same phrasal unit</td></tr><tr><td>Span</td><td>Extending to multiple phrasal units</td></tr><tr><td>Purpose</td><td>Initiation in order to realize a goal</td></tr><tr><td>Example</td><td>Specific subtypes</td></tr><tr><td>Elaboration</td><td>Providing additional details</td></tr><tr><td>Reason</td><td>Justification with intent to defend a stance</td></tr><tr><td>Sequence</td><td>Subject-matter sequence</td></tr><tr><td>Condition</td><td>Realization of dependency</td></tr><tr><td>Means</td><td>Method or instrument to improve likelihood</td></tr><tr><td>Consequence</td><td>Intended or unintended end goal</td></tr><tr><td>Topic</td><td>Central idea</td></tr><tr><td>Attribution</td><td>Contributing factor</td></tr><tr><td>Textual Organization</td><td>Part of formal text span</td></tr><tr><td>Contrast</td><td>Opposing phenomenon</td></tr><tr><td>Manner</td><td>Semantic course of occurrence</td></tr><tr><td>Antithesis</td><td>Incompatibility due to contrast</td></tr><tr><td>Concession</td><td>Potential Incompatibility</td></tr><tr><td>Explanation</td><td>Providing clarification to an established fact</td></tr><tr><td>Circumstance</td><td>Framework for interpretation</td></tr></table>


Table 6: RST relations used in Rhetoric-aware graph  $G_{DG}$  in TIMERS, with definition as provided by Mann (1987)


<table><tr><td>Relation Label</td><td>Definition</td></tr><tr><td>After</td><td>TIMEX1 starts after TIMEX2 has ended</td></tr><tr><td>Before</td><td>TIMEX1 ends before TIMEX2 started</td></tr><tr><td>Equal</td><td>TIMEX1 is numerically equal to TIMEX2 upto date resolution.</td></tr><tr><td>None</td><td>One of the timex cannot be extracted or normalized</td></tr></table>

Table 7: Timex-Timex and DCT-Timex relations used in the Time-aware graph  $G_{TG}$ .