# Syntax-based dynamic latent graph for event relation extraction

# ARTICLE INFO

Keywords:

Information extraction

Event relation extraction

Syntactic dependence

Graph modeling

Latent structure

# ABSTRACT

This paper focuses on extracting temporal and parent-child relationships between news events in social news. Previous methods have proved that syntactic features are valid. However, most previous methods directly use the static outcomes parsed by syntactic parsing tools, but task-irrelevant or erroneous parses will inevitably degrade the performance of the model. In addition, many implicit higher-order connections that are directly related and critical to tasks are not explicitly exploited. In this paper, we propose a novel syntax-based dynamic latent graph model (SDLG) for this task. Specifically, we first apply a syntactic type-enhanced attention mechanism to assign different weights to different connections in the parsing results, which helps to filter out noisy connections and better fuse the information in the syntactic structures. Next, we introduce a dynamic event pair-aware induction graph to mine the task-related latent connections. It constructs a potential attention matrix to complement and correct the supervised syntactic features, using the semantics of the event pairs as a guide. Finally, the latent graph, together with the syntactic information, is fed into the graph convolutional network to obtain an improved representation of the event to complete relational reasoning. We have conducted extensive experiments on four public benchmarks, MATRES, TCR, HiEve and TB-Dense. The results show that our model outperforms the state-of-the-art model by  $0.4\%$ ,  $1.5\%$ ,  $3.0\%$  and  $1.3\%$  in F1 scores on the four datasets, respectively. Finally, we provide detailed analyses to show the effectiveness of each proposed component.

# 1. Introduction

Events are the main elements that constitute the semantics of the natural languages in the real world, especially news messages (Hogenboom, Frasincar, Kaymak, & de Jong, 2011). News messages usually consist of multiple events, and there are often rich relationships between these events. These relationships between events can help the public better understand the evolution trends and characteristics of events, and assist in the response to risk events. Thus, understanding the relationship between events has long been one of the key tasks in information extraction. Among the multiple relationships of events, event temporal relationships and

# Input Text

In another operation, a man in his 20s was arrested during a raid on a small guardhouse in Dublin's

northern inner city for concealing 248,000 cigarettes, 51 kilograms of tobacco and a small amount of herbal cannabis.

Temporal relationship: arrested After concealing

Subevent relationship: operation  $\xrightarrow{\text{SuperSub}}$  arrested

# Syntax and Latent Dependencies

![](images/e15bf7e9b92e84c5036b4eaf0a9a56d27965109f8a30ed7a5f3b354e9de54dc6.jpg)



Fig. 1. Examples of TRE and SRE. In the syntax and latent dependencies, the gray lines represent the connections parsed by the automatic parsing tool Standard CoreNLP; the red lines indicate the ground-truth connections not correctly parsed by the parsing tool; the blue lines represent the potential connections that the parsing tool cannot obtain. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)


subevent relationships are widely used in many fields, such as event graph construction (Min et al., 2021, 2021), event prediction (Du & Zhou, 2022; Mao et al., 2021), document summarization (Kim & Kim, 2019), information threading (Narvala, McDonald, & Ounis, 2023) etc. Especially in the construction of event evolutionary graphs, the event temporal relationship and subevent relationship are the key to mining the event evolution path. In addition, they are also important aids for hot event tracking. Correspondingly, the event temporal relation extraction (TRE) and subevent relation extraction (SRE) have been introduced, aiming to predict the temporal relationship and parent-child relationship between a given event pair in the input text. As shown in Fig. 1, the event arrested occurs later than the event concealing, i.e., "arrested  $\xrightarrow{After}$  concealing". The event operation includes the event arrested, that is, "operation  $\xrightarrow{SuperSub}$  arrested".

In the event relation extraction task, there are no salient relationship cue words between the events of many texts (e.g., before, contain, etc.). Therefore, how to fully exploit the contextual information of event pairs is the key to correctly identifying the relationships between events. To this end, many studies consider introducing syntactic knowledge to model the structure of text to improve the performance of the model. Intuitively, syntactic dependency directly depicts the word-word connections, which offers effective shortcut context features for the inference of event relationships. For example, for the sentence: "Mr. John ( $e_1$ : told) the reporters that he had ( $e_2$ : received) the invitation", we can get the dependencies "received  $\xrightarrow{aux}$  had" and "told  $\xrightarrow{ccmp}$  received" through syntactic analysis. These two dependencies can help the model to more easily and accurately predict that event told occurs after event received. In previous research, Zhang, Ning, and Huang (2022) find that many temporal relationship clues are hidden in the connections between events and their surrounding contexts. Therefore, they design a graph transformer incorporating a temporal attention-oriented mechanism. Phung, Nguyen, and Nguyen (2021) design a hierarchical graph convolutional network incorporating pruned dependency trees to capture important contextual words at the sentence level. These methods demonstrate the effectiveness of syntactic information for extracting relationships between events.

Unfortunately, most of the previous syntax-based approaches directly model the dependency parse trees from third-party parsers, such as Standard CoreNLP (Manning et al., 2014), where such static parse features may largely lead to two major problems. On the one hand, the static parse trees from off-the-shelf parsers may contain incorrect or task-irrelevant connections (Zhang, Qi, & Manning, 2018). These connections can lead to the incorporation of some noisy information into the event features, which can easily bias the prediction relationships. Taking the sentence in Fig. 1 as an example, the syntactic parsing results of the automatic parsing tool have incorrectly predicted dependencies between "concealing" and "cigarettes", and the connection between "arrested  $\xrightarrow{obl} city$ , concealing  $\xrightarrow{rcmod} city$ " is missing, which are important for the mining of contextual information between event pairs. On the other hand, many task-related implicit connections in the syntax-based approach are overlooked by existing work without considering exploitation explicitly. Taking the sentence in Fig. 1 as an example, to identify the temporal relationship between arrested and concealing, constructing an "arrested  $\xleftarrow{} for \xleftarrow{}$ concealing" connection can help improve the confidence of the model's prediction of the temporal relationship between them. However, such a connection cannot be obtained by syntactic parsing alone. Therefore, how to avoid the interference of error or irrelevant information in the static parsing results and how to explore the potential connections are issues that cannot be ignored in the research of syntax-based event relation extraction.

To alleviate these issues, we propose a novel syntax-based dynamic latent graph network (SDLG) for event temporal relation extraction and subevent relation extraction. Specifically, we first perform a hard pruning of the parsing results from the automatic

Parsing tool to initially filter the event-irrelevant content, i.e., only the event-related one-hop dependencies and connections on the shortest dependency path between event pairs are retained. Second, we use a type-enhanced attention mechanism to construct a syntactic graph attention matrix based on the hard-pruned parsing results. This enables the model to focus more on task-relevant connections and reduce the effect of irrelevant or weakly relevant connections for further pruning. Then, to complement and correct the supervised syntactic features, we introduce an event pair-aware induction graph. Guided by the semantics of event pairs, it dynamically constructs a latent attention matrix through an event pair-aware attention mechanism to mine potential connections that are beneficial to the task. Subsequently, the syntactic graph attention matrix and the latent graph attention matrix are used as the input of the graph convolutional network (GCN). Finally, after encoding by a multi-layer graph network, we input the enhanced event-pair representation, the initial event-pair representation and the global representation of the sentence together to the event relation predictor.

We conduct extensive experiments on four datasets from news corpora, namely MATRES, TCR, HiEve and TB-Dense, and the experimental results show that our method outperforms the previous state-of-the-art models by  $0.4\%$ ,  $1.5\%$ ,  $3.0\%$  and  $1.3\%$  on the four datasets, respectively. In addition, we conduct specific analyses on each component, and the analyses find that mining task-specific potential dependencies can effectively complement the missing information in the syntax to further facilitate performance improvement. Meanwhile, an effective syntax pruning strategy is the key to alleviate the impact of noise for syntax-based event relation extraction.

Our main contributions can be summarized as follows:

- We propose a syntax-based dynamic latent graph (SDLG), in which both the syntactic and semantic structural features are leveraged to help capture key contextual information, improving the event temporal relation extraction and subevent relation extraction.

- We design an event pair-aware induction graph, which is guided by the semantics of event pairs, and mines task-relevant potential connections in the context to complement the missing information in the syntactic structure.

- Experimental results on four benchmark datasets show that the SDLG model achieves state-of-the-art performance.

# 2. Related work

In this subsection, we mainly introduce the research related to event temporal relation extraction, subevent relation extraction, and syntax feature modeling.

# 2.1. Event relation extraction

Event relation extraction (Liu, Chen, Liu, Zuo, & Zhao, 2020) is one of the sub-tasks of information extraction (Cao et al., 2022; Fei, Ren, & Ji, 2020; Fei, Ren, Zhang, Ji, & Liang, 2021; Li et al., 2022). It is a subsequent task to event extraction (Xiang & Wang, 2019). Event extraction aims at extracting events from the text, including the event trigger words and the argument roles of events. While event relation extraction task aims to identify the relationships between the events that have been extracted from the text, i.e., event temporal relationships, subevent relationships, etc.

Event temporal relation extraction In the early days, statistical methods were often used to extract the temporal relationship between events (Mani, Verhagen, Wellner, Lee, & Pustejovsky, 2006; Verhagen & Pustejovsky, 2008). For example, Yoshikawa, Riedel, Asahara, and Matsumoto (2009) propose to construct a Markov logic model using logical constraints between different types of temporal relations. The computational complexity of this kind of method is high and the limited effectiveness of TRE makes further development difficult.

In recent years, the introduction of large-scale pre-trained language models has led to new advances in the task of TRE. Some studies use multi-task mechanism to facilitate the learning ability of the model (Cheng, Asahara, Kobayashi, & Kurohashi, 2020; Han, Ning, & Peng, 2019; Wen & Ji, 2021). For example, Wang, Li, and Xu (2022) construct graph structures using other events, timestamps, and the creation time of the document to take advantage of the complementarity between different types of temporal relationships. Liu, Xu, Chen, and Zhang (2021) and Vo, Al-Obeidat, and Bagheri (2020) construct different event graphs to predict the temporal relationships between events, respectively. Han, Ren, and Peng (2021) propose continuous training on the large-scale pretrained language models to enhance their temporal reasoning capabilities. In order to deepen the model's understanding of events, some methods introduce external knowledge (Han, Zhou, & Peng, 2020; Ning, Subramanian, & Roth, 2019; Tan, Pergola, & He, 2021). Tan, Pergola, and He (2023) propose a novel Bayesian translation model that incorporates commonsense knowledge, which models temporal relational representations as latent variables, and infers values via Bayesian inference. In addition, some methods design specific constraints to assist in relational inference (Han, Hsu, et al., 2019; Zhou et al., 2021).

Subevent relation extraction Traditional subevent relation extraction methods mainly use machine learning algorithms (Araki, Liu, Hovy, & Mitamura, 2014; Glavaš, Šnajder, Kordjamshidi, & Moens, 2014). On the basis of the rapid development of deep learning, many new research methods have been proposed. Man, Ngo, Van, and Nguyen (2022) argue that some contextual information in documents also facilitates the recognition of subevent relation between event pair in input sentences. Therefore, they propose a long-text-based reinforcement learning algorithm aimed at extracting sentences from documents that are beneficial to the task. Zhou, Ning, Khashabi, and Roth (2020) obtain an enhanced event representation for SRE task by constructing a temporal

Common sense language model. In addition, some studies use constraint rules to guide the model to achieve global consistency inference (Hwang et al., 2022; Wang, Chen, Zhang, & Roth, 2020; Wang, Zhang, Chen, & Roth, 2021).

# 2.2. Syntax feature modeling

Syntactic information is widely used in many tasks in natural language processing, and significant results have been achieved on several tasks, such as sentiment analysis (Ma & Pang, 2022; Shi, Li, Li, Fei, & Ji, 2022; Wu et al., 2022; Wu, Fei, Ren, Ji, & Li, 2021), named entity recognition (Suttono & Hahn-Powell, 2022; Wu, Fei, Ren, Li, et al., 2021), semantic parsing (Fei, Li, Li, & Ji, 2021; Fei, Wu, Ren, Li, & Ji, 2021), paraphrase generation (Fei, Wu, Ren, & Zhang, 2022; Yang et al., 2022), code summarization (Guo, Liu, Wan, Li, & Zhou, 2022), etc.

Since syntax contains rich linguistic knowledge, many previous approaches have also considered introducing syntactic information to facilitate the identification of temporal and subevent relationships between events. The traditional methods (Glavaš & Snajder, 2014) artificially combine multiple features (i.e., events, bag of words, position, syntax and knowledge, etc.) to complete event relation extraction. Among recent approaches, Meng, Rumshisky, and Romanov (2017) and Cheng and Miyao (2017) capture syntactic information on the shortest dependency path between events using the sequence model LSTM. Aldawsari and Finlayson (2019) propose a supervised model to effectively model context by incorporating syntactic, discourse and narrative features. In recent years, some approaches adopt graph networks to model syntactic knowledge due to their good structural encoding ability. Zhang et al. (2022) design a new syntax-guided attention mechanism and applied it to graph transformer. Phung et al. (2021) utilize a hierarchical graph convolutional network to encode a pruned dependency tree to facilitate information interaction between tokens. In addition, Mathur et al. (2021) capture document-level event relations by constructing syntactic, rhetorical, and temporal graphs.

Previous studies have confirmed the effectiveness of using syntactic information for TRE and SRE. However, many previous methods directly use the static results of syntactic parsing to model the context of the input text, where the noise information may make the performance of the model suboptimal. Moreover, many task-related connections in the original text cannot be modeled explicitly (Fei, Wu, Zhang, Ren, & Ji, 2022; Fei, Zhang, & Ji, 2020). Therefore, we propose to use a type-enhanced attention mechanism to filter syntactic connections, and use an event pair-aware attention to explore potential connections to enrich event-pair representations.

# 3. Research objects

Extracting temporal and subevent relationships between news events in news texts has become crucial for businesses and governments, and can help them to cover popular hotspots by constructing the evolution of events. However, in many news texts, there are usually no clear relational clue words that allow people to quickly discover the relationship between events. Therefore, many previous approaches have tried to introduce syntax to better mine contextual information. However, on the one hand, previous approaches often directly use the results parsed by syntactic parsing tools, where noisy information can interfere with the model's judgement. On the other hand, higher-order implicit connections that are directly related to event relations are ignored. The goal of this paper is to improve the extraction of relations from news events based on syntactic features and to provide a good basis for the subsequent task (i.e. the construction of an event evolutionary graph).

The specific issues we research mainly include:

1. How to effectively filter out the information irrelevant to the event relation from the static results parsed by the syntax analysis tool (i.e. Sections 4.2 and 5.5).

2. How to mine high-order implicit connections related to news event relations in texts and the influence of different implicit graph construction methods (i.e. Sections 4.3 and 5.6).

3. Whether a latent graph guided by event pair semantics can complement supervised syntactic features, and how it improves the performance of the model (i.e. Sections 5.4 and 5.9).

# 4. Methodology

Problem formulation The input of our model is a sentence that contains two annotated events. If two events exist in two different sentences, the two sentences are concatenated and fed into the model. Given an input text  $S = \{w_{1}, w_{2}, \ldots, w_{n}\}$ , where  $w_{i}$  denotes the  $i$ -th token in the text and the length of the text is  $n$ . The sentence contains a pair of events  $< e_{s}, e_{t} >$ , corresponding to the  $s$ -th and  $t$ -th tokens in the text  $(s, t \in (1, n))$ . The purpose of our task is to identify the temporal relationship or subevent relationship between these two events. Taking the sentence in Fig. 1 as an example, for the event temporal relation extraction task, the input is the sentence "In another operation, a man in his 20 s was arrested during a raid on a small guardhouse in Dublin's northern inner city for concealing 248,000 cigarettes, 51 kilograms of tobacco and a small amount of herbal cannabis." (events "arrested" and "concealing" have been annotated in advance), and the output is the temporal relationship between "arrested" and "concealing", i.e. "After"; For the subevent relation extraction task, the input of the model is also the sentence, where the annotated events are "operation" and "arrested", and the output is the parent-child relationship between them, namely "SuperSub".

Model architecture This section will mainly introduce the specific architecture of syntax-based dynamic latent graph model (i.e., SDLG). The detailed structure of SDLG is shown in Fig. 2. SDLG contains five main parts, which are the Input and Encoding Layer, Type-enhanced Syntactic Pruning Graph, Event pair-aware Induction Graph, GCN Encoding Layer and Event Relation Predictor. Next, we will describe each part in detail separately.

![](images/a4e77f9b394c9023239eb62ca540a621f9ce9974d8074f58be72de1b5ef283f0.jpg)



Fig. 2. The overall architecture of the SDLG. It consists of five main sections: (1) Input and Encoding Layer: Initial encoding of the input text; (2) Type-enhanced Syntactic Pruning Graph: Further pruning of the syntactic dependencies after hard pruning; (3) Event pair-aware Induction Graph: Mining of task-aware latent connections; (4) GCN Encoding Layer: Encoding of the graph matrix obtained from (2) and (3); (5) Event Relation Predictor: Relation classification of event pairs.


# 4.1. Input and encoding layer

For a given text  $S$ , we first obtain the initial word vector representation for each token. The method based on the large-scale pre-training model Roberta (Liu et al., 2019), which has demonstrated excellent performance on multiple tasks, and has also achieved the best results on the TRE and SRE task. Therefore, in this paper, we choose Roberta as the initial encoder of the text to extract the hidden contextual representation of each token. After the initial encoding, we can obtain the vector representation of each token, that is,  $\mathbf{H} = PLM\{w_1,w_2,\dots ,w_n\} = \{h_{cls},h_1,h_2,\dots ,h_s,\dots ,h_t,\dots ,h_n\}$ , where  $h_i\in \mathbb{R}^d$ ,  $d$  denotes the dimension of the embedded representation, and  $i\in (1,n)$ . PLM represents the initial pre-trained language model. The initial representations of event  $e_s$  and event  $e_t$  are  $h_s$  and  $h_t$ , respectively.  $h_{cls}$  represents the global semantic representation of the input text.

# 4.2. Type-enhanced syntactic pruning graph

Syntactic information contains rich linguistic knowledge and has been widely exploited in tasks in multiple domains (Jin, Li, Lian, Jiao, & Hu, 2022; Liu, Xu, & Liu, 2021). Previous studies have also shown that syntactic information is of interest for event temporal relation extraction and subevent relation extraction tasks (Aldawsari & Finlayson, 2019; Zhang et al., 2022). However, the automatically obtained syntactic information may contain noisy information, which is detrimental to our task. Therefore, we construct a type-enhanced syntactic pruning graph, guided by the dependency types in syntactic connections, and assign different weights to different connections to filter the noisy information. Specifically, the process of constructing a type-enhanced syntactic pruning graph consists of three main steps, which are syntactic dependency acquisition, adjacency matrix construction, and syntactic graph attention matrix calculation.

Syntactic Dependency Acquisition: First, we define a set of dependency types, that is,  $\mathbf{R} = \{r_1,r_2,\dots ,r_u\}$ , where  $u$  is the total number of categories of dependency types. We set different initial vectors for each type, namely  $H_{R} = \{h_{r_{1}},h_{r_{2}},\ldots ,h_{r_{u}}\}$ . Then, we use the automatic parsing tool to parse the input text  $S$  to obtain the set of syntactic dependency triples  $Dep = \left\{(w_{i_1},r_{k_1},w_{j_1}),(w_{i_2},r_{k_2},w_{j_2}),\ldots ,(w_{i_N},r_{k_N},w_{j_N})\right\}$ . Where  $i_m$  and  $j_{m}$  are the indexes of the corresponding token,  $k_{m}$  is the category index of the corresponding dependency type ( $m\in (1,N)$ );  $i_m,j_m\in (1,n)$ ;  $k_{m}\in (1,u)$ ; The total number of triples is  $N$ .

Existing studies have shown that only a small fraction of syntactic connections are task-aware (Fei, Wu, Li, et al., 2022; Wu, Fei, Ren, Ji, & Li, 2021; Zhang et al., 2018). Therefore, we believe that designing hard pruning strategies for syntactic dependencies can help reduce the impact of irrelevant connections. Specifically, we perform a simple hard pruning operation before exploiting the syntactic information. We keep only the event-related one-hop dependencies and the connections on the shortest dependency path between two events, denoted as the set Ohp and the set Sdp, respectively. Finally, we merge the two sets to obtain the final set of syntactic dependency triples, as shown in Eq. (1).

$$
\begin{array}{l} \mathbf {O h p} = \left\{\left(w _ {i _ {m}}, r _ {k _ {m}}, w _ {j _ {m}}\right) \mid w _ {i _ {m}} = = e _ {s} \text {o r} w _ {j _ {m}} = = e _ {s} \text {o r} w _ {i _ {m}} = = e _ {t} \text {o r} w _ {j _ {m}} = = e _ {t} \right\} \\ \mathbf {S d p} = \left\{\left(w _ {i _ {m}}, r _ {k _ {m}}, w _ {j _ {m}}\right) \mid \left(w _ {i _ {m}}, r _ {k _ {m}}, w _ {j _ {m}}\right) \in S D P \right\} \tag {1} \\ \hat {\mathbf {D e p}} = \mathbf {O h p} \cup \mathbf {S d p} \\ \end{array}
$$

Where  $m\in (1,N)$  .  $i,j\in (1,n)$  . SDP denotes the shortest dependency path between event  $e_s$  and event  $e_t$

Adjacency Matrix Construction: We transform the obtained set of syntactic triples  $\hat{\mathbf{Dep}}$  into a matrix that can be encoded by the graph network. First, we convert each tuple's unidirectional connection to a bidirectional connection. Second, we add a self-loop relation for each word. Furthermore, if the input is two sentences, a "cross-sen" connection is added to capture the cross-sentence relationship. Formally, the adjacency matrix  $\mathbf{A}$  can be obtained from Eq. (2):

$$
\mathbf {A} _ {i, j} = \left\{\begin{array}{l l}1&\text {i f} w _ {i} \rightarrow w _ {j} \text {o r} w _ {i} \leftarrow w _ {j} \text {o r} i = j\\0&\text {o t h e r w i s e .}\end{array}\right. \tag {2}
$$

Syntactic Graph Attention Matrix Calculation: Although we adopt hard pruning to preserve the syntactic dependencies related to events, there may still be some incorrect or weakly relevant syntactic information. Therefore, to filter noisy information from the automatic parsing results, we introduce a type-enhanced attention mechanism for further pruning. It effectively captures significant dependencies that are beneficial to the TRE and SRE tasks. Specifically, we use the dependency types in the dependency results as a guide, thus optimizing the weight scores on different connected arcs. If we obtain the representation  $h_{i}^{(l-1)}$  of the  $i$ -th token at the  $(l-1)$ -layer after  $(l-1)$  layers graph encoding, the weight  $\eta_{i,j}^{(l)}$  between the  $i$ -th and  $j$ -th token on the  $l$ -th layer GCN is calculated as follows:

$$
\eta_ {i, j} ^ {(l)} = \frac {A _ {i , j} \cdot \exp \left(\mathbf {W} _ {0} ^ {(l)} \mathbf {g} _ {i} ^ {(l)} \cdot \mathbf {W} _ {1} ^ {(l)} \mathbf {g} _ {j} ^ {(l)}\right)}{\sum_ {c = 1} ^ {n} A _ {i , c} \cdot \exp \left(\mathbf {W} _ {0} ^ {(l)} \mathbf {g} _ {i} ^ {(l)} \cdot \mathbf {W} _ {1} ^ {(l)} \mathbf {g} _ {c} ^ {(l)}\right)} \tag {3}
$$

$$
\mathbf {g} _ {i} ^ {(l)} = \mathbf {h} _ {i} ^ {(l - 1)} \oplus \mathbf {h} _ {r _ {i, j}} \tag {4}
$$

Where  $\mathbf{h}_{r_{i,j}}$  denotes the vector representation of the dependency type between the  $i$ -th and  $j$ -th token;  $\mathbf{W}_0^{(l)}$  and  $\mathbf{W}_1^{(l)}$  are the weight matrices. Finally, we can obtain the syntactic graph attention matrix  $\mathbf{B}_{sy}^{(l)} = \{\eta_{i,j}^{(l)}\}_{i,j\in (1,n)}$  for the  $l$ -th layer.

# 4.3. Event pair-aware induction graph

Using syntax alone may fail to construct the potential connections associated with the event temporal relation extraction and subevent relation extraction task. Instead, these connections help the model to capture hard-to-identify relational cues faster. Our goal is to construct a dynamic latent graph that can effectively mine task-relevant connections and thus complement the missing information in the syntactic graph. Therefore, we construct a dynamic event pair-aware induction graph. Guided by the semantics of event pairs, it searches for important context words in the original context and establishes associations with event pairs through the event pair-aware attention mechanism.

Specifically, in order to mine task-aware implicit connections, we need to frame from the perspective of event pair semantics. Firstly, the representation of event pairs, namely  $h_{ep}^{(l)}$ , is obtained through a layer of feed-forward neural network. Then, we model specific semantic associations based on the event pairs, i.e., we use the event pair-aware attention mechanism to treat the event pairs as attentional computational queries for relevant features in the learning context. Here, we first replicate the obtained event pair representation  $n$  times, where  $n$  is the length of the sentence. An event pair representation  $H_{ep}^{(l)}$  can be obtained. Then, the activation function is used to map the cross multiplication result of the event pair representation  $(H_{ep}^{(l)})$  and the token representation  $(H)$  in the context into an  $n \times n$  matrix. Each element in the matrix represents the importance of the token to the event pair. Suppose we have obtained the event  $e_s$  and event  $e_t$  representations for the  $(l - 1)$ -th layer graph convolutional network denoted by  $\mathbf{h}_s^{(l - 1)}, \mathbf{h}_t^{(l - 1)}$ , respectively, the latent graph attention matrix  $B_{ep}^{(l)}$  for  $l$ th layer is computed as shown in Eq. (5).

$$
\mathbf {B} _ {e p} ^ {(l)} = \tanh \left(\mathbf {H} _ {e p} ^ {(l)} \mathbf {W} _ {2} ^ {(l)} \times (\mathbf {H} \mathbf {W} _ {3} ^ {(l)}) ^ {T} + \mathbf {b} _ {1} ^ {(l)}\right)
$$

$$
\mathbf {H} _ {e p} ^ {(l)} = \operatorname {C o p y} ^ {(\mathrm {n})} \left(\mathbf {h} _ {e p} ^ {(l)}\right) \tag {5}
$$

$$
\mathbf {h} _ {e p} ^ {(l)} = \operatorname {F F N} _ {1} \left(\mathbf {h} _ {s} ^ {(l - 1)} \oplus \mathbf {h} _ {t} ^ {(l - 1)}\right)
$$

Where Copy represents the copy function;  $\mathrm{FFN}_1(\pmb {x}) = \pmb {W}_4\pmb {x} + \pmb {b}_2$  .  $\mathbf{W}_2^{(l)}$ $\mathbf{W}_3^{(l)}$  , and  $\mathbf{W}_4^{(l)}\in \mathbb{R}^{d\times d}$  represent the weight matrix, and  $\mathbf{b}_1^l$  and  $\mathbf{b}_{2}$  represents the bias function. We replicate  $\mathbf{h}_{ep}$ $n$  times to obtain  $\mathbf{H}_{ep}\in \mathbb{R}^{n\times d}$  as a representation of event pairs.

# 4.4. GCN encoding layer

To combine the advantages of syntactic and latent graphs, we integrate the obtained syntactic graph attention matrix and the latent graph attention matrix:

$$
\mathbf {B} ^ {(l)} = \operatorname {s o f t m a x} \left(\mathbf {B} _ {s y} ^ {(l)} + \mathbf {B} _ {e p} ^ {(l)}\right) \tag {6}
$$

The context between events in news texts is usually complex and extensive. In this paper, we hope to make better use of syntactic information to help identify the temporal relationship and parent-child relationship between events in news texts, which are

obtained through external tools. Both the obtained syntactic tree and our latent graph attention matrix can be regarded as a graph structure, and this kind of structural information is indispensable for making full use of syntactic relations. In addition, graph convolutional networks (GCN) can encode structural information well and are often used to process graph-structured data (Kipf & Welling, 2017). They can effectively integrate local information to key nodes. Many previous studies have demonstrated its effectiveness in encoding contextual information (He, Meng, Zhang, Duan, & Wang, 2023; Zhao, Yang, Zhang, & Wang, 2022). Therefore, we encode the obtained attention matrix using a graph convolutional network. We use the integrated matrix  $\mathbf{B}^{(l - 1)}$  as the input of the  $(l)$ -th layer graph convolutional network. The output of the  $(l)$ -th layer of the GCN is the representation of each token, i.e.,  $h_i^{(l)}, i \in (1,n)$ . In each layer of the GCN, each node in the graph is updated according to the vector representation of its neighbor nodes. Formally, in the  $(l)$ -th layer of the GCN, the embedding vector of the  $(i)$ -th token is computed by Eq. (7).

$$
\mathbf {h} _ {i} ^ {(l)} = \operatorname {R e l u} \left(\sum_ {c = 1} ^ {n} B _ {i, c} ^ {(l)} \left(\mathbf {W} _ {5} ^ {(l)} \cdot \mathbf {h} _ {c} ^ {(l - 1)} + \mathbf {b} _ {3} ^ {(l)}\right)\right) \tag {7}
$$

where  $\mathbf{W}_5^{(l)}$  is the weight matrix and  $B_{i,c}^{(l)}$  denotes the value of the element in the  $i$ -th row and  $c$ -th column of matrix  $\mathbf{B}^{(l)}$ . After encoding by the  $L$ -layer GCN, we can derive the final feature representation of each token, which is recorded as  $\hat{H} = \{h_1^{(L)}, h_2^{(L)}, \dots, h_s^{(L)}, \dots, h_t^{(L)}, \dots, h_n^{(L)}\}$ . The final representations corresponding to event  $e_s$  and event  $e_t$  are  $h_s^{(L)}$  and  $h_t^{(L)}$  respectively.

# 4.5. Event relation predictor

Event Relation Predictor is used to determine the temporal relationship or subevent relationship between event pair. Our event relation predictor mainly consists of a two-layer feed-forward neural network. Its input is the event pair representation enhanced by graph encoding, the initial event pair representation and the global semantic representation of the sentence. The output is the predicted relationship between the event pair. The formula of the Event Relation Predictor is as follows:

$$
\hat {y} _ {<   e _ {s}, e _ {t} >} = \mathrm {F F N} _ {3} \left(\tanh  \left(\mathrm {F F N} _ {2} \left(\left[ \boldsymbol {h} _ {c l s}; \boldsymbol {h} _ {i} ^ {(L)}; \boldsymbol {h} _ {j} ^ {(L)}; \boldsymbol {h} _ {i}; \boldsymbol {h} _ {j} \right]\right)\right)\right) \tag {8}
$$

Where  $\hat{y}_{<e_s,e_t>}$  represents the event relationship predicted by the model;  $h_{cls}$  represents the global semantic representation of the sentence obtained by the initial encoder;  $h_i$  and  $h_j$  represent the initial event representations of  $e_s$  and  $e_t$ , respectively;  $h_i^{(L)}$  and  $h_j^{(L)}$  denote the enhanced representations of  $e_s$  and  $e_t$  encoded by the graph network, respectively.

# 4.6. Framework learning

We take the minimization of the cross-entropy function as the optimization objective, and obtain the optimal model through multiple training iterations. It is given by Eq. (9).

$$
\mathcal {L} (\boldsymbol {\Phi}) = - \mathbf {y} _ {<   e _ {s}, e _ {t} >} \log \left(\hat {\mathbf {y}} _ {<   e _ {s}, e _ {t} >}\right) + \left(1 - \mathbf {y} _ {<   e _ {s}, e _ {t} >}\right) \log \left(1 - \hat {\mathbf {y}} _ {<   e _ {s}, e _ {t} >}\right) \tag {9}
$$

where  $y_{<e_s, e_t>}$  denotes the ground-truth label of the relationship between the  $e_s$  and  $e_t$ ;  $\hat{y}_{<e_s, e_t>}$  denotes the label of the relationship predicted by our model.  $\Phi$  denotes the set of parameters of the SDLG model. After several epochs of training, we save the best-performing model parameters.

# 5. Experiments

We conduct in-depth experiments on four benchmark datasets to evaluate the validity of our proposed SDLG model. This section will describe our experiments in detail.

# 6. Theoretical and practical implications

# 6.1. Theoretical implications

Syntactic Pruning: Although third-party syntactic parsing tools have made good progress in parsing results. However, there is still error information in the parsing results, and not all connections are useful for a particular task. Therefore, an effective pruning strategy is essential for the task of event relation extraction. Previous approaches have either pruned by designing specific rules (Xu et al., 2015) or relied on a specific model (Sun, Zhang, Mao, Mensah, & Liu, 2020) structure to filter information. We want the model to learn dynamic pruning based on specific events and syntactic types. Therefore, we first apply a hard pruning strategy that retains syntactic connections that are strongly correlated with events. Then, a type-enhanced attention mechanism is designed to dynamically isolate irrelevant or weakly related dependency information.

Implicit Relation Mining: Using syntactic features alone would lose higher-order implicit connections. Especially in the relation extraction of news events, the context between events is complicated, even two events are in different sentences, and the syntax cannot capture the potential connection related to the task, which are also ignored in the previous event relation extraction research. Therefore, we use the semantics of the event pair as a guide to mine the connections related to the event pair in the context and construct an event pair-aware latent graph, which complements the missing information in the syntactic features and better facilitates the relation extraction of news events.

Model Interpretability: Deep learning is a black box that is difficult to interpret. In this paper, we compute the importance of different syntactic connections and implicit connections via a syntactic-based and a event pair-based attention mechanism, respectively. Such a method can clearly show which connections are crucial for judging the relationship between events. The weights when calculating attention scores are continuously optimized in multiple iterations. Then the fusion and extraction of features are performed by GCN. As shown in Fig. 5, the orange and blue connections are more critical for the event relation extraction task.

# 6.2. Practical implications

This paper builds a deep learning model that can extract temporal relationships and subevent relationships between events in news texts. The model proposed in this paper has some practical applications. It is an important part of constructing event evolutionary graph, which can perceive risks in advance for businesses and governments. Specifically, our model can extract the temporal relationships and parent-child relationships between events from news reports, and then the obtained news event relationships can form an event evolutionary chain. Based on the constructed event evolutionary graph, relevant businesses or governments can use it as a knowledge base to automatically link new events (Haneczok & Piskorski, 2020) and to perceive the possible evolution direction of events in advance, especially emergencies, so as to formulate scientific and reasonable countermeasures to maintain social harmony and stability.

<table><tr><td>Input</td><td>Vanilla Classifier</td><td>SDLG</td></tr><tr><td>The FAA on Friday announced it will close 149 regional airport control towers because of forced spending cuts – sparing 40 others that the FAA had been expected to shutter. A four-week, phased closure of the 149 control towers will begin on April 7, the FAA said.</td><td>Before (×)</td><td>After (√)</td></tr><tr><td>Syntactic and Implicit Connection of SDLG</td><td colspan="2">...announced it will close 149 regional airport control towers because of forced spending cuts – sparing ...closure of ...begin ..., the FAA said.</td></tr></table>


(a) A case for event temporal relation extraction.


<table><tr><td>Input</td><td>Vanilla Classifier</td><td>SDLG</td></tr><tr><td>The photographs , which date back to 2010 but were published by the Los Angeles Times on Wednesday , add to a string of damaging incidents that have ignited anti-Western feeling and complicated NATO efforts to withdraw most troops in 2014 . The first incident took place in February 2010 , when paratroopers from the 82nd Airborne Division &#x27;s 4th Brigade Combat Team were sent to an Afghan police station in Zabul province to inspect the remains of an alleged suicide bomber .</td><td>Coref (×)</td><td>SuperSub (√)</td></tr><tr><td>Syntactic and Implicit Connection of SDLG</td><td colspan="2">... a string of damaging incidents that have ignited ... The first incident took place ... suicide bomber .</td></tr></table>

(b) A case for subevent relation extraction.

Fig. 5. The case study of our SDLG model compared to the Vanilla Classifier. Where the blue lines indicate the implicit connections constructed by our event pair-aware induction graph; the orange lines indicate the significant dependencies filtered by the type-enhanced syntactic pruning graph in the syntactic parsing results; the gray lines indicate other dependencies automatically parsed by the syntactic parsing tool and considered as irrelevant or weakly relevant connections in our approach. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)


Table 8 Effect of graph convolutional network on graph structure information generated by syntactic tree on MATRES dataset.


<table><tr><td>Model</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>SDLG (Our model)</td><td>82.0</td><td>84.2</td><td>83.1</td></tr><tr><td>SDLG + direction</td><td>82.3</td><td>83.1</td><td>82.7</td></tr><tr><td>SDLG - edge type</td><td>82.1</td><td>81.7</td><td>81.9</td></tr></table>

# 7. Conclusion

In this paper, we propose a novel syntax-based dynamic latent graph model called SDLG for event temporal relation extraction and subevent relation extraction tasks. It is guided by syntactic dependency types and uses a type-enhanced attention mechanism to assign different attention to different dependencies in the automatically parsed syntactic results, thus effectively filtering out the noisy information in them. More importantly, in order to mine the task-aware implicit connections in the original context, we design a dynamic event pair-aware induction graph. It is oriented to the semantics of event pairs, and complements and corrects the supervised syntactic features through the event pair-aware attention mechanism. An enhanced event pair representation for relation prediction can be obtained after encoding by a multilayer graph convolutional network. Extensive experimental results on four datasets demonstrate that our proposed SDLG model outperforms previous state-of-the-art models. We plan to explore the effectiveness of SDLG model on other types of event relation extraction tasks in the future. In addition, we would like to continue exploring other more efficient methods for mining potential connections in the event relation extraction.

The authors declare that they have no known competing interests or personal relationships that could have appeared to influence the work reported in this paper.
