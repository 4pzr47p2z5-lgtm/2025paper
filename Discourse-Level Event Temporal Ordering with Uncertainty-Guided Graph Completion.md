# Discourse-Level Event Temporal Ordering with Uncertainty-Guided Graph Completion

# Abstract

Learning to order events at discourse-level is a crucial text understanding task. Despite many efforts for this task, the current state-of-the-art methods rely heavily on manually designed features, which are costly to produce and are often specific to tasks/domains/datasets. In this paper, we propose a new graph perspective on the task, which does not require complex feature engineering but can assimilate global features and learn inter-dependencies effectively. Specifically, in our approach, each document is considered as a temporal graph, in which the nodes and edges represent events and event-event relations respectively. In this sense, the temporal ordering task corresponds to constructing edges for an empty graph. To train our model, we design a graph mask pre-training mechanism, which can learn inter-dependencies of temporal relations by learning to recover a masked edge following graph topology. In the testing stage, we design an certain-first strategy based on model uncertainty, which can decide the prediction orders and reduce the risk of error propagation. The experimental results demonstrate that our approach outperforms previous methods consistently and can meanwhile maintain good global consistency.

# 1 Introduction

Learning to order events at discourse-level is a crucial text understanding task, which is necessary for many applications including event timeline construction [Do et al., 2012; Reimers et al., 2016], time-aware summarization [Yan et al., 2011], temporal commonsense reasoning [Zhou et al., 2019], and others. Consider the example in Figure 1. Given a document marked with events, a system should assign a temporal link, i.e., TLINK [Allen, 1984], between every pair of events. For example, a TLINK of BEFORE should be assigned between [E2 assistance] and [E4 fallen], denoted by [E2]  $\xrightarrow{\text{BEFORE}}$  [E4], indicating that [E2] occurs before [E4].

The challenge of discourse-level event temporal ordering derives from the appearance of long contexts. To succeed in the task, a system should be able to reason over global features and meanwhile maintain document-wide consistency.

It's [E1 turning] out to be another very bad financial week. The financial [E2 assistance] from the World Bank is not [E3 helping].  
... 3 sentences are omitted ...  
The value of the Indonesian stock market has [E4 fallen] by twelve percent. The Indonesian currency has [E5 lost] twenty six percent of its value. In Singapore, stocks [E6 hit] a five year low.

![](images/8f8e6b6a39c927e2dcb66d2dc153bdf7485353d5f09187d1672587187318d171.jpg)



Temporal Graph with TLINKs


![](images/8ec9b8ddcbca208ae518cfcad9bab071f376798da94933d0bfb4bcba36f192bf.jpg)


![](images/2d91ba5e94b88b825cf24e6f655804fb90ba676b7e9efd8d5dc7fa796888bb90.jpg)



Figure 1: Illustration of framing discourse-level event temporal ordering as a graph completion problem, where the nodes and edges indicate events and their temporal relations respectively.


For example, if a model has predicted [E2]  $\xrightarrow{\mathrm{BEFORE}}$  [E4] and [E4]  $\xleftarrow{\mathrm{SIMULTANEOUS}}$  [E5], it should also figure out that [E2]  $\xrightarrow{\mathrm{BEFORE}}$  [E5]. It is generally difficult to learn such patterns from texts solely, and the state-of-the-art methods usually resort to human-designed rules [Do et al., 2012; Ning et al., 2017; Ning et al., 2018; Han et al., 2019a; Han et al., 2019b]. Despite many progresses, these rules are usually costly to produce and often specific to tasks/domains/datasets, which limit the applicability of previous methods.

In this paper, we provide a new perspective for the event temporal ordering task — framing it as a graph completion problem. As noted in Figure 1 (Below), in our approach each document is considered as a temporal graph, in which each node has a one-to-one mapping relationship to an event. In this graph view, the event temporal ordering task corresponds to predicting/constructing the edges of the graph — which designate event-event temporal relations. By adopting this graph view, we can directly leverage the recent advances in graph representation learning [Chen et al., 2020] to enhance training, which endows our model the ability of global reasoning and meanwhile mining the underlying interdependencies of temporal relations effectively.

We devise an uncertainty-guided graph completion (UC-Graph) framework. For training, our framework employs a graph mask pre-training strategy, in which we randomly

mask a portion of edges out and then learn to reconstruct them following the remaining graph topology. This graph view enables our model to assimilate document-level features for reasoning and meanwhile learn the sense of global logical consistency. Nevertheless, a gap may exist in the inference stage as we would begin with an empty graph (instead of a masked graph), and the early predictions can have great impact on the late predictions. To mitigate this problem, we design a "certain-first" strategy in UCGraph based on model uncertainty [Gal and Ghahramani, 2016], which can find the optimal edge prediction orders and therefore minimize error propagation in the graph completion process ( $\S$  6.2).

To verify the effectiveness of our approach, we have conducted extensive experiments on the standard benchmark datasets [Naik et al., 2019]. The experimental results demonstrate that our approach consistently outperforms previous methods and sets up a new state-of-the-art. In addition to its superior performance, our approach also show advantages over previous methods in maintaining global consistency, which is a crucial aspect of event temporal ordering. We have released our code at https://github.com/jianliu-ml/EventTemp to facilitate further exploration.

To summarize, our contributions are three-fold:

- This paper provides a new graph view on the task of discourse-level event temporal ordering, which can effectively assimilate document-level evidence for reasoning and learn inter-dependencies of temporal relations without complex feature engineering.

- We proposes a model consisting of graph mask pre-training and uncertainty-guided graph completion, which can learn inter-dependencies of temporal relations and find optimal prediction orders respectively. To our best knowledge, this is the first work introducing graph representation learning and uncertainty modeling to the task of discourse-level event temporal ordering.

- We set up a new state-of-the-art on the standard benchmark datasets, and we will release our code to facilitate following studies.

# 2 Related Work

Event Temporal Ordering. Temporal ordering of events is a crucial text understanding task. Shaped by the proposed TimeML annotation [Pustejovsky et al., 2003a] and the related corpora such as TimeBank [Pustejovsky et al., 2003b] and TimeBank-Dense [Cassidy et al., 2014], most previous works focus on addressing local temporal relations, i.e., they focus on events in the same or adjacent sentences [Bethard et al., 2007; Verhagen et al., 2007; UzZaman and Allen, 2010; Chang and Manning, 2012; Chambers, 2013; Chambers et al., 2014; Reimers et al., 2016], Despite many progresses, the reliance on local features often restricts their ability to address global event relations. As a remedy, a few of works have explored introducing document-structure constraints [Ning et al., 2017; Han et al., 2019a], entity co-reference patterns [Do et al., 2012], and event causal clues [Do et al., 2012; Ning et al., 2018] to learn global dependencies. Nevertheless, designing such rules requires substantial domain exper

tise, which may limit the applicability of previous works. Recently, focusing on global event temporal relations specifically, [Naik et al., 2019] benchmark discourse-level event temporal ordering by proposing a new corpora TDDiscourse. As suggested by [Naik et al., 2019], global event temporal ordering is very challenging, and the previous state-of-the-art systems underperforms a simple majority-class baseline.

Graph Representation Learning. Recent years have witnessed a surge in exploring graph representation learning [Chen et al., 2020]. Graph neural networks, including Graph Convolution Network [Kipf and Welling, 2017] and its variants [Veličković et al., 2018; Schlichtkrull et al., 2018], have demonstrated state-of-the-art performance in addressing many tasks, including semantic role labeling [Marcheggiani and Titov, 2017], relation extraction [Zhang et al., 2018], event extraction [Liu et al., 2019], and others. Among all GCNs variants, Relational Graph Convolutional Networks (R-GCNs) [Schlichtkrull et al., 2018] is a particular structure that facilitate relational reasoning. To our best knowledge, this is the first work introducing R-GCNs to the task of discourse-level event temporal ordering.

# 3 Approach

Figure 2 schematically visualizes the proposed UCGraph framework. Specifically, at the training time, a graph mask pre-training mechanism is employed to learn the underlying inter-dependencies of edges, by reconstructing an edge conditioned on the remaining graph. At the test time, UCGraph employs an uncertainty-guided graph completion strategy to rank each predicted edge and select the most uncertain one as current prediction. Let a document be  $D$ , and the event set be  $E_{D}$ . Let  $\mathcal{R}$  be the TLINK set. We denote by  $e_i \in E_D$  the  $i$ th event, and  $r_{i,j} \in \mathcal{R}$  the TLINK between  $e_i$  and  $e_j$ . Our model UCGraph aims to structure  $D$  as a complete graph  $G_{D}$ , where the  $i$ th node corresponds to  $e_i$ , and  $r_{i,j}$  is the edge connecting  $e_i$  and  $e_j$ . Considering the reflexivity of TLINKs (e.g.,  $e_i \xrightarrow{\text{BEFORE}} e_j$  always implies  $e_i \xleftarrow{\text{AFTER}} e_j$ ), we assert  $i < j^1$ . In this way, the total number of edges in  $G_{D}$  is  $\frac{|E_{D}| \times (|E_{D}| - 1)}{2}$ .

# 3.1 Graph Mask Pre-Training

To learn the underlying inter-dependencies of temporal relations, UcGraph employs a graph mask pre-training strategy.

Random Edge Masking. Given a training document  $D$  and its temporal graph, we first adopt a random mask strategy to exclude some edges $^2$  in  $G_{D}$  (inspired by the masked language modeling objective in BERT [Devlin et al., 2019]), and then employ a graph model to re-construct these edges conditioned on the remaining graph. Assume a masked edges is  $r_{i',j'}$ , which connects two end nodes  $e_i'$  and  $e_j'$ . Then in the edge reconstruction phase, our goal is to recover  $r_{i',j'}$  by exploring the remaining graph's structure. This graph formulation enables our model to learn the dependency of the masked edge on other edges, and assimilate global features for reasoning.

![](images/aaaec6f7c4a903feb83495160f3de9155d8b2d770e4cc38339a110af616047da.jpg)



Figure 2: The overview of UcGraph. Up: A graph mask pre-training mechanism is adopted to learn the underlying inter-dependencies of edges. Down: An uncertainty-guided strategy is devised at the testing time to learn prediction orders for graph completion.


![](images/2cda858fce375215c391eacca392afe92a61c89146f1bf47d2465fb0b3e30546.jpg)



Figure 3: Graph representation learning via R-GCNs.


Graph Representation Learning via R-GCNs. To recover a masked edge  $r_{i',j'}$  (with two end nodes  $e_i'$  and  $e_j'$ ), we adopt Relational Graph Convolutional Networks (R-GCNs) [Schlichtkrull et al., 2018], involving two major procedures:

1) Node Representations Learning, in which we first learn the node representation of each node in the graph. Particularly, we use the event's within-sentence representation as the initialized node representation, which is computed using Bi-directional LSTM (BiLSTM) [Hochreiter and Schmidhuber, 1997] and BERT [Devlin et al., 2019]. Accordingly, the initialized node representation of  $e_i'$  is denoted by  $h_{e_i'}^{(0)}$ .

2) Graph Representations Learning, in which we further learn the graph-aware representation of a node via R-GCNs. As shown in Figure 3, the graph-aware representation of  $e_i'$  (and  $e_j'$ ) is learned by assimilating representations of its direct neighborhoods, and many R-GCNs layers are stacked to model long-range inter-dependencies. Particularly, the graph representation of  $e_i'$  at the  $l + 1^{th}$  layer is computed as:

$$
\boldsymbol {h} _ {e _ {i} ^ {\prime}} ^ {(l + 1)} = \sigma \left(\sum_ {r \in \mathcal {R}} \sum_ {e _ {k} \in \mathcal {N} ^ {r} \left(e _ {i} ^ {\prime}\right)} \frac {1}{c _ {e _ {i} ^ {\prime} , r}} \boldsymbol {W} _ {r} ^ {(l)} \boldsymbol {h} _ {e _ {k}} ^ {(l)}\right) \tag {1}
$$

where  $\sigma$  denotes the ReLU activate function;  $r$  denotes a particular TLINK label in the pre-defined label set  $\mathcal{R}$ ;  $\mathcal{N}^r(e_i')$  denotes the set of neighbor nodes of  $e_i'$  under relation  $r$ ;  $c_{e_i', r}$  is a normalization constant, and  $\mathbf{W}_r^{(l)}$  is the parameter regarding  $r$  at the  $l$ th layer. The graph representation of  $e_i'$  is set

as R-GCNs' output, denoted as  $H_{e_i'}$ , and the representation of  $e_j'$  is computed in a similar way, denoted by  $H_{e_j'}$ .

Edge Prediction and Optimization. Based on  $H_{e_i'}$  and  $H_{e_j'}$ , we conduct a multi-class classification to predict the masked edge:

$$
\boldsymbol {o} _ {e _ {i} ^ {\prime}, e _ {j} ^ {\prime}} = \operatorname {s o f t m a x} \left(\boldsymbol {W} _ {\text {o u t}} \left[ \boldsymbol {H} _ {e _ {i} ^ {\prime}}; \boldsymbol {H} _ {e _ {j} ^ {\prime}} \right] + b _ {\text {o u t}}\right) \tag {2}
$$

where  $\mathbf{o}_{e_i',e_j'}$  is an output vector containing the predictive probabilities of different TLINK labels, and the predicted result is the label having the largest value in  $\mathbf{o}_{e_i',e_j'}$ ; [;] indicates concatenation computation;  $W_{out}$  and  $b_{out}$  are model parameters. We train our model with cross-entropy loss:

$$
\mathcal {L} (\Theta) = - \sum_ {\left(e _ {i} ^ {\prime}, e _ {j} ^ {\prime}\right) \in T} \mathrm {p} \left(\mathrm {r} _ {\mathrm {e} _ {\mathrm {i}} ^ {\prime}, \mathrm {e} _ {\mathrm {j}} ^ {\prime}} \mid \left(\mathrm {e} _ {\mathrm {i}} ^ {\prime}, \mathrm {e} _ {\mathrm {j}} ^ {\prime}\right)\right) \tag {3}
$$

where  $(e_i', e_j')$  ranges over each masked edge;  $r_{e_i', e_j'}$  denotes the ground-truth TLINK label;  $\mathrm{p}(\mathrm{r}_{\mathrm{e}_i', \mathrm{e}_j'} | (\mathrm{e}_i', \mathrm{e}_j'))$  is the predictive probability of  $r_{e_i', e_j'}$  in  $\mathbf{o}_{e_i', e_j'}$ . We adopt Adam rules [Kingma and Ba, 2015] for model optimization.

# 3.2 Uncertainty-Guided Graph Completion

A gap max exist between training and testing, as in the testing stage we should build a temporal graph from the ground up (i.e., all the edges are masked), and previous predictions affect the following predictions. To minimize error propagation, we propose a "certain-first" strategy to explore the edge prediction orders, as visualized in Figure 2.

Uncertainty Modeling. Assume our model's prediction is  $r$  for a masked edge. We next estimate our model's uncertainty of  $r$  via MC-Dropout [Gal and Ghahramani, 2016]. Specifically, we conduct  $K$  forward passes with dropout layers being activated and get  $K$  output values  $\tilde{r} = [\tilde{r}_1, \tilde{r}_2, \dots, \tilde{r}_K]^4$ . According to [Gal and Ghahramani, 2016], the uncertainty of  $r$  empirically equals to the variance of  $\tilde{r}$ . Considering  $r$  is a categorical value, we adopt Shannon's entropy to

# Algorithm 1 Certain-First Graph Completion

Input: A test document  $D$  annotated with a set of events  $E_{D}$  Output: A complete graph  $G$

1: Transfer  $D$  as an empty graph  $G$  with nodes being  $E_{D}$

2: while  $G$  is not completed do

3: Predict TLINKs for all the missing edges in  $G$

4: Estimate uncertainties of TLINKs via Eq. (4)

5: Select TLINK with minimal uncertainty value as current prediction, and insert the edge into  $G$

# 6: end while

model the variance of  $\tilde{r}$ :

$$
\operatorname {U n c} (\mathrm {r}) = - \sum_ {\mathrm {i} = 1} ^ {\mathrm {K}} \mathrm {p} \left(\tilde {\mathrm {r}} _ {\mathrm {i}}\right) \log \mathrm {p} \left(\tilde {\mathrm {r}} _ {\mathrm {i}}\right) \tag {4}
$$

where  $\mathrm{p}(\tilde{\mathrm{r}}_{\mathrm{i}}) = \frac{\mathrm{N}(\tilde{r}_{\mathrm{i}})}{\mathrm{K}}$  and  $\mathrm{N}(\tilde{r}_i)$  is the frequency of  $\tilde{r}_i$  in  $\tilde{r}$ . Note lower  $\mathrm{Unc}(\mathbf{r})$  indicates smaller uncertainty of  $r$ , which also implies a more reliable prediction.

Certain-First Graph Completion Strategy. We design a certain-first graph completion strategy to decide prediction orders, as shown in Algorithm 1. Specifically, our algorithm starts with an empty graph  $G$ , which corresponds to a test document, and then conducts the following steps:

- Step 1: Pseudo-predict TLINK labels for all the missing edges in  $G$ .

- Step 2: Estimate the uncertainty value for each edge prediction using E.q. (4).

- Step 3: Select an prediction with the minimal uncertainty value, and add the corresponding edge into  $G$ .

- Step 4: Repeat the above steps until  $G$  is completed.

Note in Step 3, when a new edge is added into the  $G$ , the graph structure will change, and thus the next predictions may be different from the current ones. The above algorithm will repeat  $\frac{|E_D| \times (|E_D| - 1)}{2}$  times exactly, which equals to the total number of edges in the graph<sup>5</sup>. We compare our method with different graph completion methods in § 6.2.

# 4 Experimental Setups

Datasets and Evaluation Metrics. We use TDDiscourse, the largest discourse-level event temporal ordering benchmark, as the test bed [Naik et al., 2019]. It includes two subsets: 1) TDD-Man, which augments TimeBank-Dense (TBDense) [Cassidy et al., 2014] by manually annotating TLINKs between event pairs that are more than one sentence apart. 2) TDD-Auto, which derives new TLINKs in the document with automatic inference rules. Table 1 and Table 2 compare the sizes and label distributions of TBDense, TDD-Man and TDD-Auto. We adopt Precision (P), Recall (R), and F1 score (F1) as estimation metrics, same as previous works to ensure comparability [Ning et al., 2017; Naik et al., 2019; Han et al., 2019b].

<table><tr><td>Dataset</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>TBDense [Cassidy et al., 2014]</td><td>4,032</td><td>629</td><td>1,427</td></tr><tr><td>TDD-Man [Naik et al., 2019]</td><td>4,000</td><td>650</td><td>1,500</td></tr><tr><td>TDD-Auto [Naik et al., 2019]</td><td>32,609</td><td>1,435</td><td>4,258</td></tr></table>


Table 1: Number of temporal relations in TBDense, TDD-Man, and TDD-Auto. In TBDense, only event-event TLINKs are counted.


<table><tr><td>Dataset</td><td>a</td><td>b</td><td>s</td><td>i</td><td>ii</td></tr><tr><td>TBDense</td><td>18%</td><td>22%</td><td>2%</td><td>5%</td><td>6%</td></tr><tr><td>TDD-Man</td><td>13%</td><td>27%</td><td>3%</td><td>38%</td><td>19%</td></tr><tr><td>TDD-Auto</td><td>28%</td><td>32%</td><td>16%</td><td>11%</td><td>13%</td></tr></table>

Table 2: Distribution of TLINKs in different datasets. Assume two events are  $e_1$  and  $e_2$ . The TLINK of a defines  $e_1$  occurs after  $e_2$ ; b defines  $e_1$  occurs before  $e_2$ ; s defines  $e_1$  occurs simultaneously as  $e_2$ ; i defines  $e_1$  includes  $e_2$ ; ii defines  $e_1$  is included by  $e_2$ .

Implementation Details. The hyper-parameters of our model are tuned on the development set of TDDiscourse. Finally, for graph mask pre-training, the mask portion is set as  $5\%$  (chosen from  $1\%$  to  $50\%$ , c.f., § 6.1). The number of R-GCN layers is set at 3 (chosen from [1, 2, 3, 4, 5]), and we use DEEP GRAPH LIBRARY $^6$  (DGL) to implement graph convolution algorithm. To learn the node representations, for BERT encoder, we use BERT-Base architecture; for BiLSTM encoder, we use Glove embeddings [Pennington et al., 2014] and set the hidden dimension to 256 (chosen from [64, 128, 256, 512]). In uncertainty modeling, we set  $K$  to 20 to balance speed and efficiency. For testing, we consider event pairs which are 15 or fewer sentences apart following [Naik et al., 2019] (The evaluation is not changed).

Baselines. We compare our method with the following baselines: Majority, which assigns the majority-class label to each event pair. Despite its simplicity, Majority outperforms most existing state-of-the-art methods on TDD-Man [Naik et al., 2019]. CAEVO [Chambers et al., 2014], a previous state-of-the-art method for identifying sentence-level TLINK which heuristic rules. BiLSTM [Cheng and Miyao, 2017], which introduces BiLSTM to learn representation of events to predict TLINKs. SP+ILP [Ning et al., 2017], which adds global constraints via integer linear programming (ILP), aiming to mitigate the problem that CAEVO and BiLSTM make separate local decisions that may result in global inconsistency. Deep SSVM [Han et al., 2019a], which leverages structured support vector machine to make global predictions. Mult-Task [Han et al., 2019b], which jointly predict events and relations. BERT, which adopts BERT [Devlin et al., 2019] to learn event representations (same as our method in learning node representations), but it conducts a pair-wise classification and ignores inter-dependencies. Our approach is denoted by UcGraph. We use UcGraph+BiLSTM and UcGraph+BERT to designate learning node representations via BiLSTM and BERT respectively.
