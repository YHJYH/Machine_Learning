# content
shortcut learning
- [2-stage LCN-HCN](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-lcn-hcn)
- [2-stage just train twice](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-just-train-twice)
- [2-stage LfF](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-learning-from-failure-lff)
- [feature disentanglement](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#feature-disentanglement-in-covid-19-cxr-image-classification)
- [data imbalance](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#data-imbalance)
- [group worst-case loss](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss)
- [training group annotations]()
- [CVaR DRO](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#cvar-dro)

text summarization
- [PlanSum](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#plansum)


## 2-stage LCN-HCN
title: A Too-Good-to-be-True Prior to Reduce Shortcut Reliance

author: Nikolay Dagaev, Bradley C. Love

year: 2021

dataset: CIFAR-10

summary: 介绍了用Low-capacity networks(LCN)来检测shortcuts，然后在high-capacity networks的训练过程中 将检测到的shortcuts downweights。<br>
介绍了local和global两种shortcuts的类型。前者是图片上固定位置的色块记号；后者类似于整张图片覆盖的Gaussian noise。<br>
local的又分为congruent和incongruent两种shortcuts，前者是training set和test set上同一label下的图片上的sc标记颜色位置相同，比如ship(red), plane(blue)；incongruent是同一label下的颜色不同，记住，同一个label下的颜色都是相同的，不同的是同一个label下training set和test set的颜色，比如 train: ship(red), plane(blue); test: ship(blue), plane(red).

question: 
- how to downweights?<br>
    - 通过importance weights (IWs) $w_{i} = 1 - p(y_{i}|x_{i})$ probability of misclassification. w越大说明一个sample越容易被misclassify，说明这是一个**不**包含shortcut的image(worst-group)。用normalized w乘以sample loss，则w越大的sample就会有更大的sample loss, 但是我们希望总的loss减小，所以network会focus在worst-group sample上。但这样怎么是downweight呢？听着像是upweight。<br>
- how to get $p(y_{i}|x_{i})$ empirically?

## feature disentanglement in COVID-19 CXR image classification
title: Deep learning models for COVID-19 chest x-ray classification: Preventing shortcut learning using feature disentanglement

author: Caleb Robinson, Juan M. Lavista Ferres
 
year: 2021

summary: 这篇文章借用了一种“feature 反纠缠”的方法，这种方法的特色是，我们提前知道shortcut feature是哪个了，我们通过最大化shortcut feature的loss来使其不被我们的networks care，即把shortcut feature从所有feature里disentangle出来。<br>
同时feature entanglement也是建立在transfer learning的基础上，即一部分参数是训练好然后fronzen的。<br>
具体来说主要分为两个parts, 第一：我们通过feature extractor $g(x, \theta)=z$ (这一步类似于kernel)得到提取的feature z，但是这个z不直接参与estimator $f(z, \phi)$ ，而是再经过一个feature extractor $f_{e}(z, \phi_{e})$ 得到z'。再将这个z'喂给两个classifiers，一个classifier是用来classify shortcut feature的（recall上面提过，sc我们已经提前知道是哪个了），另一个则是其他features。第二就是使用的loss function了，是一个min-max的过程（和GAN有点像？），如下所示。（domain就是shortcut feature）<br>
![fdeq2](./pics/fdeq2.PNG)<br>
![fdeq34](./pics/fdeq34.PNG)<br>
网络结构如下所示：
![fdnet](./pics/fdnet.PNG)<br>

question: 按照0028里学的，如果要彻底disentangle，那\phi_{d}不应该直接等于0吗，表示feature z'和domain label y^{d}没有correlation。<br>
参数更新文章里写的不是很清楚，但是可以大概看出来应该是fix一个，更新另一个这种iterative的模式。

## data imbalance
title: Simple data balancing achieves competitive worst-group-accuracy

author: Badr Youbi Idrissi, David Lopez-Paz
 
year: 2022

summary: 

question: 

## PlanSum
title: Unsupervised Opinion Summarization with Content Planning

author: Reinald Kim Amplayo, Stefanos Angelidis, Mirella Lapata

year: 2021

dataset: Rotten Tomatoes, Yelp, Amazon

summary:

## group worst-case loss
title: distributionally robust neural networks for group shifts: on the importance of regularization for worst-case generalization

author: Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang

year: 2020

dataset: WaterBirds, CelebA, MultiNLI

summary: worst group指的是训练过程中表现training accuracy最低的那一些数据集合成的group，这个group的acc低的原因是因为NN学习到了一些错误的相关性（correlation），比如在识别任务中过分关注于背景而非物体本身。<br>
一般group DRO和ERM方法在训练中出现的现象如下：
- test group的average acc很
- worst group的training acc很高
- worst group的test acc很低

这说明worst-group的generalization gap很大（第三条）,尽管on average generalization gap不大（第一条）。<br>
在本文中generalization gep的定义是expected error - empirical error (在同一dateset上，一般是test set)。<br>

本文使用的method是strongly-regularized group DRO, 包括三个部分的变化：
1. L2 penalty
2. early stopping
3. *group adjustment*: 这一点证明了regularization对于整体的avg performance不一定有帮助，但是对worst-group performance还是很有帮助的。

本文使用的方法本质上还是**DRO** (distributionally robust optimization): 找到参数可以minimize empirical worst-group risk， worst-group risk通过将数据分类成不同的groups s.t. maximize expected loss of each group来obtain。具体两个公式如下所示。<br>
worst-case risk (maximum over the expected loss of each group):
![wgr1](./pics/wgr1.PNG) <br>
group DRO model (minimize the empirical worst-case risk):
![wgr2](./pics/wgr2.PNG) <br>

Y(labels) = {Y1, Y2}, A(shortcut features) = {A1, A2}, # groups m = |Y|\*|A| = 4<br>
如果一组数据，label都是Y1，都有A1 feature，且training loss很低（表示学到了A1 和 Y1的correlation），那么model在{Y1, A2}上的表现就应该很差。这种{Y1, A2}, {Y2, A1}就是worst-group。

结果： 在maintain high avg acc的同时，本文通过上述方法很大程度提升了worst-group的acc。本文是建立在overparameterized NN上（即有很多参数，使training acc很高的同时也保证了generalize well on avg, but not on the worst-group）。
- 使用strong L2 regularization和early stopping：1) 使DRO模型的training acc降低; 2) 减少了group的generalization gap（high worst-group acc -> high worst-group test acc）。
- DRO表现得都比ERM要好。
- image 任务普遍比NLI任务要好。
- group adjusted DRO 表现更进一步。
- 新介绍了一种一定会converge的gradient descent algo for group DRO: online optimization algo for group DRO.

## 2-stage just train twice
title: Just Train Twice: Improving Group Robustness without Training Group Information

author: Evan Zheran Liu, Behzad Haghgoo, Annie S. Chen, Aditi Raghunathan, Pang Wei Koh, Shiori Sagawa, Percy Liang, Chelsea Finn

year: 2021

dataset: Waterbirds, CelebA, MultiNLI, CivilComments-WILDS

summary: 

previous的解决方法：training group annotations [Sagawa et al., 2020a](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#training-group-annotations), 缺点：expensive。

JTT, two-stage approach.
- stage 1: train initial model and identify examples with high training loss (*worst-group*);
- stage 2: train again with upweighted worst-group examples.

stage 1 (identification): 通过ERM训练一个identification model $\hat{f}\_{id}$, 确定一个将 $\hat{f}\_{id}$ misclassifies的examples放进error set E。<br>
stage 2 (upweighting): train a final model $\hat{f}\_{final}$ by upweighting points in E.<br>
Both stages using ERM objective function.

![jtt](./pics/JTT.PNG)

JTT:  only requiring group annotations on a much smaller validation set to tune hyperparameters. 将misclassified examples直接当作worst-group examples。

group robustness: i.e., training models that obtain good performance on each of a set of predefined groups in the dataset

capacity control: early stopping, strong L2 regularization.

baselines:
- do not use training group annotation
    - ERM的问题：整体的avg training loss降底但是certain group还是有high error；造成这种情况的原因：spurious correlation（shortcuts）。
    - 和JTT思想相似的一个方法是(DRO) that minimizes the conditional value at risk (CVaR)： [CVaR DRO](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#cvar-dro)。但是JTT比CVaR DRO表现要好。两者的区别是JTT upweight的examples是固定的（static），CVaR DRO是动态upweight minibatch里的examples。including a uncertainty set.
    - [LfF method](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#learning-from-failure-lff).
- do use training group annotation
    - Group DRO. Using group annotation to define uncertainty set. 选定一个组，这个组的empirical risk最大。这个方法需要给training set的data annotate（JTT不需要）。

结论：
- JTT consistently achieves higher worst-group accuracy on all 4 datasets.
- JTT performs well even relative to approaches that use training group information. 
- JTT recovers a significant portion of the gap in worst-group accuracy between ERM and group DRO.
- note that simple label balancing also achieves comparably worst-group accuracy to group DRO on CivilComments.
- 有 modest drop in average accuracy。符合“ a tradeoff between average and worst-group accuracies.”的预期。[Sagawa et al., 2020a](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#training-group-annotations)
- Dynamically computing the error set in JTT lowers accuracy. (so better than CVaR DRO)

interesting related work mentioned (all require group annotations):
- synthetically expand the minority groups via generative modeling []()
- reweight or subsample the majority and minority groups []()
- impose heavy Lipschitz regularization around minority points []()
- equalize loss across groups []()

does not require group annotations:
- automatically identify groups based on clustering and improve robustness via approaches that use this learnt group information []()
- directly learn to reweight the training examples either using small amount of metadata []()
- Learning from Failure (LfF): simultaneously learns a pair of models [LfF](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#learning-from-failure-lff)

Q：
- misclassified examples确实是属于worst-group的examples但是不代表这些examples属于同一个group i.e. 有相同的shortcuts。
    - JTT不care具体的shortcuts，都在uncertainty set E里面.
- static为什么就比dynamic要好呢？(future work 1)

## training group annotations
title:

author:

year: 2020

dataset:

summary:

## CVaR DRO
title: Large-Scale Methods for Distributionally Robust Optimization

author: Daniel Levy, Yair Carmon, John C. Duchi, Aaron Sidford

year: 2020

dataset:

summary: 

## 2-stage Learning from Failure (LfF)
title: Learning from Failure: Training Debiased Classifier from Biased Classifier

author: Junhyun Nam, Hyuntak Cha, Sungsoo Ahn, Jaeho Lee, Jinwoo Shin

year: 2020

dataset: colored MNIST, Corrupted CIFAR-10.

summary: <br>
Two findings:
- 只有当bias attribute比target attribute更容易学得的时候bias才会negatively affects the model;
- classifier在训练早期会学得bias，后期会学得其他attributes。networks tend to defer learning hard concepts. 

Terms: 
- intended decision rule: decision rules that correctly classify images based on the target attribute.
- unintended decision rule: decision rules based on other attributes i.e. bias attribute.
- bias-aligned samples: samples can be correctly classified by the unintended decision rule.
- bias-conflicting samples: by intended decision rule. 
- malignant and benign bias: the bias attribute inducing malignant bias is “easier” to learn than the target attribute, e.g., Color is easier to learn than Digit.

Debiasing scheme LfF 基本概况:
1. NN1: biased. focus on easy samples (samples aligned with bias). 
2. NN2: debiased. focus on samples that the biased model struggles to learn (samples conflict with the bias i.e. worst group due to NN1).
3. re-weight training samples using relative difficulty score based on loss of two NNs.

算法细节：<br>
![lff2](./pics/lff2.PNG) <br>
![lff1](./pics/lff1.PNG) <br>
- biased model $f_{B}$ . i.e. the model following the unintended decision rule.
    - use generalized cross entropy (GCE) to amplify NN bias: $p(x; \theta)$ is softmax output, the other prob. is probability aligned to the target attribute y. $q$ controls degree of amplification, a hyperparam.
    - 

