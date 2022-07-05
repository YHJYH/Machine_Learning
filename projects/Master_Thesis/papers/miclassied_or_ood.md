# A BASELINE FOR DETECTING MISCLASSIFIED AND OUT-OF-DISTRIBUTION EXAMPLES IN NEURAL NETWORKS
_use confidence scores to determine if samples are out-of-distribution._

在这篇文章中，作者认为传统的使用softmax来做概率预测会造成high confidence but wrong prediction的局面，因此使用maximum softmax prediction probability建立了一个新的baseline来检测不同数据集和模型结构的error and ood examples.之后 introduce了一种abnormality module, 通过sigmoid function来输出一个confidence score showing an example有多abnormal (ood)，新介绍的module击败了作者自己介绍的baseline.

## Introduction
If an example is misclassified or out-of-distribution.

softmax function produces high-confidence predictions:
- Minor additions to the softmax inputs, i.e. the logits, can lead to substantial changes in the output distribution.
- a smooth approximation of an indicator function
- the prediction probability from a softmax distribution has a poor direct correspondence to confidence

**Proposed new method**: This new method evaluates the quality of a neural network’s input reconstruction to determine if an example is abnormal.

**Main contribution**: the designation of standard tasks and evaluation metrics for assessing the automatic detection of errors and out-of-distribution examples. 

**Main contribution**: For out-of-distribution detection, we provide ways to supply the out-of-distribution examples at test time.

**Conclusion**: Simple statistics **derived from softmax distributions** provide a surprisingly effective way to determine whether an example is misclassified or from a different distribution from the training data.


## Problem formulation
**Problem 1**: error and success prediction
- can we predict whether a trained classifier will make an error on a particular held-out test example;
- can we predict if it will correctly classify said example? 

**Problem 2**: in- and out-of-distribution detection
- can we predict whether a test example is from a different distribution from the training data;
-  can we predict if it is from within the same distribution?
-  adversarial example detection is considered in a seperate work. [ref]()

Used metrics:
1. AUROC  Area Under the Receiver Operating Characteristic curve 
    - a threshold-independent performance evaluation
    - the AUROC can be interpreted as the probability that a positive example has a greater detector score/value than a negative example
    - E.g., a random positive example detector corresponds to a 50% AUROC, and a “perfect” classifier corresponds to 100%
    - not ideal when the positive class and negative class have greatly differing base rates
2. AUPR  Area Under the Precision-Recall curve
    - The baseline detector has an AUPR approximately equal to the precision, and a “perfect” classifier has an AUPR of 100%


## Softmax prediction probability as a baseline
1. we separate correctly and incorrectly classified test set examples and, for each example, compute the softmax probability of the predicted class, i.e., **the maximum softmax probability**.
2. obtain the area under PR and ROC curves. These areas summarize the performance of a binary classifier discriminating with values/scores across different thresholds.

## Abnormality detection with auxiliry decoders
Softmax prediction probabilities enable abnormality detection.

1. Training a normal classifier and append an **auxiliary decoder** which reconstructs the input. The decoder and scorer (softmax) are trained jointly on in-distribution examples (blue layers).
2. the sigmoid output of the red (both in- and out-of-distribution samples) layers **scores how normal** the input is.
3. noised examples are in the abnormal class, clean examples are of the normal class, and the sigmoid is trained to output to which class an input belongs.
4. a normal classifier, an auxiliary decoder, and what we call an **abnormality module (the abnormality scorer)**.

[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
