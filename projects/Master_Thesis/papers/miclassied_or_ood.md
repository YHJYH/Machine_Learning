# A BASELINE FOR DETECTING MISCLASSIFIED AND OUT-OF-DISTRIBUTION EXAMPLES IN NEURAL NETWORKS
_use confidence scores to determine if samples are out-of-distribution._

## Introduction
If an example is misclassified or out-of-distribution.

softmax function produces high-confidence predictions:
- Minor additions to the softmax inputs, i.e. the logits, can lead to substantial changes in the output distribution.
-  a smooth approximation of an indicator function
- the prediction probability from a softmax distribution has a poor direct correspondence to confidence

**Proposed new method**: This new method evaluates the quality of a neural network’s input reconstruction to determine if an example is abnormal.

**Main contribution**: the designation of standard tasks and evaluation metrics for assessing the automatic detection of errors and out-of-distribution examples. 

[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
