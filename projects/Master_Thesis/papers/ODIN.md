# Enhancing the reliability of out-of-distribution image detection in neural networks
New method: **ODIN** (Out-of-DIstribution detector for Neural networks).

using temperature scaling and adding small perturbations to the input can separate the softmax score distributions between in- and out-of-distribution images.

## Introduction
In this [baseline](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/miclassied_or_ood.md#a-baseline-for-detecting-misclassified-and-out-of-distribution-examples-in-neural-networks) work, the authors find that a well-trained neural network tends to assign higher softmax scores to in-distribution examples than out-of-distribution examples.

In ODIN, this **gap** is enlarged by:
1. using temperature scaling in the softmax function;
2. adding small controlled perturbations to inputs

Main contributions:
1. **Simpler**. ODIN. does not require re-training the neural networ;
2. **Better performance**. ODIN can significantly improve the detection performance, and consistently outperforms the baseline method;

## ODIN
Input preprocessing:
1. idea is inspired by the idea of adversarial examples (Goodfellow);
2. in adversarial: small perturbations are added to decrease the softmax score for the true label and force the neural network to make a wrong prediction;
3. here: opposite goal. increase the softmax score of any given input, without the need for a class label at all;
4. the perturbation can have stronger effect on the indistribution images than that on out-of-distribution images;
5. the perturbations is computed by back-propagating the gradient of the cross-entropy loss w.r.t the inputs;



[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
