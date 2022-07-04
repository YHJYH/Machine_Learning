# thesis structure (new)
- Out-of-distribution generalisation in deep neural networks:
    - definition, 
    - examples of DNNs' failures when the data distribution shifts
        - texture bias, 
        - adversarial examples, [[1]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/Surface_Statistical_Regularities.md#measuring-the-tendency-of-cnns-to-learn-surface-statistical-regularities)
        - noisy data
- Current approach and hypothesis (how are ood generalization and shortcut learning related) [[4]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/adversarial_examples_are_features.md#adversarial-examples-are-not-bugs-they-are-features)[[5]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/IRM.md#invariant-risk-minimization)[[6]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/pitfall.md#the-pitfalls-of-simplicity-bias-in-neural-networks)[[7]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss)
- Shortcuts:
	- general definition, [[2]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/shortcut_learning_in_deep_NN.md#shortcut-learning-in-deep-neural-networks)
	- examples, [[3]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/bias_towards_texture.md#imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness)
	- reasons for why shortcut reliance should be avoided, 
	- existing methods for reducing shortcut reliance (loss function [[5]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/IRM.md#invariant-risk-minimization),datasets [[6]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/pitfall.md#the-pitfalls-of-simplicity-bias-in-neural-networks), regularizations [[7]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss)[[11]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/PARAMETER_FUNCTION_MAP_IS_BIASED_TOWARDS_SIMPLE_FUNCTIONS.md#deep-learning-generalizes-because-the-parameter-function-map-is-biased-towards-simple-functions), two-stage methods [[8]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-learning-from-failure-lff)[[9]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-just-train-twice)[[10]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-lcn-hcn)) 
- New mthods
    - thoery (CKA[[12]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/similarity_of_NN_CKA.md#similarity-of-neural-network-representations-revisited))


# thesis structure (before)
- Introduction
    - Shortcut learning in deep eural networks
    - Thesis overview
- Literature review/Related work (inductive bias)
    - representation similarity metrics: [CKA](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/similarity_of_NN_CKA.md#similarity-of-neural-network-representations-revisited)
    - Architecture: 2-stage methods([LCN-HCN](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-lcn-hcn), [JTT](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-just-train-twice), [LfF](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-learning-from-failure-lff)), [strong regularized group DRO](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss), [debias model](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/learn_debias.md#learning-de-biased-representations-with-biased-representations)
    - Training data: [texture bias](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/bias_towards_texture.md#imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness), [adversarial vulnerability](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/adversarial_examples_are_features.md#adversarial-examples-are-not-bugs-they-are-features), [modular synthetic and image-based datasets](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/pitfall.md#the-pitfalls-of-simplicity-bias-in-neural-networks)
    - Loss function: [IRM](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/IRM.md#invariant-risk-minimization)
    - Optimization: [online GD to train group DRO](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss)
- Methods
- Extension of methods



design exps for shortcuts in img cls, based on 2-stage LCN-HCN we may try feature disentanglement.

main and supplement exps

## code structure
`utils.py`:
- feature extraction [DONE]
- gram matrix calcularion
    - under different types of kernel
- CKA calculation
- similarity matrix plot
- 

