- Introduction
    - Shortcut learning in deep eural networks
    - Thesis overview
- Literature review/Related work (inductive bias)
    - representation similarity metrics: [CKA](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/similarity_of_NN_CKA.md#similarity-of-neural-network-representations-revisited)
    - Architecture: 2-stage methods([LCN-HCN](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-lcn-hcn), [JTT](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-just-train-twice), [LfF](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-learning-from-failure-lff)), [strong regularized group DRO](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss), [debias model](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/learn_debias.md#learning-de-biased-representations-with-biased-representations)
    - Training data: [texture bias](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/bias_towards_texture.md#imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness), [adversarial vulnerability](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/adversarial_examples_are_features.md#adversarial-examples-are-not-bugs-they-are-features)
    - Loss function: [IRM](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/IRM.md#invariant-risk-minimization)
    - Optimization: [online GD to train group DRO](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss)
- Methods
- Extension of methods



design exps for shortcuts in img cls, based on 2-stage LCN-HCN we may try feature disentanglement.

main and supplement exps

## code structure
`utils.py`:
- feature extraction
- CKA calculation
- confusion matrix plot

