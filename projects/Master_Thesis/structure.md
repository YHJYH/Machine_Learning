# thesis structure (new)
- Gaps: Deep CNNs read differently from human vision. [[22]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/Intriguing_properties_of_NN.md#intriguing-properties-of-neural-networks) [[23]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/Deep_NNs_are_Easily_Fooled.md#deep-neural-networks-are-easily-fooled-high-confidence-predictions-for-unrecognizable-images) [[24]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/comparison_of_human_and_DL_recognition_performance.md#a-study-and-comparison-of-human-and-deep-learning-recognition-performance-under-visual-distortions) [[26]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/close_gap.md#partial-success-in-closing-the-gap-between-human-and-machine-vision)
    - 深度cnn看的和human vision不一样，主要是两个原因，其一是distribution不同，cnn只学到了固定distribution内的statistics，而不是global的；第二则是因为shortcuts，即使在同一个distribution内（参考包含月亮星星位置图片那篇文章）cnn学习到的feature可能是和object本身无关的，比如背景，texture等。
    - feature map在cnn中传递的过程展示了cnn是如果抓取，以及抓取了哪些信息。如果两个layers的CKA similarity很相似，那么我们hypothesis在feature传递的过程中，NN没有学习到新的内容，那么假设从layer x到layer x+n的CKA similarity都很高的话，将layer x的feature map直接输入到layer x+n+1应当对于model的performance没有大的影响。即使对于accuracy会有一些影响，减少的parameters的数量和更加简单的network的architecture对于模型部署以及（）会更加有利。
- Out-of-distribution generalisation in deep neural networks:
    - definition, 
    - examples of DNNs' failures when the data distribution shifts
        - texture bias, [[20]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/texture_bias.md#the-origins-and-prevalence-of-texture-bias-in-convolutional-neural-networks) 
        - adversarial examples, [[1]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/Surface_Statistical_Regularities.md#measuring-the-tendency-of-cnns-to-learn-surface-statistical-regularities) [[21]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/adversarial_attacks_survey.md#threat-of-adversarial-attacks-on-deep-learning-in-computer-vision-a-survey)
        - noisy data [[23]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/Deep_NNs_are_Easily_Fooled.md#deep-neural-networks-are-easily-fooled-high-confidence-predictions-for-unrecognizable-images) [[27]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
- current approaches (detection/seperation of ood samples) [[19]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/temp_scaling.md#on-calibration-of-modern-neural-networks)[[18]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/ODIN.md#enhancing-the-reliability-of-out-of-distribution-image-detection-in-neural-networks)[[17]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/miclassied_or_ood.md#a-baseline-for-detecting-misclassified-and-out-of-distribution-examples-in-neural-networks)
- Current approach and hypothesis (how are ood generalization and shortcut learning related) [[4]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/adversarial_examples_are_features.md#adversarial-examples-are-not-bugs-they-are-features)[[5]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/IRM.md#invariant-risk-minimization)[[6]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/pitfall.md#the-pitfalls-of-simplicity-bias-in-neural-networks)[[7]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss)[[16]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/gradient_starvation.md#gradient-starvation-a-learning-proclivity-in-neural-networks)
- Shortcuts:
	- general definition, [[2]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/shortcut_learning_in_deep_NN.md#shortcut-learning-in-deep-neural-networks)
	- examples, [[3]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/bias_towards_texture.md#imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness)
	- reasons for why shortcut reliance should be avoided, 
	- existing methods for reducing shortcut reliance (loss function [[5]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/IRM.md#invariant-risk-minimization),datasets [[6]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/pitfall.md#the-pitfalls-of-simplicity-bias-in-neural-networks) [[25]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/manitest.md#manitest-are-classifiers-really-invariant) invariance of classifiers, regularizations [[7]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#group-worst-case-loss)[[11]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/PARAMETER_FUNCTION_MAP_IS_BIASED_TOWARDS_SIMPLE_FUNCTIONS.md#deep-learning-generalizes-because-the-parameter-function-map-is-biased-towards-simple-functions), two-stage methods [[8]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-learning-from-failure-lff)[[9]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-just-train-twice)[[10]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-lcn-hcn)[[13]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/learn_debias.md#learning-de-biased-representations-with-biased-representations)) 
- New mthods
    - thoery (CKA[[12]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/similarity_of_NN_CKA.md#similarity-of-neural-network-representations-revisited), complexity reduction[[15]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/lottery_ticket.md#the-lottery-ticket-hypothesis-finding-sparse-trainable-neural-networks) feature separation (if time allows)[[3]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/bias_towards_texture.md#imagenet-trained-cnns-are-biased-towards-texture-increasing-shape-bias-improves-accuracy-and-robustness))
    - experiments (一部分内容写法可以参考[[14]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/remove_inner_loop.md#rapid-learning-or-feature-reuse-towards-understanding-the-effectiveness-of-maml))
    - results (compare with lottery ticket[[15]](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/lottery_ticket.md#the-lottery-ticket-hypothesis-finding-sparse-trainable-neural-networks))


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

