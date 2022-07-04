# Adversarial Examples are not Bugs, they are Features
- Adversarial examples for images are images with intentionally perturbed pixels with the aim to deceive the model during application time. [source](https://christophm.github.io/interpretable-ml-book/adversarial.html#:~:text=is%20very%20educational.-,Adversarial%20examples%20for%20images%20are%20images%20with%20intentionally%20perturbed%20pixels%20with%20the%20aim%20to%20deceive%20the%20model%20during%20application%20time.,-The%20examples%20impressively)
- claim: *adversarial vulnerability is a direct result of sensitivity to well-generalizing features in the data.*. 文章观点：对抗易受性（准确率容易受到adversarial examples的影响）是对数据集中更容易捕获的features过于敏感的直接结果。
- 假设：模型依赖于这些non-robust features，导致adversarial perturbation可以利用这种依赖性。
- 实验设计：
    - 实验1：robust and non-robust features disentanglement. 将一个training set中的数据变换为一个robust dataset和一个non robust dataset，并在两个datasets上分别训练。在robust dataset上训练的model会有good standard & robust accuracy。第二个model则只会有good standard accuracy。robust accuracy指test set with adversarial features。
    - 实验2：一个training img无论使用robust 或者是non robust feature训练出来都会predict出正确的label，但是通过max不正确的label的probability，将training img变成adversarial example（错误的label，non robust会predict 错误label，robust会predict正确label），再在original test set上进行测试，依旧得到good accuracy。 
- 结论：
    - 结论1（实验1）：通过移除dataset的特定features可以提升模型的鲁棒性；adversarial vulnerability是由non-robust features 造成的，与training method无关。
    - 结论2（实验2）：通常model用non-robust features来做预测，即使有robust feature存在；这些通过non-robust feature做的预测并不是overfitting，而是确实有预测性的。

- setup:
    - binary classification
    - feature: function mapping from the input space to real numbers, mean-zero, uni-variance.
        - $\rho$ -useful features. A feature $f$ is $\rho$ -useful if it is correlated with the true label in expectation. Feature and label are positively correlated, i.e. same sign, product greater than $\rho$.
        - $\gamma$-robustly useful features. A useful feature f remains useful under adversarial perturbation. Perturbed feature and label are positively correlated, i.e. same sign, product greater than $\gamma$.
        - useful, non-robust features. Correlation with the labal can be flipped under perturbation.
    - classification: sgn(linear function).
    - training:
        - standard training. ERM.
        - robust training. Making useful but non-robust feature anti-correlated with the true label. Perturbation that maximises the loss. Use *adversarial loss function* discerns between robust and non-robust features.

[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
