# The Pitfalls of Simplicity Bias in Neural Networks

linear-like simple classifier vs. nonlinear complex classifier.<br>

## datasets
different coordinates/blocks define decision boundaries of varying complexity.<br>
- *feature*: each coordinate/block.
- *feature simplicity*: based on the simplicity of the corresponding decision boundary.

![pitfall1](pitfall1.PNG)
> Figure 1. A stylized version of the proposed synthetic dataset with two features, φ1 and φ2, that can perfectly predict the label with 100% accuracy, but differ in simplicity.

**modular synhexthetic dataset**:
- the simplicity of a feature is precisely determined by the minimum number of lienar pieces in the decision boundary that achieves optimal classification accuracy using that feature.
    - Fig.1. the simple feature φ1 requires a linear decision boundary to perfectly predict the label, whereas complex feature φ2 requires four linear pieces.

**image-based dataset**:
- each image concatenates MNIST images (simple feature) and CIFAR-10 images (complex feature).

observations:
1. the ideal decision boundary that achieves high accuracy and robustness relies on *all features to obtain a large margin*. 
    - the orange decision boundary in Figure 1 that learns φ1 and φ2 attains 100% accuracy and exhibits more robustness than the linear boundary because of larger margin. 
2. 
3. 

[back]()
