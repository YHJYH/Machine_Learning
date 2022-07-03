# Measuring the tendency of CNNs to Learn Surface Statistical Regularities
这篇文章注意学习一下作者是怎么先提出一个比较简陋的hypothesis,然后随着对内容的探讨愈加深入，逐渐formally state main hypothesis.

### introduction
Deep CNNs have extreme sensitivity to _adversarial examples_:  the CNNs predict the wrong label, usually with very high confidence. 

So, _how can a network that is not learning high level abstract concepts manage to generalize so well?_

The model has a tendency to overfit to superficial cues that are actually present in both the train and test datasets;<br>
Thus the statistical properties of the dataset plays a key role.

Main hypothesis: the current incarnation of deep neural networks has a tendency to learn surface statistical regularities in the dataset. 

Construct a perturbation map F:  **Fourier filtering**.
- two types of Fourier filtering schemes: radial and random.
- exhibiting up to a 28% gap in test accuracy. 
- increasing the depth of the CNN in a significant manner (going from 92 layers to 200 layers) has a very small effect on closing the generalization gap.


Our last set of experiments involves training on the fully augmented training set, which now enjoys a variance of its Fourier image statistics. We note that this sort of data augmentation was able to **close the generalization gap**. However, we stress that it is doubtful that this sort of data augmentation scheme is sufficient to enable a machine learning model to truly learn the semantic concepts present in a dataset. Rather this sort of data augmentation scheme is analogous to adversarial training: there is a nontrivial regularization benefit, but it is not a solution to the underlying problem of not learning high level semantic concepts, nor do we aim to present it as such.

## sec 2: the generalization ability of a machine learning model and its relation to the surface statistical regularities of the dataset.

Claim #1: Deep CNNs are **generalizing extremely well** to an unseen test set.<br>
Claim #2: General sensitivity to adversarial examples show that deep CNNs are **not truly capturing abstractions in the dataset**.

key intuition: __there is actually a strong statistical relationship between image statistics and visual understanding.__

To this end, we formally state our main hypothesis:
> The current incarnation of deep neural networks exhibit a tendency to learn surface statistical regularities as opposed to higher level abstractions in the dataset. For tasks such as object recognition, due to the strong statistical properties of natural images, these superficial cues that the deep neural network have learned are sufficient for high performance generalization, but in a narrow distributional sense.

[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
