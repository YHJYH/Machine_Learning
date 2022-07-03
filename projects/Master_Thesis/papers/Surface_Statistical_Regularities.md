# Measuring the tendency of CNNs to Learn Surface Statistical Regularities
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



[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
