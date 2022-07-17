# The Origins and Prevalence of Texture Bias in Convolutional Neural Networks (2020)

**A way of Data augmentation to reduce texture bias and increase shape bias.**

Finding: 
- when trained on datasets of images with **conflicting shape and texture**, CNNs learn to classify by shape at least as easily as by texture.
- apparent differences in the way humans and ImageNet-trained CNNs process images may arise from differences in the data that they see.

why texture bias is important:
- related to the vulnerability of CNNs to adversarial examples,
- difficult to generalize to different distributions than the distribution on which the model is trained (ood distribution),
-  raises an important puzzle for human-machine comparison studies.

what makes ImageNet-trained CNNs classify images by texture when humans do not?<br>
ans:  the most important factor is the data itself.

takeways keys:
- naturalistic data augmentation involving color distortion, noise, and blur substantially decreases texture bias, whereas random-crop augmentation increases texture bias.
-  architectures that perform better on ImageNet generally exhibit lower texture bias
-  which shape information is represented in an ImageNet-trained model from how much it contributes to the model’s classification decisions. 

介绍了两个和我thesis相关的部分：
- Sensitivity of CNNs to non-shape features.
- Similarity of human and CNN perceptual biases.

## The role of data augmentation in texture bias
1. Random-crop data augmentation increases texture bias.
    - Center-crop augmentation reduced texture bias relative to random-crop augmentation.
    - center-crop models had higher shape bias than random-crop models throughout the training process.
2. Appearance-modifying data augmentation reduces texture bias. 

[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
