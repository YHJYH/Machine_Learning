# Learning De-biased Representations with Biased Representations

Problem: we cannot know every bias feature exactly -> so we cannot build a totally unbiased dataset.

Method: 有点GAN的味道。<br>
1. 训练一个original task model（包含bias和non-bias features）;
2. intentionally train a set of biased models（只包含bias features）;
3. force the original model to learn a different representation;

Questions:
1. how to characterize bias with models (Method No.2).
2. how to encode "be different" (Mehod No.3).

Key points of answers:
1. a model can be intentionally biased towards texture by reducing the *receptive fields*
![debias1](debias1.PNG)
![debias2](debias2.PNG)
![debias3](debias3.PNG)

2. encode to be different through *statistical independence*: measure with **HSIC**
    - HSIC(U,V)=0 iff two representations U and V are independent;
    - "be different" = minimize HSIC
![debias4](debias4.PNG)
<br>
<br>

Proposed Method: ReBias<br>
![debias5](debias5.PNG)
![debias6](debias6.PNG)
![debias7](debias7.PNG)
![debias8](debias8.PNG)
![debias9](debias9.PNG)

<br>
<br>

Experiment results: <br>
ReBias effectively removes bias in Biased MNIST, ImageNet (-A and -C) classification, and action recognition.

[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/refs.md#content)
