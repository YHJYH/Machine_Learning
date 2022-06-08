# Similarity of Neural Network Representations Revisited

Get to know the trained NN: by looking at and comparing their representaion.

- Centred kernel alignment (CKA):
    - **meaningful similarities**：can measure meaningful similarities between representations of higher dimension than the number of data points.
    - **different initializations**: can identify correspondences between representations in networks trained from different initializations. 
<br>
<br>
- **representation**: a $n \times m$ matrix.
    - centred: substract the mean across.
    - n: # examples, m: # features.
<br>
<br>
- Comparison between representations: X and Y
    - Note: two representations of different layer but must have $n_{X} = n_{Y}$. 
<br>
<br>
- Similarity
    - simple measurement: dot product (cosine) (linear kernel)
<br>
<br>
- Comparing representation by **comparing features**
    - $X^{T}Y$, a $m_{X} \times m_{Y}$ matrix.
    - dot product between column (features), so it is the similarity between features.
- Comparing representation by **comparing exmaples**
    - $XX^{T}$, a $n_{X} \times n_{X}$ matrix.
    - dot product between row (examples), so it is the similarity between examples.
    - **Gram matrix** (ML) or representational similarity matrix (neuroscience).
<br>
<br>
- back to *Comparison between representations: X and Y*: 
    - each representation X and Y has its own Gram matrix;
    - vectorize the Gram matrix: $vec(XX^{T})$ and $vec(YY^{T})$ of size $(n \times 1)$;
    - dot produc between two Gram vectors;
    - is equal to sum of squared dot products between features: $||X^{T}Y||^{2}\_{F} = \langle vec(XX^{T}, vec(YY^{T}))\rangle$;
    - so **comparing features = comparing examples**.

[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/refs.md#content)
