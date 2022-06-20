# IMAGENET-TRAINED CNNS ARE BIASED TOWARDS TEXTURE; INCREASING SHAPE BIAS IMPROVES ACCURACY AND ROBUSTNESS
- Motivation: how do CNNs recognise objects
    - by shape or texture? Texture.
    - experiments: use style transfer generate dataset (shape of one category and texture of another)
    - human has shape bias and CNNs have texture bias
    - **texture bias is easier to learn**
- induce shape bias in CNNs
    - synthetic dataset: texture is not correlated with category.
    - experiments: CNNs move toward shape bias
    - **Using suitable dataset to let CNNs develop a shape bias**
- benefits of shape bias
    - improved performance in object recognition and detection
- **robustness towards distortion/noise**: destroying texture while shape is still intact
    - results: huamn > shape-biased CNN > texture-biased CNN
    - noise is not introduced during training


[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
