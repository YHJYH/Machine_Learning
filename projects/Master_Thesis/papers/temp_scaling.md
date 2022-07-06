# On Calibration of Modern Neural Networks
- Confidence calibration (calibrating predictions):  predicting probability estimates representative of the true correctness likelihood.
- depth, width, weight decay, and Batch Normalization are important factors influencing calibration.
- authors find *temperature scaling* – a singleparameter variant of Platt Scaling, is very effective at calibrating predictions.
- 所谓Confidence calibration指的是我们对预测出来的概率有多confident.

## Introduction
A network should provide a calibrated confidence measure in addition to its prediction:  indicate when they are likely to be incorrect. 

Observation: The average confidence of LeNet closely matches its accuracy, while the average confidence of the ResNet is substantially higher than its accuracy. 

> the confidence estimate Pˆ to be calibrated, which intuitively means that Pˆ represents a true probability.

## Conclusion
1 observation: probabilistic error and miscalibration worsen even as classification error is reduced. <br>
1 demonstration: neural network architecture and training – model capacity, normalization, and regularization – have strong effects on network calibration. <br>
1 technique: Temperature scaling is the simplest, fastest, and most straightforward of the methods that can can effectively remedy the miscalibration phenomenon in neural networks.

[back](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/111.md#content)
