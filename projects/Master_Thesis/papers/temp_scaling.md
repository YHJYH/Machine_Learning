# On Calibration of Modern Neural Networks
- Confidence calibration (calibrating predictions):  predicting probability estimates representative of the true correctness likelihood.
- depth, width, weight decay, and Batch Normalization are important factors influencing calibration.
- authors find *temperature scaling* – a singleparameter variant of Platt Scaling, is very effective at calibrating predictions.
- 所谓Confidence calibration指的是我们对预测出来的概率有多confident.

# Introduction
A network should provide a calibrated confidence measure in addition to its prediction:  indicate when they are likely to be incorrect. 

Observation: The average confidence of LeNet closely matches its accuracy, while the average confidence of the ResNet is substantially higher than its accuracy. 

