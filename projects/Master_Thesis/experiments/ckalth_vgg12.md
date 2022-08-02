# CKA LTH VGG12
| prune % | acc. |
|---------|------|
|    10     |   0.6959   |
|    20     |    0.7019   |
|    30     |  0.7003    |
|    40     |   0.6944   |
|    50     |   0.7098   |
|    60     |   0.7212   |
|    65     |   0.7228   |
|    70     |   0.7268   |
|    73     |   0.7228   |
|    75     |   0.7211   |
|    80     |   0.7054   |
|    90     |   0.3382   |

acc = [0.7217, 0.7255, 0.7231, 0.7233, 0.7257] mean: 0.7239 std: 0.0015

remained params: 10087743

CL_train_model12_x
```
12_1
Train loss: 0.298954, Valid loss: 0.477192
Updating model file...
Early stopping at: 7
----------------------------------------------
12_2
Train loss: 0.248653, Valid loss: 0.487034
Updating model file...
Early stopping at: 8
----------------------------------------------
12_3
Train loss: 0.179698, Valid loss: 0.471260
Updating model file...
Early stopping at: 9
----------------------------------------------
12_4
Train loss: 0.411762, Valid loss: 0.490199
Updating model file...
Early stopping at: 6
----------------------------------------------
12_5
Train loss: 0.233043, Valid loss: 0.476877
Updating model file...
Early stopping at: 8
----------------------------------------------
```

CL_features12_x
```
12_1
Test average loss: 1.5832, acc: 0.7217
----------
12_2
Test average loss: 1.6713, acc: 0.7255
----------
12_3
Test average loss: 1.7293, acc: 0.7231
----------
12_4
Test average loss: 1.3847, acc: 0.7233
----------
12_5
Test average loss: 1.6749, acc: 0.7257
----------
```

CKA Linear avg: <br>
![cl_vgg12_linear](cl_vgg12_linear.png)

CKA RBF avg: <br>
![cl_vgg12_rbf](cl_vgg12_rbf.png)
