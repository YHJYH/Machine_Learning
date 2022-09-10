# LTH remove 32% params on VGG16
acc. = [0.7544, 0.7566, 0.7549, 0.7452, 0.7595] mean: 0.7541 std: 0.0048

time: 57.0493 /pm 1.0401

remain params: 22865536<br>
percentage: <br>

LTH_train4_x
```
4_1
Train loss: 0.308225, Valid loss: 0.538320
Updating model file...
Early stopping at: 8
----------------------------------------------
4_2
Train loss: 0.196730, Valid loss: 0.544074
Updating model file...
Early stopping at: 10
----------------------------------------------
4_3
Train loss: 0.246846, Valid loss: 0.512801
Updating model file...
Early stopping at: 9
----------------------------------------------
4_4
Train loss: 0.257393, Valid loss: 0.528910
Updating model file...
Early stopping at: 9
----------------------------------------------
4_5
Train loss: 0.412021, Valid loss: 0.524788
Updating model file...
Early stopping at: 7
----------------------------------------------
```

LTH_features4_x.pt

[57.4172,58.8710,56.6186,56.5039,55.8360]
```
4_1
Test average loss: 1.2370, acc: 0.7544
Test time: 57.4172 s
----------
4_2
Test average loss: 1.3525, acc: 0.7566
Test time: 58.8710 s
----------
4_3
Test average loss: 1.3163, acc: 0.7549
Test time: 56.6186 s
----------
4_4
Test average loss: 1.3695, acc: 0.7452
Test time: 56.5039 s
----------
4_5
Test average loss: 1.2751, acc: 0.7595
Test time: 55.8360 s
----------
```

CKA Linear plot (averaged over 5 exps): <br>
![vgg16_32_linear](vgg16_32_linear.png)

CKA RBF plot (averaged over 5 exps): <br>
![vgg16_32_rbf](vgg16_32_rbf.png)
