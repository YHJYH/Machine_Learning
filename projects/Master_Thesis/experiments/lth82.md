# LTH remove 82% params on VGG16
acc. = [0.7045, 0.6491, 0.7035] mean:0.6857  std: 0.0259

time = [59.8993, 54.3612,61.6313] mean:58.6306 std:3.1006

lr = 0.1

remain params: 6052651<br>
percentage: <br>

LTH_features82_x.pt
```
82_1
Test average loss: 0.9594, acc: 0.7045
Test time: 59.8993 s
----------
82_2
Test average loss: 1.2149, acc: 0.6491
Test time: 54.3612 s
----------
82_3
Test average loss: 0.9986, acc: 0.7035
Test time: 61.6313 s
----------
82_4

82_5

```

LTH_train_model82_x
```
82_1
Train loss: 0.751883, Valid loss: 0.821070
Updating model file...
Early stopping at: 8
----------------------------------------------
82_2
Train loss: 0.762653, Valid loss: 0.813469
Updating model file...
Early stopping at: 8
----------------------------------------------
82_3
Train loss: 0.699773, Valid loss: 0.797136
Updating model file...
Early stopping at: 8
----------------------------------------------
82_4

82_5

```
linear:

![lth82linear](lth82linear.png)

rbf:

![lth82rbf](lth82rbf.png)


CKA Linear plot (averaged over 5 exps): <br>
![vgg16_82_linear](vgg16_82_linear.png)

CKA RBF plot (averaged over 5 exps): <br>
![vgg16_82_rbf](vgg16_82_linear.png)

```
5_1
Train loss: 2.302323, Valid loss: 2.302320
Updating model file...
Early stopping at: 100
----------------------------------------------
Train loss: 2.302537, Valid loss: 2.302538
Updating model file...
Early stopping at: 31
----------------------------------------------
5_2
Train loss: 2.302562, Valid loss: 2.302561
Updating model file...
Early stopping at: 20
----------------------------------------------
5_3
Train loss: 2.302548, Valid loss: 2.302548
Updating model file...
Early stopping at: 20
----------------------------------------------
5_4
Train loss: 2.302559, Valid loss: 2.302560
Updating model file...
Early stopping at: 20
----------------------------------------------
5_5
Train loss: 2.302566, Valid loss: 2.302566
Updating model file...
Early stopping at: 20
----------------------------------------------
```

```
5_1
Test average loss: 2.3023, acc: 0.4400
----------
Test average loss: 2.3025, acc: 0.3485
Test time: 59.8367 s
----------
5_2
Test average loss: 2.3026, acc: 0.2301
----------
5_3
Test average loss: 2.3025, acc: 0.2715
----------
5_4
Test average loss: 2.3026, acc: 0.2537
----------
5_5
Test average loss: 2.3026, acc: 0.2124
----------
```
