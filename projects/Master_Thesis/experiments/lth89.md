# LTH remove 89% parameters
acc. = [0.1683, 0.1732, 0.1746, 0.1608, 0.1528] mean:0.1659 std: 0.0082

time = [54.3240, 50.7863, 53.0205, 51.0810, 50.9574] mean:52.0334 std: 1.4029

remaining params: 3698843

LTH_features89_x
```
89_1
Test average loss: 2.3026, acc: 0.1683
Test time: 54.3240 s
----------
89_2
Test average loss: 2.3026, acc: 0.1732
Test time: 50.7863 s
----------
89_3
Test average loss: 2.3026, acc: 0.1746
Test time: 53.0205 s
----------
89_4
Test average loss: 2.3026, acc: 0.1608
Test time: 51.0810 s
----------
89_5
Test average loss: 2.3026, acc: 0.1528
Test time: 50.9574 s
----------
```

LTH_train_mode89_x
```
89_1

89_2

89_3

89_4

89_5

```

CKA Linear avg <br>
![vgg16_89_linear](vgg16_89_linear.png)

CKA RBF avg <br>
![vgg16_89_rbf](vgg16_89_rbf.png)

```
6_1
Test average loss: 2.3026, acc: 0.1683
----------
6_2
Test average loss: 2.3026, acc: 0.1732
----------
6_3
Test average loss: 2.3026, acc: 0.1746
----------
6_4
Test average loss: 2.3026, acc: 0.1608
----------
6_5
Test average loss: 2.3026, acc: 0.1528
----------
```

```
6_1
Train loss: 2.302584, Valid loss: 2.302584
Updating model file...
Early stopping at: 11
----------------------------------------------
6_2
Train loss: 2.302584, Valid loss: 2.302584
Updating model file...
Early stopping at: 2
----------------------------------------------
6_3
Train loss: 2.302584, Valid loss: 2.302584
Updating model file...
Early stopping at: 4
----------------------------------------------
6_4
Train loss: 2.302584, Valid loss: 2.302584
Updating model file...
Early stopping at: 12
----------------------------------------------
6_5
Train loss: 2.302584, Valid loss: 2.302584
Updating model file...
Early stopping at: 9
----------------------------------------------
```
