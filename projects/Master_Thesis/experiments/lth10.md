# LTH remove 10% params on VGG16

acc = [, 0.7638, 0.7638, 0.7366, 0.7403, 0.7245] mean:0.7458 std:0.0156 

remaining params.: 30263206 <br>
removed: 33625792 - 30263206 = 3362586 = 10%

LTH_train_model1_x
```
1_1
Train loss: 0.421612, Valid loss: 0.745001
Updating model file...
Early stopping at: 16
----------------------------------------------
1_2
Train loss: 0.493147, Valid loss: 0.730518
Updating model file...
Early stopping at: 15
----------------------------------------------
1_3
Train loss: 0.519747, Valid loss: 0.758077
Updating model file...
Early stopping at: 15
----------------------------------------------
1_4
Train loss: 0.423186, Valid loss: 0.774376
Updating model file...
Early stopping at: 16
----------------------------------------------
1_5
Train loss: 0.413660, Valid loss: 0.717727
Updating model file...
Early stopping at: 16
----------------------------------------------
```

LTH_features1_x.pt
```
1_1
Test average loss: 1.3225, acc: 0.7355
----------
1_2
Test average loss: 1.5377, acc: 0.7638
----------
1_3
Test average loss: 1.1445, acc: 0.7366
----------
1_4
Test average loss: 1.3590, acc: 0.7403
----------
1_5
Test average loss: 1.3289, acc: 0.7245
----------
```

CKA Linear plot (averaged over 5 exps):<br>
![vgg16_10_linear](vgg16_10_linear.png)

CKA RBF plot (averaged over 5 exps): <br>
![vgg16_10_rbf](vgg16_10_rbf.png)
  

