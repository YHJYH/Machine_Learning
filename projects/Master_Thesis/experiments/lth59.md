# 59% vgg16
acc= [0.7479, 0.7536, 0.7572, 0.7496, 0.7615] mean:0.7540 std:0.0050

time= [55.8245, 52.2588, 52.8009, 54.4774, 54.2379] mean:53.9199 std:1.2689 

remained: 13786577

LTH_features59_x
```
59_1
Test average loss: 1.3040, acc: 0.7479
Test time: 55.8245 s
----------
59_2
Test average loss: 1.3266, acc: 0.7536
Test time: 52.2588 s
----------
59_3
Test average loss: 1.3093, acc: 0.7572
Test time: 52.8009 s
----------
59_4
Test average loss: 1.5011, acc: 0.7496
Test time: 54.4774 s
----------
59_5
Test average loss: 1.3277, acc: 0.7615
Test time: 54.2379 s
----------
```

LTH_train_model59_x
```
59_1
Train loss: 0.258414, Valid loss: 0.426657
Updating model file...
Early stopping at: 8
----------------------------------------------
59_2
Train loss: 0.165709, Valid loss: 0.461849
Updating model file...
Early stopping at: 10
----------------------------------------------
59_3
Train loss: 0.266855, Valid loss: 0.419371
Updating model file...
Early stopping at: 8
----------------------------------------------
59_4
Train loss: 0.157163, Valid loss: 0.443077
Updating model file...
Early stopping at: 10
----------------------------------------------
59_5
Train loss: 0.216765, Valid loss: 0.437868
Updating model file...
Early stopping at: 9
----------------------------------------------
```

linear:

![lth59linear](lth59linear.png)

rbf:

![lth59rbf](lth59rbf.png)
