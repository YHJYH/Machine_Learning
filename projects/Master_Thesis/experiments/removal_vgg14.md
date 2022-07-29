# vgg14 remove 1 conv 1 fc
acc. = []

reduced: (33638218-16701120)/33638218 * 100 = 50.35076%
remain: 49.6492%
```
Total params: 16,701,120
Trainable params: 16,701,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.89
Params size (MB): 63.71
Estimated Total Size (MB): 65.61
----------------------------------------------------------------
```

train_model12_x:
```
12_1

12_2

12_3

12_4

12_5

```

features12_x.pt:
```
12_1

12_2

12_3

12_4

12_5

```

CKA Linear avg:<br>
![vgg14_linear](vgg14_linear.png)

CKA RBF avg: <br>
![vgg14_rbf](vgg14_rbf.png)
