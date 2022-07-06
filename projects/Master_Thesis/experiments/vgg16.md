# Content
- [raw VGG16](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/experiments/vgg16.md#raw-vgg16)
- [CKA layer removal VGG16]()
- [random param=0 VGG16]()


## raw VGG16
```
Total params: 33,625,792
Trainable params: 33,625,792
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 2.17
Params size (MB): 128.27
Estimated Total Size (MB): 130.46
----------------------------------------------------------------
-data_type=cifar10 -model_type=vgg16 -learning_rate=0.001 -momentum=0.9 -num_epoch=50 -patience=2
```
train_model1_1:
```
----------------------------------------------
Train loss: 0.590474, Valid loss: 0.798762
Updating model file...
----------------------------------------------
...
----------------------------------------------
Train loss: 0.512675, Valid loss: 0.822146
Early stopping at: 26
----------------------------------------------
```
features1_1.pt
```
-mode=test -data_type=cifar10 -model_type=vgg16
----------------------------------------------
Test average loss: 0.8804, acc: 0.7098
```
CKA (Linear) plot:
![1_1linear](1_1linear.png)

CKA (RBF) plot:
![1_1rbf](1_1rbf.png)

## CKA layer removal VGG16
