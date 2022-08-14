# CL VGG12 reinit
acc = [] mean: std:

time = [] mean: std:

model size = 

reinit_CL_features12_x
```

```

hyperparams.
```
Namespace(mode='train', data_type='cifar10', model_type='vgg16_N', seed=79, learning_rate=0.001, momentum=0.9, prune_percentage=70, num_epoch=50, patience=5)
```

weight boundary
```
"OrderedDict([(0, defaultdict(<class 'list'>, {
'conv1.weight': [0.054233514, 0.054499833658337556, 0.054529425], 
'conv2.weight': [0.030780395, 0.030780490860342978, 0.03078135], 
'conv3.weight': [0.024200888, 0.024200990609824658, 0.024201002], 
#'conv4.weight': [0.020512529, 0.020512983202934265, 0.020513438], 
'conv5.weight': [0.016116742, 0.016116873733699317, 0.01611693], 
#'conv6.weight': [0.01441928, 0.01441928669810295, 0.014419347], 
#'conv7.weight': [0.014482062, 0.014482069760560988, 0.014482137], 
'conv8.weight': [0.011718982, 0.011718985252082348, 0.011719004], 
#'conv9.weight': [0.01022359, 0.010223591700196266, 0.010223594], 
'conv10.weight': [0.00994256, 0.009942568838596344, 0.009942577], 
'conv11.weight': [0.0104539115, 0.010453915223479271, 0.010453919], 
'conv12.weight': [0.010535635, 0.010535642504692078, 0.01053565], 
'conv13.weight': [0.010355848, 0.010355865582823753, 0.010355883], 
'fc1.weight': [0.014829458, 0.01482951836660504, 0.014829544], 
'fc2.weight': [0.010924308, 0.010924309492111206, 0.010924311], 
'fc3.weight': [0.018010799, 0.018011713586747646, 0.018013848]}))])"
```

reinit_CL_train_model12_x
```

```

linear:

![recl12linear](recl12linear.png)

rbf:

![recl12rbf](recl12rbf.png)
