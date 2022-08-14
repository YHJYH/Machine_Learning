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
'conv1.weight': [0.11045775, 0.11061844676733017, 0.1106631], 
'conv2.weight': [0.053014725, 0.0530148085206747, 0.053015556], 
'conv3.weight': [0.043628804, 0.04363214783370494, 0.043633174], 
#'conv4.weight': [0.035792366, 0.0357929952442646, 0.035793], 
'conv5.weight': [0.030612448, 0.03061252515763044, 0.030612558], 
#'conv6.weight': [0.025284782, 0.02528479248285293, 0.025284968], 
#'conv7.weight': [0.025266424, 0.025266427919268607, 0.02526651], 
'conv8.weight': [0.021347042, 0.0213471470400691, 0.021347148], 
#'conv9.weight': [0.01785763, 0.017857633531093597, 0.017857637], 
'conv10.weight': [0.01819589, 0.018195901066064835, 0.018195907], 
'conv11.weight': [0.018038178, 0.01803820114582777, 0.018038215], 
'conv12.weight': [0.017992726, 0.017992740496993065, 0.017992776], 
'conv13.weight': [0.017977737, 0.017977749928832054, 0.017977763], 
'fc1.weight': [0.025371691, 0.025371709465980528, 0.025371717], 
'fc2.weight': [0.01896125, 0.018961253575980663, 0.018961256], 
'fc3.weight': [0.031973224, 0.03197539113461971, 0.03197546]}))])"
```

reinit_CL_train_model12_x
```

```

linear:

![recl12linear](recl12linear.png)

rbf:

![recl12rbf](recl12rbf.png)
