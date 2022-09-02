# CKA OOD
acc, time, similarity plot

- vgg16 0%
```
ood_features1_x = {
avg_time: 10.4931
std_time: 0.0878
avg_acc: 0.5927
std_acc: 0.0073
}
```

```
Test average loss: 3.5005, acc: 0.5935
Test time: 10.6502 s
----------
Test average loss: 3.1055, acc: 0.5975
Test time: 10.4453 s
----------
Test average loss: 3.3149, acc: 0.5790
Test time: 10.4086 s
----------
Test average loss: 3.1248, acc: 0.5935
Test time: 10.4351 s
----------
Test average loss: 3.3147, acc: 0.6000
Test time: 10.5264 s
----------
```
linear:

![vgg16_ood_linear](vgg16_ood_linear.png)

rbf:

![vgg16_ood_rbf](vgg16_ood_rbf.png)

- vgg12 10% 
```
ood_features2_x = {
avg_time: 7.2571
std_time: 0.1758
avg_acc: 0.5637
std_acc: 0.0052
}
```

```
Test average loss: 5.0388, acc: 0.5690
Test time: 7.3186 s
----------
Test average loss: 4.8026, acc: 0.5670
Test time: 7.0941 s
----------
Test average loss: 3.9518, acc: 0.5550
Test time: 7.2570 s
----------
Test average loss: 5.1057, acc: 0.5670
Test time: 7.0641 s
----------
Test average loss: 4.7684, acc: 0.5605
Test time: 7.5519 s
----------
```
linear:<br>
![vgg12_ood_linear](vgg12_ood_linear.png)<br>
rbf:<br>
![vgg12_ood_rbf](vgg12_ood_rbf.png)<br>

- vgg11 18%
```
ood_features3_x = {
avg_time: 8.1673
std_time: 1.2345
avg_acc: 0.5592
std_acc: 0.0055
}
```

```
Test average loss: 4.8428, acc: 0.5550
Test time: 7.0643 s
----------
Test average loss: 5.3311, acc: 0.5635
Test time: 8.9313 s
----------
Test average loss: 4.4372, acc: 0.5595
Test time: 10.1848 s
----------
Test average loss: 4.5807, acc: 0.5515
Test time: 7.7423 s
----------
Test average loss: 4.9220, acc: 0.5665
Test time: 6.9139 s
----------
```

linear:<br>
![vgg11oodlinear](vgg11oodlinear.png)<br>
rbf:<br>
![vgg11oodrbf](vgg11oodrbf.png)<br>

- vgg10 18%
```
ood_features4_x = {
avg_time: 
std_time: 
avg_acc: 
std_acc: 
}
```

```
Test average loss: 4.6810, acc: 0.5825
Test time: 7.2148 s
----------
Test average loss: 4.5164, acc: 0.6015
Test time: 8.8070 s
----------

```
linear:

rbf:

- vgg9 18%

linear:

rbf:

- vgg6a 20%

linear:

rbf:

- vgg8 25%

linear:

rbf:

- vgg7 32%

linear:

rbf:

- vgg15 49%

linear:

rbf:

- vgg14 50%

linear:

rbf:

- vgg13 57%

linear:

rbf:

- vgg12a 59%

linear:

rbf:

- vgg11a 66%

linear:

rbf:

- vgg8a 75%

linear:

rbf:

- vgg6 82%

linear:

rbf:

- vgg5 89%

OOD_features9_1
```
Test average loss: 13.6125, acc: 0.1010
Test time: 3.8723 s
----------
```

linear:

rbf:
