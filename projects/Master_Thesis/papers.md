## 2-stage LCN-HCN
title: A Too-Good-to-be-True Prior to Reduce Shortcut Reliance

author: Nikolay Dagaev, Bradley C. Love

year: 2021

dataset: CIFAR-10

summary: 介绍了用Low-capacity networks(LCN)来检测shortcuts，然后在high-capacity networks的训练过程中 将检测到的shortcuts downweights。<br>
介绍了local和global两种shortcuts的类型。前者是图片上固定位置的色块记号；后者类似于整张图片覆盖的Gaussian noise。<br>
local的又分为congruent和incongruent两种shortcuts，前者是training set和test set上同一label下的图片上的sc标记颜色位置相同，比如ship(red), plane(blue)；incongruent是同一label下的颜色不同，记住，同一个label下的颜色都是相同的，不同的是同一个label下training set和test set的颜色，比如 train: ship(red), plane(blue); test: ship(blue), plane(red).

