# content
shortcut learning
- [2-stage LCN-HCN](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-lcn-hcn)
- [feature disentanglement](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#feature-disentanglement-in-covid-19-cxr-image-classification)

text summarization
- [PlanSum](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#plansum)


## 2-stage LCN-HCN
title: A Too-Good-to-be-True Prior to Reduce Shortcut Reliance

author: Nikolay Dagaev, Bradley C. Love

year: 2021

dataset: CIFAR-10

summary: 介绍了用Low-capacity networks(LCN)来检测shortcuts，然后在high-capacity networks的训练过程中 将检测到的shortcuts downweights。<br>
介绍了local和global两种shortcuts的类型。前者是图片上固定位置的色块记号；后者类似于整张图片覆盖的Gaussian noise。<br>
local的又分为congruent和incongruent两种shortcuts，前者是training set和test set上同一label下的图片上的sc标记颜色位置相同，比如ship(red), plane(blue)；incongruent是同一label下的颜色不同，记住，同一个label下的颜色都是相同的，不同的是同一个label下training set和test set的颜色，比如 train: ship(red), plane(blue); test: ship(blue), plane(red).

question: how to downweights?

## feature disentanglement in COVID-19 CXR image classification
title: Deep learning models for COVID-19 chest x-ray classification: Preventing shortcut learning using feature disentanglement

author: Caleb Robinson, Juan M. Lavista Ferres
 
year: 2021

summary: 这篇文章借用了一种“feature 反纠缠”的方法，这种方法的特色是，我们提前知道shortcut feature是哪个了，我们通过最大化shortcut feature的loss来使其不被我们的networks care，即把shortcut feature从所有feature里disentangle出来。<br>
同时feature entanglement也是建立在transfer learning的基础上，即一部分参数是训练好然后fronzen的。<br>
具体来说主要分为两个parts, 第一：我们通过feature extractor g(x, \theta)=z (这一步类似于kernel)得到提取的feature z，但是这个z不直接参与estimator f(z, \phi)，而是再经过一个feature extractor f_e(z, \phi_{e})得到z'。再将这个z'喂给两个classifiers，一个classifier是用来classifyshortcut feature的（recall上面提过，sc我们已经提前知道是哪个了），另一个则是其他features。第二就是使用的loss function了，是一个min-max的过程（和GAN有点像？），如下所示。
![fdeq2](./pics/fdeq2.PNG)<br>
![fdeq34](./pics/fdeq34.PNG)<br>
网络结构如下所示：
![fdnet](./pics/fdnet.PNG)<br>

question: 按照0028里学的，如果要彻底disentangle，那\phi_{d}不应该直接等于0吗，表示feature z'和domain label y^{d}没有correlation。


## PlanSum
title: Unsupervised Opinion Summarization with Content Planning

author: Reinald Kim Amplayo, Stefanos Angelidis, Mirella Lapata

year: 2021

dataset: Rotten Tomatoes, Yelp, Amazon

summary:
