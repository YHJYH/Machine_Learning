source: [two-stage LCN-HCN approach](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers.md#2-stage-lcn-hcn)<br>
key words: shortcuts detection<br>
extension: <br>
- will generator in GAN also generate (congruent/incongruent) shortcuts?
    - do shallow Generator just do the job (detecing sc)?
    - if SG does, can we do sth like in LCN-HCN, to down-weight the items SG grabs in DG training?
- can we use MTL to find important features? such as bbox in object localization would contain shape features.
    - what feature does segmentation task grab?


implicit shortcut feature (do not use group annotation)

explicit shortcut feature (use group annotation)

for a single NN, the order of it to grab features is from easiest to hardest (it grabs shortcut feature first);

how the weights of individual sample loss affects?
- NN中参数的更新倾向于大loss的部分
    - 在LfF中，biased sample的weight小，则主要的loss来自于bias-conflict sample，NN的参数更新就会更倾向于bias-conflict sample（worst-group sample）
    - 在JTT中，则是upweight bias-conflict sample
    - 在LCN-HCN中同样是upweight bias-conflict sample/downweight bias sample

feature disentanglement method
- maybe cka
- kernel based method
- truly independent not just un-correlated



# lottery ticket hypothesis method
source: [lottery ticket](https://github.com/YHJYH/Machine_Learning/blob/main/projects/Master_Thesis/papers/lottery_ticket.md#the-lottery-ticket-hypothesis-finding-sparse-trainable-neural-networks)<br>

1. 训练一个full networks （超参数: lr，optimizer，# iterations，batch size，etc.）
2. 训练完毕，记录test accuracy
3. 检查除了output layer外每个layer的feature map的CKA（可以用不同的kernel试一试），画confusion matrix plot
4. 将高相似度数（这是一个超参数：0.5，0.7，etc.）feature map的后layer直接去掉
5. retrain，（尝试两种：使用初始的initialization(别忘了freeze)，或者reinitialize）
6. repeat step 2 3 4 5
<br>

上述实验在shallow network初始试一试(fully-connected);<br>
之后看情况在deep network上试一试(convolutional). [difference](https://medium.com/swlh/fully-connected-vs-convolutional-neural-networks-813ca7bc6ee5)

**some ideas from lab meeting**
control condition : lower bound

randonly remove compared with similarity remove (using different kernels)

20% 40% 60% 80%

receptive fields

bias = False? in conv

same architecture

try resnet

try to design a simple exp to support hypothesis

early stopping

set check points and use as sudo control conditions

check points similarity plot
