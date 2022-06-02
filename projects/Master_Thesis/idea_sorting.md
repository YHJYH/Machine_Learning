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
