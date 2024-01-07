# AAI PROJECTl BY DRO
## purpose
Train a model for handwritten digits recognition. 

## Data
Mutated handwritten digits, in the form of 10*28*28 npy files. There are 60000 'npz' files with labels in training set, 100 'npz' files with lables in val set and 9900 'npz' files without labels in test set..The data is filled with misdirection. The data is derived from variations of the data in the MNIST database, where each file has a 10-dimensional structure. In files labeled 'i' (i in 0-9), the i-th dimension should contain the corresponding handwritten digit 'i', while the other dimensions are blank. However, there are intentionally mislabeled files; under label 'i', instead of the i-th dimension containing digit 'i', it contains a different digit, thus introducing errors.

## Models

### Models Selection
According to [3], convolutional neural network (CNN) shows good performence in recognition on words. So we choose CNN as base model and for reduce the effect of the peripheral region, padding is set as 0. Comparing between [1] and [2], we finally choose [1]’s strategy for this project. The two papers mention to two regions: Invariant Risk Minimization (IRM) and Distributionally Robust Optimization (DRO). IRM focus on the datasets with different backgrounds or unstable features. DRO focus on minimizing the risk of data in worst situation. Specifically, in [1], this paper tries to learn stable features in the target task by learning unstable features from source task. So the strategy of [1] is more suitable for this task leaning stable feature of data. 

According to [1] and [3], we also have two models: CNN and MLP. The real structures of two model are showed in following code, the hidden_dim is 300 in default. And in following report, **the two models will be regarded as one model**.
```
class CNN(nn.Module)
    def __init__(self, input_channel, hidden_dim):
    super(CnnM, self).__init__()
    self.conv1 = nn.Conv2d(input_channel, 32, 3, 1, padding=0)
    self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=0)
    self.dropout1 = nn.Dropout(0.1)
    self.fc1 = nn.Linear(64*12*12, hidden_dim)
    ...
```
```
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(input_dim, output_dim)
    ...
```
###	Strategies selection
Distributionally Robust Optimization (DRO) is a mathematical optimization framework that deals with decision-making under uncertainty. Unlike traditional optimization methods which rely on a single, precise probabilistic model of the underlying data distribution, DRO takes into account a set of possible distributions that are close to or consistent with the available empirical data.

In essence, DRO aims to find solutions that perform well across this entire ambiguity set of distributions rather than just optimizing for the best-case scenario based on a point estimate. This approach enhances the robustness of the solution against potential inaccuracies in the estimated probability distribution, thereby providing a safeguard against model misspecification and changes in the data-generating process. 
Formally, the goal in DRO is to minimize the worst-case expected loss over an uncertainty set of probability distributions. This approach enhances the robustness of the solution against potential inaccuracies in the estimated probability distribution, thereby providing a safeguard against model misspecification and changes in the data-generating process. But the biggest problem is finding how to cluster data. **A simple strategy is enhancing learning on wrong predicted data**. But it generally not performs very good. **So [1] proposes a method for clustering called TOFU**. And there are **two ways for training on classified data: basing on trained model which learning unstable feature and basing on random model**. 

And there is no doubt, we want to get a model recognizing the digits with dimensional interference suppression. So there is a natural thought is to proactively exclude interference in data processing. But TAs (Teaching Assistants) refute this idea. Final, we choose training two models following the natural thought as base models to show we get a model which could recognize digit with elimination of dimensional interference. One model’s data preprocessing is **reducing the dimensions into one dimension by adding all dimensions into one dimension**. The input data for model is 28x28 tensor. The other model’s preprocessing is **shuffling the first dimension**. The input data for model is 10x28x28 tensor. We predict the two models with similar performance.
In order to validate that the effective of DRO and [1]'s strategy is indeed more effective than conventional DRO and, we train two models basing in simple DRO strategy and two models basing on TOFU strategy.

So, there are 6 models and one fake model which learning unstable features. 

|Models|Strategies|Data Preprocess|Objective|
|:---|:----|:----|:----|
|Base Model 1| Normal| Dimension Reduction| Get stable features|
|Base Model 2| Normal| First Dimension Shuffle| Get stable features|
|Fake Model 1| Normal| No | Get unstable features|
|DRO Model 1| simpel DRO; init with fake model| No | Get stable features|
|DRO Model 2| simpel DRO; init with random model| No | Get stable features|
|TOFU Model 1| TOFU; init with fake model| No | Get stable features|
|TOFU Model 2| TOFU; init with random model| No | Get stable features|

## Experiment 

### Base models and fake model
Firstly, we train three simple models: Base Model 1,2 and Fake Model in 60 epoches. In our design, two base models should recognize digits without the interference of unstable features by data preproceessing.

<img src="./pics/BASE%20MODELS%20LOSS.png" width="45%" style="float: left; margin-right: 10px;">
<img src="./pics/BASE%20MODELS%20ACCURACY.png" width="45%" style="float: left;">
<br style="clear: both;">

|models|accuracy on val dataset|
|---|---|
|Base Model 1|0.77|
|Base Model 2|0.76|
|Fake Model |0.15|

By the two pictures and table, we could see Fake Model has hightest accuracy in training set and lowest accuracy on val set which shows it overfits on straining set and learns unstable features. The good performents of two base model shows the unstable feature is the first dimension relation lables.

### Simple DRO Strategy
Next, we trains the models with simple DRO strategy in 400 epoches . In DRO, it focus on the worst loss among all classes of data. So, the loss recorded and accuracy recorded are the worst loss and accuracy of one class of data in each epoch. *This is same on TOFU Model 1 and 2*.

<img src="./pics/BASE DRO MODELS&apos; LOSS.png" width="45%" style="float: left; margin-right: 10px;">
<img src="./pics/BASE DRO MODELS&apos; ACCURACY.png" width="45%" style="float: left;">
<br style="clear: both;">

|models|accuracy on val dataset|
|---|---|
|DRO Model 1|0.36|
|DRO Model 2|0.35|

The pictures and table shows the simple DRO strategy just make small progress on learning stable features than Fake Model. Overall, it still mainly learns unstable feature.

### TOFU Strategy
By TOFU, we firstly get all classes of data: 20 data sets in the order: data set of predicted correct data by Fake Model on labe 1 and data set predicted wrong data by Fake Model on labe 1 ... Then enchan the representation difference on differant classes and similarity on same class. Thirdly, we cluster these data into 20 clusters: each label with two clusters by K-means algorithm provided by scikit-learn package. So we get all clusters of data. Then train with DRO strategy.

<img src="./pics/COMPARE ALL MODELS&apos; LOSS.png" width="45%" style="float: left; margin-right: 10px;">
<img src="./pics/COMPARE ALL MODELS&apos; ACCURACY.png" width="45%" style="float: left;">
<br style="clear: both;">

|models|accuracy on val dataset|
|---|---|
|TOFU Model 1|0.63|
|TOFU Model 2|0.75|

The pictures and tables, shows that we get TOFU models which recognizing the digits with dimensional interference suppression. And TOFU Model 2 perform better than TOFU Model 1. We should trains final model inited by random insteal of the model learning unstable feature.

## Member Contribution
Overall, all members contribute equally. In terms of specific tasks, Li Hao is responsible for determining the overall strategy and conducting experiments, Jiang Tao takes charge of the actual implementation of the code, and Yuan Fei is in charge of writing reports.

## reference
[1] Bao, Yujia, Shiyu Chang, and Regina Barzilay. "Learning stable classifiers by transferring unstable features." International Conference on Machine Learning. PMLR, 2022.

[2] Arjovsky, Martin, et al. "Invariant risk minimization." arXiv preprint arXiv:1907.02893 (2019).

[3] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

## Appendix
|sets|detail|
|---|---|
|optimizer|Adam with 0.001 as wight decay|
|learning rate| 0.001|
|Activate function of CNN|relu|
|activate function of MLP|log_softmax|
|drop out rate|0.1|