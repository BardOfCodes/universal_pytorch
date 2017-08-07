# Universal Adverserial Perturbation

This code contains implementation of the following papers in pytorch.

1) [DeepFool](https://arxiv.org/abs/1511.04599) : A batch deepfool version based on [this](https://github.com/LTS4/DeepFool) github repository.
2) [Universal Adverserial Perturbation](https://arxiv.org/abs/1707.05572): Based on [this](https://github.com/LTS4/universal) github repository
3) [Fast Feature Fool](https://arxiv.org/abs/1610.08401): Based on [this](https://github.com/utsavgarg/fast-feature-fool) github repository  

## Introduction

Universal Adverserial Perturbations are single perturbations of very small norm which are able to successfully fool networks. It is quite astonishing as it shows that a single directional shift in the image space can cause such huge errors.


## Usage

This repository provides two types of Universal Adverserial Perturbations(UAPs), Data dependent(based on Universal Adverserial Perturbations) , and Data Independent(based on fast feature fool).

#### For finding UAP for a network :
```
python find_uap model im_path im_list [options]
```
Where,
* `model`: is the network you want to find UAP for(currently 'vgg16' and 'resnet18', easily extendable)
* `im_path`: is the path to folder containing images to train perturbation on.
* `im_list` : is the path to file containing list of all images to train perturbation on.

For more information, use `python find_uap.py --h`

#### For Evaluating a perturbation's performance:
```
test_uap.py <model> <im_path> <im_list> <perturbation_path> [options]
```
Where,
* `model`: is the network you want to evaluate UAP for(currently 'vgg16' and 'resnet18', easily extendable)
* `im_path`: is the path to folder containing images to evaluate perturbation on.
* `im_list` : is the path to file containing list of all images to evaluate perturbation on.

For more information, use `python test_uap.py --h`

## Important information

As models used in works Universal Adverserial Perturbation and Fast-feature-fool were on tensorflow, these model took input of range 0-255. However, in Pytorch, the models have a different input range. The input is processed in the following fashion:

* First the input is scaled 0-1 rather than 0-255
* Each channel is normalized using the channel mean and channel standard deviations.

Using this information, we can find out the input range:<br>
```
(Lowest = min(-Channel_mean/Channel_std),Highest = max((1-Channel_mean)/ Channel_std))
```
From this we calculate the input range to be 4.765. The norm limits for the perturbations have been shifted accordingly.

## Performance of UAPs:

Data Independent UAPs for VGG16 and ResNet18 have been provided. The performance is as follows:


| **Model** | **Training Performance** | **Evaluation Performance** |
|---|---|---|
|**VGG 16** | 92.39 % | 92.726 % |
|**ResNet 18** | 88.88 % | 88.584 % |

## Future enhancements:

* The batch deepfool has to be made more efficient. Mechanism such that in each iteration, forward pass is performed only for images in batch which have not yet been fooled has to be added.
* More results for the various nets (ResNet152, VGG19, etc.)
* Provide UAPs thorugh [foolbox](https://github.com/bethgelab/foolbox).



## Acknowlegdement

I would like to thank :

* [Mopuri Reddy](https://github.com/mopurikreddy) ,[Utsav Garg](https://github.com/utsavgarg)and  [EPFL LTS4](https://github.com/LTS4) for finding out this interesting phenomena.

Also, a big thanks to [Video Analytics Lab](http://val.serc.iisc.ernet.in/valweb/).

