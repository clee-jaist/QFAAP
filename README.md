# Quality-focused Active Adversarial Policy for Safe Grasping in Human-Robot Interaction (QFAAP)

<p align="center">
<img src="The pipline of QFAAP.jpg" width="100%"/>
<br>

The QFAAP is designed to enhance the safety of vision-guided robot grasping in Human-Robot Interaction (HRI) scenarios. It introduces an Adversarial Quality Patch (AQP) and a Projected Quality Gradient Descent (PQGD) that adapts to human hand shapes from the perspective of benign adversarial attacks, which can be used to reduce the grasping priority of hands and nearby objects,  enabling robots to focus on safer, more appropriate grasping targets.


```

## Installation

This code was developed with Python 3.8 on Ubuntu 22.04.  Python requirements can installed by:

```bash
pip install -r requirements.txt
```

## Datasets

Currently, all datasets are supported.

### Cornell Grasping Dataset

1. Download the and extract [Cornell Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp). 

### OCID Grasping Dataset

1. Download and extract the [OCID Dataset](https://files.icg.tugraz.at/d/777515d0f6e74ed183c2/).

### Jacquard Grasping Dataset

1. Download and extract the [Jacquard Dataset](https://jacquard.liris.cnrs.fr/).


## Pre-trained Grasping Models

All pre-trained grasping models for GG-CNN, GG-CNN2, GR-Convnet, and others can be downloaded from [here](https://drive.google.com/drive/folders/1Yos_urL8h1A_kFrnu2y2xCD7uGeuTDGJ?usp=sharing).

## Pre-trained AQP

All AQP trained by different grasping models and datasets can be downloaded from [here](https://drive.google.com/drive/folders/1zT_yO4Zl8UAbIUHwbMYJ5S2iA2LES2PM?usp=drive_link).

## Pre-trained Hand Segmentation Models
All pre-trained Hand Segmentation models can be downloaded from [here](https://drive.google.com/drive/folders/1yd6nKRaRFG7-vIRMzp3JM11slkIvzUCr?usp=sharing).

## Training/Evaluation

Training for AQP is done by the `AQP_training.py`.  Training for Grasping model is done by the `train_grasping_network.py`. 
And the evaluation process is followed by the training.

## Predicting
1. The offline and realtime prediction of QFAAP is done by the `QFAAP_offline.py` and `QFAAP_realtime.py`.
2. For the deployment of real-time hand segmentation, please refer to this repository [https://github.com/Unibas3D/Upper-Limb-Segmentationp](https://github.com/Unibas3D/Upper-Limb-Segmentation)

<p align="center">
<img src="QFAAP prediction example.jpg" width="100%"/>
<br>


## Running on a Robot

Grasping with QFAAP is done by the `QFAAP_grasping.py` or `QFAAP_dynamic_grasping.py`. 

<p align="center">
<img src="Real Grasping example with QFAAP.jpg" width="100%"/>
<br>
