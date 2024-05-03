# Fine-grained-Automatic-Augmentation

## Overview
![image](https://github.com/IMU-MachineLearningSXD/Fine-grained-Automatic-Augmentation/blob/main/img/Framework.jpg)

## Requirements
- python 3.6+
- optuna
- opencv-python
- numpy

## Usage
We provide runnable code on the cvl dataset.
Please first download the cvl data set and record the training set and test set addresses.
Change the following configuration in the./lib/config/OWN_config.yaml file.
- DATASET-ROOT
- DATASET-JSON_FILE
- TRAIN-BEGIN_EPOCH
- TRAIN-END_EPOCH
- RESUME-FILE(Load the pre-trained model)
We provide a CRNN model pre-trained on a cvl dataset：./checkpoint_60_loss_0.0009.pth
### Run the following code
    python run_CRNN_optuna.py --cfg lib/config/OWN_config.yaml

## Policy
![image](https://github.com/IMU-MachineLearningSXD/Fine-grained-Automatic-Augmentation/blob/main/img/Policy.jpg)

## Case Study
![image](https://github.com/IMU-MachineLearningSXD/Fine-grained-Automatic-Augmentation/blob/main/img/Case_study.jpg)

