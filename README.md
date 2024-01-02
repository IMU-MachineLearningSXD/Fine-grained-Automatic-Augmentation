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
We provide a CRNN model pre-trained on a cvl augmented dataset：[pre-train model](https://pan.baidu.com/s/1nAwIwjt0am1kVQQ1hj9Z_w)
### Run the following code
    python run_CRNN_optuna.py
