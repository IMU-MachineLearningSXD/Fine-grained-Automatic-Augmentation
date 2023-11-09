# Fine-grained-Automatic-Augmentation

## Overview
![image](https://github.com/IMU-MachineLearningSXD/Fine-grained-Automatic-Augmentation/blob/main/img/Framework.jpg)

### Abstract
This paper proposes a Fine-grained Automatic Augmentation (FgAA) method for handwritten characters. Specifically, the FgAA views each word sample as composed of multiple strokes and achieves data augmentation by performing fine-grained transformations on words. Each word is automatically segmented into multiple curves, and each stroke is fitted with a Bézier curve. On such basis, this paper defines the augmentation strategies related to the fine-grained transformation and uses Bayesian optimization to automatically learn the augmentation strategies. As a result, this method achieves handwriting sample automatic augmentation, which enables the recognition task to achieve optimal performance. Experiments on four handwriting data sets of different languages show that FgAA achieves the best augmentation effect for the handwritten recognition task.

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
