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

## Additional experiments


### Performance of CRNN without augmentation

<table>
  <tr>
    <th rowspan="2">Dataset</th>
    <th colspan="2">No augmentation</th>
  </tr>
  <tr>
    <td>WER (%)</td>
    <td>CER (%)</td>
  </tr>
  <tr>
    <td>CVL</td>
    <td>60.00±0.99</td>
    <td>41.75±0.86</td>
  </tr>
  <tr>
    <td>IFN/ENIT</td>
    <td>59.75±1.59</td>
    <td>27.69±0.84</td>
  </tr>
  <tr>
    <td>HKR</td>
    <td>84.66±0.53</td>
    <td>56.23±0.52</td>
  </tr>
  <tr>
    <td>Mongolian</td>
    <td>93.56±0.11</td>
    <td>58.80±0.26</td>
  </tr>
  <tr>
    <td>IAM</td>
    <td>45.14±0.09</td>
    <td>34.55±0.13</td>
  </tr>
</table>


### The performance of FgAA compared with baselines on CVL dataset.

<table>
  
<th rowspan="4">Method</th>
<th colspan="4">CVL</th>
<tr>
<th colspan="4">Augmentation times / # Augmented samples</th>
</tr>
<tr>
<th colspan="2">10 / 133.6k</th>
<th colspan="2">15 / 194.4k</th>
</tr>
<tr>
<th>WER (%)↓</th>
<th>CER (%)↓</th>
<th>WER (%)↓</th>
<th>CER (%)↓</th>
</tr>
  <tr>
    <td>Affine</td>
    <td>20.93±0.13</td>
    <td>12.49±0.07</td>
    <td>20.10±0.07</td>
    <td>11.67±0.24</td>
  </tr>
  <tr>
    <td>SIA</td>
    <td>19.31±0.28</td>
    <td>11.25±0.15</td>
    <td>19.01±0.09</td>
    <td>10.91±0.13</td>
  </tr>
  <tr>
    <td>TPS</td>
    <td>21.53±0.32</td>
    <td>13.01±0.33</td>
    <td>20.89±0.02</td>
    <td>12.36±0.02</td>
  </tr>
  <tr>
    <td>ScrabbleGAN</td>
    <td>36.00±0.09</td>
    <td>23.06±0.27</td>
    <td>34.69±0.12</td>
    <td>21.89±0.29</td>
  </tr>
  <tr>
    <td>VAT</td>
    <td>29.35±0.23</td>
    <td>17.87±0.30</td>
    <td>28.36±0.17</td>
    <td>16.49±0.08</td>
  </tr>
  <tr>
    <td>L2A</td>
    <td>20.15±0.21</td>
    <td>11.72±0.29</td>
    <td>19.35±0.30</td>
    <td>11.09±0.20</td>
  </tr>
  <tr>
    <td>AA</td>
    <td>23.68±0.04</td>
    <td>14.56±0.18</td>
    <td>21.86±0.19</td>
    <td>13.05±0.11</td>
  </tr>
  <tr>
    <td>Fast AA [15]</td>
    <td>26.27±0.05</td>
    <td>16.28±0.04</td>
    <td>27.26±0.22</td>
    <td>16.89±0.07</td>
  </tr>
  <tr>
    <td>TA</td>
    <td>26.99±0.04</td>
    <td>16.81±0.17</td>
    <td>24.55±0.34</td>
    <td>15.09±0.11</td>
  </tr>
  <tr>
    <td><b>FgAA (ours)</b></td>
    <td><b>19.03±0.03</b></td>
    <td><b>10.86±0.13</b></td>
    <td><b>17.84±0.21</b></td>
    <td><b>10.19±0.12</b></td>
  </tr>
</table>

### The performance of FgAA compared with baselines on IFN/ENIT and HKR datasets.
<table>
  
<th rowspan="4">Method</th>
<th colspan="2">IFN/ENIT</th>
<th colspan="2">HKR</th>
<tr>
<th colspan="4">Augmentation times / # Augmented samples</th>
</tr>
<tr>
<th colspan="2">10 / 71.9k</th>
<th colspan="2">10 / 67.9k</th>
</tr>
<tr>
<th>WER (%)↓</th>
<th>CER (%)↓</th>
<th>WER (%)↓</th>
<th>CER (%)↓</th>
</tr>
  
  <tr>
    <td>Affine</td>
    <td>32.04±0.22</td>
    <td>13.22±0.20</td>
    <td>32.94±0.16</td>
    <td>20.19±0.20</td>
  </tr>
   <tr>
    <td>SIA</td>
    <td>27.56±0.24</td>
    <td>11.58±0.13</td>
    <td>29.37±0.08</td>
    <td>18.10±0.01</td>
  </tr>
  <tr>
    <td>TPS</td>
    <td>32.89±0.16</td>
    <td>13.56±0.03</td>
    <td>34.47±0.09</td>
    <td>20.77±0.40</td>
  </tr>
  <tr>
    <td>ScrabbleGAN</td>
    <td>44.47±0.37</td>
    <td>21.98±0.35</td>
    <td>66.71±0.28</td>
    <td>37.01±0.21</td>
  </tr>
  <tr>
    <td>VAT</td>
    <td>38.75±0.19</td>
    <td>16.68±0.25</td>
    <td>59.23±0.33</td>
    <td>35.01±0.18</td>
  </tr>
  <tr>
    <td>L2A</td>
    <td>28.38±0.23</td>
    <td>11.76±0.16</td>
    <td>33.06±0.30</td>
    <td>19.50±0.27</td>
  </tr>
  <tr>
    <td>AA</td>
    <td>35.41±0.09</td>
    <td>15.37±0.23</td>
    <td>51.41±0.18</td>
    <td>30.66±0.19</td>
  </tr>
  <tr>
    <td>Fast AA</td>
    <td>34.49±0.50</td>
    <td>14.33±0.37</td>
    <td>48.29±0.50</td>
    <td>29.13±0.20</td>
  </tr>
  <tr>
    <td>TA</td>
    <td>35.27±0.15</td>
    <td>15.01±0.22</td>
    <td>56.03±0.59</td>
    <td>33.12±0.52</td>
  </tr>
  <tr>
    <td><b>FgAA (ours)</b></td>
    <td><b>26.99±0.13</b></td>
    <td><b>11.53±0.04</b></td>
    <td><b>28.44±0.08</b></td>
    <td><b>17.64±0.08</b></td>
  </tr>
</table>

### The performance of FgAA compared with baselines on Mongolian and IAM datasets
<table>
  <th rowspan="4">Method</th>
  <th colspan="2">Mongolian</th>
  <th colspan="2">IAM</th>
  <tr>
  <th colspan="4">Augmentation times / # Augmented samples</th>
  </tr>
  <tr>
  <th colspan="2">10 / 110.0k</th>
  <th colspan="2">10 /592.2k</th>
  </tr>
  <tr>
  <th>WER (%)↓</th>
  <th>CER (%)↓</th>
  <th>WER (%)↓</th>
  <th>CER (%)↓</th>
  </tr>
  
  <tr>
    <td>Affine</td>
    <td>75.16±0.17</td>
    <td>40.75±0.30</td>
    <td>35.49±0.07</td>
    <td>26.17±0.05</td>
  </tr>
  <tr>
    <td>SIA</td>
    <td>72.87±0.21</td>
    <td>38.56±0.17</td>
    <td>34.36±0.11</td>
    <td>25.27±0.20</td>
  </tr>
  <tr>
    <td>TPS</td>
    <td>77.26±0.18</td>
    <td>42.36±0.22</td>
    <td>35.98±0.11</td>
    <td>26.23±0.21</td>
  </tr>
  <tr>
    <td>ScrabbleGAN</td>
    <td>88.08±0.07</td>
    <td>50.89±0.19</td>
    <td>43.86±0.29</td>
    <td>33.27±0.19</td>
  </tr>
  <tr>
    <td>VAT</td>
    <td>85.67±0.23</td>
    <td>49.09±0.15</td>
    <td>40.03±0.08</td>
    <td>30.09±0.24</td>
  </tr>
  <tr>
    <td>L2A</td>
    <td>73.65±0.31</td>
    <td>39.03±0.13</td>
    <td>34.82±0.23</td>
    <td>25.61±0.17</td>
  </tr>
  <tr>
    <td>AA</td>
    <td>77.81±0.56</td>
    <td>42.65±0.51</td>
    <td>35.90±0.05</td>
    <td>26.19±0.04</td>
  </tr>
  <tr>
    <td>Fast AA</td>
    <td>83.57±0.41</td>
    <td>47.70±0.43</td>
    <td>37.44±0.11</td>
    <td>28.14±0.05</td>
  </tr>
  <tr>
    <td>TA</td>
    <td>84.16±0.29</td>
    <td>48.22±0.39</td>
    <td>36.67±0.06</td>
    <td>27.28±0.12</td>
  </tr>
  <tr>
    <td><b>FgAA (ours)</b></td>
    <td><b>70.98±0.22</b></td>
    <td><b>37.65±0.11</b></td>
    <td><b>33.89±0.09</b></td>
    <td><b>24.92±0.18</b></td>
  </tr>
</table>

### Experimental results of FgAA's domain transferability. The number denotes the Word Accuracy Rate degradation when the policy is transferred to the target dataset.
![image](https://github.com/IMU-MachineLearningSXD/Fine-grained-Automatic-Augmentation/blob/main/img/Domain_transfer.jpg)
