# ENPH 353 Competition

Project Team: Hudson Nock (Reinforcement learning), Brianna Gopaul (Computer vision)

ENPH 353 is a project course at UBC where students create programs
designed to drive a simulated car through a race track while read signs along the way. Our team used reinforcement learning (RL)
to drive and a various filters along with a CNN to read the signs.

Above view of the robot, note that only two wheels had to be inside the road at any given time. <br />

[https://github.com/user-attachments/assets/7e9fa4be-4a7a-4b7a-b543-e153ebce588a](https://github.com/user-attachments/assets/38bc92b5-8e7c-44f7-893e-8a27f093a26c)

Camera view of the robot (what we have access to) <br />

https://github.com/user-attachments/assets/0dcdb58a-69a8-45ca-b8b4-ef9bd14a24a3

This repo contains a brief summary of our implementation, for more information refer to our report: <br /> [ENPH353 Final Report.pdf](https://github.com/user-attachments/files/16322865/ENPH353.Final.Report.pdf)

### Problem outline

The competition is aimed towards maximizing the number of points achieved. Points are gathered by reading signs scattered throughout the map, with deductions given when more than 2 wheels are off the road or in the case of a collision or respawn. Although being faster doesn't grant you more points, the time to completion does play a role in tie breakers.

### Driving Implementation

While driving, the robot is only given a camera view and has to determine its actions from that. It can then output a desired speed and steering angle.
We decided to drive using a soft actor critic (SAC, https://arxiv.org/abs/1801.01290) algorithm. The use of SAC and the reward function used was inspired by TMRL (https://github.com/trackmania-rl/tmrl).

To set up the environment for training, we determined the position of the wheels then compared them against a bitmap of the allowed path, if two or more wheels were off the road it would count as a respawn. The state space was composed of a gray scaled camera view filtered with a 2x2 average mask applied, as well as the speed set by the robot. Although the speed set does not necessarily match the actual speed (which could lead to an environment that isn't a MDP), we thought it would be a close match as long as the robot was always on the ground and so this problem wouldn't practically matter.

The main difference between a standard SAC implementation and ours is:

* Our implementation used SAC in a discrete action setting (https://arxiv.org/abs/1910.07207). This was mainly due to time constraints as we were more familiar with the discrete setting and thought it would be easier to implement
* Since we were running linux off of a usb stick, we didn't want to read and write large amounts of data on memory and so ended up not using a replay buffer. Additionally, we only had access to a Nvidia 1050 GPU which could only progress a batch size of 1. Although this decreased the sample efficiency, because of these constraints it did make it possible to use TD($\lambda$) algorithm.
* In our testing, some actions were quickly made extremely unlikely, this may be due to the memory problem above. To counter this we ran it off policy with an epsilon soft algorithm.
* In long straight sections, we found that the robot would take an indirect curved path since the difference in future expected reward was very similar. Although this didn't lead to many problems in training, running it for the competition would introduce new dynamics since we could not pause the simulation between states, which could lead to differences in performance between testing and deployment. If the robot instead took a straight path we would be more confident that the robot would perform as expected. Although we think this behavior would disappear with enough training, due to time and compute restraints, we opted to change the reward function such that each reward achieved while turning is divided by 2.5.

### Computer Vision

#### Locating Signs 
While the car drives, camera feed images were processed to locate signs and subsequently letters. To locate signs, we applied bitmaps that filtered for distinct colours of blue that the sign consisted of. Extra filtering was then applied by dilating and eroding the gray-scaled image. Contours were then found by running OpenCV’s canny Edge Detection. Once the contour is found, a perspective transform is done to fit the screen, however the image normally contains the outer blue section of the sign. To isolate the white interior of the sign, we again ran a blue mask function and repeated the steps above. The resulting image contains only the interior of the sign with the clue value and clue type.
 
#### Identifying Letters
Because the camera resolution is low, letters often have varying pixel colours with little distinction between a connection within a letter and a gap between two letters. This makes it difficult to determine the correct masking from just one threshold. After much experimentation, we settled on: (i) masking the pixels who’s blue value is 1.6 times greater than their red and green value,  (ii) taking the average of those pixels, and (iii) finding all pixels within a threshold of the average blue. To segment the letters, canny Edge Detection was applied, with a bounding rectangle created for each contour. As a safety measure, bounding boxes were merged or split in special conditions.


<img width="53" alt="letter-segment" src="https://github.com/user-attachments/assets/9f261814-8687-4e34-b71a-0b6081975dc9">

Example of a letter that has separate contours and needs to be merged. 

#### Data Generation
##### Generating Signs
Random strings of letters and numbers were generated, then, using the provided code from ‘licenseplategenerator.py’, was rendered on a sign and sent through the same preprocessing steps outlined above.

##### Data Augmentation
When comparing the fake data to real driving data, we noticed that the synthetic data and the real data had a different variance in the letter width. To address this, we decided to first blur the letter, then use a percent blue mask with a varying threshold value. 

Additionally, to better replicate the real data, random sections of letters were removed half the time in varying intensities. Various filters were also applied to blur the image further. 

### Convolutional Neural Network
The data was formatted using one hot encoding and the model was trained to predict one letter or number at a time.

This architecture was loosely inspired by AlexNet. I significantly reduced the size of the model and number of layers. 

| Layer (type)               | Output Shape         | Param #  |
|----------------------------|----------------------|----------|
| conv2d_2 (Conv2D)           | (None, 32, 27, 64)   | 640      |
| max_pooling2d_2 (MaxPooling2D) | (None, 16, 13, 64)   | 0        |
| conv2d_3 (Conv2D)           | (None, 16, 13, 64)   | 36928    |
| max_pooling2d_3 (MaxPooling2D) | (None, 8, 6, 64)    | 0        |
| flatten_1 (Flatten)         | (None, 3072)         | 0        |
| dense_3 (Dense)             | (None, 512)          | 1573376  |
| dropout_3 (Dropout)         | (None, 512)          | 0        |
| dense_4 (Dense)             | (None, 256)          | 131328   |
| dropout_4 (Dropout)         | (None, 256)          | 0        |
| dense_5 (Dense)             | (None, 128)          | 32896    |
| dropout_5 (Dropout)         | (None, 128)          | 0        |
| dense_6 (Dense)             | (None, 36)           | 4644     |
| **Total params**            | **1779812 (6.79 MB)**|          |
| **Trainable params**        | **1779812 (6.79 MB)**|          |
| **Non-trainable params**    | **0 (0.00 Byte)**    |          |


#### Training 
To avoid overfitting, new data was generated every epoch. The model was trained on 1,000 fake images per epoch for 8 epochs. 

#### Testing: Confusion Matrix 
<img width="496" alt="image" src="https://github.com/user-attachments/assets/f4d391d0-00d5-4e1f-9208-df14e1f7a0e3">


### Results

We ended up winning the competition with a perfect score on the course and the fastest time to date of 33 seconds.
