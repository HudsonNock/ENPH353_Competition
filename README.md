# ENPH 353 Competition

ENPH 353 is a project course in engineering physics at UBC where students create autonomous programs
to drive a car in simulation through a race track and read signs along the way. Our team used reinforcement learning (RL)
to drive and a various filters along with a CNN to read the signs.

Above view of the robot, note that only two wheels had to be inside the road at any given time. <br />
https://github.com/user-attachments/assets/7e9fa4be-4a7a-4b7a-b543-e153ebce588a

Camera view of the robot (what we have access to) <br />
https://github.com/user-attachments/assets/874a72b2-1b82-4684-a60d-bab6adc491f1

This repo contains a brief summary of our implementation, for more information refer to our report: <br /> [ENPH353 Final Report.pdf](https://github.com/user-attachments/files/16322865/ENPH353.Final.Report.pdf)

## Introduction

### Problem outline

The competition is aimed towards maximizing the number of points achieved. Points are gathered by reading signs scattered throughout the map, with deductions given when more than 2 wheels are off the road or in the case of a collision or respawn. Although being faster doesn't grant you more points, the time to completion does play a role in tie breakers.

While driving, the robot is only given a camera view and has to determine its actions from that. It can then output a desired speed and steering angle.
We decided to drive using a soft actor critic (SAC, https://arxiv.org/abs/1801.01290) algorithm which was inspired by TMRL (https://github.com/trackmania-rl/tmrl).

To set up the environment for training, we could determine the position of the wheels as we were using ROS and Gazebo, then compare them against a bitmap of the allowed path, if two or more wheels were off the road it would count as a respawn. The state space was composed of the camera view filtered with a gray scaled and 2x2 average mask applied, as well as the speed set by the robot. Although the speed set does not necessarily match the actual speed (which could lead to an environment that isn't a MDP), we thought it would be a close match as long as the robot was always on the ground and so this problem wouldn't practically matter.

The main difference between TMRL's implementation and ours is:

* Our implementation used SAC in a discrete action setting (https://arxiv.org/abs/1910.07207). This was mainly due to time constraints as we were more familiar with the discrete setting and thought it would be easier to implement
* Since we were running linux off of a usb stick, we didn't want to read and write large amounts of data on memory and so ended up not using a replay buffer. Additionally, we only had access to a Nvidia 1050 GPU which could only progress a batch size of 1. Although this decreased the sample efficiency, because of these constraints it did make it possible to use TD($\lambda$) algorithm.
* In my testing, some actions were quickly made extremely unlikely, this may be due to the memory problem above. To counter this we ran it off policy with an epsilon soft algorithm.
* In long straight sections, we found that the robot would take an indirect curved path since the difference in future expected reward was very similar. Although this didn't lead to many problems in training, when running it for the competition we could not pause it between states as it had to run in real time, which could lead to differences in performance between testing and deployment. If the robot instead took a straight path we would be more confident that the robot would perform as expected. Although we think this behavior would disappear with enough training, due to time and compute restraints, we opted to change the reward function such that each reward achieved while turning is divided by 2.5.
