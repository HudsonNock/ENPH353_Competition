#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import gc
import sys

sys.path.append("/home/fizzer/ros_ws/src/my_controller/src")
import read_sign

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gazebo_msgs.srv import GetModelState, SetModelState, GetLinkState
from gazebo_msgs.msg import ModelState, ContactsState
from torch.nn import init
import torch.autograd.profiler as profiler


class myModel(nn.Module):

	def __init__(self, i, channels):
		super(myModel, self).__init__()

		self.channels = channels
		self.i = i
		self.conv1_frame = nn.Conv2d(self.channels, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		self.maxpool1_frame = nn.MaxPool2d(kernel_size=(5,5), stride=(5,5))
		self.conv2_frame = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		self.maxpool2_frame = nn.MaxPool2d((3,3), stride=(3,3))
		self.conv3_frame = nn.Conv2d(32, 8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		self.maxpool3_frame = nn.MaxPool2d((3,3), stride=(3,3))

		self.dense1 = nn.Linear(896, 512)
		self.dense2 = nn.Linear(512, 256)
		self.dense3 = nn.Linear(256,128)
		if i == 0:
			self.latent = nn.Linear(128,4)

			self.decode1 = nn.Linear(6,32)
			self.decode2 = nn.Linear(32,32)
			self.output = nn.Linear(32,9)
		else:
			self.latent = nn.Linear(128, 62)

			self.decode1 = nn.Linear(64, 32)
			self.decode2 = nn.Linear(32,16)
			self.output = nn.Linear(16, 9)
		#else:
		#	self.output = nn.Linear(128,9)

		self.batchnorm1 = nn.BatchNorm2d(64)
		self.batchnorm2 = nn.BatchNorm2d(32)
		self.batchnorm3 = nn.BatchNorm2d(8)

		self.batchnorm1d = nn.BatchNorm2d(32)
		self.batchnorm2d = nn.BatchNorm2d(16)
		self.batchnorm3d = nn.BatchNorm2d(8)

		self.flatten = nn.Flatten()

		if self.i < 3:
			self.apply(self.init_weights)

	def init_weights(self, module):
		if self.i > 0 and module == self.output:
			module.weight.data.normal_(mean=0.0, std=0.1)
			if module.bias is not None:
				module.bias.data.zero_()
		elif module == self.output:
			module.weight.data.normal_(mean=0.0, std=0.1)
			if module.bias is not None:
				module.bias.data.zero_()
		elif isinstance(module, (nn.Linear, nn.Conv2d)):
			module.weight.data.normal_(mean=0.0,std=0.1)
			if module.bias is not None:
				module.bias.data.zero_()

	def forward(self, input_frame, numeric_input):
		x_frame = self.flatten(self.batchnorm3(self.maxpool3_frame(F.relu(self.conv3_frame(\
			  self.batchnorm2(self.maxpool2_frame(F.relu(self.conv2_frame(self.batchnorm1(self.maxpool1_frame(\
			  F.relu(self.conv1_frame(input_frame))))))))))))).cuda()

		#x_combined = self.flatten(torch.cat([x_frame, numeric_input], dim=1)).cuda()

		#del x_frame

		x_latent = self.latent(nn.Tanh()(self.dense3(nn.Tanh()(self.dense2(nn.Tanh()(self.dense1(x_frame))))))).cuda()
		x_combined = self.flatten(torch.cat([x_latent, numeric_input], dim=1)).cuda()

		x = self.output(nn.Tanh()(self.decode2(nn.Tanh()(self.decode1(x_combined))))).cuda()
		#print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))

		del x_combined
		if self.i == 0:
			softmx = nn.Softmax(dim=1)(x).cuda()
			del x
			return softmx
		else:
			return x


class Controller():

	def _initalize_models(self):
		model1 = myModel(0, self.channels)
		state_dict = torch.load(self.weights_folder_1)
		model1.load_state_dict(state_dict)
		model1.cuda()
		self.models.append(model1)

		model2 = myModel(0, self.channels)
		state_dict = torch.load(self.weights_folder_2)
		model2.load_state_dict(state_dict)
		model2.cuda()
		self.models.append(model2)

		model3 = myModel(0, self.channels)
		state_dict = torch.load(self.weights_folder_3)
		model3.load_state_dict(state_dict)
		model3.cuda()
		self.models.append(model3)

		model4 = myModel(0, self.channels)
		state_dict = torch.load(self.weights_folder_4)
		model4.load_state_dict(state_dict)
		model4.cuda()
		self.models.append(model4)

	def __init__(self):
		self.lin_vel = 0.0
		self.ang_vel = 0.0
		self.twist_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size = 1)
		self.sign_pub = rospy.Publisher('score_tracker', String, queue_size = 1)

		self.channels = 1
		self.input_shape = (self.channels, 360, 640)
		# pi, Q1, Q2, Qt1, Qt2
		self.models = nn.ModuleList()
		self.weights_folder_1 = '/home/fizzer/ros_ws/src/my_controller/src/weights_1.6Speed_Grass/pi.h5'
		self.weights_folder_2 = '/home/fizzer/ros_ws/src/my_controller/src/weights_2.5Speed_2/pi.h5'
		self.weights_folder_3 = '/home/fizzer/ros_ws/src/my_controller/src/weights_1.4Speed_3/pi.h5'
		self.weights_folder_4 = '/home/fizzer/ros_ws/src/my_controller/src/weights_2.0Speed_4/pi.h5'

		self._initalize_models()
		print("loaded model")
		self.batch_size = 1
		#Replay buffer
		self.s = np.zeros(shape=(self.channels, 360,640), dtype=np.float32)
		self.s_vel = np.zeros(shape=(2,), dtype=np.float32)
		self.pi = np.zeros(shape=(9,), dtype=np.float32)

		self.section = 1
		self.section_count = 0

		self.s_tf = torch.zeros(size=(1,self.channels,360,640), dtype=torch.float32).cuda()
		self.s_vel_tf = torch.zeros(size=(1,2), dtype=torch.float32).cuda()

		self.img_cnt = 0

		self.sec2 = 0
		self.sec3 = 0
		self.sec4 = 0
		self.pink3 = False

		self.iteration = 0
		self.dcount = 0
		self.fcount = 0

		self.start = 0
		self.end = 0

		self.restart_sign = False

		self.ip = read_sign.ImageProcessing()
		self.missed_signs = []

	def _publish_vel(self):
		tw = Twist()
		tw.linear.x = self.lin_vel
		tw.angular.z = self.ang_vel
		self.twist_pub.publish(tw)

	def _check_pink(self, img, num, den):
		height, width, _ = img.shape
		bottom_third = img[height * num // den:, :, :]
		diff = np.sum(np.abs(bottom_third - np.array([254,0,254])),axis=2)

		if np.any(diff < 10):
			return True
		return False

	def _take_action_distribute(self):
		if self.section == 1:
			lin_vel_add = self.pi[1] * 0.2 +  self.pi[2] * 0.2 +  self.pi[3] * 0.2 - \
					 self.pi[6] * 0.2 - self.pi[7] * 0.2 - self.pi[8] * 0.2
			if self.section_count <= 95:
				self.ang_vel = 0.9 * (4 * self.pi[1] + 4* self.pi[4] + 4* self.pi[6] - \
						4 * self.pi[3] - 4 * self.pi[5] - 4 * self.pi[8])
			else:
				self.ang_vel = 0.75 * (4 * self.pi[1] + 4* self.pi[4] + 4* self.pi[6] - \
						4 * self.pi[3] - 4 * self.pi[5] - 4 * self.pi[8])
			self.lin_vel += lin_vel_add
			#if self.section_count >= 164 and self.section_count <= 169:
			#	self.lin_vel += 0.2
			#else:
			#	self.lin_vel += lin_vel_add

			if self.section_count < 44:
				self.lin_vel += 0.13
			elif self.section_count >= 100 and self.section_count <= 107:
				self.lin_vel += 0.13

			if self.section_count == 83 or self.section_count == 84 or self.section_count == 85:
				self.ang_vel = 5.0
			#elif self.section_count == 76 or self.section_count == 77:
			#	self.ang_vel -= 2.0
			#elif self.section_count == 72 or self.section_count == 73:
			#	self.ang_vel = 0.0
			elif self.section_count == 64:
				self.ang_vel = -4.0
			#elif self.section_count >= 85 and self.section_count <= 97:
			#	self.lin_vel = 0.7
			#elif self.section_count == 126 or self.section_count == 127 or self.section_count == 128:
			#	self.ang_vel = -4.0
			#elif self.section_count == 126 or self.section_count == 127 or self.section_count == 128 or self.section_count == 129:
			#	self.ang_vel = -6.0
			#elif self.section_count == 37 or self.section_count == 38 or self.section_count == 39 or self.section_count == 40:
			#	self.ang_vel = 5.5
			#elif self.section_count == 41:
			#	self.ang_vel = 0
			elif self.section_count == 46 or self.section_count == 47 or self.section_count == 48:
				self.ang_vel = -4.0
			#elif self.section_count == 70 or self.section_count == 71 or self.section_count == 72:
			#	self.ang_vel = -5.0
			#elif self.section_count == 73 or self.section_count == 74 or self.section_count == 75:
			#	self.ang_vel = 6.5
			#elif self.section_count == 76 or self.section_count == 77:
			#	self.ang_vel = -3.2

			if self.lin_vel < 0:
				self.lin_vel = 0
			if self.lin_vel > 1.6 * 0.88:
				self.lin_vel = 1.6 * 0.88
		elif self.section == 2:
			if self.section_count >= 58 and self.section_count <= 75:
				self.ang_vel = 4 * self.pi[1] + 4* self.pi[4] + 4* self.pi[6] - \
					4.7 * self.pi[3] - 4.7 * self.pi[5] - 4.7 * self.pi[8] - 0.25
			else:
				self.ang_vel = 4 * self.pi[1] + 4* self.pi[4] + 4* self.pi[6] - \
					4.7 * self.pi[3] - 4.7 * self.pi[5] - 4.7 * self.pi[8]

			if self.section_count >= 68:
				if self.ang_vel > 0:
					self.ang_vel *= 0.8

			self.lin_vel += self.pi[1] * 0.4 +  self.pi[2] * 0.4 +  self.pi[3] * 0.4 - \
					 self.pi[6] * 0.2 - self.pi[7] * 0.2 - self.pi[8] * 0.23

			if self.section_count < 10:
				self.lin_vel += 0.15

			if self.section_count >= 41 and self.section_count <= 51 and self.ang_vel > 0.5:
				self.ang_vel += 2.0

			if self.section_count >= 41 and self.section_count <= 43:
				self.lin_vel = 1.5
				#self.ang_vel += 2.0
			elif self.section_count == 47 or self.section_count == 48:
				#self.ang_vel = -4.0
				self.lin_vel += 0.3
			elif self.section_count == 76 or self.section_count == 77:
				self.lin_vel = 1.5
				self.ang_vel -= 2.0
			elif self.section_count >= 78:
				self.lin_vel = 1.0

			if self.lin_vel < 0:
				self.lin_vel = 0
			if self.lin_vel > 2.1:
				self.lin_vel = 2.1
		elif self.section == 3:
			self.lin_vel += self.pi[1] * 0.2 +  self.pi[2] * 0.2 +  self.pi[3] * 0.2 - \
					 self.pi[6] * 0.2 - self.pi[7] * 0.2 - self.pi[8] * 0.2
			self.ang_vel = 2.0*(4 * self.pi[1] + 4* self.pi[4] + 4* self.pi[6] - \
					4 * self.pi[3] - 4 * self.pi[5] - 4 * self.pi[8])

			if self.section_count == 10 or self.section_count == 11:
				self.ang_vel -= 2.0
			#elif self.section_count == 11 or self.section_count == 12:
			#	self.ang_vel = 3

			if self.section_count >= 172 and self.section_count <= 178:
				self.ang_vel -= 1.0
			if self.lin_vel < 0:
				self.lin_vel = 0
			if self.lin_vel > 1.0:
				self.lin_vel = 1.0
		elif self.section == 4:
			action = np.argmax(self.pi)
			if action == 0:
				self.ang_vel = 0
			elif action == 1:
				self.lin_vel += 0.2
				self.ang_vel = 2
			elif action == 2:
				self.lin_vel += 0.2
			elif action == 3:
				self.lin_vel += 0.2
				self.ang_vel = -2
			elif action == 4:
				self.ang_vel = 2
			elif action == 5:
				self.ang_vel = -3
			elif action == 6:
				self.lin_vel -= 0.2
				self.ang_vel = 2
			elif action == 7:
				self.lin_vel -= 0.2
			elif action == 8:
				self.lin_vel -= 0.2
				self.ang_vel = -2
			if self.lin_vel < 0:
				self.lin_vel = 0
			if self.lin_vel > 1.8:
				self.lin_vel = 1.8

			if self.section_count == 118 or self.section_count == 119 or self.section_count == 120:
				self.lin_vel -= 0.08

			if self.img_cnt - self.sec4 < 25:
				if self.img_cnt - self.sec4 > 12:
					self.lin_vel += 0.04
					self.ang_vel += 0.07
					#if self.restart_sign == False:
					#	self.ang_vel = 4.0

		self._publish_vel()

	def check_sign(self):
		self.iteration += 1
		self.restart_sign = True
		self.missed_signs = self.ip.get_missed_signs()
		if len(self.missed_signs) == 0:
			words = self.ip.get_words()
			for i in range(len(words)):
				self.sign_pub.publish(str('nootnoot,noots,' + str(i+1) + ',' + words[i]))
				time.sleep(0.05)
			self.sign_pub.publish(str('nootnoot,noots,-1,NA'))
			while True:
				continue

		self.start = self.missed_signs[0]
		self.end = self.missed_signs[len(self.missed_signs)-1]
		self.sec2 = 0
		self.sec3 = 0
		self.sec4 = 0
		self.section_count = 0
		self.img_cnt = 0
		self.pink3 = False
		if self.start == 1 or self.start == 2 or self.start == 3 or self.start == 4:
			self.respawn(0)
			self.section = 1
		elif self.start == 5 or self.start == 6:
			self.respawn(1)
			self.section = 2
		elif self.start == 7:
			self.respawn(2)
			self.section = 3
		elif self.start == 8:
			self.respawn(3)
			self.section = 4
			self.pink3 = True
			self.fcount = 30

	def _process_image(self, msg):
		#self.img_cnt += 1
		#self.respawn(self.img_cnt % 4)

		if self.dcount > 0:
			self.dcount -= 1
			self.lin_vel = 0
			self.ang_vel = 0
			self._publish_vel()
			return

		if self.fcount > 0:
			self.fcount -= 1
			self.lin_vel += 0.1
			if self.lin_vel > 1.2:
				self.lin_vel = 1.2
			self.ang_vel = 0
			self._publish_vel()
			return

		torch.cuda.synchronize()

		bridge = CvBridge()
		cv_image = cv2.cvtColor(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), cv2.COLOR_BGR2RGB)

		self.img_cnt += 1
		self.section_count += 1 #164, 2,3

		if self.restart_sign:
			interval_bounds = [[1,15],
						[44,58],
						[70,84],
						[self.sec2-23, self.sec2],
						[self.sec2+35, self.sec2+54],
						[self.sec3-15, self.sec3],
						[self.sec4-15, self.sec4+5]]
			idx = 0
			if self.end == 1 and self.section_count > 15:
				idx = 1
			elif self.end == 2 and self.section_count > 58:
				idx = 2
			elif self.end == 3 and self.section_count > 84:
				idx = 3
			elif self.end == 4 and self.section == 2:
				idx = 4
			elif self.end == 5 and self.section_count > 54 and self.section == 2:
				idx = 5
			elif self.end == 6 and self.section == 3:
				idx = 6
			elif self.end == 7 and self.pink3:
				idx = 7
			elif self.end == 8 and self.section == 4 and self.section_count > 118:
				idx = 8

			if idx != 0:
				intervals = []
				for i in range(min(idx,7)):
					if i+1 in self.missed_signs:
						intervals.append(interval_bounds[i])
				if idx == 8:
					intervals.append([self.img_cnt - 34, self.img_cnt - 2])
				self.ip.update_group(intervals)
				self.check_sign()
				return

		if self.section == 1 and self.section_count >= 135 and self._check_pink(cv_image, num = 1, den = 2):
			self.sec2 = self.img_cnt
			self.section_count = 1
			self.section = 2
			print("section 2")
			print(self.img_cnt)
		elif self.section == 1 and self.section_count >= 165:
			self.img_cnt = 0
			self.section_count = 0
			self.respawn(0)
			return
		elif self.section == 2 and self.section_count >= 82 and self._check_pink(cv_image, num=1, den = 2):
			self.sec3 = self.img_cnt
			self.section_count = 1
			self.section = 3
			print("section 3")
			print(self.img_cnt)
		elif self.section == 2 and self.section_count >= 108:
			self.img_cnt = self.sec2
			self.section_count = 0
			self.respawn(1)
			return
		elif self.pink3 == False and self.section == 3 and self.section_count >= 50 and self._check_pink(cv_image, num=2, den=3):
			self.sec4 = self.img_cnt
			print("section 4")
			print(self.img_cnt)
			self.pink3 = True
		elif self.section == 3 and self.img_cnt - self.sec4 >= 25 and self.sec4 != 0:
			self.section_count = 1
			self.section = 4
		elif self.section == 3 and self.pink3 == False and self.section_count >= 150:
			self.section_count = 0
			self.img_cnt = self.sec3
			self.respawn(2)
			return
		elif self.section == 4 and self.section_count >= 115 and self.section_count <= 124:
			self.lin_vel = 0.5
			self.ang_vel = 0.0
			self._publish_vel()
			filename = "image_{0}".format(self.img_cnt)
			cv2.imwrite('/home/fizzer/ros_ws/src/my_controller/src/vision/image_' + str(self.iteration) + '_' + str(self.img_cnt) + '.jpg', cv_image)
			return
		elif self.section == 4 and self.section_count > 118:
			self.lin_vel = 0.0
			self.ang_vel = 0.0
			self._publish_vel()
			self.ip.update_sec(self.sec2, self.sec3, self.sec4, self.img_cnt - 2)
			self.ip.run()
			self.check_sign()
			return

		cv2.imwrite('/home/fizzer/ros_ws/src/my_controller/src/vision/image_' + str(self.iteration) + '_' + str(self.img_cnt) + '.jpg', cv_image)
		cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2GRAY)
		#cv2.imshow('Vision', cv_image)
		#cv2.waitKey(1)

		if self.section == 1:
			if self.section_count < 9:
				self.ang_vel = 0
				self.lin_vel += 0.2
				self._publish_vel()
				return

		torch_image = torch.tensor(cv_image, dtype=torch.float32)
		torch_image_batched = torch.unsqueeze(torch_image, dim=0)
		result = nn.functional.avg_pool2d(torch_image_batched, kernel_size=(2,2))
		if self.channels == 1:
			result = torch.unsqueeze(result, dim=1)
		self.s = result[0].numpy() / 128.0 - 1

		#cv2.imshow('Vision', result[0][0].numpy().astype(np.uint8))
		#cv2.waitKey(1)

		self.s_vel = np.array([self.lin_vel * 2.0 / 2.5 - 1, self.ang_vel / 4.0])

		self.s_tf.copy_(torch.unsqueeze(torch.tensor(self.s), dim=0))
		self.s_vel_tf.copy_(torch.unsqueeze(torch.tensor(self.s_vel), dim=0))

		if self.section == 1:
			with torch.no_grad():
				predict = self.models[0](self.s_tf, self.s_vel_tf)
				self.pi = predict.cpu().detach().numpy()[0]
				del predict
		elif self.section == 2:
			with torch.no_grad():
				predict = self.models[1](self.s_tf, self.s_vel_tf)
				self.pi = predict.cpu().detach().numpy()[0]
				del predict
		elif self.section == 3:
			with torch.no_grad():
				predict = self.models[2](self.s_tf, self.s_vel_tf)
				self.pi = predict.cpu().detach().numpy()[0]
				del predict
		elif self.section == 4:
			with torch.no_grad():
				predict = self.models[3](self.s_tf, self.s_vel_tf)
				self.pi = predict.cpu().detach().numpy()[0]
				del predict

		for i in range(len(self.pi)):
			if np.isnan(self.pi[i]):
				self.pi[i] = 0

		self._take_action_distribute()
		torch.cuda.empty_cache()

	#def _pause_simulation(self):
	#	rospy.wait_for_service('/gazebo/pause_physics')
	#	pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
	#	pause_physics()

	#def _unpause_simulation(self):
	#	rospy.wait_for_service('/gazebo/unpause_physics')
	#	unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
	#	unpause_physics()

	def respawn(self, index):
		msg = ModelState()
		msg.model_name = 'R1'

		pos = [[5.493696, 2.472393, 0.0525], [0.5136, 0, 0.0525], [-3.7532, 0.4235, 0.525], [-3.8523, -2.275, 0.0525]]
		orientation = [[0, 0, 0.707, -0.707], [0, 0, 0.707, 0.707], [0,0,0.93,-np.sqrt(1-0.92**2)], [0, 0, 0, -1.0]]

		msg.pose.position.x = pos[index][0]
		msg.pose.position.y = pos[index][1]
		msg.pose.position.z = pos[index][2]
		msg.pose.orientation.y = orientation[index][0]
		msg.pose.orientation.x = orientation[index][1]
		msg.pose.orientation.z = orientation[index][2]
		msg.pose.orientation.w = orientation[index][3]

		self.lin_vel = 0
		self.ang_vel = 0
		self.dcount = 5
		self._publish_vel()

		rospy.wait_for_service('/gazebo/set_model_state')
		set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		resp = set_state(msg)

	def run(self):
		rospy.init_node('adeept_awr_driver')
		rate = rospy.Rate(10)

		#self._pause_simulation()
		#time.sleep(4)
		#self._unpause_simulation()
		#time.sleep(3.5)
		input("press to start...")
		print("subscribing")

		self.sign_pub.publish(str('nootnoot,multi21,0,NA'))

		msg = ModelState()
		msg.model_name = 'R1'

		msg.pose.position.x = 5.493696
		msg.pose.position.y = 2.472393
		msg.pose.position.z = 0.0525
		msg.pose.orientation.y = 0
		msg.pose.orientation.x = 0
		msg.pose.orientation.z = 0.707
		msg.pose.orientation.w = -0.707

		rospy.wait_for_service('/gazebo/set_model_state')
		set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		resp = set_state(msg)

		self.lin_vel = 0.2
		self.ang_vel = 0.0
		self._publish_vel()

		print("spin")
		rospy.Subscriber('R1/pi_camera/image_raw', Image, self._process_image)
		rospy.spin()

if __name__=='__main__':
	print("Running Model")
	dr = Controller()
	dr.run()
