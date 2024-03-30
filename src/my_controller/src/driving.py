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
		for i in range(3):
			model = myModel(i, self.channels)
			state_dict = torch.load(self.weights_folder[i])
			model.load_state_dict(state_dict)
			model.cuda()
			self.models.append(model)
		for i in range(3,5):
			model = myModel(i, self.channels)
			state_dict = torch.load(self.weights_folder_best[i])
			model.load_state_dict(state_dict)
			#model.load_state_dict(self.models[i-2].state_dict())
			model.cuda()
			self.models.append(model)

	def __init__(self):

		self.epoch = 0
		self.lin_vel = 0.0
		self.ang_vel = 0.0
		self.twist_pub = rospy.Publisher('R1/cmd_vel', Twist, queue_size = 1)
		self.sign_pub = rospy.Publisher('score_tracker', String, queue_size = 1)

		image_path = 'binaryMap2023_restricted.png'
		path = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		self.binaryMap = (path == 255)
		self.rwd = 0.0
		self.pos = np.array([0.0,0.0,0.0])
		self.curr_idx = 0

		points_arr = np.loadtxt('path_30_v1.txt')
		self.points = points_arr.reshape(-1,3)
		#self._draw_points()

		# INITALIZE NEURAL NETWORKS

		# Policy network
		# actions are: +0.1, +1, + 10, + 100, -0.1, -1, -10, -100, (set 0) for both velocity and angular velocity,
		# so |A| = 18
		self.H = 0.8 #0.8 #0.98 * (- np.log(1 / 3.0))

		self.channels = 1
		self.input_shape = (self.channels, 360, 640)
		# pi, Q1, Q2, Qt1, Qt2
		self.models = nn.ModuleList()
		#self.weights_folder = ['weights3/pi.h', 'weights3/Q1.h5', 'weights3/Q2.h5', 'weights3/Qt1.h5', 'weights3/Qt2.h5']
		#self._initalize_models()
		self.weights_folder = ['weights5L/pi.h5', 'weights5L/Q1.h5', 'weights5L/Q2.h5', 'weights5L/Qt1.h5', 'weights5L/Qt2.h5']
		self.weights_folder_best = ['weights_1.6Speed_Grass/pi.h5', 'weights_1.6Speed_Grass/Q1.h5', 'weights_1.6Speed_Grass/Q2.h5', 'weights_1.6Speed_Grass/Qt1.h5', 'weights_1.6Speed_Grass/Qt2.h5']

		self._initalize_models()
		self.alpha = nn.Parameter(torch.tensor(0.0047), requires_grad=False)
		self.log_alpha = nn.Parameter(torch.tensor(-5.0))
		self.batch_size = 1
		self.gamma = 0.97
		self.tau = 0.002
		#Replay buffer
		self.s = np.zeros(shape=(self.batch_size+1, self.channels, 360,640), dtype=np.float32)
		self.s_vel = np.zeros(shape=(self.batch_size+1, 2), dtype=np.float32)
		self.pi = np.zeros(shape=(self.batch_size+1, 9), dtype=np.float32)
		self.a = np.zeros(shape=(self.batch_size+1,), dtype=np.int8)
		self.r = np.zeros(shape=(self.batch_size+1,), dtype=np.float32)
		self.d = np.zeros(shape=(self.batch_size+1,), dtype=bool)
		#self.q = np.zeros(shape=(self.batch_size+1,), dtype=np.float32)
		self.index = 0
		self.start = True

		self.still = 0

		optimizer_phi = optim.Adam(self.models[0].parameters(), lr = 0.00007) #, lr=0.0005) #0.002
		optimizer_q1 = optim.Adam(self.models[1].parameters(), lr = 0.00007) #, lr=0.0005)
		optimizer_q2 = optim.Adam(self.models[2].parameters(), lr = 0.00007) #, lr=0.0005) #.0009
		optimizer_alpha = optim.Adagrad([self.log_alpha], lr=0.01)
		self.optimizers = [optimizer_phi, optimizer_q1, optimizer_q2, optimizer_alpha]

		self.optimizers[0].zero_grad()
		self.optimizers[1].zero_grad()
		self.optimizers[2].zero_grad()

		self.qcnt = [0,0]
		self.acnt = 0
		self.q_lambda = 0.86

		self.s_tf = torch.zeros(size=(1,self.channels,360,640), dtype=torch.float32).cuda()
		self.s_vel_tf = torch.zeros(size=(1,2), dtype=torch.float32).cuda()

		self.s_tf_batch = torch.zeros(size=(self.batch_size+1, self.channels, 360, 640), dtype=torch.float32).cuda()
		self.s_vel_tf_batch = torch.zeros(size=(self.batch_size+1, 2), dtype=torch.float32).cuda()

		self.cnt = 0
		self.hit = False
		self.hasR = False
		self.hasC = False
		self.hasC2 = False

		self.respawn_idx = [0, 104, 195, 272]
		self.respawn_loc = 0

		self.best_index = 350
		print(torch.cuda.memory_summary())

	def _draw_points(self):
		self.bitmapPos = cv2.imread('binaryMap2023.png', cv2.IMREAD_COLOR)
		for p in self.points:
			x,y,z = p
			xy = [x,y]
			pxl = self._cord_to_pixel(xy)
			swap = [pxl[1], pxl[0]]
			color = (0,0,255)
			cv2.circle(self.bitmapPos, swap, 3, color)
			cv2.imshow('Robot Position', self.bitmapPos)
			cv2.waitKey(1)

	def _cord_to_pixel(self, coordinates):
		#returns pixel for bitmap in (row, col)
		pixel = [0,0]
		pixel[0] = int((coordinates[1] + 2.77003) *  127.009515) - 6
		pixel[1] = int((coordinates[0] - 5.943569) *  (-127.024292)) + 6
		if pixel[0] < 0:
			pixel[0] = 0
		if pixel[0] >= len(self.binaryMap):
			pixel[0] = len(self.binaryMap) - 1
		if pixel[1] < 0:
			pixel[1] = 0
		if pixel[1] >= len(self.binaryMap[0]):
			pixel[1] = len(self.binaryMap[0]) - 1
		return pixel

	def _check_offroad(self):
		#returns true if offroad
		#self.bitmapPos = cv2.imread('binaryMap2023.png', cv2.IMREAD_COLOR)
		passes = 0
		wheel_names = ['R1::rear_right_wheel', 'R1::rear_left_wheel', 'R1::front_left_wheel', 'R1::front_right_wheel']

		rospy.wait_for_service('/gazebo/get_link_state')
		get_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)

		sumX = 0
		sumY = 0
		sumZ = 0

		for wheel in wheel_names:
			response = get_link_state(link_name=wheel)
			wheel_cord = [response.link_state.pose.position.x, response.link_state.pose.position.y]
			sumX += response.link_state.pose.position.x
			sumY += response.link_state.pose.position.y
			sumZ += response.link_state.pose.position.z
			wheelPxl = self._cord_to_pixel(wheel_cord)
			if (self.binaryMap[wheelPxl[0]][wheelPxl[1]]):
				passes += 1
			#for i in range(len(self.points)):
			#	pointPxl = self._cord_to_pixel(self.points[i])
			#	self.bitmapPos[pointPxl[0]][pointPxl[1]][0] = 255
			#	self.bitmapPos[pointPxl[0]][pointPxl[1]][1] = 0
			#	self.bitmapPos[pointPxl[0]][pointPxl[1]][2] = 0

		#cv2.imshow('Robot Position', self.bitmapPos)
		#cv2.waitKey(1)
		self.pos[0] = sumX / 4.0
		self.pos[1] = sumY / 4.0
		self.pos[2] = sumZ / 4.0
		if (passes >= 2):
			return False
		return True

	def _check_collision(self, msg):
		if (msg.data == True):
			self.hit = True

	def _publish_vel(self):
		tw = Twist()
		tw.linear.x = self.lin_vel
		tw.angular.z = self.ang_vel
		self.twist_pub.publish(tw)

	def _find_closest_point(self):
		maxI = self.curr_idx
		d_curr = np.sum((self.points[self.curr_idx] - self.pos)**2)
		passed = False
		for i in range(self.curr_idx + 1, min(self.curr_idx + 21, len(self.points))):
			dp = np.sum((self.points[i] - self.pos)**2)
			if (dp < d_curr):
				passed = True
				maxI = i
				d_curr = dp
		if (passed):
			self.rwd = 0.01 * (maxI - self.curr_idx)
			self.curr_idx = maxI

	def _check_terminal_state(self):
		if self.hasC2:
			self.hasC2 = False
			return 1
		if self.rwd == -1:
			return 1
		if 1.753029 < self.pos[2] < 1.989973 and -1.791982 < self.pos[0] < -0.886508 and -0.36789 < self.pos[1] < -0.028055:
			return 1
		return 0

	def _take_action(self):
		#self._take_action_distribute()
		#return
		action = self.a[self.index]
		if action == 0:
			self.ang_vel = 0
		elif action == 1:
			self.lin_vel += 0.2
			self.ang_vel = 4
		elif action == 2:
			self.lin_vel += 0.2
		elif action == 3:
			self.lin_vel += 0.2
			self.ang_vel = -4
		elif action == 4:
			self.ang_vel = 4
		elif action == 5:
			self.ang_vel = -4
		elif action == 6:
			self.lin_vel -= 0.2
			self.ang_vel = 4
		elif action == 7:
			self.lin_vel -= 0.2
		elif action == 8:
			self.lin_vel -= 0.2
			self.ang_vel = -4
		if self.lin_vel < 0:
			self.lin_vel = 0
		if self.lin_vel > 1.6:
			self.lin_vel = 1.6
		#self.lin_vel = 1
		#if action == 1:
		#	self.ang_vel = 4
		#elif action == 2:
		#	self.ang_vel = -4
		self._publish_vel()

	def _take_action_distribute(self):
		lin_vel_add = self.pi[self.index][1] * 0.2 +  self.pi[self.index][2] * 0.2 +  self.pi[self.index][3] * 0.2 - \
				 self.pi[self.index][6] * 0.2 - self.pi[self.index][7] * 0.2 - self.pi[self.index][8] * 0.2
		self.ang_vel = 0.9 * (4 * self.pi[self.index][1] + 4* self.pi[self.index][4] + 4* self.pi[self.index][6] - \
				4 * self.pi[self.index][3] - 4 * self.pi[self.index][5] - 4 * self.pi[self.index][8])
		self.lin_vel += lin_vel_add
		if self.lin_vel < 0:
			self.lin_vel = 0
		if self.lin_vel > 1.6 * 0.88:
			self.lin_vel = 1.6 * 0.88
		self._publish_vel()

	def _process_image(self, msg):
		self.cnt += 1
		#if self.cnt % 2 != 0:
		#	return
		self._pause_simulation()
		torch.cuda.synchronize()
		if self.hasR:
			self.index = 0
			self.curr_idx = self.respawn_idx[self.respawn_loc]
			self.hasR = False
		if self.hit:
			self._respawn()
			self.sign_pub.publish(str('nootnoot,multi21,0,NA'))
			self.still = 0
			self.d[self.index-1] = 1
			self.r[self.index-1] = -0.04
		elif self._check_offroad() or self.still > 22:
			self._respawn()
			self.sign_pub.publish(str('nootnoot,multi21,0,NA'))
			self.still = 0
			self.d[self.index-1] = 1
			self.r[self.index-1] = 0.0
		if (self.curr_idx == len(self.points)-1) or (self.pos[1] > 0.03215 and self.pos[0] < 0.9):
			self.sign_pub.publish(str('nootnoot,multi21,-1,NA'))
			self._respawn()
			self.still = 0
			self.d[self.index-1] = 1
			self.r[self.index-1] = 0.5

		if (self.cnt % 10 == 0):
			print(self.cnt)
		if (self.cnt % 100 == 0):
			print("---")
			print(self.curr_idx)
			print(np.max(self.pi[self.index-1]))
			print(self.pi[self.index-1])
			print(self.alpha)
			print("---")

		if self.hasR == False:
			if (self.rwd != -1):
				self._find_closest_point()

			if (self.rwd != 0):
				self.still = 0
			else:
				self.still += 1

			bridge = CvBridge()
			cv_image = cv2.cvtColor(bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"), cv2.COLOR_BGR2GRAY)
			#cv2.imshow('Vision', cv_image)
			#cv2.waitKey(1)

			torch_image = torch.tensor(cv_image, dtype=torch.float32)
			torch_image_batched = torch.unsqueeze(torch_image, dim=0)
			result = nn.functional.avg_pool2d(torch_image_batched, kernel_size=(2,2))
			if self.channels == 1:
				result = torch.unsqueeze(result, dim=1)
			self.s[self.index] = result[0].numpy() / 128.0 - 1

			#cv2.imshow('Vision', result[0][0].numpy().astype(np.uint8))
			#cv2.waitKey(1)

			self.s_vel[self.index] = np.array([self.lin_vel * 2.0 / 2.5 - 1, self.ang_vel / 4.0])

			self.s_tf.copy_(torch.unsqueeze(torch.tensor(self.s[self.index]), dim=0))
			self.s_vel_tf.copy_(torch.unsqueeze(torch.tensor(self.s_vel[self.index]), dim=0))

			#start_time = time.time()
			with torch.no_grad():
				predict = self.models[0](self.s_tf, self.s_vel_tf)
				self.pi[self.index] = predict.cpu().detach().numpy()
				del predict
			#end_time = time.time()
			no_nans = True
			for i in range(len(self.pi[self.index])):
				if np.isnan(self.pi[self.index][i]):
					self.pi[self.index][i] = 0

			if np.random.rand() < 0.03:
				self.a[self.index] = np.random.randint(0,9)
			else:
				self.a[self.index] = np.random.choice(len(self.pi[self.index]), p=self.pi[self.index])
			#self.a[self.index] = np.argmax(self.pi[self.index])

			#if self.index == self.batch_size:
			#	self.s_tf_batch.copy_(torch.tensor(self.s[0:self.index+1]))
			#	self.s_vel_tf_batch.copy_(torch.tensor(self.s_vel[0:self.index+1]))

			#	with torch.no_grad():
			#		qt1 = self.models[3](self.s_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
			#		qt2 = self.models[4](self.s_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
			#		q_min = torch.min(qt1, qt2).cpu().detach().numpy()
			#		print(q_min)
			#		self.a[self.index] = np.argmax(q_min)
			#	time.sleep(0.5)
			#else:
			#	self.a[self.index] = np.argmax(self.pi[self.index])
			self._take_action()

			if self.index != 0:
				self.r[self.index-1] = self.rwd
				self.d[self.index-1] = 0

				if self.a[self.index-1] == 1 or self.a[self.index-1] == 3 or self.a[self.index-1] == 4 \
					or self.a[self.index-1] == 5 or self.a[self.index-1] == 6 or self.a[self.index-1] == 8:
					self.r[self.index-1] /= 2.5

		if self.index == self.batch_size:

			#self.s_tf_batch.copy_(torch.tensor(self.s[0:self.index+1]))
			#self.s_vel_tf_batch.copy_(torch.tensor(self.s_vel[0:self.index+1]))

			#with torch.no_grad():
			#	qt1 = self.models[3](self.s_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
			#	qt2 = self.models[4](self.s_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
			#	q_min = torch.min(qt1, qt2)
			#	print("--")
			#	print(self.r[self.index-1])
			#	print(self.d[self.index-1])
			#	print(q_min)
			#	print(self.pi[self.index-1])
			#	print(self._critic_loss(q_min))
			#	time.sleep(1)
			#	cv2.imshow('idxm1', ((self.s[self.index-1][0] + 1) * 128.0).astype(np.uint8))
			#	cv2.waitKey(1)
			#	cv2.imshow('idx', ((self.s[self.index][0]+1) * 128.0).astype(np.uint8))
			#	cv2.waitKey(1)
			#	time.sleep(2)
			#	del q_min
			#	del qt1
			#	del qt2

			#self.index=0
			#self.s[0] = self.s[self.batch_size]
			#self.a[0] = self.a[self.batch_size]
			#self.pi[0] = self.pi[self.batch_size]
			#self.s_vel[0] = self.s_vel[self.batch_size]
			self._train()

		self.index += 1
		self.rwd = 0
		#print(f"{end_time - start_time:.4f}")
		torch.cuda.empty_cache()
		self._unpause_simulation()

	def _pause_simulation(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		pause_physics()

	def _unpause_simulation(self):
		rospy.wait_for_service('/gazebo/unpause_physics')
		unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		unpause_physics()

	def _train(self):
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
		#print(torch.cuda.memory_allocated())
		#print(torch.cuda.memory_summary())
		# Critic Loss

		#print(self.s[0].shape)
		#img = (self.s[0][0] + 1) * 128
		#img2 = img.astype(np.uint8)
		#cv2.imshow('Vision', img2)
		#cv2.waitKey(1)
		#print("action")
		#print(self.a[0])
		#print("reward")
		#print(self.r[0])
		#print("terminal")
		#print(self.d[0])
		#print("vel")
		#print(self.s_vel[0])
		#print("----")

		self.s_tf_batch.copy_(torch.tensor(self.s[0:self.index+1]))
		self.s_vel_tf_batch.copy_(torch.tensor(self.s_vel[0:self.index+1]))

		for i in range(1,3):
			#TD Lambda
			for p in self.models[i].parameters():
				if p.grad is not None:
					p.grad *= self.q_lambda

			q = self.models[i](self.s_tf_batch, self.s_vel_tf_batch)
			loss = torch.mean(q[0][self.a[0:self.index]])
			loss.backward()

			with torch.no_grad():
				delta = self._critic_loss(q)

			for p in self.models[i].parameters():
				if p.grad is not None:
					p.grad *= delta

			self.optimizers[i].step()

			if self.d[0] == 1:
				self.optimizers[i].zero_grad()
			else:
				for p in self.models[i].parameters():
					if p.grad is not None:
						p.grad /= delta
			del delta

			#TD(0)
			#q = self.models[i](self.s_tf_batch, self.s_vel_tf_batch)
			#loss = self._critic_loss(q)
			#self.optimizers[i].zero_grad()
			#loss.backward()
			#if self.qcnt[i-1] % 32 == 0:
			#	for p in self.models[i].parameters():
			#		if p.grad is not None:
			#			p.grad /= 32.0
			#self.optimizers[i].step()

			del q
			del loss

		# Actor Loss
		#if np.random.rand() < 1.0:
		#	self.acnt += 1
		with torch.no_grad():
			qt1 = self.models[3](self.s_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
			qt2 = self.models[4](self.s_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
			q_min = torch.min(qt1, qt2)
			if self.cnt % 100 == 0:
				print(q_min)
				print(qt1)
				print(qt2)
			#if (self.epoch % 50 == 0):
			#	print("A")
			#	print(q_min)
			#	print(qt2)
			del qt1
			del qt2

		action = self.models[0](self.s_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
		loss = self._actor_loss(action, q_min)

		self.optimizers[0].zero_grad()
		loss.backward() # actor
		self.optimizers[0].step()
		#if self.acnt % 32 == 0:
		#	for p in self.models[0].parameters():
		#		if p.grad is not None:
		#			p.grad /= 32.0
		#	self.optimizers[0].step()
		#	self.optimizers[0].zero_grad()

		del action
		del q_min

		# Temperature Loss (alpha)

		loss = self._temperature_loss()

		self.optimizers[3].zero_grad()
		loss.backward() # alpha
		self.optimizers[3].step()

		self.alpha = torch.exp(self.log_alpha).detach()

		del loss

		# Update Qt
		for i in range(1,3):
			source_params = self.models[i].parameters()
			target_params = self.models[i+2].parameters()
			updated_params = [
			    (self.tau * source_param.data) + ((1 - self.tau) * target_param.data)
		  	  for source_param, target_param in zip(source_params, target_params)
			]

			for target_param, updated_param in zip(self.models[i+2].parameters(), updated_params):
				target_param.data.copy_(updated_param)

		del updated_params
		del source_params
		del target_params

		self.s[0] = self.s[self.batch_size]
		self.a[0] = self.a[self.batch_size]
		self.pi[0] = self.pi[self.batch_size]
		self.s_vel[0] = self.s_vel[self.batch_size]
		self.index = 0

		if (self.epoch) % 200 == 50:
			for i in range(len(self.models)):
				torch.save(self.models[i].state_dict(), self.weights_folder[i])
		self.epoch += 1

		torch.cuda.synchronize()
		torch.cuda.empty_cache()
		#print("D")
		#print(torch.cuda.memory_allocated())
		#print(torch.cuda.memory_summary())

		#self._unpause_simulation()

	def _critic_loss(self, q):
		pi_c = torch.tensor(self.pi[1:self.index+1], requires_grad=False).cuda()
		r = torch.tensor(self.r[0:self.index], requires_grad=False).cuda()
		d = torch.tensor(self.d[0:self.index], requires_grad=False).cuda()
		ga = torch.tensor(self.gamma, device='cuda', requires_grad=False)
		qNext = q[1:self.index+1].detach()
		log_pi_c = torch.log(pi_c)
		alpha_cuda = self.alpha.cuda()

		#TD(0)
		#target_q_values = q[0:self.index].detach().clone()
		#---vals = r + torch.logical_not(d) * ga * torch.sum(pi_c * (qNext-alpha_cuda * log_pi_c), dim=-1)
		#target_q_values[torch.arange(target_q_values.size(0)), self.a[0:self.index]] = vals
		#---loss = 0.5 * (10 * (q[0][self.a[0:self.index]] - vals))**2
		#del target_q_values

		#TD(l)
		vals = r + torch.logical_not(d) * ga * torch.sum(pi_c * (qNext-alpha_cuda * log_pi_c), dim=-1)
		loss = 10 * torch.mean(q[0][self.a[0:self.index]] - vals)

		del pi_c
		del r
		del ga
		del log_pi_c
		del alpha_cuda
		return loss

	def _actor_loss(self, action, q_min):
		alpha_cuda = self.alpha.cuda()
		log_term = alpha_cuda * torch.log(action)
		elementwise_product = action * (log_term - q_min)
		loss = torch.mean(elementwise_product)

		del log_term
		del elementwise_product

		return loss

	def _temperature_loss(self):
		log_pi = torch.log(torch.tensor(self.pi[0:self.index], requires_grad=False))
		loss = torch.sum(torch.tensor(self.pi[0:self.index], requires_grad=False) * (-self.log_alpha * (log_pi + self.H)))

		del log_pi

		return loss

	def _record_points(self):
		rospy.wait_for_service('/gazebo/get_model_state')
		get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		global_state = get_model_state(model_name='R1')
		global_coord = [global_state.pose.position.x, global_state.pose.position.y, global_state.pose.position.z]
		with open('path_30.txt', 'a') as file:
			file.write(' '.join(map(str, global_coord)) + '\n')

	def _respawn(self):
		msg = ModelState()
		msg.model_name = 'R1'

		random_num = np.random.rand()
		if random_num < 0.05:
			#up
			msg.pose.position.x = 5.493696
			msg.pose.position.y = 2.472393
			msg.pose.position.z = 0.0525
			msg.pose.orientation.y = 0
			msg.pose.orientation.x = 0
			msg.pose.orientation.z = 0.707
			msg.pose.orientation.w = -0.707
			self.respawn_loc = 0
		elif random_num < 0.95:
			#down
			msg.pose.position.x = 4.571
			msg.pose.position.y = -2.14113
			msg.pose.position.z = 0.0525
			msg.pose.orientation.y = 0
			msg.pose.orientation.x = 0
			msg.pose.orientation.z = 0.707
			msg.pose.orientation.w = 0.707
			self.respawn_loc = 2
		else:
			#left
			msg.pose.position.x = 4.80
			msg.pose.position.y = -0.3
			msg.pose.position.z = 0.0525
			msg.pose.orientation.y = 0
			msg.pose.orientation.x = 0
			msg.pose.orientation.z = 0
			msg.pose.orientation.w = -1.0
			self.respawn_loc = 1
		#else:
			#right
			#msg.pose.position.x = 2.50
			#msg.pose.position.y = -0.31
			#msg.pose.position.z = 0.0525
			#msg.pose.orientation.y = 0
			#msg.pose.orientation.x = 0
			#msg.pose.orientation.z = 1.0
			#msg.pose.orientation.w = 0
			#self.respawn_loc = 3
		#	else:
		#	msg.pose.position.x = 2.50
		#	msg.pose.position.y = -2.1
		#	msg.pose.position.z = 0.0525
		#	msg.pose.orientation.y = 0
		#	msg.pose.orientation.x = 0
		#	msg.pose.orientation.z = 1.0
		#	msg.pose.orientation.w = 0
		#	self.respawn_loc = 3

		self.lin_vel = 0.1 + np.random.randint(0,3) * 0.2
		#self.lin_vel = 0.4
		self.ang_vel = 0
		self._publish_vel()

		rospy.wait_for_service('/gazebo/set_model_state')
		set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		resp = set_state(msg)
		self.hasR = True
		self.hit = False

	def run(self):
		rospy.init_node('adeept_awr_driver')
		rate = rospy.Rate(10)

		rospy.Subscriber('/isHit', Bool, self._check_collision)
		rospy.Subscriber('R1/pi_camera/image_raw', Image, self._process_image)

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

		self.lin_vel = 0.2 #+ np.random.randint(0,4) * 0.2
		self.ang_vel = 0
		self._publish_vel()

		rospy.wait_for_service('/gazebo/set_model_state')
		set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		resp = set_state(msg)

		rospy.spin()

if __name__=='__main__':
	dr = Controller()
	dr.run()
