#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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
		self.maxpool1_frame = nn.MaxPool2d(kernel_size=(3,3), stride=(3,3))
		self.conv2_frame = nn.Conv2d(64, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		#self.maxpool2_frame = nn.MaxPool2d((3,5), stride=(2,2))
		self.conv3_frame = nn.Conv2d(32, 8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		self.maxpool3_frame = nn.MaxPool2d((3,3), stride=(3,3))

		self.conv1_diff = nn.Conv2d(self.channels, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		self.maxpool1_diff = nn.MaxPool2d(kernel_size=(3,3), stride=(3,3))
		self.conv2_diff = nn.Conv2d(32, 16, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		#self.maxpool2_diff = nn.MaxPool2d((3,3), stride=(2,2))
		self.conv3_diff = nn.Conv2d(16, 8, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		self.maxpool3_diff = nn.MaxPool2d((3,3), stride=(3,3))

		self.dense1_diff = nn.Linear(22720, 128)
		self.dense2_diff = nn.Linear(128, 32)
		self.dense3_diff = nn.Linear(32, 5)

		self.dense1 = nn.Linear(22727, 256)
		self.dense2 = nn.Linear(256, 128)
		self.output = nn.Linear(128,18)

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
		if isinstance(module, (nn.Linear, nn.Conv2d)):
			init.normal_(module.weight.data, mean=0.0,std=0.1)
		if self.i == 0 and module == self.output:
			init.normal_(module.weight.data, mean=0.0, std=0.0001)

	def forward(self, input_frame, input_diff, numeric_input):
		#with profiler.profile(record_shapes=True) as prof:
			#x_frame = self.flatten(self.maxpool3_frame(nn.ReLU()(self.conv3_frame(\
			#	  self.maxpool2_frame(nn.ReLU()(self.conv2_frame(self.maxpool1_frame(\
			#	  nn.ReLU()(self.conv1_frame(input_frame))))))))))

		x_frame = self.flatten(self.batchnorm3(self.maxpool3_frame(F.relu(self.conv3_frame(\
			  self.batchnorm2(F.relu(self.conv2_frame(self.batchnorm1(self.maxpool1_frame(\
			  F.relu(self.conv1_frame(input_frame)))))))))))).cuda()

			#x_diff_im = self.flatten(self.maxpool3_diff(nn.ReLU()(self.conv3_diff(\
			#	  self.maxpool2_diff(nn.ReLU()(self.conv2_diff(self.maxpool1_diff(\
			#	  nn.ReLU()(self.conv1_diff(input_diff))))))))))

		x_diff_im = self.flatten(self.batchnorm3d(self.maxpool3_diff(F.relu(self.conv3_diff(\
			  self.batchnorm2d(F.relu(self.conv2_diff(self.batchnorm1d(self.maxpool1_diff(\
			  F.relu(self.conv1_diff(input_diff)))))))))))).cuda()

		x_diff = nn.Tanh()(self.dense3_diff(nn.Tanh()(self.dense2_diff(nn.Tanh()(self.dense1_diff(x_diff_im)))))).cuda()

		x_combined = self.flatten(torch.cat([x_frame, x_diff, numeric_input], dim=1)).cuda()

		del x_diff
		del x_diff_im
		del x_frame

		x = self.output(nn.Tanh()(self.dense2(nn.Tanh()(self.dense1(x_combined))))).cuda()
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
			state_dict = torch.load(self.weights_folder[i])
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

		image_path = 'binaryMap2023.png'
		path = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		self.binaryMap = (path == 255)
		self.reward = -0.01
		self.pos = np.array([0,0,0])
		self.curr_idx = 0

		points_arr = np.loadtxt('path.txt')
		self.points = points_arr.reshape(-1,3)
		#self._draw_points()

		# INITALIZE NEURAL NETWORKS

		# Policy network
		# actions are: +0.1, +1, + 10, + 100, -0.1, -1, -10, -100, (set 0) for both velocity and angular velocity,
		# so |A| = 18
		self.H = 0.9 * (- np.log(1 / 18.0))

		self.channels = 1
		self.input_shape = (self.channels, 360, 640)
		# pi, Q1, Q2, Qt1, Qt2
		self.models = nn.ModuleList()
		self.weights_folder = ['weights_torch/pi.h5', 'weights_torch/Q1.h5', 'weights_torch/Q2.h5', 'weights_torch/Qt1.h5', 'weights_torch/Qt2.h5']
		self._initalize_models()

		self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=False)
		self.batch_size = 1
		self.gamma = 0.99
		self.tau = 0.02
		#Replay buffer
		self.s = np.zeros(shape=(self.batch_size+1, self.channels, 360,640), dtype=np.float32)
		self.s_diff = np.zeros(shape=(self.batch_size+1, self.channels, 360, 640), dtype=np.float32)
		self.s_vel = np.zeros(shape=(self.batch_size+1, 2), dtype=np.float32)
		self.pi = np.zeros(shape=(self.batch_size+1, 18), dtype=np.float32)
		self.a = np.zeros(shape=(self.batch_size+1,), dtype=np.int8)
		self.r = np.zeros(shape=(self.batch_size+1,), dtype=np.float32)
		self.d = np.zeros(shape=(self.batch_size+1,), dtype=bool)
		#self.q = np.zeros(shape=(self.batch_size+1,), dtype=np.float32)
		self.index = 0
		self.start = True

		self.still = 0

		optimizer_phi = optim.Adagrad(self.models[0].parameters(), lr=0.01)
		optimizer_q1 = optim.Adagrad(self.models[1].parameters(), lr=0.01)
		optimizer_q2 = optim.Adagrad(self.models[2].parameters(), lr=0.01)
		optimizer_alpha = optim.Adagrad([self.alpha], lr=0.01)
		self.optimizers = [optimizer_phi, optimizer_q1, optimizer_q2, optimizer_alpha]

		self.s_tf = torch.zeros(size=(1,self.channels,360,640), dtype=torch.float32).cuda()
		self.s_diff_tf = torch.zeros(size=(1,self.channels,360,640), dtype=torch.float32).cuda()
		self.s_vel_tf = torch.zeros(size=(1,2), dtype=torch.float32).cuda()

		self.s_tf_batch = torch.zeros(size=(self.batch_size+1, self.channels, 360, 640), dtype=torch.float32).cuda()
		self.s_diff_tf_batch = torch.zeros(size=(self.batch_size+1, self.channels, 360, 640), dtype=torch.float32).cuda()
		self.s_vel_tf_batch = torch.zeros(size=(self.batch_size+1, 2), dtype=torch.float32).cuda()

		self.cnt = 0
		self.rwd = 0

		print(torch.cuda.memory_summary())

	def _draw_points(self):
		self.bitmapPos = cv2.imread('binaryMap2023.png', cv2.IMREAD_COLOR)
		for p in self.points:
			x,y,z = p
			xy = [x,y]
			pxl = self._cord_to_pxl(xy)
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
			#self.bitmapPos[wheelPxl[0]][wheelPxl[1]][0] = 255
			#self.bitmapPos[wheelPxl[0]][wheelPxl[1]][1] = 0
			#self.bitmapPos[wheelPxl[0]][wheelPxl[1]][2] = 0

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
			self._respawn()

	def _publish_vel(self):
		tw = Twist()
		tw.linear.x = self.lin_vel
		tw.angular.z = self.ang_vel
		self.twist_pub.publish(tw)

	def _find_closest_point(self):
		maxI = self.curr_idx
		d_curr = np.sum((self.points[self.curr_idx] - self.pos)**2)
		passed = False
		for i in range(self.curr_idx + 1, min(self.curr_idx + 6, len(self.points))):
			dp = np.sum((self.points[i] - self.pos)**2)
			if (dp < d_curr):
				passed = True
				maxI = i
				d_curr = dp
		if (passed):
			self.reward = 0.01 * (maxI - self.curr_idx)
			self.curr_idx = maxI
			self.rwd += self.reward

	def _check_terminal_state(self):
		if 1.753029 < self.pos[2] < 1.989973 and -1.791982 < self.pos[0] < -0.886508 and -0.36789 < self.pos[1] < -0.028055:
			return 1
		return 0

	def _take_action(self):
		action = self.a[self.index]
		if action == 0:
			self.lin_vel = 0
		elif action == 1:
			self.lin_vel += 0.1
		elif action == 2:
			self.lin_vel += 1
		elif action == 3:
			self.lin_vel += 10
		elif action == 4:
			self.lin_vel += 100
		elif action == 5:
			self.lin_vel -= 0.1
		elif action == 6:
			self.lin_vel -= 1
		elif action == 7:
			self.lin_vel -= 10
		elif action == 8:
			self.lin_vel -= 100
		elif action == 9:
			self.ang_vel = 0
		elif action == 10:
			self.ang_vel += 0.1
		elif action == 11:
			self.ang_vel += 1
		elif action == 12:
			self.ang_vel += 10
		elif action == 13:
			self.ang_vel += 100
		elif action == 14:
			self.ang_vel -= 0.1
		elif action == 15:
			self.ang_vel -= 1
		elif action == 16:
			self.ang_vel -= 10
		elif action == 17:
			self.ang_vel -= 100
		self._publish_vel()

	def _process_image(self, msg):
		self._pause_simulation()
		self.cnt += 1
		if (self.cnt % 10 == 0):
			print(self.cnt)
		if (self.cnt % 50 == 0):
			print("---")
			print(self.rwd)
			self.rwd = 0
		if (self.reward != -1):
			self._find_closest_point()
		if (self.reward > 0):
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
		self.s[self.index] = result[0].numpy() / 256.0
		self.d[self.index] = self._check_terminal_state()
		if self.start:
			self.s_diff[self.index] = np.zeros(shape=(self.input_shape))
			self.start = False
		else:
			self.r[self.index-1] = self.reward
			self.s_diff[self.index] = np.abs(self.s[self.index] - self.s[self.index-1])

		self.s_vel[self.index] = np.array([self.lin_vel, self.ang_vel])

		if (self.d[self.index] == 1):
			self.r[self.index] = 0
			self._train(True)

		self.s_tf.copy_(torch.unsqueeze(torch.tensor(self.s[self.index]), dim=0))
		self.s_diff_tf.copy_(torch.unsqueeze(torch.tensor(self.s_diff[self.index]), dim=0))
		self.s_vel_tf.copy_(torch.unsqueeze(torch.tensor(self.s_vel[self.index]), dim=0))

		#start_time = time.time()
		with torch.no_grad():
			predict = self.models[0](self.s_tf, self.s_diff_tf, self.s_vel_tf)
			self.pi[self.index] = predict.cpu().detach().numpy()
			del predict
		#end_time = time.time()

		self.a[self.index] = np.random.choice(len(self.pi[self.index]), p=self.pi[self.index])
		self._take_action()

		if self.index == self.batch_size:
			self.index=0
			#self._train()

		self.index += 1
		self.reward = -0.01
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

	def _train(self, terminal=False):
		torch.cuda.synchronize()
		torch.cuda.empty_cache()
		#print(torch.cuda.memory_allocated())
		#print(torch.cuda.memory_summary())
		i = 1
		# Critic Loss
		if np.random.rand() < 0.5:
			i=2

		self.s_tf_batch.copy_(torch.tensor(self.s[0:self.index+1]))
		self.s_diff_tf_batch.copy_(torch.tensor(self.s_diff[0:self.index+1]))
		self.s_vel_tf_batch.copy_(torch.tensor(self.s_vel[0:self.index+1]))

		#torch.cuda.synchronize()
		#torch.cuda.empty_cache()
		#print("A")
		#print(torch.cuda.memory_allocated())

		q = self.models[i](self.s_tf_batch, self.s_diff_tf_batch, self.s_vel_tf_batch)
		loss = self._critic_loss(q)

		self.optimizers[i].zero_grad()

		#torch.cuda.synchronize()
		#torch.cuda.empty_cache()
		#print("B1")
		#print(torch.cuda.memory_allocated())

		loss.backward() # Q

		#torch.cuda.synchronize()
		#torch.cuda.empty_cache()
		#print("B2")
		#print(torch.cuda.memory_allocated())

		self.optimizers[i].step()

		del q
		del loss

		#torch.cuda.synchronize()
		#torch.cuda.empty_cache()
		#print("C")
		#print(torch.cuda.memory_allocated())

		if terminal:
			terminalQ = self.models[i](torch.tensor([self.s[self.index]]).cuda(), torch.tensor([self.s_diff[self.index]]).cuda(), torch.tensor([self.s_vel[self.index]]).cuda())
			loss = torch.mean(terminalQ**2)

			self.optimizers[i].zero_grad()
			loss.backward()
			self.optimizers[i].step()
			del terminalQ

		# Actor Loss
		with torch.no_grad():
			qt1 = self.models[3](self.s_tf_batch[0:self.index], self.s_diff_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
			qt2 = self.models[4](self.s_tf_batch[0:self.index], self.s_diff_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
			q_min = torch.min(qt1, qt2)
			del qt1
			del qt2

		action = self.models[0](self.s_tf_batch[0:self.index], self.s_diff_tf_batch[0:self.index], self.s_vel_tf_batch[0:self.index])
		loss = self._actor_loss(action, q_min)

		self.optimizers[0].zero_grad()
		loss.backward()
		self.optimizers[0].step()

		del action
		del q_min

		# Temperature Loss (alpha)

		self.alpha.requires_grad = True

		loss = self._temperature_loss()

		self.optimizers[3].zero_grad()
		loss.backward()
		self.optimizers[3].step()

		self.alpha.requires_grad = False

		# Update Qt
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
		self.s_diff[0] = self.s_diff[self.batch_size]
		self.s_vel[0] = self.s_vel[self.batch_size]
		self.d[0] = self.d[self.batch_size]
		self.index = 0

		if (terminal):
			self._respawn()

		if (self.epoch) % 200 == 150:
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
		ga = torch.tensor(self.gamma, device='cuda', requires_grad=False)
		one_hot = torch.tensor(np.eye(18)[self.a[0:self.index]]).to(q.dtype).cuda()
		log_pi_c = torch.log(pi_c)
		alpha_cuda = self.alpha.cuda()
		target_q_values = r + ga * torch.sum(pi_c * (q[1:self.index+1]-alpha_cuda * log_pi_c), dim=-1)
		selected_q_values = torch.sum(q[0:self.index] * one_hot, dim=-1)
		loss = 0.5 * torch.mean((selected_q_values - target_q_values)**2)
		del pi_c
		del target_q_values
		del selected_q_values
		del r
		del ga
		del log_pi_c
		del one_hot
		del alpha_cuda

		return loss

	def _actor_loss(self, action, q_min):
		#sum = 0
		#for i in range(self.batch_size):
		#	sum += tf.reduce_sum(action[i] * (self.alpha * tf.math.log(action[i]) - self.q[i]))
		#return sum
		alpha_cuda = self.alpha.cuda()
		log_term = alpha_cuda * torch.log(action)
		elementwise_product = action * (log_term - q_min)
		loss = torch.sum(elementwise_product)

		del log_term
		del elementwise_product

		return loss

	def _temperature_loss(self):
		log_pi = torch.log(torch.tensor(self.pi[0:self.index]))
		loss = torch.sum(torch.tensor(self.pi[0:self.index]) * (-self.alpha * (log_pi + self.H)))

		del log_pi

		return loss

	def _record_points(self):
		rospy.wait_for_service('/gazebo/get_model_state')
		get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
		global_state = get_model_state(model_name='R1')
		global_coord = [global_state.pose.position.x, global_state.pose.position.y, global_state.pose.position.z]
		self.points = np.append(self.points, global_coord)

	def _respawn(self):
		msg = ModelState()
		msg.model_name = 'R1'

		msg.pose.position.x = 5.493696
		msg.pose.position.y = 2.472393
		msg.pose.position.z = 0.0525
		msg.pose.orientation.y = 0
		msg.pose.orientation.x = 0
		msg.pose.orientation.z = 0.707
		msg.pose.orientation.w = -0.707

		self.lin_vel = 0
		self.ang_vel = 0
		self._publish_vel()

		rospy.wait_for_service('/gazebo/set_model_state')
		set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
		resp = set_state(msg)
		self.reward = -1
		self.curr_idx = 0

	def run(self):
		rospy.init_node('adeept_awr_driver')
		rate = rospy.Rate(10)

		rospy.Subscriber('/isHit', Bool, self._check_collision)
		rospy.Subscriber('R1/pi_camera/image_raw', Image, self._process_image)

		self.sign_pub.publish(str('TeamRed,multi21,0,JEDIS'))
		while not rospy.is_shutdown():
			#self._publish_vel()
			if (self._check_offroad() or self.still > 20):
				self._respawn()
				self.still = 0

if __name__=='__main__':
	dr = Controller()
	dr.run()
