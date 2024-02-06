#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
import tensorflow as tf
import time

from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from std_srvs.srv import Empty
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from gazebo_msgs.srv import GetModelState, SetModelState, GetLinkState
from gazebo_msgs.msg import ModelState, ContactsState
from tensorflow.keras import layers, models, initializers

class Controller():

	def __init__(self):

		physical_devices = tf.config.list_physical_devices('GPU')
		if not physical_devices:
			print("No GPU devices available.")
		else:
			for device in physical_devices:
				print(f"GPU device name: {device.name}")

		#tf.profiler.experimental.start('logdir')
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

		self.average_pooling_layer = layers.AveragePooling2D(pool_size=(2,2))
		self.channels = 1
		self.input_shape = (360, 640,self.channels)
		# pi, Q1, Q2, Qt1, Qt2
		self.models = []
		self.weights_folder = ['weights/pi.h5', 'weights/Q1.h5', 'weights/Q2.h5', 'weights/Qt1.h5', 'weights/Qt2.h5']
		custom_initalizer = initializers.TruncatedNormal(mean=0.0, stddev=0.1)
		for i in range(5):
			input_frame = layers.Input(shape=self.input_shape)
			input_diff = layers.Input(shape=self.input_shape)
			#conv: (3,3), 64, 32, 8
			#pool (3,3),(2,2)
			conv1_frame = layers.Conv2D(64, (3,3), activation='relu')(input_frame)
			maxpool1_frame = layers.MaxPooling2D((3,3), strides=(2,2))(conv1_frame)
			conv2_frame = layers.Conv2D(32, (3,3), activation='relu')(maxpool1_frame)
			maxpool2_frame = layers.MaxPooling2D((3,3), strides=(2,2))(conv2_frame)
			conv3_frame = layers.Conv2D(8, (3,3), activation='relu')(maxpool2_frame)
			maxpool3_frame = layers.MaxPooling2D((3,3), strides=(2,2))(conv3_frame)
			flatten_frame = layers.Flatten()(maxpool3_frame)

			#256, 50, 5
			conv1_diff = layers.Conv2D(64, (3,3), activation='relu')(input_diff)
			maxpool1_diff = layers.MaxPooling2D((3,3), strides=(2,2))(conv1_diff)
			conv2_diff = layers.Conv2D(32, (3,3), activation='relu')(maxpool1_diff)
			maxpool2_diff = layers.MaxPooling2D((3,3), strides=(2,2))(conv2_diff)
			conv3_diff = layers.Conv2D(8, (3,3), activation='relu')(maxpool2_diff)
			maxpool3_diff = layers.MaxPooling2D((3,3), strides=(2,2))(conv3_diff)
			dense1_diff = layers.Dense(256, activation='tanh')(maxpool3_diff)
			dense2_diff = layers.Dense(50, activation='tanh')(dense1_diff)
			dense3_diff = layers.Dense(5, activation='tanh')(dense2_diff)
			flatten_diff = layers.Flatten()(dense3_diff)

			if i == 0:
				numeric_input = layers.Input(shape=(2,))

				combined = layers.concatenate([flatten_frame, flatten_diff, numeric_input])
				#256, 100, 18
				dense1 = layers.Dense(256, activation='tanh')(combined)
				dense2 = layers.Dense(100, activation='tanh')(dense1)

				output = layers.Dense(18, activation='softmax')(dense2)
				pi = tf.keras.Model(inputs=[input_frame, input_diff, numeric_input], outputs=output)
				for layer in pi.layers:
					if hasattr(layer, 'kernel_initializer'):
						layer.kernel_initializer = custom_initalizer
				output_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.0001)
				pi.layers[-1].kernel_initializer = output_initializer

				self.models.append(pi)
			else:
				numeric_input = layers.Input(shape=(2,))
				combined = layers.concatenate([flatten_frame, flatten_diff, numeric_input])
				dense1 = layers.Dense(256, activation='tanh')(combined)
				dense2 = layers.Dense(100,activation='tanh')(dense1)
				output = layers.Dense(18, activation='linear')(dense2)
				Q = tf.keras.Model(inputs=[input_frame, input_diff, numeric_input], outputs=output)
				if i == 1 or i == 2:
					for layer in Q.layers:
						if hasattr(layer, 'kernel_initalizer'):
							layer.kernel_initializer = custom_initalizer
				else:
					Q.set_weights(self.models[i-2].get_weights())
				self.models.append(Q)

		self.alpha = tf.Variable(initial_value=0.03, trainable=True, dtype=tf.float32)
		self.batch_size = 16
		self.gamma = 0.99
		self.tau = 0.02
		self.models[0].summary()
		#Replay buffer
		self.s = np.zeros(shape=(self.batch_size+1,360,640,self.channels), dtype=np.float32)
		self.s_diff = np.zeros(shape=(self.batch_size+1, 360, 640, self.channels), dtype=np.float32)
		self.s_vel = np.zeros(shape=(self.batch_size+1, 2), dtype=np.float32)
		self.pi = np.zeros(shape=(self.batch_size+1, 18), dtype=np.float32)
		self.a = np.zeros(shape=(self.batch_size+1,), dtype=np.int8)
		self.r = np.zeros(shape=(self.batch_size+1,), dtype=np.float32)
		self.d = np.zeros(shape=(self.batch_size+1,), dtype=np.bool)
		#self.q = np.zeros(shape=(self.batch_size+1,), dtype=np.float32)
		self.index = 0
		self.start = True

		self.still = 0

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

		tf_image = tf.convert_to_tensor(cv_image, dtype=tf.float32)
		tf_image_batched = tf.expand_dims(tf_image, axis=0)
		if self.channels == 1:
			tf_image_batched = tf.expand_dims(tf_image_batched, axis=-1)
		result = layers.AveragePooling2D(pool_size=(2,2))(tf_image_batched)
		self.s[self.index] = np.array(result[0]) / 256.0
		#self.s[self.index] = self.average_pooling_layer([tf.convert_to_tensor(cv_image, dtype=tf.float32)])[0] / 256.0
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

		start_time = time.time()
		s_tf = tf.convert_to_tensor(self.s[self.index])
		s_tf = tf.expand_dims(s_tf, axis=0)

		s_diff_tf = tf.convert_to_tensor(self.s_diff[self.index])
		s_diff_tf = tf.expand_dims(s_diff_tf,axis=0)

		s_vel_tf = tf.convert_to_tensor(self.s_vel[self.index])
		s_vel_tf = tf.expand_dims(s_vel_tf, axis=0)
		fp_time = time.time()
		self.pi[self.index] = np.array(self.models[0]([s_tf, s_diff_tf, s_vel_tf]))
		fp_time_e = time.time()
		self.a[self.index] = np.random.choice(len(self.pi[self.index]), p=self.pi[self.index])
		self._take_action()

		if self.index == self.batch_size:
			self._train()

		self.index += 1
		self.reward = -0.01
		print(f"{fp_time_e - fp_time:.4f}")
		self._unpause_simulation()
		#if self.index == 10:
		#	tf.profiler.experimental.stop()

	def _pause_simulation(self):
		rospy.wait_for_service('/gazebo/pause_physics')
		pause_physics = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
		pause_physics()

	def _unpause_simulation(self):
		rospy.wait_for_service('/gazebo/unpause_physics')
		unpause_physics = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
		unpause_physics()

	def _train(self, terminal=False):
		#self._pause_simulation()
		optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
		i = 1
		# Critic Loss
		if tf.random.uniform(shape=()) < 0.5:
			i=2

		s_tf = tf.convert_to_tensor(self.s[0:self.index+1])
		s_diff_tf = tf.convert_to_tensor(self.s_diff[0:self.index+1])
		s_vel_tf = tf.convert_to_tensor(self.s_vel[0:self.index+1])
		with tf.GradientTape() as tape:
			q = self.models[i]([s_tf, s_diff_tf, s_vel_tf])
			loss = self._critic_loss(q)

		gradients = tape.gradient(loss, self.models[i].trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.models[i].trainable_variables))

		if terminal:
			with tf.GradientTape() as tape:
				terminalQ = self.models[i]([tf.convert_to_tensor([self.s[self.index]]), tf.convert_to_tensor([self.s_diff[self.index]]), tf.convert_to_tensor([self.s_vel[self.index]])])
				loss = tf.reduce_mean(tf.square(terminalQ))

			gradients = tape.gradient(loss, self.models[i].trainable_variables)
			optimizer.apply_gradients(zip(gradients, self.models[i].trainable_variables))

		# Actor Loss

		qt1 = self.models[3]([tf.convert_to_tensor(self.s[0:self.index]), tf.convert_to_tensor(self.s_diff[0:self.index]), tf.convert_to_tensor(self.s_vel[0:self.index])])
		qt2 = self.models[4]([tf.convert_to_tensor(self.s[0:self.index]), tf.convert_to_tensor(self.s_diff[0:self.index]), tf.convert_to_tensor(self.s_vel[0:self.index])])
		q_min = tf.math.minimum(qt1, qt2)

		with tf.GradientTape() as tape:
			action = self.models[0]([tf.convert_to_tensor(self.s[0:self.index]), tf.convert_to_tensor(self.s_diff[0:self.index]), tf.convert_to_tensor(self.s_vel[0:self.index])])
			loss = self._actor_loss(action, q_min)

		gradients = tape.gradient(loss, self.models[0].trainable_variables)
		optimizer.apply_gradients(zip(gradients, self.models[0].trainable_variables))

		# Temperature Loss (alpha)

		with tf.GradientTape() as tape:
			loss = self._temperature_loss()

		gradients = tape.gradient(loss, self.alpha)
		optimizer.apply_gradients([(gradients, self.alpha)])

		#self.models[i+2].set_weights(self.tau * self.models[i].get_weights() + (1-self.tau) * self.models[i+2].get_weights())
		updated_weights = [(self.tau * source_weight) + ((1-self.tau) * target_weight) for source_weight, target_weight in zip(self.models[i].get_weights(), self.models[i+2].get_weights())]
		self.models[i+2].set_weights(updated_weights)

		self.s[0] = self.s[self.batch_size]
		self.a[0] = self.a[self.batch_size]
		self.pi[0] = self.pi[self.batch_size]
		self.s_diff[0] = self.s_diff[self.batch_size]
		self.s_vel[0] = self.s_vel[self.batch_size]
		self.d[0] = self.d[self.batch_size]
		self.index = 0

		if (terminal):
			self._respawn()

		if self.epoch % 50 == 0:
			for i in range(len(self.models)):
				self.models[i].save_weights(self.weights_folder[i])

		self.epoch += 1

		#self._unpause_simulation()

	def _critic_loss(self, q):
		#sum=0
		#for i in range(batch_size)
		#	sum += 0.5 * tf.square(q[i][self.a[i] -
		#		(self.r[i] + self.gamma * tf.reduce_sum(self.pi[i+1] * (q[i+1] - self.alpha * tf.math.log(self.pi[i+1])))))
		#return sum

		target_q_values = self.r[0:self.index] + self.gamma * tf.reduce_sum(self.pi[1:self.index+1] * (q[1:self.index+1] - self.alpha * tf.math.log(self.pi[1:self.index+1])), axis=-1)
		selected_q_values = tf.reduce_sum(tf.multiply(q[0:self.index], tf.one_hot(self.a[0:self.index], depth=q.shape[-1], dtype=q.dtype)),axis=-1)
		loss = 0.5 * tf.reduce_mean(tf.square(selected_q_values - target_q_values))
		return loss

	def _actor_loss(self, action, q_min):
		#sum = 0
		#for i in range(self.batch_size):
		#	sum += tf.reduce_sum(action[i] * (self.alpha * tf.math.log(action[i]) - self.q[i]))
		#return sum
		log_term = self.alpha * tf.math.log(action)
		elementwise_product = action * (log_term - q_min)
		return tf.reduce_sum(elementwise_product)

	def _temperature_loss(self):
		return tf.reduce_sum(self.pi[0:self.index] * (-self.alpha * (tf.math.log(self.pi[0:self.index]) + self.H)))

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
