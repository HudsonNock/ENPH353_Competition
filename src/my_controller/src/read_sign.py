#!/usr/bin/env python3

import os
import cv2
import numpy as np
from shapely import geometry
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


class ImageProcessing():
	def __init__(self):
		self.imageCount = 1
		self.letter_index = 1384
		#with open('text_answers.txt', 'r') as file:
		#	self.lines = [line.strip() for line in file]
		#print(lines)
		self.sec2 = 0
		self.sec3 = 0
		self.sec4 = 0
		self.last_index = 0

		self.iteration = 0
		self.images_arr = []
		self.word_lengths = []
		self.words = []

		self.groups = []
		self.missed_signs = []

		self.model = Sequential([ \
			Conv2D(64, (3, 3), strides=(2, 2), activation='relu', input_shape=(65,55,1)), \
			MaxPooling2D((2, 2), strides=(2, 2)), \
			Conv2D(64, (3, 3), padding='same', activation='relu'), \
			MaxPooling2D((2, 2), strides=(2, 2)), \
			Flatten(), \
			Dense(512, activation='relu'), \
			Dropout(0.1), \
			Dense(256, activation='relu'), \
			Dropout(0.1), \
			Dense(128, activation='relu'), \
			Dropout(0.1), \
			Dense(36, activation='softmax') \
			])

		self.model.load_weights('model_weights_iteration3_8.h5')

	def update_sec(self, sec2, sec3, sec4, last_index):
		self.sec2 = sec2
		self.sec3 = sec3
		self.sec4 = sec4
		self.last_index = last_index

	def save_image(self, cv_image, output_directory, letter):
		filename = os.path.join(output_directory, f"{self.letter_index}_" + letter + ".jpg")
		cv2.imwrite(filename, cv_image)

	def run(self):
		current_group = []
		#threshold = 20
		#past_index = 0
		max_index = self.last_index + 1
		dir ='/home/fizzer/ros_ws/src/my_controller/src/vision'
		intervals = [[1,15],
			 	[44,58],
				[70,84],
				[self.sec2-23, self.sec2],
				[self.sec2 + 35, self.sec2+54],
				[self.sec3-15, self.sec3],
				[self.sec4-15, self.sec4+5],
				[self.last_index-35, self.last_index]]
		for j in range(len(intervals)):
			for i in range(intervals[j][0], intervals[j][1]+1):
				image_path = os.path.join(dir, "image_" + str(self.iteration) + "_" + str(i) + ".jpg")
				cv2_image = cv2.imread(image_path)
				found, max_approx, max_area = self.detectSign(cv2_image)
				if found:
					current_group.append((self.iteration, i, max_approx, max_area))
					#past_index = i
				#elif abs(i - past_index) > threshold and len(current_group) != 0:
				#	groups.append(current_group)
				#	current_group = []
				if i == intervals[j][1] and len(current_group) != 0:
					#print(j)
					self.groups.append(current_group)
					current_group = []
				elif i == intervals[j][1]:
					self.missed_signs.append(j+1)

		#print("group len")
		#print(len(self.groups))
		#print(self.missed_signs)
		#print("A")
			#print(self.words)

	def update_group(self, intervals):
		current_group = []
		dir ='/home/fizzer/ros_ws/src/my_controller/src/vision'
		j = 0
		self.iteration += 1
		while j < len(self.missed_signs):
			for i in range(intervals[j][0], intervals[j][1]+1):
				image_path = os.path.join(dir, "image_" + str(self.iteration) + "_" + str(i) + ".jpg")
				cv2_image = cv2.imread(image_path)
				found, max_approx, max_area = self.detectSign(cv2_image)
				if found:
					current_group.append((self.iteration, i, max_approx, max_area))
					#past_index = i
				#elif abs(i - past_index) > threshold and len(current_group) != 0:
				#	groups.append(current_group)
				#	current_group = []
				if i == intervals[j][1] and len(current_group) != 0:
					#print(j)
					index = self.missed_signs[j] - j - 1
					self.groups = self.groups[:index] + [current_group] + self.groups[index:]
					self.missed_signs.pop(j)
					current_group = []
					j -= 1
			j += 1

	def get_missed_signs(self):
		return self.missed_signs

	def get_words(self):
		dir ='/home/fizzer/ros_ws/src/my_controller/src/vision'
		for group in self.groups:
			currArea = group[0][3]
			maxAreaIdx = 0
			for i in range(1, len(group)):
				if group[i][3] > currArea: # >
					currArea = group[i][3]
					maxAreaIdx = i

			image_path = os.path.join(dir, "image_" + str(group[maxAreaIdx][0]) + "_" + str(group[maxAreaIdx][1]) + ".jpg")
			cv2_image = cv2.imread(image_path)
			self.foundSign(group[maxAreaIdx][2], cv2_image, group[maxAreaIdx][3])

		self.images_arr = np.array(self.images_arr)
		predictions = self.model.predict(self.images_arr)

		index = 0
		for i in range(len(self.word_lengths)):
			word = ""
			for j in range(self.word_lengths[i]):
				word += self.get_top_characters(predictions[index])
				index += 1
			self.words.append(word)
			print(word)
		return self.words

	def get_top_characters(self, probabilities):
		characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

		probabilities = np.array(probabilities)
		characters = np.array(list(characters))

		top_indices = np.argsort(probabilities)[-2:][::-1]
		top_characters = characters[top_indices]

		return top_characters[0]

	def order_corners(self, corners):
		corners_flat = corners.reshape(-1, 2)
		centroid = np.mean(corners_flat, axis=0)

		#angles = np.arctan2(corners_flat[:, 1] - centroid[1], corners_flat[:, 0] - centroid[0])

		#quadrants = np.zeros(len(corners_flat), dtype=np.uint8)
		#quadrants[corners_flat[:,0] - centroid[0] > 0] = 2  # Right quadrants
		quadrants = np.where(corners_flat[:,0] - centroid[0] > 0, 2, 1)
		#quadrants[angles > 0] += 1  # Top
		#match = 0
		#for i in range(1, len(quadrants)):
		#	if quadrants[i] == quadrants[0]:
		#		match = i
		#		break
		#if corners_flat[0][1] > corners_flat[match][1]:
		#	quadrants[0] += 1
		#else:
		#	quadrants[match] += 1

		#rhs = []
		#for i in range(1,4):
		#	if i != match:
		#		rhs.append(i)

		#if corners_flat[rhs[0]][1] > corners_flat[rhs[1]][1]:
		#	quadrants[rhs[0]] += 1
		#else:
		#	quadrants[rhs[1]] += 1
		#quadrants[corners_flat[:, 0] - centroid[0] > 0] += 2  # Right quadrants

		#sorted_indices = np.lexsort(quadrants)
		sorted_indices = np.lexsort((corners_flat[:,1], quadrants))
		ordered_corners = corners_flat[sorted_indices]
		ordered_corners = ordered_corners.reshape(-1, 1, 2)

		return ordered_corners

	def white_padding(self, image):
		height, width = image.shape[:2]
		padding_size = 2
		canvas = np.ones((height + 2 * padding_size, width + 2 * padding_size), dtype=np.uint8) * 255

		x_offset = padding_size
		y_offset = padding_size

		canvas[y_offset:y_offset+height, x_offset:x_offset+width] = image

		return canvas

	def split_clueT_clueV(self, img):
		height, width = img.shape[:2]
		clueType = img[0:int(height/2), 0:width]
		clueValue = img[int(height/2):height, 0:width]
		#dirValue = "/home/fizzer/ros_ws/src/live_data/processed_real_data/split_clueT_clueV/Value/"
		#dirType = "/home/fizzer/ros_ws/src/live_data/processed_real_data/split_clueT_clueV/Type/"
		#self.save_image(clueType, dirType)
		#self.save_image(clueValue, dirValue)
		return clueType, clueValue

	def perspective_transform(self, img, max_approx):
		# Assuming max_approx is ordered as TL, TR, BR, BL
		blx, bly = max_approx[0][0]
		tlx, tly = max_approx[1][0]
		brx, bry = max_approx[2][0]
		trx, try_ = max_approx[3][0]

		# Define the source points (coordinates of the corners of the skewed license plate)
		src_pts = np.array([[tlx, tly], [trx, try_], [brx, bry], [blx, bly]], dtype=np.float32)

		width = 600
		height = 400

		dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

		matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
		img = cv2.warpPerspective(img, matrix, (width, height))

		return img

	def percent_blue_mask(self, img, percent_threshold):
		blue, green, red = cv2.split(img)
		threshold = percent_threshold * np.maximum(red, green)
		mask = (blue > threshold).astype(np.uint8)
		gray_img = 255 * mask
		gray_img = gray_img.astype(np.uint8)
		return gray_img

	def blue_mask(self, img, color, threshold):
		blue, green, red = cv2.split(img)
		mask1 = np.sum(np.abs(img - color), axis=2) < threshold
		mask2 = green < color[1]
		mask3 = red < color[2]
		mask4 = np.logical_and(mask2, mask3)
		mask = np.logical_or(mask1, mask4).astype(np.uint8)

		return 255*mask


	def create_bitmap(self, image, lower_bounds, upper_bounds):
		b, g, r = cv2.split(image)

		b_mask = np.logical_and(b > lower_bounds[0], b < upper_bounds[0])
		g_mask = np.logical_and(g > lower_bounds[1], g < upper_bounds[1])
		r_mask = np.logical_and(r > lower_bounds[2], r < upper_bounds[2])

		combined_mask = np.logical_and(np.logical_and(b_mask, g_mask), r_mask)
		bitmap = combined_mask.astype(np.uint8)

		return bitmap

	def findLetters(self, img):
		# Set up the detector with default parameters.
		_, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

		# Find contours
		area_max = 0
		arrayCoords = []
		newOrderedCoords = []
		xCoords = []
		contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		for contour in contours:

			x, y, w, h = cv2.boundingRect(contour)
			# order letters
			if h < 80 and h > 20 and w > 7 and w < 80:
				arrayCoords.append([x,y,w,h])
		# Sort bounding boxes by x-coordinate
		arrayCoords.sort(key=lambda coord: coord[0])
		#print(arrayCoords)

		i = 0
		while i < len(arrayCoords)-1:
			#print(arrayCoords[i])
			#print(arrayCoords[i][1])
			if arrayCoords[i][1] + arrayCoords[i][3] < arrayCoords[i+1][1] or arrayCoords[i][1] > arrayCoords[i+1][1] + arrayCoords[i+1][3]:
				x = min(arrayCoords[i][0], arrayCoords[i+1][0])
				y = min(arrayCoords[i][1], arrayCoords[i+1][1])
				w = max(arrayCoords[i][2], arrayCoords[i+1][2])
				h = arrayCoords[i][3] + arrayCoords[i+1][3]
				del arrayCoords[i]
				del arrayCoords[i]
				arrayCoords = arrayCoords[0:i] + [[x,y,w,h]] + arrayCoords[i:]
				i -= 1
			elif arrayCoords[i][0] + arrayCoords[i][2] + 3 >= arrayCoords[i+1][0] and arrayCoords[i][2] + arrayCoords[i][2] < 50:
				x = arrayCoords[i][0]
				y = min(arrayCoords[i][1], arrayCoords[i+1][1])
				w = arrayCoords[i][2] + arrayCoords[i+1][2]
				h = max(arrayCoords[i][3], arrayCoords[i+1][3])
				del arrayCoords[i]
				del arrayCoords[i]
				arrayCoords = arrayCoords[0:i] + [[x,y,w,h]] + arrayCoords[i:]
				i -= 1
			i += 1

		# Extract and save letters
		self.word_lengths.append(len(arrayCoords))
		for i in range(len(arrayCoords)):
			x, y, w, h = arrayCoords[i]
			letter_image = img[y:y+h, x:x+w]
			#output_directory = "/home/fizzer/ros_ws/src/my_controller/src/letters/"
			#letter = self.lines[self.imageCount - 1][i]
			#self.save_image(letter_image, output_directory, letter)
			self.letter_index += 1

			padded_image = self.add_padding(letter_image)
			self.images_arr.append(np.array(padded_image))

	def add_padding(self, image):
		height, width = image.shape[:2]

		if width > 55 or height > 55:
			if width > height:
				resized_image = cv2.resize(image, (55, int(height * 55 / width)))
			else:
				resized_image = cv2.resize(image, (int(width * 55 / height), 55))
			height, width = resized_image.shape[:2]
		else:
			resized_image = image

		pad_height = max(0, 65 - height)
		pad_width = max(0, 55 - width)

		top = pad_height // 2
		bottom = pad_height - top
		left = pad_width // 2
		right = pad_width - left

		padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

		return padded_image

	def find_average_blue(self, img):
		blue, green, red = cv2.split(img)
		threshold = 1.3 * np.maximum(red, green)
		mask = blue > threshold
		mask = (255 * mask).astype(np.uint8)

		masked_image = cv2.bitwise_and(img,img, mask=mask)

		num_pixels = np.sum(mask / 255)
		sum_rgb = np.sum(masked_image, axis=(0,1))

		avg_color = sum_rgb / num_pixels

		return avg_color #avg_color[0] / np.maximum(avg_color[2], avg_color[1])

	def foundSign(self, max_approx, img, area_max):
		max_approx = self.order_corners(max_approx)
		#cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{self.imageCount}_5.jpg", img)
		#print("---")
		#print(self.imageCount)
		#print(max_approx)
		result_img_color = self.perspective_transform(img, max_approx)

		result_img = self.percent_blue_mask(result_img_color, 1.6)
		height, width = result_img.shape[:2]
		area = height*width
		result_img = self.white_padding(result_img)
		#cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{self.imageCount}_4.jpg", result_img) #
		contours = cv2.findContours(result_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[0] if len(contours) == 2 else contours[1]
		max_approx = []
		area_max = 0
		#print("A")
		for contour in contours:
			epsilon = 0.1 * cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, epsilon, True)

			for point in approx:
				if len(approx) == 4:
					poly = geometry.Polygon((approx[0][0],approx[1][0],approx[2][0],approx[3][0]))
					if poly.area > area_max and poly.area <= area:
						area_max = poly.area
						max_approx = approx

		print(self.imageCount)

		if area_max > 6000:
			max_approx = self.order_corners(max_approx)
			result_img = self.perspective_transform(result_img_color, max_approx)
			#print("-----")
			#print(self.imageCount)
			#print(max_approx)
			#cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{self.imageCount}_2.jpg", result_img)
			avg_color = self.find_average_blue(result_img)
			#if self.imageCount == 3:
			#	apb = avg_color[0] / np.maximum(avg_color[2], avg_color[1])
			#	result_img = self.percent_blue_mask(result_img, apb * 0.95)
			#elif self.imageCount == 5:
			#else:
			result_img = self.blue_mask(result_img, avg_color * 0.9, 20)
			#else:
			#	result_img = self.blue_mask(result_img, avg_color, 50)
			cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{self.imageCount}.jpg", result_img)
			clueType, clueValue = self.split_clueT_clueV(result_img)
			#cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{self.imageCount}.jpg", clueValue)
			self.findLetters(clueValue)
			self.imageCount += 1

	def detectSign(self,img):
		height = 720
		width = 1280

		map1 = self.create_bitmap(img, [80, -1, -1], [105, 5, 5])
		map2 = self.create_bitmap(img, [180, 90, 90], [215, 110, 110])
		map3 = self.create_bitmap(img, [105, 15, 15], [128, 35, 35])

		gray_img = 255 * np.logical_or(np.logical_or(map1, map2), map3)
		gray_img = gray_img.astype(np.uint8)

		kernel = np.ones((5, 5), np.uint8)
		gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_DILATE, kernel, iterations=1)
		gray_img = cv2.morphologyEx(gray_img, cv2.MORPH_ERODE, kernel, iterations=1)

		edged = cv2.Canny(gray_img,75, 150, 1)

		cnts = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if len(cnts) == 2 else cnts[1]

		area_max = 0
		max_approx = []

		for c in cnts:
			epsilon = 0.09 * cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, epsilon, True)

			if len(approx) == 4:
				poly = geometry.Polygon((approx[0][0],approx[1][0],approx[2][0],approx[3][0]))
				if poly.area > area_max:
					area_max = poly.area
					max_approx = approx

		if area_max > 6000:
			max_approx = self.order_corners(max_approx)
			result_img_color = self.perspective_transform(img, max_approx)
			#cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{area_max}_6.jpg", result_img_color)
			blue, green, red = cv2.split(result_img_color)
			avgGreen = np.sum(green) / (green.shape[0] * green.shape[1])
			if avgGreen > 25:
				return True, max_approx, area_max
		return False, None, None

if __name__ == '__main__':
	dr = ImageProcessing()  #273, 
	dr.update_sec(135,220,270,431)
	dr.run()
