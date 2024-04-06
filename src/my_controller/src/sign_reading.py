#!/usr/bin/env python3

import os
import cv2
import numpy as np
from shapely import geometry


class ImageProcessing():
	def __init__(self):
		self.imageCount = 1

	def save_image(self, cv_image, output_directory, index):
		filename = os.path.join(output_directory, f"image_{self.imageCount}_{index}.jpg")
		cv2.imwrite(filename, cv_image)

	def run(self):
		groups = []
		current_group = []
		threshold = 15
		past_index = 0
		dir ='/home/fizzer/ros_ws/src/my_controller/src/vision'
		for i in range(1, 228): #450
			image_path = os.path.join(dir, "image_" + str(i)  + ".jpg")
			cv2_image = cv2.imread(image_path)
			found, max_approx, max_area = self.detectSign(cv2_image)
			if found:
				current_group.append((i, max_approx, max_area))
				past_index = i
			elif abs(i - past_index) > threshold and len(current_group) != 0:
				groups.append(current_group)
				current_group = []
			if i == 449 and len(current_group) != 0:
				groups.append(current_group)

		for group in groups:
			currArea = group[0][2]
			maxAreaIdx = 0
			for i in range(1, len(group)):
				if group[i][2] > currArea:
					currArea = group[i][2]
					maxAreaIdx = i

			image_path = os.path.join(dir, "image_" + str(group[maxAreaIdx][0])  + ".jpg")
			cv2_image = cv2.imread(image_path)
			self.foundSign(group[maxAreaIdx][1], cv2_image, group[maxAreaIdx][2])

	def order_corners(self, corners):
		corners_flat = corners.reshape(-1, 2)
		centroid = np.mean(corners_flat, axis=0)

		angles = np.arctan2(corners_flat[:, 1] - centroid[1], corners_flat[:, 0] - centroid[0])

		quadrants = np.zeros(len(corners_flat), dtype=np.uint8)
		quadrants[angles > 0] = 1  # Top quadrants
		quadrants[angles < 0] = 2  # Bottom quadrants
		quadrants[corners_flat[:, 0] - centroid[0] > 0] += 2  # Right quadrants

		sorted_indices = np.lexsort((angles, quadrants))
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
			if h < 100 and h > 10:
				arrayCoords.append([x,y,w,h])
		# Sort bounding boxes by x-coordinate
		arrayCoords.sort(key=lambda coord: coord[0])

		# Extract and save letters
		for i in range(len(arrayCoords)):
			x, y, w, h = arrayCoords[i]
			letter_image = img[y:y+h, x:x+w]
			output_directory = "/home/fizzer/ros_ws/src/my_controller/src/letters/"
			self.save_image(letter_image, output_directory, i)

	def find_average_blue(self, img):
		blue, green, red = cv2.split(img)
		threshold = 1.3 * np.maximum(red, green)
		mask = blue > threshold
		mask = (255 * mask).astype(np.uint8)

		masked_image = cv2.bitwise_and(img,img, mask=mask)

		num_pixels = np.sum(mask / 255)
		sum_rgb = np.sum(masked_image, axis=(0,1))

		avg_color = sum_rgb / num_pixels

		return avg_color[0] / np.maximum(avg_color[2], avg_color[1])

	def foundSign(self, max_approx, img, area_max):
		max_approx = self.order_corners(max_approx)
		result_img_color = self.perspective_transform(img, max_approx)
		#apb = self.find_average_blue(result_img)
		result_img = self.percent_blue_mask(result_img_color, 1.6)
		height, width = result_img.shape[:2]
		area = height*width
		result_img = self.white_padding(result_img)
		#cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{self.imageCount}_2.jpg", result_img)
		contours = cv2.findContours(result_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contours = contours[0] if len(contours) == 2 else contours[1]

		max_approx = []
		area_max = 0
		for contour in contours:
			epsilon = 0.1 * cv2.arcLength(contour, True)
			approx = cv2.approxPolyDP(contour, epsilon, True)

			for point in approx:
				if len(approx) == 4:
					poly = geometry.Polygon((approx[0][0],approx[1][0],approx[2][0],approx[3][0]))
					if poly.area > area_max and poly.area <= area:
						area_max = poly.area
						max_approx = approx

		if area_max > 8000:
			max_approx = self.order_corners(max_approx)
			result_img = self.perspective_transform(result_img_color, max_approx)
			apb = self.find_average_blue(result_img)
			result_img = self.percent_blue_mask(result_img, apb * 0.95)
			cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{self.imageCount}_3.jpg", result_img)
			clueType, clueValue = self.split_clueT_clueV(result_img)
			cv2.imwrite(f"/home/fizzer/ros_ws/src/my_controller/src/signs/image_{self.imageCount}.jpg", clueValue)
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

		if area_max > 8000:
			return True, max_approx, area_max
		return False, None, None

if __name__ == '__main__':
	dr = ImageProcessing()
	dr.run()
