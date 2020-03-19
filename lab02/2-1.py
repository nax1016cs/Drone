import cv2
import numpy as np
import math

img = cv2.imread('HyunBin.jpg', 0)
mask_x = [-1, 0, 1,
		  -2, 0, 2,
		  -1, 0, 1]
mask_y = [-1, -2, -1,
		  0, 0, 0,
		  1, 2, 1]
height, width = img.shape
new_img = np.zeros((height, width, 1), np.uint8)

lt = [0 for i in range(256)]
new_lt = [0 for i in range(256)]
for h in range (height):
	for w in range(width):
		lt[img[h][w]] += 1
		# print(img)
current_sum = 0
for i in range(256):
	current_sum += lt[i]
	new_lt[i] = current_sum
for i in range(256):
	new_lt[i] = int (new_lt[i] * 1.0 * 255 / current_sum)

for h in range(height):
	for w in range(width):
		new_img[h][w] = new_lt[ img[h][w]]

new_img_2 = np.zeros((height, width, 1), np.uint8)
new_img_3 = np.zeros((height, width, 1), np.uint8)
new_img_4 = np.zeros((height, width, 1), np.uint8)


for h in range(1 , height -1) :
	for w in range(1 , width-1 ) :
		new_pixel_x = 0
		new_pixel_y = 0
		idx = 0
		for i in range(3):
			for j in range(3):
				new_pixel_x += img[h + i -1][w + j -1] * mask_x [idx]
				new_pixel_y += img[h + i -1][w + j -1] * mask_y [idx]
				idx += 1
		if((new_pixel_x) > 70):
			new_img_2[h][w] = 255
		else :
			new_img_2[h][w] = 0

		if((new_pixel_y) > 70):
			new_img_3[h][w] = 255
		else :
			new_img_3[h][w] = 0
		new_img_4[h][w] += new_img_3[h][w] + new_img_2[h][w]

cv2.imshow('new_img_x', new_img_2)
cv2.imshow('new_img_y', new_img_3)
cv2.imshow('new_img_output', new_img_4)

cv2.waitKey(0)
