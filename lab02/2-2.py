import cv2
import numpy as np
import math

img = cv2.imread('mj.jpg', 0)
height, width  = img.shape
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
		new_img[h][w] = new_lt[img[h][w]]
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
