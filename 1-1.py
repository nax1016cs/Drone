import cv2
import numpy as np

img = cv2.imread('kobe.jpg')
height, width, channel = img.shape
# print(img.shape)
new_img = np.zeros((height, width, 1), np.uint8)
# new_img = img.copy()
for h in range(height):
	for w in range(width):
		new_pixel = 0
		for ch in range(channel):
			new_pixel += img [h][w][ch]
		new_img[h][w] = new_pixel/3
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
