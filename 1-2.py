import cv2
import numpy as np

img = cv2.imread('IU.png')
height, width, channel = img.shape
new_img = np.zeros((height*3, width*3, 3), np.uint8)
for h in range(height*3):
	for w in range(width*3):
		for ch in range(channel):
			new_img[h][w][ch] = img[h/3][w/3][ch]
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
