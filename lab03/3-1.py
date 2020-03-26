import cv2
import numpy as np
import math

img = cv2.imread('input.jpg', 0)
height, width = img.shape
new_img = np.zeros((height, width, 1), np.uint8)

lt = [0 for i in range(256)]

total_count = 0
for h in range (height):
	for w in range(width):
		lt[img[h][w]] += 1
		total_count += 1

total_sum = 0
for i in range(256):
	total_sum += i*lt[i]

sumB = 0
wb = 0
wf = 0
varmax = 0
threshold = 0
for i in range(256):
	wb += lt[i]
	if(wb ==0):
		continue
	wf = total_count - wb
	if(wf == 0):
		continue
	sumB += float(i*lt[i])
	mb = sumB / wb
	mf = float((total_sum - sumB)) / wf
	var_between = float(wb) * float(wf) *(mb -mf) *(mb-mf)
	if(var_between > varmax):
		varmax = var_between
		threshold = i
	# print(var_between, i)
for h in range (height):
	for w in range(width):
		if (img[h][w] > threshold):
			new_img[h][w] = 255
		else:
			new_img[h][w] = 0

cv2.imshow('new_img_output', new_img)
cv2.imwrite('3-1.jpg', new_img)
cv2.waitKey(0)