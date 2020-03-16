import cv2
import numpy as np

img = cv2.imread('IU.png')
height, width, channel = img.shape
new_img = np.zeros((int (height*0.7), int (width*0.7), 3), np.uint8)
for h in range(height-1):
	for w in range(width-1):
		if(h == 0 or w == 0 or h == height or w== width) :
			new_img[int (h*0.7)][int (w*0.7)] = img[h][w]
		else:
			r1 = img[h + 1][ w ] * 0.7 + img[ h ][ w ] * 0.3
			r2 = img[h + 1][ w + 1 ] *0.7 + img[h + 1][ w + 1] * 0.3
			p = 0.7 * r2 + 0.3 * r1
			# print(p)
			new_img[int (h*0.7)][int (w*0.7)] = p
cv2.imshow('new_img', new_img)
cv2.waitKey(0)
