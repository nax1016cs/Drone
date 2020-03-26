import numpy as np
import cv2

set1 = [i for i in range(99999)]
color = [[0 for i in range(3)] for j in range(99999)]

def find_parent(idx):
    if set1[idx] == idx:
        return idx
    else:
        r = find_parent(set1[idx])
        set1[idx] = r
        return r

def union(idx1, idx2):
    root1 = find_parent(idx1)
    root2 = find_parent(idx2)

    if root1 > root2:
        set1[root1] = root2
    else:
        set1[root2] = root1

img = cv2.imread('C:\Users\lin13\Downloads\input2.jpg', 0)
height = img.shape[0]
width = img.shape[1]
new_img = np.zeros( (height, width, 3), np.uint8)

label = [[0 for i in range(width)] for j in range(height)]
cur_label = 1

for i in range(height):
    for j in range(width):
        if img[i][j] == 0:
            continue
        else:
            if i - 1 < 0:
                up = 0
            else:
                up = label[i-1][j]
            if j - 1 < 0:
                left = 0
            else:
                left = label[i][j-1]
        if up == 0 and left == 0:
            label[i][j] = cur_label
            cur_label += 1
        elif up == 0 and left != 0:
            label[i][j] = left
        elif up != 0 and left == 0:
            label[i][j] = up
        else:
            label[i][j] = min(up, left)
            union(up, left)

for i in range(height):
    for j in range(width):
        if img[i][j] != 0:
            label[i][j] = find_parent(label[i][j])

for i in range(height):
    for j in range(width):
        l = label[i][j]
        if l == 0:
            new_img[i][j] = [0, 0, 0]
        else:
            if color[l] == [0, 0, 0]:
                new_color = list(np.random.choice(range(256), size=3))
                color[l] = new_color
            new_img[i][j] = color[l]

cv2.imshow('new image', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()