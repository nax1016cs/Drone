import cv2
import numpy as np
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
size = (7,5)
objp = np.zeros((7*5,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
objpoints = [] 
imgpoints = []

# num_pic = 1
for i in range(1,20):
    pic = 'test_' + str(i) + '.jpg'
    image = cv2.imread(pic)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, size,None)
    if(ret == True ):
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        # img = cv2.drawChessboardCorners(gray, (5,7), corners2,ret)

retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None) 
f = cv2.FileStorage("t.txt", cv2.FILE_STORAGE_WRITE)
f.write("intrinsic", cameraMatrix)
f.write("distortion", distCoeffs)
f.release()
print("cameraMatrix:")
print(cameraMatrix)
print("distCoeffs: ")
print(distCoeffs)