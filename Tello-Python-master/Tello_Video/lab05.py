import tello
import cv2
from tello_control_ui import TelloUI
import time

import numpy as np
import os
import glob
import math
import string

# Defining the dimensions of checkerboard
CHECKERBOARD = (7,5)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = [] 


# Defining the world coordinates for 3D points
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Extracting path of individual image stored in a given directory
images = glob.glob('lab5\calibration\*.jpg')
for fname in images:
    img = cv2.imread(fname)
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.imread(fname, 0)
    # Find the chess board corners
    # If desired number of corners are found in the image then ret = true
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    """
    If desired number of corner are detected,
    we refine the pixel coordinates and display 
    them on the images of checker board
    """
    if ret == True:
        objpoints.append(objp)
        # refining pixel coordinates for given 2d points.
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
    
    # cv2.imshow('img',img)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()

h,w = img.shape[:2]

"""
Performing camera calibration by 
passing the value of known 3D points (objpoints)
and corresponding pixel coordinates of the 
detected corners (imgpoints)
"""
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix : \n")
print(mtx)
print("dist : \n")
print(dist)
print("rvecs : \n")
print(rvecs)
print("tvecs : \n")
print(tvecs)

# f = cv2.FileStorage('lab4\\result.xml', cv2.FILE_STORAGE_WRITE)
# f.write("intrinsic", mtx)
# f.write("distortion", dist)
# f.release()

def main():
	drone = tello.Tello('', 8889)

	time.sleep(5)



	while(1):
		frame = drone.read() 
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		# cv2.imshow("frame",frame)
		# key = cv2.waitKey(1)

		dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
		parameters =  cv2.aruco.DetectorParameters_create()
		markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
		# print(markerCorners)
		# print(rejectedCandidates)
		frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
		rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 13.7, mtx, dist) 
		# print(tvec)
		# string = str(", ".join(tvec[0]))
		try:
			string = "x: " + str(round(tvec[0][0][0], 3)) + ", " + "y: " + str( round(tvec[0][0][1], 3)) + " z: " +  ", " + str( round(tvec[0][0][2], 3))
	
			cv2.putText(frame, string , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA )
			frame = cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, 10)

			cv2.imshow("frame",frame)
			key = cv2.waitKey(1)
		except:
			cv2.imshow("frame",frame)
			key = cv2.waitKey(1)
			
			continue

		if key!= -1:
		    drone.keyboard(key)


if __name__ == "__main__":
    main()
