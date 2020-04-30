import tello
import cv2
from tello_control_ui import TelloUI
import time
import math
import numpy as np


def main():
	drone = tello.Tello('', 8889)

	time.sleep(5)
	fs = cv2.FileStorage("data.txt", cv2.FILE_STORAGE_READ)
	cameraMatrix = fs.getNode("intrinsic")
	distCoeffs = fs.getNode("distortion")
	cameraMatrix = cameraMatrix.mat()
	distCoeffs = distCoeffs.mat()
	dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
	parameters =  cv2.aruco.DetectorParameters_create()
	while(1):
		frame = drone.read() 
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

		############
		
		markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
		# print(tvec)
		# print(rvec)
		# string = str(", ".join(tvec[0]))
		if len(markerCorners) > 0:
			frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
			rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 13.7, cameraMatrix, distCoeffs) 
			try:
				t_vec = tvec[0][0]

				string = "x: " + str(round(t_vec[0], 3)) + ", " + "y: " + str( round(t_vec[1], 3)) + " z: " +  ", " + str( round(t_vec[2], 3))
				cv2.putText(frame, string , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA )
				frame = cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 10)
				
				# get rotation matrix
				rtm = cv2.Rodrigues(rvec)
				z = [0, 0, 1]
				# dot product of two vec
				v = -np.dot(np.array(rtm[0]), np.array(z))

				# project to xz plane
				v[1] = 0

				radis = math.atan2(v[0], v[2])
				angle = math.degrees(radis)
				distance = 0.2
				print('distance: ',t_vec[2] )

				if np.abs(angle) > 35:
					if angle > 0:
						# drone.rotate_cw(np.abs(angle))
						drone.rotate_cw(20)

						# print("rotate clockwise")
					else:
						# drone.rotate_ccw(np.abs(angle))
						drone.rotate_ccw(20)

						# print("rotate counter clockwise")
				markid = int(markerIds[0][0])
				if t_vec[2] > 50:						
					drone.move_forward(distance)

				elif markid == 4:
					print("one")
					drone.land()

				elif t_vec[1] < -5 :
					drone.move_up(distance)

				elif t_vec[1] > -5:
					drone.move_down(distance)

				# elif t_vec[0] > 3:
				# 	drone.move_right(distance)
				# elif t_vec[0] < -3:
				# 	drone.move_left(distance)
# # x > 0, move right
			# if t_vec[0] > 0.3: 
			# 	drone.move_right(distance)
			# # x < 0, move left
			# if t_vec[0] < -0.3:
			# 	drone.move_left(distance)
			except Exception as e:
				print(e)
	
		cv2.imshow("frame",frame)
		key = cv2.waitKey(32)

		if key!= -1:
			drone.keyboard(key)


if __name__ == "__main__":
	main()
