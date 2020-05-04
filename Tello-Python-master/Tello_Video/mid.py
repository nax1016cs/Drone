import tello
import cv2
from tello_control_ui import TelloUI
import time
import math
import numpy as np
drone = tello.Tello('', 8889)

def meet_id_11(remain_distance):
	global drone
	drone.rotate_cw(90)

def meet_id_4(remain_distance):
	global drone

	# while abs(remain_distance - 60) > 10:

	if remain_distance - 60 < 0:
		drone.move_backward(0.15)
		
	elif remain_distance - 60 > 0:
		drone.move_forward(0.15)

	# drone.land()	

def main():
	global drone
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

		if len(markerCorners) > 0:
			frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
			rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 13.7, cameraMatrix, distCoeffs) 
			try:
				# if rvec != None:
				# print('rvec: ', rvec, 'tvec: ', tvec)
				rtm = cv2.Rodrigues(rvec)
				z = [0, 0, 1]
				# dot product of two vec
				v = -np.dot(np.array(rtm[0]), np.array(z))

				# project to xz plane
				v[1] = 0

				radis = math.atan2(v[0], v[2])
				angle = math.degrees(radis)

				t_vec = tvec[0][0]

				string = ("x: " + str(round(t_vec[0], 3)) + ", " + "y: " + str( round(t_vec[1], 3)) + " z: " +  ", " + str( round(t_vec[2], 3))
						+ " angle: " +  str(angle) )
				cv2.putText(frame, string , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA )
				frame = cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 10)
				t_vec[1] += 15
				
				# get rotation matrix
				distance = 0.3
				print('distance: ',t_vec[2], 'y_distance: ', t_vec[1] )
				markid = int(markerIds[0][0])

				if np.abs(angle) > 20:
					if angle > 0:
						# drone.rotate_cw(np.abs(angle))
						drone.rotate_cw(10)

						# print("rotate clockwise")
					else:
						# drone.rotate_ccw(np.abs(angle))
						drone.rotate_ccw(10)

						# print("rotate counter clockwise")
				# the foward distance

				elif t_vec[1]  < -20 :
					drone.move_up(0.2)

				elif t_vec[1] > 20:
					drone.move_down(0.2)

				elif t_vec[0] < -18 :
					drone.move_left(0.2)

				elif t_vec[0] > 18:
					drone.move_right(0.2)

				elif t_vec[2] > 60:						
					drone.move_forward(distance)

				# elif t_vec[2] < 60:						
					# drone.land()

				elif markid == 4:
					if abs(t_vec[2] - 60) > 15:
						meet_id_4(t_vec[2])
					else:
						drone.land()

				elif markid == 11 :
					# remain_distance = t_vec[2]
					meet_id_11(t_vec[2])
					
					# drone.land()

				


				# elif t_vec[0] > 3:
				# 	drone.move_right(distance)
				# elif t_vec[0] < -3:
				# 	drone.move_left(distance)

			except Exception as e:
				print(e)
	
		cv2.imshow("frame",frame)
		key = cv2.waitKey(32)

		if key!= -1:
			drone.keyboard(key)


if __name__ == "__main__":
	main()
