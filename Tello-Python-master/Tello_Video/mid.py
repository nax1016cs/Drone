import tello
import cv2
from tello_control_ui import TelloUI
import time
import math
import numpy as np
import os
import yolo_draw
import threading
from multiprocessing.pool import ThreadPool

drone = tello.Tello('', 8889)
forward_dis = 70

def meet_id_11(remain_distance):
	global drone
	drone.rotate_cw(90)

def meet_id_4(remain_distance):
	global drone

	# while abs(remain_distance - 60) > 10:

	if remain_distance - forward_dis < 0:
		drone.move_backward(0.18)
		
	elif remain_distance - forward_dis > 0:
		drone.move_forward(0.18)

	# drone.land()	

def main():
	global drone
	time.sleep(5)
	pool = ThreadPool(processes=1)

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
		net, COLORS, LABELS = yolo_draw.initial()
		async_result = pool.apply_async(yolo_draw.draw_horse, (frame, net, COLORS, LABELS))
		frame = async_result.get()

		###########
		
		markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
		

		if len(markerCorners) > 0 and len(markerIds) != 0:
			print(markerIds)
			idx = 0
			for id_ in markerIds:
				if (id_==1):
					break
				idx += 1
			print("id:1 is in ", idx)

			frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
			rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 12.9, cameraMatrix, distCoeffs) 
			try:
				# if rvec != None:
				print('rvec: ', rvec, 'tvec: ', tvec)
				print('rvec1: ', rvec[0])
				print('rvec2: ', rvec[0][0])
				rotv = rvec[idx][0]
				print("rotv: ", rotv)
				print(len(rotv))
				rtm = cv2.Rodrigues( rotv )

				z = [0, 0, 1]
				# dot product of two vec
				v = -np.dot(np.array(rtm[0]), np.array(z))

				# project to xz plane
				v[1] = 0

				radis = math.atan2(v[0], v[2])
				angle = math.degrees(radis)

				t_vec = list(tvec[0][0])
				t_vec[1] += 18
				t_vec[0] -= 6

				string = ("x: " + str(round(t_vec[0], 3)) + ", " + "y: " + str( round(t_vec[1], 3)) + " z: " +  ", " + str( round(t_vec[2], 3))
						+ " angle: " +  str(angle))
				cv2.putText(frame, string , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA )
				frame = cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 10)
				

				
				# get rotation matrix
				distance = 0.3
				thre = 30

				print('distance: ',t_vec[2], 'y_distance: ', t_vec[1] ,'x_distance: ', t_vec[0])
				markid = int(markerIds[0][idx])
				if (markid == 4 or markid == 11):
					thre = 10
					distance = 0.2


				if np.abs(angle) > 20:
					if angle > 0:
						drone.rotate_cw(18)

					else:
						drone.rotate_ccw(18)

				elif t_vec[1]  < (-1 * thre) :
					drone.move_up(0.2)

				elif t_vec[1] > thre:
					drone.move_down(0.2)

				elif t_vec[0] < (-1 * thre):
					drone.move_left(0.2)

				elif t_vec[0] > thre:
					drone.move_right(0.2)

 				elif t_vec[2] > forward_dis:						
					drone.move_forward(distance)


				elif markid == 4:
					if t_vec[2] > 50:
						drone.move_forward(0.2);
					else:
						drone.land()
					break

				elif markid == 11 :
					meet_id_11(t_vec[2])

			except Exception as e:
				print(e)
	
		cv2.imshow("frame",frame)
		key = cv2.waitKey(30)

		if key!= -1:
			drone.keyboard(key)


if __name__ == "__main__":
	main()
