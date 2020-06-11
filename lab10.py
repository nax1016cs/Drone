import tello
import cv2
from tello_control_ui import TelloUI
import time
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

objp = np.zeros((2*2,3), np.float32)
objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
imgp = np.zeros((2*2,2), np.float32)
imgp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
objp[1][0] = objp[3][0] = 20.0
objp[2][1] = objp[3][1] = 21.0

fs = cv2.FileStorage("data.txt", cv2.FILE_STORAGE_READ)
cameraMatrix = fs.getNode("intrinsic")
distCoeffs = fs.getNode("distortion")
cameraMatrix = cameraMatrix.mat()
distCoeffs = distCoeffs.mat()

def detectHorse(frame):
	dist = 999
	(H, W) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	print("[INFO] YOLO took {:.6f} seconds".format(end - start))

	boxes = []
	confidences = []
	classIDs = []

	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > args["confidence"]:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
	
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			if LABELS[classIDs[i]] == 'horse':
				# print(x, y, w, h)
				imgp[0] = [x, y]
				imgp[1] = [x+w, y]
				imgp[2] = [x, y+h]
				imgp[3] = [x+w, y+h]
				# print('imgp: ', imgp )
				retval, rvec, tvec = cv2.solvePnP(objp, imgp, cameraMatrix, distCoeffs)
				if retval:
					dist = tvec[2]
					cv2.putText(frame, str(tvec[2]) , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA )


				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
				cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	return frame, dist	

# def detect_marker(frame, markerCorners, markerIds, cameraMatrix, distCoeffs):

# 	if len(markerCorners) > 0 and len(markerIds) != 0:
# 		print(markerIds)
# 		idx = 0
# 		for id_ in markerIds:
# 			if (id_==1):
# 				break
# 			idx += 1
# 		print("id:1 is in ", idx)

# 		frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
# 		rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 12.9, cameraMatrix, distCoeffs) 
# 		try:
# 			# if rvec != None:
# 			print('rvec: ', rvec, 'tvec: ', tvec)
# 			print('rvec1: ', rvec[0])
# 			print('rvec2: ', rvec[0][0])
# 			rotv = rvec[idx][0]
# 			print("rotv: ", rotv)
# 			print(len(rotv))
# 			rtm = cv2.Rodrigues( rotv )

# 			t_vec = list(tvec[0][0])

# 			string = ("x: " + str(round(t_vec[0], 3)) + ", " + "y: " + str( round(t_vec[1], 3)) + " z: " +  ", " + str( round(t_vec[2], 3))
# 					+ " angle: " +  str(angle))
# 			cv2.putText(frame, string , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA )
# 			frame = cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 10)
			
# 			# get rotation matrix
# 			distance = 0.3
# 			thre = 30

# 			print('distance: ',t_vec[2], 'y_distance: ', t_vec[1] ,'x_distance: ', t_vec[0])
# 			markid = int(markerIds[0][idx])

# 			if (markid == 4 or markid == 11):
# 				thre = 10
# 				distance = 0.2

def main():
	drone = tello.Tello('', 8889)

	time.sleep(5)
	detect_horse = False

	dist1 = 0.9
	dist2 = 1.3

	fs = cv2.FileStorage("data.txt", cv2.FILE_STORAGE_READ)
	cameraMatrix = fs.getNode("intrinsic")
	distCoeffs = fs.getNode("distortion")
	cameraMatrix = cameraMatrix.mat()
	distCoeffs = distCoeffs.mat()
	dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
	parameters =  cv2.aruco.DetectorParameters_create()

	detect_mark = False
	firstin = False
	while(1):
		frame = drone.read()
		frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
		
		if detect_horse == False:
			if firstin:
				drone.move_up(0.4)
				firstin = False
			frame, dist = detectHorse(frame)
			if (dist < 55):
				detect_horse = True
				continue
			drone.move_forward(0.3)
			
			cv2.imshow("frame", frame)
			key = cv2.waitKey(1)
			if key == ord('1'):
				firstin = True
			if key != -1:
				drone.keyboard(key)
			continue
		elif detect_mark == False:
			# detect marker
			markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)

			# detect_marker(frame, markerCorners, markerIds, cameraMatrix, distCoeffs)
			time.sleep(5)
			drone.move_right( dist1 )
			time.sleep(5)
			drone.move_forward( dist2 )
			time.sleep(5)
			drone.move_left( dist1 )			
			time.sleep(5)
			drone.move_forward( dist2 )
			detect_mark = True
		cv2.imshow("frame", frame)
		key = cv2.waitKey(1)
		if key == ord('1'):
			firstin = True
		if key != -1:
			drone.keyboard(key)

if __name__ == "__main__":
	main()
