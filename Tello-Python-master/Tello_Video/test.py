import tello
import cv2
from tello_control_ui import TelloUI
import time
import math
import numpy as np
drone = tello.Tello('', 8889)

forward_dis = 70

def meet_id_11(remain_distance):
    global drone
    drone.rotate_cw(90)

# def meet_id_4(remain_distance):
#     global drone

#     # while abs(remain_distance - 60) > 10:

#     if remain_distance - forward_dis < 0:
#        drone.move_backward(0.18)
       
#     elif remain_distance - forward_dis > 0:
#        drone.move_forward(0.18)

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
    print(drone.get_battery())

    while(1):
        frame = drone.read() 
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       ############
       
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
       

        if len(markerCorners) > 0 and len(markerIds) != 0:
            idx = 0
            frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
            rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 13.7, cameraMatrix, distCoeffs) 
            try:
                for i in range(len(markerIds)):
                    frame = cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec[i], tvec[i], 10)
                    # print("rvec[" + str(i) +"]: ", rvec[i])
                    # print("tvec[" + str(i) +"]: ", tvec[i])

                    # rvecs -> rotation matrix R
                    rtm = cv2.Rodrigues(rvec[i])
                    # get z' by R multiply z-axis(0, 0, 1) 
                    z = [0, 0, 1]
                    v = -np.dot(np.array(rtm[0]), np.array(z))
                    # get v by project z' to xz plane 
                    v[1] = 0
                    # get the angle 
                    radis = math.atan2(v[0], v[2])
                    # rad to degree
                    angle = math.degrees(radis)

                    id_ = int(markerIds[i][0])
                    t_vec = list(tvec[i][0])
                    # t_vec[1] += 15
                    # print("before: ", t_vec, tvec[i][0])
                    # t_vec[0] -= 3
                    # print("after: ", t_vec, tvec[i][0])

                    string = (" x: " + str(round(t_vec[0],1)) + " y: " + str(round(t_vec[1],1)) 
                            + " z: " + str(round(t_vec[2],1)) + " angle: " +  str(round(angle,1)) + " id: " + str(id))
                    cv2.putText(frame, string , (10,20+i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA )
                    idx = i
                    
                    if id_ == 1 :
                        break
                    

                print("now id: ", id_)
          
          # get rotation matrix
                distance = 0.8
                thre = 20

                print('distance: ',t_vec[2], 'y_distance: ', t_vec[1] ,'x_distance: ', t_vec[0])
                markid = int(markerIds[idx][0])
                if (markid == 4 or markid == 11):
                  thre = 10
                  distance = 0.4


                if np.abs(angle) > 20:
                    if angle > 0:
                        drone.rotate_cw(18)

                    else:
                        drone.rotate_ccw(18)

                elif (t_vec[1]  < (-1 * thre) and markid == 1) or (t_vec[2] < forward_dis + 70  and t_vec[1]  < (-1 * thre) ):

                    drone.move_up(0.2)

                elif (t_vec[1] > thre and markid == 1) or ( t_vec[2] < forward_dis + 70 and t_vec[1]  >  thre ) :
                    drone.move_down(0.2)

                elif t_vec[0] < (-1 * thre + 7 ):
                    drone.move_left(0.2)

                elif t_vec[0] > thre + 7:
                    drone.move_right(0.2)

                elif t_vec[2] > forward_dis:               
                    drone.move_forward(distance)


                elif markid == 4:
                    if t_vec[2] > 55:
                        drone.move_forward(0.2);
                    else:
                        drone.land()
                        break

                elif markid == 11 :
                    meet_id_11(t_vec[2])
            except Exception as e:
                print(e)
                traceback.print_exc()

    
        cv2.imshow("frame",frame)
        key = cv2.waitKey(300)

        if key!= -1:
            drone.keyboard(key)


if __name__ == "__main__":
    main()
