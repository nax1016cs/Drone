import cv2
import numpy as np

# img = cv2.imread('test2.jpg')
cap = cv2.VideoCapture(0)

# initialize HOG
objp = np.zeros((2*2,3), np.float32)
objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
imgp = np.zeros((2*2,2), np.float32)
imgp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
objp[1][0] = objp[3][0] = 0.41
objp[2][1] = objp[3][1] = 1.65


# initialize faces
objp_f = np.zeros((2*2,3), np.float32)
objp_f[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
imgp_f = np.zeros((2*2,2), np.float32)
imgp_f[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)

# face object point (2:3)
objp_f[1][0] = objp_f[3][0] = 0.16 + 0.04 
objp_f[2][1] = objp_f[3][1] = 0.21 



# get cameramatrix and distcoeffs
fs = cv2.FileStorage("data.txt", cv2.FILE_STORAGE_READ)
cameraMatrix = fs.getNode("intrinsic")
distCoeffs = fs.getNode("distortion")
cameraMatrix = cameraMatrix.mat()
distCoeffs = distCoeffs.mat()

hog_idx = 0
face_idx = 0
acuumulated_hog_dis = 0
accumulated_face_dis = 0
face_dis = 0
hog_dis = 0
scan_val = 3

while True:
    ret, img = cap.read()
    try: 
        # HOG
        hog = cv2.HOGDescriptor() 
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(img, winStride = (4, 4) ,  scale = 1.05, useMeanshiftGrouping= False, padding= (8,8) )

        # get the correct persom
        idx = 0
        for i in range(len(rects)):
            (x, y, w, h) = rects[i]
            # (x, y, w, h) = (rects[i][0], rects[i][1], rects[i][2], rects[i][3])
            if h < 400:
                continue
            print(x, y, w, h)
            imgp[0] = [x, y]
            imgp[1] = [x+w, y]
            imgp[2] = [x, y+h]
            imgp[3] = [x+w, y+h]

            w_ = round(w * 0.3)
            h_ = round(h * 0.1)
            _h = round(h * 0.04)
            imgp[0] += [w_, _h]
            imgp[1] += [-w_, _h]
            imgp[2] += [w_, -h_]
            imgp[3] += [-w_, -h_]
            print('imgp: ', imgp )
            cv2.rectangle(img, tuple(imgp[0]) , tuple(imgp[3]),  (0, 255, 255), 2)
        #     idx = i

        # # imgpoint
        # #   0 1        0   1
        # # 0(x,y)    1(x+w, y)
        # # 2(x,y+h)  3(x+w, y+h)
        
        # print('img', imgp)
        # print('obj', objp)
        retval, rvec, tvec = cv2.solvePnP(objp, imgp, cameraMatrix, distCoeffs)
        # print('rvec', rvec)
        # print('tvec', tvec)
        if retval:
            hog_idx += 1
            acuumulated_hog_dis += tvec[2]
            if hog_idx % scan_val == 0:
                hog_dis = acuumulated_hog_dis/scan_val/1.0
                acuumulated_hog_dis = 0

            cv2.putText(img, str(hog_dis) , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv2.LINE_AA )

        # face
        face_cascade= cv2.CascadeClassifier('C:\\Users\\Chieh-Ming Jiang\\Anaconda3\\Lib\\site-packages\\data\\haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale( img , scaleFactor = 1.02 , minNeighbors = 7 , minSize = (30,30), maxSize = (50,50)) 

        # get the correct face
        # calculate the image point
        # (x,y)   (x+w, y)
        # (x,y+h) (x+w, y+h)
        for i in range(len(faces)):

            (x, y, w, h) = faces[i]
            imgp_f[0] = [x, y]
            imgp_f[1] = [x+w, y]
            imgp_f[2] = [x, y+h]
            imgp_f[3] = [x+w, y+h]

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)

        

        # print('objp_face: ', objp_f)
        # print('imgp_face: ', imgp_f)
        retval_f, rvec_f, tvec_f = cv2.solvePnP(objp_f, imgp_f, cameraMatrix, distCoeffs)
        if retval_f:
            face_idx += 1
            accumulated_face_dis += tvec_f[2]
            if face_idx % scan_val == 0:
                face_dis = accumulated_face_dis/scan_val/1.0
                accumulated_face_dis = 0
            cv2.putText(img, str(face_dis) , (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA )

    except:
        pass    
    cv2.imshow("frame",img) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()