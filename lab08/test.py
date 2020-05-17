import cv2
import numpy as np

# img = cv2.imread('test0.jpg')
cap = cv2.VideoCapture(0)

# initialize HOG
objp = np.zeros((2*2,3), np.float32)
objp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
imgp = np.zeros((2*2,2), np.float32)
imgp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)


# initialize faces
objp_f = np.zeros((2*2,3), np.float32)
objp_f[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
imgp_f = np.zeros((2*2,2), np.float32)
imgp_f[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)


# get cameramatrix and distcoeffs
fs = cv2.FileStorage("data.txt", cv2.FILE_STORAGE_READ)
cameraMatrix = fs.getNode("intrinsic")
distCoeffs = fs.getNode("distortion")
cameraMatrix = cameraMatrix.mat()
distCoeffs = distCoeffs.mat()
while True:
    ret, img = cap.read()
    try: 
        # HOG
        hog = cv2.HOGDescriptor() 
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        rects, weights = hog.detectMultiScale(img, winStride = (2, 4) ,  scale = 1.02, useMeanshiftGrouping= False)

        # get the correct persom
        for i in range(len(rects)):
            body_i = rects[i]
            (x, y, w, h) = (rects[i][0], rects[i][1], rects[i][2], rects[i][3])
            if h < 300:
                continue
            print(x, y, w, h) 
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # imgpoint
        # (x,y)   (x+w, y)
        # (x,y+h) (x+w, y+h)
        for i in range(4):
            for j in range(2):
                imgp[i][j] = rects[0][j]
                if i == 1 and j == 0 or i == 3 and j == 0 :
                    imgp[i][j] += w
                if i >= 2 and j == 1:
                    imgp[i][j] += h

        # print('img', imgp)
        # print('obj', objp)
        retval, rvec, tvec = cv2.solvePnP(objp, imgp, cameraMatrix, distCoeffs)
        # print('rvec', rvec)
        # print('tvec', tvec)
        cv2.putText(img, str(tvec[2]) , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1, cv2.LINE_AA )

        # face
        face_cascade= cv2.CascadeClassifier('C:\\Users\\Chieh-Ming Jiang\\Anaconda3\\Lib\\site-packages\\data\\haarcascades/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale( img , scaleFactor = 1.1 , minNeighbors = 3 , minSize = (30,30), maxSize = (50,50)) 

        # get the correct face
        for i in range(len(faces)):
            body_i = faces[i]
            (x, y, w, h) = (faces[i][0], faces[i][1], faces[i][2], faces[i][3])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # calculate the image point
        for i in range(4):
            for j in range(2):
                imgp_f[i][j] = faces[0][j]
                if i == 1 and j == 0 or i == 3 and j == 0 :
                    imgp_f[i][j] += w
                if i >= 2 and j == 1:
                    imgp_f[i][j] += h
        # print(objp_f)
        # print(imgp_f)
        retval_f, rvec_f, tvec_f = cv2.solvePnP(objp_f, imgp_f, cameraMatrix, distCoeffs)
        cv2.putText(img, str(tvec_f[2]) , (50,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA )
        
    except:
        pass    
    cv2.imshow("frame",img) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()