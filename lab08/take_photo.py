import cv2
cap = cv2.VideoCapture(1)
i = 0
while(True):
  ret, frame = cap.read()
  cv2.imshow('frame', frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
    pic = 'opencv_frame_' + str(i) + '.jpg'
    i += 1
    cv2.imwrite(pic, frame)
    # cv2.imwrite('test', frame)
cap.release()
cv2.destroyAllWindows()