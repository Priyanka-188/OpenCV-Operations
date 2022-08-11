import cv2
import cv2
import numpy as np
import time

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_detect = cv2.CascadeClassifier('haarcascade_nose.xml')
smile_detect = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    # changing the color of the frame to grascale.
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray_scale, 1.6, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray_scale[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_detect.detectMultiScale(roi_gray,1.3, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        nose = nose_detect.detectMultiScale(roi_gray,2.5,5)
        for (ex,ey,ew,eh) in nose:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        smile = smile_detect.detectMultiScale(roi_gray,2.3,5)
        for (ex,ey,ew,eh) in smile:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)


    cv2.imshow("Live stream",frame)               
    if cv2.waitKey(100) & 0xFF == ord('x'):
        break

print("Number of Faces: ",len(faces))
cap.release()
cv2.destroyAllWindows()
