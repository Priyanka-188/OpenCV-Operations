import cv2

face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_detect = cv2.CascadeClassifier('haarcascade_eye.xml')
nose_detect = cv2.CascadeClassifier('haarcascade_nose.xml')
smile_detect = cv2.CascadeClassifier('haarcascade_smile.xml')

img= cv2.imread('ankit.jpg')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


faces = face_detect.detectMultiScale(img_gray, 1.3, 5)
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
    roi_gray = img_gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    eyes = eye_detect.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    
    nose = nose_detect.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in nose:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

    smile = smile_detect.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in smile:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)



print("Number of Faces: ",len(faces))
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


