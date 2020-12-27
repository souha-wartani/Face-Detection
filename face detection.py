import cv2
from random import randrange
#Load some pre-trained data on face frontals from opencv (Cascade Algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

####to capture video from webcam
webcam =cv2.VideoCapture(0)
###iterate forever frames
while True:
    succeful_frame_read, frame =webcam.read()
    #Must convert to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    #detect Faces
    for(x, y, w, h) in face_coordinates :
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)
    
    cv2.imshow('Face detector',frame)
    key = cv2.waitKey(1)
    
    ###stop if q key is pressed
    if key==81 or key==113:
        break

###release the VideoCapture object
webcam.release()
