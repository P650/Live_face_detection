import cv2

#loading the pre trained data to the model
trained_face_data = cv2.CascadeClassifier('Trained_data.xml')

#choose image for our example
#image = cv2.imread('16662224157675.jpg')
webcam = cv2.VideoCapture(0)

while True:

    #read a current frame, it will result in 2 section that is web cam is true or no 
    frame_read, frame = webcam.read()

    #must convert to gray scale\
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect the face\
    face_points = trained_face_data.detectMultiScale(gray_scale)

    #getting the points from privious data collection
    #(x, y, w, h) = face_points[0]
    for (x, y, w, h) in face_points:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 10)

    #show image with sopts
    cv2.imshow('hay its me Pavankumar', frame)

    #without this it will not be possible and we can design this to stop after pressing specific key
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break
    

#release the web cam
webcam.release()


print("code is here")
print("code is complete only for human face detection going for other models.")