import cv2

face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # webcam video

while True:
    ret, img = cap.read()
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite("detect_face/face_detect.png", img)
    cv2.imshow('img.jpg', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
