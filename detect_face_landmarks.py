import cv2
import dlib

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

cap = cv2.VideoCapture(0)  # webcame video

while True:
    ret, img = cap.read()
    faces = face_detector(img, 1)
    landmark_tuple = []
    mask = None
    for k, d in enumerate(faces):
        landmarks = landmark_detector(img, d)

        for n in range(0, 81):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_tuple.append((x, y))
            cv2.circle(img, (x, y), 2, (255, 255, 0), -1)

    cv2.imshow('img.jpg', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
