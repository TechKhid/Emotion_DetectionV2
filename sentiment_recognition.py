import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image 

model = load_model(r'C:\Users\Asustem\SentimentRecog\emotion\model.h5')
face_cascade = cv.CascadeClassifier(r"C:\Users\Asustem\SentimentRecog\haarcascade_frontalface_default.xml")
cap = cv.VideoCapture(0)

labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def preprocess(gray_img):
    roi_gray = gray_img[y:y+h, x:x+w]
    roi_gray = cv.resize(roi_gray, (48, 48), interpolation=cv.INTER_AREA)
    roi = roi_gray.astype('float')/255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
        
    return roi



while cap.isOpened():
    _, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    for x, y, w, h in faces:
        color = (100, 110, 0)
        cv.rectangle(frame, (x, y), (x+w, y+h), color)
        feed_ = preprocess(gray)
        pred_ = model.predict(feed_)[0]
        label = labels[pred_.argmax()]
        lab_pos = (x, y-10)
        cv.putText(frame, label, lab_pos, cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)
    cv.imshow("feed", frame)
    if cv.waitKey(1)&0xFF == ord('x'):
        break

cap.release()