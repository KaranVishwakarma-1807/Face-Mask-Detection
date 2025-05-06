# detect_mask_video.py

import cv2
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model('mask_detector_model.h5')

# Load OpenCV face detector (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (150,150))
        face_array = np.expand_dims(face_resized / 255.0, axis=0)

        pred = model.predict(face_array)[0][0]
        label = "Mask" if pred < 0.5 else "No Mask"
        color = (0,255,0) if label == "Mask" else (0,0,255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Mask Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
