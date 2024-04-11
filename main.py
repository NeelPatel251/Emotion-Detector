import cv2
from keras.models import model_from_json
import numpy as np
import numpy as np
import streamlit as st
from tensorflow.keras.models import model_from_json

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")
haar_file=cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def emotion_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    emotions = []
    for (p, q, r, s) in faces:
        face_image = gray[q:q+s, p:p+r]
        cv2.rectangle(image,(p,q),(p+r,q+s),(255,0,0),2)
        face_image = cv2.resize(face_image, (48, 48))
        img = extract_features(face_image)
        pred = model.predict(img)
        emotion = labels[pred.argmax()]
        emotions.append(emotion)
        cv2.putText(image, '% s' %(emotion), (p-10, q-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (255,0,0))
    return emotions, image

# Streamlit frontend
st.title("Emotion Detection Using Webcam")

# Function to capture image from webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cap.release()
        return frame
    else:
        st.error("Failed to capture image")

def main():
    start_capture = st.button('Start capturing from webcam')
    if start_capture:
        st.write('Capturing image from webcam...')
        captured_img = capture_image()
        emotions, image = emotion_detection(captured_img)
        st.write('Emotion Detected:', emotions)
        st.image(image)

if __name__ == "__main__":
    main()
