import streamlit as st
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model



labels = ['ANGRY','DISGUST','FEAR','HAPPY','NEUTRAL','SAD','SUPRISED']
emotion_counts = {label:0 for label in labels}


camera = cv2.VideoCapture('therapyTest.mp4')
count = camera.get(cv2.CAP_PROP_FRAME_COUNT)

face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
model = load_model('emotion1.h5')


st.title("Candor Canvas: Emotion Evaluater")
frame_placeholder = st.empty()
stop_button_pressed=st.button("Stop")
col2 = st.beta_column(1)

total_frames=0
target=count-1

label = st.empty()
while camera.isOpened() and not stop_button_pressed:
    
    grabbed, frame = camera.read()
    if not grabbed:
        st.write("The video capture has ended")
        break
    if total_frames==target:
        break
    
    frame_clone = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, width, height) in faces:   # Get the region of interest
        cv2.rectangle(frame_clone,(x,y),(x+width,y+height),(0,255,0),2)
            
        roi = gray[y:y+height,x:x+width] # GETS THE FACE (REGION OF INTEREST) 
                
        roi = cv2.resize(roi, (48,48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        #roi = np.expand_dims(roi, axis=-1) 
        roi = np.expand_dims(roi, axis=0) #our actual input
                
        prob = model.predict(roi)[0]
        result = np.argmax(prob) #find max index of array
        predicted_label = labels[result]
        label.text(f"Patient is feeling {predicted_label}")
        emotion_counts[predicted_label]+=1
        total_frames+=1
    frame_placeholder.image(frame_clone, channels="BGR")

camera.release()
cv2.destroyAllWindows()

for emotion, count in emotion_counts.items():
    col2.write(f"{emotion}: {(count/target)*100}%")
