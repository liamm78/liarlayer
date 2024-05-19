import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array

from glob import glob
import IPython.display as ipd
from tqdm.notebook import tqdm
import subprocess 
from keras.models import load_model

labels = ['angry','disgust','fear','happy','neutral','sad','surprised']
model = load_model('emotion1.h5')
emotion_counts = {label:0 for label in labels}
print(emotion_counts)

#subprocess.run(['ffmpeg', '-1',input_file,'-qscale','0',''])

#ipd.Video('EMDR Therapy Session demo by Psychologist.mp4', width=500)
cap = cv2.VideoCapture('EMDR Therapy Session demo by Psychologist.mp4')
face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

count = cap.get(cv2.CAP_PROP_FRAME_COUNT)


grabbed, frame = cap.read()
total_frames=0
target=100

while True:
    
    grabbed, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    if total_frames==target:
        break
    #frame_clone = frame.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, width, height) in faces:   # Get the region of interest
        cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
            
        roi = gray[y:y+height,x:x+width] # GETS THE FACE (REGION OF INTEREST) 
                
        roi = cv2.resize(roi, (48,48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0) #our actual input
                
        prob = model.predict(roi)[0]
        result = np.argmax(prob) #find max index of array
        predicted_label = labels[result]
        emotion_counts[predicted_label]+=1
        total_frames+=1
        cv2.putText(frame, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        
    cv2.imshow('Face', frame)
    
cap.release()
cv2.destroyAllWindows()

for emotion, count in emotion_counts.items():
    print(f"{emotion}: {count/target}")