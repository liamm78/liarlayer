
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
        #print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten,Dense,Dropout,BatchNormalization
from keras import regularizers
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2



training = 'C:/Users/liamm/Downloads/Emotion_Detector_AOL/train'
testing = 'C:/Users/liamm/Downloads/Emotion_Detector_AOL/test'
# Create the augumentation gens
train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, rescale = 1./255, validation_split=0.2)
test_gen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

# Add similar augmented images on top of the ones already there. 
train_generator = train_gen.flow_from_directory(directory = training, 
                                               target_size=(48,48), 
                                               batch_size=64, 
                                               color_mode='grayscale',
                                               class_mode='categorical',
                                               subset='training')

validation_generator = test_gen.flow_from_directory(directory=testing,
                                              target_size=(48,48), 
                                              batch_size=64, 
                                              color_mode='grayscale',
                                              class_mode='categorical',
                                              subset='validation')



## CREATING THE MODEL ##
# kernel_regularizer = regularizers.l2(0.01)
model=Sequential() # Sequential uses layers to build
 # 1st CONV block
model.add(Conv2D(32, kernel_size=(3,3), activation ='relu', input_shape = (48,48,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size =(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 2nd conv block
model.add(Conv2D(128, kernel_size =(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

#3rd conv block
model.add(Conv2D(256, kernel_size =(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, kernel_size =(3,3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7, activation='softmax')) #connects 500 neurons to 7 classes

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping

#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(x = train_generator,epochs = 30,validation_data = validation_generator)
print(history.history)

plt.title('Model accuracy')
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

#model.save_weights('emotion_model.h5')


#save_path = r'C:/Users/liamm/Downloads/Emotion_Detector_AOL/emotion_1.h5'
from tensorflow.keras.models import load_model

model.save('emotion1.h5')


'''
save_path = r'C:/Users/liamm\Downloads/Emotion_Detector_AOL/emotion_1.h5'

try:
    # Print the path to ensure it's correct
    print(f"Saving model weights to: {save_path}")
    
    # Save the model weights
    model.save_weights(save_path)
    
    # Check if the file now exists
    if os.path.exists(save_path):
        print('File saved successfully!')
    else:
        print('File not found after saving!')
        
except Exception as e:
    # Print any exception that occurs
    print(f"An error occurred: {e}")
#model.load_weights('C:/Users/liamm/Downloads/Emotion_Detector_AOL')
'''