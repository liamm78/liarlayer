import sklearn
import tensorflow as tf  # TensorFlow and Keras for building and training the neural network
from sklearn.preprocessing import LabelEncoder

import imutils
from imutils import paths
import matplotlib.pyplot as plt  # For plotting results
from keras.utils import to_categorical

import numpy as np  # For numerical operations
import cv2
import os
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Flatten
from keras.layers import Dense, Dropout, BatchNormalization
from keras import backend as K
from keras.regularizers import l2

df = pd.read_csv('fer2013.csv')
#print(df.head)

training_data = df.loc[df.iloc[:, 2] == 'Training']
test_data = df.loc[(df.iloc[:, 2] == 'PublicTest') | (df.iloc[:, 2] == 'PrivateTest')]
train_images = training_data['pixels'].apply(lambda x: np.array(x.split(), dtype=int))
train_labels = training_data['emotion']

test_images = test_data['pixels'].apply(lambda x: np.array(x.split(), dtype=int))
test_labels = test_data['emotion']
# Now 'pixel_data' contains NumPy arrays of pixel values for training samples
print(train_images)
print(test_images)
print(len(train_images))
print(len(test_images))
print(len(train_labels))
print(len(test_labels))

# Convert 1D Panda Series into 2D Panda Series
train_images = train_images.apply(lambda x: np.array(x).reshape(48, 48, 1))
test_images = test_images.apply(lambda x: np.array(x).reshape(48, 48, 1))
print("THIS WORKED")
# Convert 2D Panda Series into 3D Numpy Array
train_images = np.stack(train_images, axis=0)
test_images = np.stack(test_images, axis=0)

train_images = train_images / 255.0
test_images = test_images / 255.0


# GEt the Weight

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)

# Create a dictionary to map class indices to their corresponding weights
weight_dict = {class_idx: weight for class_idx, weight in enumerate(class_weights)}
print(weight_dict)

# One hot encode the labels 

train_labels = to_categorical(train_labels, num_classes=7)
test_labels = to_categorical(test_labels, num_classes=7)


#Create the CNN. 3 Conv layers, 2 pooling, 1 dense
model=Sequential() # Sequential uses layers to build
 # 1st CONV block
model.add(Conv2D(32, kernel_size=(3,3), activation ='relu', input_shape = (48,48,1)))
model.add(Conv2D(64, kernel_size =(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size =(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Conv2D(256, kernel_size =(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax')) #connects 500 neurons to 7 classes




# Balance the class weights with compute_class_weight

model.compile(loss=['categorical_crossentropy'],optimizer='adam',metrics=['accuracy'])
H = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), class_weight = weight_dict, batch_size=64,epochs=30)
#predict = model.predict(testX, batch_size=64)

#print('sucess')