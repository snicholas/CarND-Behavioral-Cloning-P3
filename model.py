import os
import csv
import cv2
import numpy as np
import sklearn
from  math import ceil
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Lambda, Conv2D, Activation, Flatten, Dense, Dropout, MaxPooling2D

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                if str(batch_sample[3])!='':

                    name = '/opt/tmp/IMG/'+batch_sample[0].split('\\')[-1]
                    
                    center_image = cv2.imread(name)
                    center_flipped = cv2.flip(center_image,1)
                    center_angle = float(batch_sample[3])                
                    
                    name = '/opt/tmp/IMG/'+batch_sample[1].split('\\')[-1]
                    left_image = cv2.imread(name)
                    left_flipped = cv2.flip(left_image,1)
                    left_angle = center_angle + 0.2

                    name = '/opt/tmp/IMG/'+batch_sample[2].split('\\')[-1]
                    right_image = cv2.imread(name)
                    right_flipped = cv2.flip(right_image,1)
                    right_angle = float(batch_sample[3]) - 0.2

                    images.append(center_image)
                    angles.append(center_angle)

                    images.append(center_flipped)
                    angles.append(-center_angle)

                    images.append(left_image)
                    angles.append(left_angle)

                    images.append(left_flipped)
                    angles.append(-left_angle)

                    images.append(right_image)
                    angles.append(right_angle)

                    images.append(right_flipped)
                    angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)



samples = []
with open('/opt/tmp/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
            
# Set our batch size
batch_size=32

# set up lambda layer
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3),output_shape=(160,320,3)))
# set up cropping2D layer
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

ch, row, col = 3, 90, 320  # Trimmed image format

model.add(Conv2D(64,3,input_shape=(row,col,ch),padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))

model.add(Conv2D(128,5,padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))

model.add(Conv2D(128,5,padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))

model.add(Conv2D(64,5,padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))

model.add(Conv2D(64,5,padding="same"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(1))




train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=10, verbose=1)

model.save('model.h5')