import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D

lines = []

def create_model_5():
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape = (3,64,64)))
    
    model.add(Conv2D(24,(5,5), strides=(2,2), activation='relu', input_shape=(3,64,64), data_format='channels_first', padding='same'))
    model.add(Conv2D(36,(5,5), strides=(2,2), activation='relu', input_shape=(3,64,64), data_format='channels_first', padding='same'))
    model.add(Conv2D(48,(5,5), strides=(2,2), activation='relu', input_shape=(3,64,64), data_format='channels_first', padding='same'))
    model.add(Conv2D(64,(3,3), strides=(2,2), activation='relu', input_shape=(3,64,64), data_format='channels_first', padding='same'))
    model.add(Conv2D(64,(3,3), strides=(2,2), activation='relu', input_shape=(3,64,64), data_format='channels_first', padding='same'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))     
    return model

def create_model_4():
    input_shape=(160,320,3)
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Convolution2D(32,3,3, activation='relu',padding='same')) #, input_shape=input_shape))
    model.add(Convolution2D(64,3,3, activation='relu',padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))
    return model

def create_model_3():
    model = Sequential()
    #normalize and mean centering
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def create_model_2():
    model = Sequential()
    #normalize and mean centering
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Flatten())
    model.add(Dense(1))
    return model

def create_model_1():
    model = Sequential()
    #width, height, and depth of images
    model.add(Flatten(input_shape=(160,320,3)))
    model.add(Dense(1))
    return model

try:
    create_model_3()
except Exception as inst:
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    print(inst)          # __str__ allows args to be printed directly,
                         # but may be overridden in exception subclasses

with open('../simulator/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../simulator/data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

returned_model = create_model_3()

returned_model.compile(loss='mse', optimizer='adam')
returned_model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10,verbose=1)

returned_model.save('test_model.h5')



