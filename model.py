import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D

lines = []

#NVIDIA
def create_model_6():
    model = Sequential()
    #normalize and mean centering
    #initial image is 160 x 320 x 3
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    #crop image: 70 from the top and 25 from the bottom, 0 from the left and 0 from the right
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    model.add(Convolution2D(24,5,5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2, 2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Convolution2D(6,5,5,activation='relu'))
    # model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
    

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
    #initial image is 160 x 320 x 3
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    #crop image: 70 from the top and 25 from the bottom, 0 from the left and 0 from the right
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
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

model_create = create_model_6
model_name = 'model_6_'
#test to see if model bombs - see error message 
try:
    model_create()
except Exception as inst:
    print(type(inst))    # the exception instance
    print(inst.args)     # arguments stored in .args
    print(inst)          # __str__ allows args to be printed directly,
                         # but may be overridden in exception subclasses

def augment_images(images, measurements):
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image, 1)) #flip image
        augmented_measurements.append(measurement * -1.0) #invert value
    return augmented_images, augmented_measurements

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

aug_data_images, aug_data_measure = augment_images(images, measurements)

for image, measurement in zip(aug_data_images, aug_data_measure):
    images.append(image)
    measurements.append(measurement)

# images.concat(aug_data_images)
# measurements.concat(aug_date_measure)

X_train = np.array(images)
y_train = np.array(measurements)

# X_train = np.append(images, aug_data_images)
# y_train = np.append(measurements, aug_data_measure)

returned_model = model_create()

returned_model.compile(loss='mse', optimizer='adam')
history = returned_model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5,verbose=2)


returned_model.save(model_name + '.h5')
#returned_model.save('test_model.h5')

def show_history(history_object):
    ### print the keys contained in the history object
    print(history_object.history.keys())

    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

show_history(history)


