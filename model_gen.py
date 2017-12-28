import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D

from sklearn.utils import shuffle

samples = []

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
    
model_create = create_model_6
model_name = 'model_6_'

data_version = '2'
csv_file_name = '../simulator/data2/driving_log.csv'
img_file_path = '../simulator/data2/IMG/'

with open(csv_file_name) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)



def get_images_and_measurements(inner_line, steering_correction):
    inner_images = []
    inner_measurements = []

    for index in range(0,3):                
        steering_angle = float(inner_line[3])        
        #left image
        if index == 1:            
            steering_angle = steering_angle + steering_correction
        #right image
        elif index == 2:            
            steering_angle = steering_angle - steering_correction
        #else - center

        source_path = inner_line[index]
        filename = source_path.split('/')[-1]    
        current_path = img_file_path + filename
        image = cv2.imread(current_path)
        inner_images.append(image)
        measurement = steering_angle
        inner_measurements.append(measurement)

        inner_images.append(cv2.flip(image, 1)) #flip image
        inner_measurements.append(measurement * -1.0) #invert value

    return (inner_images, inner_measurements)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                imgs, meas = get_images_and_measurements(batch_sample, 0.2)
                images.extend(imgs)
                angles.extend(meas)
                # name = './IMG/'+batch_sample[0].split('/')[-1]
                # center_image = cv2.imread(name)
                # center_angle = float(batch_sample[3])
                # images.append(center_image)
                # angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

returned_model = model_create()

samples_per_epoch = len(train_samples) * 6
nb_val_samples = len(validation_samples) * 6
nb_epoch = 5

returned_model.compile(loss='mse', optimizer='adam')
history = returned_model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, validation_data=validation_generator, nb_val_samples=nb_val_samples, nb_epoch=nb_epoch, verbose=2)


# returned_model.compile(loss='mse', optimizer='adam')
# history = returned_model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=number_of_epochs,verbose=2)


returned_model.save('X-5e-' + model_name + data_version + '.h5')
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