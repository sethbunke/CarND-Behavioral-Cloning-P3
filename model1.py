import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
import sklearn

epochs = 10
batch_size = 32
use_old_model = False
model_file_name = "model.h5"
old_model_file_name = "model_allData.h5"
data_dir = '../simulator/data' #"/home/alex/Documents/sim-data/data"
# validation_split = 0.2    # use 20% as validation data

#read data
samples = []
csv_file_name = data_dir + "/driving_log.csv"
with open(csv_file_name) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

def random_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image[:, :, 2] = image[:, :, 2] * random_bright
    image[:, :, 2][image[:, :, 2] > 255] = 255
    image = np.array(image, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                # load images
                center_image = cv2.imread(data_dir + "/IMG/" + batch_sample[0].split('/')[-1])
                left_image = cv2.imread(data_dir + "/IMG/" + batch_sample[1].split('/')[-1])
                right_image = cv2.imread(data_dir + "/IMG/" + batch_sample[2].split('/')[-1])

                # select region
                center_image = center_image[50:160, :, :]
                left_image = left_image[50:160, :, :]
                right_image = right_image[50:160, :, :]

                # resize images
                center_image = cv2.resize(center_image, dsize=(60, 60))
                left_image = cv2.resize(left_image, dsize=(60, 60))
                right_image = cv2.resize(right_image, dsize=(60, 60))

                # random brightness
                center_image = random_brightness(center_image)
                left_image = random_brightness(left_image)
                right_image = random_brightness(right_image)

                # create flipped images
                center_image_flipped = cv2.flip(center_image, 1)
                left_image_flipped = cv2.flip(left_image, 1)
                right_image_flipped = cv2.flip(right_image, 1)

                # append images and corresponding steering angles to data sets
                images.append(center_image)
                angles.append(float(batch_sample[3]))
                '''images.append(left_image)
                angles.append(float(batch_sample[3]) + 0.2)
                images.append(right_image)
                angles.append(float(batch_sample[3]) - 0.2)

                # append flipped images - steering angle multiplied by -1
                images.append(center_image_flipped)
                angles.append(float(batch_sample[3]) * -1.0)
                images.append(left_image_flipped)
                angles.append(-1 * (float(batch_sample[3]) + 0.2))
                images.append(right_image_flipped)
                angles.append(-1 * (float(batch_sample[3]) - 0.2))'''

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


if use_old_model:
    model = load_model(old_model_file_name)
    print("using old model " + old_model_file_name)
else:
    print("creating new model " + model_file_name)
    # model = Sequential([
    #     Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(60, 60, 3)),
    #     Convolution2D(24,5,5, subsample=(2, 2), activation='relu'),
    #     Convolution2D(36,5,5, subsample=(2, 2), activation='relu'),
    #     Convolution2D(48,5,5, subsample=(2, 2), activation='relu'),
    #     Convolution2D(64,3,3, activation='relu'),
    #     Convolution2D(64,3,3, activation='relu'),
    #     # Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation="relu"),
    #     # Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation="relu"),
    #     # Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), padding='valid', activation="relu"),
    #     # Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', activation="relu"),
    #     Flatten(),
    #     Dense(100),
    #     Dropout(0.5),
    #     Dense(50),
    #     Dropout(0.5),
    #     Dense(10),
    #     Dropout(0.5),
    #     Dense(1)
    # ])

    model = Sequential()
    #normalize and mean centering
    #initial image is 160 x 320 x 3
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    #crop image: 70 from the top and 25 from the bottom, 0 from the left and 0 from the right
    #model.add(Cropping2D(cropping=((70, 25), (0, 0))))
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
    
    model.compile(loss="mse", optimizer="adam")

n_samples = len(samples)
print("number of samples: " + str(n_samples))

train_generator = generator(samples, batch_size=batch_size)
X_batch, y_batch = train_generator.next()
print(y_batch.shape)

history = model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=n_samples / batch_size, verbose=2)

model.save(model_file_name)
print("model saved as " + model_file_name)

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