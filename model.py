import csv
import json

from keras.layers import Input, BatchNormalization, Convolution2D, Dense, Flatten, MaxPooling2D, ELU, Dropout, \
    Convolution1D, Activation, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from scipy.misc import imread, imresize, imsave
import numpy as np
import pickle
from sklearn.utils import shuffle


image_shape = (16*2, 32*2)
def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


def load_image(image_path):
    image = imread(image_path, flatten=True)
    image = imresize(image, size=image_shape)
    image = normalize_grayscale(image)
    return np.array([image])


def read_data(data):
    i = -1
    while True:
        i += 1
        if i == len(data):
            i = 0
        yield load_image(data[i][0]), np.array([data[i][1]])


def read_data_file(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            data.append([row[0], row[3]])
    return data


data = read_data_file('driving_log.csv')
data = shuffle(data)
# data = data[:int(len(data)/2)]
validation_index = int(len(data)*0.2)
test_index = validation_index + int(len(data)*0.1)
validation_data = data[0:validation_index]
test_data = data[validation_index:test_index]
train_data = data[test_index:]

model = Sequential()
model.add(Convolution1D(16, 3, border_mode="same", input_shape=image_shape))
model.add(Activation('relu'))
model.add(Convolution1D(32, 3, border_mode="same"))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Dropout(.5))
model.add(Activation('relu'))
model.add(Dense(1))

adam = Adam(0.0001)
model.compile(optimizer=adam, loss="mse")
history = model.fit_generator(read_data(train_data), validation_data=read_data(validation_data),
                              samples_per_epoch=len(train_data), nb_epoch=10, nb_val_samples=len(validation_data))
test_lost = model.evaluate_generator(read_data(test_data), len(test_data))
print("Test lost: ", test_lost)
model_json = model.to_json()
with open('model.json', 'w') as outfile:
    outfile.write(model_json)
model.save_weights('model.h5')
