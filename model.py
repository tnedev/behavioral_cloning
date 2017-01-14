import csv
import json

from keras.layers import Input, BatchNormalization, Convolution2D, Dense, Flatten, MaxPooling2D, ELU, Dropout
from keras.models import Sequential
from scipy.misc import imread
import numpy as np
import pickle
from sklearn.utils import shuffle



def normalize_grayscale(image_data):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


def read_images(data):
    for row in data:
        yield imread(row[0]).astype(np.float32)


def read_outputs(data):
    for row in data:
        yield row[3]


def read_from_pickle():
    images = pickle.load(open("data.p", "rb"))
    return images


def save_data(images):
    pickle.dump(images, open("data.p", "wb"))


def read_data():
    with open('../sim/driving_log.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            yield row


def init_data():
    data = list(read_data())
    X_data = list(read_images(data))
    y_data = list(read_outputs(data))
    X_data, y_data = shuffle(X_data, y_data)

    test_index = int(len(X_data)*0.1)

    X_test = X_data[0:test_index]
    X_train = X_data[test_index:]
    y_test = y_data[0:test_index]
    y_train = y_data[test_index:]

    return np.array(X_train), y_train, np.array(X_test), y_test

X_train, y_train, X_test, y_test = init_data()

model = Sequential()
model.add(BatchNormalization(input_shape=(160,320,3)))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, batch_size=128, nb_epoch=7, validation_split=0.2)
scores = model.evaluate(X_test, y_test, verbose=1)
print(scores)
model_json = model.to_json()
with open('model.json', 'w') as outfile:
    outfile.write(model_json)
model.save_weights('model.h5')