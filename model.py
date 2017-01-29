import csv
import json
import os.path
from keras.applications import VGG16
from keras.engine import Model
from keras.layers import Input, BatchNormalization, Convolution2D, Dense, Flatten, MaxPooling2D, ELU, Dropout, \
    Convolution1D, Activation, MaxPooling1D, Lambda, Conv1D
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
from pathlib import Path
from scipy.misc import imread, imresize, imsave
import numpy as np
from sklearn.utils import shuffle


image_shape = (160, 320, 3)

def normalize_grayscale(image_data):
    a = -1
    b = 1
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )


def load_image(image_path, reverse=False):
    image = imread(image_path)
    if reverse:
        image = np.fliplr(image)
    image = preprocess_image(image)
    return image


def preprocess_image(image):

    # image = image[60::]
    # image = imresize(image, size=image_shape)
    image = normalize_grayscale(image)
    image = np.array([image])

    return image


def read_data(data):
    i = -1
    while True:
        i += 1
        if i == len(data):
            i = 0
        reverse = data[i][2]
        if reverse:
            yield load_image(data[i][0], reverse), np.array([-float(data[i][1])])
        else:
            yield load_image(data[i][0]), np.array([float(data[i][1])])


def read_data_file(file_path):
    data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if os.path.exists(row[0]):
                data.append([row[0], row[3], False])
                data.append([row[0], row[3], True])
            else:
                print(row[0])
    return data


def get_vgg_based_model():

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)

    # Remove the last block of layers in the VGG
    [base_model.layers.pop() for _ in range(4)]
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []

    # Because the data in VGG model is different from what our task is, we disable the training on the base model
    for layer in base_model.layers:
        layer.trainable = False

    layer = base_model.outputs[0]
    layer = Convolution2D(512, 3, 3, subsample=(2, 2), activation='relu', border_mode='same')(
        layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 2), activation='relu', border_mode='same')(
        layer)
    layer = Convolution2D(512, 3, 3, subsample=(1, 2), activation='relu', border_mode='same')(
        layer)

    layer = Flatten()(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(2048, activation='relu')(layer)
    layer = Dropout(.2)(layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dropout(.5)(layer)
    layer = Dense(1, activation='tanh')(layer)

    model = Model(input=base_model.input, output=layer)

    adam = Adam(0.00001)
    model.compile(optimizer=adam, loss="mse")

    return model


def get_comma_model():

    ch, row, col = 3, 160, 320  # camera format
    model = Sequential()
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=image_shape))
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

    return model


def get_simple_model():

    model = Sequential()
    model.add(Conv1D(32, 3, input_shape=image_shape, border_mode='same', activation='relu'))
    model.add(Conv1D(64, 3, border_mode='same', activation='relu'))
    model.add(Conv1D(128, 3, border_mode='same', activation='relu'))
    model.add(Conv1D(256, 3, border_mode='same', activation='relu'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='tanh'))

    model.compile(optimizer=Adam(lr=0.00001), loss='mse')

    return model


def load_data(shuffle_data=True):

    data = []
    data += read_data_file('driving_log.csv')

    if shuffle_data:
        data = shuffle(data)

    return data


def split_data(data):
    """
    Split the data in train/validation
    Test set is missed because it did not show valuable information during the training experimentation

    :param data:
    :return:
    """
    validation_index = int(len(data) * 0.1)
    validation_data = data[0:validation_index]
    train_data = data[validation_index:]

    return train_data, validation_data


def train_generator(model, train_data, validation_data, continuation_learning = False, epoch=4):

    if continuation_learning:
        model.load_weights('model.h5')

    history = model.fit_generator(read_data(train_data), validation_data=read_data(validation_data),
                                  samples_per_epoch=len(train_data), nb_epoch=4, nb_val_samples=len(validation_data),
                                  nb_worker=1)
    return history


def save_model_and_weights(model):

    model_json = model.to_json()
    with open('model.json', 'w') as outfile:
        outfile.write(model_json)
    model.save_weights('model.h5', overwrite=True)

if __name__ == '__main__':

    data = load_data()
    train_data, validation_data = split_data(data)

    model = get_vgg_based_model()

    history = train_generator(model, train_data, validation_data)

    save_model_and_weights(model)
