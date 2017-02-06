import argparse
import base64
import numpy as np
import socketio
import eventlet.wsgi
from flask import Flask
from io import BytesIO
from model import get_vgg_based_model

# Fix error with Keras and TensorFlow
import tensorflow as tf
from scipy.misc import imresize, imread

tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
image_shape = (160, 320, 3)

def normalize_grayscale(image_data):
    a = -1
    b = 1
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = imread(BytesIO(base64.b64decode(imgString)))
    image = normalize_grayscale(image)
    image = np.array([image])

    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(image, batch_size=1))
    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    throttle = 0.15
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()

    # Saving to the model to json with Keras does not work for transfer learning
    # with open(args.model, 'r') as jfile:
    #     # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
    #     # then you will have to call:
    #     #
    #     #   model = model_from_json(json.loads(jfile.read()))\
    #     #
    #     # instead.
    #     model = model_from_json(jfile.read())

    model = get_vgg_based_model()
    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
