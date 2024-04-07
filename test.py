import argparse
import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from io import BytesIO
from PIL import Image
import cv2

from keras.models import load_model
from keras.losses import MeanSquaredError


# Initialize Flask application and socketio server
app = Flask(__name__)
sio = socketio.Server()
model = None
prev_image_array = None

# Define image preprocessing function
def preprocess_image(image):
    image = image[60:135,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image,  (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image/255
    return image

# Define socketio event for telemetry data
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image_str = data["image"]
        image = Image.open(BytesIO(base64.b64decode(image_str)))
        image_array = np.asarray(image)
        processed_image = preprocess_image(image_array)
        steering_angle = float(model.predict(np.array([processed_image]))[0])
        throttle = 0.2
        print('{} {} {}'.format(steering_angle, throttle, speed))
        send_control(steering_angle, throttle)
    else:
        # When no data is received, stop the car
        sio.emit('manual', data={}, skip_sid=True)

# Define socketio event for connecting to the simulator
@sio.on('connect')
def connect(sid, environ):
    print("Connected")
    send_control(0, 0)

# Define function to send control commands to the simulator
def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    }, skip_sid=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Udacity Self Driving Car Simulator')
    parser.add_argument('model', type=str, help='Path to model.h5 file')
    args = parser.parse_args()

    # Load the trained model
    model = load_model(args.model)

    # Wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # Deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
