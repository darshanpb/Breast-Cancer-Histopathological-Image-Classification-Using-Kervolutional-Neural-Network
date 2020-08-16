from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import SeparableConv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import Adagrad
from keras.utils import np_utils
import os
from KernelConv import *
from PIL import Image
from skimage import transform
#from keras.utils import plot_model
from keras_sequential_ascii import keras2ascii

class Model:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        channelDim = -1

        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            channelDim = 1


        model.add(KernelConv2D(32,3,padding='same',input_shape=inputShape))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(KernelConv2D(64,3,padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(KernelConv2D(64, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(KernelConv2D(128,3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(KernelConv2D(128, 3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(KernelConv2D(128,3, padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channelDim))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model
# Define a flask app

def model1():
    numEpochs = 50
    lrRate = 1e-2
    lrRateDecay = lrRate/numEpochs
    model = Model.build(width=48, height=48, depth=3, classes=2)
    opt = Adagrad(lr=lrRate, decay=lrRateDecay)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    keras2ascii(model)
    #plot_model(model, to_file='model.png')
    return model
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model40x.h5'

model=model1()
model.load_weights(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Check http://127.0.0.1:5000/')


def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (256, 256, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

 #image = load('my_file.jpg')
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(48, 48))#48

    # Preprocessing the image
    x = img
    x = np.array(x).astype('float32')/255
    x = transform.resize(x, (48,48, 3))
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    clas = preds.argmax(axis=-1)
    return clas


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        pred_class = preds  # ImageNet Decode
        result = str(pred_class[0])
        if result=="0":
            result="Benign"
        else :
            result="Malignant"
                      # Convert to string
        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
