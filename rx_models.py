import os
from rx_config import *
# from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from keras.models import save_model, load_model
from datetime import datetime


def dense_nn_bpsk():
    model = Sequential()
    model._name = 'dense_nn_bpsk'
    model.add(Dense(4, input_shape=(1,), activation='relu'))
    # model.add(Dense(2, activation='relu'))
    # model.add(Dense(1, activation='tanh'))
    model.add(Dense(1, activation=tf.keras.activations.hard_sigmoid))
    # https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid
    # Hard sigmoid
    # if x < -2.5: return 0
    # if x > 2.5: return 1
    # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def save_mdl(model):
    damga = datetime.utcnow()
    uid = model.name + '_' + damga.strftime('%Y%b%d_%H%M')
    # model.save(''+uid)
    if not os.path.isdir('models'):
        os.mkdir('models')
    save_model(model, filepath='models/'+uid, overwrite=True, save_format='tf')
