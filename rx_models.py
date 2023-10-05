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


def dense_nn_qpsk():
    model = Sequential()
    model._name = 'dense_nn_qpsk'
    model.add(Dense(4, input_shape=(2,), activation='linear'))  # input shape(2,N) : (real, imag)
    # model.add(Dense(8, activation='linear'))
    # model.add(Dense(1, activation='tanh'))
    model.add(Dense(4, activation=tf.keras.activations.hard_sigmoid))
    # model.add(Dense(4, activation='relu'))
    # https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid
    # Hard sigmoid
    # if x < -2.5: return 0
    # if x > 2.5: return 1
    # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5
    # categorical_crossentropy
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def dense_nn_deep():
    model = Sequential()
    model._name = 'dense_nn_bpsk'
    model.add(Dense(8, input_shape=(1,), activation='relu'))
    model.add(Dense(2, activation='relu'))
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
