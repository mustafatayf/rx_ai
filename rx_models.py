import os
from rx_config import *
# from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.models import save_model, load_model
from keras.metrics import BinaryAccuracy, F1Score, Precision, Recall
# from keras.optimizers import SGD, Nadam
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


def lstm_bpsk(isi=7, batch_size=32):
    # (n_samples, time_steps, features)

    model = Sequential()
    model._name = 'lstm_bpsk'
    # https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM
    # https://stackoverflow.com/questions/74811755/input-0-of-layer-lstm-is-incompatible-with-the-layer-expected-shape-1-none,
    # https://stackoverflow.com/a/74812987
    model.add(LSTM(32,  input_shape=(2*isi+1, 1),
                   return_sequences=True,
                   # stateful=True,
                   batch_input_shape=(batch_size, 2*isi+1, 1)))  # batch_size, timesteps, data_dim
    model.add(Dropout(rate=0.2))
    # https://stackoverflow.com/a/47505918
    # model.add(LSTM(units=8, return_sequences=True, stateful=True))
    # model.add(Dense(21, activation='relu'))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation=tf.keras.activations.hard_sigmoid))
    # https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid
    # Hard sigmoid
    # if x < -2.5: return 0
    # if x > 2.5: return 1
    # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5

    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss='mse',
                  metrics=[BinaryAccuracy(), F1Score()])
    # https://stackoverflow.com/a/58954176
    # model.summary()

    return model


def gru_bpsk(isi=7, batch_size=32):
    # (n_samples, time_steps, features)

    model = Sequential()
    model._name = 'gru_bpsk'
    # https://analyticsindiamag.com/lstm-vs-gru-in-recurrent-neural-network-a-comparative-study/
    model.add(GRU(32,  input_shape=(2*isi+1, 1)))
    model.add(Dropout(rate=0.2))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation=tf.keras.activations.hard_sigmoid))
    # https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid
    # Hard sigmoid
    # if x < -2.5: return 0
    # if x > 2.5: return 1
    # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5

    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss='mse',
                  metrics=[BinaryAccuracy(), F1Score()])
    # model.summary()

    return model


def gru_qpsk(isi=7, batch_size=32):
    # (n_samples, time_steps, features)

    model = Sequential()
    model._name = 'gru_qpsk'

    model.add(GRU(32,  input_shape=(2*isi+1, 2)))
    model.add(Dropout(rate=0.2))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation=tf.keras.activations.hard_sigmoid))

    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss='categorical_crossentropy',
                  metrics=[BinaryAccuracy(), F1Score()])
    # model.summary()

    return model


def dense_nn_qpsk():
    model = Sequential()
    model._name = 'dense_nn_qpsk'
    model.add(Dense(4, input_shape=(2,), activation='linear'))  # input shape(2,N) : (real, imag)
    # model.add(Dense(8, activation='linear'))
    model.add(Dense(4, activation=tf.keras.activations.hard_sigmoid))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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
    save_model(model, filepath='models/' + uid, overwrite=True, save_format='tf')
    print('{name} is saved to models/ folder..'.format(name=uid))

# references

# Activation functions:
    # Hard sigmoid
    # https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid
    # if x < -2.5: return 0
    # if x > 2.5: return 1
    # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5

# GRU and LSTM
    # https://analyticsindiamag.com/lstm-vs-gru-in-recurrent-neural-network-a-comparative-study/


