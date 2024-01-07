"""AI based RX for variable tau (VT)

RX models
    input,  RX data (FTN + AWGN, ...)
    output, 0/1 message  bit values

last update: 03 december 2023, ..
"""
import os
import pandas as pd
from rx_config import *
# from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense, LSTM, Dropout, GRU
from keras.models import save_model, load_model
from keras.metrics import BinaryAccuracy, F1Score, Precision, Recall
# from keras.optimizers import SGD, Nadam
from datetime import datetime


def base_bpsk(input_length=250, batch_size=32):
    model = Sequential()
    model._name = 'base_vt_bpsk'
    model.add(Dense(10, input_shape=(input_length,), activation='relu'))
    # model.add(Dense(3, input_shape=(170,), activation='relu'))
    model.add(Dense(10, activation=tf.keras.activations.hard_sigmoid))
    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model

#
# def song_bpsk(L, m, batch_size=32):
#     model = Sequential()
#     model._name = 'song_bpsk'
#     model.add(Dense(320, input_shape=(L, ), batch_size=batch_size, activation='relu'))
#     model.add(Dense(160, activation='relu'))
#     model.add(Dense(80, activation='relu'))
#     model.add(Dense(40, activation='relu'))
#     model.add(Dense(m, activation='tanh'))
#     # model.compile(optimizer='adam', loss='mse', metrics=[BinaryAccuracy(), F1Score()])
#     model.compile(optimizer='adam', loss='mse', metrics='accuracy')
#
#     return model
#

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

    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Nadam(),
                  loss='mse',
                  metrics=[BinaryAccuracy(), F1Score()])
    # https://stackoverflow.com/a/58954176
    # model.summary()

    return model


def gru_temel(isi=7, batch_size=32, init_lr=0.001):
    # (n_samples, time_steps, features)

    model = Sequential()
    model._name = 'gru_temel'
    model.add(Input(shape=(2*isi+1, 1),
                    batch_size=batch_size)
              )
    model.add(GRU(units=2*isi+1,  # dimensionality of OUTPUT space
                  activation='tanh',
                  recurrent_activation='sigmoid',
                  recurrent_dropout=0,
                  unroll=False,
                  use_bias=True,
                  reset_after=True,
                  kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal',
                  bias_initializer='zeros')
              )
    model.add(Dropout(rate=0.3))
    # model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation=tf.keras.activations.hard_sigmoid))

    # model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=tf.keras.optimizers.Nadam(learning_rate=init_lr),
                  loss='mse',
                  metrics=[BinaryAccuracy(), F1Score()])
    # model.summary()

    return model


def gru_plus(isi=7, batch_size=32, init_lr=0.001):
    # (n_samples, time_steps, features)
    model = Sequential()
    model._name = 'gru_plus'
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

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def save_mdl(model, tau, history=None):
    damga = datetime.utcnow()
    uid = 'tau{:.2f}_'.format(tau) + model.name + '_' + damga.strftime('%Y%b%d_%H%M')
    # model.save(''+uid)
    if not os.path.isdir('models'):
        os.mkdir('models')
    save_model(model, filepath='models/' + uid, overwrite=True, save_format='tf')
    print('{name} is saved to models/ folder..'.format(name=uid))

    if history:
        # https://stackoverflow.com/questions/41061457/
        # keras-how-to-save-the-training-history-attribute-of-the-history-object
        # convert the history.history dict to a pandas DataFrame:
        hist_df = pd.DataFrame(history.history)

        # save to json:
        hist_json_file = 'models/{name}/history.json'.format(name=uid)
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)


# references

# Activation functions:
    # Hard sigmoid
    # https://www.tensorflow.org/api_docs/python/tf/keras/activations/hard_sigmoid
    # if x < -2.5: return 0
    # if x > 2.5: return 1
    # if -2.5 <= x <= 2.5: return 0.2 * x + 0.5

# GRU and LSTM
    # https://analyticsindiamag.com/lstm-vs-gru-in-recurrent-neural-network-a-comparative-study/


