"""AI based RX

Channel Estimation models
    input,  RX symbol data (single frame, N symbol)
    output, channel tap values

last update: 12 october 2023, ..
"""
import os

import numpy as np
import pandas as pd
from rx_config import *
# from tensorflow import keras
from keras import Sequential, Model
from keras.layers import (Input, Dense, LSTM, Dropout, GRU, BatchNormalization,
                          Normalization, Multiply, Add, Subtract, Concatenate)
from keras.models import save_model, load_model
from keras.metrics import RootMeanSquaredError, R2Score, MeanSquaredError, CosineSimilarity
from keras.optimizers import Adam, SGD, RMSprop
from datetime import datetime
from sklearn.metrics import r2_score


def ce_temel(init_lr=0.001):
    model = Sequential()
    model._name = 'ce_temel'
    model.add(Input(shape=(32,)))
    model.add(Dense(10, activation='sigmoid'))  # 11 + 10 + 11 symbol
    # model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(Dense(3, activation='tanh'))  # 6 channel tap value
    model.add(Dense(3, activation='linear'))  # 6 channel tap value

    model.compile(optimizer=RMSprop(learning_rate=init_lr),
                  loss='mse',  # 'mean_squared_error',  # 'mse',
                  # metrics=tf.keras.metrics.R2Score(dtype=np.float16)  # 'cosine_similarity'
                  metrics=[r2_score]
                  # metrics=[CosineSimilarity(),
                  #          [RootMeanSquaredError(),
                  #          R2Score(num_regressors=3)
                  #          tf.keras.metrics.R2Score(dtype=np.float32)
                  #          ]
                  , run_eagerly=True
                  # https://stackoverflow.com/a/74354456
                  )
    return model


def ce_plus(init_lr=0.001):
    pilot_mask = np.concatenate((np.zeros(11), np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1]), np.zeros(11)), axis=0)
    ref = Input(shape=(32,))(pilot_mask)
    in32 = Input(shape=(32,), name='giris')

    nr32 = Normalization()(in32)
    # m1 = Multiply(np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1]), in32[:, 11:21])
    m1 = Multiply(name='mask1')([ref, nr32])
    # a1 = Add(np.array([0, 0, 0, 1, 0, 0, 0, 1, 1, 1]), in32[:, 11:21])
    s1 = Subtract(name='sub1')([m1, ref])
    nd96 = Concatenate(axis=1, name='preprocess')([nr32, m1, s1])

    dens1 = Dense(10, activation='relu')(nd96)  # 11 + 10 + 11 symbol
    # model.add(BatchNormalization())
    drop1 = Dropout(0.1)(dens1)

    nd42 = Concatenate(axis=1, name='giris_islem1')([in32, drop1])

    dens2 = Dense(10, activation='relu')(nd42)

    out3 = Dense(3, activation='linear')(dens2)

    model = Model(in32, out3)
    model._name = 'ce_plus'

    model.compile(optimizer=Adam(learning_rate=init_lr),
                  loss='mae',  # 'mean_squared_error',  # 'mse',
                  metrics=[RootMeanSquaredError(),
                           R2Score()
                           ]
                  )

    return model
