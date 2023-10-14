"""AI based RX

Channel Estimation models
    input,  RX symbol data (single frame, N symbol)
    output, channel tap values

last update: 12 october 2023, ..
"""
import os
import pandas as pd
from rx_config import *
# from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense, LSTM, Dropout, GRU, BatchNormalization
from keras.models import save_model, load_model
from keras.metrics import Accuracy, MeanSquaredError
from keras.optimizers import Adam, SGD
from datetime import datetime


def ce_temel(init_lr=0.001):
    model = Sequential()
    model._name = 'ce_temel'
    model.add(Dense(32, input_shape=(52,), activation='relu'))  # 26 symbol
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(16, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(12, activation='linear'))  # 6 channel tap value

    model.compile(optimizer=Adam(learning_rate=init_lr),
                  loss='mean_squared_error',  # 'mse',
                  metrics=[Accuracy(),
                           # MeanSquaredError()
                           ])

    return model

