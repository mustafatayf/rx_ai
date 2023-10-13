"""AI based RX
name: model training module
status: draft
version: 0.01 (05 October 2023)
"""
import wandb
import numpy as np
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
from ber_util import gen_data
from rx_utils import get_data, show_train, check_data, prep_ts_data, get_song_data
from rx_models import base_bpsk, dense_nn_qpsk, dense_nn_deep, lstm_bpsk, gru_bpsk, gru_qpsk, save_mdl, song_bpsk

TAU = 0.80  # 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
SNR = 8  # 0, 1, 2, ..., 10, nonoise  # noqa
IQ = 'bpsk'  # bpsk, qpsk   # noqa
nos = int(1e7)
MODEL = 'gru'  # 'base' 'dense', 'lstm', 'gru'  # TODO review model naming and check consistency
L = 512  # Length of symbol block/window, {(L-m)//2, m, (L-m)//2} see: song2019
m = 32  # number of symbols to process at each inference, amount of shift see: song2019

model = song_bpsk(L=L, m=m)

# train parameters
epochs = 300
batch_size = 8000  # reduce batch size for big models...
NoD = 1.6 * 10 ** 9
val_split = 0.1

DATA_MODE = 'load'  # 'load', 'generate'
WB_ON = False

if DATA_MODE == 'load':
    # Load the training data
    X_i, y_i = get_data(name='data_{iq}/tau{tau:.2f}_snr{snr}_{iq}'.format(iq=IQ, tau=TAU, snr=SNR), NoD=NoD)
else:
    raise NotImplementedError
    # TODO : include data generation flow
    # syms, bits = gen_data(n=nos, mod=IQ, seed=43523) ('bpsk', 'qpsk')
    # up-sample
    # apply FTN  (tau)
    # add AWGN noise (snr)
    # apply matched filter at RX
    # down-sample (subsample)
    # X_i, y_i

# call the model if the model is not initialized yet
if model == '':
    model = eval(MODEL + '_' + IQ)(batch_size=batch_size)  # TODO fix security risk // CAUTION
    # model = gru_qpsk(batch_size=batch_size)

print(model.summary())
confs = {
    'loss': model.loss,
    'optimizer_class_name': model.optimizer.__class__.__name__,
    'optimizer_config': model.optimizer.get_config(),
    'metrics': model.compiled_metrics._metrics,  # noqa
}
for k, v in confs.items():
    print(k, v)

# DATA pre-processing

# [DEBUG] noise generation
# Xs = add_awgn(y*2-1, snr=10)

# [DEBUG] data control
# check_data(rx_data=X_i, ref_bit=y_i, modulation=IQ)

if 'song' in model.name:
    X, y = get_song_data(X_i, y_i, L=L, m=m)
else:
    # single to time series data
    if 'lstm' in model.name or 'gru' in model.name:
        X = prep_ts_data(X_i)
    else:
        X = X_i

    # update label type to float for evaluating performance metrics
    y = y_i.astype(np.float16)

# Weight and Biases integration
# https://docs.wandb.ai/tutorials/keras_models
# TODO set configurations
configs = dict(
    tau=TAU, snr=SNR,
    modulation=IQ,
    model=MODEL,
    batch_size=batch_size,
    data_size=len(y),
    validation_split=val_split,
    # learning_rate = 1e-3,
    epochs=epochs
)
if WB_ON:
    wandb.init(project='rx_ai',
               config=configs
               )

    callbacks = [WandbMetricsLogger(log_freq='epoch',
                                    initial_global_step=0),
                 WandbModelCheckpoint('models/',
                                      save_best_only=False
                                      )
                 ]  # WandbCallback()
else:
    callbacks = None

tf.keras.backend.clear_session()
history = model.fit(X, y,
                    validation_split=val_split,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    )

save_mdl(model, tau=TAU, history=history)
# plot train process
show_train(history)
# TODO save train results

if WB_ON:
    wandb.finish()

# results = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(X[:10, :])
# y_true = y[:10, :]


# references

# https://wandb.ai/ayush-thakur/dl-question-bank/reports/LSTM-RNN-in-Keras-Examples-of-One-to-Many-Many-to-One-Many-to-Many---VmlldzoyMDIzOTM


# info
# song2019, "Receiver Design for Faster-than-Nyquist Signaling: Deep-learning-based Architectures"
# 0 1 2 ... 9 10 11 ..... 107 108 109 ...   128
# 107 108 109 ...  .. ... 215 216 217 ...   236 ?
