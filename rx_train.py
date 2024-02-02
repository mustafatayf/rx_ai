"""AI based RX
name: model training module
status: draft, G_delay 1 added,
version: 0.02 (27 Jabuary 2024)
"""
import os

import pandas as pd
import wandb
import numpy as np
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
import tensorflow as tf
from ber_util import gen_data, add_awgn
from rx_utils import get_data, show_train, check_data, prep_ts_data, get_song_data
from rx_models import gru_temel, base_bpsk, dense_nn_qpsk, dense_nn_deep, lstm_bpsk, save_mdl, song_bpsk
from rx_config import init_gpu
from constants import h_81, hh_21

init_gpu()

TAU = 0.60  # 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
SNR = 'NoNoise'  # 0, 1, 2, ..., 10, NoNoise  # noqa
IQ = 'bpsk'  # bpsk, qpsk   #

init_lr = 0.0005
model = gru_temel(init_lr=init_lr)  # 'base' 'dense', 'lstm', 'gru'  # TODO review model naming and check consistency

# model = ''
# L = 512  # Length of symbol block/window, {(L-m)//2, m, (L-m)//2} see: song2019
# m = 32  # number of symbols to process at each inference, amount of shift see: song2019
# model = song_bpsk(L=L, m=m)

# train parameters
epochs = 70
batch_size = 8192  # reduce batch size for big models...
NoS = int(1e6)  # number of symbols
val_split = 0.1

DATA_MODE = 'generate'  # 'load', 'generate' 'load_npy
WB_ON = True

ISI = 3  # bir sembole etki eden komşu sembol sayısı, örneğin ISI = 5; [ . . . . . S . . . . .], toplam 11 kayıt
FS = 10
G_DELAY = 1
step = int(TAU * FS)
if G_DELAY == 4:
    hPSF = np.array(h_81).astype(np.float16)  # TODO G_DELAY FS based h generation
elif G_DELAY == 1:
    hPSF = np.array(hh_21).astype(np.float16)  # TODO G_DELAY FS based h generation
else:
    raise NotImplementedError

assert np.array_equal(hPSF, hPSF[::-1]), 'symmetry mismatch!'

if DATA_MODE == 'load':
    # Load the training data
    X_i, y_i = get_data(name='data_{iq}/tau{tau:.2f}_snr{snr}_{iq}'.format(iq=IQ, tau=TAU, snr=SNR), NoD=NoS)
    if IQ != 'bpsk':
        # compact data into 1D, no need to consider real(I) and imaginary(Q) parts as separate dimensions
        X_i = np.reshape(X_i, (-1, ))

elif DATA_MODE == 'load_npy':
    try:
        X_i = np.load('data/snr{snr}_{iq}_tau{tau:.1f}_X_i.npy'.format(snr=SNR, iq=IQ, tau=TAU))
        y_i = np.load('data/snr{snr}_{iq}_tau{tau:.1f}_y_i.npy'.format(snr=SNR, iq=IQ, tau=TAU))
    except FileNotFoundError:
        print('data not found try to use generate option')
else:
    # raise NotImplementedError
    # TODO : include data generation flow
    assert NoS < int(1e7)+1, 'too many data to generate, load from file'
    # [SOURCE]  Data Generation
    data, bits = gen_data(n=NoS, mod=IQ, seed=43523)  # IQ options: ('bpsk', 'qpsk')
    # [TX]   up-sample
    # extend the data by up sampling (in order to be able to apply FTN)
    s_up_sampled = np.zeros(step * len(data), dtype=np.float16)
    s_up_sampled[::step] = data
    # [TX]  apply FTN  (tau)
    # apply the filter
    tx_data = np.convolve(hPSF, s_up_sampled)
    # [CHANNEL] add AWGN noise (snr)
    # Channel Modelling, add noise
    rch = add_awgn(inputs=tx_data, snr=SNR, seed=1234)
    # [RX]   apply matched filter
    mf = np.convolve(hPSF, rch)
    # [RX]  down-sample (subsample)
    # p_loc = 2 * G_DELAY * FS  # 81 for g_delay=4 and FS = 10,
    # 4*10=40 from first conv@TX, and +40 from last conv@RX
    # remove additional prefix and suffix symbols due to CONV
    rx_data = mf[2 * G_DELAY * FS:-(2 * G_DELAY * FS):int(TAU * FS)]

    # X_i, y_i
    X_i = rx_data
    y_i = bits

    # if AUTO_SAVE:
    np.save('data/snr{snr}_{iq}_tau{tau:.1f}_X_i.npy'.format(snr=SNR, iq=IQ, tau=TAU), X_i)
    np.save('data/snr{snr}_{iq}_tau{tau:.1f}_y_i.npy'.format(snr=SNR, iq=IQ, tau=TAU), y_i)

    # save to csv
    # import pandas as pd
    # df = pd.DataFrame.from_dict({'y': y_i, 'X': X_i})
    # # # df.to_csv('data/tau0.80_snrNoNoise_bpsk.csv'.format(tau=TAU, iq=IQ, model=model.name), index=False)
    # df.to_csv('data/tau0.60_snrNoNoise_bpsk.csv', index=False)


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

# if 'song' in model.name:
#     X, y = get_song_data(X_i, y_i, L=L, m=m)
# else:
# single to time series data
if 'lstm' in model.name or 'gru' in model.name:
    X = prep_ts_data(X_i, isi=ISI)
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
    model=model.name,
    isi=ISI,
    dropout=0.3,
    optimizer=model.optimizer.get_config(),
    batch_size=batch_size,
    data_source=DATA_MODE,
    num_of_syms=NoS,
    num_of_data=len(y_i),
    validation_split=val_split,
    learning_rate=init_lr,
    epochs=epochs
)
if WB_ON:
    wandb.init(project='rx_ai',
               config=configs
               )

    callbacks = [WandbMetricsLogger(log_freq='epoch',
                                    initial_global_step=0),
                 # WandbModelCheckpoint(filepath=os.getcwd()+'/models/tau{:.2f}_'.format(TAU) + model.name+'_{epoch:02d}',
                 WandbModelCheckpoint(filepath='./models/tau{:.2f}_'.format(TAU) + model.name,
                                      save_best_only=False,
                                      # monitor='val_f1_score',
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
