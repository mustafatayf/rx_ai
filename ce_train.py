"""AI based RX
name: Channel Estimation model training module
status: draft
version: 0.01 (15 October 2023)
"""
import os
import wandb
import numpy as np
import pandas as pd
import tensorflow as tf
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from ber_util import gen_data, add_awgn
from rx_utils import get_data, show_train, check_data, prep_ts_data, get_song_data
from ce_models import ce_temel, ce_plus
from rx_config import init_gpu
from constants import h_81

init_gpu()


TAU = 1.00  # 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
SNR = 10  # 0, 1, 2, ..., 10, nonoise  # noqa
IQ = 'bpsk'  # bpsk, qpsk   #


init_lr = 0.001
# model = ce_temel(init_lr=init_lr)

# train parameters
epochs = 70
batch_size = 1024  # reduce batch size for big models...
NoS = int(1e7)  # number of symbols
val_split = 0.1

DATA_MODE = 'load'  # 'load', 'generate' 'load_npy'
WB_ON = False

ISI = 7  # bir sembole etki eden komşu sembol sayısı, örneğin ISI = 5; [ . . . . . S . . . . .], toplam 11 kayıt
FS = 10
G_DELAY = 4
step = int(TAU * FS)
hPSF = np.array(h_81).astype(np.float16)  # TODO G_DELAY FS based h generation
assert np.array_equal(hPSF, hPSF[::-1]), 'symmetry mismatch!'

if DATA_MODE == 'load':
    # Load the training data
    df = pd.read_csv('data/data_ce/rx_data_DL_Ch_Est_BPSK_Ntap3_SNR10dB_alpha03_tau1_Ks10_N256.csv',
                     header=None)
    dfy = pd.read_csv('data/data_ce/tx_data_DL_Ch_Est_BPSK_Ntap3_SNR10dB_alpha03_tau1_Ks10_N256.csv',
                      usecols=[32, 33, 34, 35, 36, 37], header=None)

    for i in range(32):
        df[i] = df[i].str.replace('i', 'j').apply(lambda x: np.complex_(x))
        # df[str(i)+'r'] = df[i].apply(lambda x: np.real(x))
        # df[str(i)+'i'] = df[i].apply(lambda x: np.imag(x))

    X_r = np.real(df)
    X_i = np.imag(df)

    y_r = np.array(dfy[[32, 34, 36]])
    y_i = np.array(dfy[[33, 35, 37]])

    # X = np.concatenate((dfr[:, :26], dfi[:, :26]), axis=1)
    # y = np.concatenate((dfr[:, 26:], dfi[:, 26:]), axis=1)
    # instead of keeping real/imag as single data just flat it
    # X = np.concatenate((dfr[:, :26], dfi[:, :26]), axis=0)
    # y = np.concatenate((dfr[:, 26:], dfi[:, 26:]), axis=0)

    X = np.concatenate((X_r, X_i), axis=0).astype(np.float16)
    y = np.concatenate((y_r, y_i), axis=0).astype(np.float16)

else:
    raise NotImplementedError

model = ce_temel(init_lr=init_lr)
# model = ce_plus(init_lr=init_lr)

print(model.summary())
confs = {
    'loss': model.loss,
    'optimizer_class_name': model.optimizer.__class__.__name__,
    'optimizer_config': model.optimizer.get_config(),
    'metrics': model.compiled_metrics._metrics,  # noqa
}
for k, v in confs.items():
    print(k, v)

tf.keras.backend.clear_session()
history = model.fit(X, y,
                    validation_split=val_split,
                    epochs=epochs,
                    batch_size=batch_size,
                    # callbacks=callbacks,
                    )
show_train(history)

# results = model.evaluate(x_test, y_test, batch_size=128)
y_pred = model.predict(X[:10, :])
y_true = y[:10, :]
