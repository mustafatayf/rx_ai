"""
Variable Tau initial trials
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from ber_util import add_awgn
from constants import h_81
from vt_models import base_bpsk

# import scipy.io as sio
# tx_x = sio.loadmat('data/vt_bpsk/tx_signal.mat')


# df = pd.read_csv('data/vt_bpsk/tx_signal.csv', names=['y', 'X'], header=None, nrows=NoD)
df_x = pd.read_csv('data/vt_bpsk/tx_signal.csv', header=None)
df_y = pd.read_csv('data/vt_bpsk/data_bits.csv', header=None)
"""
upsample edilmiş ve Transmitter'da Pulse Shaping uygulnıp TX çıkışında kanala iletilen sinyal değerleri (BPSK)

her satır 110 değer içeriyor, (min 74 max 110 olabilir, eksik kalan sutunlar 0 ile doldurulmuş halde)
"""

SNR = 10
hPSF = np.array(h_81).astype(np.float16)  # TODO G_DELAY FS based h generation

# X_i, y_i
X_tx = df_x.to_numpy()
y_i = df_y.to_numpy()
DA = 1000

# X_pre = np.empty(np.shape(X_tx))
X_pre = np.empty((np.shape(X_tx)[0]*DA, np.shape(X_tx)[1]+len(hPSF)-1))
# y_pre = np.empty((np.shape(y_i)[0]*DA, np.shape(y_i)[1]))
y = np.repeat([y_i], DA, axis=0).reshape((y_i.shape[0]*DA, y_i.shape[1]))
for da in range(DA):
    for i, pkg in enumerate(X_tx):
        # [CHANNEL] add AWGN noise (snr)
        # Channel Modelling, add noise
        # X_tx_flat = np.reshape(X_tx, (-1, 1))
        # X_tx_flat = X_tx.flatten()
        rch = add_awgn(inputs=pkg, snr=SNR, seed=1234)
        # rch = pkg  # no noise
        # [RX]   apply matched filter
        # mf = np.convolve(hPSF, rch)
        X_pre[X_tx.shape[0]*da+i, :] = np.convolve(hPSF, rch)

# h = h_21 * (1/np.matmul(np.reshape(h_21, (1,-1)), h_21))


# -1,-1,-1,-1, 1,-1,-1,-1, 1,-1
# 1.0,0.6,0.6,1.0,0.9,0.7,0.6,1.0,0.8
# plt.figure()
# plt.plot(X_tx[-1, :])
# plt.plot(rch)
# plt.plot(X_pre[-1, :])
# # plt.plot(np.concatenate((np.zeros(20), X_pre[-1, :])))
# plt.plot(X_pre[-1, 10:])
# plt.legend(["TX", "TX+AWGN", "conv(TX+AWGN, H)", "conv_shift"])
# plt.show()

# plt.figure()
# plt.stem(X_tx[-1, :])
# # plt.stem(rch)
# plt.stem(X_pre[-1, :])
# # plt.plot(np.concatenate((np.zeros(20), X_pre[-1, :])))
# plt.stem(X_pre[-1, 10:])
# plt.legend(["TX", "TX+AWGN", "conv(TX+AWGN, H)", "conv_shift"])
# plt.show()
# TAU degerleri degisken oldugundan, down sample yapmadan dogrudan modele input olarak MF sonucunu gonderiyoruz

# 17 sembol değeri> 170 (tau=1)
# 17 sembol değeri> 85 (tau=0.5)
#
# 17-34 sembol >
#
# -1      1   -1       1  >>> mesaj bitleri (BPSK module edilmiş)
#     0.6   1   0.8       >>> optimum tau squence
# -0.7   0.99     -0.8    >> TX çıkış sinyali
#
# [[paket sync]  [mesaj bitleri (1K bit) (tau=1)] ]   >>      10ms
# TAU=1(Nyquist)
#
# [[paket sync]  [mesaj bitleri (1K bit) (tau=0.5)] ]   >>  5ms
#
#
# [[paket sync]  [mesaj bitleri (1K bit) (tau=1)] ] [Ending SYNC]  >>      10ms
# TAU=1(Nyquist)                                      (Nyquist)
#
# [[paket sync]  [mesaj bitleri (1K bit) (tau=0.5)] ]   >>  5ms
#
# -0.7    1     -0.8
# 1       1      -1

test = []
for pt in X_tx:
    for i in X_tx:
        test.append(np.cross(pt, i))
#
# 21.sample    22.sample   .....  26.sample
#
# RX
# match filter
# X           >>> (MODEL) >>> y  (y: bit değeri (-1, 1))
#
# X           >>> (MODEL) >>> y (y: (-1/1, shift miktarı))  >>> bu yapı üzerinden devam edelim
#                 tau_squenc------------------~|
#
#                 [ < 170 >  ]
#                   [          ]
#                     [          ]
# 0 1 2 3 4 5 6 7 8 9 10 11 .... 999 .. 10000  (samples)
# [ < 170 >  ]
#
# 1 1 -1
# ? tau değerlerini bilmeden gelen sample değerleri içerisinde kaç tane sembol olduğunu bilemiyoruz
# ? kaç sample kaydıracağız
#
# 0 1 2 3 4 5 6 7 8 9 10 11 .... 249 (samples)
# [ <       250 (10 sembol)    >  ]
# [ -1 1 -1 1 1 1 -1 1 ]


# 50 x 10 (N, 10) , 10: bir seferde iletilen bit sayısı
#
#  x x x x x a b c d e f
#  x x x x a b c d e f ..
#  x x x a b c d e f ...
#
#  0 0 0 0 0 1 -1 -1 1
#  1 -1 1 1 -1 0 0 0 0 0
# 1 adet

X = X_pre

# b = rcosdesign(beta,span,sps,shape)
# sps = 10
# span = 2*gdelay
# shape = 'sqrt'
# beta = alpha (roll of factor)

model = base_bpsk(input_length=X.shape[1])


history = model.fit(X, y,
                    # validation_split=val_split,
                    epochs=15,
                    batch_size=8,
                    # callbacks=callbacks,
                    )
