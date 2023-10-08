"""
AI based RX models BER plots

version 0.01 (05 October 2023)
"""
import matplotlib.pyplot as plt
import numpy as np

from rx_config import *
from ber_util import gen_data, add_awgn, get_h
from keras.models import load_model

IQ = 'bpsk' # bpsk qpsk # noqa
sel = 'best'  # best, latest
path = ''   # manuel path to models/gru_qpsk_2023... # noqa
# gru_bpsk_2023Oct08_0856
FS = 10
TAU = 0.8
SNR = [i for i in range(10+1)]

if path == '':
    path = 'models/{modulation}_{sel}'.format(modulation=IQ, sel=sel)

model = load_model(filepath=path,
                   custom_objects=None, compile=True, safe_mode=True)

# run id
rid = 'tau{tau}_{model}'.format(tau=TAU, model=model.name)  # TODO add unique id for each model as _suffix

# flow configurations
initial_seed = 2346
nos = int(1e6)  # number of symbol to generate, send, and decode at each turn

snr = 10
# TX Data Generation
syms, bits = gen_data(n=nos, mod=IQ, seed=initial_seed)

# https://stackoverflow.com/a/25858023
s_upsampled = np.zeros(int(TAU*FS)*len(syms), dtype=np.complex_)
# 100 bits >>> 100 sembol (BPSK)
# TAU = 0.8, FS = 10
#  8*99 + 1 = 793
s_upsampled[::int(TAU*FS)] = syms
sPSF = get_h(fs=FS)
tx_data = np.convolve(sPSF, s_upsampled)

# Channel Modelling
r = add_awgn(inputs=tx_data, snr=snr)

# RX
mf = np.convolve(sPSF, r)
ploc = 81  # TODO make parametric, FIX THIS
rx_data = mf[ploc-1:-ploc:int(TAU*FS)]
# plt.plot(np.real(mf[:250]))
# plt.plot(np.imag(mf[:250]))
# plt.plot(np.real(rx_data[:250]))
# X_ = rx_data[1:nos+1]
X_ = rx_data
if IQ == 'bpsk':
    X_i = np.real(X_)
else:
    X_i = np.vstack((np.real(X_), np.imag(X_))).T

# TODO use time series gen function
if 'lstm' in model.name or 'gru' in model.name:
    isi = 7
    # padding for initial and ending values
    d = len(X_i.shape)
    assert d < 3, 'high dimensional input does not supported, only 1D or 2D'
    if d == 1:
        tmp_pad = np.zeros(isi)  # abs(X_i[:isi, :]*0)
    else:
        tmp_pad = abs(X_i[:isi, :]*0)
    Xp = np.concatenate((tmp_pad, X_i, tmp_pad), axis=0)

    sl = list(X_i.shape)
    sl.insert(1, 2 * isi + 1)
    ls_x = np.empty(shape=tuple(sl))
    if d == 1:
        for i in range(sl[0]):
            ls_x[i, :] = Xp[i:i + 2 * isi + 1]
    else:
        for i in range(sl[0]):
            ls_x[i, :, :] = Xp[i:i + 2 * isi + 1, :]

    # X = np.reshape(ls_x, (ls_x.shape[0], ls_x.shape[1], 1))
    X = ls_x
else:
    X = X_i


y_pred = model.predict(X) # noqa
if IQ == 'bpsk':
    # hard desicion on the predictions
    y_pred_bit = (y_pred > 0.5)*1

y_hat = np.reshape(y_pred_bit, -1)

plt.figure()
plt.plot(y_hat[:40])
plt.plot(bits[:40])
plt.show()
