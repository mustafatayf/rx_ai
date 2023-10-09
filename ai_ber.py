"""
AI based RX models BER plots

version 0.01 (05 October 2023)
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rx_config import *
from ber_util import gen_data, add_awgn, get_h, bit_checker
from keras.models import load_model

IQ = 'bpsk' # bpsk qpsk # noqa
sel = 'best'  # best, latest
path = ''   # manuel path to models/gru_qpsk_2023... # noqa

FS = 10
TAU = 1.00
SNR = [i for i in range(10+1)]
G_DELAY = 4

step = int(TAU*FS)

if path == '':
    path = 'models/{modulation}_{sel}'.format(modulation=IQ, sel=sel)

model = load_model(filepath=path,
                   custom_objects=None, compile=True, safe_mode=True)

# run id
rid = 'tau{tau}_{model}'.format(tau=TAU, model=model.name)  # TODO add unique id for each model as _suffix

# flow configurations
initial_seed = 2346
nos = int(1e6)  # number of symbol to generate, send, and decode at each turn

snr = 100  # for debug

#
# Message
#

# TX Data Generation
syms, bits = gen_data(n=nos, mod=IQ, seed=initial_seed)

#
# TX side
#

# extend the data by up sampling (in order to be able to apply FTN)
# https://stackoverflow.com/a/25858023
# if len(syms.shape) == 1:  # only real
#     s_up_sampled = np.zeros(step*len(syms), dtype=np.complex_)
#     s_up_sampled[::step] = syms
# else:
s_up_sampled = np.zeros((step * len(syms), syms.shape[1]), dtype=np.float16)
for i in range(syms.shape[1]):
    s_up_sampled[::step, i] = syms[:, i]

# generate the filter
sPSF = get_h(fs=FS, g_delay=G_DELAY)
# apply the filter
# tx_data = np.convolve(sPSF, s_up_sampled)
tx_data = np.empty((len(s_up_sampled)+2*G_DELAY*FS, s_up_sampled.shape[1]), dtype=np.float16)
for i in range(syms.shape[1]):
    tx_data[:, i] = np.convolve(sPSF, s_up_sampled[:, i])

# Channel Modelling, add noise
rch = add_awgn(inputs=tx_data, snr=snr)

#
# RX side
#

# match filter
# mf = np.convolve(sPSF, r)
mf = np.empty((len(rch)+2*G_DELAY*FS, rch.shape[1]), dtype=np.float16)
for i in range(rch.shape[1]):
    mf[:, i] = np.convolve(sPSF, rch[:, i])

#  Down sampling
p_loc = 2*G_DELAY*FS  # 81 for g_delay=4 and FS = 10,
# 4*10=40 from first conv@TX, and +40 from last conv@RX
# remove additional prefix and suffix symbols due to CONV
rx_data = mf[p_loc:-p_loc:int(TAU*FS)]
# plt.plot(np.real(mf[:250]))
# plt.plot(np.imag(mf[:250]))
# plt.plot(np.real(rx_data[:250]))

if IQ == 'bpsk':  # Only Real ?
    X_i = np.real(rx_data)
else:  # Real and Imaginary
    X_i = np.vstack((np.real(rx_data), np.imag(rx_data))).T

# TODO use time series gen function
if 'lstm' in model.name or 'gru' in model.name:
    isi = 7
    # padding for initial and ending values
    d = len(X_i.shape)
    assert d < 3, 'high dimensional input does not supported, only 1D or 2D'
    # if d == 1:
    #     tmp_pad = np.zeros(isi)  # abs(X_i[:isi, :]*0)
    # else:
    #     tmp_pad = abs(X_i[:isi, :]*0)
    # Xp = np.concatenate((tmp_pad, X_i, tmp_pad), axis=0)
    tmp_pad = abs(X_i[:isi, :]*0)
    Xp = np.concatenate((tmp_pad, X_i, tmp_pad), axis=0)

    sl = list(X_i.shape)
    sl.insert(1, 2 * isi + 1)
    ls_x = np.empty(shape=tuple(sl), dtype=np.float16)
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

#
# data check point
#
# import pandas as pd
# cpn = 100
# df = pd.DataFrame()
# df['bits'] = bits[:cpn]  # message bits
# df['tx'] = tx_data[G_DELAY*FS:(G_DELAY*FS+cpn*step):step]  # TX output to channel
# df['rCH'] = rch[G_DELAY*FS:(G_DELAY*FS+cpn*step):step]  # channel effect (AWGN) added
# df['mf'] = mf[p_loc:(p_loc+cpn*step):step]  # match filter applied to RAW RX data
# df['rx_data'] = pd.DataFrame(rx_data[:cpn])  # down sampled RX after match filer
#
# df.plot()

y_pred = model.predict(X) # noqa
if IQ == 'bpsk':
    # hard desicion on the predictions
    y_pred_bit = (y_pred > 0.5)*1
else:
    y_pred_bit = ""  # TODO missing implementation

y_hat = np.reshape(y_pred_bit, -1)

# debug
# plt.figure()
# plt.plot(bits[:40])
# plt.plot(y_hat[:40])
# plt.legend(['bits', 'predictions'])
# plt.show()
noe, nob = bit_checker(bits, y_hat)
tBER = noe/nob
print("BER for given turn:\t{bit} bits\t{err} error\tBER: {ber}".format(bit=nob, err=noe, ber=tBER))

# DEBUG
# SNR : 100 dB, TAU = 1
# BER for given turn:	1000000 bits	136276 error	BER: 0.136276
# SNR : 10 dB,  TAU = 1
# BER for given turn:	1000000 bits	143702 error	BER: 0.143702
# SNR : 10 dB,  TAU = 0.8
# BER for given turn:	1000000 bits	144115 error	BER: 0.144115
