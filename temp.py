"""
 gecici çalışma dosyası,
 ! faydalı çıktıları kayıt altına almayı unutmayın..
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import snr_to_nos, h_81
from rx_config import *
from rx_utils import prep_ts_data
from ber_util import gen_data, awgn, add_awgn2, add_awgn, get_h, bit_checker
from keras.models import load_model

IQ = 'bpsk'  # bpsk qpsk # noqa
sel = 'best'  # best, latest
path = ''  # manuel path to models/gru_qpsk_2023... # noqa

THEORY = False
TAU_OFF = False

FS = 10
TAU = 0.90
SNR = [i for i in range(20 + 1)]
G_DELAY = 0

step = int(TAU * FS)

if path == '':
    path = 'models/tau{tau:.2f}_{modulation}_{sel}'.format(modulation=IQ, tau=TAU, sel=sel)

model = load_model(filepath=path,
                   custom_objects=None, compile=True, safe_mode=True)

# run id
rid = 'tau{tau}_{model}'.format(tau=TAU, model=model.name)  # TODO add unique id for each model as _suffix

# flow configurations
initial_seed = 2346
noise_seed = 0  # constant for all cases
init_nos = int(1e+2)  # number of symbol to generate, send, and decode at each turn

#
# one time process, and constants (to optimize the process, get these out of the for loop)
#

# generate the filter
sPSF = get_h(fs=FS, g_delay=G_DELAY)
result = dict({'SNR': [], 'NoE': [], 'NoB': [], 'BER': []})
#
# Message
#
# snr = 100  # for debug
# set Number of symbol for each snr run
# NOS = []
# for i in range(len(SNR)):
#     if i % 3 == 0:
#         NOS.append(min(init_nos*(10**i), int(1e+9)))
#     else:
#         NOS.append(NOS[-1])

for _i_, snr in enumerate(SNR):
    nos = snr_to_nos.get(snr, 1000000)
    # set seed value for random data
    turn_seed = initial_seed + _i_
    # TX Data Generation
    syms, bits = gen_data(n=nos, mod=IQ, seed=turn_seed)
    nos = 10

    syms = [1,
            1,
            1,
            -1,
            1,
            -1,
            -1,
            1,
            -1,
            1]
    # bits = gen_data(n=10, mod=IQ, seed=turn_seed)
    syms = np.reshape(syms, (-1,1))
    #
    # TX side
    #
    if TAU_OFF:
        tx_data = syms
    else:
        # extend the data by up sampling (in order to be able to apply FTN)
        s_up_sampled = np.zeros((step * len(syms), syms.shape[1]), dtype=np.float16)
        for i in range(syms.shape[1]):
            s_up_sampled[::step, i] = syms[:, i]

        # apply the filter
        # tx_data = np.convolve(sPSF, s_up_sampled)
        tx_data = np.empty((len(s_up_sampled) + 2 * G_DELAY * FS, s_up_sampled.shape[1]), dtype=np.float16)
        for i in range(syms.shape[1]):
            tx_data[:, i] = np.convolve(sPSF, s_up_sampled[:, i])
        # tx_data = s_up_sampled
    plt.figure()
    plt.plot(tx_data)
    plt.grid(True)
    plt.show()
    # Channel Modelling, add noise
    rch = add_awgn(inputs=tx_data, snr=snr, seed=noise_seed)
    # rch = add_awgn2(tx_data[:, 0], snr)
    # rch = awgn(signal=tx_data, desired_snr=snr)  # , seed=noise_seed)
    # rch = awgn(signal=tx_data, desired_snr=snr)  # , seed=noise_seed)
    # rch = tx_data

    #
    # RX side
    #
    if TAU_OFF:
        rx_data = rch
    else:
        # # match filter
        # mf = np.empty((len(rch)+2*G_DELAY*FS, rch.shape[1]), dtype=np.float16)
        # for i in range(rch.shape[1]):
        #     mf[:, i] = np.convolve(sPSF, rch[:, i])
        mf = rch

        #  Down sampling
        p_loc = 2 * G_DELAY * FS  # 81 for g_delay=4 and FS = 10,
        # 4*10=40 from first conv@TX, and +40 from last conv@RX
        # remove additional prefix and suffix symbols due to CONV
        # rx_data = mf[p_loc:-p_loc:int(TAU*FS)]
        rx_data = mf[::int(TAU * FS)]
        # [DEBUG]
        # plt.plot(np.real(mf[:250]))
        # plt.plot(np.imag(mf[:250]))
        # plt.plot(np.real(rx_data[:250]))

    # single to time series data
    if 'lstm' in model.name or 'gru' in model.name:
        X = prep_ts_data(rx_data)
    else:
        X = rx_data

    # [DEBUG] data check point
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

    if THEORY:
        y_hat = (X > 0) * 1
        y_hat = np.reshape(y_hat, -1)
    else:
        # y_pred = model.predict(X, batch_size=4096) # noqa
        y_pred = model.predict_on_batch(X)  # noqa
        if IQ == 'bpsk':
            # hard desicion on the predictions
            y_pred_bit = (y_pred > 0.5) * 1
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
    tBER = noe / nob
    # save data into the result dictionary
    result['SNR'].append(snr)
    result['NoE'].append(noe)
    result['NoB'].append(nob)
    result['BER'].append(tBER)
    # print("BER for given turn:\t{bit} bits\t{err} error\tBER: {ber}".format(bit=nob, err=noe, ber=tBER))
    print("{snr} dB SNR,\t{bit} bits\t{err} error\tBER: {ber}".format(snr=snr, bit=nob, err=noe, ber=tBER))
    if noe == 0:
        break

# DEBUG
# SNR : 100 dB, TAU = 1
# BER for given turn:	1000000 bits	136276 error	BER: 0.136276
# SNR : 10 dB,  TAU = 1
# BER for given turn:	1000000 bits	143702 error	BER: 0.143702
# SNR : 10 dB,  TAU = 0.8
# BER for given turn:	1000000 bits	144115 error	BER: 0.144115

df = pd.DataFrame.from_dict(result)
# df.to_csv("")

fig, ax = plt.subplots()
df.plot(ax=ax, x="SNR", y="BER", logy=True, marker='d')
plt.grid(visible=True, which='both', ls="-")
plt.show()
# references and resources and more
#
# up sampling, https://stackoverflow.com/a/25858023

dfc = pd.DataFrame()
df_ftn_hd = pd.read_csv('ftn_hd.csv')
df_ftn_dl = pd.read_csv('ftn_dl.csv')
df_no_ftn = pd.read_csv('no_ftn.csv')

fig, ax = plt.subplots()
df.plot(ax=ax, x="SNR", y="BER", logy=True, marker='d')
df_ftn_hd.plot(ax=ax, x="SNR", y="BER", logy=True, marker='v')
df_ftn_dl.plot(ax=ax, x="SNR", y="BER", logy=True, marker='.')
df_no_ftn.plot(ax=ax, x="SNR", y="BER", logy=True, marker='*')
plt.grid(visible=True, which='both', ls="-")
plt.show()


