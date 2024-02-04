"""AI based RX
name: RX data generation module
status: draft,
version: 0.00 (04 February 2024)

Naming:     Modulation_TAU_SNR  (fixed GroupDelay as 4, FS = 10)
"""
import os

import numpy as np
import pandas as pd
from ber_util import gen_data, add_awgn
from rx_utils import get_data, show_train, check_data, prep_ts_data, get_song_data
from rx_config import init_gpu
from constants import h_81, hh_21, snr_to_nos

# init_gpu()

# Modulation Type
IQ = 'bpsk'  # bpsk, qpsk
# TAU Value
TAU = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
# SNR  Level
SNR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 'NoNoise']  # 'NoNoise'  0, 1, 2, ..., 10, NoNoise  # noqa

# ISI = 3  # bir sembole etki eden komşu sembol sayısı, örneğin ISI = 5; [ . . . . . S . . . . .], toplam 11 kayıt
FS = 10
G_DELAY = 4
if G_DELAY == 4:
    hPSF = np.array(h_81).astype(np.float16)  # TODO G_DELAY FS based h generation
elif G_DELAY == 1:
    hPSF = np.array(hh_21).astype(np.float16)  # TODO G_DELAY FS based h generation
else:
    raise NotImplementedError

# [SOURCE]  Data Generation
# TODO: add other modulation type [Only bpsk supported]
for tau in TAU:  # DEBUG [TAU[i]]
    step = int(tau * FS)
    for snr in SNR:
        NoS = max(snr_to_nos[snr], 10**6)
        data_filename = 'data/data_{iq}/{iq}_tau{tau:.1f}_snr{snr}.csv'.format(iq=IQ, tau=tau, snr=snr)
        if os.path.exists(data_filename):
            print('{file} already exist, skipping..'.format(file=data_filename))
            continue

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
        rch = add_awgn(inputs=tx_data, snr=snr, seed=1234)
        # [RX]   apply matched filter
        mf = np.convolve(hPSF, rch)
        # [RX]  down-sample (subsample)
        # p_loc = 2 * G_DELAY * FS  # 81 for g_delay=4 and FS = 10,
        # 4*10=40 from first conv@TX, and +40 from last conv@RX
        # remove additional prefix and suffix symbols due to CONV
        rx_data = mf[2 * G_DELAY * FS:-(2 * G_DELAY * FS):step]

        # X_i, y_i
        X_i = rx_data
        y_i = bits

        # # if AUTO_SAVE:
        # np.save('data/snr{snr}_{iq}_tau{tau:.1f}_X_i.npy'.format(snr=SNR, iq=IQ, tau=TAU), X_i)
        # np.save('data/snr{snr}_{iq}_tau{tau:.1f}_y_i.npy'.format(snr=SNR, iq=IQ, tau=TAU), y_i)

        # save to csv
        df = pd.DataFrame.from_dict({'y': y_i, 'X': X_i})
        df.to_csv(data_filename, index=False)
        # df.to_csv('data/tau0.60_snrNoNoise_bpsk.csv', index=False)
