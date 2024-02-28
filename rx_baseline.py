""" Symbol Detector Baseline reference
name:
status: draft, model parameter added
version: 0.0.2 (28 February 2024, 22:02)
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from rx_utils import get_data, prep_ts_data, mk_dir
from constants import snr_to_nos, BERtau1, gbKSE, BCJR, TRBER


# Modulation Type
IQ = 'bpsk'  # bpsk, qpsk
# TAU Value
TAU = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# SNR  Level
SNR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 'NoNoise']

NoS = -1  # int(1e7)  # number of symbols, -1: all

#
# do not change FS and G_DELAY
FS = 10
G_DELAY = 4

datestr = datetime.now().strftime("base_%Y%m%d-%H%M%S")
results = {}
config = {'Modulation': IQ, 'TAU': TAU, 'SNR': SNR, 'Number of sample': NoS if NoS != -1 else 'all',
          'Sampling Frequency': FS, 'Group Delay': G_DELAY}

# # Test/Inference Phase
# TODO : add tic-toc time
# TODO print logs to the file, result and number of test item, + time to train
for tau in TAU:
    step = int(tau * FS)
    for snr in SNR:
        # Load the training data
        X_i, y_i = get_data(name='data_{iq}/{iq}_tau{tau:.1f}_snr{snr}'.format(iq=IQ, tau=tau, snr=snr), NoD=NoS)
        if IQ != 'bpsk':
            # compact data into 1D, no need to consider real(I) and imaginary(Q) parts as separate dimensions
            X_i = np.reshape(X_i, (-1,))

        # tree.plot_tree(dtree, feature_names=features)
        number_of_sample = len(y_i)
        number_of_correct = sum(np.equal(y_i, (X_i >= 0)*1))
        number_of_error = number_of_sample - number_of_correct
        acc = number_of_correct / number_of_sample

        # tauKEY = int(tau*FS)
        if tau in results.keys():
            results[tau][snr] = acc
        else:
            results[tau] = {snr: acc}

        print("TAU {tau}, SNR {snr}, TestData {nod}; Test accuracy : {acc}".format(tau=tau, snr=snr,
                                                                                   nod=number_of_sample, acc=acc))


# create the folder to store the result of the current run
mk_dir('run/{id}/'.format(id=datestr))
with open('run/{id}/configurations.xml'.format(id=datestr), 'w') as f:
    for key, value in config.items():
        # f.write('%s\t:\t%s\n' % (key, value))
        # f.write('{:>25}: {:<30}{}\n'.format(str(key), str(value), 'comment'))
        f.write('{:>25}: {:<30}\n'.format(str(key), str(value)))

df = pd.DataFrame.from_dict(results)
df.to_csv('run/{id}/{iq}_{date}_acc.csv'.format(id=datestr, iq=IQ, date=datestr))

# TODO: add BER plot
res_dict = {'SNR': np.array(SNR)}  # result dictionary for current run
for tau in TAU:
    res_dict['TAU_{:.1f}'.format(tau)] = np.subtract(1, np.array(list(results[tau].values()))).tolist()

df_now = pd.DataFrame.from_dict(res_dict)
df_now.to_csv('run/{id}/{iq}_{date}_pber.csv'.format(id=datestr, iq=IQ, date=datestr), index=False)

# TODO update the reference BER data
fig, ax = plt.subplots()
major_ticks = np.arange(0, 19, 2)
minor_ticks = np.arange(0, 19, 1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)

df_now.plot(ax=ax, x="SNR", logy=True, marker='v')
plt.title('DecisionTree based Symbol Detector')
plt.xlabel('Eb/No[dB], SNR')
plt.ylabel('BER')
plt.xlim([0, 18])
plt.grid(visible=True, which='both')
plt.show()

# save the figure as image
plt.savefig('run/{id}/figure.png'.format(id=datestr))
# save the figure as object
pickle.dump(fig, open('run/{id}/figure.pickle'.format(id=datestr), 'wb'))

# to load the figure back, use
# fig = pickle.load(open('run/{id}/figure.pickle'.format(id=datestr), 'rb'))
# fig.show()
