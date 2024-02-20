"""Machine Learning based RX (Symbol Detector)
name: Decision Tree training module
status: draft, configurations saved to file
version: 0.0.4 (20 February 2024, 22:18)
"""
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier, export_text
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from rx_utils import get_data, prep_ts_data, mk_dir
from rx_features import remove_isi
# from rx_config import init_gpu
from constants import snr_to_nos, BERtau1, gbKSE, BCJR, TRBER

# TODO: Add SNR value as feature
# TODO: Change y data from 0 to -1
# init_gpu()

# Modulation Type
IQ = 'bpsk'  # bpsk, qpsk
# TAU Value
TAU = [0.6, 0.7, 0.8, 0.9]  # [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# SNR  Level
SNR = [6, 8, 10, 12]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 'NoNoise']

NoS = int(1e5)  # -1  # int(1e7)  # number of symbols, -1: all

LoN = 5  # bir sembole etki eden komşu sembol sayısı, örneğin ISI = 5; [ . . . . . S . . . . .], toplam 11 kayıt
#
# do not change FS and G_DELAY
FS = 10
G_DELAY = 4

datestr = datetime.now().strftime("%Y%m%d-%H%M%S")
results = {}
config = {'Modulation': IQ, 'TAU': TAU, 'SNR': SNR, 'Number of sample': NoS if NoS != -1 else 'all',
          'Half window length': LoN, 'Sampling Frequency': FS, 'Group Delay': G_DELAY}
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

        X = prep_ts_data(X_i, isi=LoN)
        # update label type to float for evaluating performance metrics
        y = y_i.astype(np.float16)

        # prepare the features
        Xf = remove_isi(X, LoN=LoN, tau=tau, merge=True)

        # Split dataset into 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(Xf, y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=1)

        dtree = DecisionTreeClassifier(criterion='gini', max_depth=7, random_state=1)
        # dtree = RandomForestClassifier(criterion='gini', max_depth=7, random_state=1)
        dtree = dtree.fit(X_train, y_train)

        # tree.plot_tree(dtree, feature_names=features)
        y_pred = dtree.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        # tauKEY = int(tau*FS)
        if tau in results.keys():
            results[tau][snr] = acc
        else:
            results[tau] = {snr: acc}

        print("TAU {tau}, SNR {snr}, TestData {nod}; Test accuracy : {acc}".format(tau=tau, snr=snr,
                                                                                   nod=len(X_test), acc=acc))

        # text_representation = export_text(dtree)
        # print(text_representation)

        # fig = plt.figure(figsize=(25, 20))
        # fig = plt.figure()
        # _ = tree.plot_tree(dtree,
        #                    #feature_names=[...],
        #                    #class_names=["1", "-1"],
        #                    filled=True)
        # plt.show()

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
drf1 = pd.DataFrame.from_dict(TRBER)
drf2 = pd.DataFrame.from_dict(BCJR)
drf3 = pd.DataFrame.from_dict(gbKSE)
drf4 = pd.DataFrame.from_dict(BERtau1)

fig, ax = plt.subplots()
# df_comp.plot(ax=ax, x="SNR", logy=True, marker='d')
df_now.plot(ax=ax, x="SNR", logy=True, marker='v')
# df_t7.plot(ax=ax, x="SNR", logy=True, marker='v', linestyle='dashdot')
drf1.plot(ax=ax, x="SNR", logy=True, marker='X', linestyle='dotted')
drf2.plot(ax=ax, x="SNR", logy=True, marker='d', linestyle='dashed')
# drf4.plot(ax=ax, x="SNR", logy=True, marker='*', linestyle='dashdot')
plt.title('DecisionTree based Symbol Detector')
plt.xlabel('Eb/No[dB], SNR')
plt.ylabel('BER')
plt.xlim([0, 15])
plt.grid(visible=True, which='both')
plt.show()

# save the figure as image
plt.savefig('run/{id}/figure.png'.format(id=datestr))
# save the figure as object
pickle.dump(fig, open('run/{id}/figure.pickle'.format(id=datestr), 'wb'))

# to load the figure back, use
fig = pickle.load(open('run/{id}/figure.pickle'.format(id=datestr), 'rb'))
fig.show()
