"""Machine Learning based RX (Symbol Detector)
name: Decision Tree training module
status: draft
version: 0.01 (06 January 2024)
"""
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from rx_utils import get_data, prep_ts_data
from rx_config import init_gpu

# TODO: Add SNR value as feature
# TODO: Change y data from 0 to -1
init_gpu()

TAU = 0.8  # 0.50, 0.60, 0.70, 0.80, 0.90, 1.00
SNR = 10  # 0, 1, 2, ..., 10, 'NoNoise'  # noqa
IQ = 'bpsk'  # bpsk, qpsk   #

NoS = int(1e7)  # number of symbols

ISI = 7  # bir sembole etki eden komşu sembol sayısı, örneğin ISI = 5; [ . . . . . S . . . . .], toplam 11 kayıt
FS = 10
G_DELAY = 4
step = int(TAU * FS)
# hPSF = np.array(h_81).astype(np.float16)  # TODO G_DELAY FS based h generation
# assert np.array_equal(hPSF, hPSF[::-1]), 'symmetry mismatch!'

# Load the training data
X_i, y_i = get_data(name='data_{iq}/tau{tau:.2f}_snr{snr}_{iq}'.format(iq=IQ, tau=TAU, snr=SNR), NoD=NoS)
if IQ != 'bpsk':
    # compact data into 1D, no need to consider real(I) and imaginary(Q) parts as separate dimensions
    X_i = np.reshape(X_i, (-1,))

X = prep_ts_data(X_i, isi=ISI)
# update label type to float for evaluating performance metrics
y = y_i.astype(np.float16)

# Split dataset into 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=1)

dtree = DecisionTreeClassifier(criterion='entropy', max_depth=11, random_state=1)
dtree = dtree.fit(X_train, y_train)

# tree.plot_tree(dtree, feature_names=features)

y_pred = dtree.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("test accuracy {}".format(acc))

text_representation = export_text(dtree)
print(text_representation)

# fig = plt.figure(figsize=(25, 20))
# fig = plt.figure()
# _ = tree.plot_tree(dtree,
#                    #feature_names=["-7", "-6", "-5", "-4", "-3", "-2", "-1", "S", "+1", "+2", "+3", "+4", "+5", "+6", "+7"],
#                    #class_names=["1", "-1"],
#                    filled=True)
# plt.show()
