import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data(name, rec=True, NoD=int(1e6)):  # noqa
    # NoD = int(1e6)  # number of test data  # noqa
    file_path = 'data/' + name + '.csv'
    df = pd.read_csv(file_path, names=['y', 'X'], header=None, nrows=NoD)
    x = np.array(df['X'])
    y = np.array(df['y'].astype(np.int8))
    # bpsk
    if rec:
        # BPSK; [0: -1], [1: 1]
        # demapping y+1 / 2
        y = (y + 1)//2
        # y[y==-1] = 0
    return x, y


def gen_data(n):
    # n 'number of data'
    # return data
    raise NotImplementedError


def add_awgn(inputs, snr=10):

    assert len(inputs.shape) == 1, 'Only 1 dimensional data supported!'
    n = len(inputs)
    # SNR = 10*log10(Eb/No)
    # Eb/No = 10 ^(SNR/10)
    # EB = 1 for BPSK
    n0 = 10 ** (-snr/10)
    # noise = np.sqrt(N0/2)*(np.randn(n, 1) + np.randn(n, 1))
    noise = np.multiply(np.sqrt(n0/2), np.random.standard_normal(n))  # standard_normal: (mean=0, stdev=1)
    output = np.add(inputs, noise)

    # plt.hist(noise, bins=40)
    # plt.show()

    return output


def show_train(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
