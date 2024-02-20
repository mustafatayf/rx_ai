""" Utilities for RX
name:
status: initial+, mkdir added
version: 0.0.2 (12 February 2024, 13:38)
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder

def get_data(name, rec=True, NoD=int(1e6)):  # noqa
    # NoD = int(1e6)  # number of test data  # noqa
    if NoD == -1:
        NoD = None  # noqa
    file_path = 'data/' + name + '.csv'
    # infer data type
    if 'bpsk' in name:
        # df = pd.read_csv(file_path, names=['y', 'X'], header=None, nrows=NoD)
        df = pd.read_csv(file_path, header=0, nrows=NoD)  # names=['y', 'X']
        x = np.array(df['X'])
        y = np.array(df['y'].astype(np.int8))
        # bpsk
        if rec:
            # BPSK; [0: -1], [1: 1]
            # demapping y+1 / 2
            y = (y + 1)//2
            # y[y==-1] = 0

        # x = np.reshape(x, (-1, 1))
        # y = np.reshape(y, (-1, 1))
        return x, y
    elif 'qpsk' in name:
        df = pd.read_csv(file_path, names=['y1', 'y2', 'x_real', 'x_imag'], header=None, nrows=NoD,
                         dtype={'y1': np.int8, 'y2': np.int8, 'x_real':  np.float16, 'x_imag':  np.float16})
        # TODO: improve data import, replace on read; https://stackoverflow.com/a/18920156
        # x = np.array(df['X'].str.replace('i', 'j').apply(lambda s: np.singlecomplex(s)))
        x = np.array(df[['x_real', 'x_imag']].astype(np.float16))
        # y = np.array(df[['y1', 'y2']].astype(np.int8))
        # 00, 01, 10, 11 to --> 0 1 2 3 integer
        y = np.array(df['y1']*2+df['y2']).astype(np.int8)
        # integer to one hot
        #   https://www.delftstack.com/howto/numpy/one-hot-encoding-numpy/
        y_one_hot = np.zeros(shape=(y.size, y.max()+1), dtype=np.int8)
        y_one_hot[np.arange(y.size), y] = 1

        return x, y_one_hot


def acc_data(tau=0.5):
    """ accumulate data """

    raise NotImplementedError
    # return x, y


def check_data(rx_data, ref_bit, modulation='bpsk'):
    """

    :param rx_data: X
    :param ref_bit: y
    :param modulation:
    :return:
    """

    if modulation == 'qpsk':
        # qpsk check
        s3 = np.logical_and(rx_data[:, 0] > 0, rx_data[:, 1] > 0)  # D
        s2 = np.logical_and(rx_data[:, 0] > 0, rx_data[:, 1] < 0)  # C
        s1 = np.logical_and(rx_data[:, 0] < 0, rx_data[:, 1] > 0)  # B
        s0 = np.logical_and(rx_data[:, 0] < 0, rx_data[:, 1] < 0)  # A
        _df = pd.DataFrame({'s0': s0*1, 's1': s1*1, 's2': s2*1, 's3': s3*1})
        # _df.sum()
        # _df.sum().plot()
        ref = pd.DataFrame(ref_bit)
        ref.columns = ['s0', 's1', 's2', 's3']
        nof_error = 0
        for clm in _df.columns:
            nof_error += (_df[clm] != ref[clm]).sum()

    elif modulation == 'bpsk':
        # check for data
        nof_error = sum(abs((rx_data > 0)*1 - ref_bit))
        print('#data: {NoD}\t#diff: {diff}'.format(NoD=ref_bit.size, diff=nof_error))
    else:
        raise NotImplementedError

    return nof_error


def get_song_data(x_in, y_in, L, m):
    pad = (L-m)//2
    # padding for initial and ending values
    # d = len(in_data.shape)
    # assert d < 3, 'high dimensional input does not supported, only 1D or 2D'
    tmp_pad = abs(x_in[:pad]*0)
    data = np.concatenate((tmp_pad, x_in, tmp_pad), axis=0)

    nos = (len(data)-L)//m  # number of record/sample

    x_out = np.empty(shape=(nos, L))
    y_out = np.empty(shape=(nos, m))
    for i in range(nos):
        x_out[i, :] = data[i*m:(i*m + L)]
        y_out[i, :] = 2*y_in[i*m:(i*m + m)] - 1

    return x_out, y_out


def is_symmetric(L):
    """
    check if given list is symmetric or not
    """
    assert NotImplementedError
    return all(i == j for i, *j in zip(L, *L))


def prep_ts_data(in_data, isi=7):

    # padding for initial and ending values
    d = len(in_data.shape)
    assert d == 1, 'high dimensional input does not supported, only 1D allowed'
    # assert d < 3, 'high dimensional input does not supported, only 1D or 2D'
    tmp_pad = abs(in_data[:isi]*0)
    data = np.concatenate((tmp_pad, in_data, tmp_pad), axis=0)

    sl = list(in_data.shape)
    sl.insert(1, 2 * isi + 1)
    ts_x = np.empty(shape=tuple(sl))
    if d == 1:
        for i in range(sl[0]):
            ts_x[i, :] = data[i:i + 2 * isi + 1]
    else:
        for i in range(sl[0]):
            ts_x[i, :, :] = data[i:i + 2 * isi + 1, :]

    return ts_x


def show_train(history):
    n = len(history.history)
    fig, axs = plt.subplots(nrows=1, ncols=n//2)
    # TODO : add no validation support
    for i, metric in enumerate(history.history):
        # print(metric)
        if 'val_' in metric:
            break
        axs[i].plot(history.history[metric])
        axs[i].plot(history.history['val_' + metric])
        axs[i].set_title('model ' + metric)
        axs[i].set_xlabel('epoch')
        axs[i].set_ylabel(metric)
        axs[i].legend(['train', 'val'])
        # axs[i].legend(['train', 'val'], loc='upper left')

    plt.show()
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_title.html


def mk_dir(path):
    """
    Automatically create directories for the specified path if any are missing,
    Remember to add a trailing '/' backslash at the end of the input path.
    """
    assert path[-1] == '/', 'please add "/" at the end of path, To make sure that the input is not a file!'
    # check existence of each sub path
    if path[0] == '/':
        tp = ''  # tp: temporarily path
    else:
        tp = '.'
    for dr in path.split('/'):
        tp += '/'+dr
        if not os.path.isdir(tp):
            os.mkdir(tp)

