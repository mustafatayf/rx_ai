import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder

def get_data(name, rec=True, NoD=int(1e6)):  # noqa
    # NoD = int(1e6)  # number of test data  # noqa
    file_path = 'data/' + name + '.csv'
    # infer data type
    if 'bpsk' in name:
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
