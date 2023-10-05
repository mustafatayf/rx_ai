import numpy as np
import matplotlib.pyplot as plt
# from commpy.modulation import PSKModem, QAMModem  # noqa
from commpy.filters import rcosfilter  # scikit-commpy

# import pandas as pd

dict_M = dict({
    'bpsk': 1,  # noqa
    'qpsk': 2  # noqa
})


# def calc_ber():
#     raise NotImplementedError

def get_h(n, fs=10, DEBUG=False):
    # raised cosine (RC) filter (FIR) impulse response
    # TODO // verify and validate
    fd = 1
    alpha = 0.3
    h = rcosfilter(N=n, alpha=alpha, Ts=fd, Fs=fs)[1]
    if DEBUG:
        plt.figure()
        plt.plot(h)
        plt.show()
    # https://stackoverflow.com/a/25858023
    return h


def gen_data(n, mod, seed):
    np.random.seed(seed)
    #  n : 'number of symbols'
    # nb : 'number of bits'
    nb = dict_M[mod] * n
    bits = np.random.randint(low=0, high=2, size=nb, dtype=np.int8)
    if mod == 'bpsk':  # noqa
        s = 2 * bits - 1
    elif mod == 'qpsk':  # noqa
        # 1/np.sqrt(2) = 0.7071067811865475
        # s = (2 * bits - 1) /np.sqrt(2)
        # s = (2 * bits - 1) * 0.7071067811865475
        s = bits.reshape(-1, 2).astype(np.float16)
        # s[s==[0, 0]] = [-0.7071, -0.7071]
        # s[s==[0, 1]] = [-0.7071, 0.7071]
        # s[s==[1, 0]] = [0.7071, -0.7071]
        # s[s==[1, 1]] = [0.7071, 0.7071]
        s[s == 0] = -0.7071
        s[s == 1] = 0.7071

    else:
        raise NotImplementedError

    return s, bits
    # reshape notes:
    # a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # b = a.reshape(-1, 2)
    #  [[1, 2],
    #   [3, 4],
    #   [5, 6],
    #   [7, 8]]
    # c = b.flatten()
    # array([1, 2, 3, 4, 5, 6, 7, 8])

def add_awgn(inputs, snr=10):  # noqa
    assert len(inputs.shape) == 1, 'Only 1 dimensional data supported!'
    n = len(inputs)
    # SNR = 10*log10(Eb/No)
    # Eb/No = 10 ^(SNR/10)
    # EB = 1 for BPSK  # noqa
    n0 = 10 ** (-snr / 10)
    # noise = np.sqrt(N0/2)*(np.randn(n, 1) + np.randn(n, 1))  # noqa
    noise = np.multiply(np.sqrt(n0 / 2), np.random.standard_normal(n))  # standard_normal: (mean=0, stdev=1)  # noqa
    output = np.add(inputs, noise)

    # TODO add complex noise too

    # plt.hist(noise, bins=40)
    # plt.show()

    return output
