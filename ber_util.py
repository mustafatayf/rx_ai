import numpy as np
import matplotlib.pyplot as plt
# from commpy.modulation import PSKModem, QAMModem  # noqa
from commpy.filters import rrcosfilter  # scikit-commpy

# import pandas as pd

dict_M = dict({
    'bpsk': 1,  # noqa
    'qpsk': 2,  # noqa
})


# def calc_ber():
#     raise NotImplementedError

def get_h(fs=10, g_delay=4, debug=False):
    # raised cosine (RC) filter (FIR) impulse response
    # TODO // verify and validate
    fd = 1
    alpha = 0.3
    n = 2*g_delay*fs + 1
    h = rrcosfilter(N=n, alpha=alpha, Ts=fd, Fs=fs)[1]
    h_norm = np.divide(h, np.sqrt(np.square(h).sum()), dtype=np.float16)
    if debug:
        plt.figure()
        plt.plot(h_norm)
        plt.show()
    # https://stackoverflow.com/a/25858023
    return h_norm


def gen_data(n, mod, seed):
    np.random.seed(seed)
    #  n : 'number of symbols'
    # nb : 'number of bits'
    nb = dict_M[mod] * n
    bits = np.random.randint(low=0, high=2, size=nb, dtype=np.int8)
    if mod == 'bpsk':  # noqa
        s = 2 * bits - 1
        s = np.reshape(s, (-1, 1))
    elif mod == 'qpsk':  # noqa
        # 1/np.sqrt(2) = 0.7071067811865475
        # s = (2 * bits - 1) /np.sqrt(2)
        # s = (2 * bits - 1) * 0.7071067811865475
        s = bits.reshape(-1, 2).astype(np.float16)
        # s[s==[0, 0]] = [-0.7071 -0.7071i]
        # s[s==[0, 1]] = [-0.7071 +0.7071i]
        # s[s==[1, 0]] = [0.7071 -0.7071i]
        # s[s==[1, 1]] = [0.7071 +0.7071i]
        s[s == 0] = -0.7071
        s[s == 1] = 0.7071
        # INFO data format for QPSK
        # s.shape() = (N, 2) >> real(I) (x-axis) 0th col, imag(Q) (y-axis) 1th col
        # [I1, Q1]
        # [I2, Q2]
        # ... ...
        # [IN, QN]

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
    assert len(inputs.shape) == 1 or len(inputs.shape) == 2, 'Only 1D and 2D data are supported!'
    n = len(inputs)
    outputs = np.empty(inputs.shape)
    # SNR = 10*log10(Eb/No)
    # Eb/No = 10 ^(SNR/10)
    # EB = 1 for BPSK  # noqa
    n0 = 10 ** (-snr / 10)
    # noise = np.sqrt(N0/2)*(np.randn(n, 1) + np.randn(n, 1))  # noqa
    noise_r = np.multiply(np.sqrt(n0 / 2), np.random.standard_normal(n))  # standard_normal: (mean=0, stdev=1)  # noqa
    if len(inputs.shape) == 1:
        outputs = np.add(inputs, noise_r)
    else:  # 2d case
        outputs[:, 0] = np.add(inputs[:, 0], noise_r)
        if inputs.shape[1] == 2:
            noise_i = np.multiply(np.sqrt(n0 / 2), np.random.standard_normal(n))  # standard_normal: (mean=0, stdev=1)  # noqa
            outputs[:, 1] = np.add(inputs[:, 1], noise_i)
    # TODO verify noise addition in complex case

    # plt.hist(noise, bins=40)
    # plt.show()

    return outputs


def bit_checker(bit_ref, bit_tc):
    """

    :param bit_ref: reference bit
    :param bit_tc: bits to check
    :return: noe and nob
    # noe : number of error
    # nob : number of bit
    """

    assert len(bit_ref) == len(bit_tc), 'mismatch on the input lengths'
    # noe = abs(np.subtract(bit_ref, bit_tc)).sum()
    nob = len(bit_ref)
    noe = nob - np.equal(bit_ref, bit_tc).sum()

    return noe, nob
