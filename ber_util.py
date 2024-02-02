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
    """

    :param n: number of symbol to generate
    :param mod:
    :param seed:
    :return:
            syms : (n, ) 1D numpy array, modulated symbols; no differentiation for real(I) and imaginary(Q) parts
            bits : (nb, ) 1D numpy array, 0/1 message bits
    """
    np.random.seed(seed)
    #  n : 'number of symbols'
    # nb : 'number of bits'
    nb = dict_M[mod] * n
    bits = np.random.randint(low=0, high=2, size=nb, dtype=np.int8)
    if mod == 'bpsk':  # noqa
        syms = 2 * bits - 1
        # syms = np.reshape(syms, (-1, 1))
        # make 1D real valued BPSK symbols to 2D (real, 0*imag) format
        # syms = np.concatenate((syms, np.zeros((n, 1))), axis=1)
    elif mod == 'qpsk':  # noqa
        # 1/np.sqrt(2) = 0.7071067811865475
        # s = (2 * bits - 1) /np.sqrt(2)
        # s = (2 * bits - 1) * 0.7071067811865475
        # syms = bits.reshape(-1, 2).astype(np.float16)
        syms = bits.reshape(-1).astype(np.float16)  # flatten the symbols I and Q data into 1D array
        # s[s==[0, 0]] = [-0.7071 -0.7071i]
        # s[s==[0, 1]] = [-0.7071 +0.7071i]
        # s[s==[1, 0]] = [0.7071 -0.7071i]
        # s[s==[1, 1]] = [0.7071 +0.7071i]
        syms[syms == 0] = -0.7071
        syms[syms == 1] = 0.7071
        # INFO data format for QPSK
        # s.shape() = (N, 2) >> real(I) (x-axis) 0th col, imag(Q) (y-axis) 1th col
        # [I1, Q1]
        # [I2, Q2]
        # ... ...
        # [IN, QN]

    else:
        raise NotImplementedError

    return syms, bits
    # reshape notes:
    # a = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    # b = a.reshape(-1, 2)
    #  [[1, 2],
    #   [3, 4],
    #   [5, 6],
    #   [7, 8]]
    # c = b.flatten()
    # array([1, 2, 3, 4, 5, 6, 7, 8])


# def awgn(signal, desired_snr):
#     """
#     Add AWGN to the input signal to achieve the desired SNR level.
#
#     source: https://saturncloud.io/blog/python-numpy-implementing-an-additive-white-gaussian-noise-function/
#     """
#     # Calculate the power of the signal
#     signal_power = np.mean(signal ** 2)
#
#     # Calculate the noise power based on the desired SNR and signal power
#     noise_power = signal_power / (10 ** (desired_snr / 10))
#
#     # Generate the noise with the calculated power
#     noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
#
#     # Add the noise to the original signal
#     noisy_signal = signal + noise
#
#     return noisy_signal


# def add_awgn2(x_volts, target_snr_db):
#     # https://www.rfwireless-world.com/source-code/Python/AWGN-python-script.html
#
#     x_watts = x_volts ** 2
#     x_db = 10 * np.log10(x_watts)
#
#     # Calculate signal power and convert to dB
#     sig_avg_watts = np.mean(x_watts)
#     sig_avg_db = 10 * np.log10(sig_avg_watts)
#     # print("SMR in dB = ", sig_avg_db)
#
#     # Calculate noise and convert it to watts
#     noise_avg_db = sig_avg_db - target_snr_db
#     # print("Average Noise power in dB = ", noise_avg_db)
#     noise_avg_watts = 10 ** (noise_avg_db / 10)
#     # Generate samples of white noise
#     mean_noise = 0
#     noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
#
#     # Add noise to original sine waveform signal
#     y_volts = x_volts + noise_volts
#
#     return y_volts

def add_awgn(inputs, snr=10, seed=1234):  # noqa
    if snr == 'NoNoise':
        return inputs
    assert len(inputs.shape) == 1 or len(inputs.shape) == 2, 'Only 1D and 2D data are supported!'
    np.random.seed(seed)
    n = len(inputs)
    outputs = np.empty(inputs.shape).astype(np.float16)
    # SNR = 10*log10(Eb/No)
    # Eb/No = 10 ^(SNR/10)
    # EB = 1 for BPSK  # noqa
    n0 = 10 ** (-snr / 10)
    # noise = np.sqrt(N0/2)*(np.randn(n, 1) + np.randn(n, 1))  # noqa
    # noise_r = np.multiply(np.sqrt(n0 / 2), np.random.standard_normal(n))  # standard_normal: (mean=0, stdev=1)  # noqa
    noise_r = np.multiply(np.sqrt(n0 / 2), np.random.randn(n).astype(np.float16))  # standard_normal: (mean=0, stdev=1)  # noqa
    if len(inputs.shape) == 1:
        outputs = np.add(inputs, noise_r)
    else:  # 2d case
        outputs[:, 0] = np.add(inputs[:, 0], noise_r)
        if inputs.shape[1] == 2:
            # noise_i = np.multiply(np.sqrt(n0 / 2), np.random.standard_normal(n))  # standard_normal: (mean=0, stdev=1)  # noqa
            noise_i = np.multiply(np.sqrt(n0 / 2), np.random.randn(n))  # standard_normal: (mean=0, stdev=1)  # noqa
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
