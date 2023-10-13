import numpy as np
import pandas as pd
from ber_util import add_awgn

df = pd.read_csv('data/data_bpsk/tau1.00_snr10_bpsk.csv', names=['y', 'X'], header=None,)

matlab_noise = df['X']-df['y']
m_var = np.var(matlab_noise)

data_zero = np.zeros(matlab_noise.shape)

py_noise = add_awgn(data_zero, snr=10, seed=0)

py_var = np.var(py_noise)

dfc = pd.DataFrame()
dfc['matlab'] = matlab_noise
dfc['python'] = py_noise

dfc.plot(kind='hist', alpha=0.5, bins=40)

