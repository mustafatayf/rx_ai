# import numpy as np

snr_to_nos = {0: 10000,
              1: 10000,
              2: 10000,
              3: 100000,
              4: 100000,
              5: 500000,
              6: 500000,
              7: 500000,
              8: 1000000,
              9: 1000000,
              10: 1000000,
              11: 1000000,
              12: 2000000,
              13: 3000000,
              14: 3000000,
              15: 4000000,
              16: 4000000,
              'NoNoise': 5000000
              }

# roll factor 0.3, gdelay 1, fs : 10 ~ for varying theta (do not use gdelay=1, it cause additional errors to data)
# h_21 = [-0.0237175706573207, 0.00796258376002411, 0.0479172896810874, 0.0942011356918112, 0.144150842824562,
#         0.194589161559187, 0.242089589947497, 0.283276330059327, 0.315129331838160, 0.335263018086804,
#         0.342149545262556,
#         0.335263018086804, 0.315129331838160, 0.283276330059327, 0.242089589947497, 0.194589161559187,
#         0.144150842824562, 0.0942011356918112, 0.0479172896810874, 0.00796258376002411, -0.0237175706573207]
# import numpy as np
# h_21_norm = np.sqrt(sum(np.square(h_21)))
# hh_21 = h_21/h_21_norm
hh_21 = [-0.0242184, 0.00813073, 0.04892913, 0.09619033, 0.1471948,
         0.1986982, 0.24720167, 0.28925812, 0.32178375, 0.34234259,
         0.34937453, 0.34234259, 0.32178375, 0.28925812, 0.24720167,
         0.1986982, 0.1471948, 0.09619033, 0.04892913, 0.00813073,
         -0.0242184]
# hh_21_norm = np.sqrt(sum(np.square(hh_21)))

# roll factor 0.3, gdelay 4, fs : 10
h_81 = [0.00376269812419313, 0.00471337435592727, 0.00482166178880308, 0.00394250078618065, 0.00208448894874903,
        -0.000571498551744591, -0.00367916587359419, -0.00676092379084138, -0.00926712018759014, -0.0106541973225301,
        -0.0104721244330486, -0.00844906382948057, -0.00456035879416295, 0.000930166019828655, 0.00746590728241009,
        0.0142352508683435, 0.0202511199226200, 0.0244662284202333, 0.0259124907726173, 0.0238486447341953,
        0.0178975045338370, 0.00815385367042497, -0.00475397518075267, -0.0196614000546128, -0.0349300149982486,
        -0.0485738175029959, -0.0584399856646967, -0.0624276438876449, -0.0587223182124205, -0.0460204126242920,
        -0.0237175706573207, 0.00796258376002411, 0.0479172896810874, 0.0942011356918112, 0.144150842824562,
        0.194589161559187, 0.242089589947497, 0.283276330059327, 0.315129331838160, 0.335263018086804,
        0.342149545262556,
        0.335263018086804, 0.315129331838160, 0.283276330059327, 0.242089589947497, 0.194589161559187,
        0.144150842824562, 0.0942011356918112, 0.0479172896810874, 0.00796258376002411, -0.0237175706573207,
        -0.0460204126242920, -0.0587223182124205, -0.0624276438876449, -0.0584399856646967, -0.0485738175029959,
        -0.0349300149982486, -0.0196614000546128, -0.00475397518075267, 0.00815385367042497, 0.0178975045338370,
        0.0238486447341953, 0.0259124907726173, 0.0244662284202333, 0.0202511199226200, 0.0142352508683435,
        0.00746590728241009, 0.000930166019828655, -0.00456035879416295, -0.00844906382948057, -0.0104721244330486,
        -0.0106541973225301, -0.00926712018759014, -0.00676092379084138, -0.00367916587359419, -0.000571498551744591,
        0.00208448894874903, 0.00394250078618065, 0.00482166178880308, 0.00471337435592727, 0.00376269812419313]

# import numpy as np
#
# hPSF = np.array(h_81).astype(np.float16)
# coeff = np.convolve(hPSF, hPSF)
# FS = 10
# cof = coeff[len(coeff) // 2::int(tau * FS)]
# TODO: verify all the ISI coefficients cf0_7, cf0_8, cf0_9, cf1_0,
# by MATLAB, 21 feb 2024 :: Sampling frequency (FS): 10, Group Delay: 4, roll-off parameter (alpha): 03
cf0_5 = [0.999673332233851, 0.623508772458626, 0.000234953148749319, -0.175364292344045, 0.000400953412550350,
         0.0731345679138845, -0.00197335771964228, -0.0278866173295919, 0.00592672860532467, 0.0123480223510405,
         -0.00188533586384390, -0.00275200667957527, 0.00117716537510007, 0.000330967943615960, -0.000336928810743597,
         5.33680112413447e-05, 1.41578971738065e-05]
cf0_6 = [0.999673332233851, 0.489744260255461, -0.138005115346012, -0.0785836712286997, 0.0761803798302392,
         -0.00197335771964228, -0.0225892450462842, 0.0137128251658678, 0.00316030198442527, -0.00360638857981096,
         0.00117716537510007, 3.67747514858938e-05, -0.000169112840945702, 5.85008133555589e-05]
cf0_7 = [0.999673332233851, 0.353335390144568, -0.183209848767000, 0.0324297315058517, 0.0315965962904600,
         -0.0278866173295919, 0.0137128251658678, 0.000330876438046485, -0.00173386067936321, 0.000981495360903930,
         -0.000336928810743597, 7.51214746821814e-05]
cf0_8 = [0.999673332233851, 0.221937404035197, -0.152178629829995, 0.0761803798302392, -0.0244057457825951,
         0.00592672860532467, 0.00316030198442527, -0.00173386067936321, 0.000663566889945621, -0.000169112840945702,
         1.41578971738065e-05]
cf0_9 = [0.999673332233851, 0.102387303195368, -0.0785836712286997, 0.0487214573950375, -0.0225892450462842,
         0.0123480223510405, -0.00360638857981096, 0.000981495360903930, -0.000169112840945702]
cf1_0 = [0.999673332233851, 0.000234953148749319, 0.000400953412550350, -0.00197335771964228, 0.00592672860532467,
         -0.00188533586384390, 0.00117716537510007, -0.000336928810743597, 1.41578971738065e-05]
# Python results
# # tau = 0.7, FS:10, GroupDelay: 4
# cf0_7 = [9.995e-01, 3.533e-01, -1.832e-01, 3.241e-02, 3.159e-02,
#          -2.788e-02, 1.371e-02, 3.309e-04, -1.734e-03, 9.813e-04,
#          -3.371e-04, 7.516e-05]
# # tau = 0.8, FS:10, GroupDelay: 4
# cf0_8 = [9.995e-01, 2.219e-01, -1.522e-01, 7.617e-02, -2.440e-02,
#          5.928e-03, 3.160e-03, -1.734e-03, 6.638e-04, -1.692e-04,
#          1.419e-05]
# # tau = 0.9, FS:10, GroupDelay: 4
# cf0_9 = [9.9951e-01, 1.0236e-01, -7.8613e-02, 4.8706e-02, -2.2583e-02,
#          1.2352e-02, -3.6068e-03, 9.8133e-04, -1.6916e-04]
# # tau = 1.0, FS:10, GroupDelay: 4
# cf1_0 = [9.995e-01, 2.166e-04, 3.786e-04, -1.966e-03, 5.928e-03,
#          -1.885e-03, 1.177e-03, -3.371e-04, 1.419e-05]

# ref_ber_bpsk = {'SNR': [0, 2, 4, 6, 8, 10],
#                 'BER?': [0.0869565217391304, 0.0348275862068966, 0.0143884892086331,
#                          0.00209863588667366, 0.000184365781710914, 1.00000000000000e-06]
#                 }

gbKSE = {'SNR': [0, 2, 4, 6, 8, 10, 12],
         'gbKSE_BER08': [0.134666666666667, 0.0612121212121212, 0.0268421052631579, 0.00759398496240602,
                         0.00175131348511384, 0.000305576776165011, 2.30000000000000e-05]
         }

BERtau1 = {'SNR': [0, 2, 4, 6, 8, 10],
           'BER_tau1': [0.0762962962962963, 0.0403921568627451, 0.0135135135135135,
                        0.00230946882217090, 0.000167841557569654, 4.50000000000000e-06]
           }

# BCJR = {'SNR': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
BCJR = {'SNR': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'BCJR_BER07': [0.112899604885057, 0.0871930803571429, 0.0649705091059603, 0.0422838879870130,
                       0.0253703327922078, 0.0126788464656291, 0.00556695798022599, 0.00178739450273723,
                       0.000415735419327373, 6.14113004653503e-05, 7.81187504999600e-06],
        'BCJR_BER08': [0.0929208431603774, 0.0685096153846154, 0.0468189294258373, 0.0301860291280864,
                       0.0161463713842975, 0.00793387616774980, 0.00321831239703460, 0.00111099260523322,
                       0.000261392532119914, 5.06883888715873e-05, 6.03076946828877e-06],
        'BCJR_BER09': [0.0830769230769231, None, 0.0400000000000000, None,
                       0.0133333333333333, None, 0.00211416490486258, None,
                       0.000144175317185698, None, 5.45170065802027e-06]
        }

TRBER = {'SNR': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
         # 'models/tau0.50_gru_temel_2023Oct14_1540' older version of gru_temel
         'gru_tau0.5': [0.197, 0.175, 0.158, 0.1415,
                        0.1179, 0.0966, 0.0779, 0.0614,
                        0.0476, 0.03593, 0.026224, 0.020479,
                        0.016684, 0.014198, 0.012496, 0.0112875,
                        0.01053525, 0.01013, 0.009732, 0.009462, 0.00921],
         # 'models/tau0.60_gru_temel_2023Oct14_1748'
         'gru_tau0.6': [0.167, 0.141, 0.114, 0.0992,
                        0.0736, 0.0522, 0.034, 0.02123,
                        0.01246, 0.00631, 0.002926, 0.001493,
                        0.000753, 0.00042633, 0.00030067, 0.00024875,
                        0.00022725, 0.0002185, 0.00021125, 0.00020475, 0.0002005],
         # 'models/tau0.70_gru_temel_2023Oct14_1833'
         'gru_tau0.7': [1.42000000e-01, 1.16000000e-01, 9.10000000e-02, 5.43000000e-02,
                        3.61000000e-02, 2.22000000e-02, 1.13000000e-02, 5.57000000e-03,
                        2.43000000e-03, 9.50000000e-04, 4.19000000e-04, 2.14000000e-04,
                        1.23000000e-04, 9.06666667e-05, 7.36666667e-05, 5.95000000e-05,
                        5.50000000e-05, 5.27500000e-05, 4.95000000e-05, 4.82500000e-05,
                        4.72500000e-05],
         # 'models/tau0.80_gru_temel_2023Oct14_1719'
         'gru_tau0.8': [1.21000000e-01, 8.50000000e-02, 6.40000000e-02, 3.43000000e-02,
                        2.05000000e-02, 9.70000000e-03, 4.30000000e-03, 1.72000000e-03,
                        5.30000000e-04, 7.00000000e-05, 3.90000000e-05, 1.10000000e-05,
                        3.00000000e-06, 1.66666667e-06, 1.33333333e-06, 7.50000000e-07,
                        2.50000000e-07, 2.50000000e-07, 0.00000000e+00, 0, 0],  # son iki 0 değeri el ile eklendi
         # 'models/tau0.90_gru_temel_2023Oct14_1732'
         'gru_tau0.9': [7.70e-02, 6.00e-02, 4.40e-02, 2.44e-02,
                        1.32e-02, 6.70e-03, 2.80e-03, 9.30e-04,
                        2.60e-04, 4.00e-05, 7.00e-06, 0.00e+00,
                        0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0 değerleri el ile eklendi
         # 'models/tau1.00_gru_temel_2023Oct14_1744'
         'gru_tau1.0': [9.00e-02, 5.90e-02, 3.50e-02, 2.16e-02,
                        1.15e-02, 5.20e-03, 2.10e-03, 8.20e-04,
                        1.70e-04, 3.00e-05, 5.00e-06, 2.00e-06,
                        0.00e+00, 0, 0, 0, 0, 0, 0, 0, 0]  # 0 değerleri el ile eklendi
         }

# just looking for the sign of the sample at the rx side
base = {'SNR': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 'NoNoise'],
        'tau0.5': [0.79667, 0.804071, 0.809898, 0.813937, 0.816667, 0.817662, 0.817761, 0.816952, 0.815478, 0.813695,
                   0.811862, 0.810087, 0.807851, 0.806266, 0.80512167, 0.80426975, 0.80351775, 0.7998086],
        'tau0.6': [0.83111, 0.842521, 0.852244, 0.860167, 0.866422, 0.871072, 0.874439, 0.876807, 0.878458, 0.879612,
                   0.880385, 0.880937, 0.8809055, 0.88113567, 0.88131533, 0.88165275, 0.88177125, 0.8847634],
        'tau0.7': [0.862663, 0.878292, 0.892234, 0.904047, 0.914038, 0.922503, 0.929516, 0.935432, 0.940181, 0.94396,
                   0.946959, 0.949341, 0.9508545, 0.952176, 0.953066, 0.9537535, 0.95385525, 0.9481226],
        'tau0.8': [0.891089, 0.909884, 0.926825, 0.940862, 0.952811, 0.962656, 0.970306, 0.976472, 0.981283, 0.985074,
                   0.987994, 0.99032, 0.992023, 0.99348267, 0.994616, 0.99559275, 0.99636225, 1.0],
        'tau0.9': [0.913665, 0.935095, 0.953565, 0.968557, 0.979954, 0.987958, 0.993257, 0.996547, 0.998374, 0.999324,
                   0.999705, 0.999894, 0.999975, 0.999991, 0.9999993, 1.0, 1.0, 1.0],
        'tau1.0': [0.921067, 0.943641, 0.962434, 0.977043, 0.987601, 0.994137, 0.997714, 0.999216, 0.999801, 0.999962,
                   0.999996, 0.999999, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }

bpsk = {'SNR': [-4, -2, 0, 2, 4, 6, 8, 10, 12],
        # 'tau1.0': [-0.729917, -0.884020, -1.104215, -1.425779, -1.901612,
        # -2.622876, -3.719422, -5.522879, -np.Infinity] # in dB

        }
