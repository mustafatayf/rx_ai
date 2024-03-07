""" Feature Extraction and Generations
name: ~
desc:   obtaining features from the sequences of sampled data on the RX side
status: draft, reorganize features, GetFeature class initiated
version: 0.0.3 (07 March 2024, 16:22)
"""

import numpy as np
from constants import cf0_5, cf0_6, cf0_7, cf0_8, cf0_9, cf1_0
# from rx_utils import is_symmetric


class GetFeature:
    """
    possible valuable features
    > Difference of received samples

    #                           # 1,1,0,1,1
    # possibilities;    0->0,       0->1,       1->0,       1->1
    # possibilities;    no change   increase    decrease    no change


    """
    def __init__(self, lon=2):
        assert lon >= 2, "minimum allowed LoN is 2!"
        self.lon = lon

    def s_diff(self, sequence):
        """
        sequence :

        """
        features = []
        for s in sequence:
            # get a single row of data
            # target sample : f
            # a b c d e >> f << g h i j k

            # get 1st diff
            # b c d e f g h i j k
            # a b c d e f g h i j
            # (b-a) (c-b) ...  (e-d) (f-e) :: (g-f) (h-g)... (k-j)
            df1 = s[1:] - s[:-1]
            # if rs:
            #     df1 = df1[lon:lon+2]

            # get the 2nd diff
            # c d e f g h i j k
            # a b c d e f g h i
            # ...  (e-c) (f-d) (g-e) (h-f) (i-g)...
            df2 = s[2:] - s[:-2]
            # df2 //= 2  :: Do not SCALE down
            # if rs:
            #     df2 = df2[lon-2], df2[lon]

            # get the 3rd diff
            # d e f g h i j k
            # a b c d e f g h
            # (d-a) (e-b) (f-c) (g-d) (h-e) (i-f) ...
            df3 = s[3:] - s[:-3]
            # df3 //= 4  :: Do not SCALE down

            # 2nd order dif1
            # dff1 = df1[1:] - df1[:-1]

            features.append(np.concatenate((df1, df2, df3), axis=0))

        # fit type and dimension
        np.array(features)
        return features


def get_feature(sequence, lon, tau, merge=False):
    """
    sequence : input data, in form of [row, column] : each row corresponds to a single sample,
                columns are the element of each sample
    ** note that the features are based on individual rows, so that no correlation nor dependency between the rows

    merge : if True, concat the input data with the extracted features
            if False, return only the obtained features
    """
    assert lon >= 2, "minimum allowed LoN is 2!"

    # TODO : improve if-else structure: use direct coefficient
    if tau == 0.5:
        cf = cf0_5  # isi coefficients
    elif tau == 0.6:
        cf = cf0_6  # isi coefficients
    elif tau == 0.7:
        cf = cf0_7  # isi coefficients
    elif tau == 0.8:
        cf = cf0_8
    elif tau == 0.9:
        cf = cf0_9
    elif tau == 1.0:
        cf = cf1_0
    else:
        assert False, "invalid tau value, "  # TODO improve assert case

    # diff margin threshold
    dmt = 0.2

    # DEBUG sequence = [[*range(7)]]
    n = len(sequence[0])
    if len(cf) > n:
        cf = cf[:n]  # crop the coefficient vector
    elif len(cf) < n:
        cf += [0] * (n - len(cf))  # zero padding
    # else: len(cf) == n:
    #       use the cf as is
    # [g f e d c b a b c d e f g]
    cf_ext = cf[:0:-1] + [0] + cf[1:]
    # assert is_symmetric(cf_ext), "ISI coefficient have to be symmetric!"  # TODO

    # isi_removed = []
    features = []
    c = n // 2  # index of center element: symbol value index (len-1)/2
    #  -3 -2 -1 S 1 2 3  e.g. LoN: 3
    # >> : circular shift
    # e.g. cf [a b c d e f g], : cf involves the ISI coefficients
    #                  -3 -2 -1 S 1 2 3
    # s(-3) * cf        [a b c d e f g]
    # s(-2) * cf >> 1   [b a b c d e f]
    # s(-1) * cf >> 2   [c b a b c d e]
    # s(0)  * cf >> 3   [d c b a b c d] note that a: 0.96, b: 0.35, c:-0.18, d:
    # s(1)  * cf >> 4   [e d c b a b c]
    # s(2)  * cf >> 5   [f e d c b a b]
    # s(3)  * cf >> 6   [g f e d c b a]

    for s in sequence:
        # get a single row of data
        # sv = s[c]
        # sr = np.multiply(sv, cf)
        # sr = np.subtract(np.array(sequence), sr)
        # isi_removed.append(sr.tolist())
        # acsm = [0]*n  # accumulated sum of ISI effect
        # acsm = np.array(acsm).astype('float32')
        # s.append(0)
        #                           # 1,1,0,1,1
        # possibilities;    0->0,       0->1,       1->0,       1->1
        # possibilities;    no change   increase    decrease    no change

        # target sample : f
        # a b c d e >> f << g h i j k

        # get 1st diff
        # b c d e f g h i j k
        # a b c d e f g h i j
        # (b-a) (c-b) ...  (e-d) (f-e) :: (g-f) (h-g)... (k-j)
        df1 = s[1:] - s[:-1]
        # if rs:
        #     df1 = df1[lon:lon+2]

        # get the 2nd diff
        # c d e f g h i j k
        # a b c d e f g h i
        # ...  (e-c) (f-d) (g-e) (h-f) (i-g)...
        df2 = s[2:] - s[:-2]
        # df2 //= 2  :: Do not SCALE down
        # if rs:
        #     df2 = df2[lon-2], df2[lon]

        # get the 3rd diff
        # d e f g h i j k
        # a b c d e f g h
        # (d-a) (e-b) (f-c) (g-d) (h-e) (i-f) ...
        df3 = s[3:] - s[:-3]
        # df3 //= 4  :: Do not SCALE down

        # 2nd order dif1
        # dff1 = df1[1:] - df1[:-1]

        # cf_ext
        # acsm -= np.multiply(df1, np.array(cf_ext[(n-i-1):(2*n-i-1)]))  # [0 ... 0 0 g f e d c b a b c d e f g 0 0 ... 0] 2n-1

        # features.append(np.subtract(s, dif1))
        features.append(np.concatenate((df1, df2, df3), axis=0))
        # features.append(np.concatenate((df1, df2, dff1), axis=0))

        # for i in range(n-1):
        #        # TODO optimize the zero multiplications
        #        # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 ...  34 35 36 ... : current sequence (samples)
        #    #a    a ~ ~ ~ ~ ~ f g 0 0 0 0   0   0          >>> ISI coefficients
        #    #b    b ~ ~ ~ ~ ~ e f g 0 0 0   0   0
        #    #c    c ~ ~ ~ ~ ~ d e f g 0 0   0   0
        #    #d    d ~ ~ ~ ~ ~ ~ ~ ~ ~ 0   0   0
        #    #...
        #    #0    0 0 0 0 0 0 ...         0   0 g f e d c b a b c d e f g 0  0
        # acsm += np.multiply(s[i], cf_ext[(n-i-1):(2*n-i-1)])  # [0 .. 0 g f e d c b a b c d e f g 0 .. 0] 2n-1
        #   df1 = s[i+1] - s[i]
        #   df2 = s[i] - s[i-1]
        #   df3 = s[i+2] - s[i+1]
        #   df4 = s[i+3] - s[i+2]
        #
        # s[i+1] - s[i]  # adj symb ISI
        # s[i+2] - s[i]  # 2nd adj ISI, (should have less effect) less weighted

        # # if s[i] > 0:
        # acsm -= np.multiply(df, np.array(cf_ext[(n-i-1):(2*n-i-1)]))  # [0 ... 0 0 g f e d c b a b c d e f g 0 0 ... 0] 2n-1
        # # else:  # s[i] <= 0
        # #     acsm -= np.array(cf_ext[(n-i-1):(2*n-i-1)])  # [0 ... 0 0 g f e d c b a b c d e f g 0 0 ... 0] 2n-1

        # isi_removed.append(np.subtract(s, acsm))

    # fit type and dimension
    np.array(features)
    # features = np.expand_dims(np.array(features), axis=1)
    if merge:
        # isi_removed = [a + b for a, b in zip(isi_removed, sequence)]
        # return [a + b for a, b in zip(sequence, features)]
        # if rs:
        #     return np.concatenate((sequence[:, lon-1:2*lon-2], features), axis=1)
        #     # return np.concatenate((sequence[lon-k:2*lon-k], features), axis=1)
        # else:
        return np.concatenate((sequence, features), axis=1)
    else:
        return features
