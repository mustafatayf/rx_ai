""" Feature Extraction and Generations
name: ~
desc:   obtaining features from the sequences of sampled data on the RX side
status: draft, fix major error on the ISI removal
version: 0.0.2 (21 February 2024, 07:21)
"""

import numpy as np
from constants import cf0_5, cf0_6, cf0_7, cf0_8, cf0_9, cf1_0
from rx_utils import is_symmetric


def remove_isi(sequence, LoN, tau, merge=False):
    """
    sequence : input data, in form of [row, column] : each row corresponds to a single sample,
                columns are the element of each sample
    ** note that the features are based on individual rows, so that no correlation nor dependency between the rows

    merge : if True, concat the input data with the extracted features
            if False, return only the obtained features
    """

    # TODO : improve if-else structure: use direct coeffient
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

    isi_removed = []
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
        acsm = [0]*n  # accumulated sum of ISI effect
        acsm = np.array(acsm).astype('float32')
        for i in range(n):
            #        # TODO optimize the zero multiplications
            #        # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 ...  34 35 36 ... : current sequence (samples)
            #    #a    a ~ ~ ~ ~ ~ f g 0 0 0 0   0   0          >>> ISI coefficients
            #    #b    b ~ ~ ~ ~ ~ e f g 0 0 0   0   0
            #    #c    c ~ ~ ~ ~ ~ d e f g 0 0   0   0
            #    #d    d ~ ~ ~ ~ ~ ~ ~ ~ ~ 0   0   0
            #    #...
            #    #0    0 0 0 0 0 0 ...         0   0 g f e d c b a b c d e f g 0  0
            # acsm += np.multiply(s[i], cf_ext[(n-i-1):(2*n-i-1)])  # [0 .. 0 g f e d c b a b c d e f g 0 .. 0] 2n-1
            if s[i] > 0:
                acsm += np.array(cf_ext[(n-i-1):(2*n-i-1)])  # [0 ... 0 0 g f e d c b a b c d e f g 0 0 ... 0] 2n-1
            else:  # s[i] <= 0
                acsm -= np.array(cf_ext[(n-i-1):(2*n-i-1)])  # [0 ... 0 0 g f e d c b a b c d e f g 0 0 ... 0] 2n-1

        isi_removed.append(np.subtract(s, acsm))

    if merge:
        isi_removed = [a + b for a, b in zip(isi_removed, sequence)]

    return isi_removed
