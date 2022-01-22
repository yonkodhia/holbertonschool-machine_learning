#!/usr/bin/env python3
"""function pool_forward"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Convol prop manually"""
    bot = A_prev.shape[1] - kernel_shape[0] + 1
    r = A_prev.shape[2] - kernel_shape[1] + 1
    res = np.ndarray((A_prev.shape[0], int((bot - 1) / stride[0] + 1),
                      int((r - 1) / stride[1] + 1), A_prev.shape[3]))
    in1 = 0
    out1 = 0
    while in1 < bot:
        in2 = 0
        out2 = 0
        while in2 < r:
            slice = A_prev[:, in1:in1 + kernel_shape[0],
                           in2:in2 + kernel_shape[1], :]
            if mode == 'max':
                res[:, out1, out2] = np.amax(slice, axis=(1, 2))
            else:
                res[:, out1, out2] = np.mean(slice, axis=(1, 2))
            in2 += stride[1]
            out2 += 1
        in1 += stride[0]
        out1 += 1
    return res
