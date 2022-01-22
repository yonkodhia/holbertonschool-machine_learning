#!/usr/bin/env python3
""" Pool_backward """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape,
                  stride=(1, 1), mode='max'):
    """function backward to pool layers"""
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)
    for a in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    init = i * sh
                    end = i * sh + kh
                    init1 = j * sw
                    end1 = j * sw + kw
                    if mode == 'max':
                        value = np.max(A_prev[a,
                                              init:end,
                                              init1:end1,
                                              k])
                        mask = np.where(A_prev[a,
                                               init:end,
                                               init1:end1,
                                               k] == value, 1, 0)
                        mask = mask * dA[a, i, j, k]
                    elif mode == 'avg':
                        mask = np.ones(kernel_shape) * (dA[a, i, j, k] /
                                                        (kh * kw))
                    dA_prev[a, init:end,
                            init1:end1, k] += mask
    return dA_prev
