#!/usr/bin/env python3
""" function thatConvol Forward Prop """
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """ function prefrom forward prop"""
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        p1 = int(((h_prev * (sh - 1)) - sh + kh) / 2)
        p2 = int(((w_prev * (sw - 1)) - sw + kw) / 2)
    elif type(padding) == tuple:
        p1, p2 = padding
    else:
        p1 = 0
        p2 = 0
    img_pad = np.pad(A_prev, ((0, 0), (p1, p1),
                              (p2, p2), (0, 0)),
                     'constant', constant_values=(0))
    img_pad_h = img_pad.shape[1]
    img_pad_w = img_pad.shape[2]
    h_out = int((img_pad_h - kh) / sh) + 1
    w_out = int((img_pad_w - kw) / sw) + 1
    result = np.zeros((m, h_out, w_out, c_new))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                result[:, i, j, k] = np.sum(img_pad[:,
                                                    i * sh: i * sh + kh,
                                                    j * sw: j * sw + kw] *
                                            W[:, :, :, k],
                                            axis=(1, 2, 3))
    return activation(result + b)
