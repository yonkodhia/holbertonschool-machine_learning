#!/usr/bin/env python3
"""Calculate symmetric P"""


import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """function P affinities"""
    tol = 1 + tol
    D, P, betas, H = P_init(X, perplexity)
    for i in range(P.shape[0]):
        print(i)
        _, P[i] = HP(P[i], betas[i])
        inflect = 0
        while True:
            print(betas[i], H)
            if betas[i] < H / tol:
                print("too low")
                if inflect == -1:
                    betas[i] *= 1.5
                else:
                    betas[i] *= 2
                inflect = 0
                _, P[i] = HP(P[i], betas[i])
            elif betas[i] > H * tol:
                print("too high")
                if inflect == 1:
                    betas[i] *= .75
                else:
                    betas[i] *= .5
                inflect = -1
                _, P[i] = HP(P[i], betas[i])
            else:
                break
    return P
