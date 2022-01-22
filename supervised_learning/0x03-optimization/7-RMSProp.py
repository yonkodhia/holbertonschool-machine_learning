#!/usr/bin/env python3

"""7-RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ function thatupdate variable"""
    rs = beta2 * s + (1 - beta2) * (grad ** 2)
    a = var - alpha * grad / (rs ** 0.5 + epsilon)
    return a, rs
