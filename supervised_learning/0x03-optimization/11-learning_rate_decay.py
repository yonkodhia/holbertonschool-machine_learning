#!/usr/bin/env python3
"""
11-learning_rate_decay.py"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ updates the learning rate using inverse time decay"""

    rt = decay_rate
    ur = alpha / (1 + rt * int(global_step / decay_step))
    return ur
