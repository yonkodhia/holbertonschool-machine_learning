#!/usr/bin/env python3
"""l2_reg_cost"""
import tensorflow as tf


def l2_reg_cost(cost):
    """alculates the cost of a neural network """

    cost += tf.losses.get_regularization_losses()
    return cost
