#!/usr/bin/env python3
""" confusion matrix function """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ create confusion matrix """
    return np.matmul(labels.T, logits)
