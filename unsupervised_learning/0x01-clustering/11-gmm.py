#!/usr/bin/env python3
"""Do GMM with sklearn"""


import sklearn.mixture


def gmm(X, k):
    """Do GMM with sklearn"""
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)
    labels = gmm.predict(X)
    return gmm.weights_, gmm.means_, gmm.covariances_, labels, gmm.bic(X)
