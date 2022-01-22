#!/usr/bin/env python3
"""0-bag_of_words file"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """

    --  Function that create Bag of wards embedding wards --
    |______________________________________________________|

    sentences ==> a list of sentences to analyze
    vocab ==> a list of the vocabulary words to use for the analysis
    s ==> the number of sentences in sentences
    f ==> the number of features analyzed
    features ==> a list of the features used for embeddings
    Return embeddings & features
    """
    # Analyse vocabulary using CountVectorize:
    vect = CountVectorizer(vocabulary=vocab)

    # Transfrom sentences:
    V = vect.fit_transform(sentences)

    # Print features names selected:
    features = vect.get_feature_names()

    # Return embedding after modification using toarray:
    embeddings = V.toarray()

    # return embeddings & features:
    return embeddings, features
