#!/usr/bin/env python3
"""1-tf_idf.py"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
      ____________________________________________________
    |      --  Function that create tf_idf embedding --    |
    | ____________________________________________________ |
    sentences ==> a list of sentences to analyze
    vocab ==> a list of the vocabulary words to use for the analysis
    s ==> the number of sentences in sentences
    f ==> the number of features analyzed
    features ==> a list of the features used for embeddings
    features is a list of the features used for embeddings
    """

    # Highlight words that are most important:
    vect = TfidfVectorizer(vocabulary=vocab)

    # Fit data:
    V = vect.fit_transform(sentences)

    # Print features names selected:
    features = vect.get_feature_names()

    # Return embedding after modification using toarray:
    embeddings = V.toarray()

    # Return embeddings & features:
    return embeddings, features
