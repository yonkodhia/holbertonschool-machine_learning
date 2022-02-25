#!/usr/bin/env python3
""" function to train model """
import tensorflow.keras as K


def train_model(network, data, labels,
                batch_size, epochs,
                validation_data=None,
                early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True,
                shuffle=False):
    """function to train model with keras"""
    def scheduler(epoch):
        """function scheduler"""
        return alpha / (1 + decay_rate * epoch)

    x = []
    if early_stopping is True:
        p = K.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=patience)
        x.append(p)
    if learning_rate_decay and validation_data:
        p = K.callbacks.LearningRateScheduler(scheduler,
                                              verbose=1)
        x.append(p)
    if save_best and validation_data:
        a = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        x.append(a)
    re = network.fit(x=data, y=labels,
                     callbacks=x,
                     epochs=epochs,
                     batch_size=batch_size,
                     validation_data=validation_data,
                     verbose=verbose,
                     shuffle=shuffle)
    return re
