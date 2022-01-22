#!/usr/bin/env python3
"""
file lenet5
"""

import tensorflow as tf


def lenet5(x, y):
    """function lent 5"""
    it_ = tf.contrib.layers.variance_scaling_initializer()
    cv_lyr1 = tf.layers.Conv2D(filters=6,
                               kernel_size=(5, 5),
                               padding='same',
                               kernel_initializer=it_,
                               activation=tf.nn.relu)(x)
    pool_lyr_2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(cv_lyr1)
    mv_lyr3 = tf.layers.Conv2D(filters=16,
                               kernel_size=(5, 5),
                               padding='valid',
                               kernel_initializer=it_,
                               activation=tf.nn.relu)(pool_lyr_2)
    pool_lyr_4 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(mv_lyr3)
    flatten5 = tf.layers.Flatten()(pool_lyr_4)
    fc_lyr_5 = tf.contrib.layers.fully_connected(inputs=flatten5,
                                                 num_outputs=120,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=it_)
    fc_lyr_6 = tf.contrib.layers.fully_connected(inputs=fc_lyr_5,
                                                 num_outputs=84,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=it_)
    sfmx_ = tf.contrib.layers.fully_connected(inputs=fc_lyr_6,
                                              num_outputs=10,
                                              activation_fn=None,
                                              weights_initializer=it_)
    sfmx_lyr = tf.nn.softmax(sfmx_)
    losses = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=sfmx_)
    predictions = tf.equal(tf.argmax(sfmx_lyr, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))
    train_operation = tf.train.AdamOptimizer().minimize(losses)

    return (sfmx_lyr, train_operation, losses, accuracy)
