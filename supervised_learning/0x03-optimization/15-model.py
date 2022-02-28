#!/usr/bin/env python3
"""Optimization module"""
import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """Shuffles the data points in two matrices the same way
    Args:
        X (np.ndarray): Is the first array with shape (m, nx) where m is the
            number of data points and nx is the number of features in X.
        Y (np.ndarray): is the second array with shape (m, ny) where ny is the
            number of features in Y
    Returns:
        np.ndarray: The shuffled X and Y matrices
    """
    m = X.shape[0]
    indices = list(np.random.permutation(m))
    X_shuffled = X[indices, :]
    Y_shuffled = Y[indices, :]

    return X_shuffled, Y_shuffled


def create_placeholders(nx, classes):
    """Creates two placeholder x and y
    Args:
        nx (int): Is the number of feature columns in our data
        classes (int): Is the number of classes in the classifier
    Returns:
        tf.placeholder: returns a placeholder for the input data to the neural
            network.
        tf.placeholder: returns a placeholder for the one-hot labels for the
            input data
    """
    x = tf.placeholder("float", shape=(None, nx), name="x")
    y = tf.placeholder("float", shape=(None, classes), name="y")
    return x, y


def create_layer(prev, n, activation):
    """Creates a tensorflow layer
    Args:
        prev (tf.tensor): Is tensor output of the previous layer
        n (int): Is the number of nodes in the layer.
        activation: Is the activation function that the layer should use
    Returns:
        tf.tensor: Returns the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    model = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name="layer",
    )
    output = model(prev)
    return output


def create_batch_norm_layer(prev, n, activation):
    """Creates a batch normalization layer for a neural network using
    tensorflow
    Args:
        prev (tf.Tensor): Is the activated output of the previous layer
        n (int): Is the number of nodes in the layer be created.
        acitvation (tf.nn.activation): Is the activation function that should
            be used on the output of the layer
    Returns:
        tf.Tensor: Is the activated output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, kernel_initializer=init)
    layered = layer(prev)
    gamma = tf.Variable(
        tf.constant(1.0, shape=[n]), name="gamma", trainable=True
    )
    beta = tf.Variable(
        tf.constant(0.0, shape=[n]), name="beta", trainable=True
    )
    mean, variance = tf.nn.moments(layered, axes=[0])
    epsilon = 1e-8
    normed = tf.nn.batch_normalization(
        layered, mean, variance, beta, gamma, epsilon
    )
    return activation(normed)


def forward_prop(x, layer_sizes=[], activations=[]):
    """Creates the forward propagation graph for the neural netword
    Args:
        x (tf.placeholder): Is the input data
        layer_sizes (list): Is containing the number of nodes in each layer
            of the network
        activations (list): Is containing the activation functions for each
            layer of the network.
    Returns:
        tf.Tensor: Returns the prediction of the network in tensor form
    """
    prediction = x
    for layer, activation in zip(layer_sizes, activations):
        if activation is None:
            prediction = create_layer(prediction, layer, activation)
        else:
            prediction = create_batch_norm_layer(prediction, layer, activation)
    return prediction


def calculate_accuracy(y, y_pred):
    """Calculates the accuracy of a prediction
    Args:
        y (tf.placeholder): Is the placeholder for the input data
        y_pred (tf.Tensor): Is a tensor containing the network's prediction
    Returns:
        tf.Tensor: Containing the decimal accuracy of the predcition
    """
    truth_max = tf.argmax(y, 1)
    pred_max = tf.argmax(y_pred, 1)
    difference = tf.equal(truth_max, pred_max)
    accuracy = tf.reduce_mean(tf.cast(difference, "float"))
    return accuracy


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction
    Args:
        y (tf.placeholder): the labels of the input data
        y_pred (tf.Tensor): Is a tensor containing the network's prediction
    Returns:
        tf.Tensor: Containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Creates the learning rate decay operation in tensorflow using inverse
    time decay
    Args:
        alpha (float): Is the original learning rate
        decay_rate (float): Is the weight used to determine the rate at which
            alpha will decay.
        global_step (int): Is the number of passes of gradient descent that
            have elapsed.
        decay_step (int): Is the number of passes of gradient descent that
            occur before alpha is decayed further.
    Returns:
        tf.operation: The learning rate decay operation
    """
    return tf.train.inverse_time_decay(
        alpha, global_step, decay_step, decay_rate, staircase=True
    )


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """Creates the training operation for a neural network in tensorflow
    using Adam algorithm
    Args:
        loss (tf.Tensor): Is the loss of the network
        alpha (float): Is the learning rate
        beta1 (float): Is the weight for the first moment
        beta2 (float): Is the weight for the second moment
        epsilon (float): Is the small number to avoid division by zero
    Returns:
        tf.operation: Adam optimization operation
    """
    adam = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    return adam.minimize(loss)


def model(
        Data_train, Data_valid, layers, activations,
        alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
        decay_rate=1, batch_size=32, epochs=5,
        save_path='/tmp/model.ckpt'
):
    """Builds, Trains and saves a neural network model in tensorflow using:
    Adam optimization, mini batch gradient descent, learning rate decay and
    batch normalization.

    Args:
        Data_train (tuple(numpy.ndarray)): Is a tuple containing the training
            inputs and training labels.
        Data_valid (tuple(numpy.ndarray)): Is a tuple containing the validi
            inputs and valid labels
        layers (list): Is a list containing the number of nodes in each layer
            of the network.
        activation (list): Is a list containing the activation functions used
            for each layer in the network.
        alpha (float): Is the learning rate
        beta1 (float): Is the weight for the first moment of Adam Optimization.
        beta2 (float): Is the weight for the second moment of Adam.
        epsilon (float): Is a small number used to avoid division by zero.
        decay_rate (float): Is the decay rate for inverse time decay of the
            learning rate.
        batch_size (int): Is the number of data points that should be in
            a mini-batch.
        epochs (int): Is the number of times the training should pass through
            the whole dataset.
        save_path (str): Is the path where the model should be saved to.

    Returns:
        str: The path where the model was saved

    """
    x_train, y_train = Data_train
    x_valid, y_valid = Data_valid

    m, nx = x_train.shape
    classes = y_train.shape[1]

    batches = m / batch_size
    if batches % 1 != 0:
        batches = int(batches + 1)
    else:
        batches = int(batches)

    x, y = create_placeholders(nx, classes)
    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection("y_pred", y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)

    global_step = tf.Variable(0)
    decay = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, decay, beta1, beta2, epsilon)
    tf.add_to_collection("train_op", train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init)
        for epoch in range(epochs + 1):
            train_cost, train_accuracy = session.run(
                [loss, accuracy],
                feed_dict={x: x_train, y: y_train}
            )
            valid_cost, valid_accuracy = session.run(
                [loss, accuracy],
                feed_dict={x: x_valid, y: y_valid}
            )
            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                x_shuffled, y_shuffled = shuffle_data(x_train, y_train)
                for batch in range(batches):
                    start = batch * batch_size
                    end = start + batch_size
                    if end > m:
                        end = m
                    x_batch = x_shuffled[start:end]
                    y_batch = y_shuffled[start: end]
                    session.run(
                        train_op,
                        feed_dict={x: x_batch, y: y_batch}
                    )
                    if (batch + 1) % 100 == 0 and batch > 0:
                        batch_cost, batch_accuracy = session.run(
                            [loss, accuracy],
                            feed_dict={x: x_batch, y: y_batch}
                        )
                        print("\tStep {}:".format(batch + 1))
                        print("\t\tCost: {}".format(batch_cost))
                        print("\t\tAccuracy: {}".format(batch_accuracy))
            session.run(tf.assign(global_step, global_step + 1))
        return saver.save(session, save_path)
