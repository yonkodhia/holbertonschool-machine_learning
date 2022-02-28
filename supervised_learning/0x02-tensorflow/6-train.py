#!/usr/bin/env python3
"""
    Train
"""
import tensorflow.compat.v1 as tf
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    Method:
         builds, trains, and saves a neural network classifier:

    Parameters:
        @X_train (numpy.ndarray): training input data.
        @Y_train (numpy.ndarray):training labels.
        @X_valid (numpy.ndarray): validation input data.
        @Y_valid (numpy.ndarray):  validation labels.
        @layer_sizes (array): list containing the number of nodes
            in each layer of the network.
        @activations (array): list containing the activation
            functions for each layer of the network.
        @alpha (float): learning rate
        @iterations (int): number of iterations to train over
        @save_path (string): where to save the model.


    Returns:
        the path where the model was saved.
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection("x", x)
    tf.add_to_collection("y", y)
    tf.add_to_collection("y_pred", y_pred)
    tf.add_to_collection("loss", loss)
    tf.add_to_collection("accuracy", accuracy)
    tf.add_to_collection("train_op", train_op)

    # g = tf.Graph()

    # with g.as_default():
    #
    #     # create_placeholders(nx, classes)
    #     x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    #     # forward_prop(x, layer_sizes=[], activations=[])
    #     # creates the forward propagation graph for the neural network
    #     y_pred = forward_prop(x, layer_sizes, activations)
    #     # calculate_loss(y, y_pred)
    #     loss = calculate_loss(y, y_pred)
    #     # calculate_accuracy(y, y_pred)
    #     accuracy = calculate_accuracy(y, y_pred)

    #     # create_train_op(loss, alpha)
    #     # creates the training operation for the network
    #     train_op = create_train_op(loss, alpha)
    #

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            train_loss = sess.run(loss, feed_dict={x: X_train, y: Y_train})
            train_accuracy = sess.run(accuracy,
                                      feed_dict={x: X_train, y: Y_train})
            valid_loss = sess.run(loss,
                                  feed_dict={x: X_valid, y: Y_valid})
            valid_accuracy = sess.run(accuracy,
                                      feed_dict={x: X_valid, y: Y_valid})
            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})
            if i == iterations or i % 100 == 0:
                print('After {} iterations:'.format(i))
                print('\tTraining Cost: {}'.format(train_loss))
                print('\tTraining Accuracy: {}'.format(train_accuracy))
                print('\tValidation Cost: {}'.format(valid_loss))
                print('\tValidation Accuracy: {}'.format(valid_accuracy))

        save_path = saver.save(sess, save_path)
        return save_path
