#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions to manipulate the data.
Date: 30/6/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from numpy.random import default_rng 
from numpy.linalg import norm


def extract_data(data_name):
    """ Extracts the data from tensorflow datasets.
    Input:  data_name -> str {'cifar', 'mnist'}, name of the data
            to extract.
    Output: train_data -> tf.tensor[float32], list
                of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            test_data -> tf.tensor[float32], list
                of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> np.array[int], list of labels associated
                with the train examples.
            test_labels -> np.array[int], list of labels associated
                with the test examples.
    """

    try:
        assert(data_name == 'cifar' or data_name == 'mnist')

    except AssertionError:
        print("Error in function extract_data: the data name must be cifar or mnist")
        exit(1)

    if data_name == 'cifar':
        # Extract cifar10 data and labels
        train, test = tf.keras.datasets.cifar10.load_data()
        train_data = train[0]
        test_data = test[0]

    else:
        # Extract mnist data and labels and reshape
        train, test = tf.keras.datasets.mnist.load_data()
        train_data = tf.reshape(train[0], (train[0].shape[0], train[0].shape[1], train[0].shape[2], 1))
        test_data = tf.reshape(test[0], (test[0].shape[0], test[0].shape[1], test[0].shape[2], 1))

    train_labels = np.squeeze(train[1])
    test_labels = np.squeeze(test[1])

    return tf.cast(train_data[:20000], dtype=tf.float32), tf.cast(test_data[:4000], dtype=tf.float32), train_labels[:20000], test_labels[:4000]
    #return tf.cast(train_data, dtype=tf.float32), tf.cast(test_data, dtype=tf.float32), train_labels, test_labels


def prep_data(labels, class_nb=0):
    """ Prepares the labels for a one vs all strategy given a class.
    The first class must be 0.
    Input:  labels -> np.array[int], list of labels.
            class_nb -> int, desired class to be classified vs the others.
    Output: labels -> np.array[int], list of labels modified for
                one vs all strategy.
    """

    try:
        assert(isinstance(class_nb, int))
        assert(class_nb >= 0)
        assert(np.max(labels) > class_nb)

    except AssertionError:
        print("Error in function prep_data: class_nb must be a positive integer less then the number of classes.")
        exit(1)

    # Convert to two labels given the class number
    labels[labels != class_nb] = 0
    labels[labels == class_nb] = 1

    return labels
