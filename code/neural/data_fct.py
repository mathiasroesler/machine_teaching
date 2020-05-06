#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the function to read and edit the mnist data.
Date: 6/3/2020
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

        # Convert data to grayscale
        train_data = tf.image.rgb_to_grayscale(train[0]) 
        test_data = tf.image.rgb_to_grayscale(test[0])

    else:
        # Extract mnist data and labels and reshape
        train, test = tf.keras.datasets.mnist.load_data()
        train_data = tf.reshape(train[0], (train[0].shape[0], train[0].shape[1], train[0].shape[2], 1))
        test_data = tf.reshape(test[0], (test[0].shape[0], test[0].shape[1], test[0].shape[2], 1))

    train_labels = np.squeeze(train[1])
    test_labels = np.squeeze(test[1])

    return tf.cast(train_data[:20000], dtype=tf.float32), tf.cast(test_data[:4000], dtype=tf.float32), train_labels[:20000], test_labels[:4000]


def prep_data(labels, class_nb):
    """ Prepares the labels for a one vs all strategy.
    Changes the labels that are not equal to class_nb to 0 and the
    others to 1.
    Input:  labels -> np.array[int], list of labels associated
                with the train data.
            class_nb -> int, desired class to be classified vs the others.
    Output: labels -> tf.tensor[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, one hot label.
    """
    
    try:
        assert(isinstance(class_nb, int))
        assert(class_nb >= 0)
        assert(max(labels) > class_nb)

    except AssertionError:
        if max(labels) == 1:
            # If there are only two classes already
            return tf.one_hot(labels, 2)

        else:
            print("Error in function prep_data: class_nb must be a positive integer less then the number of classes.")
            exit(1)

    # Labels for the data
    positive_indices = np.nonzero(labels == class_nb)[0]
    negative_indices = np.nonzero(labels != class_nb)[0]

    labels[positive_indices] = 1
    labels[negative_indices] = 0

    return tf.one_hot(labels, 2)
