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

    assert(data_name == 'cifar' or data_name == 'mnist')

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

    return tf.cast(train_data[:10000], dtype=tf.float32), tf.cast(test_data[:2000], dtype=tf.float32), train_labels[:10000], test_labels[:2000]


def prep_data(train_labels, test_labels, class_nb):
    """ Prepares the MNIST data for a one vs all strategy.
    Changes the labels that are not equal to class_nb to 0 and the
    others to 1.
    Returns None, None if an error occured.
    Input:  train_labels -> np.array[int], list of labels associated
                with the train data.
            test_labels -> np.array[int], list of labels associated
                with the test data.
            class_nb -> int, desired class to be classified vs the others.
    Output: train_labels -> tf.tensor[int], list of labels associated
                with the train data.
                First dimension, number of examples.
                Second dimension, one hot label.
            test_labels -> tf.tensor[int], list of labels associated
                with the test data.
                First dimension, number of examples.
                Second dimension, one hot label.
    """
    
    # Consistency check
    if isinstance(class_nb, int):
        if class_nb < 0:
            print("Error in function prep_data: the selected class must be a positive integer.")
            return None, None

        elif class_nb > max(train_labels):
            # If the selected class is greater than the labels
            print("Error in function prep_data: the selected class must be less then the number of classes.")
            return None, None

    else:
        print("Error in function prep_data: the selected class must be an integer.")
        return None, None
    
    # Labels for thetrain data
    positive_train_indices = np.nonzero(train_labels == class_nb)[0]
    negative_train_indices = np.nonzero(train_labels != class_nb)[0]

    # Labels for thetest data
    positive_test_indices = np.nonzero(test_labels == class_nb)[0]
    negative_test_indices = np.nonzero(test_labels != class_nb)[0]

    # Change indices for the svm student
    train_labels[positive_train_indices] = 1
    train_labels[negative_train_indices] = 0
    test_labels[positive_test_indices] = 1
    test_labels[negative_test_indices] = 0

    return tf.one_hot(train_labels, 2), tf.one_hot(test_labels, 2)
