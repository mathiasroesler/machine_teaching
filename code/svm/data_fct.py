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
    Output: train_data -> np.array[np.array[int]], list
                of examples. 
                First dimension number of examples.
                Second dimension features.
            test_data -> np.array[np.array[int]], list
                of examples.
                First dimension number of examples.
                Second dimension features.
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
        train_data = np.squeeze(tf.image.rgb_to_grayscale(train[0])) 
        test_data = np.squeeze(tf.image.rgb_to_grayscale(test[0]))

    else:
        # Extract mnist data and labels
        train, test = tf.keras.datasets.mnist.load_data()
        train_data = train[0]
        test_data = test[0]

    train_labels = np.squeeze(train[1])
    test_labels = np.squeeze(test[1])

    # Flatten the data for SVM
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1]*train_data.shape[2])
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1]*test_data.shape[2])

    return train_data[:10000], test_data[:2000], train_labels[:10000], test_labels[:2000]


def prep_data(train_labels, test_labels, class_nb):
    """ Prepares the data for a one vs all strategy.
    Changes the labels that are not equal to class_nb to -1 and the others to 1
    for the svm. 
    Returns None, None if an error occured.
    Input:  train_labels -> np.array[int], list of labels associated
                with the train data.
            test_labels -> np.array[int], list of labels associated
                with the test data.
            class_nb -> int, desired class to be classified vs the others.
    Output: train_labels -> np.array[int], list of labels associated
                with the train data, the labels not equal to class_nb are -1.
                Coded in a one hot fashion if cnn student model.
            test_labels -> np.array[int], list of labesl associated
                with the test data, the labels not equal to class_nb are -1.
                Coded in a one hot fashion if cnn student model.
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
    
    # Labels for the train data
    positive_train_indices = np.nonzero(train_labels == class_nb)[0]
    negative_train_indices = np.nonzero(train_labels != class_nb)[0]

    # Labels for the test data
    positive_test_indices = np.nonzero(test_labels == class_nb)[0]
    negative_test_indices = np.nonzero(test_labels != class_nb)[0]

    # Change indices for the svm student
    train_labels[positive_train_indices] = 1
    train_labels[negative_train_indices] = 0
    test_labels[positive_test_indices] = 1
    test_labels[negative_test_indices] = 0

    return train_labels, test_labels

