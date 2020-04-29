#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the function to read and edit the mnist data.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
from numpy.random import default_rng 
from numpy.linalg import norm


def extract_mnist_data(model_type, normalize=False):
    """ Extracts the data for the mnist data files.
    Input:  model_type -> str, {'svm', 'cnn'} model type for the student.
            normalize -> boolean, normalize the data if True.
    Output: mnist_train_data -> np.array[np.array[int]], list
                of examples. 
                First dimension number of examples.
                Second dimension features.
            mnist_test_data -> np.array[np.array[int]], list
                of examples.
                First dimension number of examples.
                Second dimension features.
            mnist_train_labels -> np.array[int], list of labels associated
                with the train examples.
            mnist_test_labels -> np.array[int], list of labels associated
                with the test examples.
    """

    mnist_train_data = np.load('../../data/basetrain.npy')
    mnist_train_labels = np.load('../../data/labeltrain.npy')
    mnist_test_data = np.load('../../data/basetest.npy')
    mnist_test_labels = np.load('../../data/labeltest.npy')

    if normalize:
    # Normalize data
        mnist_train_data = mnist_train_data/norm(mnist_train_data, axis=0)
        mnist_test_data = mnist_test_data/norm(mnist_test_data, axis=0)
    
    return np.transpose(mnist_train_data), np.transpose(mnist_test_data), mnist_train_labels, mnist_test_labels


def prep_data(mnist_train_labels, mnist_test_labels, class_nb):
    """ Prepares the MNIST data for a one vs all strategy.
    Changes the labels that are not equal to class_nb to -1 and the others to 1
    for the svm. Changes the labels that are not equal to class_nb to 0 and the
    others to 1 for the cnn.
    Returns None, None if an error occured.
    Input:  mnist_train_labels -> np.array[int], list of labels associated
                with the train data.
            mnist_test_labels -> np.array[int], list of labels associated
                with the test data.
            class_nb -> int, desired class to be classified vs the others.
    Output: mnist_train_labels -> np.array[int], list of labels associated
                with the train data, the labels not equal to class_nb are -1.
                Coded in a one hot fashion if cnn student model.
            mnist_test_labels -> np.array[int], list of labesl associated
                with the test data, the labels not equal to class_nb are -1.
                Coded in a one hot fashion if cnn student model.
    """
    
    # Consistency check
    if isinstance(class_nb, int):
        if class_nb < 0:
            print("Error in function prep_data: the selected class must be a positive integer.")
            return None, None

        elif class_nb > max(mnist_train_labels):
            # If the selected class is greater than the labels
            print("Error in function prep_data: the selected class must be less then the number of classes.")
            return None, None

    else:
        print("Error in function prep_data: the selected class must be an integer.")
        return None, None
    
    # Labels for the train data
    positive_train_indices = np.nonzero(mnist_train_labels == class_nb)[0]
    negative_train_indices = np.nonzero(mnist_train_labels != class_nb)[0]

    # Labels for the test data
    positive_test_indices = np.nonzero(mnist_test_labels == class_nb)[0]
    negative_test_indices = np.nonzero(mnist_test_labels != class_nb)[0]

    # Change indices for the svm student
    mnist_train_labels[positive_train_indices] = 1
    mnist_train_labels[negative_train_indices] = 0
    mnist_test_labels[positive_test_indices] = 1
    mnist_test_labels[negative_test_indices] = 0

    return mnist_train_labels, mnist_test_labels

