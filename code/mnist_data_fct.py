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

def extract_mnist_data(normalize=False):
    """ Extracts the data for the mnist data files.
    Input:  normalize -> boolean, normalize the data if True
    Output: mnist_train -> np.array[np.array[int]], list of examples with
                the last element the label. 
                First dimension number of examples.
                Second dimension features.
            mnist_test -> np.array[np.array[int]], list of examples with
                the last element the label.
                First dimension number of examples.
                Second dimension features.
    """

    mnist_train_data = np.load('../data/basetrain.npy')
    mnist_train_label = np.load('../data/labeltrain.npy')
    mnist_test_data = np.load('../data/basetest.npy')
    mnist_test_label = np.load('../data/labeltest.npy')
    
    if normalize:
    # Normalize data
        mnist_train_data = mnist_train_data/norm(mnist_train_data, axis=0)
        mnist_test_data = mnist_test_data/norm(mnist_test_data, axis=0)
    

    # Add labels
    mnist_train = np.append(mnist_train_data, [mnist_train_label], axis=0) 
    mnist_test = np.append(mnist_test_data, [mnist_test_label], axis=0)   

    return np.transpose(mnist_train), np.transpose(mnist_test)


def prep_data(mnist_train, mnist_test, class_nb):
    """ Prepares the MNIST data for a one vs all SVM.
    Changes the labels that are not equal to class_nb to -1.
    Returns None, None if an error occured.
    Input:  mnist_train -> np.array[np.array[int]], list of examples with
                the last element the label.
                First dimension number of examples.
                Second dimension features.
            mnist_test -> np.array[np.array[int]], list of examples with
                the last element the label
                First dimension number of examples.
                Second dimension features.
            class_nb -> int, desired class to be classified vs the others.
    Output: mnist_train -> np.array[np.array[int]], list of examples with
                the last element the label -1 if not equal to class_nb
                First dimension number of examples.
                Second dimension features.
            mnist_test -> np.array[np.array[int]], list of examples with
                the last element the label -1 if not equal to class_nb
                First dimension number of examples.
                Second dimension features.
    """
    
    # Consistency check
    if isinstance(class_nb, int):
        if class_nb < 0:
            print("Error in function prep_data: the selected class must be a positive integer.")
            return None, None

        elif class_nb > max(mnist_train[:, -1]):
            # If the selected class is greater than the labels
            print("Error in function prep_data: the selected class must be less then the number of classes.")
            return None, None

    else:
        print("Error in function prep_data: the selected class must be an integer.")
        return None, None

    for example in mnist_train:
        if example[-1] != class_nb:
            example[-1] = -1

    for example in mnist_test:
        if example[-1] != class_nb:
            example[-1] = -1

    return mnist_train, mnist_test
