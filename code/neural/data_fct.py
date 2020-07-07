#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions to manipulate the data.
Date: 11/7/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from numpy.random import default_rng 


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
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.cifar10.load_data()

    else:
        # Extract mnist data and labels and reshape
        (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()
        train_data = tf.reshape(train_data, (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1))
        test_data = tf.reshape(test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

    return prep_data(train_data[:20000]), prep_data(test_data[:4000]), train_labels[:20000], test_labels[:4000]
    #return prep_data(train_data), prep_data(test_data), train_labels, test_labels


def prep_labels(labels, class_nb=0):
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
        print("Error in function prep_labels: class_nb must be a positive integer less then the number of classes.")
        exit(1)

    # Convert to two labels given the class number
    labels[labels != class_nb] = 0
    labels[labels == class_nb] = 1

    return labels


def prep_data(data):
    """ Normalizes the data.
    Input:  data -> tf.tensor[float32], list
                of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
    Output: norm_data -> tf.tensor[float32], list
                of normalized examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
    """

    return tf.cast(data/255, dtype=tf.float32)


def split_data(train_set, optimal_indices):
    """ Divides the train set into a validation set and a train set. 
    The validation set is composed of a tenth of the train set, the
    examples are chosen so that they are not in the optimal set. The
    validation set can be used for both sets.
    Input:  train_set -> tuple(tf.tensor[float32], tf.tensor[int]),
                train data and labels.
                First dimension, examples.
                Second dimension, labels.
            optimal_indices -> np.array[int], list of the indices of
                the optimal examples in the train set.
    Output: reduced_set -> tuple(tf.tensor[float32], tf.tensor[int]),
                train data and labels.    
                of examples. 
                First dimension, examples.
                Second dimension, labels
            optimal_set -> tuple(tf.tensor[float32], tf.tensor[int]), validation
                data and labels.
                First dimension, examples.
                Second dimension, labels.
            val_set -> tuple(tf.tensor[float32], tf.tensor[int]), validation
                data and labels.
                First dimension, examples.
                Second dimension, labels.
    """

    rng = default_rng(120) # Set seed 
    example_nb = train_set[1].shape[0]

    possible_indices = np.delete(np.arange(example_nb), optimal_indices)
    val_indices = rng.choice(possible_indices, example_nb//10, replace=False)
    train_indices = np.delete(np.arange(example_nb), val_indices)

    # Create validation set
    val_data = tf.gather(train_set[0], val_indices)
    val_labels = tf.gather(train_set[1], val_indices)

    # Create optimal set
    optimal_data = tf.gather(train_set[0], optimal_indices)
    optimal_labels = tf.gather(train_set[1], optimal_indices)

    # Create a reduced train set 
    reduced_train_data = tf.gather(train_set[0], train_indices)
    reduced_train_labels = tf.gather(train_set[1], train_indices)

    return (reduced_train_data, reduced_train_labels), (optimal_data, optimal_labels), (val_data, val_labels)
