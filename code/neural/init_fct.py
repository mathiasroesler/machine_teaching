#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for the initialization of the teacher.
Date: 7/5/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Conv2D, ZeroPadding2D, Flatten
from numpy.random import default_rng 
from custom_fct import *


def model_init(data_shape, max_class_nb):
    """ Initializes the model.
    Input:  data_shape -> tuple[int], shape of the input data. 
            max_class_nb -> int, number of classes.
    Output: model -> CNN model
    """

    try:
        assert(np.issubdtype(type(max_class_nb), np.integer))
        assert(max_class_nb > 1)

    except AssertionError:
        print("Error in function model_init: max_class_nb must be an integer greater or equal to 2.")
        exit(1)

    try:
        assert(len(data_shape) == 3)
        assert(data_shape[0] == data_shape[1])
        input_shape = data_shape

    except AssertionError:
        print("Error in function model_init: data_shape must have 3 elements with the two first ones equal.")
        exit(1)

    model = tf.keras.models.Sequential() # Sequential neural network

    # Add layers to model
    if (data_shape[0] == 28):
        # Pad the input to be 32x32
        model.add(ZeroPadding2D(2, input_shape=input_shape))
        input_shape = (input_shape[0]+4, input_shape[1]+4, input_shape[2])

    model.add(Conv2D(6, (5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten(data_format='channels_last'))
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(max_class_nb, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy']
                )

    return model


def rndm_init(labels):
    """ Randomly selects the index of an example for each class. The label 
    associated with the first class must be 0.
    Input:  labels -> np.array[int] | tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, label | one hot label.
    Output: init_indices -> np.array[int], list of indices of the 
                initial example to be used for each class.
    """

    if tf.is_tensor(labels):
        # If the labels are one hot convert to simple labels
        labels = np.argmax(labels, axis=1)

    rng = default_rng() # Set seed 

    max_class_nb = np.max(labels)
    indices = find_indices(labels) # List of example indices for each class
    init_indices = np.zeros(max_class_nb+1, dtype=np.intc) # List of initial examples indices

    for i in range(max_class_nb+1):
        # Randomly select an index for each class
        init_indices[i] = rng.choice(indices[i])

    return init_indices


def nearest_avg_init(data, labels):
    """ Selects the index of the example nearest to the average examples of 
    each class. The label associated with the first class must be 0.
    Input:  data -> tf.tensor[float32], list of examples. 
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> np.array[int] | tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, label | one hot label.
    Output: init_indices -> np.array[int], list of indices of the 
                initial example to be used for each class.
    """

    if tf.is_tensor(labels):
        # If the labels are one hot convert to simple labels
        labels = np.argmax(labels, axis=1)

    max_class_nb = np.max(labels)
    indices = find_indices(labels)             # List of example indices for each class
    examples = find_examples(data, labels)     # List of examples for each class
    averages = average_examples(data, labels)  # List of average example for each class
    init_indices = np.zeros(max_class_nb+1, dtype=np.intc) # List of initial examples indices

    for i in range(max_class_nb+1):
        # Select nearest example for each class
        dist = tf.norm(averages[i]-examples[i], axis=(1, 2)) # Estimate distance to average

        init_indices[i] = indices[i][np.argmin(np.mean(dist, axis=1))]

    return init_indices
