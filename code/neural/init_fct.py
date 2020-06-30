#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the initializations functions for the teacher model.
Date: 30/6/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from numpy.random import default_rng 
from misc_fct import *


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
