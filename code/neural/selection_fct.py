#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for selection of misclassified examples
Date: 16/7/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from numpy.random import default_rng 
from misc_fct import *


def select_examples(missed_indices, thresholds, weights):
    """ Selects the indices of the examples for the teaching set.

    The indice is selected if its weight is greater then the threshold.
    Input:  missed_indices -> np.array[int], list of indices of 
                missclassified examples.
            thresholes -> np.array[int], threshold for each example.
            weights -> np.array[int], weight of each example.
    Output: added_indices -> np.array[int], list of indices of examples
                to be added to the teaching set.

    """
    added_indices = np.array([], dtype=np.intc)

    while np.sum(weights[missed_indices]) < 1:
        weights[missed_indices] = 2*weights[missed_indices]
        added_indices = np.nonzero(weights[missed_indices] >= thresholds[missed_indices])[0]

    return added_indices


def select_rndm_examples(missed_indices, max_nb):
    """ Selects the indices of the examples for the teaching set.
    
    The indices are selected randomly among the missclassified ones.
    Input:  missed_indices -> np.array[int], list of indices of 
                missclassified examples.
            max_nb -> int, maximal number of examples to add.
    Output: added_indices -> np.array[int], list of indices of examples
                to be added to the teaching set.

    """
    rng = default_rng() # Set random seed

    if len(missed_indices) > max_nb:
        added_indices = rng.choice(missed_indices, max_nb, replace=False)

    else:
        added_indices = missed_indices

    return added_indices


def select_min_avg_dist(missed_indices, max_nb, train_data, train_labels, positive_average, negative_average):
    """ Selects the indices of the examples for the teaching set.

    The indices selected are the max//2 associated with positive
    examples that are closest to the negative average example and the
    max//2 negative examples that are closest to the positive average
    example. For binary classification only.
    Input:  missed_indices -> np.array[int], list of indices of
                missclassified examples.
            max_nb -> int, maximal number of examples to add.
            train_ data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            train_labels -> np.array[int], list of labels associated
                with the train data.
            positive_average -> tf.tensor[float32], average positive 
                example.
            negative_average -> tf.tensor[float32], average negative 
                example.
    Output: added_indices -> np.array[int], list of indices of examples
                to be added to the teaching set.

    """
    # Extract misclassified examples and labels 
    missed_data = tf.gather(train_data, missed_indices, axis=0)
    missed_labels = tf.gather(train_labels, missed_indices, axis=0)

    indices = find_indices(missed_labels)
    negative_indices = indices[0]
    positive_indices = indices[1]

    examples = find_examples(missed_data, missed_labels)
    negative_examples = examples[0]
    positive_examples = examples[1]

    if max_nb//2 < positive_indices.shape[0]:
        positive_dist = tf.norm(positive_examples-negative_average, axis=(1, 2))
        positive_indices = np.argsort(positive_dist, axis=0)[:max_nb//2]

    if max_nb//2 < negative_indices.shape[0]:
        negative_dist = tf.norm(negative_examples-positive_average, axis=(1, 2))
        negative_indices = np.argsort(negative_dist, axis=0)[:max_nb//2]

    try:
        added_indices = np.concatenate((np.squeeze(missed_indices[positive_indices]), np.squeeze(missed_indices[negative_indices])), axis=0)

    except ValueError:
        if len(positive_indices) != 1: 
            # If negative_indices is a scalar
            added_indices = np.concatenate((np.squeeze(positive_indices), negative_indices), axis=0)

        else:
            # If positive_indices is a scalar
            added_indices = np.concatenate((positive_indices, np.squeeze(negative_indices)), axis=0)

    return added_indices
